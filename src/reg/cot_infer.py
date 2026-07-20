"""Self-contained GT-free CoT inference for the REG2026 container (features -> chain-of-thought).
Ports the SOTA deployable pipeline (honest_test.py D+E2): None-aware TransMIL ensemble
(5x20x + 5x10x, softmax-avg, drop diagnosis=='None') -> traverse from root with routing +
edge_disc disambiguation -> report from all answers at the forced final-report node.
Loads all weights/artifacts from MODEL_PATH=/opt/ml/model. No ground truth, no train_CoT."""
import os, json
from collections import deque
import numpy as np
import torch

from .mil import MultiHeadMIL, ORGANS
from . import report_gen, derive_heads

M = os.environ.get("REG_MODEL_PATH", "/opt/ml/model")
DISC = os.environ.get("REG_DISC", "1") == "1"        # edge-disc fan-out disambiguation (on)
MASK = os.environ.get("REG_MASK", "0") == "1"        # organ-conditioned class mask (off)
N_MODE = os.environ.get("REG_N_MODE", "none")        # diagnosis count source (none = None-MIL)
GROUPS = [("m1024none_uni2_transmil", "20x"), ("m1024none_uni2_transmil10x", "10x")]
NSEEDS = int(os.environ.get("REG_NSEEDS", "5"))

C2F = {"breast": "Breast", "colon": "Colon", "stomach": "Stomach", "prostate": "Prostate",
       "bladder": "Urinary bladder", "lung": "Lung", "cervix": "Uterine cervix"}
NOTT_GRADE = {3: "Grade I", 4: "Grade I", 5: "Grade I", 6: "Grade II", 7: "Grade II",
              8: "Grade III", 9: "Grade III"}


def norm(s):
    s = " ".join((s or "").strip().split()); s = s.rstrip(".?! ")
    return s.replace(" an ", " a ").lower()


FINAL_Q = norm("What is the final pathology report?")
DIAGQS = [norm(f"What is the #{k} diagnosis?") for k in range(1, 5)]
AF_Q = norm("Is there any additional finding present?")
ROOTS = [norm("What is the organ?"), norm("What is the procedure?")]

_S = {}


def _lazy():
    if _S:
        return _S
    _R = json.load(open(f"{M}/routing_smart.json", encoding="utf-8"))
    _S["routing"], _S["ambig"] = _R["routing"], _R["ambig"]
    _S["FB"] = json.load(open(f"{M}/routing_fallback.json", encoding="utf-8"))
    _S["EDGE_DISC"] = json.load(open(f"{M}/edge_disc.json", encoding="utf-8")) if os.path.exists(f"{M}/edge_disc.json") else {}
    _S["ORGAN_VALID"] = json.load(open(f"{M}/organ_valid_none.json", encoding="utf-8")) if os.path.exists(f"{M}/organ_valid_none.json") else []
    _S["readable"] = json.load(open(f"{M}/canonical_questions.json", encoding="utf-8"))
    d = np.load(f"{M}/organ_clf_1024.npz", allow_pickle=True)
    _S["CM"], _S["CS"] = d["mean"].astype(np.float32), d["scale"].astype(np.float32)
    _S["CF"], _S["CI"] = d["coef"].astype(np.float32), d["intercept"].astype(np.float32)
    _S["CC"] = [str(x) for x in d["classes"]]
    groups = []
    for pre, scale in GROUPS:
        for s in range(NSEEDS):
            ck = f"{M}/{pre}_s{s}/best.pt"
            if not os.path.exists(ck):
                continue
            c = torch.load(ck, map_location="cpu", weights_only=False)
            net = MultiHeadMIL(1536, c["head_classes"], c.get("hid", 512), agg=c.get("agg", "abmil"),
                               n_organ=7, use_film=c.get("film", True), agg_cap=c.get("agg_cap", 2048))
            net.load_state_dict(c["model"]); net.eval().cuda()
            groups.append((net, scale))
            _S["questions"], _S["lspace"] = c["questions"], c["label_space"]
    _S["groups"] = groups
    return _S


def predict_organ(f20):
    s = _lazy()
    f = np.asarray(f20, np.float32)
    if len(f) == 0:
        return "breast"
    pooled = np.concatenate([f.mean(0), f.max(0), f.std(0)])
    z = (pooled - s["CM"]) / s["CS"]
    return s["CC"][int((z @ s["CF"].T + s["CI"]).argmax())]


def _fwd(f20, f10, oidx):
    s = _lazy(); probs = None; oi = torch.tensor([oidx], device="cuda")
    t20 = torch.from_numpy(np.asarray(f20, np.float32)).cuda()
    t10 = torch.from_numpy(np.asarray(f10, np.float32)).cuda()
    with torch.inference_mode():
        for net, scale in s["groups"]:
            feat = t20 if scale == "20x" else t10
            lg, _ = net([feat], oi)
            p = [torch.softmax(l[0], 0) for l in lg]
            probs = p if probs is None else [a + b for a, b in zip(probs, p)]
    ov = s["ORGAN_VALID"][oidx] if (MASK and 0 <= oidx < len(s["ORGAN_VALID"])) else {}
    out = {}
    for hi, q in enumerate(s["questions"]):
        pr = probs[hi]; idxs = ov.get(q)
        if idxs:
            m = torch.zeros_like(pr)
            for j in idxs:
                if j < pr.shape[0]:
                    m[j] = 1.0
            if float(m.sum()) > 0:
                pr = pr * m
        out[q] = s["lspace"][q]["classes"][int(pr.argmax())]
    return out


def apply_derived(ans):
    a = dict(ans)
    try:
        sc = (int(a.get("What is the score for tubular differentiation?", 0))
              + int(a.get("What is the score for nuclear pleomorphism?", 0))
              + int(a.get("What is the score for mitotic rate?", 0)))
        if sc >= 3:
            a["What is the overall score?"] = str(sc)
            a.setdefault("What is the grade of neoplasm?", NOTT_GRADE.get(sc, ""))
    except Exception:
        pass
    ndc = sum(1 for q in a if q.startswith("What is the #") and "diagnosis" in q)
    if ndc >= 1:
        a["What is the number of diagnoses to includes?"] = str(ndc)
    if "What is the #1 diagnosis?" in a:
        a.setdefault("What is the final pathology report?", "report")
    return a


def _traverse(organ, ans):
    s = _lazy(); routing, ambig = s["routing"], s["ambig"]; FB = s["FB"]; EDGE_DISC = s["EDGE_DISC"]
    o = norm(organ); qa = {norm(q): norm(a) for q, a in ans.items()}
    nN = sum(1 for q in DIAGQS if q in qa)
    af = qa.get(AF_Q, "").startswith("yes")
    if N_MODE == "union":   N = max(nN, 2 if af else 1)
    elif N_MODE == "inter": N = nN if (af or nN <= 1) else 1
    elif N_MODE == "af":    N = (max(2, nN) if af else 1)
    else:                   N = nN
    N = max(1, N)
    edges, seen, dq = set(), set(), deque(ROOTS)
    while dq:
        q = dq.popleft()
        if q in seen:
            continue
        seen.add(q)
        if q == FINAL_Q:
            edges.add((FINAL_Q, "")); continue
        if q in DIAGQS:
            k = DIAGQS.index(q) + 1
            nxt = DIAGQS[k] if (k < N and k < len(DIAGQS)) else FINAL_Q
            edges.add((q, nxt)); dq.append(nxt); continue
        a = qa.get(q); nqs = None
        if a is not None:
            bk = f"{o}|||{q}|||{a}"
            if bk in ambig:
                sig = "&&".join(f"{cq}={qa.get(cq,'')}" for cq in ambig[bk]); nqs = routing.get(f"{bk}##{sig}")
            if nqs is None:
                nqs = routing.get(bk)
        if nqs is None:
            fb = FB.get(f"{o}|||{q}")
            if fb:
                nqs = [fb]
        if nqs:
            for nq in nqs:
                if DISC and a is not None:
                    dd = EDGE_DISC.get(f"{o}|||{q}|||{a}|||{nq}")
                    if dd:
                        if dd.get("prune"):
                            continue
                        if qa.get(dd["cq"], "") not in set(dd["allowed"]):
                            continue
                edges.add((q, nq)); dq.append(nq)
    return edges, seen


def predict_chain_of_thought_from_feats(f20, f10):
    """f20,f10: [N,1536] UNI2 cap=1024 feature bags (20x, 10x). Returns list of CoT steps."""
    s = _lazy()
    organ = predict_organ(f20)
    oidx = ORGANS.index(organ) if organ in ORGANS else 0
    if len(f20) == 0:
        return [{"question": "What is the organ?", "answer": C2F.get(organ, "Breast"), "next_question": ""}]
    pred = _fwd(f20, f10, oidx)
    pred["What is the organ?"] = C2F.get(organ, pred.get("What is the organ?", ""))
    # None-aware models: drop diagnosis slots marked None -> routing stops there
    pred = {q: a for q, a in pred.items()
            if not (q.startswith("What is the #") and "diagnosis" in q and a in ("None", "", "none"))}
    used = apply_derived(derive_heads.derive(dict(pred)))
    n2q = {norm(q): q for q in used}; an = {norm(q): used[q] for q in used}
    report = report_gen.gen_report({q: used.get(q, "") for q in used}, level=2)   # report from ALL answers
    edges, _ = _traverse(organ, used)
    steps = []
    has_final = False
    for (qn, nqn) in sorted(edges):
        qc = n2q.get(qn, s["readable"].get(qn, qn))
        ans = report if qn == FINAL_Q else an.get(qn, "")
        nqc = "" if nqn == "" else s["readable"].get(nqn, n2q.get(nqn, nqn))
        if qn == FINAL_Q:
            has_final = True
        steps.append({"question": qc, "answer": ans, "next_question": nqc})
    if not has_final:   # ensure the 40%-weighted report node is always present
        steps.append({"question": "What is the final pathology report?", "answer": report, "next_question": ""})
    # match the REG2026 annotation convention: newlines stored as the literal two chars "\n"
    _LIT = chr(92) + "n"
    for st in steps:
        if chr(10) in st["answer"]:
            st["answer"] = st["answer"].replace(chr(10), _LIT)
    return steps
