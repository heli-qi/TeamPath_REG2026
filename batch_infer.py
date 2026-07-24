#!/usr/bin/env python3
"""Batch-run the REG2026 interf1 pipeline over a folder of test WSIs (Test Phase 1 has NO GT,
so this validates that the Metric-A model pipeline runs end-to-end and inspects its behaviour:
predicted organ, #patches, #CoT steps, report). Uses the exact container code path:
memory-bounded tiled patching -> UNI2-h -> two-stage organ MIL -> derived heads -> report
-> routing -> chain-of-thought. Writes <out>/<slide>.json (bare CoT array, submission shape)
and appends a row to <out>/_summary_r<rank>.tsv. Resumable (skips slides already done).

Shard across GPUs by launching one process per GPU with CUDA_VISIBLE_DEVICES=<gpu> and
--workers W --rank R (slide i handled by worker i % W). Optional --ckpts a.pt,b.pt averages
softmax over an MIL ensemble (default: single TransMIL == the packaged container model)."""
import os, sys, json, time, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("REG_MODEL_PATH", os.path.join(HERE, "model"))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="/data/MICCAI/test1")
ap.add_argument("--outdir", default="/data/MICCAI/test1_cot")
ap.add_argument("--ckpts", default=os.path.join(HERE, "model", "mil_transmil_s0.pt"),
                help="comma-separated MIL checkpoints; >1 -> softmax ensemble")
ap.add_argument("--cap", type=int, default=8192)
ap.add_argument("--threads", type=int, default=6, help="decode threads per slide in patch_wsi_path")
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--rank", type=int, default=0)
args = ap.parse_args()

import numpy as np, torch
from src.reg.patching import patch_wsi_path
from src.reg.uni2 import build_uni2h, extract_features
from src.reg.mil import load_mil, ORGANS
from src.reg import cot, organ, derive_heads, report_gen

os.makedirs(args.outdir, exist_ok=True)
M = os.environ["REG_MODEL_PATH"]
uni2 = build_uni2h(os.path.join(M, "uni2-h.bin"))
ORGAN_CLF = organ.load_organ_clf(os.path.join(M, "organ_clf.npz"))
ckpts = [c for c in args.ckpts.split(",") if c]
nets = []
questions = lspace = None
for c in ckpts:
    net, questions, lspace = load_mil(c)
    nets.append(net)
print(f"[r{args.rank}] loaded UNI2-h + organ_clf + {len(nets)} MIL ckpt(s)", flush=True)


def mil_logits(feat, oidx):
    """mean softmax across the ensemble; returns list of [1,C] log-probs per head."""
    accs = None
    with torch.inference_mode():
        for net in nets:
            lg, _ = net([feat], torch.tensor([oidx], device="cuda"))
            sm = [torch.softmax(h, 1) for h in lg]
            accs = sm if accs is None else [a + s for a, s in zip(accs, sm)]
    return [a / len(nets) for a in accs]


def run_one(wsi):
    sid = os.path.splitext(os.path.basename(wsi))[0]
    out = os.path.join(args.outdir, sid + ".json")
    if os.path.exists(out):
        return None
    t0 = time.time()
    imgs, coords, _ = patch_wsi_path(wsi, 256, 0.25, 2048, cap=args.cap, cap_seed=0, threads=args.threads)
    if len(imgs) == 0:
        steps = [{"question": "What is the organ?", "answer": "Breast", "next_question": ""}]
        json.dump(steps, open(out, "w"), ensure_ascii=False, indent=2)
        return (sid, 0, len(steps), "NA", "EMPTY", round(time.time() - t0, 1))
    feats = extract_features(imgs, uni2, 64)
    org = organ.predict_coarse(feats, ORGAN_CLF)
    oidx = ORGANS.index(org) if org in ORGANS else 0
    feat = torch.from_numpy(feats.astype(np.float32)).cuda()
    pr = mil_logits(feat, oidx)
    pred = {q: lspace[q]["classes"][int(pr[hi].argmax(1))] for hi, q in enumerate(questions)}
    pred["What is the organ?"] = organ.COARSE2FINE.get(org, pred.get("What is the organ?", ""))
    used = derive_heads.derive(pred); used = cot.apply_derived(used)
    report = report_gen.gen_report({q: used.get(q, "") for q in used}, level=2)
    norm2q = {cot.norm(q): q for q in used}; ans_norm = {cot.norm(q): used[q] for q in used}
    steps = []
    for (qn, nqn) in sorted(cot.assemble_edges(org, used)):
        qc = norm2q.get(qn, cot.canonical.get(qn, qn))
        ans = report if qn == cot.FINAL_Q else ans_norm.get(qn, "")
        nqc = "" if nqn == "" else cot.canonical.get(nqn, norm2q.get(nqn, nqn))
        steps.append({"question": qc, "answer": ans, "next_question": nqc})
    json.dump(steps, open(out, "w"), ensure_ascii=False, indent=2)
    rep1 = report.splitlines()[0] if report else ""
    return (sid, len(coords), len(steps), org, rep1, round(time.time() - t0, 1))


files = sorted(f for f in glob.glob(os.path.join(args.indir, "*.tiff"))
               if not os.path.exists(f + ".aria2"))
files = [f for i, f in enumerate(files) if i % args.workers == args.rank]
sumf = os.path.join(args.outdir, f"_summary_r{args.rank}.tsv")
print(f"[r{args.rank}] {len(files)} slides to process", flush=True)
for k, wsi in enumerate(files):
    try:
        r = run_one(wsi)
        if r is None:
            continue
        with open(sumf, "a") as sf:
            sf.write("\t".join(map(str, r)) + "\n")
        print(f"[r{args.rank}] {k+1}/{len(files)} {r[0]} patch={r[1]} steps={r[2]} "
              f"organ={r[3]} {r[5]}s", flush=True)
    except Exception as e:
        print(f"[r{args.rank}] ERROR {os.path.basename(wsi)}: {type(e).__name__}: {e}", flush=True)
        with open(os.path.join(args.outdir, f"_errors_r{args.rank}.txt"), "a") as ef:
            ef.write(f"{wsi}\t{type(e).__name__}: {e}\n")
print(f"[r{args.rank}] DONE", flush=True)
