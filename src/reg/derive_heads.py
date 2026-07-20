#!/usr/bin/env python3
"""Derive the 7 'derived' heads from the 85 PREDICTED heads (no GT). Used to get
the real submittable score instead of GT-filling them."""
import re
NOTT_GRADE = {3: "Grade I", 4: "Grade I", 5: "Grade I", 6: "Grade II", 7: "Grade II",
              8: "Grade III", 9: "Grade III"}
GLEASON_GG = {"6 (3+3)": "Grade group 1", "7 (3+4)": "Grade group 2", "7 (4+3)": "Grade group 3",
              "8 (4+4)": "Grade group 4", "8 (3+5)": "Grade group 4", "8 (5+3)": "Grade group 4",
              "9 (4+5)": "Grade group 5", "9 (5+4)": "Grade group 5", "10 (5+5)": "Grade group 5"}


def _pn(s):
    m = re.search(r"(\d)", s or "")
    return int(m.group(1)) if m else None


def derive(ans):
    """ans: dict of 85-head predicted answers (+ organ/procedure). Returns dict with
    the derived heads filled in from predictions."""
    a = dict(ans)
    # number of diagnoses = count of #N diagnosis present
    ndc = sum(1 for q in a if q.startswith("What is the #") and "diagnosis" in q and a.get(q))
    if ndc >= 1:
        a["What is the number of diagnoses to includes?"] = str(ndc)
    # breast: overall score (sum Nottingham) + grade of neoplasm
    try:
        s = (int(a["What is the score for tubular differentiation?"])
             + int(a["What is the score for nuclear pleomorphism?"])
             + int(a["What is the score for mitotic rate?"]))
        a["What is the overall score?"] = str(s)
        a["What is the grade of neoplasm?"] = NOTT_GRADE.get(s, "")
    except (KeyError, ValueError):
        pass
    # prostate: Gleason score / grade group / worst pattern from predominant + present patterns
    pred = _pn(a.get("What is the pridominant pattern?"))
    if pred:
        present = [n for n in (3, 4, 5)
                   if a.get(f"Is there any Gleason pattern {n} present?", "").strip().lower().startswith("yes")]
        if present:
            others = [x for x in present if x != pred]
            sec = max(others) if others else pred
            gl = f"{pred + sec} ({pred}+{sec})"
            a["What is the Gleason score?"] = gl
            a["What is the grade group?"] = GLEASON_GG.get(gl, "")
            a["What is the worst grade pattern?"] = f"Gleason pattern {max(present)}"
    # colon/stomach: grade of neoplasm = differentiation embedded in #1 diagnosis
    d1 = a.get("What is the #1 diagnosis?", "")
    m = re.search(r"(well|moderately|poorly) differentiated", d1.lower())
    if m and not a.get("What is the grade of neoplasm?"):
        a["What is the grade of neoplasm?"] = m.group(1).capitalize() + " differentiated"
    return a


if __name__ == "__main__":
    # verify: derive from GT 85-head answers, check it reproduces GT derived heads
    import json, sys, collections
    BASE = "/workspace/sota_run"
    HEADS = set(json.load(open(f"{BASE}/models/mil_transmil/head_acc.json")).keys())
    cot = json.load(open(f"{BASE}/train_CoT.json", encoding="utf-8"))

    def answers(c):
        d = {}
        for s in c["chain-of-thought"]:
            q = (s.get("question") or "").strip()
            if q not in d:
                d[q] = (s.get("answer") or "").strip()
        return d
    DERIVED = ["What is the Gleason score?", "What is the grade group?", "What is the worst grade pattern?",
               "What is the overall score?", "What is the grade of neoplasm?",
               "What is the number of diagnoses to includes?"]
    hit = collections.Counter(); tot = collections.Counter()
    for c in cot:
        a = answers(c)
        # 85-head-only input (drop derived to simulate prediction-only)
        inp = {q: v for q, v in a.items() if q in HEADS or q in ("What is the organ?",)}
        d = derive(inp)
        for q in DERIVED:
            if q in a and a[q]:
                tot[q] += 1
                if d.get(q) == a[q]:
                    hit[q] += 1
    for q in DERIVED:
        if tot[q]:
            print("  %5d/%5d (%.0f%%)  %s" % (hit[q], tot[q], 100 * hit[q] / tot[q], q))
