#!/usr/bin/env python3
"""Pre-extract the norm->canonical question/next_question map from train_CoT.json so the
container can emit EXACT canonical strings without shipping the 200MB train_CoT.
Run on tianyu; writes canonical_questions.json into the submission model/ folder."""
import json, sys
BASE = "/xuanwu-tank/south/MICCAI"
OUT = sys.argv[1] if len(sys.argv) > 1 else "canonical_questions.json"
cot = json.load(open(f"{BASE}/train_CoT.json", encoding="utf-8"))


def norm(s):
    s = " ".join((s or "").strip().split())
    s = s.rstrip(".?! ")
    return s.replace(" an ", " a ").lower()


canon = {}
for c in cot:
    for s in c.get("chain-of-thought", []):
        for k in ("question", "next_question"):
            t = (s.get(k) or "").strip()
            if t:
                canon.setdefault(norm(t), t)
json.dump(canon, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"canonical_questions: {len(canon)} -> {OUT}")
