#!/usr/bin/env bash
# One-shot: build the REG2026 interf1 image, then run it on the d021e460 sample WSI.
# Requires: docker daemon access + NVIDIA GPU + nvidia-container-toolkit.
set -e
cd "$(dirname "$0")"
IMG=reg2026_algorithm

echo "=== [1/3] docker build -> $IMG ==="
docker build --platform=linux/amd64 -t "$IMG" .

echo "=== [2/3] docker run on test/input/interf1 (d021e460) ==="
mkdir -p test/output
rm -f test/output/chain-of-thought.json
docker run --rm --gpus all --platform=linux/amd64 \
  -v "$PWD/test/input/interf1:/input:ro" \
  -v "$PWD/test/output:/output" \
  "$IMG"

echo "=== [3/3] result summary ==="
OUT=test/output/chain-of-thought.json
if [ ! -f "$OUT" ]; then echo "ERROR: $OUT not produced"; exit 1; fi
python3 - "$OUT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert isinstance(s, list) and s, "output is not a non-empty array"
keys_ok   = all(set(x) == {"question","answer","next_question"} for x in s)
organ_ok  = any(x["question"] == "What is the organ?" for x in s)
blanks    = sum(1 for x in s if not x["question"].strip())
terminals = [x for x in s if x["next_question"] == ""]
print(f"steps           : {len(s)}")
print(f"keys ok         : {keys_ok}")
print(f"organ step      : {organ_ok}")
print(f"blank questions : {blanks}")
print(f"terminal steps  : {len(terminals)} (expect 1)")
print("--- first 6 steps ---")
for x in s[:6]:
    print(f"  Q: {x['question']}")
    print(f"     A: {x['answer'][:70]}  ->  {x['next_question'][:45]}")
if terminals:
    print("--- final-report step ---")
    print("  Q:", terminals[0]["question"])
    print(" ", terminals[0]["answer"][:400].replace("\n", "\n  "))
ok = keys_ok and organ_ok and blanks == 0 and len(terminals) == 1
print("\nVALID" if ok else "\nCHECK FAILED")
sys.exit(0 if ok else 2)
PY
echo "=== done. full output at: $OUT ==="
