#!/usr/bin/env python3
"""No-Docker smoke test: run the EXACT container entry (predict_chain_of_thought) on one WSI
and dump its chain-of-thought, so you can validate the pipeline without building the image.

Usage:  REG_MODEL_PATH=./model python dryrun.py /path/to/slide.tiff
"""
import os, sys, json
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("REG_MODEL_PATH", os.path.join(HERE, "model"))
import core
core.MODEL_PATH = Path(os.environ["REG_MODEL_PATH"])   # override /opt/ml/model for local runs
from src.interf1 import model

wsi = sys.argv[1]
steps = model.predict_chain_of_thought(wsi_path=Path(wsi))
json.dump(steps, open("/tmp/dryrun_cot.json", "w"), ensure_ascii=False, indent=2)

keys_ok = all(set(s) == {"question", "answer", "next_question"} for s in steps)
term = [s for s in steps if s["next_question"] == ""]
print(f"\n=== {len(steps)} steps | keys_ok={keys_ok} | terminals={len(term)} (expect 1) ===")
for s in steps[:6]:
    print(json.dumps({"q": s["question"], "a": s["answer"][:60], "nq": s["next_question"][:40]},
                     ensure_ascii=False))
if term:
    print("--- final-report step ---")
    print(term[0]["question"], "::", term[0]["answer"][:320])
