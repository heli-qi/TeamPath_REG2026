"""Interface-1 (Workflow Reasoning / Metric A) — deployable GT-free pipeline.
WSI -> cap=1024 tissue patches (20x + 10x) -> UNI2-h features -> None-aware TransMIL
ensemble + routing/edge_disc traversal -> chain-of-thought. (SOTA test Metric A = 0.8624.)"""
from __future__ import annotations
import os
from pathlib import Path
from typing import TypedDict

from src.reg.patching import patch_wsi_path
from src.reg.uni2 import build_uni2h, extract_features
from src.reg.cot_infer import predict_chain_of_thought_from_feats

MODEL_PATH = Path(os.environ.get("REG_MODEL_PATH", "/opt/ml/model"))
CAP = int(os.environ.get("REG_PATCH_CAP", "1024"))
_UNI2 = {}


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


def _uni2():
    if "m" not in _UNI2:
        _UNI2["m"] = build_uni2h(str(MODEL_PATH / "uni2-h.bin"))
    return _UNI2["m"]


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    # 20x + 10x tissue patches (cap=1024, fixed seed -> reproducible), memory-bounded.
    i20, _, _ = patch_wsi_path(str(wsi_path), patch_size=256, tissue_thresh=0.25, seg_thumb=2048,
                               cap=CAP, cap_seed=0, downsample=1)
    i10, _, _ = patch_wsi_path(str(wsi_path), patch_size=256, tissue_thresh=0.25, seg_thumb=2048,
                               cap=CAP, cap_seed=0, downsample=2)
    print(f"[interf1] {len(i20)} 20x + {len(i10)} 10x patches", flush=True)
    uni2 = _uni2()
    f20 = extract_features(i20, uni2, batch_size=64) if len(i20) else __import__("numpy").zeros((0, 1536), "float16")
    f10 = extract_features(i10, uni2, batch_size=64) if len(i10) else __import__("numpy").zeros((0, 1536), "float16")
    steps = predict_chain_of_thought_from_feats(f20, f10)
    print(f"[interf1] {len(steps)} CoT steps", flush=True)
    return steps
