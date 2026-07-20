"""
Interface 0 — Visual Grounding (Metric B).

Runs in the algorithm env (transformers>=5.6 that supports qwen3_5).
Loads a fine-tuned Qwen3.5-0.8B VL checkpoint with plain transformers — NO LlamaFactory
is needed at inference time. REQUIRES a fine-tuned checkpoint at
`/opt/ml/model/metric_b/sft/`; there is NO base-model fallback (see _resolve_model_dir).

Returns a plain string answer (written verbatim to visual-context-response.json
by the runner). Do not change the return type (str).
"""

from __future__ import annotations

import re
from pathlib import Path

from core import MODEL_PATH, load_json_file, load_roi_image
try:
    from .gate import is_visibility_question, detect_tissue, TISSUE_CANON, BG_CANON
except ImportError:  # flat-import fallback (interf0 not imported as a package)
    from gate import is_visibility_question, detect_tissue, TISSUE_CANON, BG_CANON

_MODEL = None
_PROCESSOR = None


def _resolve_model_dir() -> Path:
    """Return the FINETUNED Metric B SFT checkpoint. NO base-model fallback (CLAUDE.md §0.7):
    running the untrained base Qwen3.5 is forbidden, and a missing/incomplete sft must crash
    visibly — not silently score 0 on an untrained model. Also rejects a dangling weight link."""
    sft = MODEL_PATH / "metric_b" / "sft"
    has_weights = (sft / "model.safetensors").is_file() or (sft / "model.safetensors.index.json").is_file()
    if (sft / "config.json").exists() and has_weights:
        return sft
    raise FileNotFoundError(
        f"[interf0] finetuned Metric B checkpoint missing/incomplete at {sft} "
        "(need config.json + a real model.safetensors, not a dangling symlink). Refusing to "
        "fall back to the untrained base Qwen3.5 (CLAUDE.md no-silent-degradation rule)."
    )


def _load_model():
    """Lazy-load processor + model once (one case per container, but cheap to guard)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_dir = _resolve_model_dir()
    print(f"[interf0] loading Qwen3.5-0.8B VLM from {model_dir}")

    _PROCESSOR = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    _MODEL.eval()


def _extract_question(question_path: Path) -> str:
    q = load_json_file(location=question_path)
    if isinstance(q, dict):
        # platform shape is typically {"question": "..."}
        return str(q.get("question", q))
    return str(q)


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    """Run Visual Grounding inference for a single ROI -> answer string."""
    import torch

    question = _extract_question(question_path)
    roi_image = load_roi_image(location=roi_image_path)  # RGB PIL.Image

    # ---- 1) 背景 → 永远拒答(B1), 与问题类型无关(诊断/描述也拒); 纯像素规则, 不碰 GPU(毫秒级) ----
    if not detect_tissue(roi_image):
        return BG_CANON

    # ---- 2) 组织 + 可见性问句 → 罐头肯定答案; 不碰 GPU ----
    if is_visibility_question(question):
        return TISSUE_CANON

    # ---- 3) 组织 + 诊断/描述问句 → 0.8B VLM 生成内容 ----
    roi_image = roi_image.resize((256, 256))  # 与训练尺度(256@5x)一致, 修 train/inference 失配
    _load_model()
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": roi_image},
            {"type": "text", "text": question},
        ]}
    ]
    inputs = _PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,            # qwen3_5 关 thinking, 与训练 qwen3_5_nothink 一致
    ).to(_MODEL.device)

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = _MODEL.generate(**inputs, max_new_tokens=256, do_sample=False)

    answer = _PROCESSOR.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()  # 剥 <think>(普通token, skip_special 剥不掉)
    return answer or TISSUE_CANON  # 已知是组织(步骤1已过), 空兜底用肯定答案, 绝不回背景拒答
