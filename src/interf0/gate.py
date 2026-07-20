# -*- coding: utf-8 -*-
"""
interf0/gate.py — Metric B 规则门控(hack)。在 0.8B VLM 之前拦截「组织是否可见」类问题:
  问题命中可见性问法 → 纯像素规则判 组织/背景 → 返回固定 canned 答案(毫秒级, 不碰 GPU)。
  命中不了(诊断/分级/功能/复合句) → 返回 None → 调用方 fall through 到 VLM。
依赖仅 numpy+PIL(推理容器已有)。判别器与训练侧 H3 一致: 纹理结构(色相无关) + 饱和度/OD,
  并对 绿/青墨迹(非H&E、无结构) 做色相 veto。canned 与训练 canonical 逐字一致(train==infer==hack)。
阈值在 reg_b_data/reg26_ground 18 张 + 真实背景 holdout 上标定(目标: 背景误判组织率≈0 优先, 兼顾组织召回)。
"""
import re
import numpy as np
from PIL import Image

TISSUE_CANON = "Tissue is clearly visible in this ROI."
BG_CANON     = "This ROI is background only; no tissue is visible."

# ---- 问题分类器: 命中=可见性二元题(可被规则答); 否则 None(fall through 到 VLM) ----
_VIS_WHITELIST = {
    "is tissue visible in this roi answer briefly",
    "does this roi contain analyzable histological tissue answer briefly",
    "what is the dominant content in this roi answer briefly",
    "is this roi informative for histological image analysis answer briefly",
    "is there any tissue present in this roi answer briefly",
    "does this region show histological tissue answer briefly",
}
_NORM = re.compile(r"[^a-z0-9 ]+")
def _norm(q): return re.sub(r"\s+", " ", _NORM.sub(" ", (q or "").lower())).strip()

_INTENT = re.compile(r"\b(tissue|histolog\w*|content|informative|analyz\w*|background|empty|blank)\b")
_REGION = re.compile(r"\b(roi|region|image|patch|field|area)\b")
# 诊断/分级/功能/描述词 → 一律 fall through 到 VLM(规则不答这类题, 让 0.8B 生成内容)
_VETO = re.compile(r"\b(diagnos\w*|grade|gleason|tumou?r|carcinoma|malignan\w*|benign|cell type|"
                   r"mitos\w*|subtype|stage|margin|function\w*|express\w*|biomarker|grading|"
                   r"differentiat\w*|invasi\w*|metasta\w*|what disease|"
                   r"describe|explain|report|characteri\w*|comment|finding|abnormal\w*|"
                   r"identif\w*|assess|evaluat\w*|morpholog\w*|architectur\w*|cytolog\w*|"
                   r"lesion|structure|feature|pattern)\b")

def is_visibility_question(question: str) -> bool:
    n = _norm(question)
    if n in _VIS_WHITELIST:
        return True
    if _VETO.search(n):
        return False
    return bool(_INTENT.search(n) and _REGION.search(n))

# ---- 组织/背景 判别器 (256 RGB) ----
def _feats(im: Image.Image):
    a = np.asarray(im.convert("RGB").resize((256, 256))).astype(np.float32)
    g = a.mean(2)
    sat = a.max(2) - a.min(2)
    s15 = float((sat > 15).mean())
    od = -np.log10((g + 1.0) / 256.0)
    o = float((od > 0.15).mean())
    adj = float(np.abs(np.diff(g, axis=1)).mean())
    std = float(g.std())
    hsv = np.asarray(im.convert("RGB").resize((256, 256)).convert("HSV")).astype(np.float32)
    H, S = hsv[..., 0], hsv[..., 1]
    hi = S > 64
    fgreen = float(((H[hi] >= 60) & (H[hi] <= 145)).mean()) if hi.any() else 0.0
    return dict(s15=s15, o=o, adj=adj, std=std, fgreen=fgreen)

# 阈值(在 reg26_ground 18 张标定; 训练侧 H3 同款纹理门)
_TEX_ADJ, _TEX_STD = 2.0, 8.0      # 成片结构(纹理)门 — 区分"染料on组织(高纹理→tissue)"vs"染色玻璃cast(低纹理→bg)"
_S15, _S15_OD, _O = 0.05, 0.02, 0.10

def detect_tissue(im: Image.Image) -> bool:
    f = _feats(im)
    has_structure = (f["adj"] >= _TEX_ADJ and f["std"] >= _TEX_STD)
    colored = (f["s15"] >= _S15) or (f["o"] >= _O and f["s15"] >= _S15_OD)
    # 色相 veto: 高饱和主色为绿/青(非H&E墨迹) 且 无成片结构 → 背景
    if f["fgreen"] >= 0.5 and not has_structure:
        return False
    return bool(colored and has_structure)

def rule_answer(im: Image.Image) -> str:
    return TISSUE_CANON if detect_tissue(im) else BG_CANON
