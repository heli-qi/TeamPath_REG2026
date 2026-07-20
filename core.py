"""
core.py — Platform I/O utilities for the REG2026 submission container.

This file is not meant to be edited. It handles all fixed platform contracts:
  - canonical path constants
  - interface detection from /input/inputs.json
  - WSI path resolution (<uid>.tiff under images/whole-slide-image/)
  - file I/O helpers (JSON, ROI JPEG, WSI TIFF)
  - GPU diagnostics

Import what you need from inference.py; do not modify this file.
"""

import json
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image

# ---------------------------------------------------------------------------
# Fixed platform paths — do not change these
# ---------------------------------------------------------------------------

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

WSI_IMAGE_DIR = INPUT_PATH / "images" / "whole-slide-image"


# ---------------------------------------------------------------------------
# Interface detection
# ---------------------------------------------------------------------------

def get_interface_key() -> tuple:
    """
    Read /input/inputs.json (injected by the platform) and return a sorted
    tuple of socket slugs. Used in inference.py to dispatch to the correct
    handler without any manual branching.
    """
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    slugs = [entry["socket"]["slug"] for entry in inputs]
    return tuple(sorted(slugs))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def load_roi_image(*, location: Path) -> Image.Image:
    """Load the platform ROI thumbnail (.jpeg) as RGB; verify and log pixel stats."""
    with Image.open(location) as opened:
        image = opened.convert("RGB")
    print(f"[ROI] Load verification passed: {location}")
    return image


def load_wsi_array(*, location: Path) -> np.ndarray:
    """Load the platform WSI (<uid>.tiff) with tifffile; verify and log array stats."""
    array = tifffile.imread(location)
    print(f"[WSI] Load verification passed: {location}")
    return array


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def show_torch_cuda_info():
    print("=+=" * 10)
    print("Torch CUDA available:", (available := torch.cuda.is_available()))
    if available:
        print(f"  devices          : {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        print(f"  current device   : {current}")
        print(f"  device properties: {torch.cuda.get_device_properties(current)}")
    print("=+=" * 10)
