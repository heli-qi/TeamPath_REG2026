"""
REG2026 Challenge — Algorithm Entry Point
=========================================

This file is the container's entrypoint (see Dockerfile).
It detects which interface is active and calls the right handler.

  Interface 0 — Visual Grounding (Metric B)
    Input  : paths to ROI thumbnail (.jpeg) + question (JSON)
    Output : visual-context-response.json  — a plain JSON string

  Interface 1 — Workflow Reasoning (Metric A)
    Input  : WSI at /input/images/whole-slide-image/<uid>.tiff  (uid = opaque hash)
    Output : chain-of-thought.json  — a JSON array of {question, answer, next_question}

Where to add YOUR code
-----------------------
  - src/interf0/model.py  →  predict_visual_context_response()
  - src/interf1/model.py  →  predict_chain_of_thought()

  The functions in core.py (I/O helpers, path constants, interface detection)
  do not need to be changed.

See README.md for a full walkthrough.
"""

from core import (
    INPUT_PATH,
    OUTPUT_PATH,
    get_interface_key,
    write_json_file,
    show_torch_cuda_info,
)

# ---------------------------------------------------------------------------
# Import your inference functions from src/
# ---------------------------------------------------------------------------
from src.interf0.model import predict_visual_context_response
from src.interf1.model import predict_chain_of_thought


# ---------------------------------------------------------------------------
# Entry point — dispatches to the correct handler automatically
# ---------------------------------------------------------------------------

def run():
    interface_key = get_interface_key()

    handler = {
        (
            "histopathology-region-of-interest-thumbnail",
            "visual-context-question",
        ): interf0_handler,
        ("whole-slide-image",): interf1_handler,
    }[interface_key]

    return handler()


# ---------------------------------------------------------------------------
# Interface 0 — Visual Grounding
# ---------------------------------------------------------------------------

def interf0_handler():
    # --- Fixed input paths (do not change) -----------------------------------
    question_path   = INPUT_PATH / "visual-context-question.json"
    roi_image_path  = INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg"
    output_path     = OUTPUT_PATH / "visual-context-response.json"

    print(f"[interf0] Question path : {question_path}")
    print(f"[interf0] ROI path      : {roi_image_path}")

    # --- Run inference -------------------------------------------------------
    # predict_visual_context_response lives in src/interf0/model.py
    answer = predict_visual_context_response(
        question_path=question_path,
        roi_image_path=roi_image_path,
    )

    # --- Write output --------------------------------------------------------
    # Output format: a plain JSON string — just the answer text.
    write_json_file(location=output_path, content=answer)
    print(f"[interf0] Answer written: {answer}")
    return 0


# ---------------------------------------------------------------------------
# Interface 1 — Workflow Reasoning
# ---------------------------------------------------------------------------

def interf1_handler():
    # --- WSI path (platform: /input/images/whole-slide-image/<uid>.tiff) -----
    wsi_dir = INPUT_PATH / "images" / "whole-slide-image"
    wsi_tiff_files = list(wsi_dir.glob("*.tiff"))
    if not wsi_tiff_files:
        raise FileNotFoundError(f"No .tiff files found in {wsi_dir}")
    wsi_path = wsi_tiff_files[0]
    output_path = OUTPUT_PATH / "chain-of-thought.json"
    show_torch_cuda_info()

    # --- Run inference -------------------------------------------------------
    # predict_chain_of_thought lives in src/interf1/model.py
    chain_of_thought = predict_chain_of_thought(wsi_path=wsi_path)

    # --- Write output --------------------------------------------------------
    # Output format: a bare JSON array of step objects.
    write_json_file(location=output_path, content=chain_of_thought)
    print(f"[interf1] Chain-of-thought written ({len(chain_of_thought)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
