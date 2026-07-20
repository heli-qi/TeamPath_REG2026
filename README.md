# TeamPath — REG² 2026 (MICCAI) Challenge Submission

Code submission for the **Pathologist REasoning-Guided REport Generation Challenge (REG² 2026)**.
This repository contains the **inference / container code** used to produce our Test Phase #2
submission, for both challenge interfaces.

> **Note on training code.** The training pipeline is being cleaned up for release and is
> **coming soon**. The code in this repository is the complete inference stack that builds the exact Docker container we submitted.

---

## Status

| Item | State |
|---|---|
| Inference / container code (both interfaces) | ✅ in this repository |
| Environment + dependency setup | ✅ `Dockerfile`, `requirements.txt` |
| Scripts to build & run the submitted container | ✅ `do_build.sh`, `do_test_run.sh`, `do_save.sh` |
| Model weights | ⬇️ hosted externally — see [Model weights](#model-weights) |
| Training code | 🚧 **coming soon** |

---

## Contents

- [Team](#team)
  - [Team Members](#team-members)
- [Method](#method)
  - [Interface-1 — Workflow Reasoning (Metric A)](#interface-1--workflow-reasoning-metric-a)
  - [Interface-0 — Visual Grounding (Metric B)](#interface-0--visual-grounding-metric-b)
- [Model weights](#model-weights)
  - [Download](#download)
  - [Required layout](#required-layout)
- [Reproducing inference](#reproducing-inference)
  - [Building the submission artifacts](#building-the-submission-artifacts)

---

## Team

| Field | Value |
|---|---|
| **Team Name** | TeamPath |
| **Primary Participant** | Tianyu Liu |

### Team Members

| Name | Grand Challenge Username | Grand Challenge Profile URL |
|---|---|---|
| Tianyu Liu | `superyalecbb` | https://grand-challenge.org/users/superyalecbb/ |
| Heli Qi | `qiheli` | https://grand-challenge.org/users/qiheli/ |
| Zeqi Zhou | `JZZQ` | https://grand-challenge.org/users/JZZQ/ |
| Dingyuan Dai | `Dingyuan` | https://grand-challenge.org/users/Dingyuan/ |
| Xitong Ling | `Tobycat` | https://grand-challenge.org/users/Tobycat/ |

---

## Method

Our submission covers both challenge interfaces with two independent pipelines that share one
container image.

### Interface-1 — Workflow Reasoning (Metric A)

Whole-slide image → chain-of-thought + final pathology report.

1. **Patching** (`src/reg/patching.py`) — tissue segmentation by HSV-saturation Otsu +
   morphology; 256-px tiles kept at tissue fraction ≥ 0.25. Tiles are read directly from the
   tiled TIFF via `tifffile` + `zarr`, so memory stays bounded regardless of slide size.
   Two magnifications are extracted (20x and 10x), each capped at **1024 patches** with a
   fixed seed (`cap_seed=0`) so runs are reproducible.
2. **Feature extraction** (`src/reg/uni2.py`) — each patch is embedded by **UNI2-h**
   (timm `vit_giant_patch14_224`, SwiGLU, 8 register tokens, 1536-d, fp16).
3. **Organ prediction** (`src/reg/cot_infer.py`) — multinomial logistic regression over pooled
   (mean‖max‖std) features. The predicted organ
   conditions the MIL heads via FiLM and selects the routing table then.
4. **TransMIL ensemble** (`src/reg/mil.py`, `src/reg/cot_infer.py`) — 10 models
   (5 seeds × {20x, 10x}), softmax-averaged.
   A multi-head design emits one answer per canonical question.
5. **Rule-based derivation** (`src/reg/derive_heads.py`) — derives dependent answers
   (Gleason pattern → grade group, Nottingham sub-scores → overall grade, etc.).
6. **Report generation** (`src/reg/report_gen.py`) — assembles the structured pathology report
   from the answered heads.
7. **CoT assembly** (`src/reg/cot_infer.py`) — deterministic per-organ traversal over
   `routing_smart.json`, with `edge_disc.json` disambiguating fan-out edges and
   `routing_fallback.json` covering unseen states. The final-report node carries the generated
   report and terminates the chain.

Reported test Metric A of this configuration: **Pending**.

### Interface-0 — Visual Grounding (Metric B)

ROI thumbnail + visual-context question → answer string.

1. **Rule gate** (`src/interf0/gate.py`) — visibility/tissue-presence questions and background
   ROIs are answered deterministically from a saturation/intensity tissue detector, using the
   canonical answer strings from training.
2. **VLM** (`src/interf0/model.py`) — everything else is answered by a **fine-tuned
   Qwen3.5-0.8B vision-language model**, loaded with plain `transformers` (no LlamaFactory at
   inference). Decoding is greedy (`do_sample=False`) for determinism. There is deliberately
   **no base-model fallback**: a missing fine-tuned checkpoint raises rather than silently
   scoring an untrained model.

---

## Model weights

Weights are **not** stored in this repository (size). They are hosted on Hugging Face as a single
archive that unpacks directly into `model/`.

**Host:** https://huggingface.co/weihao1115/TeamPath_REG2026
**File:** `model_v1_hybrid_alldata.tar.gz`

The repository is public — **no token and no access request are needed**.

### Download

From the repository root:

```bash
curl -L -o model_v1_hybrid_alldata.tar.gz \
  https://huggingface.co/weihao1115/TeamPath_REG2026/resolve/main/model_v1_hybrid_alldata.tar.gz

tar -xzf model_v1_hybrid_alldata.tar.gz -C model/
rm model_v1_hybrid_alldata.tar.gz
```

Equivalently, with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download weihao1115/TeamPath_REG2026 model_v1_hybrid_alldata.tar.gz \
  --repo-type model --local-dir .
tar -xzf model_v1_hybrid_alldata.tar.gz -C model/
```

The archive stores paths relative to its root, so `-C model/` places everything where the
container expects it. No files need to be moved or renamed afterwards.

### Required layout

After extraction, `model/` contains everything the container loads from `/opt/ml/model`:

```
model/
├── uni2-h.bin                                # UNI2-h patch encoder (~2.7 GB)
├── organ_clf_1024.npz                        # organ classifier (numpy: mean/scale/coef/intercept/classes)
├── routing_smart.json                        # per-organ CoT routing table
├── routing_fallback.json                     # fallback routing for unseen states
├── canonical_questions.json                  # normalized → canonical question strings
├── edge_disc.json                            # fan-out edge disambiguation
├── organ_valid_none.json                     # per-organ valid-None list
├── m1024none_uni2_transmil_s0 … _s4/         # 20x MIL ensemble (5 seeds)
│   ├── best.pt                               #   loaded at inference
│   └── val_perhead.json                      #   per-head val accuracy (not loaded)
├── m1024none_uni2_transmil10x_s0 … _s4/      # 10x MIL ensemble (5 seeds)
│   ├── best.pt
│   └── val_perhead.json
└── metric_b/sft/                             # fine-tuned Qwen3.5-0.8B VLM
    ├── config.json  generation_config.json
    ├── model.safetensors
    ├── tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
    ├── processor_config.json  preprocessor_config.json  video_preprocessor_config.json
    └── chat_template.jinja
```

**Verify all ten `best.pt` checkpoints are present.** `src/reg/cot_infer.py` loads them
opportunistically (`if not os.path.exists(ck): continue`), so a missing checkpoint silently
shrinks the ensemble and changes the scores rather than raising an error. Quick check:

```bash
ls -d model/m1024none_uni2_transmil{,10x}_s{0..4}/best.pt | wc -l   # must print 10
```

By contrast `uni2-h.bin`, `organ_clf_1024.npz`, `routing_smart.json`, `routing_fallback.json`,
`canonical_questions.json` and `metric_b/sft/{config.json, model.safetensors}` are mandatory —
a missing file raises `FileNotFoundError` immediately.

---

## Reproducing inference

```bash
git clone https://github.com/heli-qi/TeamPath_REG2026.git
cd TeamPath_REG2026

# 1. populate model/ as described above

# 2. build the image
bash do_build.sh

# 3. run both interfaces on the bundled sample cases
bash do_test_run.sh
```

To run a single case manually:

```bash
# Interface-1 (Metric A) — expects images/whole-slide-image/<uid>.tiff under the input dir
docker run --rm --gpus all --platform=linux/amd64 \
  -v "$PWD/test/input/interf1:/input:ro" \
  -v "$PWD/test/output:/output" \
  reg2026_algorithm
cat test/output/chain-of-thought.json

# Interface-0 (Metric B)
docker run --rm --gpus all --platform=linux/amd64 \
  -v "$PWD/test/input/interf0:/input:ro" \
  -v "$PWD/test/output:/output" \
  reg2026_algorithm
cat test/output/visual-context-response.json
```

`inference.py` dispatches on the interface by inspecting `/input/inputs.json`, matching the
Grand Challenge platform contract. Interface-1 writes a bare JSON array of
`{question, answer, next_question}` objects (last `next_question` is `""`); Interface-0 writes a
single answer string.

### Building the submission artifacts

```bash
bash do_save.sh
```

This produces `reg2026_algorithm_<timestamp>.tar.gz` (the container image, uploaded to
*Algorithm → Container Images*) and `model.tar.gz` (the contents of `model/`, uploaded
**separately** to *Algorithm → Models*). The platform mounts the model tarball at
`/opt/ml/model`, so the weights must be in `model.tar.gz` — not baked into the image.

