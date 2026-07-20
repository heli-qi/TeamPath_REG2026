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
   (mean‖max‖std) features, exported as pure NumPy in `organ_clf_1024.npz`. The predicted organ
   conditions the MIL heads via FiLM and selects the routing table — no ground truth is used.
4. **None-aware TransMIL ensemble** (`src/reg/mil.py`, `src/reg/cot_infer.py`) — 10 models
   (5 seeds × {20x, 10x}), softmax-averaged, with `diagnosis == "None"` predictions dropped.
   85 classification heads cover the directly predictable questions, one answer each.
5. **Rule-based derivation** (`src/reg/derive_heads.py`) — fills in the questions the heads do not
   predict, deriving them from the ones they do (Gleason score → grade group, Nottingham
   sub-scores → overall grade, differentiation, worst grade pattern, and others).
6. **Report generation** (`src/reg/report_gen.py`) — assembles the structured pathology report
   from the answered heads.
7. **CoT assembly** (`src/reg/cot_infer.py`) — deterministic per-organ traversal over
   `routing_smart.json`, with `edge_disc.json` disambiguating fan-out edges and
   `routing_fallback.json` covering unseen states. The final-report node carries the generated
   report and is the chain's only terminal node (the one with an empty `next_question`).

Reported test Metric A of this configuration: **0.8624**.

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

| | |
|---|---|
| **Host** | https://huggingface.co/weihao1115/TeamPath_REG2026 |
| **File** | `model_v1_hybrid_alldata.tar.gz` (4.1 GB) |

The repository is public — **no token and no access request are needed**.

### Download

The archive is **4.1 GB** and unpacks to roughly the same again, so allow **~9 GB free disk**
(both snippets extract before deleting the archive). From the repository root:

```bash
curl -fL -o model_v1_hybrid_alldata.tar.gz \
  https://huggingface.co/weihao1115/TeamPath_REG2026/resolve/main/model_v1_hybrid_alldata.tar.gz

tar -xzf model_v1_hybrid_alldata.tar.gz -C model/
rm model_v1_hybrid_alldata.tar.gz
```

`curl -f` matters: without it an HTTP error is written into the `.tar.gz` and only surfaces later
as a confusing `tar` error.

Equivalently, with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download weihao1115/TeamPath_REG2026 model_v1_hybrid_alldata.tar.gz \
  --repo-type model --local-dir .
tar -xzf model_v1_hybrid_alldata.tar.gz -C model/
rm model_v1_hybrid_alldata.tar.gz
```

The archive stores paths relative to its root, so `-C model/` places everything where the
container expects it. No files need to be moved or renamed afterwards.

The container runs as a non-root user, so the extracted files must be world-readable. If your
umask is stricter than `022`, fix it up afterwards:

```bash
chmod -R o+rX model
```

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

Two classes of artifact behave differently when missing, and the difference matters.

**Silently optional — absence changes the score without raising.** Verify these yourself; nothing
in the pipeline will warn you.

- The ten `best.pt` checkpoints. `src/reg/cot_infer.py:56-58` loads them opportunistically
  (`if not os.path.exists(ck): continue`), so missing checkpoints just shrink the ensemble.
  (If *all* ten are absent the run does fail, but with an unrelated `KeyError` rather than a
  clear missing-weights message.)
- `edge_disc.json` (`cot_infer.py:46`). Edge disambiguation is **on by default**
  (`REG_DISC`, `cot_infer.py:15`), so a missing file silently changes the emitted edge set and
  therefore Metric A.

`organ_valid_none.json` is loaded through the same `os.path.exists` guard (`cot_infer.py:47`) but
is inert in the shipped configuration: its contents are consulted only when the organ-conditioned
class mask is enabled (`REG_MASK`, `cot_infer.py:16,89`), which defaults to off and is set nowhere
in this repository. It is included in the archive for completeness.

```bash
ls -d model/m1024none_uni2_transmil{,10x}_s{0..4}/best.pt | wc -l   # must print 10
ls model/edge_disc.json                                             # must exist
```

**Mandatory — absence raises `FileNotFoundError`.** `uni2-h.bin`, `organ_clf_1024.npz`,
`routing_smart.json`, `routing_fallback.json`, `canonical_questions.json`, and
`metric_b/sft/`. The explicit check in `src/interf0/model.py:33-34` only tests for
`config.json` plus either `model.safetensors` or `model.safetensors.index.json`, but the
subsequent `AutoProcessor.from_pretrained` and `apply_chat_template` calls require the tokenizer,
processor and chat-template files too — ship `metric_b/sft/` complete, exactly as the archive
provides it.

The failure is raised **on first use, not at startup**: Interface-1 loads `uni2-h.bin` only after
patching, and the routing/classifier files only after the full UNI2-h feature pass, so an
Interface-1 run can proceed for several minutes before a missing file surfaces. On Interface-0 the
rule gate is pure NumPy and runs first, so `metric_b/sft` is resolved only for questions that fall
through it (`src/interf0/gate.py`) — and then fails within a second.

---

## Reproducing inference

**An NVIDIA GPU is required.** Interface-1 has no CPU path — the UNI2-h encoder and every MIL
checkpoint are moved to CUDA unconditionally (`src/reg/uni2.py:22`, `src/reg/cot_infer.py:60`).
Note that `do_test_run.sh` silently falls back to CPU when it cannot find working GPU passthrough
(`do_test_run.sh:20-32`); Interface-0 will still complete in that mode, but Interface-1 will fail
with a CUDA error.

```bash
git clone https://github.com/heli-qi/TeamPath_REG2026.git
cd TeamPath_REG2026

# 1. populate model/ as described above

# 2. build the image
bash do_build.sh

# 3. run both interfaces on the bundled sample cases
bash do_test_run.sh
```

`do_test_run.sh` rebuilds the image, then runs it once per interface. Results land in
`test/output/interf0/visual-context-response.json` and
`test/output/interf1/chain-of-thought.json`.

To run a single case manually:

```bash
# The container runs as a non-root user (uid 999), so the output directory must exist and be
# world-writable before the mount. Note the explicit chmod: `mkdir -p -m` is a no-op when the
# directory already exists, and do_test_run.sh leaves test/output itself at 0755.
mkdir -p test/output && chmod o+rwX test/output

# Interface-1 (Metric A) — expects images/whole-slide-image/<uid>.tiff under the input dir
docker run --rm --gpus all --platform=linux/amd64 \
  --network none \
  -v "$PWD/test/input/interf1:/input:ro" \
  -v "$PWD/model:/opt/ml/model:ro" \
  -v "$PWD/test/output:/output" \
  reg2026_algorithm
cat test/output/chain-of-thought.json

# Interface-0 (Metric B)
docker run --rm --gpus all --platform=linux/amd64 \
  --network none \
  -v "$PWD/test/input/interf0:/input:ro" \
  -v "$PWD/model:/opt/ml/model:ro" \
  -v "$PWD/test/output:/output" \
  reg2026_algorithm
cat test/output/visual-context-response.json
```

Each input directory must also contain the platform's `inputs.json` — `core.py:37-45` reads it to
decide which interface to dispatch to, so the same image serves both. The bundled sample cases
already include it.

The `model/` mount is what actually supplies the weights at run time: a bind mount at
`/opt/ml/model` shadows whatever the image contains, which is what `do_test_run.sh:113` does too.
`--network none` mirrors the platform's offline constraint.

### Building the submission artifacts

```bash
bash do_save.sh
```

This produces `reg2026_algorithm_<timestamp>.tar.gz` (the container image, uploaded to
*Algorithm → Container Images*) and `model.tar.gz` (the contents of `model/`, uploaded
**separately** to *Algorithm → Models*).

The platform mounts `model.tar.gz` at `/opt/ml/model`, and that mount is the authoritative source
of weights at evaluation time. Note that `Dockerfile` also copies `model/` into the image, so an
image built with a populated `model/` carries a second copy — the platform mount shadows it, so
only `model.tar.gz` determines what the algorithm actually loads.

