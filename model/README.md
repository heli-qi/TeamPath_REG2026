# `model/`

Saved **model files for inference** — checkpoints and any small files your loader needs alongside them (configs, tokenizers, etc.).

## What to put here

| Kind | Examples |
|---|---|
| Checkpoints / weights | `weights.pt`, `model.safetensors`, `checkpoint.ckpt` |
| Bundled weights | `pytorch_model.bin`, `model.pt` |
| Config & tokenizer files | `config.json`, `tokenizer.json` |

Subfolders are fine. Use whatever layout your code expects.

**Example:**

```
model/
├── weights.pt
└── my_vlm/
    ├── config.json
    └── pytorch_model.bin
```

## Loading in code

At run time this directory is available as **`/opt/ml/model/`**. Open files via `MODEL_PATH` in [`core.py`](../core.py):

```python
from core import MODEL_PATH

model.load_state_dict(torch.load(MODEL_PATH / "weights.pt"))
```

Point `MODEL_PATH / "…"` at the paths you use under `model/`.
