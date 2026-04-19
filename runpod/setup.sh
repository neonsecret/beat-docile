#!/bin/bash
# One-shot setup for a fresh RunPod pod (PyTorch 2.x base image).
# Run once after SSH-ing in. Takes ~5 minutes.
set -euo pipefail

echo "=== [1/4] System packages ==="
apt-get update -qq
apt-get install -y git poppler-utils unzip tmux htop

echo "=== [2/4] Python packages ==="
pip install --quiet \
    docile-benchmark \
    torchmetrics \
    pyarrow \
    seqeval \
    transformers \
    accelerate \
    tensorboard \
    pillow \
    tqdm \
    pydantic \
    typer \
    rich

echo "=== [3/4] Clone DocILE repo ==="
if [ ! -d /workspace/docile-repo ]; then
    git clone --depth=1 https://github.com/rossumai/docile.git /workspace/docile-repo
else
    echo "docile-repo already present, skipping clone."
fi

echo "=== [4/4] Apply compatibility patches ==="

TRAIN_SCRIPT=/workspace/docile-repo/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py
INFER_SCRIPT=/workspace/docile-repo/baselines/NER/docile_inference_NER_multilabel_layoutLMv3.py
DOC_IMAGES=/workspace/docile-repo/docile/dataset/document_images.py

# --- PyArrow 23 removed set_auto_load ---
# Wrap pyarrow.PyExtensionType.set_auto_load(True) in try/except in train script
python3 - <<'PYEOF'
import re, pathlib

def patch_pyarrow(path):
    text = pathlib.Path(path).read_text()
    old = "pyarrow.PyExtensionType.set_auto_load(True)"
    new = (
        "try:\n"
        "    pyarrow.PyExtensionType.set_auto_load(True)\n"
        "except AttributeError:\n"
        "    pass  # PyArrow >= 23 removed set_auto_load"
    )
    if old in text:
        pathlib.Path(path).write_text(text.replace(old, new))
        print(f"  patched pyarrow set_auto_load in {path}")
    else:
        print(f"  pyarrow patch already applied or not needed in {path}")

import os
for p in [
    "/workspace/docile-repo/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py",
    "/workspace/docile-repo/baselines/NER/docile_inference_NER_multilabel_layoutLMv3.py",
]:
    if os.path.exists(p):
        patch_pyarrow(p)
PYEOF

# --- PIL truncated images ---
python3 - <<'PYEOF'
import pathlib, os

PIL_PATCH = (
    "from PIL import ImageFile\n"
    "ImageFile.LOAD_TRUNCATED_IMAGES = True\n"
)

targets = [
    "/workspace/docile-repo/docile/dataset/document_images.py",
    "/workspace/docile-repo/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py",
    "/workspace/docile-repo/baselines/NER/docile_inference_NER_multilabel_layoutLMv3.py",
]
for path in targets:
    if not os.path.exists(path):
        print(f"  skip (not found): {path}")
        continue
    text = pathlib.Path(path).read_text()
    if "LOAD_TRUNCATED_IMAGES" in text:
        print(f"  PIL patch already applied: {path}")
        continue
    # insert after first 'from PIL import' line
    lines = text.splitlines(keepends=True)
    insert_after = -1
    for i, line in enumerate(lines):
        if line.startswith("from PIL import") or line.startswith("import PIL"):
            insert_after = i
            break
    if insert_after == -1:
        # no PIL import yet; add at top after imports block
        lines.insert(0, PIL_PATCH)
    else:
        lines.insert(insert_after + 1, PIL_PATCH)
    pathlib.Path(path).write_text("".join(lines))
    print(f"  patched PIL truncated images: {path}")
PYEOF

# --- transformers 5.x API changes in train script ---
python3 - <<'PYEOF'
import pathlib, re

path = pathlib.Path("/workspace/docile-repo/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py")
text = path.read_text()
changed = False

# evaluation_strategy -> eval_strategy
if 'evaluation_strategy=' in text:
    text = text.replace('evaluation_strategy=', 'eval_strategy=')
    changed = True
    print("  patched: evaluation_strategy -> eval_strategy")

# tokenizer= -> processing_class= in Trainer(...) call
# Only inside the Trainer( block, not in variable assignment
if 'tokenizer=tokenizer,' in text:
    text = text.replace('tokenizer=tokenizer,', 'processing_class=tokenizer,')
    changed = True
    print("  patched: tokenizer= -> processing_class= in Trainer")

# Remove warmup_ratio= line from TrainingArguments (keep warmup_steps)
if 'warmup_ratio=args.warmup_ratio,' in text:
    text = re.sub(r'\s*warmup_ratio=args\.warmup_ratio,\n', '\n', text)
    changed = True
    print("  patched: removed warmup_ratio= from TrainingArguments")

# init_weights() -> post_init()
if 'init_weights()' in text:
    text = text.replace('init_weights()', 'post_init()')
    changed = True
    print("  patched: init_weights() -> post_init()")

if changed:
    path.write_text(text)
else:
    print("  no transformers API patches needed (already up to date)")
PYEOF

# --- _tied_weights_keys in MyLayoutLMv3 ---
python3 - <<'PYEOF'
import pathlib, re

path = pathlib.Path("/workspace/docile-repo/baselines/NER/my_layoutlmv3.py")
if not path.exists():
    print("  my_layoutlmv3.py not found, skip")
    exit()
text = path.read_text()
if '_tied_weights_keys' in text:
    print("  _tied_weights_keys already present")
else:
    # Insert class attribute after the first 'class MyLayoutLMv3' line
    text = re.sub(
        r'(class MyLayoutLMv3\w+[^:]*:)',
        r'\1\n    _tied_weights_keys = {}',
        text,
        count=1
    )
    path.write_text(text)
    print("  patched: added _tied_weights_keys = {} to MyLayoutLMv3 class")
PYEOF

# --- NumPy 2.x str repr fix for get_sorted_field_candidates ---
python3 - <<'PYEOF'
import pathlib, re

path = pathlib.Path("/workspace/docile-repo/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py")
text = path.read_text()

# The problematic pattern: str([np.int64(0)]) emits 'np.int64(0)' in NumPy 2.x
# The fix: use a helper that parses via split rather than relying on str repr
OLD = "gid = str(ft.groups)"
NEW = (
    "# NumPy 2.x: str([np.int64(x)]) is 'np.int64(x)', not '[0]'; use repr-safe key\n"
    "            _g = ft.groups\n"
    "            gid = str([int(v) for v in (_g if hasattr(_g, '__iter__') else [_g])])"
)
if OLD in text:
    text = text.replace(OLD, NEW)
    path.write_text(text)
    print("  patched: NumPy 2.x str repr in get_sorted_field_candidates")
else:
    print("  NumPy str repr patch already applied or pattern changed")
PYEOF

echo ""
echo "Setup complete."
