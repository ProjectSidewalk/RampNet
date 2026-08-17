#!/bin/bash
# Build a Tillicum training environment that MATCHES the klone #51 runs exactly.
#
# WHY EXACT PINS AND NOT "latest"
# The whole point of the Tillicum move is to finish the supervised-YOLO baseline so it
# can be compared against RampNet without a "you undertrained the baseline" objection.
# Resuming klone checkpoints into a different torch/ultralytics, or comparing a
# Tillicum-trained arm against a klone-trained one across library versions, reintroduces
# exactly the uncontrolled variable that #71's protocol exists to eliminate.
#
# The reference is not the README -- it is the training job logs themselves
# (logs/yolo_train_37745363.out on klone), which report:
#
#     Python-3.11.15   Ultralytics 8.4.105   torch-2.13.0+cu126
#
# A naive `pip install torch ultralytics` on Tillicum gives torch 2.8.0 + ultralytics
# 8.4.113, because Tillicum's SYSTEM python is 3.9.25 and the cu126 index tops out at
# 2.8.0 for 3.9. Hence the conda module: it is the only way to get 3.11 here.
#
# WHERE IT LIVES
# /gpfs/projects/makelab -- the 1 TB backed-up allocation, not /gpfs/scrubbed. The
# dataset belongs on scrubbed (it is huge and reproducible); the environment is small
# and annoying to rebuild, so it belongs where there are backups.
#
#   wsl-ssh.ps1 tillicum script scripts/tillicum_setup_env.sh

set -euo pipefail

PYVER="3.11.15"
TORCH="2.13.0+cu126"
ULTRA="8.4.105"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

ENVROOT="${ENVROOT:-/gpfs/projects/makelab/$USER/envs}"
ENVDIR="$ENVROOT/rampnet-yolo"

echo "=== target: python $PYVER / torch $TORCH / ultralytics $ULTRA ==="
mkdir -p "$ENVROOT"

module load conda
conda --version

if [ ! -x "$ENVDIR/bin/python" ]; then
    echo "=== creating conda env at $ENVDIR ==="
    conda create -y -p "$ENVDIR" "python=$PYVER"
else
    echo "=== env exists, reusing $ENVDIR ==="
fi

PY="$ENVDIR/bin/python"
"$PY" -V

echo "=== is the pinned torch actually available for this interpreter? ==="
# Fail loudly HERE rather than silently installing a different version: a mismatched
# torch is the kind of thing that produces plausible numbers and an unpublishable
# comparison.
if ! "$PY" -m pip index versions torch --index-url "$TORCH_INDEX" 2>&1 | grep -q "${TORCH}"; then
    echo "!! torch $TORCH NOT available for $("$PY" -V). Available:"
    "$PY" -m pip index versions torch --index-url "$TORCH_INDEX" 2>&1 | head -3
    echo "!! Refusing to install a substitute -- that would break comparability with #51."
    exit 1
fi

echo "=== installing pinned torch (+ torchvision, from the SAME index) ==="
"$PY" -m pip install --upgrade pip
# torchvision must come from the cu126 index too, exactly as the klone runbook does it
# (`pip install torch torchvision --index-url .../cu126`, hyak_yolo_runbook.sh). Install
# torch alone and ultralytics later drags torchvision in from PyPI instead -- a build
# with its own hard `torch==` pin, which either downgrades the torch we just pinned or
# lands a copy compiled against a different CUDA. The second case is the dangerous one:
# it is invisible unless the gate below looks for it, which is why it now does.
"$PY" -m pip install "torch==$TORCH" torchvision --index-url "$TORCH_INDEX"

echo "=== installing pinned ultralytics + the dataset deps ==="
# --no-deps would risk a broken install; instead pin ultralytics and let it resolve its
# own deps, then assert torch was not silently upgraded underneath us.
#
# datasets + huggingface_hub are NOT optional here, and they are not ultralytics deps.
# run_yolo_data_prep_tillicum.slurm points PYBIN at this env and calls the runbook's
# `data` stage, which runs download_dataset.py -- whose first import is
# `from datasets import load_dataset`. Without these the prep job dies on import under
# `set -euo pipefail`, after the H200 allocation has already started billing. The klone
# runbook installs the same three (hyak_yolo_runbook.sh).
"$PY" -m pip install "ultralytics==$ULTRA" datasets huggingface_hub

echo "=== VERIFY (this is the gate, not the install) ==="
"$PY" - <<PY
import sys, torch, torchvision, ultralytics
ok = True
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("ultralytics:", ultralytics.__version__)
if not sys.version.startswith("$PYVER"):        print("!! python mismatch");      ok = False
if torch.__version__ != "$TORCH":               print("!! torch mismatch");       ok = False
if ultralytics.__version__ != "$ULTRA":         print("!! ultralytics mismatch"); ok = False
# torchvision has no pinned target -- it just has to be a build whose compiled
# extension actually links against THIS torch. Test that by behaviour rather than by
# parsing a version string: local version labels (+cu126) are a packaging convention
# that has changed before, and a gate that fails a good env is worse than the hole it
# closes. Running one real op exercises the C++ extension, which is what breaks when
# ultralytics drags a mismatched torchvision in from PyPI.
try:
    import torch as _t
    from torchvision.ops import nms
    nms(_t.tensor([[0.0, 0.0, 1.0, 1.0]]), _t.tensor([0.5]), 0.5)
except Exception as e:
    print(f"!! torchvision does not link against this torch: {type(e).__name__}: {e}")
    ok = False
# The prep job imports these before it touches a GPU; catching it here costs nothing,
# catching it there costs an H200 allocation.
for mod in ("datasets", "huggingface_hub"):
    try:
        __import__(mod)
    except ImportError:
        print(f"!! {mod} missing -- download_dataset.py cannot run in this env")
        ok = False
print("cuda available:", torch.cuda.is_available(), "(False is EXPECTED on a login node)")
print("MATCHES KLONE" if ok else "DOES NOT MATCH KLONE -- do not train with this")
sys.exit(0 if ok else 1)
PY

echo
echo "=== done ==="
echo "env: $ENVDIR"
echo "Use it from the launcher with:  PYTHON=$ENVDIR/bin/python"
