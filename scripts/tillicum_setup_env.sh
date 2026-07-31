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

echo "=== installing pinned torch ==="
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==$TORCH" --index-url "$TORCH_INDEX"

echo "=== installing pinned ultralytics ==="
# --no-deps would risk a broken install; instead pin ultralytics and let it resolve its
# own deps, then assert torch was not silently upgraded underneath us.
"$PY" -m pip install "ultralytics==$ULTRA"

echo "=== VERIFY (this is the gate, not the install) ==="
"$PY" - <<PY
import sys, torch, ultralytics
ok = True
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("ultralytics:", ultralytics.__version__)
if not sys.version.startswith("$PYVER"):        print("!! python mismatch");      ok = False
if torch.__version__ != "$TORCH":               print("!! torch mismatch");       ok = False
if ultralytics.__version__ != "$ULTRA":         print("!! ultralytics mismatch"); ok = False
print("cuda available:", torch.cuda.is_available(), "(False is EXPECTED on a login node)")
print("MATCHES KLONE" if ok else "DOES NOT MATCH KLONE -- do not train with this")
sys.exit(0 if ok else 1)
PY

echo
echo "=== done ==="
echo "env: $ENVDIR"
echo "Use it from the launcher with:  PYTHON=$ENVDIR/bin/python"
