#!/bin/bash
# Rebuild the YOLO training environment for issue #51, PINNED and on DURABLE storage.
#
# Why this exists as a committed launcher rather than the runbook's `env` stage:
#
# 1. **Durable, not scratch.** The runbook builds to `$SCRATCH/envs/yolo`. On 2026-08-17
#    that environment was found DESTROYED by the /gscratch/scrubbed purge -- it deletes
#    files by access time and leaves the directory tree standing, so `ls` looked healthy
#    while `lib/python3.11` had 38 entries instead of 207 and `encodings/__init__.py` was
#    gone. The interpreter dies at startup with "no codec search functions registered".
#    The same thing had already happened to the `sidewalkcv2` copy. Environments belong on
#    /gscratch/makelab, which is purchased and never purged.
#
# 2. **Pinned, not latest.** The runbook does a bare `pip install torch torchvision
#    ultralytics`. The #51 grid trained under specific versions, and a resumed run must not
#    silently change training library mid-run -- `y11x_tiles` is 36+ epochs in. The pins
#    below were recovered from the surviving dist-info directories of the destroyed env,
#    so they are what actually trained, not what the docs assume:
#
#        torch 2.13.0+cu126   torchvision 0.28.0+cu126   ultralytics 8.4.105
#
#    (For reference, `pip install ultralytics` on 2026-08-17 resolves to 8.4.118.)
#
# Submit (NOT on a login node -- klone reaps heavy login processes, and that reap also
# kills the SSH control master):
#
#   cd ~/RampNet && mkdir -p logs
#   export YOLO_ENV=/gscratch/makelab/jonf/envs/yolo
#   export YOLO_CONDA_PKGS=/gscratch/scrubbed/jfroehli/conda_pkgs
#   sbatch -A ckpt-all -p ckpt-all --export=ALL,YOLO_ENV,YOLO_CONDA_PKGS \
#          scripts/model_comparison/run_build_yolo_env.slurm
#
# Afterwards, point the training launcher at it:
#
#   PYTHON=$YOLO_ENV/bin/python ... sbatch scripts/model_comparison/run_yolo_train.slurm
#
#SBATCH --job-name=yolo_env_build
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/yolo_env_build_%j.out
#SBATCH --error=logs/yolo_env_build_%j.err

# NOT `set -u`: klone's lmod init dereferences an unbound LD_LIBRARY_PATH, so nounset
# makes `module load` fail before conda ever exists. Use explicit ${VAR:-default}.
set -eo pipefail

ENV_PREFIX="${YOLO_ENV:-/gscratch/makelab/jonf/envs/yolo}"
# The conda package cache must live off $HOME: klone homes are capped at 10 GB and the
# torch/CUDA download alone is ~5 GB, so the default ~/.conda/pkgs blows the quota
# mid-solve. Scratch is the right home for it -- it is regenerable, and /gscratch/makelab
# was at 92% when this was written, so ~8 GB of cache does not belong there.
export CONDA_PKGS_DIRS="${YOLO_CONDA_PKGS:-/gscratch/scrubbed/$USER/conda_pkgs}"

# ...but a scratch cache is exposed to the same purge that destroyed the env, and a
# PARTIALLY purged cache is worse than no cache: conda trusts what it finds, then fails
# the transaction late with
#
#   CondaVerificationError: The package for readline ... appears to be corrupted.
#   The path 'share/man/man3/readline.3' specified in the package manifest cannot be found
#
# because the purge reaped the rarely-read man pages out of an otherwise-present package.
# That is a real 2026-08-17 failure of this very script, not a hypothetical. Clearing the
# cache costs a re-download and nothing else, since the versions below are pinned.
CLEAN_PKGS="${YOLO_CLEAN_PKGS:-1}"

# The versions that actually trained the #51 grid. Do not bump these to resume a run.
PIN_TORCH="${PIN_TORCH:-2.13.0}"
PIN_TORCHVISION="${PIN_TORCHVISION:-0.28.0}"
PIN_ULTRALYTICS="${PIN_ULTRALYTICS:-8.4.105}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"

echo "--- YOLO env rebuild (#51) ---"
echo "prefix      : $ENV_PREFIX"
echo "pkgs cache  : $CONDA_PKGS_DIRS"
echo "pins        : torch==$PIN_TORCH  torchvision==$PIN_TORCHVISION  ultralytics==$PIN_ULTRALYTICS"
echo "cuda index  : $CUDA_INDEX"
echo "node        : ${SLURMD_NODENAME:-?}"
echo "------------------------------"

mkdir -p "$CONDA_PKGS_DIRS" "$(dirname "$ENV_PREFIX")"

module load conda/Miniforge3-25.9.1-0

if [ "$CLEAN_PKGS" != "0" ]; then
    echo "Clearing the package cache first (YOLO_CLEAN_PKGS=$CLEAN_PKGS) -- see the note above."
    conda clean --all -y || echo "conda clean returned non-zero; continuing"
    # conda clean leaves extracted dirs it considers in use by an env on the same
    # filesystem. Nothing here is, and a half-reaped extracted dir is exactly what fails
    # verification, so remove them outright.
    rm -rf "${CONDA_PKGS_DIRS:?}"/*
fi

# Do NOT treat "bin/python exists" as "env is fine". The whole reason this script exists
# is that a purged env keeps its directory tree and its executable while having lost the
# stdlib underneath -- and a failed conda transaction leaves the same shape. Ask the
# interpreter to prove it runs; recreate from scratch if it cannot.
if [ -x "$ENV_PREFIX/bin/python" ] && "$ENV_PREFIX/bin/python" -c "import encodings, ssl, sys" 2>/dev/null; then
    echo "Env at $ENV_PREFIX is present and its interpreter runs -- reusing."
else
    if [ -e "$ENV_PREFIX" ]; then
        echo "Env at $ENV_PREFIX is absent, hollow or half-built -- removing and recreating."
        rm -rf "${ENV_PREFIX:?}"
    fi
    conda create -p "$ENV_PREFIX" python=3.11 -y
fi

PYBIN="$ENV_PREFIX/bin/python"
"$PYBIN" -m pip install --upgrade pip

# torch first, from the CUDA index, so the resolver cannot substitute a CPU-only wheel.
"$PYBIN" -m pip install --index-url "$CUDA_INDEX" \
    "torch==$PIN_TORCH" "torchvision==$PIN_TORCHVISION"
# ultralytics from PyPI, pinned. --no-deps would be wrong (it needs its own deps), but
# torch is already satisfied above so it will not be re-resolved.
"$PYBIN" -m pip install "ultralytics==$PIN_ULTRALYTICS"

echo
echo "--- Verify ---"
"$PYBIN" - <<'PY'
import sys
import torch
import torchvision
import ultralytics

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("ultralytics ", ultralytics.__version__)

# A CPU-only torch is the classic silent failure of this install: it imports fine and
# then trains at 1/50th speed, or not at all on a GPU-only partition.
if "cpu" in torch.__version__:
    raise SystemExit("STOP: CPU-only torch -- reinstall from the cu126 index.")

expected = {"torch": "2.13.0+cu126", "torchvision": "0.28.0+cu126", "ultralytics": "8.4.105"}
actual = {"torch": torch.__version__, "torchvision": torchvision.__version__,
          "ultralytics": ultralytics.__version__}
bad = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
if bad:
    # Loud, because a resumed run under a different training library is exactly the
    # confound this file exists to prevent.
    raise SystemExit(f"STOP: version drift vs the #51 grid: {bad}")
print("\nOK -- matches the versions the #51 grid trained under.")
PY

echo
echo "Env ready: $PYBIN"
echo "Resume y11x_tiles with:"
echo "  PYTHON=$PYBIN YOLO_DATA=/gscratch/scrubbed/\$USER/yolo/tiles/data.yaml NAME=y11x_tiles \\"
echo "    sbatch -A gpu-l40s-makelab -p gpu-l40s -q normal --time=7-00:00:00 \\"
echo "           --dependency=afterany:<current_jobid> \\"
echo "           --export=ALL,PYTHON,YOLO_DATA,NAME scripts/model_comparison/run_yolo_train.slurm"
