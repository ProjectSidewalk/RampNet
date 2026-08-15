#!/bin/bash
# One-time build of the `sidewalkcv2` conda environment on klone, from the
# committed environment.yml. Runs as a batch job rather than on a login node:
# klone reaps heavy login processes, and that reap also kills the SSH master.
#
# Paths come from the environment so the committed script stays site-neutral:
#   RAMPNET_REPO       repo checkout           (default: $HOME/RampNet)
#   RAMPNET_ENV        conda prefix to create  (default: $HOME/envs/sidewalkcv2)
#   RAMPNET_CONDA_PKGS conda package cache     (default: $HOME/conda_pkgs)
#
# The conda package cache is redirected off $HOME on purpose: klone home
# directories are capped at 10 GB and the pytorch/CUDA packages alone are ~5 GB,
# so the default ~/.conda/pkgs blows the quota mid-solve.
#
#SBATCH --job-name=rampnet_env_build
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/env_build_%j.out
#SBATCH --error=logs/env_build_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

# NOTE: deliberately no `set -u`. klone's lmod init dereferences an unbound
# LD_LIBRARY_PATH, so nounset makes `module load` fail before conda exists.
# Every variable below carries an explicit ${VAR:-default} anyway.
set -eo pipefail

REPO="${RAMPNET_REPO:-$HOME/RampNet}"
ENV_PREFIX="${RAMPNET_ENV:-$HOME/envs/sidewalkcv2}"
export CONDA_PKGS_DIRS="${RAMPNET_CONDA_PKGS:-$HOME/conda_pkgs}"

mkdir -p "$CONDA_PKGS_DIRS" "$(dirname "$ENV_PREFIX")"

echo "--- env build configuration ---"
echo "repo        : $REPO"
echo "env prefix  : $ENV_PREFIX"
echo "pkgs cache  : $CONDA_PKGS_DIRS"
echo "node        : $(hostname)"
echo "-------------------------------"

module load conda/Miniforge3-25.9.1-0

if [ -x "$ENV_PREFIX/bin/python" ]; then
  echo "Environment already exists at $ENV_PREFIX; skipping create."
else
  conda env create --file "$REPO/environment.yml" --prefix "$ENV_PREFIX"
fi

# The `rampnet` package itself: no declared dependencies, so this only puts the
# repo on the path (CLAUDE.md: "no build step beyond pip install -e .").
"$ENV_PREFIX/bin/python" -m pip install --no-deps --no-build-isolation -e "$REPO"

echo "--- verification ---"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch, torchvision, timm
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("timm        ", timm.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available    ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            ", torch.cuda.get_device_name(0))
if torch.version.cuda is None:
    raise SystemExit("STOP: CPU-only torch build -- the CUDA selector in environment.yml did not take.")

# The two imports train.py needs from the repo package.
from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE
from rampnet.loading import load_checkpoint
m = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False)
n = sum(p.numel() for p in m.parameters())
print("KeypointModel params", f"{n:,}")

# Everything else train.py imports at module scope.
import numpy, PIL, matplotlib, tqdm
from torch.utils.tensorboard import SummaryWriter
print("tensorboard/matplotlib/PIL/numpy/tqdm OK")
PY

echo "--- resolved environment (recorded for the run log) ---"
conda list --prefix "$ENV_PREFIX" --explicit
echo "--- env build finished OK ---"
