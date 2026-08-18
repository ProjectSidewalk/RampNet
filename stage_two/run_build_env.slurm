#!/bin/bash
# One-time build of the `sidewalkcv2` conda environment on klone, from the
# committed environment.yml. Runs as a batch job rather than on a login node:
# klone reaps heavy login processes, and that reap also kills the SSH master.
#
# It also pre-warms the timm backbone cache, so this is the ONLY step that needs
# the network before training: `--preset scratch` builds the model with
# pretrained_backbone=True, and without a warm cache all 16 ranks race to
# download convnextv2_base.fcmae_ft_in22k_in1k_384 at step 0. Doing it here
# rather than in a login shell keeps the download off a login node too.
#
# Paths come from the environment so the committed script stays site-neutral:
#   RAMPNET_REPO       repo checkout           (default: $HOME/RampNet)
#   RAMPNET_ENV        conda prefix to create  (default: $HOME/envs/sidewalkcv2)
#   RAMPNET_CONDA_PKGS conda package cache     (default: $HOME/conda_pkgs)
#   RAMPNET_HF_HOME    timm/HF download cache  (default: the HF default, ~/.cache/huggingface --
#                                               must match what the training job sees)
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
if [ -n "${RAMPNET_HF_HOME:-}" ]; then
  export HF_HOME="$RAMPNET_HF_HOME"
fi

# Written only after the verification block below passes. The existence of
# $ENV_PREFIX/bin/python is NOT a completeness test: this job runs on a
# preemptible partition with --requeue, so a preemption between conda linking
# python and finishing the install would leave a half-built prefix that a
# python-only guard then skips forever -- every later submit would run pip and
# the verification against a broken env, fail under `set -e`, and never rebuild.
ENV_STAMP="$ENV_PREFIX/.rampnet_env_complete"

mkdir -p "$CONDA_PKGS_DIRS" "$(dirname "$ENV_PREFIX")"

echo "--- env build configuration ---"
echo "repo        : $REPO"
echo "env prefix  : $ENV_PREFIX"
echo "pkgs cache  : $CONDA_PKGS_DIRS"
echo "hf cache    : ${HF_HOME:-<default ~/.cache/huggingface>}"
echo "node        : $(hostname)"
echo "restart     : ${SLURM_RESTART_COUNT:-0}"
echo "-------------------------------"

module load conda/Miniforge3-25.9.1-0

if [ -f "$ENV_STAMP" ]; then
  echo "Environment already complete at $ENV_PREFIX (stamp present); skipping create."
else
  # Clear any partial prefix left by a preempted incarnation before recreating.
  # Guarded on the stamp path being non-empty so this can never widen into $HOME.
  if [ -e "$ENV_PREFIX" ] && [ -n "$ENV_PREFIX" ] && [ "$ENV_PREFIX" != "/" ]; then
    echo "Incomplete environment at $ENV_PREFIX (no stamp) -- removing and rebuilding."
    rm -rf "$ENV_PREFIX"
  fi
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

# Pre-warm the timm backbone so the 16 training ranks do not race to download it
# at step 0. Non-fatal: a transient network failure here must not throw away a
# 15-50 minute env build, and training still works (the ranks just race).
echo "--- pre-warming the timm backbone ---"
if "$ENV_PREFIX/bin/python" - <<'PY'
import timm
timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
print("backbone cached")
PY
then
  echo "backbone pre-warm OK"
else
  echo "WARNING: backbone pre-warm failed. The env is fine; the training ranks will" >&2
  echo "         download it themselves at step 0, racing on a cold cache." >&2
fi

echo "--- resolved environment (recorded for the run log) ---"
conda list --prefix "$ENV_PREFIX" --explicit

# Last: the stamp the skip-create guard reads. Nothing above may be skipped on a
# requeue unless everything above it succeeded.
date -u +"built %Y-%m-%dT%H:%M:%SZ on $(hostname) by job ${SLURM_JOBID:-none}" > "$ENV_STAMP"
echo "--- env build finished OK (stamped $ENV_STAMP) ---"
