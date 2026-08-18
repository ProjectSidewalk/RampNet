#!/bin/bash
# Run A of issue #84 -- the Stage 2 epoch curve.
#
# WHAT IS AND IS NOT CHANGED
# --------------------------
# Training-wise this is the released recipe (stage_two/run_train.slurm) verbatim:
# 4 nodes x 4 GPUs => world size 16, and train.py uses batch_size=1 per rank, so
# the global batch is 16 exactly as in the paper run. Same constant lr 1e-5 (the
# 'scratch' preset, no scheduler), same data, same splits, same seed 42, same
# ImageNet-initialised backbone.
#
# The ONLY deliberate change is --epochs 8 instead of 1. That is the experiment:
# per-epoch checkpoints so `manual_gold` F1 can be read across the curve and
# compared against the auto-label val loss the script already selects on. The
# paper run's auto-label val loss bottoms at epoch 5, so 8 covers the turn with
# margin (see the pre-registration in issue #84).
#
# The world size is load-bearing, not a speed knob: with batch_size=1 per rank,
# changing the node/GPU count changes the global batch and therefore the
# optimisation regime. Do not "scale this down to fit" -- it stops being a
# replicate of the released model.
#
# PATHS
# -----
# Site-neutral: everything comes from the environment, nothing is hardcoded.
#
#   RAMPNET_REPO  repo checkout                       (default: $HOME/RampNet)
#   RAMPNET_ENV   conda prefix built by
#                 stage_two/run_build_env.slurm       (default: $HOME/envs/sidewalkcv2)
#   RAMPNET_DATA  dataset root with train/ and val/   (default: $RAMPNET_REPO/dataset)
#   RAMPNET_EPOCHS                                    (default: 8)
#
# Submit with --chdir=<run dir> on a DURABLE volume. train.py writes several
# artefacts relative to the working directory -- latest_checkpoint.pth (the
# resume file), best_model.pth, runs/experiment_1 (tensorboard), peek_training/
# -- and on a preemptible partition the resume file is what makes requeue work.
# It must not sit on a volume that purges or on a home directory that can hit
# quota mid-write: a silent quota-exhausted write is what corrupted y26_pano's
# last.pt in the #51 YOLO grid.
#
#SBATCH --job-name=rampnet_run_a_84
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --output=logs/run_a_84_%j.out
#SBATCH --error=logs/run_a_84_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

# NOTE: deliberately no `set -u`. klone's lmod init dereferences an unbound
# LD_LIBRARY_PATH, so nounset makes `module load` fail. Every variable below
# carries an explicit ${VAR:-default} anyway.
set -eo pipefail

REPO="${RAMPNET_REPO:-$HOME/RampNet}"
ENV_PREFIX="${RAMPNET_ENV:-$HOME/envs/sidewalkcv2}"
DATA_ROOT="${RAMPNET_DATA:-$REPO/dataset}"
EPOCHS="${RAMPNET_EPOCHS:-8}"

PYBIN="$ENV_PREFIX/bin/python"
TORCHRUN="$ENV_PREFIX/bin/torchrun"

# Call the env's interpreter by absolute path rather than activating it. Under
# srun this is one less thing that has to survive the launch, and it is the
# pattern already proven on this cluster by the #51 YOLO runbook.
if [ ! -x "$TORCHRUN" ]; then
  echo "FATAL: no torchrun at $TORCHRUN -- build the env first:" >&2
  echo "       sbatch stage_two/run_build_env.slurm" >&2
  exit 1
fi

# Belt and braces alongside the editable install the env build does: train.py
# imports rampnet.model / rampnet.loading, and the working directory is the run
# dir, not the repo.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

mkdir -p checkpoints logs

echo "--- Run A (#84) configuration ---"
echo "Job ID          : ${SLURM_JOBID}"
echo "Restart count   : ${SLURM_RESTART_COUNT:-0}"
echo "Node List       : ${SLURM_NODELIST}"
echo "Nodes           : ${SLURM_NNODES}"
echo "GPUs per node   : ${NPROC_PER_NODE}"
echo "World size      : ${WORLD_SIZE}   (== global batch; train.py is batch_size=1/rank)"
echo "Master          : ${MASTER_ADDR}:${MASTER_PORT}"
echo "CPUs per task   : ${SLURM_CPUS_PER_TASK}"
echo "Repo            : ${REPO}"
echo "Env             : ${ENV_PREFIX}"
echo "Data root       : ${DATA_ROOT}"
echo "Working dir     : $(pwd)"
echo "Epochs          : ${EPOCHS}"
echo "Resume file     : $(pwd)/latest_checkpoint.pth$([ -f latest_checkpoint.pth ] && echo ' (PRESENT -- resuming)' || echo ' (absent -- fresh start)')"
echo "---------------------------------"

if [ "${WORLD_SIZE}" -ne 16 ]; then
  echo "WARNING: world size ${WORLD_SIZE} != 16. The released model's global batch was 16;" >&2
  echo "         this run is NOT a budget-matched replicate at any other world size." >&2
fi

echo "Launching torchrun..."
srun --export=ALL \
    "$TORCHRUN" --nnodes $SLURM_NNODES \
             --nproc_per_node $NPROC_PER_NODE \
             --rdzv_id $SLURM_JOB_ID \
             --rdzv_backend c10d \
             --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
             "$REPO/stage_two/train.py" \
             --data-root "$DATA_ROOT" \
             --epochs "$EPOCHS" \
             --preset scratch \
             --checkpoint-dir checkpoints

echo "--- Slurm job finished ---"
