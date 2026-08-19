#!/bin/bash
# The 8-epoch cosine rung (#135) -- and, at RAMPNET_EPOCHS=30, Run B itself.
#
# WHAT IS AND IS NOT CHANGED
# --------------------------
# Against Run A (stage_two/run_train_epoch_curve.slurm) exactly ONE thing changes:
# the learning rate decays. Same 4 nodes x 4 GPUs => world size 16, same
# batch_size=1 per rank => global batch 16, same PEAK lr 1e-5 ('scratch' preset),
# same data, same splits, same seed 42, same ImageNet-initialised backbone, same
# 8 epochs. So a comparison against Run A at the same epoch is budget-matched and
# isolates the SCHEDULE -- which B - A at 30 epochs cannot do, because it changes
# length and schedule together (the confound the #84 amendment flagged).
#
# There is deliberately NO WARMUP. Adding one would be a second change.
#
# WHY THE SCHEDULE IS STATELESS
# -----------------------------
# train.py computes the rate from the absolute step index (see lr_at_step), not
# from a scheduler object. This partition preempts -- Run A was requeued 5x, per sacct -D --
# and a stateful scheduler restarts its decay from the peak on every requeue
# unless its state is also serialized, turning the cosine into a sawtooth. That
# failure is SILENT: the job completes and the loss curve looks plausible. The
# per-step 'LR' scalar in tensorboard is the check; read it before trusting any
# result from this run.
#
# The world size is load-bearing, not a speed knob: with batch_size=1 per rank,
# changing the node/GPU count changes the global batch and therefore the
# optimisation regime. Do not "scale this down to fit".
#
# PATHS
# -----
# Site-neutral: everything comes from the environment, nothing is hardcoded.
#
#   RAMPNET_REPO       repo checkout                      (default: $HOME/RampNet)
#   RAMPNET_ENV        conda prefix from
#                      stage_two/run_build_env.slurm      (default: $HOME/envs/sidewalkcv2)
#   RAMPNET_DATA       dataset root with train/ and val/  (default: $RAMPNET_REPO/dataset)
#   RAMPNET_EPOCHS                                        (default: 8)
#   RAMPNET_LR_FINAL_FRAC  cosine floor as a fraction of peak lr (default: 0.0)
#
# Submit with --chdir=<run dir> on a DURABLE volume, and NOT the same run dir as
# Run A: train.py writes latest_checkpoint.pth, best_model.pth, runs/experiment_1
# and checkpoints/ relative to the working directory, and pointing this at Run A's
# directory would resume Run A's weights and overwrite its artefacts.
#
#SBATCH --job-name=rampnet_cosine_rung_135
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --output=logs/cosine_rung_135_%j.out
#SBATCH --error=logs/cosine_rung_135_%j.err
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
LR_FINAL_FRAC="${RAMPNET_LR_FINAL_FRAC:-0.0}"

PYBIN="$ENV_PREFIX/bin/python"
TORCHRUN="$ENV_PREFIX/bin/torchrun"

if [ ! -x "$TORCHRUN" ]; then
  echo "FATAL: no torchrun at $TORCHRUN -- build the env first:" >&2
  echo "       sbatch stage_two/run_build_env.slurm" >&2
  exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

mkdir -p checkpoints logs

echo "--- Cosine rung (#135) configuration ---"
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
echo "LR schedule     : cosine, peak 1e-5 -> ${LR_FINAL_FRAC} x peak, no warmup"
echo "Resume file     : $(pwd)/latest_checkpoint.pth$([ -f latest_checkpoint.pth ] && echo ' (PRESENT -- resuming)' || echo ' (absent -- fresh start)')"
echo "----------------------------------------"

if [ "${WORLD_SIZE}" -ne 16 ]; then
  echo "WARNING: world size ${WORLD_SIZE} != 16. Run A's global batch was 16;" >&2
  echo "         this run is NOT budget-matched to it at any other world size." >&2
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
             --lr-schedule cosine \
             --lr-final-frac "$LR_FINAL_FRAC" \
             --checkpoint-dir checkpoints

echo "--- Slurm job finished ---"
