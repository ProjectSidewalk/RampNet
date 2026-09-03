#!/bin/bash
# Train ONE Stage 2 seed replicate on klone, for the seed-variance campaign
# (docs/seed_variance_51_135.md; issues #51 and #135).
#
# WHY THIS IS A SEPARATE FILE AND NOT A FLAG ON run_train.slurm
# run_train.slurm is the preserved record of the published run and of #135's rungs. It
# should not churn, and more importantly a seed replicate needs two things that file
# deliberately does not do: a per-seed working directory, and a bounded epoch count. The
# same reasoning kept run_yolo_train_tillicum.slurm separate from its klone original.
#
# WHY A PER-SEED WORKING DIRECTORY -- THIS IS THE LOAD-BEARING PART
# train.py writes `best_model.pth` and `latest_checkpoint.pth` to the CURRENT DIRECTORY,
# not to --checkpoint-dir (see stage_two/train.py: torch.save(..., "best_model.pth")).
# Three seeds launched from one directory would therefore overwrite each other's best
# model AND each other's resume state, and the second failure is worse than the first:
# a resume file from another seed is silently loaded as if it were this run's own, so
# the arms converge on one lineage and nothing in the log says so. Each seed gets its
# own RUNDIR and cd's into it before torchrun.
#
# USAGE
#   SEED=1 sbatch stage_two/run_train_seed.slurm
#   SEED=2 RUNDIR=/gscratch/scrubbed/$USER/seedvar/rampnet_s2 sbatch stage_two/run_train_seed.slurm
#
# COST. klone ckpt is FREE and preemptable. One replicate is 1 epoch = ~3.5 h on 16
# GPUs (~56 GPU-h, measured -- docs/stage2_training_cost.md). Preemption is handled by
# --requeue plus train.py's own latest_checkpoint.pth resume, so a requeued replicate
# continues rather than restarting. Calendar, not money, is the risk here: ckpt-all's
# duty cycle was 3.9% in 2026-08 (#135), so a replicate can sit pending for days.
#SBATCH -p ckpt-all
#SBATCH --job-name=rampnet_seedvar
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/seedvar_%j.out
#SBATCH --error=logs/seedvar_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

set -euo pipefail

# The seed IS the experiment, so it is required rather than defaulted -- a replicate
# that silently ran at 42 would be a duplicate of the published run wearing a new name.
SEED="${SEED:?set SEED to the replicate's seed, e.g. SEED=1}"

REPO="${REPO:-$HOME/RampNet}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset}"
# NOT $HOME: klone home is a separate 10 GB quota that gscratch cleanup does not touch,
# and one replicate's per-epoch checkpoints alone would blow it.
RUNDIR="${RUNDIR:-/gscratch/scrubbed/$USER/seedvar/rampnet_s${SEED}}"
EPOCHS="${EPOCHS:-1}"        # the published recipe is 1 epoch / 9,378 steps (#84)

mkdir -p "$RUNDIR/checkpoints" "$REPO/logs"

echo "Loading Conda environment..."
source activate sidewalkcv2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# train.py lives in stage_two/ but imports the `rampnet` package from the repo root, and
# we run from RUNDIR, so neither is on sys.path by default.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

echo "--- Stage 2 seed replicate (#51 / #135) ---"
echo "Job ID:      ${SLURM_JOBID}"
echo "Seed:        ${SEED}"
echo "Run dir:     ${RUNDIR}   (cwd -- best_model.pth and latest_checkpoint.pth land here)"
echo "Data root:   ${DATA_ROOT}"
echo "Epochs:      ${EPOCHS}"
echo "Nodes:       ${SLURM_NNODES} x ${NPROC_PER_NODE} GPU  (world size ${WORLD_SIZE})"
echo "Node list:   ${SLURM_NODELIST}"
echo "Restarts:    ${SLURM_RESTART_COUNT:-0}   (requeue resumes from latest_checkpoint.pth)"
echo "-------------------------------------------"

cd "$RUNDIR"

srun --export=ALL \
    torchrun --nnodes $SLURM_NNODES \
             --nproc_per_node $NPROC_PER_NODE \
             --rdzv_id $SLURM_JOB_ID \
             --rdzv_backend c10d \
             --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
             "$REPO/stage_two/train.py" \
             --seed "$SEED" \
             --epochs "$EPOCHS" \
             --data-root "$DATA_ROOT" \
             --checkpoint-dir "$RUNDIR/checkpoints"

echo "--- Slurm job finished ---"
