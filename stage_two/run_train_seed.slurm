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
# USAGE. Submit from the repo root; logs/ must ALREADY exist, because Slurm opens
# --output relative to the SUBMIT directory before this script runs (the mkdir below
# cannot help with that -- it is there for a RUNDIR that does not exist yet).
#   mkdir -p logs
#   SEED=1 sbatch stage_two/run_train_seed.slurm
#   SEED=2 RUNDIR=/gscratch/scrubbed/$USER/seedvar/rampnet_s2 sbatch stage_two/run_train_seed.slurm
#
# RUNDIR IS ON A VOLUME THAT PURGES. best_model.pth -- the only artifact this campaign
# produces -- lands in RUNDIR, and /gscratch/scrubbed purges on a ~21-day idle window
# (docs/stage2_epoch_curve_84.md). Campaign B's calendar is unbounded (ckpt-all duty
# cycle 3.9%), so a replicate can finish and then sit unscored past that window while
# its siblings are still pending. COPY IT OUT as soon as the job completes:
#   cp "$RUNDIR/best_model.pth" /gscratch/makelab/$USER/seedvar/rampnet_s${SEED}_best.pth
# /gscratch/makelab is purchased and never purged, which is where #84 put Run A's
# checkpoints for the same reason. The default is deliberately left on scrubbed so it
# matches the replicates already queued; changing it is a decision, not a cleanup.
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
# NO APOSTROPHE in this message. Inside ${VAR:?...} bash parses the word for quoting
# even within double quotes, so a lone ' opens a quote and the closing } is never found:
# "unexpected EOF while looking for matching `}'". The script then dies at submit time
# with exit 2 in about one second, having printed nothing -- which is how jobs
# 39515025/26/27 failed on 2026-09-03 and then sat unnoticed for a day.
SEED="${SEED:?set SEED to the seed for this replicate, e.g. SEED=1}"

REPO="${REPO:-$HOME/RampNet}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset}"
# NOT $HOME: klone home is a separate 10 GB quota that gscratch cleanup does not touch,
# and one replicate's per-epoch checkpoints alone would blow it.
RUNDIR="${RUNDIR:-/gscratch/scrubbed/$USER/seedvar/rampnet_s${SEED}}"
EPOCHS="${EPOCHS:-1}"        # the published recipe is 1 epoch / 9,378 steps (#84)

mkdir -p "$RUNDIR/checkpoints" "$REPO/logs"

# Interpreter. Default behaviour is unchanged -- `source activate sidewalkcv2`, then
# torchrun off PATH -- but every other klone launcher here carries an escape hatch
# (PYTHON= in run_yolo_train.slurm and run_gold_bundle.slurm, RAMPNET_ENV in
# run_train_epoch_curve.slurm) and this one did not. It matters because #84's env was
# built at a PREFIX, /gscratch/scrubbed/$USER/envs/sidewalkcv2, which `source activate
# <name>` cannot resolve: under `set -e` the job would then die here having printed one
# line. Set RAMPNET_ENV to a conda prefix and the env's own torchrun is called by
# absolute path instead -- the pattern run_train_epoch_curve.slurm documents as the one
# proven on this cluster.
RAMPNET_ENV="${RAMPNET_ENV:-}"
if [ -n "$RAMPNET_ENV" ]; then
    TORCHRUN="$RAMPNET_ENV/bin/torchrun"
    if [ ! -x "$TORCHRUN" ]; then
        echo "FATAL: RAMPNET_ENV=$RAMPNET_ENV has no executable bin/torchrun" >&2
        exit 1
    fi
    echo "Using conda prefix ${RAMPNET_ENV} (no activation)"
else
    echo "Loading Conda environment..."
    source activate sidewalkcv2
    TORCHRUN=torchrun
fi

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

# World size IS the global batch: train.py uses batch_size=1 per rank, so a replicate
# that lands on any other node/GPU count is a different optimisation regime, not a seed
# replicate of the published recipe. Same guard as run_train_epoch_curve.slurm.
if [ "${WORLD_SIZE}" -ne 16 ]; then
    echo "WARNING: world size ${WORLD_SIZE} != 16. The published recipe's global batch" >&2
    echo "         was 16; this is NOT a seed replicate of it at any other world size." >&2
fi

cd "$RUNDIR"

srun --export=ALL \
    "$TORCHRUN" --nnodes $SLURM_NNODES \
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
