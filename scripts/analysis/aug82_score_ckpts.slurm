#!/bin/bash
# Score #82 screen checkpoints on all 12 bundles, on klone (one GPU, ckpt-all, free).
# A thin wrapper around scripts/analysis/aug82_score_ckpts.sh (read its header): the
# makelab2 A40 was busy with the Step 1 probe and shared with another job, so scoring
# moved to klone, where the checkpoints already are.
#
# Prerequisites in the checkout this is submitted from (the repo root):
#   benchmark/<split>/panos -> /gscratch/makelab/jonf/rampnet_benchmark/<split>/panos
#   benchmark/manual_gold/panos -> /gscratch/scrubbed/jfroehli/manual_gold/panos
#   logs/ exists; checkpoints in $CKPTS (default /gscratch/makelab/jonf/aug82) as <label>.pth
#     with a SHA256SUMS beside them.
# Usage:
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS="released" \
#       sbatch scripts/analysis/aug82_score_ckpts.slurm
#SBATCH -p ckpt-all
#SBATCH --job-name=aug82_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/aug82_score_%j.out
#SBATCH --error=logs/aug82_score_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'
set -u
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix}"
LABELS="${LABELS:?set LABELS, e.g. LABELS=released or LABELS=control_s1 both_s1}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONDONTWRITEBYTECODE=1   # see run_finetune_aug82.slurm
export HF_HUB_OFFLINE=1          # the released weights come from the existing hub cache
# Node-local copy of the env when its tarball exists (see run_finetune_aug82.slurm: reading
# the env from GPFS stalled imports for 25-36 min on 2026-10-02).
ENV_TAR="${ENV_TAR:-/gscratch/scrubbed/$USER/aug82/sidewalkcv2_env.tar}"
if [ -f "$ENV_TAR" ]; then
    LOCAL_ENV_DIR="/scr/$USER/aug82score_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "$LOCAL_ENV_DIR"
    trap 'rm -rf "$LOCAL_ENV_DIR"' EXIT
    t_env=$(date +%s)
    tar xf "$ENV_TAR" -C "$LOCAL_ENV_DIR"
    RAMPNET_ENV="$LOCAL_ENV_DIR/sidewalkcv2"
    echo "env unpacked to $RAMPNET_ENV in $(( $(date +%s) - t_env )) s"
fi
echo "job ${SLURM_JOB_ID} restarts ${SLURM_RESTART_COUNT:-0} node ${SLURM_NODELIST} labels: ${LABELS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# OUT is per job label set; extract skips splits already written, so a requeue resumes.
PY="$RAMPNET_ENV/bin/python" CKPTS="${CKPTS:-/gscratch/makelab/jonf/aug82}" \
    bash "$REPO/scripts/analysis/aug82_score_ckpts.sh" $LABELS
