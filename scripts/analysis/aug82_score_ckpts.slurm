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
export HF_HUB_OFFLINE=1          # the released weights come from the existing hub cache
echo "job ${SLURM_JOB_ID} restarts ${SLURM_RESTART_COUNT:-0} node ${SLURM_NODELIST} labels: ${LABELS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# OUT is per job label set; extract skips splits already written, so a requeue resumes.
PY="$RAMPNET_ENV/bin/python" CKPTS="${CKPTS:-/gscratch/makelab/jonf/aug82}" \
    bash "$REPO/scripts/analysis/aug82_score_ckpts.sh" $LABELS
