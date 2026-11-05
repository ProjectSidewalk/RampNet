#!/bin/bash
# Inference for the #274 crosswalk detector's pre-registered scoring, on klone (one GPU, ckpt-all, free).
# Runs `score_274.py infer` for the fine-tuned checkpoint and for the released curb-ramp weights (the floor),
# over the Stage 2 root's val + test splits. Each label refuses a checkpoint whose sha256 differs from the one
# pinned below, and resumes per pano after a requeue. Scoring itself (`score_274.py score`) is CPU-only and
# runs wherever the predictions are copied to.
#
# Usage, from the repo root of a checkout at the scorer's commit:
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 sbatch scripts/analysis/stage1_crosswalk_sdot_274/score_274.slurm
#SBATCH -p ckpt-all
#SBATCH --job-name=cw274_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cw274_score_%j.out
#SBATCH --error=logs/cw274_score_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'
set -u
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="${ROOT:-/gscratch/scrubbed/$USER/crosswalk274/root}"
PREDS="${PREDS:-/gscratch/scrubbed/$USER/crosswalk274/score_preds}"
FT_CKPT="${FT_CKPT:-/gscratch/makelab/jonf/crosswalk274/s1/final_step_1000.pth}"
FT_SHA=6f467ff4afd7d092e159daa0451f327f755bfd6767769c417a7729f92b6c8c84
REL_CKPT="${REL_CKPT:-/gscratch/scrubbed/$USER/aug82/released_rampnet_state_dict.pth}"
REL_SHA=024a987cabed7cc9f70dea9ea4003aef416cf5d8d3ce35a3f0c16c31ef35c4b1   # Hub 606a119, docs/control_arm_263.md
export PYTHONDONTWRITEBYTECODE=1
ENV_TAR="${ENV_TAR:-/gscratch/scrubbed/$USER/control263/sidewalkcv2_env.tar}"
if [ -f "$ENV_TAR" ]; then
    LOCAL_ENV_DIR="/scr/$USER/cw274score_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "$LOCAL_ENV_DIR"
    trap 'rm -rf "$LOCAL_ENV_DIR"' EXIT
    t_env=$(date +%s)
    tar xf "$ENV_TAR" -C "$LOCAL_ENV_DIR"
    RAMPNET_ENV="$LOCAL_ENV_DIR/sidewalkcv2"
    echo "env unpacked to $RAMPNET_ENV in $(( $(date +%s) - t_env )) s"
fi
PY="$RAMPNET_ENV/bin/python"
echo "job ${SLURM_JOB_ID:-local} restarts ${SLURM_RESTART_COUNT:-0} node ${SLURM_NODELIST:-?} commit $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
S="$REPO/scripts/analysis/stage1_crosswalk_sdot_274/score_274.py"
status=0
"$PY" "$S" infer --root "$ROOT" --preds "$PREDS" --label crosswalk_s1 --checkpoint "$FT_CKPT" --sha256 "$FT_SHA" || status=1
"$PY" "$S" infer --root "$ROOT" --preds "$PREDS" --label released --checkpoint "$REL_CKPT" --sha256 "$REL_SHA" || status=1
echo "done, status $status"
exit $status
