#!/bin/bash
# Score the #263 control arm (released weights fine-tuned on GSV only, 2,000 steps, seeds
# 1-3) and the released checkpoint on all 14 reviewed benchmark splits, on klone (one GPU,
# ckpt-all, free). One job per label; the four run side by side.
#
# A thin wrapper around scripts/analysis/aug82_score_ckpts.sh -- the #82 screen's scorer,
# itself the seed-variance campaign's frozen call (operating_point_curve.py extract
# [--checkpoint ...] --score-floor 0.05 --min-distance 10, no TTA, fp32, one --cache
# directory per checkpoint). Read that header. What differs from aug82_score_ckpts.slurm:
# the split list (bayonne and juneau were reviewed after #82 and are added), OUT, CKPTS,
# LABEL_PREFIX=control263_, and the env tarball path (the #82 one may be purged; the
# control-arm training jobs rebuilt theirs under control263/).
#
# Prerequisites in the checkout this is submitted from (the repo root):
#   benchmark/<split>/panos -> /gscratch/makelab/jonf/rampnet_benchmark/<split>/panos
#     for all 14 splits (the counts must match analysis_out/op_cache/<split>.json n_panos;
#     extract scores only the panos present, so a short directory gives a short cache);
#   logs/ exists;
#   $CKPTS/control_s<seed>.pth (symlinks to s<seed>/final_step_2000.pth, which the training
#     jobs copied there) with a SHA256SUMS beside them, which the scorer checks first.
# Usage (docs/control_arm_263.md has the exact sequence):
#   for L in released control_s1 control_s2 control_s3; do
#       RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS=$L \
#           sbatch scripts/analysis/control263_score.slurm
#   done
# The #82 screen's two controls are the same recipe (2,000 steps, global batch 16, seeds 1
# and 2) on the earlier copy of the dataset, and were scored on 12 splits before bayonne and
# juneau existed. Their two missing splits, into their own OUT so nothing of #82's moves:
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 LABELS="control_s1 control_s2" \
#       CKPTS=/gscratch/makelab/jonf/aug82 LABEL_PREFIX=aug82_ SPLITS=bayonne,juneau \
#       OUT=$PWD/analysis_out/control_arm_263/caches_aug82_controls \
#       sbatch scripts/analysis/control263_score.slurm
#SBATCH -p ckpt-all
#SBATCH --job-name=control263_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/control263_score_%j.out
#SBATCH --error=logs/control263_score_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'
set -u
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix}"
LABELS="${LABELS:?set LABELS, e.g. LABELS=released or LABELS=control_s1}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONDONTWRITEBYTECODE=1   # see run_finetune_aug82.slurm
export HF_HUB_OFFLINE=1          # released weights + backbone from the existing hub cache;
                                 # the klone home is over quota, so nothing may write there
ENV_TAR="${ENV_TAR:-/gscratch/scrubbed/$USER/control263/sidewalkcv2_env.tar}"
if [ -f "$ENV_TAR" ]; then
    LOCAL_ENV_DIR="/scr/$USER/ctrl263score_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "$LOCAL_ENV_DIR"
    trap 'rm -rf "$LOCAL_ENV_DIR"' EXIT
    t_env=$(date +%s)
    tar xf "$ENV_TAR" -C "$LOCAL_ENV_DIR"
    RAMPNET_ENV="$LOCAL_ENV_DIR/sidewalkcv2"
    echo "env unpacked to $RAMPNET_ENV in $(( $(date +%s) - t_env )) s"
fi
echo "job ${SLURM_JOB_ID} restarts ${SLURM_RESTART_COUNT:-0} node ${SLURM_NODELIST} labels: ${LABELS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# OUT is shared by the four jobs: each label writes its own subdirectory, and extract skips
# splits already written, so a requeue resumes. driver.log interleaves the jobs' lines.
PY="$RAMPNET_ENV/bin/python" \
    CKPTS="${CKPTS:-/gscratch/makelab/jonf/control263}" \
    OUT="${OUT:-$REPO/analysis_out/control_arm_263/caches}" \
    LABEL_PREFIX="${LABEL_PREFIX:-control263_}" \
    SPLITS="${SPLITS:-annapolis,bayonne,bend,budapest_district5,clovis,gainesville,juneau,laurens_gsv,laurens_mapillary,manual_gold,morgantown,paterson,richmond,sao_paulo}" \
    bash "$REPO/scripts/analysis/aug82_score_ckpts.sh" $LABELS
