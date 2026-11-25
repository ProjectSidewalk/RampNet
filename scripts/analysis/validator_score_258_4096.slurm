#!/bin/bash
# The #258 manual_gold resolution control (PR #269 review, finding 4): richmond's native
# panos downsampled to 4096x2048 with manual_gold's own transform
# (scripts/analysis/downsample_panos_258.py), then scored by the same validator scorer
# into analysis_out/verifier_258/validator_scores_4096/richmond.json. Same inputs as
# validator_score_258.slurm, plus STAGE, a shared-storage directory for the downsampled
# copy (never /tmp: klone's /tmp is node-local).
#
#   PY=/gscratch/scrubbed/$USER/envs/rampnet-eval/bin/python \
#   PANOS_ROOT=/gscratch/makelab/jonf/rampnet_benchmark \
#   STAGE=/gscratch/makelab/jonf/verifier258_4096 \
#   DAV2_DIR=... DAV2_CKPT=... HF_HOME=... \
#       sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/validator_score_258_4096.slurm
#SBATCH -p ckpt-g2
#SBATCH --job-name=validator_258_4096
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --output=logs/validator_258_4096_%j.out
set -euo pipefail
: "${PY:?set PY}" "${PANOS_ROOT:?set PANOS_ROOT}" "${STAGE:?set STAGE}" \
  "${DAV2_DIR:?set DAV2_DIR}" "${DAV2_CKPT:?set DAV2_CKPT}"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}" HF_HUB_OFFLINE=1
echo "HOST: $(hostname)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The native panos must be the reviewed bytes before they are downsampled.
ln -sfn "$PANOS_ROOT/richmond/panos" "benchmark/richmond/panos"
"$PY" scripts/analysis/imagery_manifest.py --verify --panos-root . --cities richmond
"$PY" scripts/analysis/downsample_panos_258.py --src "$PANOS_ROOT/richmond/panos" \
    --dst "$STAGE/richmond/panos"
"$PY" -u scripts/analysis/validator_score_258.py --panos-root "$STAGE" --splits richmond \
    --dav2-dir "$DAV2_DIR" --dav2-ckpt "$DAV2_CKPT" \
    --out-dir analysis_out/verifier_258/validator_scores_4096 \
    --panos-note "richmond native panos downsampled to 4096x2048 by downsample_panos_258.py (manual_gold's transform), then resized to 8192x4096 as serving does"
