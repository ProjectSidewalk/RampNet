#!/bin/bash
# Step 2 of #258: score every verifier candidate with the production CurbRamp validator
# (scripts/analysis/validator_score_258.py), on klone, one GPU, on the free ckpt partition.
# The scorer resumes per pano, so a preempted (requeued) job picks up where it stopped.
#
# Inputs on klone, as run on 2026-10-22:
#   PANOS_ROOT  /gscratch/makelab/jonf/rampnet_benchmark  (<split>/panos for all 14 splits;
#               juneau and bayonne copied in on 2026-10-22, every split checked against its
#               imagery_manifest.json first)
#   DAV2_DIR    a Depth-Anything-V2 checkout at a561b84 (sidewalk-ai-api's submodule pin)
#   DAV2_CKPT   depth_anything_v2_metric_vkitti_vitb.pth (sha256 asserted by the scorer)
#   HF_HOME     a cache holding projectsidewalk/sidewalk-validator-ai-curbramp@969ad5a3;
#               compute nodes cannot reach HF Xet, so it must be pre-cached (HF_HUB_OFFLINE=1)
#
#   PY=/gscratch/scrubbed/$USER/envs/rampnet-eval/bin/python \
#   PANOS_ROOT=... DAV2_DIR=... DAV2_CKPT=... HF_HOME=... \
#       sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/validator_score_258.slurm
#
# Afterwards, scripts/analysis/slurm_usage.py records the GPU-hours in
# analysis_out/compute_log.jsonl (docs/compute_cost.md).
#SBATCH -p ckpt-g2
#SBATCH --job-name=validator_258
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --output=logs/validator_258_%j.out
set -euo pipefail
: "${PY:?set PY}" "${PANOS_ROOT:?set PANOS_ROOT}" "${DAV2_DIR:?set DAV2_DIR}" "${DAV2_CKPT:?set DAV2_CKPT}"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}" HF_HUB_OFFLINE=1
echo "HOST: $(hostname)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The panos must be the bytes every review was made against; set -e stops the run otherwise.
for s in ${SPLITS:-$("$PY" -c 'from rampnet.eval import PINNED_SPLITS; print(*PINNED_SPLITS)')}; do
  ln -sfn "$PANOS_ROOT/$s/panos" "benchmark/$s/panos"
done
"$PY" scripts/analysis/imagery_manifest.py --verify --panos-root . ${SPLITS:+--cities ${SPLITS// /,}}
"$PY" -u scripts/analysis/validator_score_258.py --panos-root "$PANOS_ROOT" \
    --dav2-dir "$DAV2_DIR" --dav2-ckpt "$DAV2_CKPT" ${SPLITS:+--splits $SPLITS}
