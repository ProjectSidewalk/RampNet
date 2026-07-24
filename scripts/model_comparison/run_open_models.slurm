#!/bin/bash
# Open-weight leg of the model comparison, on Hyak (klone): the open-vocabulary
# detectors (OWLv2, Grounding DINO) and Ai2 Molmo. Sibling of run_qwen.slurm.
#
# Pass your Slurm account with -A; there is no usable default, and a submission
# without one is rejected. Find yours with:
#   sacctmgr -nP show assoc user=$USER format=Account,QOS
#
#   sbatch -A <account> scripts/model_comparison/run_open_models.slurm
#   BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_open_models.slurm
#   MODELS=rampnet,molmo:allenai/Molmo2-8B \
#       sbatch -A <account> scripts/model_comparison/run_open_models.slurm
#
# Sizes: OWLv2-large and Grounding DINO-base are ~1-2 GB and run in minutes, so
# both fit one card with room to spare. Molmo-8B is ~16 GB in bf16 — still one
# L40S, but hours rather than minutes (it generates text per view).
#
# The open-vocabulary detectors emit calibrated per-box scores, so the harness
# caches detections down to a low score FLOOR (--score-threshold, default 0.05)
# and every higher operating point is then a free local re-score (--op-threshold /
# --sweep). Lowering the floor later re-runs the model: it is in the cache key.
#
# PYTHON is the interpreter to run with; default is whatever `python` resolves to
# after the env is activated (see the runbook in docs/model_comparison.md).
#
# Detections land in .model_cache/, whose key contains nothing machine-specific —
# rsync that directory back and score locally with no GPU.
#SBATCH -p gpu-l40s
#SBATCH --job-name=open_curb_ramp_compare
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/open_compare_%j.out
#SBATCH --error=logs/open_compare_%j.err

set -euo pipefail

MODELS="${MODELS:-rampnet,owlv2,gdino}"
BUNDLE="${BUNDLE:-benchmark/richmond}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Home quota on klone is small; keep the HF cache on scratch and pre-download the
# weights on the login node so the GPU isn't billed for the transfer.
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    source activate sidewalkcv2
    PYTHON=python
fi

echo "--- open-model comparison ---"
echo "Models: ${MODELS}"
echo "Bundle: ${BUNDLE}"
echo "GPUs:   ${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}} on ${SLURMD_NODENAME:-?}"
echo "HF_HOME: ${HF_HOME}"
echo "Python:  $($PYTHON -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "-----------------------------"

"$PYTHON" -u scripts/model_comparison/compare.py "$BUNDLE" \
    --models "$MODELS" \
    --sweep \
    $EXTRA_ARGS

echo "--- done; rsync .model_cache/ back to score alongside the other models ---"
