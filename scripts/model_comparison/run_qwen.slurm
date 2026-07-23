#!/bin/bash
# Qwen3-VL leg of the model comparison, on Hyak (klone).
#
#   sbatch scripts/model_comparison/run_qwen.slurm
#   BUNDLE=benchmark/bend sbatch scripts/model_comparison/run_qwen.slurm
#   QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch --gpus=2 scripts/model_comparison/run_qwen.slurm
#
# 32B is ~64 GB in bf16, so it needs two 48 GB cards; device_map="auto" shards it
# across whatever the job is allocated. 8B fits one.
#
# Detections land in .model_cache/, whose key contains nothing machine-specific —
# rsync that directory back and score locally with no GPU (see docs/model_comparison.md).
#SBATCH -p gpu-l40s
#SBATCH --job-name=qwen_curb_ramp_compare
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/qwen_compare_%j.out
#SBATCH --error=logs/qwen_compare_%j.err

set -euo pipefail

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BUNDLE="${BUNDLE:-benchmark/richmond}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Home quota on klone is small and the checkpoints are tens of GB; keep the HF
# cache on scratch. Pre-download on the login node so the GPU isn't billed for it:
#   HF_HOME=... hf download "$QWEN_MODEL"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"

source activate sidewalkcv2

echo "--- Qwen model comparison ---"
echo "Model:  ${QWEN_MODEL}"
echo "Bundle: ${BUNDLE}"
echo "GPUs:   ${SLURM_GPUS:-1} on ${SLURMD_NODENAME:-?}"
echo "HF_HOME: ${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "-----------------------------"

python -u scripts/model_comparison/compare.py "$BUNDLE" \
    --models "rampnet,qwen:${QWEN_MODEL}" \
    --qwen-model "$QWEN_MODEL" \
    $EXTRA_ARGS

echo "--- done; rsync .model_cache/ back to score alongside Gemini ---"
