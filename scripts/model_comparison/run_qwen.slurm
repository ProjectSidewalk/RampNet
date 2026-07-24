#!/bin/bash
# Qwen3-VL leg of the model comparison, on Hyak (klone).
#
# Pass your Slurm account with -A; there is no usable default, and a submission
# without one is rejected. Find yours with:
#   sacctmgr -nP show assoc user=$USER format=Account,QOS
#
#   sbatch -A <account> scripts/model_comparison/run_qwen.slurm
#   BUNDLE=benchmark/bend sbatch -A <account> scripts/model_comparison/run_qwen.slurm
#   QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch -A <account> --gpus=2 \
#       scripts/model_comparison/run_qwen.slurm
#
# 32B is ~64 GB in bf16, so it needs two 48 GB cards; device_map="auto" shards it
# across whatever the job is allocated. 8B fits one.
#
# --nodes=1 below is load-bearing for the multi-GPU case. `--gpus=2` alone lets
# Slurm satisfy the request with one GPU on each of two nodes; the process then
# sees a single card, and accelerate silently CPU-offloads the rest ("Some
# parameters are on the meta device...") instead of failing. The run does not
# crash, it just becomes unusably slow. Ask for GPUs per node, on one node:
#   sbatch -A <account> --nodes=1 --gpus-per-node=2 ...
# (--mem is per node, so it is not multiplied by the node count either.)
#
# PYTHON is the interpreter to run with; default is whatever `python` resolves to
# after the env is activated. On klone the comparison leg needs only
# numpy/PIL/torch/transformers (the RampNet baseline reads detections from the
# bundle, so no timm and no model load), so a lean venv works — see the runbook.
#
# Detections land in .model_cache/, whose key contains nothing machine-specific —
# rsync that directory back and score locally with no GPU (see docs/model_comparison.md).
#SBATCH -p gpu-l40s
#SBATCH --job-name=qwen_curb_ramp_compare
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
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

# PYTHON points at a ready interpreter (e.g. a venv on gscratch). Falls back to
# activating the full conda env when unset.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    source activate sidewalkcv2
    PYTHON=python
fi

echo "--- Qwen model comparison ---"
echo "Model:  ${QWEN_MODEL}"
echo "Bundle: ${BUNDLE}"
echo "GPUs:   ${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}} on ${SLURMD_NODENAME:-?}"
echo "HF_HOME: ${HF_HOME}"
echo "Python:  $($PYTHON -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "-----------------------------"

"$PYTHON" -u scripts/model_comparison/compare.py "$BUNDLE" \
    --models "rampnet,qwen:${QWEN_MODEL}" \
    --qwen-model "$QWEN_MODEL" \
    $EXTRA_ARGS

echo "--- done; rsync .model_cache/ back to score alongside Gemini ---"
