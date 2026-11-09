#!/bin/bash
# #276 Step 3 on an open-weight model, on Hyak (klone): the dev-split prompt battery
# that the Gemini legs ran, so the definition result is measured on three model
# families rather than one.
#
#   sbatch -A gpu-l40s-makelab --nodes=1 --gpus-per-node=2 --mem=128G \
#       scripts/model_comparison/run_qwen_276.slurm                 # Qwen3-VL-32B (default)
#   QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct sbatch -A gpu-l40s-makelab \
#       scripts/model_comparison/run_qwen_276.slurm                 # 8B fits one card
#
# Legs, in order (each is an ordinary compare.py run with --pano-list, so each writes
# its own ledger row and lands in the detection cache under its own key):
#   seed_repeat   seed prompt on dev_train in a FRESH cache dir  -> the noise floor
#   para1..3      three meaning-preserving paraphrases on dev_select -> the wording null
#   expert        Jon's definition on dev_select
#   expert-long   its 515-word structured form on dev_select
# The seed on dev_select itself is already in the published 14-split leg.
#
# LEGS overrides the list (space-separated names from the set above). Detections come
# home with `rsync klone:RampNet/.model_cache/ .model_cache/` and
# `rsync klone:RampNet/.model_cache_rep1/ .model_cache_rep1/`; scoring needs no GPU.
#SBATCH -p gpu-l40s
#SBATCH --job-name=qwen276_prompt_battery
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/qwen276_%j.out
#SBATCH --error=logs/qwen276_%j.err

set -euo pipefail

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
LEGS="${LEGS:-seed_repeat para1 para2 para3 expert expert-long}"
SPLIT="benchmark/prompt_opt_276/sao_paulo_split.json"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    source activate sidewalkcv2
    PYTHON=python
fi

echo "--- #276 prompt battery ---"
echo "Model:  ${QWEN_MODEL}"
echo "Legs:   ${LEGS}"
echo "GPUs:   ${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}} on ${SLURMD_NODENAME:-?}"
echo "HF_HOME: ${HF_HOME}"
echo "Python:  $($PYTHON -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "---------------------------"

run() {  # name, then extra compare.py args
    local name="$1"; shift
    echo; echo "=== $(date '+%F %T') leg ${name}"
    "$PYTHON" -u scripts/model_comparison/compare.py benchmark/sao_paulo \
        --models "qwen:${QWEN_MODEL}" --qwen-model "$QWEN_MODEL" "$@"
    echo "=== $(date '+%F %T') leg ${name} done"
}

for leg in $LEGS; do
    case "$leg" in
        seed_repeat) run "$leg" --pano-list "${SPLIT}@dev_train" --cache-dir .model_cache_rep1 ;;
        para1|para2|para3|expert|expert-long)
                     run "$leg" --pano-list "${SPLIT}@dev_select" --definition "$leg" ;;
        *) echo "unknown leg ${leg}"; exit 2 ;;
    esac
done
echo "--- $(date '+%F %T') battery complete; rsync .model_cache/ and .model_cache_rep1/ home ---"
