#!/bin/bash
#SBATCH --job-name=mv48
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40|l40|a100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/scrubbed/jfroehli/mv48/logs/%x_%A_%a.out
# #48 klone offload of makelab2's pass-2 legs. One array task = one shard of the 1,307
# panos the 30 m widening added (plus the 124 judged panos, which are already cached
# from makelab2 and must report as such). Same compare.py args as
# scripts/analysis/multiview_challengers_48.sh. Resumable: detections cache per pano.
#   sbatch --array=0-3 --job-name=mv48-molmo run_mv48.slurm molmo
# GPU: any one 40 GB+ GPU. The 2026-09-27 arrays were submitted with --gres=gpu:l40s:1 and
# widened while pending, before 8 of the 9 tasks started, with
#   scontrol update jobid=<id> gres=gpu:1 features="l40s|a40|l40|a100|h200"
# so this file now requests what those 8 tasks ran under (1 L40S, 6 A40, 2 A100).
# Paths: W below and --output above are Jon's scratch dir; --account is his lab's.
set -uo pipefail
LEG=$1
W=${W:-/gscratch/scrubbed/jfroehli/mv48}
SH=$W/shards/$LEG/s${SLURM_ARRAY_TASK_ID}
case $LEG in
  molmo)  PY=$W/envs/molmo/bin/python; M=molmo:allenai/Molmo2-8B ;;
  qwen8b) PY=$W/envs/eval/bin/python;  M=qwen:Qwen/Qwen3-VL-8B-Instruct ;;
  open)   PY=$W/envs/eval/bin/python;  M=owlv2,gdino ;;
  *) echo "unknown leg $LEG"; exit 2 ;;
esac
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/RampNet
echo "=== $LEG shard ${SLURM_ARRAY_TASK_ID} $(hostname) $(git rev-parse --short HEAD) restart=${SLURM_RESTART_COUNT:-0} $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import sys,torch,transformers;print('python',sys.version.split()[0],'torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())"
$PY -u scripts/model_comparison/compare.py $SH/benchmark/richmond_neighbourhood --detect-unjudged \
    --models "$M" --cache-dir $W/model_cache --usage-log $W/usage_log.jsonl
echo "=== exit=$? $(date -Is)"
