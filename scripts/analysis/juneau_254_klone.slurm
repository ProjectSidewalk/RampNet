#!/bin/bash
# Run scripts/analysis/juneau_254_gpu.sh on Hyak (klone) as one Slurm job, for a
# subset of its STEPS. #254 moved here from makelab2 on 2026-10-19, when a stuck NFS
# mount wedged that host. The steps need different envs (YOLO has its own; Molmo
# needs transformers 4.57.1), so it is submitted once per env:
#
#   E=/gscratch/scrubbed/$USER/envs
#   STEPS="repro extract parity open qwen" PY=$E/qwenvl/bin/python \
#       sbatch -A gpu-l40s-makelab scripts/analysis/juneau_254_klone.slurm
#   STEPS="yolo" PY=$E/yolo/bin/python YOLO_DIR=<dir of y11l_pano.pt, y26_pano.pt,
#       y11x_pano_h200.pt> sbatch -A gpu-l40s-makelab scripts/analysis/juneau_254_klone.slurm
#   STEPS="molmo" MOLMO_PY=$E/molmo/bin/python PY=$E/qwenvl/bin/python \
#       sbatch -A gpu-l40s-makelab scripts/analysis/juneau_254_klone.slurm
#
# Qwen3-VL-32B needs two cards and goes through run_qwen.slurm instead
# (BUNDLE=benchmark/juneau QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct --gpus-per-node=2).
# Afterwards, scripts/analysis/slurm_usage.py records the GPU-hours in
# analysis_out/compute_log.jsonl.
#SBATCH -p gpu-l40s
#SBATCH --job-name=juneau_254
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/juneau_254_%j.out
set -u
: "${STEPS:?set STEPS}" "${PY:?set PY}"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
echo "HOST: $(hostname)  job ${SLURM_JOB_ID:-?}  STEPS: $STEPS"
export STEPS PY
bash scripts/analysis/juneau_254_gpu.sh
