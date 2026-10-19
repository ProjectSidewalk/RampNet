#!/bin/bash
# Run scripts/analysis/juneau_254_gpu.sh on Hyak (klone) as one Slurm job, for a
# subset of its STEPS. #254 moved here from makelab2 on 2026-10-19, when a stuck NFS
# mount wedged that host. Build the envs first (scripts/analysis/juneau_254_klone_env.sh);
# the steps need different envs, so it is submitted once per env. As run 2026-10-19, on
# the free ckpt-g2 partition because gpu-l40s-makelab was at its AssocGrpCpuLimit:
#
#   E=/gscratch/scrubbed/$USER/envs
#   export HF_HOME=/gscratch/scrubbed/$USER/hf XDG_CACHE_HOME=... TORCH_HOME=...   # off $HOME
#   STEPS="repro extract parity open" PY=$E/rampnet-eval/bin/python \
#       sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/juneau_254_klone.slurm
#   STEPS="yolo" PY=$E/rampnet-eval/bin/python YOLO_DIR=<dir of y11l_pano.pt, y26_pano.pt,
#       y11x_pano_h200.pt> sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/juneau_254_klone.slurm
#   STEPS="molmo" PY=$E/rampnet-eval2/bin/python MOLMO_PY=$E/molmo3/bin/python \
#       sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/juneau_254_klone.slurm
#   STEPS="qwen" PY=$E/rampnet-eval2/bin/python \
#       sbatch -A ckpt-makelab -p ckpt-g2 --export=ALL scripts/analysis/juneau_254_klone.slurm
#
# The YOLO_DIR links are rampnet_yolo_baseline_51/<arm>/weights/best.pt, named <arm>.pt (the
# leg's label comes from the file name; sha1 must equal the published signature's
# weights_hash). Qwen3-VL-32B needs two cards and goes through run_qwen.slurm instead
# (BUNDLE=benchmark/juneau QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct PYTHON=$E/rampnet-eval2/bin/python,
# --gpus-per-node=2). Afterwards, scripts/analysis/slurm_usage.py records the GPU-hours in
# analysis_out/compute_log.jsonl (docs/compute_cost.md).
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
