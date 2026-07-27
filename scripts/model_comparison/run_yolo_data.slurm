#!/bin/bash
# Download the RampNet dataset (~214k panos) from HF onto scratch, as a Slurm job.
# This is the `data` stage of hyak_yolo_runbook.sh, wrapped so the multi-hour,
# 8-process JPEG re-encode runs on a COMPUTE node rather than the login node —
# klone's login nodes kill heavy, long-running processes, which would silently
# abort an unattended download partway through.
#
# The GPU is requested only to satisfy the gpu-l40s partition's constraints; the
# download itself is CPU/IO-bound (HF parquet fetch + `.map` re-encode) and does
# not touch it. Idempotent: HF caches shards and the per-pano files are written
# incrementally, so a re-run resumes rather than starting over.
#
#   mkdir -p logs
#   sbatch -A gpu-l40s-makelab scripts/model_comparison/run_yolo_data.slurm
#
#SBATCH -p gpu-l40s
#SBATCH --job-name=yolo_data_dl
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/yolo_data_%j.out
#SBATCH --error=logs/yolo_data_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/RampNet}"
cd "$REPO"

echo "--- RampNet dataset download (issue #51 data stage) ---"
echo "node:   ${SLURMD_NODENAME:-?}"
echo "cpus:   ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?}M"
echo "start:  $(date -Is)"
echo "-------------------------------------------------------"

# Single source of truth for the download procedure: the runbook's `data` stage
# sets HF_HOME/PYBIN/DATA, symlinks ./dataset -> scratch, and runs download_dataset.py.
bash "$REPO/hyak_yolo_runbook.sh" data

echo "--- download job done: $(date -Is) ---"
