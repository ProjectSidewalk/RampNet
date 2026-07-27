#!/bin/bash
# Full YOLO prep (issue #51): build the tiles + pano training datasets from the ~214k-pano
# RampNet dataset with box-size=pitch (distance-aware; fixed:0.03 was too small and gps fell
# back to pitch on ~85% of panos, so pitch is the consistent choice). Runs on a COMPUTE node:
# the 150k-pano tiling is multi-hour and CPU-heavy, and klone's login nodes reap heavy procs
# (that reap also kills the ssh control-master). Writes to shared scratch.
#
#   sbatch -A gpu-l40s-makelab scripts/model_comparison/run_yolo_prep.slurm
#   # override box model: BOX_SIZE=pitch RAMP_SIZE_M=1.8 sbatch ...
#
#SBATCH -p gpu-l40s
#SBATCH --job-name=yolo_prep
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/home/jfroehli/RampNet/logs/yolo_prep_%j.out
#SBATCH --error=/mmfs1/home/jfroehli/RampNet/logs/yolo_prep_%j.err
set -uo pipefail

REPO="${REPO:-/mmfs1/home/jfroehli/RampNet}"
PY="${PYBIN:-/gscratch/scrubbed/jfroehli/envs/yolo/bin/python}"
Y="${YOLODATA:-/gscratch/scrubbed/jfroehli/yolo}"
BOX="${BOX_SIZE:-pitch}"
RAMP="${RAMP_SIZE_M:-1.8}"
BG="${BG_KEEP:-0.15}"
W="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
export OPENBLAS_NUM_THREADS=1   # one BLAS thread per worker (no RLIMIT_NPROC storm)
cd "$REPO"

echo "node: $(hostname)   start: $(date -Is)   cpus: ${SLURM_CPUS_PER_TASK:-?}   box=$BOX ramp=${RAMP}m workers=$W"

echo "=== TILES ==="
"$PY" scripts/model_comparison/prepare_yolo_dataset.py \
  --dataset-root "$REPO/dataset" --out "$Y/tiles" --geometry tiles \
  --box-size "$BOX" --ramp-size-m "$RAMP" --bg-keep-frac "$BG" --workers "$W"

echo "=== PANO ==="
"$PY" scripts/model_comparison/prepare_yolo_dataset.py \
  --dataset-root "$REPO/dataset" --out "$Y/pano" --geometry pano \
  --box-size "$BOX" --ramp-size-m "$RAMP" --workers "$W"

echo "=== output image/label counts (glob-free) ==="
for g in tiles pano; do
  for d in images labels; do
    for s in train val; do
      printf "  %-5s %-6s %-5s = %s\n" "$g" "$d" "$s" "$(ls -U "$Y/$g/$d/$s" 2>/dev/null | wc -l)"
    done
  done
done
echo "ALL DONE @ $(date -Is)"
