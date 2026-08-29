#!/bin/bash
# Rebuild the purged YOLO tile labels from the rescued Ultralytics caches (#51).
#
# This is a batch job rather than a login-node one-liner for one reason: it creates
# ~718,000 small files on GPFS (557,413 train + 161,002 val), which is metadata-heavy
# and sustained. klone reaps heavy login processes and that reap also kills the SSH
# control master.
#
# It reads the DURABLE cache copies under /gscratch/makelab (purchased, never purged),
# not the ones on /gscratch/scrubbed -- those are the copies whose siblings were already
# lost once, and rebuilding from them would make the recovery depend on the thing that
# failed. The rescued copies were staged 2026-08-17 with sha256sums.txt beside them.
#
#   sbatch scripts/model_comparison/run_rebuild_yolo_labels.slurm
#
# Env overrides: RAMPNET_REPO, YOLO_CACHE_DIR, YOLO_DATA_ROOT.
#
#SBATCH --job-name=yolo_label_rebuild_51
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/yolo_label_rebuild_%j.out
#SBATCH --error=logs/yolo_label_rebuild_%j.err
#SBATCH --requeue

set -eo pipefail

REPO="${RAMPNET_REPO:-/gscratch/makelab/jonf/lrcheck_135}"
CACHE_DIR="${YOLO_CACHE_DIR:-/gscratch/makelab/jonf/rampnet_yolo_baseline_51/label_cache_rescue}"
DATA_ROOT="${YOLO_DATA_ROOT:-/gscratch/scrubbed/jfroehli/yolo/tiles}"
PY="${YOLO_PY:-/gscratch/makelab/jonf/envs/yolo/bin/python}"
SCRIPT="$REPO/scripts/model_comparison/rebuild_yolo_labels_from_cache.py"

echo "--- YOLO label rebuild (#51) ---"
echo "Job ID     : ${SLURM_JOBID}"
echo "Node       : ${SLURMD_NODENAME:-?}"
echo "Cache dir  : $CACHE_DIR"
echo "Data root  : $DATA_ROOT"
echo "Script     : $SCRIPT"
echo "Python     : $PY"
echo "-------------------------------"

# The rescued caches carry their own checksums. Verify before trusting them as the
# single source for 968,227 boxes.
if [ -f "$CACHE_DIR/sha256sums.txt" ]; then
    echo "=== verifying rescued caches against sha256sums.txt ==="
    ( cd "$CACHE_DIR" && sha256sum -c sha256sums.txt ) || { echo "FATAL: cache checksum mismatch"; exit 1; }
else
    echo "WARNING: no sha256sums.txt beside the caches -- proceeding unverified"
fi

for split in train val; do
    echo
    echo "=== $split $(date -Is) ==="
    "$PY" "$SCRIPT" \
        --cache "$CACHE_DIR/${split}.cache" \
        --labels-dir "$DATA_ROOT/labels/${split}" \
        --verify
done

echo
echo "=== stale .cache files ==="
# Ultralytics rebuilds a cache whose hash no longer matches, so these are not harmful --
# but they are what made the failure confusing, so move them aside rather than trust them.
for split in train val; do
    if [ -f "$DATA_ROOT/labels/${split}.cache" ]; then
        mv -v "$DATA_ROOT/labels/${split}.cache" "$DATA_ROOT/labels/${split}.cache.pre-rebuild"
    fi
done

echo
echo "=== final counts ==="
for split in train val; do
    echo "  labels/$split : $(ls "$DATA_ROOT/labels/$split" | wc -l) files"
    echo "  images/$split : $(ls "$DATA_ROOT/images/$split" | wc -l) files"
done

echo "--- done $(date -Is) ---"
