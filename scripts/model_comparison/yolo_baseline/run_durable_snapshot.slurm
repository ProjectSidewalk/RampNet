#!/bin/bash
# Put everything #51 still needs onto durable storage, in one job (#51).
#
# WHY THIS IS ONE JOB
#   Two different things on /gscratch/scrubbed are one purge away from unreproducible,
#   and they fail together for the same reason, so they are protected together:
#
#     1. The pano label CACHES. The 2026-08 purge took every label .txt under
#        yolo/pano/labels/ (0 files remain against 150,063 + 42,875 surviving images).
#        pano/labels/{train,val}.cache survived and can reconstitute them exactly --
#        but those caches sit on the same volume, on the same access-time clock, that
#        already deleted their own labels once. Unlike the tiles pair (staged
#        2026-08-17) they had never been copied anywhere durable.
#
#     2. The tiles CHECKPOINTS. The durable snapshot under /gscratch/makelab was last
#        refreshed 2026-08-16, and every tiles arm has trained past it since:
#        y11x_tiles ep44 vs ep32 saved, y26_tiles ep12 vs ep9, y11l_tiles ep11 vs ep9,
#        y11x_pano ep38 vs ep30. The checkpoints an evaluation would actually score
#        exist at their current epoch in exactly one place, and it is the volume that
#        has already demonstrated what it does to cold files.
#
#   Neither step needs a GPU and neither is destructive: both copy, verify by sha256,
#   and leave the source alone.
#
#   This is a batch job rather than a login-node one-liner because it reads and hashes
#   ~2.4 GB off GPFS. klone reaps heavy login processes and that reap also kills the SSH
#   control master.
#
#   sbatch scripts/model_comparison/yolo_baseline/run_durable_snapshot.slurm
#
# Env overrides: RAMPNET_REPO, YOLO_SCRATCH_ROOT, YOLO_RUNS_ROOT, YOLO_DURABLE_ROOT,
#                CACHE_VARIANTS ("" to skip the cache step), SNAPSHOT_ARMS ("" to skip).
#
#SBATCH --job-name=yolo_durable_snapshot_51
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/yolo_durable_snapshot_%j.out
#SBATCH --error=logs/yolo_durable_snapshot_%j.err
#SBATCH --requeue

set -uo pipefail

REPO="${RAMPNET_REPO:-$HOME/RampNet}"
SCRATCH_ROOT="${YOLO_SCRATCH_ROOT:-/gscratch/scrubbed/jfroehli/yolo}"
RUNS_ROOT="${YOLO_RUNS_ROOT:-/gscratch/scrubbed/jfroehli/yolo_runs}"
DURABLE_ROOT="${YOLO_DURABLE_ROOT:-/gscratch/makelab/jonf/rampnet_yolo_baseline_51}"

# pano only by default. The tiles pair is already staged and its live cache is a LATER
# regeneration than the durable copy, so re-rescuing tiles would be refused by design.
CACHE_VARIANTS="${CACHE_VARIANTS-pano}"
SNAPSHOT_ARMS="${SNAPSHOT_ARMS-}"

BASE="$REPO/scripts/model_comparison/yolo_baseline"

echo "--- YOLO durable snapshot (#51) ---"
echo "Job ID        : ${SLURM_JOBID:-?}"
echo "Node          : ${SLURMD_NODENAME:-?}"
echo "Scratch root  : $SCRATCH_ROOT"
echo "Runs root     : $RUNS_ROOT"
echo "Durable root  : $DURABLE_ROOT"
echo "Cache variants: ${CACHE_VARIANTS:-<skipped>}"
echo "Snapshot arms : ${SNAPSHOT_ARMS:-<all found>}"
echo "Free on durable volume:"
df -h "$DURABLE_ROOT" | tail -1
echo "-----------------------------------"

rc=0

echo
echo "=== STEP 1/2: rescue label caches  $(date -Is) ==="
if [ -z "$CACHE_VARIANTS" ]; then
    echo "(skipped: CACHE_VARIANTS is empty)"
else
    # shellcheck disable=SC2086
    SRC="$SCRATCH_ROOT" DST="$DURABLE_ROOT/label_cache_rescue" \
        bash "$BASE/rescue_label_caches.sh" $CACHE_VARIANTS || rc=1
fi

echo
echo "=== STEP 2/2: snapshot run directories  $(date -Is) ==="
# shellcheck disable=SC2086
SRC="$RUNS_ROOT" DST="$DURABLE_ROOT" \
    bash "$BASE/snapshot_runs.sh" $SNAPSHOT_ARMS || rc=1

echo
echo "=== what is durable now  $(date -Is) ==="
for d in "$DURABLE_ROOT"/*/; do
    arm=$(basename "$d")
    r="$d/results.csv"
    if [ -f "$r" ]; then
        printf '  %-18s %s epochs, best.pt %s\n' "$arm" \
            "$(( $(wc -l <"$r") - 1 ))" \
            "$([ -f "$d/weights/best.pt" ] && echo present || echo MISSING)"
    else
        printf '  %-18s (no results.csv)\n' "$arm"
    fi
done
echo
echo "  label caches:"
find "$DURABLE_ROOT/label_cache_rescue" -name '*.cache' -printf '    %p (%s bytes)\n' 2>/dev/null | sort

echo
echo "--- done $(date -Is), rc=$rc ---"
exit "$rc"
