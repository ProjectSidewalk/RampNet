#!/bin/bash
# YOLO geometry-pair benchmark eval (#51): does the equirect input explain the gap?
#
# THE QUESTION
#   #51's headline is that RampNet beats a supervised YOLO baseline by 0.12-0.37 F1 on
#   nine out-of-distribution city splits.  The obvious objection is that this is not
#   architecture at all -- YOLO is being fed 2048x4096 equirectangular panoramas, which
#   is not what a COCO-shaped detector expects, so the gap could be a geometry handicap.
#   The tiles arms exist to answer exactly that: same data, same schedule, but fed
#   through the same perspective-view rig the VLMs get.  Until now no tiles checkpoint
#   had ever been scored on the benchmark, so the objection was live and unmeasured.
#
# WHY THESE THREE LEGS
#   y11x_tiles  ep44  perspective, imgsz 1024  } same architecture, near-matched budget,
#   y11x_pano   ep38  whole-pano,  imgsz 1280  } opposite geometry -> isolates geometry
#   y11x_pano_h200 ep60  whole-pano, imgsz 1280   CONTROL, already published
#
#   The control is not redundant.  The published y11x_pano_h200 numbers were produced on
#   2026-08-14 against a repo that predates the #132 seam fix, which changed how the
#   matcher wraps the 360 seam and therefore changes scores.  Re-scoring it here, under
#   the same code as the two new legs, does two things: it keeps all three numbers
#   mutually comparable, and it measures how far the published number moved.  Comparing
#   a fresh tiles number against a stale pano number would attribute a code change to
#   geometry.
#
#   y11x_pano ep38 and y11x_tiles ep44 are NOT equal budget.  They are the epochs that
#   exist.  Read the pair as "roughly matched", and read the h200 control for what a
#   converged pano arm does.
#
# PROTOCOL (pre-registered, issue #71; scripts/model_comparison/yolo_baseline/README.md)
#   best.pt as-saved; each arm in its TRAINING geometry; headline F1 at conf 0.25;
#   full sweep printed but flagged tune-on-test; selection never on test.
#
# USAGE
#   ./run_yolo_geometry_eval.sh                      # all 10 splits
#   ./run_yolo_geometry_eval.sh manual_gold paterson # only the named splits
#   OUT=... REPO=... ./run_yolo_geometry_eval.sh
#
set -u

REPO="${REPO:-/homes/gws/jonf/RampNet}"
PY="${PY:-$REPO/.venv-eval/bin/python}"
OUT="${OUT:-$REPO/yolo_eval_results_geometry_51}"
CKPTS="${CKPTS:-$REPO/yolo_ckpts}"

cd "$REPO" || exit 2
mkdir -p "$OUT"

SPLITS_DEFAULT=(manual_gold bend richmond annapolis budapest_district5 clovis gainesville morgantown paterson sao_paulo)
if [ "$#" -gt 0 ]; then SPLITS=("$@"); else SPLITS=("${SPLITS_DEFAULT[@]}"); fi

# Provenance first: which code, which weights. A number whose checkpoint hash is not
# written down cannot be re-derived by someone else.
{
  echo "run started      : $(date -Is)"
  echo "host             : $(hostname)"
  echo "repo HEAD        : $(git rev-parse HEAD)  $(git log -1 --format=%s)"
  echo "repo dirty       : $(git status --porcelain | wc -l) paths"
  "$PY" -c "import torch,ultralytics;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('ultralytics',ultralytics.__version__)"
  echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "checkpoint sha256:"
  sha256sum "$CKPTS/y11x_tiles.pt" "$CKPTS/y11x_pano.pt" "$CKPTS/y11x_pano_h200.pt"
  echo "splits           : ${SPLITS[*]}"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

for b in "${SPLITS[@]}"; do
  if [ ! -d "benchmark/$b" ]; then
    echo "=== $b SKIPPED (no bundle dir)"
    continue
  fi

  # Leg 1: tiles geometry. Its own invocation because --tiling/--yolo-imgsz are global
  # flags, so a tiles arm and a pano arm cannot share one call.
  echo "=== $b tiles  start $(date -Is)"
  t0=$(date +%s)
  "$PY" scripts/model_comparison/compare.py "benchmark/$b" \
      --models "yolo:$CKPTS/y11x_tiles.pt" \
      --tiling perspective --yolo-imgsz 1024 \
      --op-threshold 0.25 --sweep --pr-out "$OUT/pr_${b}_tiles" \
      > "$OUT/${b}_tiles.txt" 2>&1
  echo "=== $b tiles  exit=$? elapsed=$(( $(date +%s) - t0 ))s"

  # Leg 2+3: both pano arms share a geometry, so they go in one call and land in one
  # table -- which is also the direct ep38-vs-ep60 read.
  echo "=== $b pano   start $(date -Is)"
  t0=$(date +%s)
  "$PY" scripts/model_comparison/compare.py "benchmark/$b" \
      --models "yolo:$CKPTS/y11x_pano.pt,yolo:$CKPTS/y11x_pano_h200.pt" \
      --tiling none --yolo-imgsz 1280 \
      --op-threshold 0.25 --sweep --pr-out "$OUT/pr_${b}_pano" \
      > "$OUT/${b}_pano.txt" 2>&1
  echo "=== $b pano   exit=$? elapsed=$(( $(date +%s) - t0 ))s"
done

echo "ALL_SPLITS_DONE $(date -Is)"
