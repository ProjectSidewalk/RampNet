#!/usr/bin/env bash
# GPU half of #218 (makelab2, A40). Run from the repo root of a clone of this branch:
#
#   bash scripts/analysis/perspective_photos_218.sh <richmond-img-dir> <seoul-img-dir> \
#        [venv-activate] [usage-log]
#
# <richmond-img-dir> holds <image_id>.jpg from `perspective_photos_218.py fetch`;
# <seoul-img-dir> holds the photos from `seoul_photos_218.py fetch` (either may be "-" to
# skip that half). Every image is checked against the committed sha256 before inference.
# usage-log defaults to analysis_out/perspective_photos_218/usage_rows.jsonl in THIS
# clone; copy those rows into the main checkout's analysis_out/usage_log.jsonl.
set -euo pipefail
RICH=${1:?richmond image dir or -}
SEOUL=${2:?seoul image dir or -}
VENV=${3:-/homes/gws/jonf/RampNet/.venv/bin/activate}
UL=${4:-analysis_out/perspective_photos_218/usage_rows.jsonl}
source "$VENV"
{
  echo "=== $(date -u +%FT%TZ) $(hostname) $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  if [[ $RICH != - ]]; then
    # NSHARD processes share the GPU (~5.7 GB each); the canvas build is CPU-bound, so
    # this is what makes the run minutes rather than hours
    N=${NSHARD:-4}
    for k in $(seq 0 $((N - 1))); do
      python scripts/analysis/perspective_photos_218.py infer --images "$RICH" \
        --arms "${ARMS:-canvas_level,canvas_sfm,stretch}" --verify-sha --usage-log "$UL" \
        --shard "$k/$N" &
    done
    wait
    python scripts/analysis/perspective_photos_218.py merge \
      --arms "${ARMS:-canvas_level,canvas_sfm,stretch}"
  fi
  if [[ $SEOUL != - ]]; then
    python scripts/analysis/seoul_photos_218.py infer --images "$SEOUL" --verify-sha \
      --usage-log "$UL"
  fi
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1
