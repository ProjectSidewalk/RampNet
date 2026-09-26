#!/usr/bin/env bash
# Launcher for the #25 frozen-model input-size sweep (arm 1, no retraining).
#
#   bash scripts/analysis/input_res_sweep_25.sh <panos-root> [venv-activate] \
#       [--cache-root DIR] [--cities a,b,...] [--force] [--limit N]
#
# <panos-root> holds benchmark/<city>/panos/*.jpg at native resolution (on makelab2 the
# existing clone, /homes/gws/jonf/RampNet, already has all 11 splits; from a clean clone,
# scripts/unpack_benchmark_panos.py writes that layout for the 9 splits on the Hub).
# Order is the point: the r2048 control is extracted and CHECKED against the committed
# analysis_out/op_cache before any other arm is run, and a failed check stops the run.
#
# Every cache is committed, so with the defaults every arm/split is skipped and only
# check + report run. To actually re-run, either
#   --cache-root DIR   write fresh caches under DIR (results.json, results.md and
#                      instrument_check.json go in DIR/.. beside it), or
#   --force            re-extract over the committed caches in place.
# --cities restricts extract, check and report (default: all 11 splits; laurens_gsv and
# laurens_mapillary are not on the Hub, so a clean clone passes the other nine).
# --limit N (smoke test) is refused unless --cache-root is a scratch dir; under it the
# check cannot pass (truncated pano sets) and is reported rather than stopping the run.
# Without --force every step is resumable (extract skips an arm/split whose cache exists).
set -euo pipefail
USAGE="usage: input_res_sweep_25.sh <panos-root> [venv-activate] [--cache-root DIR] [--cities LIST] [--force] [--limit N]"
PANOS_ROOT=${1:?$USAGE}
shift
VENV=/homes/gws/jonf/RampNet/.venv/bin/activate
if [[ $# -gt 0 && $1 != --* ]]; then VENV=$1; shift; fi
DEFAULT_CACHE=analysis_out/input_res_sweep_25/cache
CACHE_ROOT=$DEFAULT_CACHE
CITIES=""
FORCE=()
LIMIT=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --cache-root) CACHE_ROOT=${2:?--cache-root needs a dir}; shift 2 ;;
    --cities) CITIES=${2:?--cities needs a list}; shift 2 ;;
    --force) FORCE=(--force); shift ;;
    --limit) LIMIT=(--limit "${2:?--limit needs N}"); shift 2 ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/../.."
if [[ ${#LIMIT[@]} -gt 0 && $(realpath -m "$CACHE_ROOT") == $(realpath -m "$DEFAULT_CACHE") ]]; then
  echo "--limit writes truncated caches: pass a scratch --cache-root, not $DEFAULT_CACHE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV"
# less fragmentation when the 5500x11000 arm follows smaller ones in one process
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$(dirname "$CACHE_ROOT")
mkdir -p "$OUT"
S=scripts/analysis/input_res_sweep_25.py
CR=(--cache-root "$CACHE_ROOT")
CI=()
if [[ -n $CITIES ]]; then CI=(--cities "$CITIES"); fi
{
  echo "=== $(date -u +%FT%TZ) $(hostname) $(git rev-parse HEAD) cache-root=$CACHE_ROOT"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  python "$S" extract --arms r2048 --panos-root "$PANOS_ROOT" ${CR[@]+"${CR[@]}"} ${CI[@]+"${CI[@]}"} ${FORCE[@]+"${FORCE[@]}"} ${LIMIT[@]+"${LIMIT[@]}"}
  if [[ ${#LIMIT[@]} -gt 0 ]]; then
    # a --limit run's pano sets are truncated, so the check cannot pass; report, don't stop
    python "$S" check ${CR[@]+"${CR[@]}"} ${CI[@]+"${CI[@]}"} --out "$OUT/instrument_check.json"         || echo "check FAILED, expected under --limit (truncated pano sets); continuing"
  else
    python "$S" check ${CR[@]+"${CR[@]}"} ${CI[@]+"${CI[@]}"} --out "$OUT/instrument_check.json"
  fi
  python "$S" extract --arms r3072,r4096,rnative,u4096,r4096_hm1024 --panos-root "$PANOS_ROOT" \
      ${CR[@]+"${CR[@]}"} ${CI[@]+"${CI[@]}"} ${FORCE[@]+"${FORCE[@]}"} ${LIMIT[@]+"${LIMIT[@]}"}
  python "$S" report ${CR[@]+"${CR[@]}"} ${CI[@]+"${CI[@]}"} --out "$OUT/results.json"
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$OUT/run.log"
