#!/usr/bin/env bash
# makelab2 launcher for the #25 frozen-model input-size sweep (arm 1, no retraining).
#
#   bash scripts/analysis/input_res_sweep_25.sh <panos-root> [venv-activate]
#
# <panos-root> holds benchmark/<city>/panos/*.jpg at native resolution (on makelab2 the
# existing clone, /homes/gws/jonf/RampNet, already has all 11 splits). Order is the
# point: the r2048 control is extracted and CHECKED against the committed
# analysis_out/op_cache before any other arm is run, and a failed check stops the run.
# Every step is resumable (extract skips an arm/city whose cache exists).
set -euo pipefail
PANOS_ROOT=${1:?usage: input_res_sweep_25.sh <panos-root> [venv-activate]}
VENV=${2:-/homes/gws/jonf/RampNet/.venv/bin/activate}
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1090
source "$VENV"
# less fragmentation when the 5500x11000 arm follows smaller ones in one process
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=analysis_out/input_res_sweep_25
mkdir -p "$OUT"
S=scripts/analysis/input_res_sweep_25.py
{
  echo "=== $(date -u +%FT%TZ) $(hostname) $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  python "$S" extract --arms r2048 --panos-root "$PANOS_ROOT"
  python "$S" check
  python "$S" extract --arms r3072,r4096,rnative,u4096,r4096_hm1024 --panos-root "$PANOS_ROOT"
  python "$S" report
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$OUT/run.log"
