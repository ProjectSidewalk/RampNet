#!/bin/bash
# Launcher for the #82 Step 1 frozen-model probe (makelab2 A40). Runs, in order:
#   1. extract the untransformed arm ("none") on all seven probe splits;
#   2. check: it must reproduce the committed #25 r2048 caches -- STOP if it does not;
#   3. extract every other arm (stats.json -> arm table, see `aug_probe_82.py arms`);
#   4. report.
# Usage (from the checkout root; panos from the main checkout's benchmark/<split>/panos):
#   tmux new -s aug82probe 'scripts/analysis/aug_probe_82.sh > logs82/probe.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 2
PY="${PY:-/homes/gws/jonf/RampNet/.venv/bin/python}"
PANOS="${PANOS:-/homes/gws/jonf/RampNet}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "=== aug probe 82 $(date -Is) commit $(git rev-parse --short HEAD) host $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
"$PY" scripts/analysis/aug_probe_82.py extract --panos-root "$PANOS" --arms none \
    --note "control arm first; instrument check follows" || exit 1
"$PY" scripts/analysis/aug_probe_82.py check || { echo "INSTRUMENT CHECK FAILED -- stopping"; exit 1; }
"$PY" scripts/analysis/aug_probe_82.py extract --panos-root "$PANOS" || exit 1
"$PY" scripts/analysis/aug_probe_82.py report > /dev/null || exit 1
echo "=== done $(date -Is)"
