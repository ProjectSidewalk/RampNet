#!/usr/bin/env bash
# #276 Step 7: replicates of the Flash zs+conf and expert+conf arms on the open pool
# (laurens_mapillary + bayonne). One chain = one replicate, its own cache dir, so every
# call is paid again (same cache dir would make zero calls). Runs from the MAIN checkout.
# Usage: step7_chain.sh <k>      -> cache .model_cache_rep<k>
set -u
K="$1"; MODEL=gemini-3.8-flash
ROOT=/d/Git/RampNet
PY=$ROOT/.venv/Scripts/python.exe
LOGS=$ROOT/logs/step7_276
cd "$ROOT" || exit 1
for split in laurens_mapillary bayonne; do
  for arm in zs_conf expert_conf; do
    extra=()
    [ "$arm" = expert_conf ] && extra+=(--definition expert)
    log="$LOGS/step7_rep${K}_${MODEL}_${arm}_${split}.log"
    echo "$(date '+%F %T') [rep$K] start $arm $split"
    timeout 43200 "$PY" scripts/model_comparison/compare.py "benchmark/$split" --models "gemini:$MODEL" \
        --gemini-concurrency 6 --gemini-confidence --cache-dir ".model_cache_rep$K" "${extra[@]}" > "$log" 2>&1
    rc=$?
    echo "$(date '+%F %T') [rep$K] done  $arm $split exit $rc  $(grep -E '^gemini' "$log" | tail -n 1 | awk '{print "F1",$8}')"
    if [ $rc -ne 0 ]; then echo "STOPPING rep$K at $arm $split (exit $rc); see $log"; exit $rc; fi
  done
done
echo "$(date '+%F %T') [rep$K] chain complete"
