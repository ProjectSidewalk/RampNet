#!/usr/bin/env bash
# #276 rewrite arm follow-up: replicates of Argon zs+conf and prompt276-argon-r1+conf on the
# open pool (laurens_mapillary + bayonne), to test the single-run Mapillary gain in RampNet#288.
# Jon approved 2026-11-14 ("two more runs of both prompts on the open-imagery pool", ~$20).
# One chain = one replicate with its OWN fresh cache dir (.model_cache_argon_rep<k>), so every
# call is paid again; the Step 3/7 dirs .model_cache_rep{1,2} are deliberately not reused.
# Runs from the MAIN checkout (no junctions, absolute paths). Same flags as heldout_chain.sh.
# Usage: replicate_chain.sh <k>
set -u
K="$1"; MODEL=gemini-4-argon
ROOT=/d/Git/RampNet
PY=$ROOT/.venv/Scripts/python.exe
LOGS=$ROOT/logs/rewrite_276
CACHE="D:/Git/RampNet/.model_cache_argon_rep$K"
cd "$ROOT" || exit 1
for split in laurens_mapillary bayonne; do
  for arm in zs_conf r1_conf; do
    extra=()
    [ "$arm" = r1_conf ] && extra+=(--prompt-body prompt276-argon-r1)
    log="$LOGS/rep${K}_${MODEL}_${arm}_${split}.log"
    echo "$(date '+%F %T') [rep$K] start $arm $split"
    timeout 43200 "$PY" scripts/model_comparison/compare.py "D:/Git/RampNet/benchmark/$split" \
        --models "gemini:$MODEL" --gemini-concurrency 6 --gemini-confidence \
        --cache-dir "$CACHE" "${extra[@]}" > "$log" 2>&1
    rc=$?
    echo "$(date '+%F %T') [rep$K] done  $arm $split exit $rc  $(grep -E '^gemini' "$log" | tail -n 1)"
    if [ $rc -ne 0 ]; then echo "STOPPING rep$K at $arm $split (exit $rc); see $log"; exit $rc; fi
  done
done
echo "$(date '+%F %T') [rep$K] chain complete"
