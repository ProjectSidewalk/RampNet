#!/usr/bin/env bash
# #276 rewrite arm, held-out run of the one candidate that passed the dev_select gate:
# Argon prompt276-argon-r1 (+confidence), compared afterwards with the cached zs+conf legs.
# Jon approved the spend on 2026-11-13 (about $44 estimated, ~$37 from the para3 arm's rows).
# Runs from the MAIN checkout (no junctions, absolute cache dir). manual_gold uses the
# committed 250-pano list, as every Step 5 arm but zs+conf did.
# Usage: heldout_chain.sh <tag> <split>...
set -u
TAG="$1"; shift
MODEL=gemini-4-argon; BODY=prompt276-argon-r1
ROOT=/d/Git/RampNet
PY=$ROOT/.venv/Scripts/python.exe
LOGS=$ROOT/logs/rewrite_276
cd "$ROOT" || exit 1
for split in "$@"; do
  extra=()
  [ "$split" = manual_gold ] && extra+=(--pano-list benchmark/prompt_opt_276/manual_gold_250.json@panos)
  log="$LOGS/heldout_${BODY}_${split}.log"
  echo "$(date '+%F %T') [$TAG] start $split"
  timeout 86400 "$PY" scripts/model_comparison/compare.py "D:/Git/RampNet/benchmark/$split" --models "gemini:$MODEL" \
      --gemini-concurrency 6 --gemini-confidence --prompt-body "$BODY" \
      --cache-dir D:/Git/RampNet/.model_cache "${extra[@]}" > "$log" 2>&1
  rc=$?
  echo "$(date '+%F %T') [$TAG] done  $split exit $rc  $(grep -E '^gemini' "$log" | tail -n 1)"
  if [ $rc -ne 0 ]; then echo "STOPPING $TAG at $split (exit $rc); see $log"; exit $rc; fi
done
echo "$(date '+%F %T') [$TAG] chain complete"
