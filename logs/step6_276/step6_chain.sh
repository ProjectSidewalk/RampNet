#!/usr/bin/env bash
# #276 Step 6: in-context-images control. exemplars=shared4 + confidence, on dev_select and
# the open pool (laurens_mapillary + bayonne). Code = the vlm-276-followups worktree (holds the
# committed exemplar set); pixels and cache = the MAIN checkout by absolute path, no junctions.
# Usage: step6_chain.sh <gemini-model>
set -u
MODEL="$1"
CODE=/d/Git/RampNet-wt/vlm276b
DATA=D:/Git/RampNet
PY=/d/Git/RampNet/.venv/Scripts/python.exe
LOGS=/d/Git/RampNet/logs/step6_276
cd "$CODE" || exit 1
for split in sao_paulo laurens_mapillary bayonne; do
  extra=()
  [ "$split" = sao_paulo ] && extra+=(--pano-list benchmark/prompt_opt_276/sao_paulo_split.json@dev_select)
  log="$LOGS/step6_${MODEL}_shared4_conf_${split}.log"
  echo "$(date '+%F %T') [$MODEL] start shared4_conf $split"
  timeout 86400 "$PY" scripts/model_comparison/compare.py "$DATA/benchmark/$split" --models "gemini:$MODEL" \
      --gemini-concurrency 6 --gemini-confidence --gemini-exemplars shared4 \
      --cache-dir "$DATA/.model_cache" "${extra[@]}" > "$log" 2>&1
  rc=$?
  echo "$(date '+%F %T') [$MODEL] done  shared4_conf $split exit $rc  $(grep -E '^gemini' "$log" | tail -n 1 | awk '{print "F1",$8}')"
  if [ $rc -ne 0 ]; then echo "STOPPING $MODEL at $split (exit $rc); see $log"; exit $rc; fi
done
echo "$(date '+%F %T') [$MODEL] chain complete"
