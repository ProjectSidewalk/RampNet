#!/usr/bin/env bash
# #276 follow-up A: whole-prompt-rewrite dev loop for one model. Code = the vlm276c worktree;
# pixels and cache = the MAIN checkout by absolute path (no junctions).
# Usage: rewrite_chain.sh <gemini-model> <short> <floor>
set -u
M="$1"; SHORT="$2"; FLOOR="$3"
cd /d/Git/RampNet-wt/vlm276c || exit 1
PY=/d/Git/RampNet/.venv/Scripts/python.exe
P=scripts/model_comparison/prompt_optimize.py
R=(--panos-root D:/Git/RampNet --cache-dir D:/Git/RampNet/.model_cache)
run() { echo "$(date '+%F %T') >>> $*"; "$@"; echo "$(date '+%F %T') <<< exit $?"; }
for it in 01 02; do
  [ "$it" = 02 ] && run "$PY" $P pick --scope prompt --model "$M" --reuse-from "optimize/$M/iter_$it" "${R[@]}"
  run "$PY" $P rewrite  --scope prompt --model "$M" "${R[@]}"
  run "$PY" $P evaluate --scope prompt --model "$M" --floor "$FLOOR" "${R[@]}"
done
run "$PY" $P alternative --scope prompt --model "$M" "${R[@]}"
run "$PY" $P status --scope prompt --model "$M" "${R[@]}"
# dev_select for every candidate body written (the gate and the paraphrase control)
for f in benchmark/prompt_opt_276/prompts/prompt276-"$SHORT"-*.txt; do
  [ -e "$f" ] || continue
  name=$(basename "$f" .txt)
  echo "$(date '+%F %T') >>> dev_select $name"
  "$PY" scripts/model_comparison/compare.py D:/Git/RampNet/benchmark/sao_paulo --models "gemini:$M" \
      --gemini-concurrency 6 --gemini-confidence --prompt-body "$name" \
      --pano-list benchmark/prompt_opt_276/sao_paulo_split.json@dev_select \
      --cache-dir D:/Git/RampNet/.model_cache > "/d/Git/RampNet/logs/rewrite_276/devselect_${name}.log" 2>&1
  echo "$(date '+%F %T') <<< exit $?  $(grep -E '^gemini' "/d/Git/RampNet/logs/rewrite_276/devselect_${name}.log" | tail -n 1)"
done
echo "$(date '+%F %T') [$M] rewrite chain complete"
