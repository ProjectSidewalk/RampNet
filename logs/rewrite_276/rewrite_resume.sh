#!/usr/bin/env bash
# Resume of rewrite_chain.sh after the 2026-11-12 08:05 stall (dropped connections, no
# request timeout). Runs the named steps in order, then dev_select for every candidate.
# Steps: pick2 rewrite evaluate alternative status
# Usage: rewrite_resume.sh <gemini-model> <short> <floor> <step>...
set -u
M="$1"; SHORT="$2"; FLOOR="$3"; shift 3
cd /d/Git/RampNet-wt/vlm276c || exit 1
PY=/d/Git/RampNet/.venv/Scripts/python.exe
P=scripts/model_comparison/prompt_optimize.py
R=(--panos-root D:/Git/RampNet --cache-dir D:/Git/RampNet/.model_cache)
run() { echo "$(date '+%F %T') >>> $*"; "$@"; rc=$?; echo "$(date '+%F %T') <<< exit $rc"; return $rc; }
for step in "$@"; do
  case "$step" in
    pick2)       run "$PY" $P pick --scope prompt --model "$M" --reuse-from "optimize/$M/iter_02" "${R[@]}" ;;
    rewrite)     run "$PY" $P rewrite --scope prompt --model "$M" "${R[@]}" ;;
    evaluate)    run "$PY" $P evaluate --scope prompt --model "$M" --floor "$FLOOR" "${R[@]}" ;;
    alternative) run "$PY" $P alternative --scope prompt --model "$M" "${R[@]}" ;;
    status)      run "$PY" $P status --scope prompt --model "$M" "${R[@]}" ;;
  esac || { echo "STOPPING at $step"; exit 1; }
done
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
