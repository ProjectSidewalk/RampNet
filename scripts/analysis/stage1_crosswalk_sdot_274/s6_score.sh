#!/bin/bash
# crosswalk_s6 scoring, CPU only (docs/crosswalk_followups_285.md, section A4): the primary and guards 1 to 6 for
# both doses and the step control. Predictions must already be in the repo layout (see the doc's replication block).
# Usage, from the repo root: PY=<python> ROOT=<Stage 2 root> BASELINE=<baseline_nodes.csv> CENSUS=<census out>
#        PANO_DATA=<seattle_pano_data.psv> VERDICTS=<recall285 verdicts json> bash s6_score.sh
set -eu
PY=${PY:-python}
S=scripts/analysis/stage1_crosswalk_sdot_274
D=analysis_out/stage1_crosswalk_detector_274; N=$D/hardneg; F=analysis_out/crosswalk_followups_285
S4="crosswalk_s4 crosswalk_s4_seed2 crosswalk_s4_seed3"; S4T=(0.23 0.3 0.13)
ALL=crosswalk_s6full,crosswalk_s6full_seed2,crosswalk_s6full_seed3,crosswalk_s6half,crosswalk_s6half_seed2,crosswalk_s6half_seed3,crosswalk_s4_steps1397,crosswalk_s4,crosswalk_s4_seed2,crosswalk_s4_seed3
score() { "$PY" $S/score_274.py score --root "$ROOT" --preds $D/v2_labels/preds --labels $ALL,released,label_ceiling \
              --baseline-nodes "$BASELINE" --census "$CENSUS" "$@" > /dev/null; }
score --out $D/v2_labels/s6
match() { "$PY" $S/score_274.py match-threshold --sweep $D/v2_labels/s6/threshold_sweep.csv --base "$1" --new "$2" \
              --base-threshold "$3" | tail -1; }
declare -A T
i=0
for b in $S4; do
    T[$b]=${S4T[$i]}
    sfx=${b#crosswalk_s4}
    for d in full half; do T[crosswalk_s6$d$sfx]=$(match "$b" "crosswalk_s6$d$sfx" "${S4T[$i]}"); done
    i=$((i + 1))
done
T[crosswalk_s4_steps1397]=$(match crosswalk_s4 crosswalk_s4_steps1397 0.23)
fixed=(); for k in "${!T[@]}"; do fixed+=("$k=${T[$k]}"); done
printf 'thresholds: %s\n' "$(printf '%s\n' "${fixed[@]}" | sort | tr '\n' ' ')"
score --fixed-threshold "${fixed[@]}" --out $D/v2_labels/s6_matched
"$PY" $S/score_274.py bprime-score --bprime $D/v2_labels/bprime --scores $D/v2_labels/s6_matched --pano-data "$PANO_DATA" \
    --labels $ALL --out $D/v2_labels/s6_matched/bprime > /dev/null
at() { echo "$1@${T[$1]}"; }                       # label@threshold
for d in full half steps; do
    if [ $d = steps ]; then new="crosswalk_s4_steps1397"; base="crosswalk_s4_klone@0.23"; lab1=crosswalk_s4_steps1397
    else new="crosswalk_s6$d crosswalk_s6${d}_seed2 crosswalk_s6${d}_seed3"; lab1=crosswalk_s6$d
         base="crosswalk_s4_klone@0.23 crosswalk_s4_seed2@0.3 crosswalk_s4_seed3@0.13"; fi
    "$PY" $S/recall_check.py compare --verdicts "$VERDICTS" --cities laurens-ia juneau-ak --base $base \
        --new $(for m in $new; do at $m; done) --out-label s6_$d > /dev/null
    t=${T[$lab1]}
    "$PY" $S/score_274.py label-recall --labels $N/s4_inputs/stepA/columbia_gt.csv \
        --base crosswalk_s4=$N/s4_eval/columbia/preds/crosswalk_s4/columbia.jsonl@0.23 \
        --new $lab1=$N/s6_eval/columbia/preds/$lab1/columbia.jsonl@$t --out $N/s6_eval/columbia/$d > /dev/null
    "$PY" $S/score_274.py retention --ratings $N/confirm_s3/seattle/s4/ratings.csv \
        --models crosswalk_s4=$D/v2_labels/bprime/preds/crosswalk_s4/bprime.jsonl@0.23 \
                 $(for m in $new; do echo "$m=$D/v2_labels/bprime/preds/$m/bprime.jsonl@${T[$m]}"; done) \
        --out $F/s6/$d/seattle > /dev/null
    "$PY" $S/score_274.py retention --ratings $N/s5_inputs/juneau_eval/s4/ratings.csv \
        --models crosswalk_s4=$N/s5_eval/juneau_eval/preds/crosswalk_s4/juneau_eval.jsonl@0.23 \
                 $(for m in $new; do echo "$m=$N/s6_eval/juneau_eval/preds/$m/juneau_eval.jsonl@${T[$m]}"; done) \
        --out $F/s6/$d/juneau_eval > /dev/null
done
echo done
