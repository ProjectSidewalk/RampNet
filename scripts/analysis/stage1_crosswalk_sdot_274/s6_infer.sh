#!/bin/bash
# crosswalk_s6 inference on klone (docs/crosswalk_followups_285.md, section A4), one job per model and panorama set.
# Every seed: val+test, the (b') sample, the 175 missed-crosswalk panos and the fresh Juneau draw. Seed 1 and the
# step control also: the Seattle/Laurens confirmation draw, Columbia, and the Bayonne and PS-label probes.
# Usage: bash s6_infer.sh <run> [<run> ...]   e.g. v6full_s1 v6half_s3 v4steps1397_s1 (a finished s6_train.sh run)
set -eu
W=/gscratch/scrubbed/$USER/crosswalk274
C=/gscratch/makelab/jonf/crosswalk274
cd /gscratch/scrubbed/$USER/RampNet274
mkdir -p logs
S=scripts/analysis/stage1_crosswalk_sdot_274/score_274.slurm
export RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 RELEASED=0 ROOT=$W/root_v4
# the confirmation draw's id lists, from the panos crosswalk_s4 was run on
for s in cities ps_labels; do
    [ -s "$W/confirm_${s}_ids.txt" ] || python3 -c "
import json, sys
for line in open(sys.argv[1]):
    print(json.loads(line)['pano_id'] + '.jpg')" "$W/confirm_preds/crosswalk_s4/$s.jsonl" > "$W/confirm_${s}_ids.txt"
done
flat() {  # split_name ids_file pano_dir preds_dir
    BPRIME=1 SPLIT_NAME=$1 IDS_FILE=$2 BPRIME_PANOS=$3 BPRIME_PREDS=$4 sbatch --parsable --job-name="cw285_inf_${1}_$name" $S
}
for name in "$@"; do
    seed=${name##*_s}
    case $name in
        v6full_s*|v6half_s*) d=${name#v6}; d=${d%_s*}; label=crosswalk_s6$d ;;
        v4steps*_s1) label=crosswalk_s4_${name%_s1}; label=${label/v4steps/steps} ;;
        *) echo "FATAL: unknown run $name" >&2; exit 1 ;;
    esac
    [ "$seed" != 1 ] && label=${label}_seed$seed
    ckpt=$(ls "$C/$name"/final_step_*.pth)
    export FT_LABEL=$label FT_CKPT=$ckpt FT_SHA=$(cut -d' ' -f1 "$ckpt.sha256")
    jobs="$(PREDS=$W/score_preds_v6 sbatch --parsable --job-name="cw285_inf_valtest_$name" $S)"
    jobs="$jobs $(BPRIME=1 BPRIME_PREDS=$W/bprime_preds_v6 sbatch --parsable --job-name="cw285_inf_bprime_$name" $S)"
    jobs="$jobs $(flat juneau-ak $W/recall285/juneau-ak_ids.txt $W/recall285/prep_4096 $W/recall285_preds)"
    jobs="$jobs $(flat laurens-ia $W/recall285/laurens-ia_ids.txt $W/recall285/prep_4096 $W/recall285_preds)"
    jobs="$jobs $(flat juneau_eval $W/juneau_eval_ids.txt $W/juneau_eval_4096 $W/juneau_eval_preds)"
    if [ "$seed" = 1 ]; then
        jobs="$jobs $(flat cities $W/confirm_cities_ids.txt $W/confirm_4096 $W/confirm_preds_v6)"
        jobs="$jobs $(flat ps_labels $W/confirm_ps_labels_ids.txt $W/ps_label_4096 $W/confirm_preds_v6)"
        jobs="$jobs $(flat columbia $W/columbia_infer_ids.txt $W/columbia_4096 $W/columbia_preds)"
        jobs="$jobs $(flat bayonne $W/bayonne_infer_ids.txt $W/bayonne_4096 $W/bayonne_preds)"
    fi
    echo "$name $label ${FT_SHA:0:12}: $jobs"
done
