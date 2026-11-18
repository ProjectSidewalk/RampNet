#!/bin/bash
# crosswalk_s6 calibration (docs/crosswalk_followups_285.md, A4): crosswalk_s4 seeds 1-3 on the 175
# missed-crosswalk panos, and seeds 2-3 on the (b') sample and the fresh Juneau draw. Inference only.
set -eu
cd /gscratch/scrubbed/$USER/RampNet274
mkdir -p logs
W=/gscratch/scrubbed/$USER/crosswalk274
C=/gscratch/makelab/jonf/crosswalk274
S=scripts/analysis/stage1_crosswalk_sdot_274/score_274.slurm
export RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 RELEASED=0 BPRIME=1 ROOT=$W/root_v4
for n in 1 2 3; do
    label=crosswalk_s4; [ "$n" != 1 ] && label=crosswalk_s4_seed$n
    export FT_LABEL=$label FT_CKPT=$C/v4_s$n/final_step_1000.pth FT_SHA=$(cut -d' ' -f1 $C/v4_s$n/final_step_1000.pth.sha256)
    j=$(IDS_FILE=$W/recall285/laurens-ia_ids.txt BPRIME_PANOS=$W/recall285/prep_4096 SPLIT_NAME=laurens-ia \
        BPRIME_PREDS=$W/recall285_preds sbatch --parsable --job-name=cw285_cal_lau_s$n $S)
    k=$(IDS_FILE=$W/recall285/juneau-ak_ids.txt BPRIME_PANOS=$W/recall285/prep_4096 SPLIT_NAME=juneau-ak \
        BPRIME_PREDS=$W/recall285_preds sbatch --parsable --dependency=afterany:$j --job-name=cw285_cal_jun_s$n $S)
    echo "seed $n recall panos: $j $k"
    if [ "$n" != 1 ]; then
        b=$(BPRIME_PREDS=$W/bprime_preds_v4 sbatch --parsable --job-name=cw285_cal_bp_s$n $S)
        e=$(IDS_FILE=$W/juneau_eval_ids.txt BPRIME_PANOS=$W/juneau_eval_4096 SPLIT_NAME=juneau_eval \
            BPRIME_PREDS=$W/juneau_eval_preds sbatch --parsable --job-name=cw285_cal_je_s$n $S)
        echo "seed $n bprime $b, juneau_eval $e"
    fi
done
squeue -u $USER
