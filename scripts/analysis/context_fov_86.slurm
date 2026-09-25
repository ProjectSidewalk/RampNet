#!/bin/bash
# One arm of the #86 context experiment (plan item 4) on klone: train the tagger's recipe on
# the arm's crops, score its best checkpoint on the common test rows, write the ledger row.
# Launched per arm by `context_fov_86.sh train`; the lab's L40S allocation, so no preemption
# (train has no resume). ARM, WORK, PY, REPO arrive through --export.
#
#SBATCH --account=gpu-l40s-makelab
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ctx_%x_%j.out
set -eo pipefail
ARM=${ARM:?}; WORK=${WORK:?}; REPO=${REPO:?}
PY=${PY:-/gscratch/makelab/jonf/envs/tagger/bin/python}
cd "$REPO"
S=scripts/analysis/tag_benchmark_86.py
OUT=analysis_out/context_fov_86
TAGGER=$WORK/sidewalk-tagger-ai
START=$(date -u +%FT%TZ); T0=$SECONDS
HOST=$(hostname -f); GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
RUN_ID="context-fov-86:train-$ARM:$START"
WHAT="tag_benchmark_86.py train (100 ep) + infer on the $ARM arm of the #86 context experiment, Slurm job $SLURM_JOB_ID"
echo "$START $HOST $GPU arm=$ARM job=$SLURM_JOB_ID"
$PY $S log-usage --label "train-$ARM" --elapsed-s 0 --status in_progress --run-id "$RUN_ID" \
  --host "$HOST" --gpu "$GPU" --ts "$START" --what "$WHAT" --log analysis_out/usage_log.jsonl
$PY $S train --tagger-repo "$TAGGER" --labels $OUT/labels_$ARM.csv --split-csv $OUT/split_common.csv \
  --images "$WORK/crops" --backbone "$WORK/dinov2_vitb14_reg4_pretrain.pth" --out-dir "$WORK/train_$ARM"
$PY $S infer --tagger-repo "$TAGGER" --checkpoint "$WORK/train_$ARM/best.pth" \
  --csv $OUT/labels_$ARM.csv --images "$WORK/crops" --out "$WORK/train_${ARM}_predictions.csv"
$PY $S test-only --pred "$WORK/train_${ARM}_predictions.csv" --labels $OUT/labels_$ARM.csv \
  --split-csv $OUT/split_common.csv --out $OUT/train_${ARM}_final_test_predictions.csv
$PY $S score --pred $OUT/train_${ARM}_final_test_predictions.csv --labels $OUT/labels_$ARM.csv \
  --split-csv $OUT/split_common.csv --out $OUT/train_${ARM}_final_scores.json \
  --per-label-out $OUT/train_${ARM}_final_per_label.csv
cp "$WORK/train_$ARM/train_log.csv" $OUT/train_${ARM}_train_log.csv
cp "$WORK/train_$ARM/train_meta.json" $OUT/train_${ARM}_train_meta.json
sha256sum "$WORK/train_$ARM/best.pth" > $OUT/train_${ARM}_best_pth.sha256
$PY $S log-usage --label "train-$ARM" --elapsed-s $((SECONDS - T0)) --status ok --run-id "$RUN_ID" \
  --host "$HOST" --gpu "$GPU" --ts "$START" --what "$WHAT" --log analysis_out/usage_log.jsonl
echo "EXIT 0"; date -u +%FT%TZ
