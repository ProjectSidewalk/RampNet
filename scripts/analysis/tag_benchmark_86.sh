#!/usr/bin/env bash
# The #86 tag benchmark of record, end to end, in run order (docs/tag_benchmark_86.md).
#
# Run from the root of a RampNet checkout on a Linux box with one CUDA GPU (the committed
# numbers came from makelab2, 1x A40). Everything large lands under $WORK, a local cache
# that is never committed: the 30.8 GB HF zip, the extracted crops, a pinned checkout of
# sidewalk-tagger-ai, and the checkpoints.
#
#   WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/tag_benchmark_86.sh [stage...]
#
# Stages: fetch prepare eval infer score  (the default: the released checkpoint)
#         train      launch the three retrain arms concurrently, plus the snapshot watcher,
#                    in the background (~17 h on one A40), exactly as the committed run did
#         snapshots  while or after they run: score every best_after_ep<E>.pth that exists
#         finish     once all three train_<arm>.log end in "EXIT 0": snapshots + final best.pth
# snapshots and finish write analysis_out/tag_benchmark_86/train_<arm>_{ep<E>,final}_* and
# rows in analysis_out/usage_log.jsonl; commit both. Re-running either is safe: outputs are
# rewritten and usage rows carry a run_id, so a re-written row replaces its predecessor.
# PY needs torch + torchvision (CUDA), scikit-learn, pandas, pillow, matplotlib, timm,
# requests. The committed runs used Python 3.10, torch 2.4.1+cu121, torchvision 0.19.1,
# scikit-learn 1.7.2, pandas 2.3.3, without xformers.
set -euo pipefail

WORK=${WORK:?set WORK to a scratch directory with ~70 GB free}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)   # absolute: the train stage's background jobs and later stages share it
PY=${PY:-python}
TAGGER_SHA=3b7405cd3206ece631cb7a65e22b1ab219df4b75
HF_DATASET_REV=6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50   # projectsidewalk/sidewalk-tagger-ai-validated
HF_MODEL_REV=65959dbc80b87e4f39385204c4b639cbcf58e1a8     # projectsidewalk/sidewalk-tagger-ai-models
TAGGER=$WORK/sidewalk-tagger-ai
DATA=$TAGGER/datasets/crops-curbramp-tags     # the layout the tagger's evaluate.py expects
OUT=analysis_out/tag_benchmark_86
S=scripts/analysis/tag_benchmark_86.py
STAGES=${*:-fetch prepare eval infer score}

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has fetch; then
  mkdir -p "$WORK"
  [ -d "$TAGGER" ] || git clone -q https://github.com/ProjectSidewalk/sidewalk-tagger-ai.git "$TAGGER"
  git -C "$TAGGER" checkout -q $TAGGER_SHA
  cd "$WORK"
  # pinned revisions (docs/tag_benchmark_86.md §6); the sha256 check below stops on any mismatch
  [ -f CurbRamp.zip ] || curl -fsSL -C - -o CurbRamp.zip \
    https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/$HF_DATASET_REV/Validated/CurbRamp.zip
  [ -f validated-dino-cls-b-curbramp-tags-best.pth ] || curl -fsSL -o validated-dino-cls-b-curbramp-tags-best.pth \
    https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/$HF_MODEL_REV/validated-dino-cls-b-curbramp-tags-best.pth
  [ -f dinov2_vitb14_reg4_pretrain.pth ] || curl -fsSL -o dinov2_vitb14_reg4_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
  sha256sum -c - <<SUMS
5a8568353d720084ce4ec170ad9e3cc2b57b8d549bdd5d93099c180b82ed9a2d  CurbRamp.zip
4d00193aed73fc199049f31cebade51f236bfca92ad9f76adf08a9d08a272833  validated-dino-cls-b-curbramp-tags-best.pth
73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  dinov2_vitb14_reg4_pretrain.pth
SUMS
  cd - >/dev/null
  # the layout REPRODUCE_RESULTS.md sets up
  mkdir -p "$TAGGER/notebooks/models"
  ln -sf "$WORK/validated-dino-cls-b-curbramp-tags-best.pth" "$TAGGER/notebooks/models/"
  ln -sf "$WORK/dinov2_vitb14_reg4_pretrain.pth" "$TAGGER/"
fi

if has prepare; then
  # extract, then run the tagger's own crop.py:crop_image over each split (640 px box)
  $PY $S prepare --tagger-repo "$TAGGER" --zip "$WORK/CurbRamp.zip" --out "$DATA" --splits test train
fi

if has eval; then
  # reproduction of record: the tagger's notebooks/evaluate.py, unmodified
  $PY $S tagger-eval --tagger-repo "$TAGGER" --log "$WORK/tagger_eval.log" \
    --out $OUT/released_tagger_evaluate_py.json
fi

if has infer; then
  # the same checkpoint through this repo's loop, to get per-label scores
  $PY $S infer --tagger-repo "$TAGGER" --checkpoint "$WORK/validated-dino-cls-b-curbramp-tags-best.pth" \
    --csv "$DATA/test/test.csv" --images "$DATA/test" --out $OUT/released_test_predictions.csv
fi

if has score; then
  # CPU: full / leak-free / leaked, pano-clustered bootstrap
  $PY $S score --pred $OUT/released_test_predictions.csv --out $OUT/released_scores.json \
    --per-label-out $OUT/released_test_per_label.csv
fi

ARMS="control pano cell"
split_args() {  # the split each arm trains and is scored on (control: the published HF split)
  case $1 in
    control) ;;
    pano) echo "--split-csv $OUT/resplit_pano_grouped_seed86.csv" ;;
    cell) echo "--split-csv $OUT/resplit_cell100m_seed86.csv" ;;
  esac
}

if has train; then
  # The tagger's training recipe, three arms at once on one GPU (the committed run: makelab2
  # A40, ~605 s/epoch each while sharing it), each followed by inference of its final best.pth
  # on all 10,857 crops. Paths are absolute so a later stage finds them. The wrapper's first
  # line (date -u) is the start time `collect` reads, and "EXIT <rc>" marks the end.
  REPO=$(pwd)
  for ARM in $ARMS; do
    [ -e "$WORK/train_$ARM.log" ] && { echo "$WORK/train_$ARM.log exists; refusing to start $ARM over it" >&2; exit 1; }
    nohup bash -c "date -u +%FT%TZ; \
      $PY $S train --tagger-repo $TAGGER $(split_args $ARM) --images $DATA/train $DATA/test \
        --backbone $WORK/dinov2_vitb14_reg4_pretrain.pth --out-dir $WORK/train_$ARM \
      && $PY $S infer --tagger-repo $TAGGER --checkpoint $WORK/train_$ARM/best.pth \
        --csv $REPO/$OUT/hf_curbramp_labels.csv --images $DATA/train $DATA/test \
        --out $WORK/train_${ARM}_predictions.csv; \
      echo EXIT \$?; date -u +%FT%TZ" > "$WORK/train_$ARM.log" 2>&1 &
  done
  sleep 5
  WORK=$WORK ARMS="$ARMS" nohup bash scripts/analysis/tag_benchmark_86_snap.sh >> "$WORK/snap.log" 2>&1 &
  echo "launched; watch $WORK/train_{control,pano,cell}.log and $WORK/snap.log"
fi

snapshot_infer() {  # GPU: score each best_after_ep<E>.pth that has no predictions yet
  for E in 4 9 19 49; do
    for ARM in $ARMS; do
      CK=$WORK/train_$ARM/best_after_ep$E.pth
      P=$WORK/snap_ep${E}_${ARM}_predictions.csv
      [ -f "$CK" ] && [ ! -f "$P.meta.json" ] || continue
      $PY $S infer --tagger-repo "$TAGGER" --checkpoint "$CK" --csv $OUT/hf_curbramp_labels.csv \
        --images "$DATA/train" "$DATA/test" --out "$P"
    done
  done
}

if has snapshots; then
  snapshot_infer
  # CPU: test rows, scores, one usage row per inference (train rows stay as they are)
  $PY $S collect --work "$WORK"
fi

if has finish; then
  for ARM in $ARMS; do
    grep -qx 'EXIT 0' "$WORK/train_$ARM.log" || { echo "$ARM has not ended with EXIT 0; not finishing" >&2; exit 1; }
  done
  snapshot_infer
  # CPU: everything `snapshots` does, plus each arm's final best.pth (epoch read from the
  # checkpoint and train_meta.json, not assumed), its train log/meta under arm-prefixed
  # names, and the final training row, which replaces the in_progress row (same run_id)
  $PY $S collect --work "$WORK" --final
fi
