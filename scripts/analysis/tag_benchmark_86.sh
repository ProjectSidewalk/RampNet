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
# Stages: fetch prepare eval infer score train  (default: all but train)
# PY needs torch + torchvision (CUDA), scikit-learn, pandas, pillow, matplotlib, timm,
# requests. The committed runs used Python 3.10, torch 2.4.1+cu121, torchvision 0.19.1,
# scikit-learn 1.7.2, pandas 2.3.3, without xformers.
set -euo pipefail

WORK=${WORK:?set WORK to a scratch directory with ~70 GB free}
PY=${PY:-python}
TAGGER_SHA=3b7405cd3206ece631cb7a65e22b1ab219df4b75
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
  # HF dataset projectsidewalk/sidewalk-tagger-ai-validated @ 6e3a116a (LFS sha256 5a856835...)
  [ -f CurbRamp.zip ] || curl -sSL -C - -o CurbRamp.zip \
    https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/main/Validated/CurbRamp.zip
  # HF model projectsidewalk/sidewalk-tagger-ai-models @ 65959dbc (LFS sha256 4d00193a...)
  [ -f validated-dino-cls-b-curbramp-tags-best.pth ] || curl -sSL -o validated-dino-cls-b-curbramp-tags-best.pth \
    https://huggingface.co/projectsidewalk/sidewalk-tagger-ai-models/resolve/main/validated-dino-cls-b-curbramp-tags-best.pth
  [ -f dinov2_vitb14_reg4_pretrain.pth ] || curl -sSL -o dinov2_vitb14_reg4_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
  sha256sum CurbRamp.zip validated-dino-cls-b-curbramp-tags-best.pth dinov2_vitb14_reg4_pretrain.pth
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

if has train; then
  # the tagger's training recipe, three arms; ~GPU-hours each, see the doc for measured times
  FIXED=missing-tactile-warning,narrow,not-enough-landing-space,not-level-with-street,points-into-traffic,pooled-water,steep,surface-problem
  for ARM in control pano cell; do
    case $ARM in
      control) SPLIT=() ;;
      pano) SPLIT=(--split-csv $OUT/resplit_pano_grouped_seed86.csv) ;;
      cell) SPLIT=(--split-csv $OUT/resplit_cell100m_seed86.csv) ;;
    esac
    $PY $S train --tagger-repo "$TAGGER" "${SPLIT[@]}" --images "$DATA/train" "$DATA/test" \
      --backbone "$WORK/dinov2_vitb14_reg4_pretrain.pth" --out-dir "$WORK/train_$ARM"
    cp "$WORK/train_$ARM/train_log.csv" $OUT/train_${ARM}_log.csv
    cp "$WORK/train_$ARM/train_meta.json" $OUT/train_${ARM}_meta.json
    $PY $S infer --tagger-repo "$TAGGER" --checkpoint "$WORK/train_$ARM/best.pth" \
      --csv $OUT/hf_curbramp_labels.csv --images "$DATA/train" "$DATA/test" \
      --out $OUT/train_${ARM}_predictions.csv
    $PY $S score --pred $OUT/train_${ARM}_predictions.csv "${SPLIT[@]}" --fixed-tags $FIXED \
      --out $OUT/train_${ARM}_scores.json
  done
fi
