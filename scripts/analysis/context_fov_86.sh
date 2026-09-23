#!/usr/bin/env bash
# The #86 context experiment (RampNet 2.0 plan item 4), end to end, in run order
# (docs/context_fov_86.md). Two machines: the pano store is on makelab2, the GPUs on klone.
#
#   WORK=/path/to/scratch PY=/path/to/python bash scripts/analysis/context_fov_86.sh <stage...>
#
# Stages, in order:
#   cut-input   anywhere, CPU: the 10,853 HF labels with geometry -> analysis_out/context_fov_86/cut_input.csv
#   cut         makelab2, CPU (60 min at 12 workers, 18 crops/s): the four arms' crops -> $WORK/crops (+ manifests)
#   labels      wherever the crops are mounted (the committed run: makelab1, same NFS home), CPU:
#               per-arm label tables on the common labels; the viewport arm's 640 px box, in place.
#               Before pack, so the 1440x960 viewport crops (6 of the 11 GB) never travel.
#   pack        makelab1/2: $WORK/context_fov_86_crops.tar, to move to klone
#   unpack      klone: the tar -> $WORK/crops
#   train       klone: one Slurm job per arm on the lab's L40S allocation (context_fov_86.slurm)
#   (the tar moves through a third machine: neither makelab nor klone holds a key for the other)
#   control     CPU: the #178 control arm re-scored on the common test rows
#   report      CPU: the summary table
# The committed numbers came from: cut on makelab2 (48 cores), train on klone gpu-l40s (1x L40S per arm).
# PY on klone is the env context_fov_86_env.slurm builds; on makelab2 the tagger venv.
set -euo pipefail
WORK=${WORK:?set WORK to a scratch directory}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
PY=${PY:-python}
S=scripts/analysis/context_fov_86.py
TB=scripts/analysis/tag_benchmark_86.py
OUT=analysis_out/context_fov_86
ARMS=${ARMS:-"viewport fov25 fov50 fov90"}
STORE=${STORE:-/projects/makeabilitylab/sidewalk_panos/Panoramas}
WORKERS=${WORKERS:-12}
TAGGER=$WORK/sidewalk-tagger-ai
CONTROL_PRED=${CONTROL_PRED:-analysis_out/tag_benchmark_86/train_control_final_test_predictions.csv}
STAGES=${*:-}
[ -n "$STAGES" ] || { echo "usage: $0 <stage...>  (cut-input cut labels pack unpack train control report)" >&2; exit 2; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has cut-input; then
  $PY $S cut-input
fi
if has cut; then
  # label-centred square arms in one pass (one decode per pano), then the viewport arm at
  # the HF crop's size. Both resumable; both leave a manifest beside the crops.
  $PY scripts/crop_cutter.py --labels $OUT/cut_input.csv --store "$STORE" \
    --fov 25 --fov 50 --fov 90 --size 640 --aspect 1.0 \
    --out "$WORK/crops" --manifest "$WORK/crops/manifest_fov.jsonl" --summary $OUT/cut_summary_fov.json \
    --workers "$WORKERS"
  $PY scripts/crop_cutter.py --labels $OUT/cut_input.csv --store "$STORE" \
    --fov viewport \
    --out "$WORK/crops" --manifest "$WORK/crops/manifest_viewport.jsonl" --summary $OUT/cut_summary_viewport.json \
    --workers "$WORKERS"
fi
if has labels; then
  $PY $S labels --manifest "$WORK/crops/manifest_fov.jsonl" "$WORK/crops/manifest_viewport.jsonl" \
    --images "$WORK/crops" --arms $ARMS --out-dir $OUT
  $PY $S crop640 --labels $OUT/labels_viewport.csv --images "$WORK/crops"
fi
if has pack; then
  tar -cf "$WORK/context_fov_86_crops.tar" -C "$WORK" crops
  sha256sum "$WORK/context_fov_86_crops.tar"
fi
if has unpack; then
  tar -xf "$WORK/context_fov_86_crops.tar" -C "$WORK"
fi
if has train; then
  mkdir -p logs
  for ARM in $ARMS; do
    sbatch --job-name="ctx_$ARM" --export=ALL,ARM="$ARM",WORK="$WORK",PY="$PY",REPO="$(pwd)" \
      scripts/analysis/context_fov_86.slurm
  done
fi
if has control; then
  # the #178 control (the HF crops, the published split) on exactly the common test rows
  $PY $TB score --pred "$CONTROL_PRED" --labels analysis_out/tag_benchmark_86/hf_curbramp_labels.csv \
    --split-csv $OUT/split_common.csv --out $OUT/control_scores.json --per-label-out $OUT/control_per_label.csv
fi
if has report; then
  $PY $S report --out-dir $OUT --control-scores $OUT/control_scores.json --arms $ARMS
fi
