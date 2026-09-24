#!/usr/bin/env bash
# The #86 context experiment (RampNet 2.0 plan item 4), end to end, in run order
# (docs/context_fov_86.md). Two machines: the pano store is on makelab2, the GPUs on klone.
#
#   [WORK=/path/to/scratch] PY=/path/to/python [CONTROL=final|interim] \
#     bash scripts/analysis/context_fov_86.sh <stage...>
#
# Stages, in order:
#   cut-input   anywhere, CPU: the 10,853 HF labels with geometry -> analysis_out/context_fov_86/cut_input.csv
#   cut         makelab2, CPU (48 min at 12 workers in the committed run, both passes): the four
#               arms' crops -> $WORK/crops (+ manifests). Use WORKERS=6 if anything trains there.
#   labels      wherever the crops are mounted (the committed run: makelab1, same NFS home), CPU:
#               per-arm label tables on the common labels; the viewport arm's 640 px box, in place.
#               Before pack, so the 1440x960 viewport crops (6 of the 11 GB) never travel.
#   pack        makelab1/2: $WORK/context_fov_86_crops.tar, to move to klone; its sha256 ->
#               analysis_out/context_fov_86/crops_tar.sha256
#   unpack      klone: the tar -> $WORK/crops, then the sha256 of every crop as trained ->
#               analysis_out/context_fov_86/crops_as_trained.sha256
#   train       klone: one Slurm job per arm on the lab's L40S allocation (context_fov_86.slurm)
#   (the tar moves through a third machine: neither makelab nor klone holds a key for the other)
#   interim-infer  makelab2, GPU: only for CONTROL=interim, the dead #178 control run's epoch-49
#               snapshot over all 10,857 HF crops (TB_WORK = the #178 runbook's WORK), + ledger row
#   control     CPU: the #178 control's test rows restricted to the common labels (test-only),
#               then scored on them
#   contrast    CPU: every arm minus the control (all test rows, then leak-free rows only), and
#               every wider arm minus viewport, paired on the same rows and pano draws
#   report      CPU: the summary table
# CONTROL picks which control the last three stages read (default final):
#   final    #178's 100-epoch control, analysis_out/tag_benchmark_86/train_control_final_test_predictions.csv
#            (exists once `tag_benchmark_86.sh finish` has run and #178 is merged into this branch)
#            -> control_{test_predictions.csv,scores.json,per_label.csv}, contrast_vs_control{,_leak_free}.json,
#               summary.{json,md}
#   interim  the epoch-49 snapshot of the #178 run that died on 2026-09-23 (what §4 read first)
#            -> control_interim_ep49_*, contrast_vs_control_interim_ep49{,_leak_free}.json,
#               summary_interim_ep49.{json,md}
# WORK is needed only by cut, labels, pack, unpack, train; the CPU stages run in any checkout.
# The committed numbers came from: cut on makelab2 (48 cores), train on klone gpu-l40s (1x L40S per arm).
# PY on klone is the env context_fov_86_env.slurm builds; on makelab2 the tagger venv.
set -euo pipefail
PY=${PY:-python}
S=scripts/analysis/context_fov_86.py
TB=scripts/analysis/tag_benchmark_86.py
OUT=analysis_out/context_fov_86
TBOUT=analysis_out/tag_benchmark_86
ARMS=${ARMS:-"viewport fov25 fov50 fov90"}
STORE=${STORE:-/projects/makeabilitylab/sidewalk_panos/Panoramas}
WORKERS=${WORKERS:-12}
CONTROL=${CONTROL:-final}
STAGES=${*:-}
[ -n "$STAGES" ] || { echo "usage: $0 <stage...>  (cut-input cut labels pack unpack train interim-infer control contrast report)" >&2; exit 2; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

for st in cut labels pack unpack train; do
  if has $st; then
    WORK=${WORK:?set WORK to a scratch directory (stage $st needs it)}
    mkdir -p "$WORK"
    WORK=$(cd "$WORK" && pwd)
    TAGGER=$WORK/sidewalk-tagger-ai
    break
  fi
done

# The as-run makelab2 layout of #178 (docs/tag_benchmark_86.md): the dead run was moved to
# dead_2026-09-23/ before the re-launch (recorded on #178, 2026-09-24 04:15 UTC).
TB_WORK=${TB_WORK:-/homes/gws/jonf/nobackup/tagbench86}
INTERIM_CKPT=${INTERIM_CKPT:-$TB_WORK/dead_2026-09-23/train_control/best_after_ep49.pth}
INTERIM_ALL=${INTERIM_ALL:-$TB_WORK/control_interim_ep49_predictions_all.csv}
case $CONTROL in
  final)
    CONTROL_PRED=${CONTROL_PRED:-$TBOUT/train_control_final_test_predictions.csv}
    CP=control; CONTROL_LABEL="control (#178, HF crops)"; STEM=summary ;;
  interim)
    CONTROL_PRED=${CONTROL_PRED:-$INTERIM_ALL}
    CP=control_interim_ep49; CONTROL_LABEL="control (#178 epoch-49 snapshot, INTERIM)"; STEM=summary_interim_ep49 ;;
  *) echo "CONTROL must be final or interim, not '$CONTROL'" >&2; exit 2 ;;
esac

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
  # the manifests carry each crop's sha256 as cut; gzip -n so the committed copy is byte-stable
  for M in fov viewport; do gzip -n -9 -c "$WORK/crops/manifest_$M.jsonl" > $OUT/manifest_$M.jsonl.gz; done
fi
if has labels; then
  $PY $S labels --manifest "$WORK/crops/manifest_fov.jsonl" "$WORK/crops/manifest_viewport.jsonl" \
    --images "$WORK/crops" --arms $ARMS --out-dir $OUT
  # rewrites the viewport crops in place: their manifest sha256 is of the 1440x960 cut, not of
  # what trains; crops_as_trained.sha256 (unpack) is the hash of the trained-on files
  $PY $S crop640 --labels $OUT/labels_viewport.csv --images "$WORK/crops"
fi
if has pack; then
  tar -cf "$WORK/context_fov_86_crops.tar" -C "$WORK" crops
  (cd "$WORK" && sha256sum context_fov_86_crops.tar) | tee $OUT/crops_tar.sha256
fi
if has unpack; then
  tar -xf "$WORK/context_fov_86_crops.tar" -C "$WORK"
  (cd "$WORK/crops" && find . -maxdepth 1 -name '*.jpg' -printf '%f\n' | LC_ALL=C sort | xargs sha256sum) \
    > $OUT/crops_as_trained.sha256
fi
if has train; then
  mkdir -p logs
  for ARM in $ARMS; do
    sbatch --job-name="ctx_$ARM" --export=ALL,ARM="$ARM",WORK="$WORK",PY="$PY",REPO="$(pwd)" \
      scripts/analysis/context_fov_86.slurm
  done
fi
if has interim-infer; then
  # Reconstructed from the committed meta (control_interim_ep49_test_predictions.csv.meta.json:
  # checkpoint sha256, csv, 10,857 crops) and ledger row (infer-control-ep49-interim); the
  # as-run shell line was not saved. Only CONTROL=interim reads its output.
  DATA=$TB_WORK/sidewalk-tagger-ai/datasets/crops-curbramp-tags
  START=$(date -u +%FT%TZ)
  $PY $TB infer --tagger-repo "$TB_WORK/sidewalk-tagger-ai" --checkpoint "$INTERIM_CKPT" \
    --csv $TBOUT/hf_curbramp_labels.csv --images "$DATA/train" "$DATA/test" --out "$INTERIM_ALL"
  ELAPSED=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['elapsed_s'])" "$INTERIM_ALL.meta.json")
  $PY $TB log-usage --label infer-control-ep49-interim --elapsed-s "$ELAPSED" --n 10857 \
    --run-id "context-fov-86:infer-control-ep49-interim:$START" --ts "$START" \
    --concurrent-with train-control,train-pano,train-cell \
    --what "INTERIM control for the #86 context experiment: tag_benchmark_86.py infer with the dead #178 control run's best_after_ep49.pth over all 10,857 HF crops" \
    --log analysis_out/usage_log.jsonl
fi
if has control; then
  # the #178 control (the HF crops, the published split) on exactly the common test rows,
  # through the same test-only filter the arms went through, then scored on them
  $PY $TB test-only --pred "$CONTROL_PRED" --labels $TBOUT/hf_curbramp_labels.csv \
    --split-csv $OUT/split_common.csv --out $OUT/${CP}_test_predictions.csv
  $PY $TB score --pred $OUT/${CP}_test_predictions.csv --labels $TBOUT/hf_curbramp_labels.csv \
    --split-csv $OUT/split_common.csv --out $OUT/${CP}_scores.json --per-label-out $OUT/${CP}_per_label.csv
fi
if has contrast; then
  # each arm minus the control, then each wider arm minus viewport (the re-cut at the
  # control's framing). The viewport contrast does not read the control; re-running it
  # rewrites the same rows (seeded draws) with a new ts.
  $PY $S contrast --reference $CP --reference-pred $OUT/${CP}_test_predictions.csv \
    --reference-labels $TBOUT/hf_curbramp_labels.csv --arms $ARMS --out $OUT/contrast_vs_${CP}.json
  # the same on the leak-free rows only (the test rows whose pano has no train label)
  $PY $S contrast --reference $CP --reference-pred $OUT/${CP}_test_predictions.csv \
    --reference-labels $TBOUT/hf_curbramp_labels.csv --arms $ARMS --subset leak_free \
    --out $OUT/contrast_vs_${CP}_leak_free.json
  WIDER=$(for A in $ARMS; do [ "$A" = viewport ] || printf '%s ' "$A"; done)
  $PY $S contrast --reference viewport --reference-pred $OUT/train_viewport_final_test_predictions.csv \
    --reference-labels $OUT/labels_viewport.csv --arms $WIDER --out $OUT/contrast_vs_viewport.json
fi
if has report; then
  $PY $S report --out-dir $OUT --control-scores $OUT/${CP}_scores.json --control-label "$CONTROL_LABEL" \
    --arms $ARMS --out-stem $STEM
fi
