#!/bin/bash
# Score the #82 fine-tune screen's checkpoints on all 12 benchmark bundles.
#
# Every committed cache was made on klone through aug82_score_ckpts.slurm, which calls this
# script with PY and CKPTS set (CKPTS=/gscratch/makelab/jonf/aug82). The makelab2 defaults
# below are only a fallback for running it by hand there.
#
# The frozen scorer, exactly as the seed-variance campaign ran it
# (scripts/model_comparison/yolo_baseline/run_seedvar_eval.sh, RampNet half):
#   operating_point_curve.py extract --checkpoint ... --score-floor 0.05 --min-distance 10,
#   no TTA, fp32 (it falls back to fp16 only on OOM and records it in the cache meta),
#   one --cache directory per checkpoint, because extract skips splits that already have
#   a file and does not check who wrote them.
# The released checkpoint is scored by the same call with no --checkpoint (the Hub
# safetensors), into its own directory, so every contrast is checkpoint-vs-checkpoint
# through one code path.
#
# USAGE by hand (makelab2, from a checkout of this branch whose benchmark/<split>/panos are
# present or symlinked to /homes/gws/jonf/RampNet/benchmark/<split>/panos):
#   CKPTS=/homes/gws/jonf/aug82_ckpts \
#     nohup scripts/analysis/aug82_score_ckpts.sh released control_s1 res_s1 ... > score.out 2>&1 &
# Each label other than "released" must exist as $CKPTS/<label>.pth (the klone
# final_step_<N>.pth, renamed on copy). SHA256SUMS must sit beside them and is checked
# first; scoring a checkpoint without it needs ALLOW_NO_SHA256SUMS=1.
#
# Exit status: 2 on a setup error, 1 if any label's extract failed or wrote fewer caches
# than there are splits (the remaining labels are still scored), 0 otherwise. There is no
# `set -e`: each extract's status is read and logged explicitly instead, so one failed
# label does not hide the result of the others.
#
# OUTPUT: analysis_out/aug_transfer_82/finetune/<label>/<split>.json, plus
# analysis_out/aug_transfer_82/finetune/driver.log (env, hashes, timings).
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-/homes/gws/jonf/RampNet/.venv/bin/python}"
CKPTS="${CKPTS:-/homes/gws/jonf/aug82_ckpts}"
OUT="${OUT:-$REPO/analysis_out/aug_transfer_82/finetune}"
SPLITS="${SPLITS:-annapolis,bend,budapest_district5,clovis,gainesville,laurens_gsv,laurens_mapillary,manual_gold,morgantown,paterson,richmond,sao_paulo}"
cd "$REPO" || exit 2
mkdir -p "$OUT"
LOG="$OUT/driver.log"
log() { echo "$*" | tee -a "$LOG"; }

need_sums=0
for label in "$@"; do [ "$label" = "released" ] || need_sums=1; done
if [ -f "$CKPTS/SHA256SUMS" ]; then
    (cd "$CKPTS" && sha256sum -c --quiet SHA256SUMS) || { log "FATAL: checkpoint hashes do not match $CKPTS/SHA256SUMS"; exit 2; }
elif [ "$need_sums" = 1 ] && [ "${ALLOW_NO_SHA256SUMS:-0}" != 1 ]; then
    log "FATAL: no $CKPTS/SHA256SUMS (set ALLOW_NO_SHA256SUMS=1 to score unhashed checkpoints)"
    exit 2
fi
n_splits=$(echo "$SPLITS" | tr ',' '\n' | grep -c .)
status=0
log "=== aug82 scoring (job ${SLURM_JOB_ID:-none}) $(date -Is) host $(hostname) repo commit $(git rev-parse --short HEAD)"
log "python: $PY ($("$PY" -c 'import torch,timm;print("torch",torch.__version__,"timm",timm.__version__)'))"
log "gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tr '\n' ' ')"
log "splits: $SPLITS"

for label in "$@"; do
    t0=$(date +%s)
    if [ "$label" = "released" ]; then
        ck=(); log "--- released: Hub projectsidewalk/rampnet-model (no --checkpoint)"
    else
        f="$CKPTS/$label.pth"
        [ -f "$f" ] || { log "missing $f"; exit 2; }
        ck=(--checkpoint "$f" --model-label "aug82_$label")
        log "--- $label: $f sha256 $(sha256sum "$f" | cut -c1-16)"
    fi
    "$PY" scripts/analysis/operating_point_curve.py extract ${ck[@]+"${ck[@]}"} \
        --cities "$SPLITS" --score-floor 0.05 --min-distance 10 --cache "$OUT/$label" \
        > "$OUT/${label}_extract.txt" 2>&1
    rc=$?
    n=$(ls "$OUT/$label"/*.json 2>/dev/null | wc -l)
    log "--- $label exit=$rc splits=$n elapsed=$(( $(date +%s) - t0 ))s"
    if [ "$rc" -ne 0 ] || [ "$n" -lt "$n_splits" ]; then
        log "FAILED: $label (exit $rc, $n of $n_splits caches); see $OUT/${label}_extract.txt"
        status=1
    fi
done
log "=== done $(date -Is) status=$status"
exit "$status"
