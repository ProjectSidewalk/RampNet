#!/bin/bash
# Score the seed-variance replicates (#51, #135) on the benchmark -- both campaigns.
#
# THE QUESTION
#   docs/operating_point_parity_51.md put RampNet 0.039 F1 ahead of the y11x_tiles
#   baseline at matched operating points, on ONE training run of each. Two campaigns of
#   three seed replicates each (docs/seed_variance_51_135.md) exist to say whether 0.039
#   is bigger than seed noise. This driver produces the per-replicate inputs that
#   scripts/analysis/seed_variance_read_51_135.py turns into the pre-registered statistic.
#   It runs no statistics itself.
#
# WHAT IT SCORES
#   YOLO, six legs in one compare.py call per split (same geometry, so one table):
#     y11x_tiles_s{1,2,3}_ep{44,44,42}.pt   PRIMARY   best metrics/mAP50-95(B) epoch <= 44
#     y11x_tiles_s{1,2,3}_best.pt            SECONDARY as-saved best.pt (<= 60 epochs)
#   The file STEM is the leg label -- compare.py has no --yolo-label, so the checkpoints
#   were renamed on copy (their sha256s are in seedvar_ckpts/SHA256SUMS).
#   _ep<N> IS THE 1-BASED results.csv EPOCH, AND THAT IS NOT THE FILE NAME ON DISK.
#   Ultralytics writes epoch{self.epoch}.pt with the 0-based counter, so results.csv
#   row 44 is epoch43.pt. The 2026-09-15 run copied epoch44/44/42.pt as _ep44/44/42 and
#   scored one epoch late (two legs outside the <=44 window; PR #161). Every _ep<N> leg
#   is now checked against ckpt["epoch"]+1 by check_epoch_ckpt.py before anything runs.
#   Flags are the seed-0 arm's exactly: perspective tiling, imgsz 1024, cache floor 0.05,
#   full sweep. See run_yolo_geometry_eval.sh.
#
#   RampNet, one extract per replicate into its own cache dir:
#     rampnet_s{1,2,3}_best.pth              PRIMARY   Stage 2 best_model.pth, 1 epoch
#   operating_point_curve.py extract --checkpoint, floor 0.05, min_distance 10, no TTA --
#   the committed analysis_out/op_cache arm. One --cache per replicate is load-bearing:
#   extract skips splits that already have a file and does not check who wrote them.
#
# SPLITS
#   The seven pooled US splits + sao_paulo (the dev split the threshold is chosen on)
#   is the pre-registered set; manual_gold is a labeled secondary read added 2026-09-15.
#
# OUTPUT
#   $OUT/yolo/<split>_tiles.txt, $OUT/rampnet_s<N>/<split>.json, env.txt, driver.log.
#   Copy those (not the pr_* dirs) into docs/data/seed_variance_51_135/.
#
# USAGE (makelab2)
#   nohup scripts/model_comparison/yolo_baseline/run_seedvar_eval.sh > seedvar_driver.out 2>&1 &
#   ... SPLITS can be overridden positionally, e.g. `run_seedvar_eval.sh richmond bend`.
#   ARMS="yolo" (or "rampnet") scores one half only -- the 2026-09-17 re-score touched
#   the YOLO legs and left the RampNet caches as scored. Point OUT at a fresh dir for a
#   re-score so the earlier outputs stay on disk beside the new ones.
#   PY_YOLO is the ultralytics env, PY_RN the Stage 2 (torch+timm) env; they differ on
#   makelab2, which is why there are two.
#
set -u

REPO="${REPO:-/homes/gws/jonf/RampNet}"
PY_YOLO="${PY_YOLO:-$REPO/.venv-eval/bin/python}"
PY_RN="${PY_RN:-$REPO/.venv/bin/python}"
OUT="${OUT:-$REPO/seedvar_eval_51_135}"
CKPTS="${CKPTS:-$REPO/seedvar_ckpts}"

YOLO_LEGS=(y11x_tiles_s1_ep44 y11x_tiles_s2_ep44 y11x_tiles_s3_ep42
           y11x_tiles_s1_best y11x_tiles_s2_best y11x_tiles_s3_best)
RN_SEEDS=(1 2 3)
ARMS="${ARMS:-yolo rampnet}"

cd "$REPO" || exit 2
mkdir -p "$OUT/yolo"

SPLITS_DEFAULT=(richmond bend clovis morgantown annapolis paterson gainesville sao_paulo manual_gold)
if [ "$#" -gt 0 ]; then SPLITS=("$@"); else SPLITS=("${SPLITS_DEFAULT[@]}"); fi
SPLITS_CSV=$(IFS=,; echo "${SPLITS[*]}")

models=""
for leg in "${YOLO_LEGS[@]}"; do
  [ -f "$CKPTS/$leg.pt" ] || { echo "missing checkpoint: $CKPTS/$leg.pt" >&2; exit 2; }
  models="${models:+$models,}yolo:$CKPTS/$leg.pt"
done
for s in "${RN_SEEDS[@]}"; do
  [ -f "$CKPTS/rampnet_s${s}_best.pth" ] || { echo "missing checkpoint: $CKPTS/rampnet_s${s}_best.pth" >&2; exit 2; }
done

# The label must be the epoch the file holds. Refuses to score otherwise (see header).
if ! (cd "$CKPTS" && "$PY_YOLO" "$REPO/scripts/model_comparison/yolo_baseline/check_epoch_ckpt.py" "${YOLO_LEGS[@]/%/.pt}"); then
  echo "epoch label check FAILED -- not scoring" >&2; exit 3
fi

# Provenance first. A number whose checkpoint hash is not written down cannot be
# re-derived by someone else; the full dirty list, not a count, says whether the
# uncommitted paths were outputs or scoring code.
{
  echo "run started      : $(date -Is)"
  echo "host             : $(hostname)"
  echo "repo HEAD        : $(git rev-parse HEAD)  $(git log -1 --format=%s)"
  echo "repo dirty       : $(git status --porcelain | wc -l) paths"
  git status --porcelain | sed 's/^/                   /'
  echo "yolo env         : $PY_YOLO"
  "$PY_YOLO" -c "import torch,ultralytics;print('  torch',torch.__version__,'cuda',torch.cuda.is_available());print('  ultralytics',ultralytics.__version__)"
  echo "rampnet env      : $PY_RN"
  "$PY_RN" -c "import torch,timm;print('  torch',torch.__version__,'cuda',torch.cuda.is_available());print('  timm',timm.__version__)"
  echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "checkpoint sha256:"
  (cd "$CKPTS" && sha256sum "${YOLO_LEGS[@]/%/.pt}" rampnet_s{1,2,3}_best.pth)
  echo "splits           : ${SPLITS[*]}"
  echo "arms             : $ARMS"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

log() { echo "$*" | tee -a "$OUT/driver.log"; }

# --- YOLO: one call per split, all six legs ----------------------------------------
case " $ARMS " in *" yolo "*) for b in "${SPLITS[@]}"; do
  if [ ! -d "benchmark/$b" ]; then
    log "=== $b SKIPPED (no bundle dir)"
    continue
  fi
  log "=== yolo $b start $(date -Is)"
  t0=$(date +%s)
  "$PY_YOLO" scripts/model_comparison/compare.py "benchmark/$b" \
      --models "$models" \
      --tiling perspective --yolo-imgsz 1024 \
      --op-threshold 0.25 --sweep --pr-out "$OUT/pr_${b}_tiles" \
      > "$OUT/yolo/${b}_tiles.txt" 2>&1
  rc=$?
  n=$(grep -c "threshold sweep" "$OUT/yolo/${b}_tiles.txt")
  log "=== yolo $b exit=$rc sweeps=$n/${#YOLO_LEGS[@]} elapsed=$(( $(date +%s) - t0 ))s"
done ;; esac

# --- RampNet: one extract per replicate, own cache dir -----------------------------
case " $ARMS " in *" rampnet "*) for s in "${RN_SEEDS[@]}"; do
  log "=== rampnet_s$s start $(date -Is)"
  t0=$(date +%s)
  "$PY_RN" scripts/analysis/operating_point_curve.py extract \
      --checkpoint "$CKPTS/rampnet_s${s}_best.pth" --model-label "rampnet_s$s" \
      --cities "$SPLITS_CSV" --score-floor 0.05 --min-distance 10 \
      --cache "$OUT/rampnet_s$s" \
      > "$OUT/rampnet_s${s}_extract.txt" 2>&1
  rc=$?
  n=$(ls "$OUT/rampnet_s$s"/*.json 2>/dev/null | wc -l)
  log "=== rampnet_s$s exit=$rc splits=$n/${#SPLITS[@]} elapsed=$(( $(date +%s) - t0 ))s"
done ;; esac

log "ALL_DONE $(date -Is)"
