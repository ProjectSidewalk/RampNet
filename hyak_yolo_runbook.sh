#!/bin/bash
# ============================================================================
#  YOLO supervised-baseline training on Hyak (klone) — self-contained runbook.
#  The training half of issue #51 (the 'yolo' provider + eval land elsewhere).
#
#  Tracked operator runbook (like hyak_qwen_runbook.sh). Paths/identity are
#  parameterized ($USER, $SCRATCH), so edit nothing to run it. Claude cannot drive
#  klone (UW password + Duo per connection), so YOU run these stages; each is
#  idempotent and safe to re-run.
#
#  Usage on klone:   bash hyak_yolo_runbook.sh <stage>
#  Stages:           env | data | prepsmoke | prep | train | status | collect
#  Windows-side:     bash hyak_yolo_runbook.sh push   (prints the rsync commands)
#
#  Unlike the VLM runbook, this does NOT upload benchmark panos: YOLO is trained on
#  the RampNet dataset (pulled from HF straight onto the cluster), and EVAL RUNS
#  LOCALLY — you only rsync the small best.pt files back (see `collect`).
#
#  CRITICAL PATH (the two slow, unattended things; start them first):
#    1. `env`   — lean venv on scratch with ultralytics          (minutes)
#    2. `data`  — download the 214k-pano dataset from HF          (HOURS, big IO)
#    3. `prepsmoke` -> eyeball overlays -> `prep`                 (CPU, ~hours)
#    4. `train` — 6 configs, concurrent single-GPU jobs           (~hours, parallel)
# ============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/RampNet}"
USER="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/gscratch/scrubbed/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$SCRATCH/ultralytics}"

ENVDIR="${ENVDIR:-$SCRATCH/envs/yolo}"
PYBIN="${PYBIN:-$ENVDIR/bin/python}"
DATA="${DATA:-$SCRATCH/rampnet_dataset}"      # HF dataset lands here (via ./dataset symlink)
YOLODATA="${YOLODATA:-$SCRATCH/yolo}"         # prepared tiles/ and pano/ datasets
PROJECT="${PROJECT:-$SCRATCH/yolo_runs}"      # training outputs (weights)
# -A for the `train` sbatch: run_yolo_train.slurm pins the ckpt (scavenger) partition,
# which needs the ckpt account. The data/prep slurm wrappers use gpu-l40s-makelab.
ACCOUNT="${ACCOUNT:-ckpt-makelab}"
# As-run choice (see run_yolo_prep.slurm): fixed:0.03 was too small on the overlays,
# and gps fell back to pitch on ~85% of panos — pitch is the consistent strategy.
BOX_SIZE="${BOX_SIZE:-pitch}"
RAMP_SIZE_M="${RAMP_SIZE_M:-1.8}"             # physical ramp size the pitch model assumes
BG_KEEP="${BG_KEEP:-0.15}"                    # keep 15% of background tiles (tames the skew)
# YOLO26's checkpoint name/availability depends on the ultralytics version — check
# `python -c "from ultralytics import YOLO; YOLO('yolo26l.pt')"` and override YOLO26.
YOLO26="${YOLO26:-yolo26l.pt}"
STAGE="${1:-help}"

banner() { echo; echo "=== $* ==="; echo; }

case "$STAGE" in

# ---------------------------------------------------------------------------
help|push)
  cat <<'TXT'
Run from a shell with rsync (Windows: WSL). Define a `klone` host in ~/.ssh/config
(klone.hyak.uw.edu, your UW netid) with ControlMaster/ControlPath/ControlPersist so
Duo is entered once. Only the repo code goes up — no imagery, no venv, no caches:

  cd <parent dir of your RampNet checkout>
  rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
        --exclude 'benchmark/*/gallery' --exclude view_dump --exclude dataset \
        --exclude runs --exclude '*.pt' \
        RampNet/ klone:~/RampNet/

The dataset is pulled from HF ON the cluster (`data` stage), not uploaded. When
training finishes, pull ONLY the small weight files back and score locally:

  rsync -av --include '*/' --include 'best.pt' --include 'args.yaml' --exclude '*' \
        klone:/gscratch/scrubbed/<netid>/yolo_runs/ ./yolo_runs/
TXT
  ;;

# ---------------------------------------------------------------------------
env)
  banner "Lean env on scratch: python 3.11 + torch(cu126) + ultralytics"
  module load conda/Miniforge3-25.9.1-0
  if [ ! -x "$PYBIN" ]; then
    conda create -p "$ENVDIR" python=3.11 -y
  else
    echo "$ENVDIR already exists; skipping create"
  fi
  # cu126 wheels — the lean-env path from docs/model_comparison.md, no CPU-fallback trap.
  "$PYBIN" -m pip install --upgrade pip
  "$PYBIN" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  # ultralytics pulls numpy/pillow/opencv/pyyaml; datasets+hf_hub for download_dataset.py.
  "$PYBIN" -m pip install ultralytics datasets huggingface_hub
  banner "Verify — CUDA build survived, ultralytics imports"
  "$PYBIN" - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "| ultralytics", ultralytics.__version__)
if "cpu" in torch.__version__:
    raise SystemExit("STOP: CPU-only torch — reinstall from the cu126 index.")
PY
  echo "OK. PYBIN=$PYBIN   Next: bash hyak_yolo_runbook.sh data"
  ;;

# ---------------------------------------------------------------------------
data)
  banner "Download the RampNet dataset (~214k panos) from HF onto scratch"
  # Login nodes reap heavy long-running processes (and the SSH master with them) —
  # for the real multi-hour download, submit this stage as a compute-node job:
  #   sbatch -A gpu-l40s-makelab scripts/model_comparison/run_yolo_data.slurm
  mkdir -p "$DATA"
  ln -sfn "$DATA" "$REPO/dataset"     # download_dataset.py writes ./dataset/{train,val,test}
  cd "$REPO"
  echo "This is the long pole (tens of GB). Landing in: $DATA"
  "$PYBIN" download_dataset.py
  du -sh "$DATA" | tail -1
  # Glob-free count: `ls "$DATA/$s"/*.jpg` overflows ARG_MAX on the ~150k-file train
  # split, so `ls` errors out (to the suppressed stderr) and `wc -l` reports 0 — a
  # false "train: 0 panos" even on a healthy download. Enumerate via readdir instead.
  for s in train val test; do printf "  %s: %s panos\n" "$s" "$(ls -U "$DATA/$s" 2>/dev/null | grep -c '\.jpg$')"; done
  echo "OK. Next: bash hyak_yolo_runbook.sh prepsmoke"
  ;;

# ---------------------------------------------------------------------------
prepsmoke)
  banner "200-pano tiles build + overlays — eyeball boxes before the full run"
  cd "$REPO"
  "$PYBIN" scripts/model_comparison/prepare_yolo_dataset.py \
      --dataset-root "$REPO/dataset" --out "$YOLODATA/tiles_smoke" \
      --geometry tiles --box-size "$BOX_SIZE" --ramp-size-m "$RAMP_SIZE_M" \
      --bg-keep-frac "$BG_KEEP" \
      --subset 200 --overlay-dir "$YOLODATA/overlay_smoke"
  echo
  echo "scp $YOLODATA/overlay_smoke/*.jpg back and check red boxes sit ON ramps."
  echo "If using --box-size gps, confirm the 'fell back to pitch' count is near 0."
  echo "OK. Next: bash hyak_yolo_runbook.sh prep"
  ;;

# ---------------------------------------------------------------------------
prep)
  banner "Full tiles + pano datasets (box-size=$BOX_SIZE, ramp=${RAMP_SIZE_M}m, bg-keep=$BG_KEEP)"
  # The 150k-pano tiling is multi-hour and CPU-heavy — on klone, submit it as a
  # compute-node job instead of running here on a login node:
  #   sbatch -A gpu-l40s-makelab scripts/model_comparison/run_yolo_prep.slurm
  cd "$REPO"
  "$PYBIN" scripts/model_comparison/prepare_yolo_dataset.py \
      --dataset-root "$REPO/dataset" --out "$YOLODATA/tiles" \
      --geometry tiles --box-size "$BOX_SIZE" --ramp-size-m "$RAMP_SIZE_M" \
      --bg-keep-frac "$BG_KEEP"
  "$PYBIN" scripts/model_comparison/prepare_yolo_dataset.py \
      --dataset-root "$REPO/dataset" --out "$YOLODATA/pano" \
      --geometry pano --box-size "$BOX_SIZE" --ramp-size-m "$RAMP_SIZE_M"
  echo "OK. Next: bash hyak_yolo_runbook.sh train"
  ;;

# ---------------------------------------------------------------------------
train)
  banner "6 training jobs, concurrent (yolo11l/x/26 x tiles/pano)"
  cd "$REPO"
  mkdir -p logs
  # sbatch inherits the environment (--export=ALL default), so PYTHON/HF_HOME/etc.
  # reach the job. PYTHON=the lean venv; the .slurm falls back to conda otherwise.
  export PYTHON="$PYBIN" HF_HOME YOLO_CONFIG_DIR PROJECT
  SLURM=scripts/model_comparison/run_yolo_train.slurm
  sub() { # sub <ckpt> <data.yaml> <imgsz> <batch> <name>
    YOLO_CKPT="$1" YOLO_DATA="$2" YOLO_IMGSZ="$3" BATCH="$4" NAME="$5" \
      sbatch -A "$ACCOUNT" "$SLURM"
  }
  # Batches PINNED to the as-run values (sized for >=45G GPUs) so the LR schedule is
  # identical wherever a ckpt job lands or resumes — see run_yolo_train.slurm.
  # Caveat from the 2026-07 runs: YOLO11-pano collapsed at physical batch 2-4 (#70).
  sub yolo11l.pt "$YOLODATA/tiles/data.yaml" 1024 6  y11l_tiles
  sub "$YOLO26"  "$YOLODATA/tiles/data.yaml" 1024 6  y26_tiles
  sub yolo11l.pt "$YOLODATA/pano/data.yaml"  1280 4  y11l_pano
  sub yolo11x.pt "$YOLODATA/pano/data.yaml"  1280 2  y11x_pano
  sub "$YOLO26"  "$YOLODATA/pano/data.yaml"  1280 4  y26_pano

  # y11x_tiles is the one arm that does NOT go to ckpt. At batch 3 and again at batch
  # 12 it never finished a single epoch inside a ckpt scheduling slice, so it was
  # dropped 2026-07-27 with zero epochs. Restarted 2026-08-03 on the lab's dedicated
  # gpu-l40s — ONE node, a bounded exception to the "students keep gpu-l40s" rule,
  # taken only after the y26_tiles_l40s fork released its node on 2026-08-01.
  #
  # The evidence that this is worth a dedicated node: that fork ran the SAME config as
  # its ckpt twin y26_tiles, un-preempted, and reached epoch 18 / mAP50-95 0.425 while
  # the twin was still at epoch 1. The one-epoch wall was the scheduling slice, not the
  # model. 14 days because the fork's 72 h limit truncated it mid-schedule at ep18.
  #
  # Resources are the script's own #SBATCH defaults (12 CPU / 64G, workers 8) — the SAME
  # footprint as the fork, and identical dataloader width to all five ckpt arms, so
  # partition and wall limit are the only intentional differences in this arm.
  #
  # A 32-CPU / 28-worker version was submitted first (job 38063462) and cancelled before
  # it started. gpu-l40s-makelab is capped at cpu=32, gres/gpu=2, mem=386952M for the
  # WHOLE ACCOUNT — all 11 lab members, not per user — so 32 CPUs would have consumed
  # the lab's entire CPU budget, and could not start at all while another student held
  # 16. That cap, not grid parity, is why the 2026-08-02 I/O probe's "more workers are
  # free on klone" conclusion does NOT hold on this partition. Verify before assuming:
  #   sacctmgr -nP show assoc account=gpu-l40s-makelab format=Account,GrpTRES
  #
  # The 14-day limit is what actually rescues this arm; the worker count was never the
  # binding constraint. At the fork's ~4.5 h/epoch it buys tens of epochs against the
  # 18 the fork managed inside 72 h.
  YOLO_CKPT=yolo11x.pt YOLO_DATA="$YOLODATA/tiles/data.yaml" \
    YOLO_IMGSZ=1024 BATCH=12 NAME=y11x_tiles \
    sbatch -A gpu-l40s-makelab -p gpu-l40s -q normal --time=14-00:00:00 "$SLURM"
  squeue -u "$USER"
  echo "OK. Watch: bash hyak_yolo_runbook.sh status"
  ;;

# ---------------------------------------------------------------------------
status)
  squeue -u "$USER" || true
  echo
  for f in $(ls -t "$REPO"/logs/yolo_train_*.out 2>/dev/null | head -4); do
    echo "--- $f"; tail -12 "$f"
  done
  echo
  echo "Weights so far:"; find "$PROJECT" -name best.pt 2>/dev/null
  ;;

# ---------------------------------------------------------------------------
collect)
  banner "Trained weights (pull these home; eval runs locally)"
  find "$PROJECT" -name best.pt 2>/dev/null
  echo
  echo "From your machine (small files only):"
  echo "  rsync -av --include '*/' --include 'best.pt' --include 'args.yaml' --exclude '*' \\"
  echo "        klone:$PROJECT/ ./yolo_runs/"
  echo
  echo "Then locally, per weight (default tiling = tiles; --tiling none for pano):"
  echo "  for c in bend richmond clovis; do"
  echo "    python scripts/model_comparison/compare.py benchmark/\$c \\"
  echo "        --models rampnet,yolo --yolo-model ./yolo_runs/y11l_tiles/weights/best.pt \\"
  echo "        --sweep --pr-out pr_out/\$c; done"
  ;;

*)
  echo "unknown stage '$STAGE' (env | data | prepsmoke | prep | train | status | collect | push)"
  exit 2
  ;;
esac
