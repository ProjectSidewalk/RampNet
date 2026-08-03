#!/bin/bash
# Train one YOLO curb-ramp detector on the RampNet dataset, on Hyak (klone).
# The supervised baseline of issue #51.
#
# Pass your Slurm account with -A (no usable default; a submission without one is
# rejected). Find it with: sacctmgr -nP show assoc user=$USER format=Account,QOS
#
# ONE config per job, parameterized by env vars, so the 6 configs
# (yolo11l/x/26 x tiles/pano) are the same script sbatch'd concurrently. They run on
# the ckpt (scavenger) partition so they NEVER touch the lab's dedicated gpu-l40s
# allocation — students keep that. ckpt is PREEMPTABLE (PreemptMode=REQUEUE): an
# owner reclaiming a node requeues the job, which then RESUMES from weights/last.pt
# (see the resume block below), so a preemption costs at most the in-progress epoch.
#
#   YOLO_CKPT=yolo11l.pt YOLO_DATA=$DATA/tiles/data.yaml BATCH=6 NAME=y11l_tiles \
#       sbatch -A ckpt-makelab scripts/model_comparison/run_yolo_train.slurm
#   YOLO_CKPT=yolo11x.pt YOLO_DATA=$DATA/tiles/data.yaml BATCH=3 NAME=y11x_tiles \
#       sbatch -A ckpt-makelab scripts/model_comparison/run_yolo_train.slurm
#   YOLO_CKPT=yolo11l.pt YOLO_DATA=$DATA/pano/data.yaml YOLO_IMGSZ=1280 BATCH=4 NAME=y11l_pano \
#       sbatch -A ckpt-makelab scripts/model_comparison/run_yolo_train.slurm
#
# EXCEPTION — y11x_tiles (2026-08-03 restart). This one config never finished an epoch
# inside a ckpt slice, at batch 3 and again at batch 12, so it is the one arm that runs
# on the lab's dedicated gpu-l40s rather than ckpt. Approved as a deliberate, bounded
# exception to the "never touch gpu-l40s" rule above: ONE node, and only after the
# y26_tiles_l40s fork released its node on 2026-08-01. It is a FRESH start (no last.pt,
# so no inherited args), which is also the only reason WORKERS can be raised here.
#
#   WORKERS=28 YOLO_CKPT=yolo11x.pt YOLO_DATA=$DATA/tiles/data.yaml BATCH=12 NAME=y11x_tiles \
#       sbatch -A gpu-l40s-makelab -p gpu-l40s -q normal \
#              --time=14-00:00:00 --cpus-per-task=32 \
#              scripts/model_comparison/run_yolo_train.slurm
#
# BATCH is pinned per config (not auto -1) so the LR schedule is identical no matter
# which ckpt GPU (l40/l40s/h200) the job lands or resumes on. Sizes are tuned for >=48G.
#
# imgsz MUST match the eval side: tiles are rendered at 1024 (default), so the
# 'yolo' provider defaults --yolo-imgsz 1024; a pano-geometry model wants the
# training imgsz passed to --yolo-imgsz at eval time too.
#
# Only best.pt is needed downstream — rsync it back and score locally (the eval is
# ~110-125 panos/city and runs fine on a laptop GPU; see hyak_yolo_runbook.sh).
#SBATCH -p ckpt-g2
#SBATCH -q ckpt-gpu
#SBATCH --requeue
#SBATCH --job-name=yolo_curb_ramp_train
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/yolo_train_%j.out
#SBATCH --error=logs/yolo_train_%j.err

set -euo pipefail

YOLO_CKPT="${YOLO_CKPT:-yolo11l.pt}"          # base weights (or an absolute path)
YOLO_DATA="${YOLO_DATA:?set YOLO_DATA to a prepared data.yaml}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1024}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:--1}"                          # pin per config (e.g. 6/3) for ckpt; -1 = auto
PATIENCE="${PATIENCE:-20}"                    # early-stop if val mAP plateaus
# Fixed-epoch, resume-safe training (default). On ckpt a job can be preempted and
# requeued any number of times; a fixed EPOCHS + resume-from-last.pt accumulates
# cleanly across requeues, whereas ultralytics `time=<hours>` restarts its clock on
# every requeue (unbounded). So TRAIN_HOURS defaults to 0. Set it >0 only for a
# one-shot run on a non-preemptable partition, where it fits the LR schedule to the
# wall budget and saves cleanly.
TRAIN_HOURS="${TRAIN_HOURS:-0}"
NAME="${NAME:-yolo_run}"
# Dataloader width. Defaults to 8, ultralytics' own default and the value baked into
# every 2026-07 ckpt run — do NOT change that default, or a re-run stops matching the
# committed args.yaml. It is a knob only for a FRESH run: `resume=True` reuses the
# args saved in the checkpoint, so raising it cannot affect an in-flight arm even if
# it is set. Sized against the 2026-08-02 I/O probe (logs/yolo_io_probe_38017333.out),
# which measured read+decode scaling to 301 img/s at 30 workers against the ~38 img/s
# training consumes; note that probe measured decode WITHOUT augmentation, and the
# companion end-to-end CPU probe (38016766) produced no usable samples, so "the
# dataloader is the wall" remains plausible-but-unproven. Requesting more workers than
# --cpus-per-task oversubscribes the allocation, so raise both together.
WORKERS="${WORKERS:-8}"

# Keep everything on scratch (home quota is tiny; runs + base-weight downloads are
# large). /gscratch/scrubbed auto-purges after ~21 idle days.
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
PROJECT="${PROJECT:-/gscratch/scrubbed/$USER/yolo_runs}"
# Base-weight autodownload + Ultralytics settings/datasets dir on scratch, not $HOME.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/gscratch/scrubbed/$USER/ultralytics}"
mkdir -p "$PROJECT" "$YOLO_CONFIG_DIR"

# PYTHON points at a ready interpreter with ultralytics (a lean venv on scratch —
# see the runbook 'env' stage). Falls back to the conda env, which has ultralytics
# only if requirements-vlm.txt was installed into it.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    # shellcheck disable=SC1091
    source activate sidewalkcv2
    PYTHON=python
fi

echo "--- YOLO train (issue #51 supervised baseline) ---"
# On a resume the echoed YOLO_CKPT/BATCH/IMGSZ below are the ENV defaults, NOT what
# trains — resume=True takes every arg from the checkpoint. Reading this header as
# the run's config has already caused one false "arm is mislabelled" alarm, so say so.
if [ -f "${PROJECT}/${NAME}/weights/last.pt" ]; then
    echo "MODE:   RESUME — the values below are env defaults and are IGNORED;"
    echo "        the real config comes from ${NAME}/weights/last.pt (see args.yaml)."
else
    echo "MODE:   FRESH — the values below are the run's actual config."
fi
echo "base:   ${YOLO_CKPT}"
echo "data:   ${YOLO_DATA}"
echo "imgsz:  ${YOLO_IMGSZ}   epochs: ${EPOCHS}   batch: ${BATCH}   patience: ${PATIENCE}   time_hours: ${TRAIN_HOURS}"
echo "workers:${WORKERS}   cpus: ${SLURM_CPUS_PER_TASK:-?}   (workers apply to a FRESH run only; a resume reuses the checkpoint's)"
echo "out:    ${PROJECT}/${NAME}/weights/best.pt"
echo "GPU:    ${SLURM_GPUS_PER_NODE:-1} on ${SLURMD_NODENAME:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "--------------------------------------------------"

# Programmatic call (interpreter-explicit; avoids depending on the `yolo` console
# script being on PATH for whichever venv PYTHON is).
"$PYTHON" - "$YOLO_CKPT" "$YOLO_DATA" "$YOLO_IMGSZ" "$EPOCHS" "$BATCH" "$PATIENCE" \
          "$PROJECT" "$NAME" "$TRAIN_HOURS" "$WORKERS" <<'PY'
import os, sys
from ultralytics import YOLO
ckpt, data, imgsz, epochs, batch, patience, project, name, hours, workers = sys.argv[1:11]
last = os.path.join(project, name, "weights", "last.pt")
if os.path.exists(last):
    # Preempted-and-requeued: resume the in-progress run from its last checkpoint.
    # resume=True reuses every training arg saved in the checkpoint, so the LR
    # schedule continues seamlessly no matter how many times ckpt requeues us.
    print(f"[resume] {last} exists -> resuming", flush=True)
    YOLO(last).train(resume=True)
else:
    # A pinned batch must be a Python int — torch's BatchSampler rejects a float
    # (batch=6.0 -> ValueError). -1 (auto) and 0<f<1 (mem-fraction) stay as-is.
    bf = float(batch)
    batch_arg = int(bf) if bf.is_integer() else bf
    kw = dict(
        data=data, imgsz=int(imgsz), epochs=int(epochs), batch=batch_arg,
        patience=int(patience), project=project, name=name, device=0, exist_ok=True,
        workers=int(workers),
    )
    if float(hours) > 0:
        # Overrides epochs: ultralytics fits the LR schedule to this wall budget.
        # (Only safe on a non-preemptable partition — see TRAIN_HOURS note above.)
        kw["time"] = float(hours)
    YOLO(ckpt).train(**kw)
PY

echo "--- done: ${PROJECT}/${NAME}/weights/best.pt ---"
echo "rsync it back, then: compare.py benchmark/<city> --models rampnet,yolo \\"
echo "    --yolo-model best.pt --yolo-imgsz ${YOLO_IMGSZ} --sweep"
