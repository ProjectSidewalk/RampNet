#!/bin/bash
# Train one YOLO curb-ramp detector on the RampNet dataset, on TILLICUM.
#
# Port of run_yolo_train.slurm (klone). Kept as a SEPARATE file rather than
# branching inside the klone script, because nearly every scheduler directive
# differs and the klone version is the preserved record of the #51 runs -- it
# should not churn. See docs/tillicum.md for the directive-by-directive diff.
#
# WHAT CHANGED FROM THE KLONE VERSION, AND WHY
#   -p ckpt-g2 / -q ckpt-gpu  ->  --qos=normal   Tillicum has NO partitions; QoS only.
#   --requeue                 ->  dropped        Nothing preempts on Tillicum.
#   --time=72:00:00           ->  24:00:00       normal QoS caps at 24 h. 7-day jobs
#                                                need `long` QoS (pre-approval).
#   --gpus-per-node=1         ->  --gres=gpu:N   Tillicum's documented spelling.
#   --cpus-per-task=12        ->  8 x N GPUs     HARD 8:1 ratio; 12-on-1 is REJECTED.
#   (no mail directive)       ->  --mail-type=NONE   Be explicit so the site default
#                                                cannot decide for us (on klone an
#                                                inherited END,FAIL,TIME_LIMIT produced
#                                                ~1 email per minute under preemption).
#
#   YOLO_DATA=$DATA/pano/data.yaml YOLO_IMGSZ=1280 BATCH=4 NAME=y26_pano_till \
#       sbatch scripts/model_comparison/run_yolo_train_tillicum.slurm
#
# COST. Tillicum BILLS: $0.90/GPU-hour where GPU-hour = elapsed x N GPUs. A 24 h
# 2-GPU job is 48 GPU-hours (~$43). Do not leave allocations idle. Check `hyakusage`.
#SBATCH --qos=normal
#SBATCH --job-name=yolo_curb_ramp_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --output=logs/yolo_train_till_%j.out
#SBATCH --error=logs/yolo_train_till_%j.err
# gres/cpus/mem are NOT hardcoded -- see the GPUS block below, which must be kept in
# step with them. Override at submit time with: sbatch --gres=gpu:2 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail

# ---------------------------------------------------------------------------------
# The 2-GPU dataloader trick, and why DEVICE is separate from the allocation.
#
# Tillicum fixes CPUs at 8 per GPU. Our tiles arm is I/O-bound (the same epoch took
# 4.1-9.5 h on klone depending only on which node it landed on), so more dataloader
# workers is the obvious lever -- but on Tillicum more CPUs means more GPUs.
#
# The naive move -- allocate 2 GPUs and let ultralytics use both -- is NOT a clean
# experiment: 2 GPUs puts ultralytics into DDP, which changes the effective batch and
# the LR dynamics, so the result is no longer comparable to the klone single-GPU runs.
#
# So: allocate 2 GPUs for their 16 CPUs, but pin DEVICE=0 so training math is
# IDENTICAL to a 1-GPU run. The second GPU sits idle and we pay for it. That is the
# point -- it isolates dataloader throughput as the only changed variable, and it
# answers whether the 2x bill buys more than a 2x speedup. Set DEVICE=0,1 only if you
# deliberately want a DDP run, and label it as a different arm.
# ---------------------------------------------------------------------------------
DEVICE="${DEVICE:-0}"

YOLO_CKPT="${YOLO_CKPT:-yolo11l.pt}"
YOLO_DATA="${YOLO_DATA:?set YOLO_DATA to a prepared data.yaml}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1024}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:--1}"                          # pin per config to match the klone runs
PATIENCE="${PATIENCE:-20}"
# On klone this defaulted to 0 because ultralytics `time=` restarts its clock on every
# preemption. Tillicum never preempts, so `time=` is safe here -- but leave it 0 for
# any run meant to be comparable to the #51 klone results, which are epoch-based.
TRAIN_HOURS="${TRAIN_HOURS:-0}"
NAME="${NAME:-yolo_run_till}"

# Tillicum storage. /gpfs/projects/makelab is 1 TB, backed up daily, and PURGED AT END
# OF PROJECT -- it is NOT archival. Only best.pt matters downstream: rsync it back to
# klone /gscratch/makelab, which stays the system of record.
GPFS="${GPFS:-/gpfs/projects/makelab/$USER}"
export HF_HOME="${HF_HOME:-$GPFS/hf}"
PROJECT="${PROJECT:-$GPFS/yolo_runs}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$GPFS/ultralytics}"
mkdir -p "$PROJECT" "$YOLO_CONFIG_DIR" "$HF_HOME"

# PYTHON points at an interpreter with ultralytics. On Tillicum the DOCUMENTED path is
# an Apptainer container (NGC PyTorch + `pip install ultralytics`), not a conda module
# -- our environment.yml pins CUDA 11.8 against what is now a Rocky 9 / H200 host and
# should not be assumed to transfer. Set PYTHON, or set APPTAINER_IMG to run in one.
PYTHON="${PYTHON:-python}"
APPTAINER_IMG="${APPTAINER_IMG:-}"

echo "--- YOLO train on TILLICUM (issue #51 / #70) ---"
echo "base:   ${YOLO_CKPT}"
echo "data:   ${YOLO_DATA}"
echo "imgsz:  ${YOLO_IMGSZ}   epochs: ${EPOCHS}   batch: ${BATCH}   patience: ${PATIENCE}"
echo "alloc:  ${SLURM_GPUS_ON_NODE:-?} GPU(s), ${SLURM_CPUS_PER_TASK:-?} CPUs on ${SLURMD_NODENAME:-?}"
echo "device: ${DEVICE}   (allocation and device differ on purpose -- see header)"
echo "out:    ${PROJECT}/${NAME}/weights/best.pt"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "------------------------------------------------"

run_train() {
    "$@" - "$YOLO_CKPT" "$YOLO_DATA" "$YOLO_IMGSZ" "$EPOCHS" "$BATCH" "$PATIENCE" \
            "$PROJECT" "$NAME" "$TRAIN_HOURS" "$DEVICE" <<'PY'
import os, sys
from ultralytics import YOLO
ckpt, data, imgsz, epochs, batch, patience, project, name, hours, device = sys.argv[1:11]
last = os.path.join(project, name, "weights", "last.pt")
# Still needed on Tillicum, but for a different reason than on klone: not preemption,
# but the 24 h normal-QoS ceiling. A 60-epoch tiles schedule does not fit in one job,
# so chain jobs and resume. Unlike klone the boundary is predictable, not random.
if os.path.exists(last):
    print(f"[resume] {last} exists -> resuming", flush=True)
    YOLO(last).train(resume=True)
else:
    bf = float(batch)
    batch_arg = int(bf) if bf.is_integer() else bf
    dev = [int(d) for d in device.split(",")] if "," in device else int(device)
    kw = dict(
        data=data, imgsz=int(imgsz), epochs=int(epochs), batch=batch_arg,
        patience=int(patience), project=project, name=name, device=dev, exist_ok=True,
    )
    if float(hours) > 0:
        kw["time"] = float(hours)
    YOLO(ckpt).train(**kw)
PY
}

# run_train takes the interpreter as "$@", so the container case is just a longer
# command word -- the heredoc still lands on its stdin and every shell variable is
# expanded HERE, outside the container, before apptainer is ever invoked.
# --nv exposes the GPUs; bind /gpfs so the container sees data, runs and cache.
if [ -n "$APPTAINER_IMG" ]; then
    run_train apptainer exec --nv --bind /gpfs:/gpfs "$APPTAINER_IMG" python
else
    run_train "$PYTHON"
fi
