#!/bin/bash
# Paired fine-tune screen for issue #82 (docs/aug_transfer_82.md, Step 3), on klone.
#
# One job = one (ARM, SEED): start from the RELEASED RampNet weights, train a fixed number
# of optimizer steps at the recipe's constant LR 1e-5 and global batch 16, with or without
# the #82 pixel-statistics augmentation, and write checkpoints/final_step_<STEPS>.pth.
# Arms with the same SEED share data order and flip draws (train.py --aug docs), so the
# control at a seed is the paired baseline for every augmented arm at that seed.
#
# WHY A NEW FILE. run_train.slurm is the preserved record of the published run and is not
# edited; run_train_seed.slurm is the record of the #51/#135 seed replicates. This one
# needs a step budget, an arm, gradient accumulation and a warm start, none of which
# those runs had. Its failure-mode guards are copied from run_train_seed.slurm, whose
# header documents every klone failure already hit (stale REPO, unset RAMPNET_ENV,
# apostrophes inside ${VAR:?...}): read it before changing anything here.
#
# USAGE. Submit from the repo root of a checkout of this branch; logs/ must ALREADY exist
# (Slurm opens --output relative to the submit directory before this script runs).
#
#   cd /gscratch/scrubbed/$USER/RampNet_aug82 && mkdir -p logs
#   for s in 1 2; do for a in control res photo both; do
#       RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 ARM=$a SEED=$s \
#           sbatch stage_two/run_finetune_aug82.slurm
#   done; done
#
# THEN CHECK, ABOUT AN HOUR LATER, that the jobs actually started and are training:
#   sacct -D -u $USER -S $(date -d '1 day ago' +%Y-%m-%d) -o JobID%16,JobName%20,State%14,Elapsed
#   head -40 logs/aug82_<jobid>.err      # the FIRST lines, not the tail (see run_train_seed.slurm)
#
# GLOBAL BATCH. train.py runs batch 1 per rank, so the global batch is
# world size x --grad-accum. This launcher asks for one node with 4 GPUs (a 4-node job
# waits far longer on ckpt-all) and sets --grad-accum 16 / world size = 4, which keeps the
# recipe's global batch of 16 with 4x fewer GPUs. Not bit-identical to 16 GPUs x 1:
# the AMP loss scale updates once per optimizer step either way, but the per-rank data
# shard differs. The GPU count is printed in the log.
#
# RESUME. Preemption is handled by --requeue plus train.py's latest_checkpoint.pth, written
# every CKPT_INTERVAL micro-batches (a multiple of the accumulation). On resume the data
# order is identical (DistributedSampler is a pure function of seed and epoch, and
# ResumeSkipSampler drops the batches already done); the augmentation draws are identical
# (keyed on seed, epoch and sample index, never on a running counter); the horizontal flip
# draws are NOT identical to an uninterrupted run, because the DataLoader re-seeds its
# workers on every start and the flip uses each worker's `random` -- true of the published
# recipe as well. Which incarnations a job had: sacct -D.
#
# COST. ckpt-all is FREE and preemptible. 2,000 optimizer steps x 4 micro-batches at the
# recipe's ~1.34 s per micro-batch is ~3 h on 4 GPUs, ~12 GPU-h per job, ~96 GPU-h for the
# eight-job screen. After the jobs finish run scripts/analysis/slurm_usage.py per
# docs/compute_cost.md, into THIS checkout's analysis_out/compute_log.jsonl.
#SBATCH -p ckpt-all
#SBATCH --job-name=rampnet_aug82
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/aug82_%j.out
#SBATCH --error=logs/aug82_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

set -euo pipefail

# NO APOSTROPHES inside any ${VAR:?...} message (see run_train_seed.slurm).
SEED="${SEED:?set SEED, e.g. SEED=1}"
ARM="${ARM:?set ARM to one of control res photo both}"
STEPS="${STEPS:-2000}"                 # optimizer steps (global batch 16 each)
GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
LR="${LR:-1e-5}"                       # the recipe LR, constant, no scheduler

# The augmentation each arm adds, as train.py --aug OP=P:LO:HI. Levels are placed on the
# rig statistics measured in Step 1 (analysis_out/aug_transfer_82/stats.json; the
# derivation is in docs/aug_transfer_82.md, "Training arms"). Every arm keeps the
# recipe's horizontal flip.
AUG_RES=(--aug downscale=0.5:0.5:1.0 --aug blur=0.3:0.0:1.15 --aug jpeg=0.5:50:95)
AUG_PHOTO=(--aug brightness=0.5:0.6:1.2 --aug contrast=0.5:0.85:1.25
           --aug saturation=0.5:0.7:1.6 --aug gamma=0.5:0.75:1.5
           --aug wb=0.5:-0.5:0.3 --aug hue=0.3:-6:6)
case "$ARM" in
    control) AUG=() ;;
    res)     AUG=("${AUG_RES[@]}") ;;
    photo)   AUG=("${AUG_PHOTO[@]}") ;;
    both)    AUG=("${AUG_RES[@]}" "${AUG_PHOTO[@]}") ;;
    *) echo "FATAL: ARM=$ARM is not one of control res photo both" >&2; exit 1 ;;
esac

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! grep -q -- "--max-steps" "$REPO/stage_two/train.py" 2>/dev/null \
   || ! grep -q -- "--aug" "$REPO/stage_two/train.py"; then
    echo "FATAL: $REPO/stage_two/train.py has no --max-steps / --aug (stale checkout?)" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset}"
RUNDIR="${RUNDIR:-/gscratch/scrubbed/$USER/aug82/${ARM}_s${SEED}}"
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix, e.g. RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2}"
TORCHRUN="$RAMPNET_ENV/bin/torchrun"
PY="$RAMPNET_ENV/bin/python"
[ -x "$TORCHRUN" ] || { echo "FATAL: RAMPNET_ENV=$RAMPNET_ENV has no bin/torchrun" >&2; exit 1; }

# The released weights, as a plain state_dict .pth train.py --init-weights can load. The
# Hub file is model.safetensors with a "model." key prefix; converted once, atomically, so
# eight jobs starting together cannot read a half-written file.
# The Hub revision is 606a119; its model.safetensors blob is sha256 f2119e3b..., the same
# blob the makelab2 scorer loads. The copy under /gscratch/scrubbed/$USER/hf was purged
# (dangling symlink; jobs 41103263-70 died on it in seconds), so this reads the home cache
# (read only -- nothing is written to home). The sha is checked before conversion.
HF_SAFETENSORS="${HF_SAFETENSORS:-$HOME/.cache/huggingface/hub/models--projectsidewalk--rampnet-model/snapshots/606a11956743f7eb328d9207769034752f6191f4/model.safetensors}"
HF_SHA256=f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5
INIT="${INIT:-/gscratch/scrubbed/$USER/aug82/released_rampnet_state_dict.pth}"
mkdir -p "$(dirname "$INIT")" "$RUNDIR/checkpoints" "$REPO/logs"
if [ ! -f "$INIT" ]; then
    echo "$HF_SHA256  $HF_SAFETENSORS" | sha256sum -c --quiet         || { echo "FATAL: $HF_SAFETENSORS is not the released blob $HF_SHA256" >&2; exit 1; }
    tmp="$INIT.tmp.$SLURM_JOB_ID"
    "$PY" -c "import sys, torch, safetensors.torch as st
sd = st.load_file(sys.argv[1])
sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
torch.save(sd, sys.argv[2])" "$HF_SAFETENSORS" "$tmp"
    mv -n "$tmp" "$INIT" || true
    rm -f "$tmp"
fi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=4
# Never write .pyc files into the shared conda env. /gscratch/makelab is at its file-count
# cap, and on 2026-10-02 every rank of 41106892/93/94 sat 14-25 min in uninterruptible GPFS
# I/O inside `import torch`, holding a sympy/filelock __pycache__ file, before any line of
# train.py ran. The scoring job (one process) had imported the same env fine an hour earlier.
export PYTHONDONTWRITEBYTECODE=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))
if [ $((GLOBAL_BATCH % WORLD_SIZE)) -ne 0 ]; then
    echo "FATAL: world size $WORLD_SIZE does not divide global batch $GLOBAL_BATCH" >&2
    exit 1
fi
ACCUM=$((GLOBAL_BATCH / WORLD_SIZE))
# latest_checkpoint.pth every 100 optimizer steps (~9 min at 4 GPUs): the granularity a
# preemption rewinds to.
CKPT_INTERVAL="${CKPT_INTERVAL:-$((100 * ACCUM))}"

if commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null); then
    dirty=$(git -C "$REPO" status --porcelain | wc -l)
else
    commit=unknown; dirty=unknown
fi
echo "--- #82 fine-tune screen ---"
echo "Job ID:      ${SLURM_JOBID}   restarts: ${SLURM_RESTART_COUNT:-0}"
echo "Arm / seed:  ${ARM} / ${SEED}"
echo "Aug flags:   ${AUG[*]:-none}"
echo "Repo:        ${REPO}  commit ${commit}  dirty files: ${dirty}"
echo "Run dir:     ${RUNDIR}"
echo "Data root:   ${DATA_ROOT}"
echo "Init:        ${INIT}  sha256 $(sha256sum "$INIT" | cut -c1-16)"
echo "GPUs:        ${SLURM_NNODES} x ${NPROC_PER_NODE} (world ${WORLD_SIZE}) x accum ${ACCUM} = global batch ${GLOBAL_BATCH}"
echo "Steps:       ${STEPS} optimizer steps, LR ${LR} constant, ckpt every ${CKPT_INTERVAL} micro-batches"
echo "Node list:   ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
echo "----------------------------"

# Read the env's big shared libraries once, sequentially, before the ranks start. On
# 2026-10-02 GPFS reads of /gscratch/makelab ran at ~20-30 MB/s on the compute nodes, and four
# ranks page-faulting libtorch_cuda.so (1.4 GB) and friends through mmap sat 25+ min in
# `import torch` (41106892 on g3115, 41106893 on g3045). One `cat` of libtorch_cuda.so took
# 47 s, and 41106893 got past its imports within minutes of that cat warming the page cache.
t_warm=$(date +%s)
cat "$RAMPNET_ENV"/lib/libtorch*.so "$RAMPNET_ENV"/lib/libc10*.so "$RAMPNET_ENV"/lib/libmagma.so \
    "$RAMPNET_ENV"/lib/libnccl.so* "$RAMPNET_ENV"/lib/libcudnn*.so* "$RAMPNET_ENV"/lib/libcublas*.so* \
    "$RAMPNET_ENV"/lib/libmkl*.so* > /dev/null 2>&1 || true
echo "Warmed env shared libraries in $(( $(date +%s) - t_warm )) s"

cd "$RUNDIR"
srun --export=ALL \
    "$TORCHRUN" --nnodes $SLURM_NNODES \
             --nproc_per_node $NPROC_PER_NODE \
             --rdzv_id $SLURM_JOB_ID \
             --rdzv_backend c10d \
             --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
             "$REPO/stage_two/train.py" \
             --seed "$SEED" \
             --preset scratch --lr "$LR" \
             --init-weights "$INIT" \
             --data-root "$DATA_ROOT" \
             --checkpoint-dir "$RUNDIR/checkpoints" \
             --checkpoint-interval-steps "$CKPT_INTERVAL" \
             --grad-accum "$ACCUM" \
             --max-steps "$STEPS" \
             ${AUG[@]+"${AUG[@]}"}

echo "--- Slurm job finished ---"
ls -la "$RUNDIR/checkpoints"
