#!/bin/bash
# Control arm for issue #263 (RampNet 2.0 phase 0, Track 3): fine-tune the RELEASED RampNet
# weights on the GSV training set only, 2,000 optimizer steps at global batch 16, one job
# per SEED. Every later #263 arm (Lyon mixes, masked loss) is read against these seeds, so
# this arm changes nothing but the step budget and the warm start: no augmentation, no
# --loss-mask, the recipe's constant LR 1e-5 and horizontal flip.
#
# DERIVED FROM run_finetune_aug82.slurm, whose `control` arm is the same recipe. Every
# preflight there is kept verbatim (node-local env tar, sha-checked released weights,
# shared-library warm-up, world-size/batch check); read that header and
# run_train_seed.slurm's before changing anything here. Differences: no ARM, no --aug, its
# own RUNDIR under /gscratch/scrubbed/$USER/control263, the re-staged dataset with a
# file-count check, an env tarball it REBUILDS when missing, and a COPY STEP at the end.
# INIT stays the #82 screen's converted released weights, so this arm starts from the same
# file the #82 control did (and the sha-checked conversion re-creates it if purged).
#
# THE ENV TARBALL IS REBUILT WHEN MISSING. The #82 tar was purged from scrubbed by
# 2026-10-20. The aug82 header's GPFS note is why this rebuilds it rather than running
# from the GPFS env: four ranks importing torch from /gscratch/makelab sat 25-30 min in
# uninterruptible I/O, while one sequential `tar cf` read of the same env (job 41123104)
# did not stall. So a job that finds no tar makes one -- into a job-unique temp file, then
# `mv -n`, so seeds starting together each build their own and the first to finish wins
# (a few minutes of duplicated reading, once) -- and every later start and requeue unpacks
# it to node-local disk. If the env is not named sidewalkcv2 (the unpack step expects that
# name) or the tar fails, the job falls back to the GPFS env plus the library warm-up. A job
# preempted mid-tar removes its partial temp file on the way out. The tar is a snapshot:
# AFTER CHANGING THE CONDA ENV, DELETE $ENV_TAR, or jobs keep unpacking the old one.
#
# THE DATASET is /gscratch/scrubbed/$USER/rampnet_dataset_v2, written by
# run_restage_263.slurm (the older rampnet_dataset copy was partly purged: 114,667 of
# ~300,126 train files, val empty). The job refuses to start unless train/ and val/ hold
# the full 150,063 and 42,875 jpg+json PAIRS, because a purge that thins the data silently
# is exactly what happened to the last copy, and download_dataset.py writes each JSON before
# its JPEG and logs-and-continues on a failed pano, so a jpg count alone can pass a short
# split. EXPECT_TRAIN= EXPECT_VAL= (empty) skip the check.
#
# WHY A COPY STEP. /gscratch/scrubbed is purged (the Stage 2 dataset copy there was partly
# gone by 2026-10, #263), and this arm's weights are the baseline every later arm is scored
# against. So the job copies the final weights, a sha256 of them and the TensorBoard
# events to /gscratch/makelab/jonf/control263/s<SEED> (literal `jonf`: on klone $USER is
# `jfroehli`, and /gscratch/makelab holds both). The job log is not copied: logs/ is in the
# submit checkout, which already lives on lab storage. Only a handful of files:
# /gscratch/makelab sits near its file-count cap. The resume file (latest_checkpoint.pth,
# ~1 GB with optimizer state) is NOT copied.
#
# DATA. GSV only means the Stage 2 dataset (train/ + val/) exactly as staged, with no
# sparse-labeled panos mixed in. train.py enforces that: a pano JSON with a `supervision`
# block is an error when --loss-mask is off (docs/sparse_supervision.md).
#
# USAGE. Submit from the repo root of a checkout of this branch; logs/ must ALREADY exist.
#
#   cd /gscratch/scrubbed/$USER/RampNet_263 && mkdir -p logs
#   for s in 1 2 3; do
#       RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 SEED=$s \
#           sbatch stage_two/run_finetune_control.slurm
#   done
#
# THEN CHECK, ABOUT AN HOUR LATER, that the jobs actually started and are training:
#   sacct -D -u $USER -S $(date -d '1 day ago' +%Y-%m-%d) -o JobID%16,JobName%20,State%14,Elapsed
#   head -40 logs/control263_<jobid>.err      # the FIRST lines, not the tail
#
# SCORING. On all 14 benchmark splits, as #263 specifies, from
# /gscratch/makelab/jonf/control263/s<SEED>/final_step_2000.pth.
#
# COST. ckpt-all is FREE and preemptible: ~3 h on 4 GPUs, ~12 GPU-h per seed, ~36 GPU-h for
# three (the #82 launcher's estimate for the same shape of job). After the jobs finish run
# scripts/analysis/slurm_usage.py per docs/compute_cost.md, with `sacct -D`.
#SBATCH -p ckpt-all
#SBATCH --job-name=rampnet_ctrl263
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/control263_%j.out
#SBATCH --error=logs/control263_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

set -euo pipefail

# NO APOSTROPHES inside any ${VAR:?...} message (see run_train_seed.slurm).
SEED="${SEED:?set SEED, e.g. SEED=1}"
STEPS="${STEPS:-2000}"                 # optimizer steps (global batch 16 each)
GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
LR="${LR:-1e-5}"                       # the recipe LR, constant, no scheduler

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if ! grep -q -- "--max-steps" "$REPO/stage_two/train.py" 2>/dev/null \
   || ! grep -q -- "--grad-accum" "$REPO/stage_two/train.py"; then
    echo "FATAL: $REPO/stage_two/train.py has no --max-steps / --grad-accum (stale checkout?)" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}"
EXPECT_TRAIN="${EXPECT_TRAIN-150063}"  # jpg+json pairs per split (run_restage_263.slurm);
EXPECT_VAL="${EXPECT_VAL-42875}"       # set either to empty to skip its check
for split in train val; do
    if [ ! -d "$DATA_ROOT/$split" ]; then
        echo "FATAL: no $DATA_ROOT/$split -- run stage_two/run_restage_263.slurm first (#263)" >&2
        exit 1
    fi
    if [ "$split" = train ]; then want="$EXPECT_TRAIN"; else want="$EXPECT_VAL"; fi
    if [ -n "$want" ]; then
        # pairs: stems that have BOTH a .jpg and a .json (train.py skips a lone jpg)
        have=$(comm -12 \
            <(find "$DATA_ROOT/$split" -maxdepth 1 -name "*.jpg" -printf "%f\n" | sed "s/[.]jpg$//" | sort) \
            <(find "$DATA_ROOT/$split" -maxdepth 1 -name "*.json" -printf "%f\n" | sed "s/[.]json$//" | sort) \
            | wc -l)
        if [ "$have" -ne "$want" ]; then
            echo "FATAL: $DATA_ROOT/$split has $have jpg+json pairs, expected $want (purged or partial?)" >&2
            exit 1
        fi
    fi
done
RUNDIR="${RUNDIR:-/gscratch/scrubbed/$USER/control263/s${SEED}}"
COPY_DEST="${COPY_DEST:-/gscratch/makelab/jonf/control263/s${SEED}}"
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix, e.g. RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2}"
TORCHRUN="$RAMPNET_ENV/bin/torchrun"
PY="$RAMPNET_ENV/bin/python"
[ -x "$TORCHRUN" ] || { echo "FATAL: RAMPNET_ENV=$RAMPNET_ENV has no bin/torchrun" >&2; exit 1; }

# --- env tarball rebuild (see header) ---
ENV_TAR="${ENV_TAR:-/gscratch/scrubbed/$USER/control263/sidewalkcv2_env.tar}"
if [ ! -f "$ENV_TAR" ] && [ "$(basename "$RAMPNET_ENV")" = sidewalkcv2 ]; then
    mkdir -p "$(dirname "$ENV_TAR")"
    tmp_tar="$ENV_TAR.tmp.${SLURM_JOB_ID:-$$}"
    # a preemption mid-tar must not leave a multi-GB partial file behind
    trap 'rm -f "$tmp_tar"' EXIT
    trap 'exit 143' TERM INT
    t_tar=$(date +%s)
    if tar cf "$tmp_tar" -C "$(dirname "$RAMPNET_ENV")" sidewalkcv2; then
        mv -n "$tmp_tar" "$ENV_TAR" || true
        if [ -e "$tmp_tar" ]; then
            echo "Env tarball $ENV_TAR was finished first by another job; using that one"
        else
            echo "Built env tarball $ENV_TAR from $RAMPNET_ENV in $(( $(date +%s) - t_tar )) s"
        fi
    else
        echo "WARNING: could not tar $RAMPNET_ENV; running from the GPFS env" >&2
    fi
    rm -f "$tmp_tar"
    trap - EXIT TERM INT
fi
# --- end env tarball rebuild ---

# Run the env from node-local NVMe when a tarball of it exists. On 2026-10-02, ranks reading
# the env from /gscratch/makelab sat 25-30 min in uninterruptible GPFS I/O inside
# `import torch` before train.py's first line, on four different nodes (41106892/93/94/95/
# 903) -- and every requeue would pay that again. One sequential read of a tarball does not
# hit the per-file stall. The tar is made once by `tar cf $ENV_TAR -C
# /gscratch/makelab/jonf/envs sidewalkcv2` (job 41123104); without it the job falls back to
# the GPFS env.
ENV_TAR="${ENV_TAR:-/gscratch/scrubbed/$USER/control263/sidewalkcv2_env.tar}"
if [ -f "$ENV_TAR" ]; then
    LOCAL_ENV_DIR="/scr/$USER/ctrl263env_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "$LOCAL_ENV_DIR"
    trap 'rm -rf "$LOCAL_ENV_DIR"' EXIT
    t_env=$(date +%s)
    tar xf "$ENV_TAR" -C "$LOCAL_ENV_DIR"
    RAMPNET_ENV="$LOCAL_ENV_DIR/sidewalkcv2"
    echo "Env unpacked to node-local $RAMPNET_ENV in $(( $(date +%s) - t_env )) s"
fi
PY="$RAMPNET_ENV/bin/python"

# The released weights, as a plain state_dict .pth train.py --init-weights can load. The
# Hub file is model.safetensors with a "model." key prefix; converted once, atomically, so
# three seeds starting together cannot read a half-written file.
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
echo "--- #263 control arm: released weights, GSV only ---"
echo "Job ID:      ${SLURM_JOBID}   restarts: ${SLURM_RESTART_COUNT:-0}"
echo "Seed:        ${SEED}   (no --aug, no --loss-mask)"
echo "Repo:        ${REPO}  commit ${commit}  dirty files: ${dirty}"
echo "Run dir:     ${RUNDIR}"
echo "Copy to:     ${COPY_DEST}"
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
if [ -z "${LOCAL_ENV_DIR:-}" ]; then
    t_warm=$(date +%s)
    cat "$RAMPNET_ENV"/lib/libtorch*.so "$RAMPNET_ENV"/lib/libc10*.so "$RAMPNET_ENV"/lib/libmagma.so \
        "$RAMPNET_ENV"/lib/libnccl.so* "$RAMPNET_ENV"/lib/libcudnn*.so* "$RAMPNET_ENV"/lib/libcublas*.so* \
        "$RAMPNET_ENV"/lib/libmkl*.so* > /dev/null 2>&1 || true
    echo "Warmed env shared libraries in $(( $(date +%s) - t_warm )) s"
fi

cd "$RUNDIR"
# python -m torch.distributed.run, not bin/torchrun: torchrun's shebang names the GPFS
# prefix, which would defeat the node-local copy.
srun --export=ALL \
    "$PY" -m torch.distributed.run --nnodes $SLURM_NNODES \
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
             --max-steps "$STEPS"

echo "--- Slurm job finished ---"
ls -la "$RUNDIR/checkpoints"

# --- copy step: the durable half, off scrubbed (see header) ---
# Fails the job when the final weights are missing, so a run that never reached STEPS
# cannot pass for a finished one.
FINAL="$RUNDIR/checkpoints/final_step_${STEPS}.pth"
if [ ! -f "$FINAL" ]; then
    echo "FATAL: no $FINAL to copy" >&2
    exit 1
fi
mkdir -p "$COPY_DEST"
cp -p "$FINAL" "$COPY_DEST/"
(cd "$COPY_DEST" && sha256sum "final_step_${STEPS}.pth" > "final_step_${STEPS}.pth.sha256")
if [ -d "$RUNDIR/runs" ]; then
    cp -rp "$RUNDIR/runs" "$COPY_DEST/"
fi
echo "Copied to $COPY_DEST: $(cat "$COPY_DEST/final_step_${STEPS}.pth.sha256")"
# --- end copy step ---
