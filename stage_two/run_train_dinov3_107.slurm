#!/bin/bash
# Arm A1 of issue #107: the Stage 2 recipe with the backbone swapped to DINOv3-ConvNeXt-base.
#
# WHAT IS AND IS NOT CHANGED
# --------------------------
# This is stage_two/run_train_epoch_curve.slurm (Run A of #84) with ONE training change:
# --backbone convnext_base.dinov3_lvd1689m instead of the default
# convnextv2_base.fcmae_ft_in22k_in1k_384. Same 4 nodes x 4 GPUs (world size 16 = global
# batch 16), same constant lr 1e-5 ('scratch' preset, no scheduler), same seed 42, same data
# and splits. Run A's per-epoch checkpoints are therefore the control (A0) at every epoch.
#
# Two things that differ from Run A and are NOT part of the comparison:
#   * 5 epochs by default instead of 8. Run A's auto-label val loss bottoms at epoch 5 and
#     its manual_gold F1 is flat from epoch 2 (#84), so 5 covers the turn. Compare epoch N
#     here with epoch N of Run A.
#   * The dataset is a later staging of the same Hub revision (docs/restage_263.md): the
#     parquet's own jpg bytes rather than a PIL re-encode. #263's control arm trained on the
#     same kind of copy.
# The LR is the recipe's, chosen for the default backbone. If A1 loses, rule out the LR
# before concluding anything about pretraining (#107, "Implementation notes").
#
# The DINOv3 weights carry the DINOv3 License, and so does every checkpoint this writes.
# timm downloads them from timm/convnext_base.dinov3_lvd1689m on the Hub (not gated, checked
# 2026-11-19; Meta's own facebook/dinov3-* repos are gated).
#
# PATHS (all from the environment; nothing is hardcoded)
#   RAMPNET_REPO      repo checkout                     (default: the submitting checkout)
#   RAMPNET_ENV       conda prefix                      (required)
#   RAMPNET_DATA      dataset root with train/ and val/ (default: /gscratch/scrubbed/$USER/rampnet_dataset_v2)
#   RAMPNET_EPOCHS                                      (default: 5)
#   RAMPNET_BACKBONE  timm id                           (default: convnext_base.dinov3_lvd1689m)
#   HF_HOME           Hub cache for the backbone        (default: /gscratch/scrubbed/$USER/hf)
#   HF_TOKEN_PATH     token file, used if present       (default: $HOME/.cache/huggingface/token)
#   MAX_STEPS         smoke test only: stop after this many optimizer steps of global batch
#                     16, whatever the world size (uses --grad-accum). Unset for the real run.
#
# Submit with --chdir=<run dir> on a DURABLE volume, as for Run A: train.py writes
# latest_checkpoint.pth (the resume file), best_model.pth, runs/ and peek_training/ relative
# to the working directory, and the #SBATCH log paths below are relative to it too.
#
#   export RUN_DIR=/gscratch/makelab/jonf/rampnet_dinov3_107
#   mkdir -p "$RUN_DIR"/{checkpoints,logs}
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 \
#       sbatch --chdir="$RUN_DIR" stage_two/run_train_dinov3_107.slurm
#
# Smoke test first (one GPU, 20 optimizer steps, its own run dir on scratch):
#   mkdir -p /gscratch/scrubbed/$USER/dinov3_107_smoke/logs
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 MAX_STEPS=20 \
#       sbatch --chdir=/gscratch/scrubbed/$USER/dinov3_107_smoke --nodes=1 --gpus-per-node=1 \
#              --time=2:00:00 --job-name=dinov3_107_smoke stage_two/run_train_dinov3_107.slurm
#
#SBATCH --job-name=rampnet_dinov3_107
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --output=logs/dinov3_107_%j.out
#SBATCH --error=logs/dinov3_107_%j.err
#SBATCH --requeue
#SBATCH --constraint='l40s|l40|a40|a100'

# NOTE: deliberately no `set -u` (see run_train_epoch_curve.slurm: klone's lmod init
# dereferences an unbound variable). Every variable below has an explicit default.
set -eo pipefail

REPO="${RAMPNET_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix, e.g. RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2}"
DATA_ROOT="${RAMPNET_DATA:-/gscratch/scrubbed/$USER/rampnet_dataset_v2}"
EPOCHS="${RAMPNET_EPOCHS:-5}"
BACKBONE="${RAMPNET_BACKBONE:-convnext_base.dinov3_lvd1689m}"
PYBIN="$ENV_PREFIX/bin/python"
TORCHRUN="$ENV_PREFIX/bin/torchrun"
[ -x "$TORCHRUN" ] || { echo "FATAL: RAMPNET_ENV=$ENV_PREFIX has no bin/torchrun" >&2; exit 1; }
grep -q -- "--backbone" "$REPO/stage_two/train.py" 2>/dev/null \
    || { echo "FATAL: $REPO/stage_two/train.py has no --backbone (stale checkout?)" >&2; exit 1; }

# The staged copy lives on storage that purges. Refuse a partial one instead of training
# on it: stems that have BOTH a .jpg and a .json, as train.py reads them.
EXPECT_TRAIN="${EXPECT_TRAIN-150063}"
EXPECT_VAL="${EXPECT_VAL-42875}"
for split in train val; do
    if [ "$split" = train ]; then want="$EXPECT_TRAIN"; else want="$EXPECT_VAL"; fi
    [ -d "$DATA_ROOT/$split" ] || { echo "FATAL: no $DATA_ROOT/$split (docs/restage_263.md)" >&2; exit 1; }
    [ -n "$want" ] || continue
    have=$(comm -12 \
        <(find "$DATA_ROOT/$split" -maxdepth 1 -name "*.jpg" -printf "%f\n" | sed "s/[.]jpg$//" | sort) \
        <(find "$DATA_ROOT/$split" -maxdepth 1 -name "*.json" -printf "%f\n" | sed "s/[.]json$//" | sort) \
        | wc -l)
    [ "$have" -eq "$want" ] || { echo "FATAL: $DATA_ROOT/$split has $have jpg+json pairs, expected $want (purged or partial?)" >&2; exit 1; }
done

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [ -f "$HF_TOKEN_PATH" ]; then export HF_TOKEN_PATH; else unset HF_TOKEN_PATH; fi

# Fetch the backbone once, here, then go offline: sixteen ranks downloading the same file
# into one cache at the same moment is a race, and a rank that cannot reach the Hub after a
# requeue should fail on a missing cache entry, not hang.
"$PYBIN" -c "
import sys, timm
m = timm.create_model(sys.argv[1], pretrained=True)
c = m.pretrained_cfg
print('backbone', sys.argv[1], 'from', c.get('hf_hub_id'), 'licence', c.get('license'),
      'params', sum(p.numel() for p in m.parameters()), 'timm', timm.__version__)
" "$BACKBONE"
export HF_HUB_OFFLINE=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

EXTRA=()
if [ -n "${MAX_STEPS:-}" ]; then
    [ $((16 % WORLD_SIZE)) -eq 0 ] || { echo "FATAL: world size $WORLD_SIZE does not divide global batch 16" >&2; exit 1; }
    EXTRA=(--max-steps "$MAX_STEPS" --grad-accum $((16 / WORLD_SIZE)) --checkpoint-interval-steps $((10 * 16 / WORLD_SIZE)))
elif [ "$WORLD_SIZE" -ne 16 ]; then
    echo "FATAL: world size $WORLD_SIZE != 16. Run A's global batch was 16; at any other world" >&2
    echo "       size this is not comparable with it. (Set MAX_STEPS for a smoke test.)" >&2
    exit 1
fi

mkdir -p checkpoints logs
if commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null); then
    dirty=$(git -C "$REPO" status --porcelain | wc -l)
else
    commit=unknown; dirty=unknown
fi
echo "--- #107 arm A1 configuration ---"
echo "Job ID          : ${SLURM_JOBID}   restarts: ${SLURM_RESTART_COUNT:-0}"
echo "Node List       : ${SLURM_NODELIST}"
echo "World size      : ${WORLD_SIZE}   (${SLURM_NNODES} nodes x ${NPROC_PER_NODE} GPUs)"
echo "Backbone        : ${BACKBONE}"
echo "Repo            : ${REPO}  commit ${commit}  dirty files: ${dirty}"
echo "Env             : ${ENV_PREFIX}"
echo "Data root       : ${DATA_ROOT}"
echo "Working dir     : $(pwd)"
echo "Epochs          : ${EPOCHS}   extra args: ${EXTRA[*]:-none}"
echo "Resume file     : $(pwd)/latest_checkpoint.pth$([ -f latest_checkpoint.pth ] && echo ' (PRESENT -- resuming)' || echo ' (absent -- fresh start)')"
echo "---------------------------------"

srun --export=ALL \
    "$TORCHRUN" --nnodes $SLURM_NNODES \
             --nproc_per_node $NPROC_PER_NODE \
             --rdzv_id $SLURM_JOB_ID \
             --rdzv_backend c10d \
             --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
             "$REPO/stage_two/train.py" \
             --data-root "$DATA_ROOT" \
             --epochs "$EPOCHS" \
             --preset scratch \
             --backbone "$BACKBONE" \
             --checkpoint-dir checkpoints \
             "${EXTRA[@]}"

# A smoke test must fail loudly: exit 0 is what lets a dependent job start.
if [ -n "${MAX_STEPS:-}" ]; then
    "$PYBIN" -c "
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu')
bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
print('smoke:', sys.argv[1], len(sd), 'tensors,', len(bad), 'non-finite')
sys.exit(1 if bad or not sd else 0)
" "checkpoints/final_step_${MAX_STEPS}.pth"
fi

echo "--- Slurm job finished ---"
