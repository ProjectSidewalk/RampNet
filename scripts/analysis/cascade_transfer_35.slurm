#!/bin/bash
# The #35 cascade transfer run (docs/cascade_cost_35.md, "Transfer"): the Vistas parity arm
# (mask2former-vistas-curb-cut-1024x1024) on bend, paterson, gainesville and annapolis, and
# RampNet's 1x pass timed on the same GPU in the same job, so the per-pano ratio is fair.
#
# One L40S on the lab's allocation (no preemption, so each leg is timed in one piece). Every
# step is resumable: compare.py skips panos already in $W/model_cache and input_res_sweep_25.py
# skips a split whose arm cache exists, so a resubmission only redoes what is missing (and only
# what it redoes gets a usage row).
#
# Before submitting (full commands in docs/cascade_cost_35.md, "Reproduction (transfer)"):
#   1. the native-resolution panoramas at $REPO/benchmark/<split>/panos, e.g.
#        python scripts/unpack_benchmark_panos.py --out $REPO --cities bend,paterson,gainesville,annapolis
#      (all four splits are on projectsidewalk/rampnet-benchmark);
#   2. the two checkpoints in $W/hf (the job runs offline by default), e.g.
#        HF_HOME=$W/hf python -c 'from huggingface_hub import snapshot_download as d; \
#          d("facebook/mask2former-swin-large-mapillary-vistas-semantic"); d("projectsidewalk/rampnet-model")'
#      or submit with HF_OFFLINE=0 to let the job download them itself (needs network on the node);
#   3. mkdir -p logs  (the --output path below is relative to the submit directory, the repo).
#
#   cd <clone at this branch> && mkdir -p logs
#   PY=<env python> sbatch --export=ALL scripts/analysis/cascade_transfer_35.slurm
#
# Variables (all optional except that PY must name an env with torch/transformers/timm/skimage):
#   REPO   clone to run from                default: the submit directory
#   W      scratch root for caches/outputs  default: /gscratch/scrubbed/$USER/cascade_35
#   PY     python interpreter               default: python on PATH
#   SPLITS splits to run                    default: bend paterson gainesville annapolis
#   HF_OFFLINE  1 = no network in the job   default: 1 (needs step 2)
#
# Outputs, all under $W: model_cache/ (the Vistas detections, exported afterwards with
# export_model_cache.py), rampnet_r2048_fp32/ and rampnet_r2048_fp16/ (RampNet's floor peaks,
# checked against analysis_out/op_cache), usage_log.jsonl (paid: false rows, appended to the
# committed ledger afterwards), env.txt (including the checkpoint snapshot each model resolved
# to), and one log per step under $W/logs/.
#
# Deviation recorded: the committed 2026-09-27 run (job 40774146) used the pre-review version
# of this file, which hardcoded W, PY and the --output path to jfroehli's scratch; the steps it
# ran are the same.
#
#SBATCH --job-name=cascade35_transfer
#SBATCH --account=gpu-l40s-makelab
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
W=${W:-/gscratch/scrubbed/$USER/cascade_35}
PY=${PY:-python}
SPLITS=${SPLITS:-bend paterson gainesville annapolis}
HF_OFFLINE=${HF_OFFLINE:-1}
export HF_HOME=$W/hf
if [ "$HF_OFFLINE" = 1 ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$W/logs"
NOTE="cascade-transfer-35, Slurm job $SLURM_JOB_ID"
snapshot() {  # the commit refs/main resolved to in $HF_HOME, i.e. what an offline load uses
  local d="$HF_HOME/hub/models--${1//\//--}"
  echo "$1 refs/main=$(cat "$d/refs/main" 2>/dev/null || echo MISSING) snapshots=$(ls "$d/snapshots" 2>/dev/null | tr '\n' ' ')"
}
{
  echo "=== $(date -u +%FT%TZ) job=$SLURM_JOB_ID host=$(hostname -f) commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
  $PY -c "import sys,torch,torchvision,transformers,timm,skimage;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('torchvision',torchvision.__version__);print('transformers',transformers.__version__);print('timm',timm.__version__);print('scikit-image',skimage.__version__)"
  lscpu | grep -E '^Model name|^CPU\(s\)'
  echo "HF_HOME=$HF_HOME HF_OFFLINE=$HF_OFFLINE"
  snapshot facebook/mask2former-swin-large-mapillary-vistas-semantic
  snapshot projectsidewalk/rampnet-model
} > "$W/env.txt" 2>&1
cat "$W/env.txt"

# 1. The imagery is what the GT was reviewed on (sha256 per pano, against the committed
#    manifests). A mismatch stops the job before any GPU time is spent.
CITIES=$(echo $SPLITS | tr ' ' ',')
$PY scripts/analysis/imagery_manifest.py --verify --cities "$CITIES" --panos-root "$REPO" \
  2>&1 | tee "$W/logs/imagery_verify_$SLURM_JOB_ID.log"

# 2. The Vistas parity arm, exactly the richmond flags (docs/vistas_transfer_126.md, Resolution
#    parity): --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024, fp16 and
#    min_area_px 16 by default, no --vistas-revision (setting it changes the signature and so
#    the leg; the snapshot actually loaded is recorded in env.txt above instead). One process
#    per split, so each split's row carries its own model load, inference seconds and s/pano.
for s in $SPLITS; do
  echo "=== vistas $s $(date -u +%FT%TZ)"
  $PY scripts/model_comparison/compare.py "benchmark/$s" \
      --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024 \
      --cache-dir "$W/model_cache" --usage-log "$W/usage_log.jsonl" \
      2>&1 | tee "$W/logs/vistas1024_${s}_$SLURM_JOB_ID.log"
done

# 3. RampNet's 1x pass (2048x4096 input, one forward, peaks to the 0.05 floor), same GPU.
#    fp32 is the precision the committed op_caches were made at, and `check` compares
#    against them; the fp16 arm is the like-for-like precision against Mask2Former's fp16.
for prec in fp32 fp16; do
  if [ $prec = fp16 ]; then FP=on; else FP=off; fi
  echo "=== rampnet r2048 $prec $(date -u +%FT%TZ)"
  $PY scripts/analysis/input_res_sweep_25.py extract --arms r2048 --cities "$CITIES" \
      --panos-root "$REPO" --cache-root "$W/rampnet_r2048_$prec/cache" --fp16 $FP \
      --usage-log "$W/usage_log.jsonl" --note "$NOTE, RampNet 1x pass $prec" --issue 35 \
      2>&1 | tee "$W/logs/rampnet_r2048_${prec}_$SLURM_JOB_ID.log"
  $PY scripts/analysis/input_res_sweep_25.py check --cities "$CITIES" \
      --cache-root "$W/rampnet_r2048_$prec/cache" --out "$W/rampnet_r2048_$prec/instrument_check.json" \
      2>&1 | tee "$W/logs/rampnet_check_${prec}_$SLURM_JOB_ID.log" || echo "check $prec did not pass (see log)"
done
echo "EXIT 0 $(date -u +%FT%TZ)"
