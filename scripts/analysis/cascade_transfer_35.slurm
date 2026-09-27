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
#   REPO=<clone at this branch, benchmark/<split>/panos present> W=<scratch dir> \
#       sbatch --export=ALL scripts/analysis/cascade_transfer_35.slurm
#
# Outputs, all under $W: model_cache/ (the Vistas detections, exported afterwards with
# export_model_cache.py), rampnet_r2048_fp32/ and rampnet_r2048_fp16/ (RampNet's floor peaks,
# checked against analysis_out/op_cache), usage_log.jsonl (paid: false rows, appended to the
# committed ledger afterwards), env.txt, and one log per step under logs/.
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
#SBATCH --output=/gscratch/scrubbed/jfroehli/cascade_35/logs/%x_%j.out
set -eo pipefail
REPO=${REPO:?}; W=${W:?}
PY=${PY:-/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python}
SPLITS=${SPLITS:-bend paterson gainesville annapolis}
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$W/logs"
NOTE="cascade-transfer-35, Slurm job $SLURM_JOB_ID"
{
  echo "=== $(date -u +%FT%TZ) job=$SLURM_JOB_ID host=$(hostname -f) commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
  $PY -c "import sys,torch,torchvision,transformers,timm,skimage;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('torchvision',torchvision.__version__);print('transformers',transformers.__version__);print('timm',timm.__version__);print('scikit-image',skimage.__version__)"
  lscpu | grep -E '^Model name|^CPU\(s\)'
} > "$W/env.txt" 2>&1
cat "$W/env.txt"

# 1. The imagery is what the GT was reviewed on (sha256 per pano, against the committed
#    manifests). A mismatch stops the job before any GPU time is spent.
CITIES=$(echo $SPLITS | tr ' ' ',')
$PY scripts/analysis/imagery_manifest.py --verify --cities "$CITIES" --panos-root "$REPO" \
  2>&1 | tee "$W/logs/imagery_verify_$SLURM_JOB_ID.log"

# 2. The Vistas parity arm, exactly the richmond flags (docs/vistas_transfer_126.md, Resolution
#    parity): --models rampnet,vistas:curb-cut --vistas-input-size 1024 1024, fp16 and
#    min_area_px 16 by default. One process per split, so each split's row carries its own
#    model load, inference seconds and s/pano.
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
      --usage-log "$W/usage_log.jsonl" --note "$NOTE, RampNet 1x pass $prec" \
      2>&1 | tee "$W/logs/rampnet_r2048_${prec}_$SLURM_JOB_ID.log"
  $PY scripts/analysis/input_res_sweep_25.py check --cities "$CITIES" \
      --cache-root "$W/rampnet_r2048_$prec/cache" --out "$W/rampnet_r2048_$prec/instrument_check.json" \
      2>&1 | tee "$W/logs/rampnet_check_${prec}_$SLURM_JOB_ID.log" || echo "check $prec did not pass (see log)"
done
echo "EXIT 0 $(date -u +%FT%TZ)"
