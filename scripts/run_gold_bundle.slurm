#!/bin/bash
# Build the benchmark/manual_gold bundle on Hyak (klone), end to end (issue #58):
# fetch the gold imagery out of the HF test split, export RampNet's detections,
# and finish with the reproduction gate against the published gold-set numbers
# (P 0.949 / R 0.873 @ conf >= 0.55, TTA). Run this BEFORE spending anything on
# challenger models — the gate is this split's validity check.
#
# Pass your Slurm account with -A (find it: sacctmgr -nP show assoc user=$USER
# format=Account,QOS) and the stage-2 checkpoint via CHECKPOINT:
#
#   mkdir -p logs
#   CHECKPOINT=/path/to/stage2.pth sbatch -A <account> scripts/run_gold_bundle.slurm
#
# The fetch is CPU/network-bound (~44 GB of test-split parquet through HF_HOME);
# the GPU is used only by the export. Both halves are idempotent: a complete
# bundle skips the fetch, HF caches shards, and heatmaps are cached per
# checkpoint+TTA, so a preempted or timed-out job resumes where it stopped.
#
# Home quota on klone is small (10 GB) and the bundle + heatmap cache are a few
# GB: point benchmark/manual_gold/panos and .../heatmap_cache at scratch
# (symlinks are fine — every path here goes through the repo dir) and keep
# HF_HOME on scratch as the runbook does.
#
# PYTHON is the interpreter to run with; default is whatever `python` resolves
# to after the env is activated (see the runbook in docs/model_comparison.md).
# On top of the runbook's lean comparison env this needs timm + scikit-image
# (the export loads the model) and datasets (the fetch reads the HF parquet).
#SBATCH -p gpu-l40s
#SBATCH --job-name=gold_bundle
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/gold_bundle_%j.out
#SBATCH --error=logs/gold_bundle_%j.err

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT=<stage-2 .pth> (passed to export_gold_records.py --checkpoint)}"
BUNDLE_DIR=benchmark/manual_gold
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    source activate sidewalkcv2
    PYTHON=python
fi

echo "--- manual_gold bundle build ---"
echo "Checkpoint: ${CHECKPOINT}"
echo "HF_HOME:    ${HF_HOME}"
echo "Node:       ${SLURMD_NODENAME:-?}"
echo "Python:     $($PYTHON -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "--------------------------------"

# 1. Imagery fetch. A complete bundle is skipped; a partial one is rebuilt.
#    --force on the partial path is safe: detections only exist after step 2,
#    and a bundle that got that far is complete and never reaches this branch.
n_panos=$(find "$BUNDLE_DIR/panos" -name '*.jpg' 2>/dev/null | wc -l)
if [ -f "$BUNDLE_DIR/records.jsonl" ] && [ "$n_panos" -ge 1000 ]; then
    echo "fetch: records.jsonl + ${n_panos} panos already present -- skipping"
elif [ -f "$BUNDLE_DIR/records.jsonl" ]; then
    "$PYTHON" -u scripts/fetch_manual_gold.py --force
else
    "$PYTHON" -u scripts/fetch_manual_gold.py
fi

# 2. RampNet detections + the reproduction gate (printed at the end of the run).
"$PYTHON" -u scripts/export_gold_records.py --checkpoint "$CHECKPOINT"

echo "--- done; check the gate output above before running challengers ---"
