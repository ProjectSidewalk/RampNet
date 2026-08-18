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
# The fetch is CPU/network-bound. Measured 2026-08-14 on makelab2: load_dataset
# downloaded and arrow-materialized ALL THREE splits (~2.5 h) despite split="test",
# so budget for that, not for a ~44 GB test-only pull;
# the GPU is used only by the export. Both halves are idempotent: the fetch skips
# panos already on disk, HF caches shards, and heatmaps are cached per
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

# 1. Imagery fetch. records.jsonl is COMMITTED and carries the detections step 2
#    exports into it, so on a fresh clone it is already present and only the
#    git-ignored panos/ are missing — that is exactly --images-only, which never
#    writes a committed file and skips panos already on disk (so a preempted job
#    resumes). This branch used --force until 2026-08-18; on a fresh clone that
#    silently discarded the committed detections and rewrote bundle_meta.json,
#    which is what happened on makelab2 on 2026-08-14 (#118). The old
#    "n_panos >= 1000" skip is gone with it: --images-only does the same job
#    correctly, checking each present pano against records.jsonl instead of
#    trusting a file count that a 999-pano directory silently fails.
if [ -f "$BUNDLE_DIR/records.jsonl" ]; then
    "$PYTHON" -u scripts/fetch_manual_gold.py --images-only
else
    "$PYTHON" -u scripts/fetch_manual_gold.py
fi

# 2. RampNet detections + the reproduction gate (printed at the end of the run).
"$PYTHON" -u scripts/export_gold_records.py --checkpoint "$CHECKPOINT"

echo "--- done; check the gate output above before running challengers ---"
