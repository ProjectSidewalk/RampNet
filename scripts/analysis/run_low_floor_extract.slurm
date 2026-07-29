#!/bin/bash
# Low-floor peak extraction over every benchmark split, on Hyak (klone) — the one
# GPU step of the operating-point sweep (issue #54; #55 and #78 ride on the same
# cache). Everything downstream (curves, per-tier pooling, calibration histograms,
# the incremental-FP gallery) is CPU-only and reads the cache this writes.
#
# Pass your Slurm account with -A; there is no usable default, and a submission
# without one is rejected. Find yours with:
#   sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS
#
#   mkdir -p logs
#   sbatch -A <account> scripts/analysis/run_low_floor_extract.slurm
#   CITIES=clovis,morgantown sbatch -A <account> scripts/analysis/run_low_floor_extract.slurm
#
# Runs on ONE GPU: inference is a single 2048x4096 forward pass per pano (~1-2 s),
# so ~1,750 panos across seven splits is well under an hour. This deliberately uses
# the dedicated gpu-l40s allocation rather than a ckpt-* partition — a preempted job
# would be recoverable (see --force below) but there is no reason to risk it for a
# sub-hour run.
#
# Resumable: `extract` skips any split whose cache already exists, so a preempted or
# timed-out job re-submits with no argument changes and picks up at the split it died
# in. Pass FORCE=1 to re-extract splits that already have a cache.
#
# PARITY IS THE GATE. After this finishes, run (CPU, locally or on a login node):
#   python scripts/analysis/low_floor_sweep.py parity
# The peaks at >= 0.55 must reproduce each split's committed records.jsonl. Bundle
# panos are native-res while production GSV went through a 4096x2048 intermediate —
# one extra resample — so expect agreement to a few pixels, not bit-for-bit. A gross
# mismatch means preprocessing diverged and every downstream number inherits it.
# manual_gold is exempt: its committed detections were exported WITH flip-TTA, so
# its row is a TTA delta rather than a drift check.
#
# PYTHON is the interpreter to run with; default is whatever `python` resolves to
# after the env is activated (see the runbook in docs/model_comparison.md). Needs
# timm + scikit-image + huggingface_hub (the model is loaded from the HF hub).
#SBATCH -p gpu-l40s
#SBATCH --job-name=low_floor_extract
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/low_floor_extract_%j.out
#SBATCH --error=logs/low_floor_extract_%j.err

set -euo pipefail

# All seven splits. manual_gold is included deliberately: it is the in-distribution
# reference with independently-labelled (non-RampNet-anchored) GT, and its imagery is
# already staged on klone. It is swept alongside the deployment cities but held out of
# the pooled recommendation — see low_floor_sweep.py's HELD_OUT.
CITIES="${CITIES:-richmond,bend,clovis,morgantown,annapolis,budapest_district5,manual_gold}"
SCORE_FLOOR="${SCORE_FLOOR:-0.05}"
MIN_DISTANCE="${MIN_DISTANCE:-10}"
CACHE="${CACHE:-analysis_out/op_cache}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    source activate sidewalkcv2
    PYTHON=python
fi

FORCE_FLAG=""
if [ -n "${FORCE:-}" ]; then
    FORCE_FLAG="--force"
fi

echo "--- low-floor extraction (issue #54) ---"
echo "Splits:     ${CITIES}"
echo "Score floor:${SCORE_FLOOR}   min_distance: ${MIN_DISTANCE}"
echo "Cache:      ${CACHE}"
echo "Node:       ${SLURMD_NODENAME:-?}"
echo "Python:     $($PYTHON -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "----------------------------------------"

# min_distance stays at 10: issue #25 found 10->3 buys ~0.5 recall points and costs
# precision, and peak_nms_check.py (#62) showed no suppression radius separates real
# adjacent ramp pairs from the rare duplicate. The threshold is the lever here.
"$PYTHON" -u scripts/analysis/operating_point_curve.py extract \
    --cities "$CITIES" \
    --score-floor "$SCORE_FLOOR" \
    --min-distance "$MIN_DISTANCE" \
    --cache "$CACHE" \
    $FORCE_FLAG

echo
echo "--- extraction done; caches in ${CACHE} ---"
ls -la "$CACHE"
echo
echo "NEXT: run the parity gate before trusting anything downstream:"
echo "  python scripts/analysis/low_floor_sweep.py parity --cache ${CACHE}"
