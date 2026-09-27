#!/bin/bash
# DA3 extraction for the #101 calibration, on Hyak (klone): the one GPU step of
# scripts/analysis/da3_calibration_101.py. Everything after it is CPU-only.
#
# Layout it expects (all under ROOT, default /gscratch/scrubbed/jfroehli/da3_101):
#   ROOT/repo                     a clone of this repo at the commit being run
#   ROOT/env                      python venv (--system-site-packages over a sidewalkcv2 env)
#                                 plus: omegaconf einops addict opencv-python-headless plyfile
#                                 pycolmap trimesh evo  (scripts/analysis/README.md, DA3 setup)
#   ROOT/src/Depth-Anything-3     git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
#                                 && git -C ... checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
#                                 (extract refuses other DA3 code; the weights revision is pinned in the script)
#   ROOT/src/Depth-Anything-3/stubs/moviepy/{__init__,editor}.py   empty stubs
#   PANOS_ROOT/<split>/panos/*.jpg  the benchmark panos, each sha256-checked against the bundle's
#                                 imagery_manifest.json. HF projectsidewalk/rampnet-benchmark has nine of
#                                 the eleven splits; laurens_gsv and laurens_mapillary are NOT on the Hub
#                                 (scripts/unpack_benchmark_panos.py), so their panos are not public input.
#
#   mkdir -p logs   # the default log path below is relative to the submit directory; or pass --output=
#   sbatch -A ckpt-makelab -p ckpt-all scripts/analysis/da3_calibration_101.slurm
#   EXTRA="--splits bend --limit 3" sbatch ... (smoke test)
#
# Resumable: the script appends one JSONL line per pano and skips panos already written,
# so a preempted (requeued) job continues where it stopped. Each incarnation appends
# one paid:false usage row to OUT/usage_rows.jsonl.
#SBATCH --job-name=da3_calib_101
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT="${ROOT:-/gscratch/scrubbed/jfroehli/da3_101}"
PANOS_ROOT="${PANOS_ROOT:-/gscratch/makelab/jonf/rampnet_benchmark}"
OUT="${OUT:-$ROOT/out/raw}"
export HF_HOME="$ROOT/hf"
export DA3_SRC="$ROOT/src/Depth-Anything-3/src"
export PYTHONUNBUFFERED=1
cd "$ROOT/repo"
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  restart ${SLURM_RESTART_COUNT:-0}  commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$ROOT/env/bin/python" scripts/analysis/da3_calibration_101.py extract \
    --panos-root "$PANOS_ROOT" --out-dir "$OUT" ${LABEL:+--label "$LABEL"} ${EXTRA:-}
