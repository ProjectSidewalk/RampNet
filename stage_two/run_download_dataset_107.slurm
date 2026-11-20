#!/bin/bash
# Download the Stage 2 dataset's parquet shards from the Hugging Face Hub into $HF_HOME, as a
# batch job (issue #107). This is step 1 of the re-stage in docs/restage_263.md, which until
# now was run by hand; 431 GiB is too long a transfer for a klone login node, where heavy
# processes are reaped. Step 2 is stage_two/run_restage_parquet_263.slurm, which reads what
# this job writes.
#
# Reads:  the Hub dataset projectsidewalk/rampnet-dataset at $HF_REVISION (public; a token at
#         $HF_TOKEN_PATH is used when present, for rate limits only).
# Writes: $HF_HOME/hub/datasets--projectsidewalk--rampnet-dataset/snapshots/$HF_REVISION/
# Resumable: `hf download` skips files already complete, so a requeued job continues.
#
# USAGE, from the root of the checkout (logs/ must already exist):
#   mkdir -p logs
#   DL=$(RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 sbatch --parsable stage_two/run_download_dataset_107.slurm)
#   RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2 HF_REVISION=ee882e3f3c779dc13182f307bca616e50d9b8c5c \
#       sbatch --dependency=afterok:$DL stage_two/run_restage_parquet_263.slurm
# Optional: HF_HOME (default /gscratch/scrubbed/$USER/hf), HF_REVISION (default: the revision
# of record below), HF_TOKEN_PATH (default $HOME/.cache/huggingface/token).
#
#SBATCH -p ckpt
#SBATCH -A ckpt-makelab
#SBATCH --qos=ckpt
#SBATCH --job-name=rampnet_download_107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o logs/download_107_%j.out
#SBATCH -e logs/download_107_%j.err
set -euo pipefail
RAMPNET_ENV="${RAMPNET_ENV:?set RAMPNET_ENV to the conda prefix, e.g. RAMPNET_ENV=/gscratch/makelab/jonf/envs/sidewalkcv2}"
HF_BIN="$RAMPNET_ENV/bin/hf"
[ -x "$HF_BIN" ] || { echo "FATAL: RAMPNET_ENV=$RAMPNET_ENV has no bin/hf (huggingface_hub >= 1.0)" >&2; exit 1; }
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
# The token lives in the home directory (not purged); HF_HOME is on scrubbed storage. The
# variable is exported only when the file exists, and its content is never printed.
HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [ -f "$HF_TOKEN_PATH" ]; then export HF_TOKEN_PATH; else unset HF_TOKEN_PATH; fi
# The revision of record for every staged copy since 2026-10-20 (docs/restage_263.md).
HF_REVISION="${HF_REVISION:-ee882e3f3c779dc13182f307bca616e50d9b8c5c}"
echo "host $(hostname)  start $(date -u +%FT%TZ)  HF_HOME $HF_HOME  revision $HF_REVISION  restart ${SLURM_RESTART_COUNT:-0}"
"$HF_BIN" download projectsidewalk/rampnet-dataset --repo-type dataset --revision "$HF_REVISION"
SNAP="$HF_HOME/hub/datasets--projectsidewalk--rampnet-dataset/snapshots/$HF_REVISION"
echo "snapshot $SNAP: $(find -L "$SNAP" -name '*.parquet' | wc -l) parquet files, $(du -shL "$SNAP" | cut -f1)"
echo "end $(date -u +%FT%TZ)"
