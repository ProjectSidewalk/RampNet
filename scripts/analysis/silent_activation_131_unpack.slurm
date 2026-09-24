#!/bin/bash
# #131 step 1 of 2: the published benchmark panoramas, from the Hub, onto klone scratch.
#
# CPU only. Fetches the `native` Parquet for the seven US splits Phase 1 pools from
# projectsidewalk/rampnet-benchmark and unpacks it into a checkout-shaped root, checking every
# byte against the Parquet's own sha256 column and the committed imagery_manifest.json. The
# unpack is resumable, so `--requeue` on ckpt is safe. Compute nodes have outbound network
# (context_fov_86_env.slurm built an env the same way); the login node does no download.
#
#   cd /gscratch/makelab/jonf/silent_activation_131/RampNet
#   mkdir -p logs    # --output below is relative to this directory, and logs/ is not tracked
#   sbatch scripts/analysis/silent_activation_131_unpack.slurm      # ~8 GB, minutes
#
# The repo's conda env (environment.yml) is enough: it has huggingface_hub, and pyarrow comes
# in with `datasets` (20.0.0 on klone). The 2026-09-23 run (job 40546626) used a separate uv
# venv, /gscratch/makelab/jonf/envs/hfunpack, on the mistaken belief that the conda env had
# no pyarrow; UNPACK_PY still overrides the interpreter. The unpack only copies bytes and
# checks their hashes, so the interpreter cannot change what lands on disk.
#
#SBATCH --job-name=sa131_unpack
#SBATCH --account=ckpt-makelab
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/sa131_unpack_%j.out
set -eo pipefail
WORK=${WORK:-/gscratch/makelab/jonf/silent_activation_131}
PY=${UNPACK_PY:-/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python}
REV=${BENCHMARK_REV:-main}
CITIES=${CITIES:-richmond,bend,clovis,morgantown,annapolis,paterson,gainesville}
export HF_HOME=$WORK/hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "$(date -u +%FT%TZ) $(hostname -f) job=$SLURM_JOB_ID rev=$REV cities=$CITIES"
$PY scripts/unpack_benchmark_panos.py --out "$WORK/benchmark_root" --cities "$CITIES" \
    --config native --revision "$REV" --cache-dir "$HF_HOME/hub"
# the Parquet cache is the download itself (~8 GB); the loose files are what the run reads.
# Kept until the run has finished so a re-unpack costs no network; delete by hand after.
du -sh "$WORK/benchmark_root" "$HF_HOME" 2>/dev/null
echo "EXIT 0"; date -u +%FT%TZ
