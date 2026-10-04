#!/bin/bash
# After #82 fine-tune jobs finish on klone: copy each final checkpoint off the purging
# scrubbed volume, hash it, and submit one scoring job per checkpoint.
#
#   cd /gscratch/scrubbed/$USER/RampNet_aug82
#   bash scripts/analysis/aug82_finish_klone.sh control_s1 both_s1 ...
#
# Copies $RUNDIR_ROOT/<label>/checkpoints/final_step_$STEPS.pth to $DST/<label>.pth (DST is
# on /gscratch/makelab, purchased and never purged -- NOTE the literal jonf; few large files
# only, that volume is near its file-count cap), checks the copy with cmp, rewrites
# $DST/SHA256SUMS over every .pth there, then submits aug82_score_ckpts.slurm per label.
set -eu
RUNDIR_ROOT="${RUNDIR_ROOT:-/gscratch/scrubbed/$USER/aug82}"
DST="${DST:-/gscratch/makelab/jonf/aug82}"
STEPS="${STEPS:-2000}"
RAMPNET_ENV="${RAMPNET_ENV:-/gscratch/makelab/jonf/envs/sidewalkcv2}"
mkdir -p "$DST"
for label in "$@"; do
    src="$RUNDIR_ROOT/$label/checkpoints/final_step_$STEPS.pth"
    [ -f "$src" ] || { echo "not finished: $src" >&2; exit 1; }
    cp "$src" "$DST/$label.pth"
    cmp "$src" "$DST/$label.pth"
    echo "copied $label"
done
(cd "$DST" && sha256sum *.pth > SHA256SUMS && cat SHA256SUMS)
for label in "$@"; do
    RAMPNET_ENV="$RAMPNET_ENV" LABELS="$label" CKPTS="$DST" \
        sbatch --job-name="aug82_score_$label" scripts/analysis/aug82_score_ckpts.slurm
done
