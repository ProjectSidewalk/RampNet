#!/bin/bash
# Rebuild the YOLO training dataset ON TILLICUM, from Hugging Face. Download + prep.
#
# WHY REGENERATE RATHER THAN TRANSFER FROM KLONE
# There is no automated path from klone to Tillicum: no shared filesystem (/gscratch and
# /mmfs1 do not exist here), and Tillicum cannot authenticate to klone non-interactively
# — both ends are Duo/keyboard-interactive with no publickey, so BatchMode fails. Globus
# CLI is on neither side. Verified 2026-07-30.
#
# Regenerating sidesteps all of it, and is SAFE because the prep is deterministic by
# construction: prepare_yolo_dataset.py thins background tiles with an md5 of the file
# stem specifically so the choice is stable across processes and runs (see
# `_keep_background`, and its comment about salted hash()).
#
# VERIFY, DO NOT ASSUME. The committed record gives the exact counts klone produced:
#     tiles 557,413 train / 161,002 val      pano 150,063 train / 42,875 val
# This job prints its own counts at the end. If they do not match, the datasets are NOT
# equivalent and nothing trained here is comparable to the #51 runs — stop and diff.
#
# WHY A GPU JOB FOR CPU WORK
# Tillicum prohibits CPU-only jobs; every job must request >=1 GPU. The runbook also
# warns that login nodes reap heavy long-running processes, and on Tillicum that would
# take the Duo-authenticated SSH master with it. So we hold one GPU and eat the cost:
# roughly 6-12 h at $0.90/h, i.e. $5-11. Cheap against a $1,500/month cap.
#
#   sbatch scripts/model_comparison/run_yolo_data_prep_tillicum.slurm
#
#SBATCH --job-name=yolo_data_prep
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --output=logs/yolo_data_prep_%j.out
#SBATCH --error=logs/yolo_data_prep_%j.err

set -euo pipefail

export REPO="${REPO:-/gpfs/home/$USER/RampNet}"
# /gpfs/scrubbed, NOT the 1 TB /gpfs/projects/makelab quota: the HF source is ~462 GB
# and the prepared YOLO layout another ~286 GB, so this needs ~750 GB. scrubbed has
# ~484 TB free and is what UW-IT recommends for active many-small-file datasets.
export SCRATCH="${SCRATCH:-/gpfs/scrubbed/$USER}"
export PYBIN="${PYBIN:-/gpfs/projects/makelab/$USER/envs/rampnet-yolo/bin/python}"
export HF_HOME="$SCRATCH/hf"
export YOLO_CONFIG_DIR="$SCRATCH/ultralytics"

mkdir -p "$SCRATCH"

echo "=== node ==="
hostname; date
df -h "$SCRATCH" | tail -1
echo "=== interpreter (must match the klone #51 runs exactly) ==="
"$PYBIN" -c "import sys, torch, ultralytics; print('python', sys.version.split()[0], '| torch', torch.__version__, '| ultralytics', ultralytics.__version__)"

cd "$REPO"

echo
echo "=== STAGE: data (HF download, ~462 GB) ==="
date
bash hyak_yolo_runbook.sh data

echo
echo "=== STAGE: prep (tiles + pano) ==="
date
# The runbook's prep stage reads SLURM_CPUS_PER_TASK and passes --workers plus
# OPENBLAS_NUM_THREADS=1 when it is set, matching what run_yolo_prep.slurm does on
# klone. Without that, prepare_yolo_dataset.py sizes its pool from os.cpu_count(),
# which reports the whole node and ignores our 8-CPU cgroup.
bash hyak_yolo_runbook.sh prep

echo
echo "=== VERIFICATION — counts must match the klone record ==="
date

# This is a GATE, not a report. The header promises "stop and diff" on a mismatch, so a
# mismatch has to exit non-zero: a job that prints 557,398 next to 557,413 and still
# exits 0 reads as green to anyone skimming the tail of the log, and the next person
# trains an arm that is not comparable to #51 without ever knowing.
#
# Expected counts come from scripts/model_comparison/yolo_baseline/README.md provenance.
# Both images/ and labels/ are checked — they are written as a pair by _write_pair, so a
# divergence between them is exactly the kind of partial-run damage worth catching.
declare -A EXPECTED=(
    [tiles/train]=557413
    [tiles/val]=161002
    [pano/train]=150063
    [pano/val]=42875
)

fail=0
for ds in tiles pano; do
    for split in train val; do
        want="${EXPECTED[$ds/$split]}"
        for kind in images labels; do
            d="$SCRATCH/yolo/$ds/$kind/$split"
            if [ "$kind" = images ]; then
                got="$(ls -U "$d" 2>/dev/null | grep -c '\.jpg$' || true)"
            else
                got="$(ls -U "$d" 2>/dev/null | grep -c '\.txt$' || true)"
            fi
            if [ "$got" -eq "$want" ]; then
                printf "  ok   %-5s %-6s %-5s %8s\n" "$ds" "$kind" "$split" "$got"
            else
                printf "  FAIL %-5s %-6s %-5s %8s  (expected %s)\n" \
                    "$ds" "$kind" "$split" "$got" "$want"
                fail=1
            fi
        done
    done
done

if [ "$fail" -ne 0 ]; then
    cat <<'EOF'

!! MISMATCH. The regenerated dataset is NOT the one the #51 arms trained on.
!! Do not train against it until the difference is understood.

Next step is the filename-level check, which is stronger than counts and is what the
doc's equivalence table reports. Per directory, on BOTH clusters:

    ls -U <root>/yolo/<dir> | sort | md5sum

  klone root    /gscratch/scrubbed/jfroehli
  Tillicum root /gpfs/scrubbed/jfroehli
EOF
    date
    exit 1
fi

echo
echo "  All eight directories match the klone record."
echo "  Counts are necessary, not sufficient — for the filename-level check see"
echo "  docs/tillicum.md, 'Verified equivalent to klone'."
date
echo 'Cost so far: run `hyakusage`'
