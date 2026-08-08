#!/bin/bash
# Tillicum environment smoke test — settles "does our stack run on H200/Rocky 9?"
#
# COSTS NOTHING. The `debug` QoS bills at UsageFactor 0.000000 (verified with
# `sacctmgr show qos`), so this is free, and it carries a HIGHER priority than normal
# (50 vs 25) so it should also start fast. Limits: 1 h, 1 GPU, 8 CPU, 200G, 1 node.
# Do every environment/throughput probe here before spending a cent of the $90 credit.
#
#   sbatch scripts/tillicum_smoke.slurm
#
# The open question this answers: our environment.yml pins linux-64 packages against
# CUDA 11.8, and Tillicum is H200 (sm_90) on Rocky 9. The recon found BOTH a
# conda/Miniforge3 module and apptainer 1.5.2, so either path is available and we do
# not yet know which is less work. This tries conda first because it reuses what we
# already have; the container fallback is the documented UW-IT path if it fails.
#
#SBATCH --job-name=tillicum_smoke
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --output=logs/tillicum_smoke_%j.out
#SBATCH --error=logs/tillicum_smoke_%j.err

set -uo pipefail   # NOT -e: this is a probe. A failing step is a RESULT, not a reason
                   # to abort and lose the rest of the report.

say() { echo; echo "=== $* ==="; }

say "NODE"
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}   GPUs: ${SLURM_GPUS_ON_NODE:-?}"

say "STORAGE — where the dataset should land"
for d in /gpfs/scrubbed /gpfs/scrubbed/$USER /gpfs/projects/makelab /gpfs/home/$USER; do
    if [ -d "$d" ]; then
        printf "%-32s exists  writable=%s  " "$d" "$( [ -w "$d" ] && echo yes || echo NO )"
        df -h "$d" 2>/dev/null | tail -1 | awk '{printf "size=%s avail=%s\n", $2, $4}'
    else
        printf "%-32s MISSING\n" "$d"
    fi
done

say "SQUASHFS SUPPORT — can we mount the packed dataset at all?"
for t in mksquashfs unsquashfs squashfuse apptainer; do
    printf "  %-12s %s\n" "$t" "$(command -v $t 2>/dev/null || echo NOT_FOUND)"
done
echo "  kernel squashfs: $(grep -qw squashfs /proc/filesystems && echo yes || echo 'no (use apptainer --bind image-src=/)')"

say "CONDA MODULE PATH"
module load conda 2>&1 | head -5
command -v conda && conda --version

# Build somewhere durable and shared, not node-local /tmp. Keep it out of the 286 GB
# dataset's way: envs are small but numerous-filed, so scrubbed is the better home.
ENVROOT="${ENVROOT:-/gpfs/scrubbed/$USER/envs}"
if ! mkdir -p "$ENVROOT" 2>/dev/null; then
    echo "  (could not write $ENVROOT -- falling back to the project dir)"
    ENVROOT="/gpfs/projects/makelab/$USER/envs"
    mkdir -p "$ENVROOT"
fi
VENV="$ENVROOT/ultra"
echo "env root: $ENVROOT"

say "BUILD A MINIMAL ULTRALYTICS ENV"
# A venv on top of the module python is far lighter than solving environment.yml, and
# the only question that matters here is whether a CUDA-12 torch wheel drives an H200.
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" 2>&1 | tail -2
fi
"$VENV/bin/python" -m pip install -q --upgrade pip 2>&1 | tail -2
# cu126 wheels: Tillicum is sm_90, and our klone stack is torch 2.13.0+cu126 (see the
# yolo_baseline record), so matching it keeps the comparison honest.
"$VENV/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
"$VENV/bin/python" -m pip install -q ultralytics 2>&1 | tail -3

say "DOES IT SEE THE GPU?"
"$VENV/bin/python" - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0), "(expect (9,0) for H200)")
        import time
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50):
            a @ a
        torch.cuda.synchronize()
        dt = time.time() - t0
        tflops = 50 * 2 * 8192**3 / dt / 1e12
        print(f"bf16 matmul: {tflops:.0f} TFLOP/s  ({dt:.2f}s for 50 iters)")
        print("  ref: L40S ~180 dense. A large ratio here is the H200 speedup we are betting on.")
except Exception as e:
    print("TORCH FAILED:", type(e).__name__, e); sys.exit(0)

try:
    from ultralytics import YOLO
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
    m = YOLO("yolo11n.yaml")          # from config, no download needed
    print("model constructed OK")
except Exception as e:
    print("ULTRALYTICS FAILED:", type(e).__name__, e)
PY

say "SLURM VIEW OF THIS JOB (confirm debug really is free)"
scontrol show job "$SLURM_JOB_ID" | grep -oE "QOS=[a-z]+|Partition=[a-z0-9-]+|TRES=[^ ]+" | head -5

say "DONE"
date
echo "Check cost with: hyakusage   (this job should add 0.00)"
