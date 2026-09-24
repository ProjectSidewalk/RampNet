#!/bin/bash
# #131 step 2 of 2: re-run #46 Phase 1 (silent_activation.py) on klone, on the panoramas
# step 1 unpacked, and write the replica beside the committed result -- never over it.
#
# One L40S on the lab's allocation (no preemption; the run is ~20 min and has no resume).
# The environment is the repo's own conda env built from environment.yml
# (/gscratch/makelab/jonf/envs/sidewalkcv2: torch 2.6.0+cu126, timm 1.0.28 -- the same
# library versions as the Windows venv that produced the committed file, so what differs
# between the two runs is the GPU, the OS, the driver and cuDNN, not the code path).
#
#   cd /gscratch/makelab/jonf/silent_activation_131/RampNet
#   sbatch --dependency=afterok:<unpack job> scripts/analysis/silent_activation_131.slurm
#   # optional second copy, first-come on the scavenger partition (same GPU class, another node):
#   sbatch --account=ckpt-makelab --partition=ckpt-all --gpus=l40s:1 --requeue \
#       --export=ALL,OUT=analysis_out/silent_activation_replica_ckpt.json,TAG=silent-activation-replica-131-ckpt \
#       scripts/analysis/silent_activation_131.slurm
#
# Outputs, all under the checkout (copied back and committed):
#   analysis_out/silent_activation_replica.json       the replica itself
#   analysis_out/silent_activation_replica.run.json   host, GPU, driver, library versions,
#                                                     model + dataset commits, elapsed
#   analysis_out/usage_log.jsonl                      one paid:false row (elapsed, host, GPU)
#   logs/sa131_phase1_<job>.out                       the script's printed tables
#
#SBATCH --job-name=sa131_phase1
#SBATCH --account=gpu-l40s-makelab
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/sa131_phase1_%j.out
set -eo pipefail
WORK=${WORK:-/gscratch/makelab/jonf/silent_activation_131}
PY=${PY:-/gscratch/makelab/jonf/envs/sidewalkcv2/bin/python}
PANOS_ROOT=${PANOS_ROOT:-$WORK/benchmark_root}
# OUT and TAG can be overridden (--export=ALL,OUT=...,TAG=...) so a second copy of the job on the
# preemptable ckpt partition writes a second replica beside this one instead of over it.
OUT=${OUT:-analysis_out/silent_activation_replica.json}
TAG=${TAG:-silent-activation-replica-131}
export HF_HOME=$WORK/hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
START=$(date -u +%FT%TZ); T0=$SECONDS
HOST=$(hostname -f)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "$START $HOST $GPU driver=$DRIVER job=$SLURM_JOB_ID panos_root=$PANOS_ROOT"
[ -f "$PANOS_ROOT/unpack_manifest.json" ] || { echo "no unpack_manifest.json under $PANOS_ROOT; run the unpack job first" >&2; exit 2; }

$PY scripts/analysis/silent_activation.py --panos-root "$PANOS_ROOT" --json-out "$OUT"
ELAPSED=$((SECONDS - T0))
echo "phase 1 done in ${ELAPSED}s"

# the run record: what ran, on what, with which library and artifact versions
START="$START" HOST="$HOST" GPU="$GPU" DRIVER="$DRIVER" ELAPSED="$ELAPSED" PANOS_ROOT="$PANOS_ROOT" \
JOB="$SLURM_JOB_ID" OUT="$OUT" TAG="$TAG" $PY - <<'EOF'
import json, os, platform, sys, hashlib
import torch, timm, numpy, PIL, skimage
from huggingface_hub import HfApi
from rampnet import ledger
unpack = json.load(open(os.path.join(os.environ["PANOS_ROOT"], "unpack_manifest.json")))
model_sha = HfApi().model_info("projectsidewalk/rampnet-model").sha
out = os.environ["OUT"]
digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
rec = {
    "what": "scripts/analysis/silent_activation.py --json-out (the #46 Phase 1 study), re-run for #131",
    "result": out, "result_sha256": digest,
    "started": os.environ["START"], "elapsed_s": int(os.environ["ELAPSED"]),
    "cluster": "klone", "slurm_job_id": os.environ["JOB"], "host": os.environ["HOST"],
    "gpu": os.environ["GPU"], "driver": os.environ["DRIVER"],
    "os": platform.platform(), "python": sys.version.split()[0],
    "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "timm": timm.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
    "scikit_image": skimage.__version__,
    "model": {"repo_id": "projectsidewalk/rampnet-model", "revision": model_sha, "file": "model.safetensors"},
    "panos": {"repo_id": unpack["repo_id"], "revision": unpack["resolved_sha"] or unpack["revision"],
              "config": unpack["config"], "cities": {c: v["n"] for c, v in unpack["cities"].items()},
              "checked_against_committed_manifests": all(v["checked_against_manifest"] for v in unpack["cities"].values())},
}
with open(out.replace(".json", ".run.json"), "w", encoding="utf-8", newline="") as fh:
    json.dump(rec, fh, indent=1, sort_keys=True)
row = {"ts": os.environ["START"], "bundle": "pooled-7-us-splits", "label": os.environ["TAG"],
       "provider": "rampnet", "model_id": "projectsidewalk/rampnet-model", "paid": False,
       "panos_scored": 108, "elapsed_s": float(os.environ["ELAPSED"]),
       "hardware": {"host": os.environ["HOST"], "gpus": [os.environ["GPU"]]}, "status": "ok",
       "what": rec["what"] + f", klone job {os.environ['JOB']}, single-pass fp32, 128 misses / 108 panos",
       "est_cost_usd": 0.0, "pricing": None, "run_id": f"{os.environ['TAG']}:{os.environ['START']}"}
ledger.append_rows("analysis_out/usage_log.jsonl", [row])
print(json.dumps(rec, indent=1))
EOF
echo "EXIT 0"; date -u +%FT%TZ
