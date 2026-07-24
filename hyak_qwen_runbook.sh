#!/bin/bash
# ============================================================================
#  Qwen3-VL benchmark run on Hyak (klone) — self-contained runbook.
#
#  UNTRACKED: this file is a scratch operator script, like the earlier
#  hyak_runbook.sh. Do not commit it.
#
#  Claude cannot drive klone (gssapi/keyboard-interactive only — UW password +
#  Duo on every connection), so you run these stages. Each is idempotent and
#  safe to re-run.
#
#  Usage on klone:   bash hyak_qwen_runbook.sh <stage>
#  Stages:           env | weights | smoke | run | run32b | status | collect
#  Open models (#39): weightsopen | smokeopen | runopen | runmolmo
#  Windows-side:     bash hyak_qwen_runbook.sh push   (prints the rsync commands)
#
#  CRITICAL PATH — start the two slow things first, they overlap:
#    1. (Windows/WSL) start the richmond pano upload   ~364 MB
#    2. (klone login) `env` then `weights`             ~16 GB checkpoint
#    3. (klone login) `smoke`, then `run`
#  Bend (~1.7 GB) can upload while richmond is already running.
# ============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/RampNet}"
USER="${USER:-$(whoami)}"          # set -u would trip on this off-cluster
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/$USER/hf}"
QWEN_8B="Qwen/Qwen3-VL-8B-Instruct"
QWEN_32B="Qwen/Qwen3-VL-32B-Instruct"
OWLV2="google/owlv2-large-patch14-ensemble"
GDINO="IDEA-Research/grounding-dino-base"
MOLMO="${MOLMO:-allenai/Molmo2-8B}"
STAGE="${1:-help}"

banner() { echo; echo "=== $* ==="; echo; }

case "$STAGE" in

# ---------------------------------------------------------------------------
help|push)
  cat <<'TXT'
Run these from a shell that has rsync (on Windows: WSL). Define a `klone` host in
~/.ssh/config pointing at klone.hyak.uw.edu with your UW netid, or substitute
<netid>@klone.hyak.uw.edu below. Each connection wants your UW password + Duo,
so add ControlMaster/ControlPath/ControlPersist to that host block and every
later connection reuses the first one. Start step 1 now and let it run in the
background.

  cd <parent directory of your RampNet checkout>

  # 1. Repo (fast). Excludes the venv, the local Gemini cache, and the imagery.
  rsync -av --exclude .venv --exclude .model_cache --exclude 'benchmark/*/panos' \
        --exclude 'benchmark/*/gallery' --exclude view_dump \
        RampNet/ klone:~/RampNet/

  # 2. Richmond imagery — 364 MB, the fast path to a first number.
  rsync -av --partial --progress \
        RampNet/benchmark/richmond/panos/ klone:~/RampNet/benchmark/richmond/panos/

  # 3. Bend imagery — 1.7 GB. Start it after richmond is running on the cluster.
  rsync -av --partial --progress \
        RampNet/benchmark/bend/panos/ klone:~/RampNet/benchmark/bend/panos/

Send the NATIVE panos, not downscaled copies. The harness downscales in-process
to 4096 px; pre-resizing would re-encode the JPEG, and a past gold-set re-eval
showed JPEG re-encoding alone moved P by +2.2 and R by -1.8 pts. Not worth
saving upload time on a benchmark number.

  # 4. When the run finishes, pull the detections back (they score with no GPU):
  rsync -av klone:~/RampNet/.model_cache/ RampNet/.model_cache/
TXT
  ;;

# ---------------------------------------------------------------------------
env)
  banner "Conda env + VLM deps (login node)"
  cd "$REPO"
  if ! conda env list | grep -q '^sidewalkcv2 '; then
    # CONDA_OVERRIDE_CUDA is load-bearing: solving on a GPU-less login node makes
    # conda-forge silently pick CPU-only pytorch, and Qwen then crawls on CPU
    # while the allocated GPU idles.
    CONDA_OVERRIDE_CUDA=12.6 conda env create -f environment.yml
  else
    echo "sidewalkcv2 already exists; skipping create"
  fi
  # shellcheck disable=SC1091
  source activate sidewalkcv2

  # environment.yml pins an unpinned conda `transformers`, usually older than the
  # 4.57 that first shipped Qwen3-VL. Upgrade via pip; transformers has no torch
  # dependency, so this cannot clobber the CUDA build.
  pip install -r requirements-vlm.txt

  banner "Verify — CUDA build survived, and Qwen3-VL is importable"
  python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("transformers", transformers.__version__)
ok = hasattr(transformers, "Qwen3VLForConditionalGeneration")
print("Qwen3VLForConditionalGeneration present:", ok)
if "cpu" in torch.__version__ or not ok:
    raise SystemExit("STOP: CPU-only torch or missing Qwen3-VL. Recreate the env "
                     "with CONDA_OVERRIDE_CUDA=12.6 / upgrade transformers.")
PY
  echo "(torch.cuda.is_available() is False on a login node — that's expected; "
  echo " the check above is that the build is not the '+cpu' one.)"
  echo "OK. Next: bash hyak_qwen_runbook.sh weights"
  ;;

# ---------------------------------------------------------------------------
weights)
  banner "Pre-download checkpoints to $HF_HOME (login node, not GPU time)"
  cd "$REPO"
  # shellcheck disable=SC1091
  source activate sidewalkcv2
  mkdir -p "$HF_HOME"
  for M in "$QWEN_8B" "${DOWNLOAD_32B:+$QWEN_32B}"; do
    [ -z "$M" ] && continue
    echo "--- $M"
    python - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=[
    "*.json", "*.txt", "*.safetensors", "*.model", "*.py"])
print("cached at", p)
PY
  done
  df -h "$HF_HOME" | tail -1
  echo "OK. Next: bash hyak_qwen_runbook.sh smoke"
  echo "(For 32B later: DOWNLOAD_32B=1 bash hyak_qwen_runbook.sh weights)"
  ;;

# ---------------------------------------------------------------------------
smoke)
  banner "2-pano GPU smoke — proves env + weights + box mapping on the cluster"
  cd "$REPO"
  mkdir -p logs
  INNER="$(mktemp)"
  cat > "$INNER" <<INNER_EOF
set -euo pipefail
source activate sidewalkcv2
export HF_HOME="$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u scripts/model_comparison/compare.py benchmark/richmond \\
    --models rampnet,qwen:$QWEN_8B --limit 2
# Visual check: boxes should sit on the ramps. Copy the contact sheet back and
# look at it before committing to the full run.
python -u scripts/model_comparison/dump_detections.py benchmark/richmond \\
    --model qwen:$QWEN_8B --out view_dump/qwen8b
INNER_EOF
  # salloc + srun (not a heredoc piped into salloc's shell, which is unreliable
  # and non-interactive). HF_HOME is baked into the inner script above.
  salloc -p gpu-l40s --gpus=1 --mem=64G --cpus-per-task=8 --time=1:00:00 \
      srun bash "$INNER" || true
  rm -f "$INNER"
  echo
  echo "Sanity: the rampnet row must read P 0.9-1.0 on those 2 panos (it reads"
  echo "detections from the bundle, so anything else means the bundle is wrong)."
  echo "Then scp view_dump/qwen8b/contact_*.png back and eyeball it."
  echo "OK. Next: bash hyak_qwen_runbook.sh run"
  ;;

# ---------------------------------------------------------------------------
run)
  banner "Full 8B runs (richmond first, then bend)"
  cd "$REPO"
  mkdir -p logs
  sbatch scripts/model_comparison/run_qwen.slurm
  if [ -d benchmark/bend/panos ] && [ "$(ls -A benchmark/bend/panos)" ]; then
    BUNDLE=benchmark/bend sbatch scripts/model_comparison/run_qwen.slurm
  else
    echo "benchmark/bend/panos is empty — submit it once the upload lands:"
    echo "  BUNDLE=benchmark/bend sbatch scripts/model_comparison/run_qwen.slurm"
  fi
  squeue -u "$USER"
  ;;

# ---------------------------------------------------------------------------
run32b)
  banner "32B runs — two GPUs, device_map=auto shards the ~64 GB checkpoint"
  cd "$REPO"
  mkdir -p logs
  QWEN_MODEL="$QWEN_32B" sbatch --gpus=2 scripts/model_comparison/run_qwen.slurm
  QWEN_MODEL="$QWEN_32B" BUNDLE=benchmark/bend sbatch --gpus=2 scripts/model_comparison/run_qwen.slurm
  squeue -u "$USER"
  ;;

# ---------------------------------------------------------------------------
status)
  squeue -u "$USER"
  echo
  for f in $(ls -t "$REPO"/logs/qwen_compare_*.out 2>/dev/null | head -3); do
    echo "--- $f"
    tail -20 "$f"
  done
  echo
  echo "Cached detections so far: $(find "$REPO/.model_cache" -name '*.json' 2>/dev/null | wc -l)"
  ;;

# ---------------------------------------------------------------------------
collect)
  banner "Detections are in $REPO/.model_cache — pull them home"
  find "$REPO/.model_cache" -name '*.json' | wc -l
  echo "From your machine:"
  echo "  rsync -av klone:~/RampNet/.model_cache/ <local RampNet checkout>/.model_cache/"
  echo
  echo "Then locally, no GPU needed (score_model skips the model load when"
  echo "everything is cached):"
  echo "  python scripts/model_comparison/compare.py benchmark/richmond \\"
  echo "      --models rampnet,gemini:gemini-3.6-flash,gemini:gemini-3.1-pro-preview,qwen:$QWEN_8B"
  ;;

# ---------------------------------------------------------------------------
#  Open detectors + pointing models (issue #39). OWLv2 and Grounding DINO are
#  ~1-2 GB and finish in minutes; they emit calibrated scores, so their runs give
#  AP / PR curves / a threshold sweep. Molmo-8B is a text generator and takes
#  hours -- and its wiring has never met real weights, so eyeball it first.
# ---------------------------------------------------------------------------
weightsopen)
  banner "Pre-download the open-detector checkpoints (~2 GB total)"
  cd "$REPO"
  # shellcheck disable=SC1091
  source activate sidewalkcv2
  mkdir -p "$HF_HOME"
  for M in "$OWLV2" "$GDINO" ${DOWNLOAD_MOLMO:+$MOLMO}; do
    echo "--- $M"
    python - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("cached at", snapshot_download(sys.argv[1], allow_patterns=[
    "*.json", "*.txt", "*.safetensors", "*.model", "*.py"]))
PY
  done
  echo "OK. Next: bash hyak_qwen_runbook.sh smokeopen"
  echo "(Molmo too: DOWNLOAD_MOLMO=1 bash hyak_qwen_runbook.sh weightsopen)"
  ;;

# ---------------------------------------------------------------------------
smokeopen)
  banner "2-pano GPU smoke for OWLv2 + Grounding DINO, then a box overlay"
  cd "$REPO"
  mkdir -p logs
  INNER="$(mktemp)"
  cat > "$INNER" <<INNER_EOF
set -euo pipefail
source activate sidewalkcv2
export HF_HOME="$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u scripts/model_comparison/compare.py benchmark/richmond \\
    --models rampnet,owlv2,gdino --limit 2 --sweep
# Boxes should sit on ramps. 0.15 just declutters the overlay.
python -u scripts/model_comparison/dump_detections.py benchmark/richmond \\
    --model owlv2 --score-threshold 0.15 --out view_dump/owlv2
INNER_EOF
  salloc -p gpu-l40s --gpus=1 --mem=64G --cpus-per-task=8 --time=1:00:00 \
      srun bash "$INNER" || true
  rm -f "$INNER"
  echo
  echo "Expect: both models FP-heavy at the 0.05 floor but with high recall, and"
  echo "the sweep showing what tightening the threshold buys. scp the contact"
  echo "sheet back before committing to the full run."
  echo "OK. Next: bash hyak_qwen_runbook.sh runopen"
  ;;

# ---------------------------------------------------------------------------
runopen)
  banner "Full OWLv2 + Grounding DINO runs (minutes, one card)"
  cd "$REPO"
  mkdir -p logs
  sbatch scripts/model_comparison/run_open_models.slurm
  if [ -d benchmark/bend/panos ] && [ "$(ls -A benchmark/bend/panos)" ]; then
    BUNDLE=benchmark/bend sbatch scripts/model_comparison/run_open_models.slurm
  else
    echo "benchmark/bend/panos is empty — submit it once the upload lands:"
    echo "  BUNDLE=benchmark/bend sbatch scripts/model_comparison/run_open_models.slurm"
  fi
  squeue -u "$USER"
  ;;

# ---------------------------------------------------------------------------
runmolmo)
  banner "Molmo — VERIFY THE OVERLAY FIRST, this path has never seen real weights"
  cd "$REPO"
  mkdir -p logs
  # Molmo runs its own Hub code and its card pins transformers==4.57.1; if the
  # load fails on a newer transformers, build it a separate env at that pin
  # rather than downgrading the env the other models use.
  INNER="$(mktemp)"
  cat > "$INNER" <<INNER_EOF
set -euo pipefail
source activate sidewalkcv2
export HF_HOME="$HF_HOME"
python -u scripts/model_comparison/dump_detections.py benchmark/richmond \\
    --model molmo:$MOLMO --out view_dump/molmo
INNER_EOF
  salloc -p gpu-l40s --gpus=1 --mem=64G --cpus-per-task=8 --time=1:00:00 \
      srun bash "$INNER" || true
  rm -f "$INNER"
  echo
  echo "Look at view_dump/molmo/contact_*.png. Red crosshairs must sit ON ramps."
  echo "Nothing detected at all => the coordinate scale is wrong; re-run with"
  echo "  --molmo-coord-scale 100   (or 1000)"
  echo "Only once that looks right:"
  echo "  MODELS=rampnet,molmo:$MOLMO sbatch scripts/model_comparison/run_open_models.slurm"
  ;;

*)
  echo "unknown stage '$STAGE' (env | weights | smoke | run | run32b | status |"
  echo "collect | push | weightsopen | smokeopen | runopen | runmolmo)"
  exit 2
  ;;
esac
