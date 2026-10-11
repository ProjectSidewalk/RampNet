#!/bin/bash
#SBATCH --job-name=mv48-setup
#SBATCH --account=makelab
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/scrubbed/jfroehli/mv48/logs/setup_%j.out
# #48 klone offload: build the two envs from makelab2's exact freezes (torch from the
# cu126 wheel index), clone RampNet at REF, and pull the four HF models. Runs on a compute
# node so nothing heavy lands on a login node.
# REF defaults to 4cef05c, the commit the 2026-09-27 run used (its logs print it). It has
# the same compare.py and benchmark/richmond_neighbourhood as later commits on this PR.
# Override with: REF=<commit> sbatch setup_mv48.slurm
# Paths: W, UV and --output are Jon's; --account is his lab's. Change them for your own.
set -euo pipefail
W=${W:-/gscratch/scrubbed/jfroehli/mv48}
REF=${REF:-4cef05c}
export UV_CACHE_DIR=$W/uv_cache UV_LINK_MODE=copy HF_HOME=$W/hf
UV=${UV:-/mmfs1/home/jfroehli/.local/bin/uv}
cd $W
[ -d RampNet ] || git clone -q https://github.com/ProjectSidewalk/RampNet.git
git -C RampNet checkout -q "$REF"
git -C RampNet log --oneline -1
IDX="--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match"
[ -x envs/eval/bin/python ] || { $UV venv -q --python 3.12.13 envs/eval && $UV pip install -q --python envs/eval/bin/python $IDX -r req_eval_cu126.txt; }
[ -x envs/molmo/bin/python ] || { $UV venv -q --python 3.11.15 envs/molmo && $UV pip install -q --python envs/molmo/bin/python $IDX -r req_molmo.txt; }
for e in eval molmo; do envs/$e/bin/python -c "import sys,torch,transformers;print('$e',sys.version.split()[0],torch.__version__,transformers.__version__)"; done
envs/eval/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["allenai/Molmo2-8B","Qwen/Qwen3-VL-8B-Instruct","google/owlv2-large-patch14-ensemble","IDEA-Research/grounding-dino-base"]:
    print(r, snapshot_download(r))
PY
du -sh $W/hf $W/envs/*
echo SETUP_DONE
