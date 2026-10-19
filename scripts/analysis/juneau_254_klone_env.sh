#!/bin/bash
# The klone envs of the #254 Juneau run, as built on a login node on 2026-10-19. The
# /gscratch/scrubbed purge had removed the old conda envs' interpreters and every cached HF
# weight, so they were rebuilt with uv. Exact package versions: analysis_out/juneau_254/envs/.
#
#   rampnet-eval   klone's system python 3.12   extract, parity, YOLO, OWLv2, Grounding DINO
#   rampnet-eval2  uv-managed CPython 3.12.13   Qwen3-VL-8B, Qwen3-VL-32B
#   molmo3         uv-managed CPython 3.12.13   Molmo2-8B (transformers 4.57.1)
#
# klone's system python has no Python.h, so Triton cannot compile its CUDA launcher there and
# Molmo and Qwen fail on every pano (jobs 42034545 and the qwen step of 42034543). Build the
# VLM envs on a uv-managed CPython, which ships the headers. Caches go on scratch because the
# /mmfs1/home quota is small (10 GB) and was full.
#
# This installs version RANGES (unpinned torch on the cu126 index, transformers==5.15.* /
# ==4.57.1, ultralytics==8.4.*). The exact versions of the 2026-10-19 run are the freezes in
# analysis_out/juneau_254/envs/; to rebuild those exactly, after `uv venv` install from the
# freeze instead of the two `uv pip install` lines:
#   uv pip install --python <env>/bin/python --extra-index-url https://download.pytorch.org/whl/cu126 \
#       -r analysis_out/juneau_254/envs/<env>.txt
# On 2026-10-19 the managed CPython 3.12.13 was already installed under ~/.local/share/uv; this
# script installs it if absent and resolves its path, so it also runs on a fresh account.
set -euo pipefail
echo "HOST: $(hostname)"
S=/gscratch/scrubbed/$USER
export UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/cache/uv-python XDG_CACHE_HOME=$S/cache/xdg
UV=${UV:-$HOME/.local/bin/uv}
if [ -z "${MANAGED:-}" ]; then
  $UV python install 3.12.13
  MANAGED=$($UV python find --managed-python 3.12.13)
fi
echo "managed CPython: $MANAGED"
COMMON="timm scikit-image scipy pandas matplotlib tqdm pillow numpy huggingface_hub accelerate einops shapely pyproj geopy requests opencv-python-headless"

mk() {  # name, python, extra pip specs
  local d=$S/envs/$1; shift; local py=$1; shift
  if [ ! -x "$d/bin/python" ] || ! "$d/bin/python" -c "import torch" 2>/dev/null; then
    rm -rf "$d"
    $UV venv --python "$py" "$d"
    $UV pip install --python "$d/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu126
    $UV pip install --python "$d/bin/python" $COMMON "$@"
  fi
  "$d/bin/python" -c "import sys,torch,transformers;print('$d', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__)"
}
mk rampnet-eval 3.12 "transformers==5.15.*" "ultralytics==8.4.*"
mk rampnet-eval2 "$MANAGED" "transformers==5.15.*" "ultralytics==8.4.*"
mk molmo3 "$MANAGED" "transformers==4.57.1"
for e in rampnet-eval2 molmo3; do
  $S/envs/$e/bin/python -c "import sysconfig,os;p=os.path.join(sysconfig.get_paths()['include'],'Python.h');print('$e Python.h', os.path.exists(p))"
done
echo ENVS_OK
