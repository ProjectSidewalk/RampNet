#!/bin/bash
# Scratch envs for the #48 multi-view 3D arms (docs/crossview_align_48/multiview_3d.md) on
# makelab2 (A40, sm_86). The repo's conda env is not changed.
#
# *** NOT REBUILT OR VERIFIED. *** Written 2026-09-29, after the runs, from the versions the
# committed predictions/*.meta.json record (python 3.12.13, torch 2.6.0+cu124, numpy 2.5.2,
# pycolmap 4.2.0, kornia 0.8.3, cv2 5.0.0) and the code commits the doc lists. The original
# envs were built by hand and their install commands were not recorded, so the install ORDER
# and the editable/--no-deps choices below are a reconstruction. Until someone rebuilds
# from this script and re-predicts one arm byte-identically, treat it as a starting point.
# Known gaps: the exact opencv-python-headless wheel build (meta records cv2.__version__
# 5.0.0 only) and the full sha256 of InstantSplat's MASt3R checkpoint (only e28f91b4...6eb2
# was written down).
#
# Usage:  bash scripts/analysis/crossview_mv3d_setup.sh [main|splat|both]
set -e
B=${MV3D_ROOT:-/homes/gws/jonf/crossview48_sfm}
WHICH=${1:-both}
mkdir -p "$B/src" && cd "$B"
export UV_CACHE_DIR=$B/uv_cache
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="8.6"

clone() {  # clone URL DIR COMMIT
  [ -d "src/$2" ] || git clone -q --recursive "$1" "src/$2"
  git -C "src/$2" checkout -q "$3"
  git -C "src/$2" submodule update -q --init --recursive
  echo "COMMIT $2 $(git -C "src/$2" rev-parse HEAD)"
}

if [ "$WHICH" = main ] || [ "$WHICH" = both ]; then
  [ -d venv ] || uv venv -q -p 3.12.13 venv
  PY=venv/bin/python
  uv pip install -q --python $PY torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
  uv pip install -q --python $PY numpy==2.5.2 pycolmap==4.2.0 kornia==0.8.3 \
      "opencv-python-headless>=5.0.0,<5.0.1" uniception==0.1.7
  clone https://github.com/facebookresearch/vggt.git vggt a288dd0
  clone https://github.com/facebookresearch/map-anything.git map-anything 3d10cf7
  clone https://github.com/Nik-V9/mast3r.git mast3r 6b9f163   # MapAnything-packaged fork
  clone https://github.com/naver/dust3r.git dust3r bb9f9f5
  clone https://github.com/naver/croco.git croco 87244aa
  for d in vggt map-anything mast3r dust3r croco; do
    uv pip install -q --python $PY -e "src/$d" || echo "WARN: editable install of $d failed"
  done
  # re-assert the recorded pins in case a package pulled something else in
  uv pip install -q --python $PY numpy==2.5.2 "opencv-python-headless>=5.0.0,<5.0.1"
  uv pip freeze --python $PY > freeze_main.txt
  $PY - <<'EOF'
import cv2, numpy, pycolmap, kornia, torch
print("cv2", cv2.__version__, "numpy", numpy.__version__, "pycolmap", pycolmap.__version__,
      "kornia", kornia.__version__, "torch", torch.__version__, torch.cuda.is_available())
EOF
fi

if [ "$WHICH" = splat ] || [ "$WHICH" = both ]; then
  [ -d venv_splat ] || uv venv -q -p 3.12.13 venv_splat
  PY=venv_splat/bin/python
  uv pip install -q --python $PY torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
  clone https://github.com/NVlabs/InstantSplat.git InstantSplat b951567
  for sub in simple-knn diff-gaussian-rasterization fused-ssim; do
    uv pip install -q --python $PY --no-build-isolation "src/InstantSplat/submodules/$sub" \
      || echo "WARN: build of $sub failed"
  done
  uv pip install -q --python $PY -r src/InstantSplat/requirements.txt \
      || echo "WARN: InstantSplat requirements.txt install failed"
  uv pip freeze --python $PY > freeze_splat.txt
  echo "InstantSplat loads its MASt3R checkpoint with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1;"
  echo "the committed record has only a truncated sha256 (e28f91b4...6eb2): record the full"
  echo "hash of whatever checkpoint this build downloads."
fi
echo SETUP_DONE
