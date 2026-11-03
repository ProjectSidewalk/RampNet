#!/bin/bash
# Scratch env for the flat-Mapillary 3D experiment (#214) on makelab2. Pinned versions;
# the repo's conda env is not changed. Run once:  bash setup_makelab2.sh
set -e
B=${FLAT3D_ROOT:-/homes/gws/jonf/flat3d}
mkdir -p "$B" && cd "$B"
export UV_CACHE_DIR=$B/uv_cache
[ -d venv ] || uv venv -q -p 3.12 venv
. venv/bin/activate
uv pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q numpy==2.2.6 scipy==1.15.3 opencv-python-headless==4.10.0.84 pillow==11.2.1 \
    kornia==0.8.3 pycolmap==4.2.0 plyfile==1.1 requests==2.32.3 ninja==1.11.1.4 jaxtyping==0.3.2 \
    rich==14.0.0
# gsplat compiles its CUDA kernels on first use (A40 = sm_86)
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="8.6"
uv pip install -q gsplat==1.5.3
python - <<'EOF'
import torch, pycolmap, kornia, gsplat
print("torch", torch.__version__, torch.cuda.is_available(), "pycolmap", pycolmap.__version__,
      "kornia", kornia.__version__, "gsplat", gsplat.__version__)
print("pycolmap cuda:", getattr(pycolmap, "has_cuda", "unknown"))
EOF
echo SETUP_DONE
