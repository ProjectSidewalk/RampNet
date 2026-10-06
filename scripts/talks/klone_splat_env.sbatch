#!/bin/bash
#SBATCH --job-name=talk-splat-env
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/gscratch/makelab/jonf/talk_splat/logs/env_%j.out
# Build a Gaussian-splatting environment on klone for the UChicago fly-around: python 3.10,
# torch 2.4.1+cu121 and gsplat from its prebuilt wheel index, falling back to a source build
# against /sw/cuda/12.2.2 if the wheel is missing. Idempotent; re-run to repair.
set -x
R=/gscratch/makelab/jonf/talk_splat
export UV_CACHE_DIR=/gscratch/makelab/jonf/uv_cache UV_PYTHON_INSTALL_DIR=/gscratch/makelab/jonf/uv_python
mkdir -p $R/logs $R/data
cd $R
[ -d venv ] || uv venv -q -p 3.10 venv
. venv/bin/activate
uv pip install -q "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install -q numpy pillow tqdm imageio imageio-ffmpeg scipy packaging setuptools wheel ninja
# No usable prebuilt wheel: the pt24cu121 wheel needs glibc 2.29 and klone is Rocky 8
# (glibc 2.28). Build from source with the gcc under /sw (the system gcc 8.5 is too old for
# torch's headers) against /sw/cuda/12.2.2; gsplat JIT-compiles its kernels on first import.
export PATH=/sw/gcc/11.2.0/bin:/sw/cuda/12.2.2/bin:$PATH CC=/sw/gcc/11.2.0/bin/gcc CXX=/sw/gcc/11.2.0/bin/g++
export LD_LIBRARY_PATH=/sw/gcc/11.2.0/lib64:$LD_LIBRARY_PATH CUDA_HOME=/sw/cuda/12.2.2 TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
python -c "from gsplat import csrc" 2>/dev/null || { uv pip uninstall -q gsplat; uv pip install -q --no-build-isolation "gsplat==1.4.0"; }
python - <<'EOF'
import torch, gsplat
print("torch", torch.__version__, torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("gsplat", gsplat.__version__)
# force the CUDA backend to load (JIT-compiles on first use for source builds)
from gsplat import rasterization
n = 100
means = torch.randn(n, 3, device="cuda")
quats = torch.randn(n, 4, device="cuda")
scales = torch.rand(n, 3, device="cuda") * 0.1
opac = torch.rand(n, device="cuda")
colors = torch.rand(n, 3, device="cuda")
K = torch.tensor([[[500., 0, 256], [0, 500., 256], [0, 0, 1]]], device="cuda")
viewmat = torch.eye(4, device="cuda")[None]
viewmat[0, 2, 3] = 5
out, alpha, info = rasterization(means, quats, scales, opac, colors, viewmat, K, 512, 512)
print("rasterization ok", out.shape, float(out.mean()))
EOF
echo ENV_DONE
