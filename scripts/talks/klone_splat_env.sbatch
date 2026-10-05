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
uv pip install -q numpy pillow tqdm imageio imageio-ffmpeg scipy packaging
uv pip install -q gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121 \
  || { echo "wheel index failed; building gsplat from source";
       export CUDA_HOME=/sw/cuda/12.2.2 PATH=/sw/cuda/12.2.2/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9";
       uv pip install -q --no-build-isolation gsplat; }
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
