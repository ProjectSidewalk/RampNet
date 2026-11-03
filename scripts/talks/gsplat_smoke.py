"""Smoke test for a gsplat install: one rasterization call. On a source install this is where
the CUDA kernels get JIT-compiled (minutes), so run it once before any training job."""
import torch
import gsplat
from gsplat import rasterization

print("torch", torch.__version__, torch.version.cuda, "gpu", torch.cuda.get_device_name(0), flush=True)
print("gsplat", gsplat.__version__, flush=True)
n = 100
means = torch.randn(n, 3, device="cuda")
quats = torch.randn(n, 4, device="cuda")
scales = torch.rand(n, 3, device="cuda") * 0.1
opac = torch.rand(n, device="cuda")
colors = torch.rand(n, 3, device="cuda")
K = torch.tensor([[[500.0, 0, 256], [0, 500.0, 256], [0, 0, 1]]], device="cuda")
viewmat = torch.eye(4, device="cuda")[None]
viewmat[0, 2, 3] = 5
out, alpha, info = rasterization(means, quats, scales, opac, colors, viewmat, K, 512, 512)
print("rasterization ok", tuple(out.shape), float(out.mean()), flush=True)
