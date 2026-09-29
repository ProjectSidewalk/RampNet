"""Compile and smoke-test gsplat's CUDA kernels (first use JIT-compiles them)."""
import time

import torch
from gsplat import rasterization

t0 = time.time()
n = 1000
dev = "cuda"
means = torch.randn(n, 3, device=dev) + torch.tensor([0, 0, 5.0], device=dev)
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
scales = torch.full((n, 3), 0.05, device=dev)
op = torch.full((n,), 0.8, device=dev)
sh0 = torch.rand(n, 1, 3, device=dev)
K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], device=dev)[None]
vm = torch.eye(4, device=dev)[None]
out, alpha, info = rasterization(means, quats, scales, op, sh0, vm, K, 320, 240, sh_degree=0,
                                 render_mode="RGB+ED")
print("ok", out.shape, float(out[..., 3].max()), f"{time.time() - t0:.1f} s")
