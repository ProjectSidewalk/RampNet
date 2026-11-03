#!/usr/bin/env python3
"""The two ``--mask-norm`` choices on a silent miss (#263, review of #266).

Prints the table in docs/sparse_supervision.md, "The normalization is a hyperparameter":
the loss and the summed |gradient| for one ramp the model does not respond to at all
(output 0), under the dense ``nn.MSELoss`` and under the masked loss with each norm, plus
the per-pixel up-weighting that ``weight`` gives a pano with 5 positives and one whose disc
is clipped at the top edge.

The functions are lifted from ``stage_two/train.py`` (importing it would start training),
so these are the numbers the training script itself produces. CPU only, a few seconds.

    python scripts/analysis/mask_norm_263.py
"""
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402
from rampnet import augment as A  # noqa: E402

N = 512 * 1024


def main():
    mod = load_from_train_py(
        "generate_heatmap_from_points", "generate_weight_from_points", "_nearest_dist_sq",
        "masked_mse", "MASK_NORMS", os=os, json=json, math=math, np=np, torch=torch,
        Image=Image, F=F, random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)

    def run(points, norm):
        tgt = mod.generate_heatmap_from_points(points).unsqueeze(0)
        w = mod.generate_weight_from_points(points).unsqueeze(0)
        out = torch.zeros_like(tgt, requires_grad=True)          # a silent miss
        loss = (torch.nn.MSELoss()(out, tgt) if norm == "dense"
                else mod.masked_mse(out, tgt, w, norm))
        loss.backward()
        return loss.item(), out.grad.abs().sum().item(), int(w.sum())

    one = [(0.3, 0.6)]
    dense, pixels, weight = (run(one, n) for n in ("dense", "pixels", "weight"))
    print("| normalization | loss | gradient vs a dense pano with the same miss |")
    print("|---|---|---|")
    print(f"| dense `nn.MSELoss` | {dense[0]:.6f} | 1x |")
    print(f"| sparse, `--mask-norm pixels` (default) | {pixels[0]:.6f} | "
          f"{pixels[1] / dense[1]:.2f}x |")
    print(f"| sparse, `--mask-norm weight` | {weight[0]:.4f} | {weight[1] / dense[1]:.1f}x |")
    print(f"\none positive: disc {weight[2]:,} px, N / sum(w) = {N / weight[2]:.1f}")
    five = run([(0.1 + 0.15 * i, 0.6) for i in range(5)], "weight")
    print(f"5 positives: sum(w) {five[2]:,} px, N / sum(w) = {N / five[2]:.1f}")
    edge = run([(0.3, 0.02)], "weight")
    print(f"one positive at y = 0.02 (disc clipped): {edge[2]:,} px, "
          f"N / sum(w) = {N / edge[2]:.1f}")


if __name__ == "__main__":
    main()
