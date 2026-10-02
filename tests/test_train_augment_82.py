"""Stage 2 augmentation flags (#82): off means off, and arms stay paired by seed.

Three properties are load-bearing, and each fails silently on the cluster:

1. **With no ``--aug`` flag the dataset output is bit-identical to the published
   recipe's.** ``GOLDEN_OFF`` below is the sha256 of the lifted ``EquiHeatmapDataset``'s
   output on synthetic PNGs, computed from ``stage_two/train.py`` *before* the
   augmentation hook was added (origin/main 459ea9e). If a later edit changes what the
   dataset returns with augmentation off -- even by one pixel -- this hash moves.
2. **Augmentation draws never touch the global ``random`` stream.** The horizontal flip
   uses ``random.random()``; if an augmentation consumed the same stream, an
   augmented arm would see a different flip sequence from its same-seed control and the
   arms would no longer be paired.
3. **An augmentation draw is a pure function of (aug seed, epoch, sample index)**, so a
   job that resumes from ``latest_checkpoint.pth`` re-draws exactly what an
   uninterrupted run would have drawn for the same sample.

CPU only, synthetic images, no network, no checkpoint.
"""
import hashlib
import io
import json
import math
import os
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from torchvision import transforms  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
from check_lr_schedule_135 import load_from_train_py  # noqa: E402

from rampnet import augment as A  # noqa: E402

#: sha256 over (image bytes, heatmap bytes) for two passes over the synthetic set with
#: random.seed(42), computed with the ORIGINAL EquiHeatmapDataset (origin/main 459ea9e,
#: before #82 touched train.py). Do not regenerate this to make a failure go away: a
#: changed hash means the published recipe's data path changed.
GOLDEN_OFF = "3d3cbe4949b938374c0a00cfc068c3bf58c72430be57bf2931c85f9e451526ef"


def _lift():
    return load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)


def _make_split(root, n=6, size=(128, 64)):
    """Deterministic synthetic PNG panos (PNG, not JPEG: no decoder-version drift)."""
    split = os.path.join(root, "train")
    os.makedirs(split)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(split, f"p{i}.png"))
        pts = [[float(rng.random()), float(rng.random())] for _ in range(i % 3)]
        with open(os.path.join(split, f"p{i}.json"), "w") as f:
            json.dump({"curb_ramp_points_normalized": pts}, f)
    return root


def _digest(ds, passes=2, seed=42):
    random.seed(seed)
    h = hashlib.sha256()
    for _ in range(passes):
        for i in range(len(ds)):
            img, hm = ds[i]
            h.update(img.numpy().tobytes())
            h.update(hm.numpy().tobytes())
    return h.hexdigest()


def _dataset(mod, root, **kw):
    return mod.EquiHeatmapDataset(
        root_dir=root, split="train", target_heatmap_shape=(16, 32),
        transform_input=transforms.ToTensor(),
        points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
        apply_horizontal_flip=True, **kw)


def test_augmentation_off_is_bit_identical_to_the_published_dataset(tmp_path):
    mod = _lift()
    root = _make_split(str(tmp_path))
    assert _digest(_dataset(mod, root)) == GOLDEN_OFF
