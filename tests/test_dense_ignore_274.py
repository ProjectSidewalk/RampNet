"""Stage 2 'dense_ignore' supervision mode (#274, crosswalks): weight maps, flip, schema, off-means-off.

A label-complete crowd-label pano is trained everywhere (it teaches background: lane lines and stop bars are
not crosswalks) except inside ignore discs and ignore boxes, where crowd, SDOT and Vistas disagree or Vistas
sees paint no label explains. Load-bearing: a positive's disc always survives an overlapping ignore region, a
seam-wrapping box masks both edges, a flipped pano mirrors its boxes with the target, and nothing changes
without --loss-mask (the #263 suite's GOLDEN_OFF test still guards that path). CPU only, synthetic images.
"""
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

SHAPE = (512, 1024)
RADIUS = 30


def _lift():
    return load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points", "generate_weight_from_points",
        "generate_dense_ignore_weight", "supervision_ignore_boxes",
        "supervision_mode", "SUPERVISION_MODES", "_point_list", "_nearest_dist_sq",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)


def _disc(cx, cy, shape=SHAPE, r=RADIUS):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _one_pano(root, record, size=(64, 32)):
    split = os.path.join(root, "train")
    os.makedirs(split, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 90, np.uint8)).save(os.path.join(split, "p.png"))
    with open(os.path.join(split, "p.json"), "w") as f:
        json.dump(record, f)
    return root


def _rec(points, ignore_points=(), boxes=()):
    return {"curb_ramp_points_normalized": points,
            "supervision": {"mode": "dense_ignore", "ignore_points": list(map(list, ignore_points)),
                            "ignore_boxes": list(map(list, boxes))}}


def test_no_ignore_is_exactly_dense():
    mod = _lift()
    w = mod.generate_dense_ignore_weight([(0.5, 0.5)], SHAPE)
    assert torch.equal(w, torch.ones(1, *SHAPE))


def test_ignore_point_zeroes_its_disc_but_never_a_positive_peak():
    mod = _lift()
    w = mod.generate_dense_ignore_weight([(0.5, 0.5)], SHAPE, ignore_points=[(0.2, 0.5), (0.52, 0.5)])[0].numpy()
    far = _disc(205, 256)
    assert w[far].sum() == 0 and w[~far & ~_disc(532, 256)].min() == 1     # far ignore disc fully off
    assert w[256, 512] == 1                                                 # the positive's own peak stays
    assert w[256, 540] == 0                                                 # nearer the ignore point: off


def test_box_masks_everything_but_positive_discs():
    mod = _lift()
    w = mod.generate_dense_ignore_weight([(0.5, 0.6)], SHAPE, ignore_boxes=[(0.4, 0.6, 0.5, 0.7)])[0].numpy()
    pos = _disc(512, 307)
    assert w[pos].min() == 1                                  # positive disc survives inside the box
    assert w[260, 420] == 0 and w[350, 600] == 0               # rest of the box is ignored
    assert w[100, 100] == 1 and w[400, 900] == 1               # outside the box: trained


def test_seam_wrapping_box_masks_both_edges():
    mod = _lift()
    w = mod.generate_dense_ignore_weight([], SHAPE, ignore_boxes=[(0.95, 0.05, 0.4, 0.6)])[0].numpy()
    assert w[256, 1020] == 0 and w[256, 10] == 0 and w[256, 512] == 1 and w[100, 1020] == 1


def test_schema_accepts_dense_ignore_and_rejects_bad_boxes():
    mod = _lift()
    assert "dense_ignore" in mod.SUPERVISION_MODES
    assert mod.supervision_mode(_rec([], ignore_points=[(0.1, 0.6)]), []) == ("dense_ignore", [(0.1, 0.6)])
    assert mod.supervision_ignore_boxes(_rec([], boxes=[(0.9, 0.1, 0.5, 0.7)])) == [(0.9, 0.1, 0.5, 0.7)]
    for bad in ([0.1, 0.2, 0.7, 0.5], [0.1, 0.2, 0.5], [0.1, 1.5, 0.5, 0.6], "box"):
        with pytest.raises(ValueError):
            mod.supervision_ignore_boxes({"supervision": {"mode": "dense_ignore", "ignore_boxes": [bad]}})


def test_through_the_dataset_and_the_flip(tmp_path):
    mod = _lift()
    rec = _rec([[0.2, 0.6]], boxes=[(0.6, 0.7, 0.5, 0.7)])
    ds = mod.EquiHeatmapDataset(root_dir=_one_pano(str(tmp_path), rec), split="train", target_heatmap_shape=SHAPE,
                                transform_input=transforms.ToTensor(),
                                points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
                                apply_horizontal_flip=True, loss_mask=True)
    seen = set()
    random.seed(0)
    for _ in range(12):
        _, hm, w = ds[0]
        hm, w = hm.numpy()[0], w.numpy()[0]
        cx = int(np.unravel_index(hm.argmax(), hm.shape)[1])
        seen.add(cx)
        box_col = 665 if cx == 205 else 1024 - 665    # the box's centre column, mirrored with the target
        assert w[307, box_col] == 0 and w[307, cx] == 1 and w[50, 50] == 1
    assert seen == {205, 819}


def test_mask_off_refuses_a_dense_ignore_pano(tmp_path):
    mod = _lift()
    ds = mod.EquiHeatmapDataset(root_dir=_one_pano(str(tmp_path), _rec([[0.5, 0.5]])), split="train",
                                target_heatmap_shape=SHAPE, transform_input=transforms.ToTensor(),
                                points_to_heatmap_transform_fn=mod.generate_heatmap_from_points)
    with pytest.raises(ValueError):
        ds[0]
