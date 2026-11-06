"""Stage 2 `supervision.negative_points` (#274 hard negatives): verified "not a crosswalk" spots train as background.

Load-bearing: a negative switches the loss on in its own disc (target 0) even where the pano's mode would not (a
sparse pano's background, an ignore point or box); a positive's disc keeps its weight; a sparse pano may carry
negatives and no positive; a negative on a labelled crosswalk is refused; negatives mirror with the flip; and a
pano without negatives is untouched (the #263 and dense_ignore suites guard that path). CPU only, synthetic images.
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
        "generate_dense_ignore_weight", "supervision_ignore_boxes", "supervision_negative_points",
        "add_negative_weight", "NEGATIVE_MIN_DIST",
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


def test_negative_disc_turns_on_sparse_background():
    mod = _lift()
    w = mod.generate_weight_from_points([(0.5, 0.5)], SHAPE)
    w = mod.add_negative_weight(w, [(0.5, 0.5)], [(0.2, 0.5)], SHAPE)[0].numpy()
    assert w[_disc(205, 256)].min() == 1 and w[_disc(512, 256)].min() == 1     # negative and positive discs on
    assert w[~_disc(205, 256) & ~_disc(512, 256)].max() == 0                    # background stays off (sparse)


def test_negative_overrides_ignore_but_not_a_positive():
    mod = _lift()
    w = mod.generate_dense_ignore_weight([(0.5, 0.5)], SHAPE, ignore_points=[(0.2, 0.5)],
                                         ignore_boxes=[(0.3, 0.45, 0.4, 0.6)])
    w = mod.add_negative_weight(w, [(0.5, 0.5)], [(0.2, 0.5), (0.4, 0.5)], SHAPE)[0].numpy()
    assert w[_disc(205, 256)].min() == 1                       # the ignore point's disc trains as background now
    assert w[_disc(410, 256)].min() == 1                       # inside the ignore box too
    assert w[256, 330] == 0                                    # rest of the box stays ignored
    assert w[256, 512] == 1                                    # the positive's peak is untouched


def test_schema_sparse_with_only_negatives_and_refusals():
    mod = _lift()
    rec = {"curb_ramp_points_normalized": [], "supervision": {"mode": "sparse", "negative_points": [[0.3, 0.6]]}}
    assert mod.supervision_mode(rec, []) == ("sparse", [])
    assert mod.supervision_negative_points(rec, []) == [(0.3, 0.6)]
    with pytest.raises(ValueError):                                            # nothing at all to teach
        mod.supervision_mode({"supervision": {"mode": "sparse"}}, [])
    with pytest.raises(ValueError):                                            # on a labelled crosswalk
        mod.supervision_negative_points(rec, [(0.31, 0.62)])
    with pytest.raises(ValueError):                                            # seam-wrapped distance counts
        mod.supervision_negative_points({"supervision": {"negative_points": [[0.995, 0.6]]}}, [(0.01, 0.6)])
    with pytest.raises(ValueError):
        mod.supervision_negative_points({"supervision": {"negative_points": [[0.3]]}}, [])
    assert mod.supervision_negative_points(rec, [(0.36, 0.6)]) == [(0.3, 0.6)]   # 0.06 away: allowed


def test_through_the_dataset_and_the_flip(tmp_path):
    mod = _lift()
    rec = {"curb_ramp_points_normalized": [[0.2, 0.6]],
           "supervision": {"mode": "sparse", "negative_points": [[0.65, 0.6]]}}
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
        neg_col = 666 if cx == 205 else 1024 - 666           # the negative mirrors with the target
        assert w[307, neg_col] == 1 and hm[307, neg_col] == 0 and w[307, cx] == 1 and w[50, 50] == 0
    assert seen == {205, 819}
