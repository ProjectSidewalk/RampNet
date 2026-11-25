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
    assert w[256, 512] == 1      # the positive's peak is untouched (weights only rise, so this alone does not
    #                              pin the d_neg < d_pos term: test_a_negative_never_unmasks_pixels_nearer_a_positive)


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


def test_min_distance_halves_y_and_sits_at_the_scorer_radius():
    """The refusal boundary itself: 0.05 in matcher units (x in pano widths, y counted half), the scorer's match
    radius. Brackets it within 0.0005 on x and checks that y counts half."""
    mod = _lift()
    assert mod.NEGATIVE_MIN_DIST == 0.05
    neg = {"supervision": {"negative_points": [[0.3, 0.6]]}}
    assert mod.supervision_negative_points(neg, [(0.3505, 0.6)]) == [(0.3, 0.6)]     # 0.0505 away: allowed
    with pytest.raises(ValueError):
        mod.supervision_negative_points(neg, [(0.3495, 0.6)])                        # 0.0495 away: refused
    with pytest.raises(ValueError):                                                  # dy 0.08 counts 0.04
        mod.supervision_negative_points(neg, [(0.3, 0.68)])
    assert mod.supervision_negative_points(neg, [(0.3, 0.702)]) == [(0.3, 0.6)]      # dy 0.102 counts 0.051


def test_negatives_apply_on_a_dense_ignore_pano(tmp_path):
    """root_v3 has 4 negatives on dense_ignore panos. A negative inside one of the pano's ignore boxes must train
    as background there (weight 1); the rest of the box stays ignored."""
    mod = _lift()
    rec = {"curb_ramp_points_normalized": [[0.2, 0.6]],
           "supervision": {"mode": "dense_ignore", "ignore_boxes": [[0.6, 0.7, 0.5, 0.7]],
                           "negative_points": [[0.65, 0.6]]}}
    ds = mod.EquiHeatmapDataset(root_dir=_one_pano(str(tmp_path), rec), split="train", target_heatmap_shape=SHAPE,
                                transform_input=transforms.ToTensor(),
                                points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
                                apply_horizontal_flip=False, loss_mask=True)
    _, _, w = ds[0]
    w = w.numpy()[0]
    assert w[307, 666] == 1                    # the negative's own spot, inside the box
    assert w[307, 700] == 0                    # elsewhere in the box, outside the negative's disc
    assert w[50, 50] == 1                      # dense background


def test_a_negative_never_unmasks_pixels_nearer_a_positive():
    """The `d_neg < d_pos` term of add_negative_weight. It matters only where an ignore point sits between a
    positive and a negative: the pixels there that are nearer the positive (inside its disc, where the target is
    the positive's Gaussian) but ignored because the ignore point is nearer still must stay ignored. Columns, at
    1024 px: positive 500, ignore point 525, negative 555 (0.0537 from the positive, so the schema allows it)."""
    mod = _lift()
    pos, ign, neg = [(500 / 1024, 0.5)], [(525 / 1024, 0.5)], [(555 / 1024, 0.5)]
    w = mod.generate_dense_ignore_weight(pos, SHAPE, ignore_points=ign)
    assert w[0, 256, 527] == 0                 # ignored: nearer the ignore point than the positive
    w = mod.add_negative_weight(w, pos, neg, SHAPE)[0].numpy()
    assert w[256, 527] == 0                    # 27 px from the positive, 28 from the negative: stays ignored
    assert w[256, 545] == 1                    # nearer the negative: trains as background
