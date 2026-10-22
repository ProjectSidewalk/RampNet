"""Pure helpers of the #258 step 2 scorer and read-out (no GPU, no imagery)."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))

import validator_auc_258 as va  # noqa: E402
import validator_score_258 as vs  # noqa: E402


def brute_auc(score, positive):
    pos = [s for s, p in zip(score, positive) if p]
    neg = [s for s, p in zip(score, positive) if not p]
    wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_average_ranks_ties():
    assert va.average_ranks([3, 1, 3, 2]).tolist() == [3.5, 1.0, 3.5, 2.0]


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 6, 40).astype(float)   # many ties
    positive = rng.random(40) < 0.4
    assert va.auc(score, positive) == pytest.approx(brute_auc(score, positive))


def test_auc_degenerate():
    assert np.isnan(va.auc([0.1, 0.2], [True, True]))
    assert va.auc([0.9, 0.1], [True, False]) == 1.0
    assert va.auc([0.1, 0.9], [True, False]) == 0.0


def test_prf():
    p, r, f = va.prf(3, 1, 6)
    assert (p, r) == (0.75, 0.5)
    assert f == pytest.approx(0.6)
    assert va.prf(0, 0, 5)[2] == 0.0


def test_cluster_bootstrap_resamples_whole_panos():
    clusters = [np.array([0, 1]), np.array([2])]
    seen = []

    def stat(idx):
        seen.append(sorted(idx.tolist()))
        return np.array([len(idx)])

    va.cluster_bootstrap(clusters, stat, b=50, seed=1)
    # Every draw is a union of whole clusters: rows 0 and 1 always travel together.
    assert all(s.count(0) == s.count(1) for s in seen)


def test_crop_box_matches_serving_arithmetic():
    # 2048x2048 perspective, depth 10 m: half = int(0.1 * 6100) = 610.
    assert vs.crop_box(10.0, 2048, 2048) == (414, 414, 1634, 1634, 610)
    # Near the camera the crop clips to the image.
    assert vs.crop_box(2.0, 2048, 2048) == (0, 0, 2048, 2048, 3050)


def test_resize_dims_long_side_640():
    assert vs.resize_dims(1220, 1220) == (640, 640)
    assert vs.resize_dims(2048, 1000) == (640, int(640 * 1000 / 2048))
    assert vs.resize_dims(1000, 2048) == (int(640 * 1000 / 2048), 640)


def test_perspective_center_params():
    assert vs.get_perspective_center_params_from_equi_point(4096, 2048, 8192, 4096) == (0.0, 0.0)
    theta, phi = vs.get_perspective_center_params_from_equi_point(6144, 3072, 8192, 4096)
    assert (theta, phi) == pytest.approx((90.0, -45.0))
