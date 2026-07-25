"""Unit tests for the operating-point analysis core (issue #54).

Covers the pure scoring/curve logic (no GPU, no panos): the load-bearing
guarantee is that ``classify_predictions`` reproduces ``score_pano``'s tp/fp/
ignored counts exactly, so the incremental-FP gallery audits the same false
positives the PR curve counts.
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from rampnet.detection_eval import GroundTruth, radius_sq_for, score_pano  # noqa: E402

import operating_point_curve as opc  # noqa: E402

RSQ = radius_sq_for()


def _random_gt(rng):
    gt = [(rng.random(), rng.random()) for _ in range(rng.randint(0, 5))]
    ignore = [(rng.random(), rng.random()) for _ in range(rng.randint(0, 3))]
    return GroundTruth(gt, ignore, fn_confirmed=True)


def _random_preds(rng, gt):
    """A mix of on-GT, near-GT-jittered, on-ignore, and random-scatter preds."""
    preds = []
    for gx, gy in gt.gt_points:
        preds.append((gx + rng.uniform(-0.03, 0.03), gy + rng.uniform(-0.03, 0.03), rng.random()))
    for ix, iy in gt.ignore_points:
        preds.append((ix, iy, rng.random()))
    for _ in range(rng.randint(0, 6)):
        preds.append((rng.random(), rng.random(), rng.random()))
    rng.shuffle(preds)
    return preds


def test_threshold_grid():
    grid = opc._threshold_grid(0.05, 0.05)
    assert grid[0] == 0.05
    assert grid[-1] == 1.0
    assert len(grid) == 20
    assert all(b > a for a, b in zip(grid, grid[1:]))  # strictly increasing


def test_classify_matches_score_pano():
    """Counting classify_predictions outcomes must equal score_pano exactly."""
    rng = random.Random(0)
    for _ in range(300):
        gt = _random_gt(rng)
        preds = _random_preds(rng, gt)
        ps = score_pano(preds, gt, radius_sq=RSQ)
        outcomes = [o for _, _, _, o in opc.classify_predictions(preds, gt, RSQ)]
        assert outcomes.count("tp") == ps.tp
        assert outcomes.count("fp") == ps.fp
        assert outcomes.count("ignore") == ps.ignored
        assert len(outcomes) == len(preds)


def test_sweep_recall_monotone_and_reproduces_floor():
    """Recall is non-increasing in threshold; the floor row matches a direct score."""
    rng = random.Random(1)
    panos = []
    for i in range(8):
        gt = _random_gt(rng)
        panos.append({"pano": f"p{i}", "preds": _random_preds(rng, gt), "gt": gt})
    rows = opc.sweep_operating_points(panos, 0.05, 0.05, RSQ)
    recalls = [r["recall"] for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(recalls, recalls[1:]))
    # the lowest-threshold row keeps every pred >= floor == the full aggregate at that floor
    floor_rep = opc._score_at(panos, 0.05, RSQ)
    assert abs(rows[0]["precision"] - floor_rep.precision) < 1e-9
    assert abs(rows[0]["recall"] - floor_rep.recall) < 1e-9


def test_pr_curve_and_ap_present():
    rng = random.Random(2)
    panos = []
    for i in range(6):
        gt = GroundTruth([(0.5, 0.5), (0.3, 0.7)], [], fn_confirmed=True)
        preds = [(0.5, 0.5, 0.9), (0.3, 0.7, 0.4), (0.1, 0.1, 0.6)]  # 2 tp, 1 fp
        panos.append({"pano": f"p{i}", "preds": preds, "gt": gt})
    rep = opc.pr_curve_and_ap(panos, RSQ)
    assert rep.ap is not None and 0.0 <= rep.ap <= 1.0
    assert rep.pr_curve is not None and len(rep.pr_curve) == 2


def test_incremental_fps_band():
    gt = GroundTruth([(0.5, 0.5)], [], fn_confirmed=True)
    preds = [
        (0.5, 0.5, 0.90),    # tp
        (0.10, 0.10, 0.30),  # fp, in band [0.25, 0.55)
        (0.80, 0.80, 0.70),  # fp, above upper -> excluded
        (0.20, 0.90, 0.10),  # fp, below op -> excluded
    ]
    panos = [{"pano": "pano1", "preds": preds, "gt": gt}]
    items = opc.incremental_fps(panos, op_threshold=0.25, upper=0.55, radius_sq=RSQ)
    assert len(items) == 1
    assert abs(items[0]["score"] - 0.30) < 1e-9
    assert items[0]["pano"] == "pano1"
    assert items[0]["id"] == "pano1_0.1_0.1"


def test_corrected_precision_band():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    tags = {"a": "A", "b": "A", "c": "B"}
    res = opc.corrected_precision(tp=10, fp=5, items=items, tags=tags)
    assert abs(res["uncorrected"] - 10 / 15) < 1e-9
    assert abs(res["corrected"] - 12 / 15) < 1e-9      # 2 As promoted to TP
    assert abs(res["upper_bound"] - 13 / 15) < 1e-9    # if all 3 were real
    assert res["n_A"] == 2 and res["n_tagged"] == 3 and res["n_incremental"] == 3


def test_cache_roundtrip(tmp_path):
    gt = GroundTruth([(0.5, 0.5)], [(0.2, 0.2)], fn_confirmed=True)
    panos = [{"pano": "p1", "preds": [(0.5, 0.5, 0.9), (0.1, 0.1, 0.3)], "gt": gt}]
    path = os.path.join(tmp_path, "richmond.json")
    opc.write_cache(path, "richmond", panos, {"score_floor": 0.05})
    back, meta = opc.read_cache(path)
    assert meta["score_floor"] == 0.05
    assert back[0]["pano"] == "p1"
    assert back[0]["preds"] == [(0.5, 0.5, 0.9), (0.1, 0.1, 0.3)]
    assert back[0]["gt"].gt_points == [(0.5, 0.5)]
    assert back[0]["gt"].ignore_points == [(0.2, 0.2)]
    assert back[0]["gt"].fn_confirmed is True
    # round-tripped GT scores identically
    assert opc._score_at(panos, 0.0, RSQ).tp == opc._score_at(back, 0.0, RSQ).tp
