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
        rows = opc.classify_predictions(preds, gt, RSQ)
        outcomes = [r[3] for r in rows]
        assert outcomes.count("tp") == ps.tp
        assert outcomes.count("fp") == ps.fp
        assert outcomes.count("ignore") == ps.ignored
        assert len(outcomes) == len(preds)
        # redundancy is a property of FPs only — a TP claimed its ramp, an ignore
        # is out of scoring entirely.
        assert all(not r[4] for r in rows if r[3] != "fp")


def test_redundant_flags_second_hit_on_one_ramp():
    """Two predictions on a single GT ramp: the winner is a TP, the loser is an FP
    that must be marked redundant — not treated as a ramp the GT missed."""
    gt = GroundTruth([(0.5, 0.5)], [], fn_confirmed=True)
    preds = [(0.5, 0.5, 0.9), (0.505, 0.5, 0.4)]      # second is well inside the radius
    rows = opc.classify_predictions(preds, gt, RSQ)
    assert [r[3] for r in rows] == ["tp", "fp"]
    assert rows[0][4] is False and rows[1][4] is True
    # and a genuinely isolated FP is not flagged
    rows = opc.classify_predictions([(0.5, 0.5, 0.9), (0.1, 0.9, 0.4)], gt, RSQ)
    assert [r[3] for r in rows] == ["tp", "fp"]
    assert rows[1][4] is False


def test_duplicate_risk_buckets():
    assert opc.duplicate_risk(True, 0.5) == "redundant"
    assert opc.duplicate_risk(True, None) == "redundant"   # redundancy wins
    assert opc.duplicate_risk(False, 1.4) == "near"
    assert opc.duplicate_risk(False, 2.5) == "mid"
    assert opc.duplicate_risk(False, 9.0) == "isolated"
    assert opc.duplicate_risk(False, None) == "isolated"   # no GT on the pano


def test_visible_radii_and_draft_width():
    """The crop must reach past the duplicate zone, and the decode must not upscale."""
    assert abs(opc.visible_radii(0.08) - 1.818) < 0.01     # the old default: too small
    assert opc.visible_radii(opc.DEFAULT_CROP_FRAC) > opc.DUP_MID_R
    # a 1024px crop cut from 15% of the pano needs a >=6827px decode
    assert opc.draft_width_for(0.15, 1024) == 6827


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
    # far from the only GT ramp, so it is a clean A candidate
    assert items[0]["dup_risk"] == "isolated"
    assert items[0]["redundant"] is False


def test_incremental_fps_carry_duplicate_geometry():
    """A second hit on a labelled ramp must reach the reviewer flagged, since the
    crop alone cannot show it."""
    gt = GroundTruth([(0.5, 0.5)], [], fn_confirmed=True)
    panos = [{"pano": "p", "preds": [(0.5, 0.5, 0.9), (0.508, 0.5, 0.3)], "gt": gt}]
    items = opc.incremental_fps(panos, op_threshold=0.25, upper=0.55, radius_sq=RSQ)
    assert len(items) == 1
    it = items[0]
    assert it["redundant"] is True and it["dup_risk"] == "redundant"
    assert it["d_gt_r"] < 1.0                       # inside the match radius
    assert it["neighbour_outcome"] == "tp"          # the neighbour that claimed the ramp


def test_corrected_precision_band():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    tags = {"a": "A", "b": "A", "c": "B"}
    res = opc.corrected_precision(tp=10, fp=5, items=items, tags=tags)
    assert abs(res["uncorrected"] - 10 / 15) < 1e-9
    assert abs(res["corrected"] - 12 / 15) < 1e-9      # 2 As promoted to TP
    # fully tagged A/B -> nothing left to be uncertain about, so the band is a point
    assert abs(res["band_high"] - 12 / 15) < 1e-9
    assert abs(res["ceiling_all_real"] - 13 / 15) < 1e-9
    assert res["n_A"] == 2 and res["n_B"] == 1 and res["n_U"] == 0
    assert res["n_tagged"] == 3 and res["n_untagged"] == 0 and res["n_incremental"] == 3


def test_unsure_and_untagged_widen_the_band():
    """'Unsure' must widen the reported band rather than silently counting as B."""
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    res = opc.corrected_precision(tp=10, fp=5, items=items,
                                  tags={"a": "A", "b": "B", "c": "U"})  # d untagged
    assert abs(res["corrected"] - 11 / 15) < 1e-9          # only the confirmed A
    assert abs(res["band_high"] - 13 / 15) < 1e-9          # + the unsure + the untagged
    assert res["n_U"] == 1 and res["n_untagged"] == 1


def test_corrected_recall_moves_both_numerator_and_denominator():
    """An A is a ramp the GT lacked: the model found it (numerator) and it should
    always have been findable (denominator). Correcting only precision is the bug."""
    items = [{"id": "a", "fn_confirmed": True}, {"id": "b", "fn_confirmed": True},
             {"id": "c", "fn_confirmed": True}]
    res = opc.corrected_recall(tp_recall=80, n_gt_recall=100, items=items,
                               tags={"a": "A", "b": "B", "c": "U"})
    assert abs(res["uncorrected"] - 0.80) < 1e-9
    assert abs(res["corrected"] - 81 / 101) < 1e-9      # one A on both sides
    assert abs(res["band_high"] - 82 / 102) < 1e-9      # + the unsure
    assert res["n_A_recall"] == 1 and res["n_A_unscanned"] == 0
    # correcting must raise recall, never lower it (tp_recall <= n_gt_recall)
    assert res["corrected"] > res["uncorrected"]


def test_corrected_recall_ignores_unscanned_panos():
    """Recall is computed over recall-confirmed panos only, so an A on an unscanned
    pano corrects precision but must leave recall untouched."""
    items = [{"id": "a", "fn_confirmed": False}, {"id": "b", "fn_confirmed": True}]
    res = opc.corrected_recall(80, 100, items, {"a": "A", "b": "B"})
    assert abs(res["corrected"] - 0.80) < 1e-9     # unchanged
    assert res["n_A_recall"] == 0 and res["n_A_unscanned"] == 1


def test_A_on_a_duplicate_is_flagged_not_silently_credited():
    items = [{"id": "a", "dup_risk": "near"}, {"id": "b", "dup_risk": "isolated"}]
    res = opc.corrected_precision(tp=10, fp=5, items=items, tags={"a": "A", "b": "A"})
    assert res["n_A"] == 2
    assert res["n_A_suspect"] == 1     # the 'near' one is probably a second hit


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
