"""Unit tests for the cross-split low-floor sweep core (issue #54).

Pure logic only — no GPU, no panos, no cache on disk. The load-bearing guarantees:
the sweep's per-threshold rows agree with the shared scorer, the parity gate
actually fails when preprocessing drifts, and monotone quantities stay monotone.
"""
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

from rampnet.detection_eval import (  # noqa: E402
    GroundTruth, aggregate, radius_sq_for, score_pano)

import low_floor_sweep as lfs  # noqa: E402

RSQ = radius_sq_for()


def _pano(rng, pid, n_gt=3):
    gt_pts = [(rng.uniform(0.05, 0.95), rng.uniform(0.5, 0.9)) for _ in range(n_gt)]
    gt = GroundTruth(gt_pts, [], fn_confirmed=True)
    preds = []
    for gx, gy in gt_pts:                       # near-hits with varied confidence
        preds.append((gx + rng.uniform(-0.01, 0.01), gy + rng.uniform(-0.005, 0.005),
                      rng.random()))
    for _ in range(rng.randint(0, 4)):          # scatter
        preds.append((rng.random(), rng.uniform(0.5, 0.95), rng.random()))
    return {"pano": pid, "preds": preds, "gt": gt, "tier": "gsv", "city": "test"}


def _panos(seed=0, n=12):
    rng = random.Random(seed)
    return [_pano(rng, f"p{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def test_threshold_grid_is_sorted_unique_and_within_bounds():
    grid = lfs.threshold_grid(0.05, 0.90)
    assert grid[0] == 0.05 and grid[-1] == 0.90
    assert len(grid) == len(set(grid))
    assert all(b > a for a, b in zip(grid, grid[1:]))
    assert all(0.05 <= v <= 0.90 for v in grid)


def test_threshold_grid_is_fine_in_the_candidate_band_only():
    """0.20-0.50 at 0.01; the tails stay coarse. This is the whole point of the grid."""
    grid = lfs.threshold_grid(0.05, 0.90)
    for v in (0.21, 0.27, 0.33, 0.49):
        assert any(abs(v - g) < 1e-9 for g in grid), f"{v} missing from fine band"
    for v in (0.06, 0.13, 0.62, 0.87):
        assert not any(abs(v - g) < 1e-9 for g in grid), f"{v} should not be swept"


def test_threshold_grid_degenerate_range():
    assert lfs.threshold_grid(0.9, 0.5) == []
    assert lfs.threshold_grid(0.5, 0.5) == [0.5]


# --------------------------------------------------------------------------- #
# tiers
# --------------------------------------------------------------------------- #
def test_tier_of_covers_every_benchmark_rig():
    assert lfs.tier_of("Trimble", "Trimble mx7", "mapillary") == "survey"
    assert lfs.tier_of("Trimble", "Trimble MX7", "mapillary") == "survey"   # case variant
    assert lfs.tier_of("NCTECH LTD", "iSTAR Pulsar", "mapillary") == "pro360"
    assert lfs.tier_of("GoPro", "GoPro Max", "mapillary") == "action-modern"
    assert lfs.tier_of("GoPro", "Fusion", "mapillary") == "action-legacy"
    assert lfs.tier_of("GoPro", "GoPro Fusion FS1.04.01.80.00", "mapillary") == "action-legacy"
    assert lfs.tier_of(None, None, "launch") == "gsv"


def test_tier_of_does_not_invent_a_tier_for_absent_provenance():
    """'none'/None must land in `unknown`, never be folded into a neighbouring tier."""
    assert lfs.tier_of("none", "none", "mapillary") == "unknown"
    assert lfs.tier_of(None, None, "mapillary") == "unknown"
    assert lfs.tier_of("LG Electronics", "LG-R105", "mapillary") == "unknown"


def test_gsv_source_wins_over_camera_fields():
    """A GSV pano is GSV regardless of what camera metadata happens to be attached."""
    assert lfs.tier_of("GoPro", "GoPro Max", "gsv") == "gsv"


# --------------------------------------------------------------------------- #
# sweep rows
# --------------------------------------------------------------------------- #
def test_sweep_row_matches_the_shared_scorer():
    """A swept row must equal what score_pano/aggregate say at that threshold."""
    panos = _panos()
    grid = [0.1, 0.3, 0.55, 0.8]
    rows = lfs.sweep_rows(panos, grid, RSQ)
    for row in rows:
        expected = aggregate([
            score_pano([p for p in pd["preds"] if p[2] >= row["threshold"]],
                       pd["gt"], radius_sq=RSQ)
            for pd in panos])
        assert row["tp"] == expected.tp
        assert row["fp"] == expected.fp
        assert row["fn"] == expected.fn
        assert math.isclose(row["precision"], expected.precision)
        assert math.isclose(row["recall"], expected.recall)


def test_recall_is_monotone_non_increasing_in_threshold():
    rows = lfs.sweep_rows(_panos(seed=3), lfs.threshold_grid(0.05, 0.90), RSQ)
    recalls = [r["recall"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(recalls, recalls[1:]))


def test_density_counts_ignored_detections():
    """dets_per_pano is review burden, so an `unsure`-covered detection still counts."""
    gt = GroundTruth([], [(0.5, 0.6)], fn_confirmed=True)
    panos = [{"pano": "a", "preds": [(0.5, 0.6, 0.9)], "gt": gt}]
    row = lfs.sweep_rows(panos, [0.1], RSQ)[0]
    assert row["tp"] == 0 and row["fp"] == 0 and row["ignored"] == 1
    assert row["dets_per_pano"] == 1.0


def test_best_f1_breaks_ties_toward_the_higher_threshold():
    rows = [{"threshold": 0.2, "f1": 0.5}, {"threshold": 0.4, "f1": 0.5},
            {"threshold": 0.6, "f1": 0.4}]
    assert lfs.best_f1_row(rows)["threshold"] == 0.4


def test_highest_threshold_meeting_picks_the_lowest_qualifying_point():
    """Recall-first: among points clearing the precision floor, take the lowest."""
    rows = [{"threshold": 0.2, "precision": 0.80}, {"threshold": 0.3, "precision": 0.92},
            {"threshold": 0.4, "precision": 0.95}]
    assert lfs.highest_threshold_meeting(rows, 0.92)["threshold"] == 0.3
    assert lfs.highest_threshold_meeting(rows, 0.99) is None


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_calibration_bins_partition_every_scored_prediction():
    panos = _panos(seed=7)
    edges = [round(0.05 + i * 0.05, 4) for i in range(20)]
    bins = lfs.confidence_calibration(panos, RSQ, edges)
    binned = sum(b["n"] for b in bins)
    expected = sum(r["tp"] + r["fp"] for r in lfs.sweep_rows(panos, [edges[0]], RSQ))
    assert binned == expected


def test_calibration_excludes_ignored_predictions():
    gt = GroundTruth([(0.2, 0.6)], [(0.8, 0.6)], fn_confirmed=True)
    panos = [{"pano": "a", "preds": [(0.2, 0.6, 0.9), (0.8, 0.6, 0.7)], "gt": gt}]
    bins = lfs.confidence_calibration(panos, RSQ, [0.05, 0.5, 1.0])
    assert sum(b["n"] for b in bins) == 1          # the ignore-covered one is dropped
    assert sum(b["n_true"] for b in bins) == 1


def test_calibration_places_a_perfect_score_in_the_last_bin():
    gt = GroundTruth([(0.2, 0.6)], [], fn_confirmed=True)
    panos = [{"pano": "a", "preds": [(0.2, 0.6, 1.0)], "gt": gt}]
    bins = lfs.confidence_calibration(panos, RSQ, [0.05, 0.5, 1.0])
    assert bins[-1]["n_true"] == 1


# --------------------------------------------------------------------------- #
# distance
# --------------------------------------------------------------------------- #
def test_ground_distance_is_monotone_decreasing_below_the_horizon():
    ys = [0.51, 0.55, 0.6, 0.7, 0.8]
    ds = [lfs.ground_distance(y) for y in ys]
    assert all(b < a for a, b in zip(ds, ds[1:]))


def test_ground_distance_at_or_above_horizon_is_infinite():
    assert lfs.ground_distance(0.5) == float("inf")
    assert lfs.ground_distance(0.3) == float("inf")


def test_recall_by_distance_bands_sum_to_the_recall_denominator():
    panos = _panos(seed=11)
    rows = lfs.recall_by_distance(panos, RSQ, 0.2, 0.55)
    total_gt = sum(len(pd["gt"].gt_points) for pd in panos if pd["gt"].fn_confirmed)
    assert sum(r["n_gt"] for r in rows) == total_gt
    assert all(r["gained"] >= 0 for r in rows)   # a lower threshold cannot lose a ramp


# --------------------------------------------------------------------------- #
# parity gate
# --------------------------------------------------------------------------- #
def _records_from(panos, threshold=0.55):
    return {pd["pano"]: {"detections": [
        {"x_normalized": x, "y_normalized": y, "confidence": s}
        for (x, y, s) in pd["preds"] if s >= threshold]} for pd in panos}


def test_parity_passes_and_is_exact_when_the_cache_reproduces_records():
    panos = _panos(seed=5)
    res = lfs.parity_for(panos, _records_from(panos))
    assert res["ok"]
    assert res["n_records"] == res["n_cache"] == res["matched"]
    assert res["exact_frac"] == 1.0
    assert res["max_displacement_r"] == 0.0


def test_parity_tolerates_sub_radius_drift_but_reports_it_as_inexact():
    """The GSV path fed the model a different resample, so peaks shift a cell or two.

    That must still pass — nothing it moved can change a scoring outcome — while the
    exact-match fraction drops, which is the sensitive early warning.
    """
    panos = _panos(seed=6)
    records = _records_from(panos)
    for rec in records.values():
        for d in rec["detections"]:
            d["x_normalized"] += 0.004         # ~4 heatmap cells, ~0.18 R
    res = lfs.parity_for(panos, records)
    assert res["ok"]
    assert res["exact_frac"] == 0.0
    assert 0 < res["max_displacement_r"] <= lfs.PARITY_TOL_RADII


def test_parity_fails_when_detections_move_beyond_a_scoring_radius():
    """The gate has to actually fail, or it is decoration."""
    panos = _panos(seed=8)
    records = _records_from(panos)
    for rec in records.values():
        for d in rec["detections"]:
            d["x_normalized"] = (d["x_normalized"] + 0.25) % 1.0
    res = lfs.parity_for(panos, records)
    assert not res["ok"]


def test_parity_fails_when_many_detections_are_missing():
    panos = _panos(seed=9)
    records = _records_from(panos)
    for rec in records.values():
        if rec["detections"]:
            rec["detections"].pop()
    res = lfs.parity_for(panos, records)
    assert not res["ok"]
    assert res["panos_with_count_mismatch"]


def test_parity_reports_a_count_mismatch_even_when_it_is_within_gate_tolerance():
    """A single dropped detection is recorded per pano, not silently absorbed."""
    panos = _panos(seed=9)
    records = _records_from(panos)
    next(r for r in records.values() if r["detections"])["detections"].pop()
    res = lfs.parity_for(panos, records)
    assert res["panos_with_count_mismatch"]
    assert res["n_cache"] == res["n_records"] + 1


# --------------------------------------------------------------------------- #
# pooling policy
# --------------------------------------------------------------------------- #
def test_pool_holds_out_budapest_and_gold_by_default():
    pool = lfs.pool_of(lfs.ALL_SPLITS)
    assert set(pool) == set(lfs.US_SPLITS)
    assert "budapest_district5" not in pool and "manual_gold" not in pool


def test_pool_flags_opt_the_held_out_splits_back_in():
    assert "budapest_district5" in lfs.pool_of(lfs.ALL_SPLITS, include_budapest=True)
    assert "manual_gold" in lfs.pool_of(lfs.ALL_SPLITS, include_gold=True)


def test_every_held_out_split_carries_a_stated_reason():
    """An omission with no reason is indistinguishable from a withheld result."""
    for split in set(lfs.ALL_SPLITS) - set(lfs.pool_of(lfs.ALL_SPLITS)):
        assert lfs.HELD_OUT.get(split), f"{split} is held out with no documented reason"
