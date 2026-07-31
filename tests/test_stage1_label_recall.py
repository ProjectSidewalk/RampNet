"""Unit tests for E1, the Stage-1 label-ceiling experiment (#59).

Pure logic only — no network, no Hub fetch, no cache on disk. The load-bearing
guarantees: both curves are computed over an identical population (so a coverage
difference cannot masquerade as a recall difference), the drop-off ignores tiny
tail buckets (the gold set has 4 ramps beyond 40 m, and letting them anchor the
comparison inverted its sign), and the pre-registered verdict is applied to the
numbers rather than chosen after seeing them.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_label_recall as e1  # noqa: E402


# --------------------------------------------------------------------------- #
# geom — shared with size_analysis.py, so the curves land on the same axis
# --------------------------------------------------------------------------- #
def test_geom_is_monotone_and_reciprocal():
    near, far = e1.geom(0.75), e1.geom(0.55)
    assert near[0] < far[0]            # lower in frame => closer
    assert near[1] > far[1]            # closer => larger apparent size
    # Apparent size is inversely proportional to distance.
    d, px = e1.geom(0.7)
    assert abs(px * d - e1.RAMP_W * e1.PX_PER_RAD) < 1e-6


def test_geom_clamps_at_the_horizon():
    # At or above the horizon the flat-ground model diverges; it must not blow up.
    assert e1.geom(0.5)[0] == 150.0
    assert e1.geom(0.3)[0] == 150.0


def test_bucket_of_is_half_open():
    b = [(0, 8), (8, 12)]
    assert e1.bucket_of(0, b) == (0, 8)
    assert e1.bucket_of(7.999, b) == (0, 8)
    assert e1.bucket_of(8, b) == (8, 12)
    assert e1.bucket_of(99, b) is None


# --------------------------------------------------------------------------- #
# hit_indices — "was this ramp found?" via the shared matcher
# --------------------------------------------------------------------------- #
def test_exact_hits_claim_their_ramps():
    gold = [(0.2, 0.7), (0.8, 0.7)]
    assert e1.hit_indices(list(gold), gold) == {0, 1}


def test_far_away_predictions_claim_nothing():
    assert e1.hit_indices([(0.05, 0.55)], [(0.9, 0.9)]) == set()


def test_one_prediction_cannot_claim_two_ramps():
    # Two coincident gold ramps, one prediction -> exactly one is credited.
    gold = [(0.5, 0.7), (0.5, 0.7)]
    assert len(e1.hit_indices([(0.5, 0.7)], gold)) == 1


def test_no_predictions_means_no_hits():
    assert e1.hit_indices([], [(0.5, 0.7)]) == set()


# --------------------------------------------------------------------------- #
# build_rows — both curves must share one population
# --------------------------------------------------------------------------- #
def test_panos_missing_from_either_source_are_excluded():
    gold = {"a": [(0.5, 0.7)], "b": [(0.5, 0.7)], "c": [(0.5, 0.7)]}
    stage1 = {"a": [(0.5, 0.7)], "b": [(0.5, 0.7)]}       # c absent
    model = {"a": [(0.5, 0.7)], "c": [(0.5, 0.7)]}        # b absent
    rows = e1.build_rows(gold, stage1, model)
    assert {r["pano"] for r in rows} == {"a"}


def test_negative_panos_contribute_no_rows():
    # The gold set's 207 empty label files are real negatives, not missing data.
    rows = e1.build_rows({"a": []}, {"a": []}, {"a": []})
    assert rows == []


def test_rows_record_each_source_independently():
    gold = {"p": [(0.5, 0.75)]}
    rows = e1.build_rows(gold, {"p": [(0.5, 0.75)]}, {"p": []})
    assert len(rows) == 1 and rows[0]["stage1"] is True and rows[0]["model"] is False


# --------------------------------------------------------------------------- #
# recall_table / dropoff
# --------------------------------------------------------------------------- #
def _rows(spec):
    """spec: [(dist, stage1_hit, model_hit), ...] -> row dicts."""
    return [{"pano": "p", "x": 0.5, "y": 0.7, "dist": d, "px": 100.0,
             "stage1": s, "model": m} for d, s, m in spec]


def test_recall_table_counts_both_sources_per_bucket():
    rows = _rows([(1, True, True), (1, True, False),
                  (30, False, False), (30, True, False)])
    t = e1.recall_table(rows, "dist", [(0, 8), (25, 40)])
    assert t[0]["n"] == 2 and t[0]["stage1_recall"] == 1.0 and t[0]["model_recall"] == 0.5
    assert t[1]["n"] == 2 and t[1]["stage1_recall"] == 0.5 and t[1]["model_recall"] == 0.0
    assert abs(t[0]["gap"] - 0.5) < 1e-9


def test_dropoff_ignores_undersized_tail_buckets():
    # Mirrors the real failure: a 4-ramp 40m+ bucket at recall 1.0 inverted the
    # sign of Stage-1's drop-off and would have decided the experiment on n=4.
    table = [{"lo": 0, "hi": 8, "n": 1000, "stage1_recall": 0.96, "model_recall": 0.94},
             {"lo": 25, "hi": 40, "n": 113, "stage1_recall": 0.78, "model_recall": 0.49},
             {"lo": 40, "hi": 1e9, "n": 4, "stage1_recall": 1.0, "model_recall": 0.25}]
    d = e1.dropoff(table, min_n=30)
    assert d["far"] == (25, 40)                    # not the n=4 bucket
    assert abs(d["stage1"] - 0.18) < 1e-9
    assert abs(d["model"] - 0.45) < 1e-9
    # With no floor, the tiny bucket flips Stage-1's sign.
    assert e1.dropoff(table, min_n=0)["stage1"] < 0


def test_dropoff_is_nan_without_two_populated_buckets():
    d = e1.dropoff([{"lo": 0, "hi": 8, "n": 5, "stage1_recall": 1.0,
                     "model_recall": 1.0}], min_n=30)
    assert math.isnan(d["stage1"]) and math.isnan(d["model"])


# --------------------------------------------------------------------------- #
# verdict — #59's pre-registered decision rule
# --------------------------------------------------------------------------- #
def _table(s_near, s_far, m_near, m_far):
    return [{"lo": 0, "hi": 8, "n": 500, "stage1_recall": s_near, "model_recall": m_near},
            {"lo": 25, "hi": 40, "n": 500, "stage1_recall": s_far, "model_recall": m_far}]


def test_verdict_mirrors_when_labels_fall_off_as_steeply():
    v = e1.verdict(_table(0.95, 0.50, 0.95, 0.50))
    assert v.startswith("MIRRORS")
    assert "contraindicated" in v


def test_verdict_flat_when_labels_hold_up_and_the_model_does_not():
    # The observed case: Stage-1 drops 0.18, the model 0.46.
    v = e1.verdict(_table(0.959, 0.779, 0.943, 0.487))
    assert v.startswith("FLAT")
    assert "MODEL/RESOLUTION" in v


def test_verdict_is_inconclusive_without_enough_data():
    assert e1.verdict([{"lo": 0, "hi": 8, "n": 5, "stage1_recall": 1.0,
                        "model_recall": 1.0}]).startswith("INCONCLUSIVE")


def test_tolerance_band_is_what_separates_the_two_calls():
    # Stage-1 drops 0.30, model 0.40 -> within tol=0.15, so still "inherited".
    assert e1.verdict(_table(0.95, 0.65, 0.95, 0.55), tol=0.15).startswith("MIRRORS")
    # Tighten the band and the same numbers read as flat.
    assert e1.verdict(_table(0.95, 0.65, 0.95, 0.55), tol=0.05).startswith("FLAT")
