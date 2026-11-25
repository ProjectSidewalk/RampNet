"""Tests for scripts/analysis/merged_separation_263.py (Stage 0a, #263).

CPU only: the trigger logic on synthetic pairs, and a drift guard that regenerates the
committed analysis_out/horizon_band_263/merged_pairs.json and .md from the op_caches.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import merged_separation_263 as ms  # noqa: E402


def _pairs(spec):
    return [{"city": c, "chebyshev": v, "euclidean": e} for c, v, e in spec]


def test_histogram_half_open_bins():
    h = ms.histogram([0.0, 0.99, 1.0, 7.99, 8.0])
    assert h[0] == [0.0, 1.0, 2] and h[1] == [1.0, 2.0, 1]
    assert h[7] == [7.0, 8.0, 1] and h[8] == [8.0, 9.0, 1]
    assert ms.histogram([]) == []


def test_arm_c_needs_more_than_one_third_under_8px():
    # exactly one third under 8 px: not triggered (the rule is "more than")
    t = ms.triggers(_pairs([("a", 7.9, 9.0), ("a", 9.0, 9.0), ("a", 10.0, 10.0)]))
    assert t["chebyshev"]["under_8px"] == 1 and not t["chebyshev"]["arm_c_triggered"]
    t = ms.triggers(_pairs([("a", 7.9, 7.9), ("a", 7.0, 9.0), ("a", 10.0, 10.0)]))
    assert t["chebyshev"]["arm_c_triggered"]
    assert not t["euclidean"]["arm_c_triggered"]          # metrics are evaluated separately
    assert t["chebyshev"]["under_8px"] == 2 and t["euclidean"]["under_8px"] == 1


def test_a_pair_exactly_8px_apart_is_not_under_8px():
    # the real data comes within 0.014 px of the line (7.9861 and 8.0319), so pin it
    t = ms.triggers(_pairs([("a", 8.0, 8.0), ("a", 7.999, 8.0), ("a", 10.0, 10.0)]))
    assert t["chebyshev"]["under_8px"] == 1 and t["euclidean"]["under_8px"] == 0
    assert t["chebyshev"]["between_8_and_12px"] == 2           # 8.0 and 10.0


def test_arm_a_counts_12px_or_more_outside_paterson_only():
    spec = [("bend", 12.0, 12.0)] * 29 + [("paterson", 20.0, 20.0)] * 10 + [("bend", 11.99, 13)]
    t = ms.triggers(_pairs(spec))
    assert t["chebyshev"]["ge_12px_outside_paterson"] == 29
    assert not t["chebyshev"]["arm_a_runs"]
    assert t["euclidean"]["ge_12px_outside_paterson"] == 30 and t["euclidean"]["arm_a_runs"]
    assert t["chebyshev"]["ge_12px_all"] == 39


def test_committed_outputs_reproduce():
    js, md = ms.outputs()
    with open(ms.OUT_JSON, encoding="utf-8", newline="") as f:
        assert f.read() == js
    with open(os.path.splitext(ms.OUT_JSON)[0] + ".md", encoding="utf-8", newline="") as f:
        assert f.read() == md
