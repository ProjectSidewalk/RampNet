"""Tests for the gated-cascade cost read (#35).

CPU only, committed inputs only: ``benchmark/richmond`` (GT),
``analysis_out/op_cache/richmond.json`` (RampNet floor peaks) and the published
Vistas parity detections. The pure rule is checked on toy inputs; the instrument
(baseline reproduces the committed 0.30 row) and the committed primary artifact are
checked against a fresh re-derivation.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import cascade_cost_35 as cc  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402

VISTAS = "mask2former-vistas-curb-cut-1024x1024"
PRIMARY = os.path.join(REPO, "analysis_out", "cascade_cost_35",
                       f"richmond__{VISTAS}.json")


# --------------------------------------------------------------------------- #
# the pure rule
# --------------------------------------------------------------------------- #
PEAKS = [(0.5, 0.5, 0.9), (0.7, 0.5, 0.15), (0.2, 0.5, 0.15)]
CANDS = [(0.7, 0.5, 0.8)]


def test_toy_promotes_only_the_gated_peak():
    preds, n = cc.cascade_preds(PEAKS, CANDS, 0.30, 0.10, radius_sq_for(0.022))
    assert preds == [PEAKS[0], PEAKS[1]] and n == 1


def test_toy_tight_gate_promotes_nothing():
    near = [(0.705, 0.5, 0.8)]          # 5.1 px from the 0.15 peak at x = 0.7
    assert cc.cascade_preds(PEAKS, near, 0.30, 0.10, radius_sq_for(0.022))[1] == 1
    assert cc.cascade_preds(PEAKS, near, 0.30, 0.10, radius_sq_for(0.001))[1] == 0


def test_toy_t_lo_above_the_peak_promotes_nothing():
    _, n = cc.cascade_preds(PEAKS, CANDS, 0.30, 0.20, radius_sq_for(0.022))
    assert n == 0


def test_gate_wraps_at_the_seam():
    preds, n = cc.cascade_preds([(0.999, 0.5, 0.15)], [(0.001, 0.5, 0.8)],
                                0.30, 0.10, radius_sq_for(0.022))
    assert n == 1 and preds == [(0.999, 0.5, 0.15)]


def test_c_min_filter_keeps_scoreless_boxes():
    assert cc.filter_cands([(0.1, 0.1, 0.2), (0.2, 0.2, None), (0.3, 0.3)], 0.5) == [
        (0.2, 0.2, None), (0.3, 0.3)]


def test_verdict_rule_branches():
    base = {"F1": 0.86}
    thr = {"0.1": {"F1": 0.80}}

    def row(att, f1):
        return {"t_lo": 0.1, "attributable_dR": att, "F1": f1}
    assert cc.verdict_of(base, thr, [row(0.03, 0.87)]) == "VIABLE"
    assert cc.verdict_of(base, thr, [row(0.01, 0.87)]) == "NOT VIABLE"
    assert cc.verdict_of(base, thr, [row(0.03, 0.83)]) == "PARTIAL"
    assert cc.verdict_of(base, {"0.1": {"F1": 0.90}}, [row(0.03, 0.87)]) == "NOT VIABLE"


# --------------------------------------------------------------------------- #
# identities on the real split
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def richmond():
    gts = cc.load_gts("richmond")
    peaks = cc.load_floor_peaks("richmond")
    cands, _ = cc.load_challenger("richmond", VISTAS)
    return gts, peaks, cands


def _score(gts, peaks, fn):
    return cc.score_split(fn, gts, radius_sq_for(0.022))


def test_instrument_baseline_reproduces_the_committed_030_row(richmond):
    gts, peaks, _ = richmond
    r = _score(gts, peaks, lambda pid: cc.threshold_preds(peaks.get(pid, []), 0.30))
    assert (r["tp"], r["fp"], r["fn"]) == (257, 28, 53)
    assert round(r["P"], 4) == 0.9018 and round(r["R"], 4) == 0.8290
    assert round(r["F1"], 4) == 0.8639


def test_shipped_point_reproduces_the_published_row(richmond):
    gts, peaks, _ = richmond
    r = _score(gts, peaks, lambda pid: cc.threshold_preds(peaks.get(pid, []), 0.5519))
    assert (r["tp"], r["fp"], r["fn"]) == (238, 9, 72)


def test_identities(richmond):
    gts, peaks, cands = richmond
    # r_gate -> infinity reduces to threshold-only only where the challenger put at
    # least one box on the pano; on a pano it left empty nothing is promoted. So the
    # identity is checked with one box on every pano.
    one_each = {pid: [(0.5, 0.5, 1.0)] for pid in gts}
    for t_lo in (0.05, 0.15):
        huge = _score(gts, peaks, lambda pid: cc.cascade_preds(
            peaks.get(pid, []), one_each[pid], 0.30, t_lo, 1e12)[0])
        thr = _score(gts, peaks, lambda pid: cc.threshold_preds(peaks.get(pid, []), t_lo))
        assert huge == thr
    base = _score(gts, peaks, lambda pid: cc.threshold_preds(peaks.get(pid, []), 0.30))
    same = _score(gts, peaks, lambda pid: cc.cascade_preds(
        peaks.get(pid, []), cands.get(pid, []), 0.30, 0.30, radius_sq_for(0.044))[0])
    none = _score(gts, peaks, lambda pid: cc.cascade_preds(
        peaks.get(pid, []), [], 0.30, 0.05, radius_sq_for(0.044))[0])
    assert same == base and none == base


def test_c_min_grid(richmond):
    _, _, cands = richmond
    g = cc.c_min_grid(cands)
    assert len(g) == 4 and g == sorted(g) and g[0] == 0.0
    gem, _ = cc.load_challenger("richmond", "gemini-3.6-flash")
    assert cc.c_min_grid(gem) == [0.0]


# --------------------------------------------------------------------------- #
# the committed primary artifact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def committed():
    with open(PRIMARY, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fresh(richmond):
    gts, peaks, _ = richmond
    return cc.run_pair("richmond", VISTAS, null="none", gts=gts, peaks=peaks,
                       bootstrap=0)


def test_naive_union_both_conventions(committed):
    u = committed["naive_union"]
    assert round(u["complementarity"]["F1"], 3) == 0.549     # docs/model_comparison.md
    assert round(u["aggregate"]["F1"], 4) == 0.4632


def test_grid_does_not_drift(committed, fresh):
    keys = ("t_lo", "r_gate", "c_min", "tp", "fp", "fn", "n_promoted", "promoted_tp",
            "promoted_fp", "promoted_ignored")
    assert [{k: r[k] for k in keys} for r in committed["grid"]] == \
           [{k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in keys}
            for r in fresh["grid"]]
    assert committed["baseline"] == cc._round(fresh["baseline"])
    assert committed["threshold_only_best"] == cc._round(fresh["threshold_only_best"])


def test_pinned_null_and_verdict(committed):
    assert committed["verdict"] == "VIABLE"
    assert committed["null"] == {"mode": "shift", "shifts": 123, "seed": None}
    bv = committed["best_viable"]
    assert (bv["t_lo"], bv["r_gate"], bv["tp"], bv["fp"], bv["fn"]) == (0.05, 0.011, 269, 38, 41)
    assert bv["attributable_dR"] == pytest.approx(0.03572, abs=1e-6)
    assert committed["best_by_f1"]["attributable_dR"] == pytest.approx(0.017912, abs=1e-6)
    assert committed["ceiling"]["promotable"] == 19
