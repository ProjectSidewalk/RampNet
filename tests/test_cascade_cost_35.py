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
# the committed primary artifact, re-derived in full
# --------------------------------------------------------------------------- #
# ``fresh`` is the primary pair exactly as committed: the full 60-setting grid, all 123
# null shifts, both 2,000-resample bootstraps and the calibration -- about 10 s on the
# Windows desktop. It is byte-compared against the committed file, so a regression in
# the null loop, ``attributable_dR``, the union code, either bootstrap, the calibration,
# the ceiling matching or ``verdict_of``'s wiring fails here, not silently.
OUT_DIR = os.path.join(REPO, "analysis_out", "cascade_cost_35")


@pytest.fixture(scope="module")
def committed_text():
    with open(PRIMARY, encoding="utf-8", newline="") as f:
        return f.read()


@pytest.fixture(scope="module")
def committed(committed_text):
    return json.loads(committed_text)


@pytest.fixture(scope="module")
def fresh(richmond):
    gts, peaks, _ = richmond
    payload, run = cc.run_pair("richmond", VISTAS, gts=gts, peaks=peaks)
    assert set(run) == {"elapsed_s", "host", "python"}
    return payload


def test_primary_artifact_reproduces_byte_for_byte(committed_text, fresh):
    assert cc.dumps(fresh) == committed_text


def test_volatile_fields_are_not_in_the_payload(committed):
    assert "elapsed_s" not in committed and "host" not in committed
    assert committed["args"] == {"t_lo": list(cc.T_LO), "r_gate": list(cc.R_GATE),
                                 "c_min": "auto", "null": "shift", "null_shifts": "all",
                                 "seed": 0, "bootstrap": cc.BOOTSTRAP}


def test_naive_union_both_conventions(fresh):
    u = fresh["naive_union"]
    c, a = u["complementarity"], u["aggregate"]
    assert round(c["F1"], 3) == 0.549                       # docs/model_comparison.md
    assert (c["tp"], c["fp"], c["rampnet_fp"], c["challenger_fp"]) == (295, 470, 28, 442)
    assert round(a["F1"], 4) == 0.4632 and (a["tp"], a["fp"]) == (302, 692)
    # 0.30-point price of a recovered ramp, each convention against the same baseline
    assert c["fp_per_recovered_ramp"] == pytest.approx((470 - 28) / (295 - 257))
    assert a["fp_per_recovered_ramp"] == pytest.approx((692 - 28) / (302 - 257))


def test_union_gap_is_dedup_and_reassignment_not_pano_sets(richmond):
    """Every richmond pano is fn_confirmed, so the two conventions see the same panos;
    the gap is the merged list's second hits (+222 FP) and greedy reassignment (+7 TP)."""
    gts, peaks, cands = richmond
    assert all(g.fn_confirmed for g in gts.values()) and len(gts) == 124
    assert sum(1 for g in gts.values() if g.gt_points) == 92
    rsq = radius_sq_for(0.022)
    kept = {p: cc.threshold_preds(peaks.get(p, []), 0.30) for p in gts}
    sep_tp = sep_fp = sep_ig = mer_ig = 0
    for p, g in gts.items():
        r = cc.score_pano(kept[p], g, rsq)
        c = cc.score_pano(list(cands.get(p, [])), g, rsq)
        m = cc.score_pano(cc.union_preds(kept[p], cands.get(p, [])), g, rsq)
        sep_tp += r.tp + c.tp
        sep_fp += r.fp + c.fp
        sep_ig += r.ignored + c.ignored
        mer_ig += m.ignored
    assert (sep_tp, sep_fp) == (257 + 274, 470)
    # 531 separate hits -> 302 merged TPs: 229 lost, 222 of them to FP and 7 to ignored
    assert 692 - sep_fp == 222 and mer_ig - sep_ig == 7


def test_verdict_rederived_on_the_fresh_grid(fresh):
    assert cc.verdict_of(fresh["baseline"], fresh["threshold_only"], fresh["grid"]) \
        == fresh["verdict"] == "VIABLE"
    assert fresh["null"] == {"mode": "shift", "shifts": 123, "seed": None}
    bv = fresh["best_viable"]
    assert (bv["t_lo"], bv["r_gate"], bv["tp"], bv["fp"], bv["fn"]) == (0.05, 0.011, 269, 38, 41)
    assert bv["attributable_dR"] == pytest.approx(0.03572, abs=1e-5)
    assert fresh["best_by_f1"]["attributable_dR"] == pytest.approx(0.017912, abs=1e-5)
    assert len(fresh["viable_rows"]) == 4


def test_selected_rows_are_copies(fresh):
    """Annotating the selected rows must not write into the grid (review item 9)."""
    assert not any("threshold_only_at_matched_recall" in r for r in fresh["grid"])
    assert fresh["best_viable"]["threshold_only_at_matched_recall"]["t"] == 0.15


def test_selection_aware_bootstrap(fresh):
    """``bootstrap_selection_aware`` itself asserts that the vectorised rule with every
    pano weighted once reproduces the in-sample pick and verdict; ``fresh`` ran it."""
    sa = fresh["bootstrap_selection_aware"]
    assert sa["resamples"] == cc.BOOTSTRAP and 0 < sa["viable_frac"] <= 1
    assert sum(sa["verdict_counts"].values()) == cc.BOOTSTRAP
    assert sa["dF1_ci95"][0] >= 0            # a viable row never has F1 below baseline
    cond = fresh["bootstrap_vs_baseline"]["best_viable"]
    assert cond["conditioning"].startswith("conditional on the in-sample selection")


def test_calibration_wrong_pano_challengers(fresh):
    c = fresh["calibration"]
    assert c["challengers"] == 123 and c["exact"] is True
    assert sum(c["verdicts"].values()) == 123
    assert c["verdicts"]["VIABLE"] == 0 and c["false_viable_rate"] == 0.0
    assert c["max_attr_dR_max"] < c["real_max_attr_dR"]


def test_ceiling_crosscheck_recomputed(richmond, fresh):
    gts, peaks, cands = richmond
    c = fresh["ceiling"]
    assert (c["promotable"], c["handoff_ge_t_hi"], c["no_peak_in_radius"]) == (19, 4, 15)
    assert c["best_viable_gained_in_promotable"] <= c["best_viable_gained_ramps"]
    # recount the gained ramps independently of run_pair
    bv = fresh["best_viable"]
    rsq = radius_sq_for(0.022)
    gained = 0
    for p, g in gts.items():
        pk = peaks.get(p, [])
        before = cc.matched_gt(cc.threshold_preds(pk, 0.30), g.gt_points, rsq)
        preds, _ = cc.cascade_preds(pk, cc.filter_cands(cands.get(p, []), bv["c_min"]),
                                    0.30, bv["t_lo"], radius_sq_for(bv["r_gate"]))
        gained += len(cc.matched_gt(preds, g.gt_points, rsq) - before)
    assert gained == c["best_viable_gained_ramps"] == c["best_viable_gained_in_promotable"] == 12


def test_null_none_is_a_gap_not_a_crash(richmond, tmp_path):
    gts, peaks, _ = richmond
    p, _ = cc.run_pair("richmond", VISTAS, null="none", gts=gts, peaks=peaks, bootstrap=0)
    assert p["verdict"] is None and p["calibration"] is None
    assert p["bootstrap_selection_aware"] is None
    assert cc.verdict_of(p["baseline"], p["threshold_only"], p["grid"]) is None
    cc.write_json(str(tmp_path / cc.out_name("richmond", VISTAS, cc.T_HI)), p)
    out = cc.summary(str(tmp_path), str(tmp_path / "summary.json"), quiet=True)
    assert out["rows"] == [] and out["verdict_counts"] == {}
    assert any(g.get("challenger") == VISTAS and "--null none" in g["reason"]
               for g in out["gaps"])


def test_summary_rederives_the_committed_summary(tmp_path):
    out = tmp_path / "summary.json"
    cc.summary(OUT_DIR, str(out), quiet=True)
    with open(os.path.join(OUT_DIR, "summary.json"), encoding="utf-8", newline="") as f:
        assert out.read_text(encoding="utf-8") == f.read()
