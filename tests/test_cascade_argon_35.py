"""Tests for the #35 Argon cascade read (``cascade_argon_35.py``).

CPU only, committed inputs only. The full byte-for-byte regeneration is ``--check`` (about
three minutes, in ``check_all.py --all``); here one pair file and one split of
``argon_pairs.json`` are re-derived, and the headline rows are pinned. The synthetic tests
at the end check behaviour on inputs whose right answer is worked out by hand: the
none/sub/kept split, the blind-GT count, the radius sensitivity, and the pooling.
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
import cascade_argon_35 as ca  # noqa: E402
from rampnet.detection_eval import GroundTruth, radius_sq_for  # noqa: E402

SPLIT = "laurens_mapillary"


@pytest.fixture(scope="module")
def committed():
    with open(os.path.join(ca.OUT_DIR, ca.RESULT), encoding="utf-8") as f:
        return json.load(f)


def _split(res, name):
    return next(r for r in res["splits"] if r["split"] == name)


def test_one_pair_file_reproduces_byte_for_byte():
    assert cc.check(ca.OUT_DIR, [cc.out_name(SPLIT, ca.ARGON[0], cc.T_HI)]) == []


def test_one_split_of_the_comparison_reproduces(committed):
    fresh = json.loads(cc.dumps(ca.compare_split(SPLIT, ca.OUT_DIR)))
    assert fresh == _split(committed, SPLIT)


def test_study_files_are_read_not_duplicated():
    """A pair the study already committed is never re-run into this directory, no Argon
    pair file lands in the study's own directory, and the study's summary.json still lists
    every Argon pair of this note as a gap (so it was not rewritten)."""
    study = set(cc.pair_files(cc.DEFAULT_OUT))
    own = set(ca.own_pair_files(ca.OUT_DIR))
    assert own and not (own & study)
    assert ca.RESULT not in own
    with open(os.path.join(cc.DEFAULT_OUT, "summary.json"), encoding="utf-8") as f:
        gaps = {(g["split"], g["challenger"]) for g in json.load(f)["gaps"]}
    for leg in ca.ARGON:
        name = cc.out_name(SPLIT, leg, cc.T_HI)
        assert name in own and name not in study
        assert (SPLIT, leg) in gaps


def test_baseline_is_the_studys(committed):
    with open(os.path.join(cc.DEFAULT_OUT, cc.out_name(SPLIT, "y26_pano", cc.T_HI)),
              encoding="utf-8") as f:
        study = json.load(f)
    r = _split(committed, SPLIT)
    assert r["baseline"] == study["baseline"]
    assert (r["baseline"]["tp"], r["baseline"]["fp"], r["baseline"]["fn"]) == (131, 17, 118)


def test_headline_rows(committed):
    r = _split(committed, SPLIT)
    f1 = {leg: round(a["best_by_f1"]["F1"], 4) for leg, a in r["arms"].items()}
    assert f1 == {"gemini-4-argon": 0.7775, "gemini-4-argon-confidence-on": 0.7877,
                  "y11l_pano": 0.7868, "y11x_pano_h200": 0.7803, "y26_pano": 0.8018}
    assert r["best_open_weight"]["best_by_f1"] == "y26_pano"
    none = r["argon_only"]["gemini-4-argon"][0]["none"]
    assert (none["boxes"], none["tp"], none["fp"], none["ignored"]) == (36, 17, 18, 1)
    pools = {n: {leg: round(p["arms"][leg]["fixed"]["F1"], 4) for leg in p["arms"]}
             for n, p in committed["pools"].items()}
    assert pools["open_rig"] == {"gemini-4-argon": 0.7243,
                                 "gemini-4-argon-confidence-on": 0.7191,
                                 "y11l_pano": 0.7192, "y11x_pano_h200": 0.7153,
                                 "y26_pano": 0.7162}
    assert committed["pools"]["open_276"]["splits"] == ["laurens_mapillary", "bayonne"]


def test_unique_point_comparison_and_scoreboard_pins(committed):
    """The PR #289 review numbers, as the script computes them (open_rig pool, R)."""
    u = committed["pools_unique_points"]["open_rig"]["R"]
    assert u["n_blind_gt"] == 196
    reached = {leg: {v: o["blind_gt_reached"] for v, o in vs.items()}
               for leg, vs in u["arms"].items()}
    assert reached == {"gemini-4-argon": {"all": 65},
                       "gemini-4-argon-confidence-on": {"all": 95},
                       "y11l_pano": {"all": 44, "conf_ge_0.25": 12},
                       "y11x_pano_h200": {"all": 40, "conf_ge_0.25": 8},
                       "y26_pano": {"all": 54, "conf_ge_0.25": 11}}
    r = _split(committed, SPLIT)
    z = r["unique_points"]
    assert (z["R"]["n_blind_gt"], z["2R"]["n_blind_gt"]) == (36, 23)
    a = {k: (z[k]["arms"]["gemini-4-argon"]["all"]["tp"],
             z[k]["arms"]["gemini-4-argon"]["all"]["fp"]) for k in z}
    assert a == {"R": (17, 18), "2R": (14, 7), "R_group_2R_score": (26, 9)}
    sb = r["scoreboard_reconciliation"]
    assert sb["operating_point"] == 0.55
    assert round(sb["op_cache_at_operating_point"]["F1"], 6) == sb["scoreboard"]["f1"] == 0.543417


# --------------------------------------------------------------------------- #
# behaviour on synthetic inputs (right answers worked out by hand)
# --------------------------------------------------------------------------- #
R = cc.RADIUS
RSQ = radius_sq_for(R)
#: Pano p has two RampNet peaks: 0.40 (kept at T_hi 0.30) at x 0.5 and 0.15 (sub) at x 0.2.
PEAKS = {"p": [(0.5, 0.5, 0.40), (0.2, 0.5, 0.15)], "q": [(0.5, 0.5, 0.40)]}
KEPT = {"p": (0,), "q": (0,)}


def _gts(p_points, q_confirmed=False):
    return {"p": GroundTruth(list(p_points), [], True),
            "q": GroundTruth([(0.3, 0.5)], [], q_confirmed)}


def test_argon_only_splits_boxes_by_the_nearest_peak():
    """Box on the kept peak -> kept, on the sub peak -> sub, far from both -> none. GT at
    the none box and the sub box makes both TPs and new; the kept box is an FP; the
    unconfirmed pano q is skipped entirely."""
    cands = {"p": [(0.5, 0.5), (0.2, 0.5), (0.8, 0.5)], "q": [(0.9, 0.5)]}
    gts = _gts([(0.2, 0.5), (0.8, 0.5)])
    out = ca.argon_only(cands, PEAKS, gts, KEPT, RSQ, 0.05)
    assert out["recall_confirmed_panos"] == 1
    assert {g: (out[g]["boxes"], out[g]["tp"], out[g]["fp"], out[g]["tp_new"])
            for g in ("none", "sub", "kept")} == {
        "none": (1, 1, 0, 1), "sub": (1, 1, 0, 1), "kept": (1, 0, 1, 0)}
    assert out["none"]["precision"] == 1.0 and out["kept"]["precision"] == 0.0
    # A floor above the sub peak's 0.15 removes it, so its box becomes "none".
    hi = ca.argon_only(cands, PEAKS, gts, KEPT, RSQ, 0.20)
    assert (hi["none"]["boxes"], hi["sub"]["boxes"], hi["kept"]["boxes"]) == (2, 0, 1)


def test_argon_only_tp_new_excludes_ramps_the_baseline_matches():
    """A TP is not automatically a ramp the baseline misses: with GT on the kept peak, the
    kept box is a TP whose tp_new is 0."""
    out = ca.argon_only({"p": [(0.5, 0.5)]}, PEAKS, _gts([(0.5, 0.5)]), KEPT, RSQ, 0.05)
    assert (out["kept"]["tp"], out["kept"]["tp_new"]) == (1, 0)


def test_argon_only_wraps_the_seam():
    """A box at x 0.999 is within R of a peak at x 0.001 across the seam -> kept."""
    peaks = {"p": [(0.001, 0.5, 0.40)]}
    out = ca.argon_only({"p": [(0.999, 0.5)]}, peaks, _gts([]), {"p": (0,), "q": ()},
                        RSQ, 0.05)
    assert (out["none"]["boxes"], out["kept"]["boxes"]) == (0, 1)


def test_score_radius_turns_a_loose_box_into_a_tp():
    """A box 1.5 R from a GT ramp and far from every peak: FP at R, TP when rescored at 2R
    with the grouping left at R (the reviewer's read)."""
    gts, cands = _gts([(0.8, 0.5)]), {"p": [(0.8 + 1.5 * R, 0.5)]}
    at_r = ca.argon_only(cands, PEAKS, gts, KEPT, RSQ, 0.05)["none"]
    at_2r = ca.argon_only(cands, PEAKS, gts, KEPT, RSQ, 0.05, radius_sq_for(2 * R))["none"]
    assert (at_r["tp"], at_r["fp"]) == (0, 1)
    assert (at_2r["tp"], at_2r["fp"]) == (1, 0)


def test_blind_gt_and_reached():
    """GT on the kept peak is not blind; GT 1.5 R from it is blind at R but not at 2R; a
    GT far from everything is blind at both. A box within R of a blind ramp reaches it."""
    gts = _gts([(0.5, 0.5), (0.5 + 1.5 * R, 0.5), (0.9, 0.5)])
    b_r = ca.blind_gt(PEAKS, gts, RSQ, 0.05)
    b_2r = ca.blind_gt(PEAKS, gts, radius_sq_for(2 * R), 0.05)
    assert b_r == {"p": [(0.5 + 1.5 * R, 0.5), (0.9, 0.5)]}    # q is unconfirmed: absent
    assert b_2r == {"p": [(0.9, 0.5)]}
    assert ca.blind_reached({"p": [(0.9 + 0.5 * R, 0.5)]}, b_r, RSQ) == 1
    assert ca.blind_reached({"p": [(0.7, 0.5)]}, b_r, RSQ) == 0
    # The floor matters: a 0.06 peak hides a ramp at floor 0.05, not at 0.10.
    pk = {"p": [(0.9, 0.5, 0.06)]}
    assert ca.blind_gt(pk, gts, RSQ, 0.05)["p"] == [(0.5, 0.5), (0.5 + 1.5 * R, 0.5)]
    assert len(ca.blind_gt(pk, gts, RSQ, 0.10)["p"]) == 3


def test_arm_variants_filter_only_yolo_by_confidence():
    cands = {"p": [(0.1, 0.5, 0.9), (0.2, 0.5, 0.1)]}
    y = ca.arm_variants("y26_pano", cands)
    assert set(y) == {"all", "conf_ge_0.25"}
    assert y["conf_ge_0.25"] == {"p": [(0.1, 0.5, 0.9)]}
    assert set(ca.arm_variants("gemini-4-argon", {"p": [(0.1, 0.5)]})) == {"all"}


def _row(split, base, arm_counts):
    """A minimal compare_split row; counts are (tp, fp, fn, n_gt)."""
    def rep(c):
        return dict(zip(("tp", "fp", "fn", "n_gt"), c))
    return {"split": split, "baseline": rep(base), "threshold_only_best": rep(base),
            "arms": {leg: {"best_by_f1": rep(c), "fixed": rep(c)}
                     for leg, c in arm_counts.items()}}


def test_pool_sums_counts_and_picks_the_best_yolo_arm():
    """Two splits; pooled F1 is 2 TP / (2 TP + FP + FN) on summed counts (recall TP = TP
    here). An arm missing on one split pools to None; a pool missing a split is None."""
    legs = ca.ARGON + ca.OPEN_WEIGHT
    a = {leg: (8, 2, 2, 10) for leg in legs}
    b = {leg: (3, 1, 7, 10) for leg in legs}
    a["y26_pano"], b["y26_pano"] = (9, 0, 1, 10), (5, 0, 5, 10)    # the best YOLO arm
    del b["gemini-4-argon"]
    rows = [_row("s1", (8, 2, 2, 10), a), _row("s2", (3, 1, 7, 10), b)]
    p = ca.pool(rows, "x", ("s1", "s2"))
    base = p["baseline"]
    assert (base["tp"], base["fp"], base["fn"], base["n_gt"]) == (11, 3, 9, 20)
    assert base["F1"] == pytest.approx(22 / 34)
    assert base["P"] == pytest.approx(11 / 14) and base["R"] == pytest.approx(0.55)
    assert p["arms"]["y26_pano"]["fixed"]["F1"] == pytest.approx(28 / 34)
    assert p["arms"]["gemini-4-argon"] == {"best_by_f1": None, "fixed": None}
    assert p["best_open_weight_best_by_f1"] == p["best_open_weight_fixed"] == "y26_pano"
    assert ca.pool(rows, "x", ("s1", "s2", "s3")) is None


def test_pool_unique_sums_per_arm_and_variant():
    def u(n_blind, tp, fp, reached):
        o = {"boxes": tp + fp, "tp": tp, "fp": fp, "ignored": 0, "tp_new": tp,
             "precision": None, "blind_gt_reached": reached}
        return {"n_blind_gt": n_blind, "arms": {"y26_pano": {"all": o, "conf_ge_0.25": o}}}
    rows = [{"split": s, "unique_points": {k: u(*v) for k in ca.UNIQUE_CONFIGS}}
            for s, v in (("s1", (4, 3, 1, 2)), ("s2", (6, 1, 3, 1)))]
    r = ca.pool_unique(rows, ("s1", "s2"))["R"]
    assert r["n_blind_gt"] == 10
    y = r["arms"]["y26_pano"]["all"]
    assert (y["tp"], y["fp"], y["blind_gt_reached"], y["precision"]) == (4, 4, 3, 0.5)
    assert r["arms"]["gemini-4-argon"] == {}
