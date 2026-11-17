"""Tests for Stage 0 of the Argon-teacher plan (``argon_teacher_s0.py``).

CPU only. The synthetic tests check each rule on inputs whose answer is worked out by hand:
the 2R ignore disc and the far band (leak H / H2), the empty-pano match (leak E), the snap
and the blind / strict-silent counts (candidate scoring), the LOSO choice, and the three
gate verdicts. The last tests pin the headline numbers of the committed ``results.json``;
the full byte-for-byte rebuild is ``argon_teacher_s0.py --check``.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import argon_teacher_s0 as a  # noqa: E402
from rampnet.detection_eval import GroundTruth  # noqa: E402

R = a.R


def _split(gt_points, peaks=(), zs=(), conf=(), yolo=(), seq="s1", rig="Max", pid="p1",
           confirmed=True):
    """A one-pano split in ``load_split``'s shape."""
    return {"gts": {pid: GroundTruth(list(gt_points), [], confirmed)},
            "peaks": {pid: list(peaks)}, "zs": {pid: list(zs)}, "conf": {pid: list(conf)},
            "yolo": {arm: {pid: list(yolo)} for arm in a.YOLO_ARMS},
            "records": {pid: {"sequence_id": seq, "camera_model": rig}}}


def _merge(*splits):
    out = {"gts": {}, "peaks": {}, "zs": {}, "conf": {}, "records": {},
           "yolo": {arm: {} for arm in a.YOLO_ARMS}}
    for s in splits:
        for k in ("gts", "peaks", "zs", "conf", "records"):
            out[k].update(s[k])
        for arm in a.YOLO_ARMS:
            out["yolo"][arm].update(s["yolo"][arm])
    return out


# ---- leak H / H2 ------------------------------------------------------------------- #
def test_H_disc_is_2R_and_floors_apply():
    # GT at x=0.5; a 0.06 peak 1.5R away covers it (disc is 2R), a 0.04 peak does not.
    covered = _split([(0.5, 0.6)], peaks=[(0.5 + 1.5 * R, 0.6, 0.06)])
    below = _split([(0.5, 0.6)], peaks=[(0.5 + 1.5 * R, 0.6, 0.04)])
    assert a.leak_H(covered)["leaked"] == 0
    assert a.leak_H(below)["leaked"] == 1
    # a YOLO box at 0.04 does not count, at 0.05 it does
    assert a.leak_H(_split([(0.5, 0.6)], yolo=[(0.5, 0.6, 0.04)]))["leaked"] == 1
    assert a.leak_H(_split([(0.5, 0.6)], yolo=[(0.5, 0.6, 0.05)]))["leaked"] == 0
    # an Argon box (no score) counts; the disc wraps round the seam
    assert a.leak_H(_split([(0.005, 0.6)], zs=[(0.995, 0.6, None)]))["leaked"] == 0


def test_H2_far_band_and_neighbour_discs():
    far = _split([(0.5, a.FAR_BAND_Y - 0.001)])          # 5.9 deg below the horizon
    near_ = _split([(0.5, a.FAR_BAND_Y + 0.01)])
    assert a.leak_H(far, far_band=True)["leaked"] == 0
    assert a.leak_H(near_, far_band=True)["leaked"] == 1
    assert a.leak_H(near_, extra={"p1": [(0.5, a.FAR_BAND_Y + 0.01)]},
                    far_band=True)["leaked"] == 0


def test_leak_counts_only_recall_confirmed_panos():
    s = _split([(0.5, 0.6)], confirmed=False)
    assert a.leak_H(s) == {"n_gt": 0, "leaked": 0, "leak": None}


# ---- leak E ------------------------------------------------------------------------ #
def test_E_matches_empties_by_sequence_and_rig():
    labelled = _split([(0.5, 0.6)], peaks=[(0.5, 0.6, 0.9)], pid="lab", seq="A", rig="Max")
    empty_match_ramp = _split([(0.3, 0.6)], pid="e1", seq="A", rig="Max")   # leaks
    empty_match_clean = _split([], pid="e2", seq="A", rig="Max")
    empty_other_rig = _split([(0.3, 0.6)], pid="e3", seq="A", rig="Trimble")  # unmatched
    d = _merge(labelled, empty_match_ramp, empty_match_clean, empty_other_rig)
    e = a.leak_E(d)
    assert (e["n_empty"], e["n_empty_matched"], e["matched_with_ramp"]) == (3, 2, 1)
    assert e["unmatched_with_ramp"] == 1
    assert e["leak"] == pytest.approx(0.5)


def test_empty_needs_all_three_models_silent():
    assert a.is_empty(_split([]), "p1")
    assert not a.is_empty(_split([], zs=[(0.5, 0.6, None)]), "p1")
    assert not a.is_empty(_split([], yolo=[(0.5, 0.6, 0.05)]), "p1")
    assert a.is_empty(_split([], yolo=[(0.5, 0.6, 0.049)]), "p1")
    assert not a.is_empty(_split([], peaks=[(0.5, 0.6, 0.05)]), "p1")


# ---- gate B ------------------------------------------------------------------------ #
def _gate_B(h2_all, h2_have, e):
    return a.gate_B({"pooled": {"H2": {"leak": h2_all},
                                "H2_splits_with_neighbours": {"leak": h2_have},
                                "E": {"leak": e}}})


def test_gate_B_order():
    assert _gate_B(0.04, 0.03, 0.5)["rule_passed"] == "H2"
    g = _gate_B(0.06, 0.07, 0.04)
    assert (g["H2"], g["verdict"], g["rule_passed"]) == ("fail", "go", "E")
    # four-split read fails but the split-with-data read passes: unreadable, fall to E
    g = _gate_B(0.06, 0.04, 0.10)
    assert (g["H2"], g["verdict"]) == ("unreadable", "stop")
    assert _gate_B(0.06, 0.07, None)["verdict"] == "stop"


# ---- features, candidates, scoring -------------------------------------------------- #
THR = {"elev_y": [0.52, 0.70], "conf": {"median": 0.5, "q75": 0.7}}


def test_80_candidates():
    cands, tests = a.candidates(THR)
    assert len(tests) == 9
    assert len(cands) == 80
    assert sum(1 for n, _, _ in cands if n.endswith("+snap")) == 40
    # no pair joins two levels of the same feature
    assert not any(len(c) == 2 and tests[c[0]][0] == tests[c[1]][0] for _, c, _ in cands)


def _row(x, y, **kw):
    r = {"split": "toy", "pano": "p1", "box": 0, "x": x, "y": y, "f1_heatmap_max_R": 0.0,
         "f2_support_m": None, "f2_available": True, "f3_conf_score": None,
         "f4_yolo_2R": False, "f5_elev_in_range": True, "f6_snap": None}
    r.update(kw)
    return r


def test_snap_moves_a_box_onto_the_ramp():
    # GT at (0.5, 0.6). Box 1.5R off; the heatmap mode sits on the ramp. Blind and
    # strict-silent: no peak at all.
    gt = (0.5, 0.6)
    d = {"toy": _split([gt])}
    rows = [_row(0.5 + 1.5 * R, 0.6, f6_snap=[0.5, 0.6, 0.02])]
    _, tests = a.candidates(THR)
    off = a.score_candidate(rows, d, ("f5_elev",), False, tests, "toy")
    on = a.score_candidate(rows, d, ("f5_elev",), True, tests, "toy")
    assert (off["tp"], off["fp"], off["blind_kept"]) == (0, 1, 0)
    assert (on["tp"], on["fp"], on["blind_kept"], on["ss_kept"]) == (1, 0, 1, 1)


def test_blind_but_not_strict_silent():
    # a 0.05 peak 1.5R from the ramp: blind (none within R) but not strict-silent
    gt = (0.5, 0.6)
    d = {"toy": _split([gt], peaks=[(0.5 + 1.5 * R, 0.6, 0.05)])}
    o = a.score_candidate([_row(0.5, 0.6)], d, ("f5_elev",), False, a.candidates(THR)[1], "toy")
    assert (o["tp"], o["blind_kept"], o["ss_kept"]) == (1, 1, 0)


def test_feature_tests_thresholds():
    t = a.feature_tests(THR)
    assert t["f1>=0.02"][1](_row(0, 0, f1_heatmap_max_R=0.02))
    assert not t["f1>=0.03"][1](_row(0, 0, f1_heatmap_max_R=0.02))
    assert t["f2<=3m"][1](_row(0, 0, f2_support_m=3.0))
    assert not t["f2<=3m"][1](_row(0, 0, f2_support_m=None))     # no data fails
    assert t["f3>=median"][1](_row(0, 0, f3_conf_score=0.5))
    assert not t["f3>=q75"][1](_row(0, 0, f3_conf_score=0.6))


def test_quantile_matches_numpy_default():
    assert a.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert a.quantile([0, 10], 0.05) == pytest.approx(0.5)


def test_disc_max_wraps_the_seam():
    np = pytest.importorskip("numpy")
    h = np.zeros((64, 128))
    h[32, 0] = 0.7                       # on the seam's right side of column 0
    assert a.disc_max(h, 127.5 / 128, 32 / 64, 3) == pytest.approx(0.7)
    assert a.disc_max(h, 0.5, 0.5, 3) == 0.0


# ---- gates P / Y / S --------------------------------------------------------------- #
def _counts(ss_reached=40, blind_reached=65):
    per = {"ss_reached": ss_reached // 4, "blind_reached": blind_reached // 4,
           "n_strict_silent": 25}
    return {s: per for s in a.SPLITS}


def _loso(per_split):
    return {s: {"held_out": dict(zip(("boxes", "precision"), per_split[i]))}
            for i, s in enumerate(a.SPLITS)}


def test_gate_P_go_kill_between():
    ok = _loso([(5, 0.8)] * 4)
    pooled = {"tp": 85, "fp": 15, "ignored": 0, "blind_kept": 30, "ss_kept": 10}
    pooled["precision"] = 0.85
    assert a.gates(ok, pooled, _counts())["P"]["verdict"] == "go"
    # one split with 10+ boxes under 0.60 blocks go
    bad = _loso([(12, 0.5), (5, 0.9), (5, 0.9), (5, 0.9)])
    assert a.gates(bad, pooled, _counts())["P"]["verdict"] == "in-between"
    low = dict(pooled, tp=60, fp=40, precision=0.60)
    assert a.gates(ok, low, _counts())["P"]["verdict"] == "kill"
    # ignored boxes counted true lift a 0.60 split above the kill bar
    lifted = dict(low, ignored=20)
    assert a.gates(ok, lifted, _counts())["P"]["verdict"] == "in-between"


def test_gate_Y_and_S():
    loso = _loso([(5, 0.9)] * 4)
    base = {"tp": 9, "fp": 1, "ignored": 0, "precision": 0.9}
    g = a.gates(loso, dict(base, blind_kept=30, ss_kept=10), _counts(ss_reached=40))
    assert (g["Y"]["verdict"], g["S"]["verdict"]) == ("go", "go")
    g = a.gates(loso, dict(base, blind_kept=25, ss_kept=9), _counts(ss_reached=40))
    assert (g["Y"]["verdict"], g["S"]["verdict"]) == ("in-between", "kill")
    g = a.gates(loso, dict(base, blind_kept=19, ss_kept=10), _counts(ss_reached=48))
    assert (g["Y"]["verdict"], g["S"]["verdict"]) == ("kill", "kill")   # 10/48 < 25%


# ---- committed results ------------------------------------------------------------- #
@pytest.fixture(scope="module")
def committed():
    with open(os.path.join(a.OUT_DIR, a.RESULTS), encoding="utf-8") as f:
        return json.load(f)


def test_reproduces_the_plans_counts(committed):
    p = committed["counts"]["pooled"]
    assert (p["n_panos"], p["n_gt"], p["n_blind"], p["blind_reached"]) == (469, 1147, 196, 65)


def test_headline_verdicts(committed):
    assert committed["leak"]["gate_B"]["verdict"] == "go"
    g = committed["filters"]["gates"]
    assert {k: g[k]["verdict"] for k in "PYS"} == HEADLINE


#: The committed run's verdicts (docs/argon_teacher.md, Result).
HEADLINE = {"P": "kill", "Y": "kill", "S": "kill"}


def test_no_candidate_reaches_the_kill_bar_in_sample(committed):
    """The doc's claim that the kill is not a LOSO artefact: pooled over all four splits,
    no candidate reaches precision 0.65."""
    best = max((a._add([per[s] for s in a.SPLITS])["precision"] or 0.0)
               for per in committed["filters"]["candidates"].values())
    assert best < a.P_KILL
