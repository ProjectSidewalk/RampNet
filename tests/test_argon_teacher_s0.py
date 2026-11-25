"""Tests for Stage 0 of the Argon-teacher plan (``argon_teacher_s0.py``).

CPU only. The synthetic tests check each rule on inputs whose answer is worked out by hand:
the 2R ignore disc and the far band (leak H / H2), the empty-pano match (leak E), the base
counts (blind, strict-silent, reached), the feature table's drops (Argon-only, rig mask,
0.30-within-2R) and features, the snap and the blind / strict-silent counts (candidate
scoring), the leave-one-split-out choice end to end through ``filters()`` on four toy
splits where the in-sample best and the held-out choice differ, the selection tie-breaks,
and the three gate verdicts at their bars. The last tests pin the headline numbers of the
committed ``results.json``; the full byte-for-byte rebuild is ``argon_teacher_s0.py --check``.
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


def test_far_band_is_6_degrees_and_stays_in_the_denominator():
    # absolute positions, so the test also pins the 6 deg band itself
    five = _split([(0.5, 0.5 + 5 / 180)])     # 5 deg below the horizon: in the band
    seven = _split([(0.5, 0.5 + 7 / 180)])    # 7 deg: outside it, nothing covers it
    assert a.leak_H(five, far_band=True) == {"n_gt": 1, "leaked": 0, "leak": 0.0}
    assert a.leak_H(seven, far_band=True) == {"n_gt": 1, "leaked": 1, "leak": 1.0}
    assert a.far_band_count(_merge(five, _split([(0.5, 0.6)], pid="p2"))) == 1


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


def test_kept_ramp_with_a_peak_within_R_is_not_blind():
    gt = (0.5, 0.6)
    d = {"toy": _split([gt], peaks=[(0.5 + 0.5 * R, 0.6, 0.9)])}
    o = a.score_candidate([_row(0.5, 0.6)], d, ("f5_elev",), False, a.candidates(THR)[1], "toy")
    assert (o["tp"], o["blind_kept"], o["ss_kept"]) == (1, 0, 0)


def test_pairs_are_and():
    d = {"toy": _split([(0.5, 0.6)])}
    _, tests = a.candidates(THR)
    row = _row(0.5, 0.6, f4_yolo_2R=True, f5_elev_in_range=False)
    assert a.score_candidate([row], d, ("f4_yolo", "f5_elev"), False, tests, "toy")["boxes"] == 0
    assert a.score_candidate([row], d, ("f4_yolo",), False, tests, "toy")["tp"] == 1
    # the OR read (amendment 5) keeps it
    o = a.score_candidate([row], d, ("f4_yolo", "f5_elev"), False, tests, "toy", join=any)
    assert o["tp"] == 1


def test_match_radius_2R_read():
    # a box 1.5R from the ramp: FP at R, TP (and a kept blind ramp) at 2R
    d = {"toy": _split([(0.5, 0.6)])}
    _, tests = a.candidates(THR)
    rows = [_row(0.5 + 1.5 * R, 0.6)]
    at_r = a.score_candidate(rows, d, ("f5_elev",), False, tests, "toy")
    at_2r = a.score_candidate(rows, d, ("f5_elev",), False, tests, "toy", match_r_sq=a.R2_SQ)
    assert (at_r["tp"], at_r["fp"], at_r["blind_kept"]) == (0, 1, 0)
    assert (at_2r["tp"], at_2r["fp"], at_2r["blind_kept"], at_2r["ss_kept"]) == (1, 0, 1, 1)


def test_candidate_table_scores_at_R_by_default(monkeypatch):
    # the filters() path: a box 1.5R from the ramp is an FP unless 2R is asked for
    monkeypatch.setattr(a, "SPLITS", ("toy",))
    d = {"toy": _split([(0.5, 0.6)])}
    cands, tests = a.candidates(THR)
    rows = [_row(0.5 + 1.5 * R, 0.6)]
    one = [c for c in cands if c[0] == "f5_elev"]
    assert (a.candidate_table(d, rows, one, tests)["f5_elev"]["toy"]["fp"]) == 1
    assert (a.candidate_table(d, rows, one, tests, match_r_sq=a.R2_SQ)["f5_elev"]["toy"]["tp"]) == 1


# ---- base counts -------------------------------------------------------------------- #
def test_base_counts_blind_strict_silent_reached(monkeypatch):
    # g0: nothing near, Argon box 0.5R off -> blind, strict-silent, reached (both)
    # g1: a 0.05 peak 1.5R off -> blind, not strict-silent; Argon box 1.5R off -> not reached
    # g2: a 0.9 peak 0.5R off -> not blind
    # g3: nothing near, no box -> blind, strict-silent, not reached
    g = [(0.1, 0.6), (0.3, 0.6), (0.5, 0.6), (0.7, 0.6)]
    d = _merge(_split(g, peaks=[(0.3 + 1.5 * R, 0.6, 0.05), (0.5 + 0.5 * R, 0.6, 0.9)],
                      zs=[(0.1 + 0.5 * R, 0.6, None), (0.3 - 1.5 * R, 0.6, None)]),
               _split([(0.5, 0.6)], pid="unconfirmed", confirmed=False))
    c = a.base_counts(d)
    assert c == {"n_panos": 2, "n_confirmed": 1, "n_gt": 4, "n_gt_all_panos": 5,
                 "n_blind": 3, "blind_reached": 1, "n_strict_silent": 2, "ss_reached": 1}
    # within 2R, g1's box 1.5R off reaches it too
    monkeypatch.setattr(a, "SPLITS", ("toy",))
    assert a.reached_within({"toy": d}, a.R2_SQ) == {"blind_reached": 2, "ss_reached": 1}


# ---- the feature table ---------------------------------------------------------------- #
def _toy_feature_inputs():
    """One pano, five zero-shot boxes (x positions 0.1 ... 0.9, so no box sees another's
    peak): b0 kept, with every feature set; b1 kept although a 0.05 peak sits 1.5R off
    (Argon-only is "no peak within R"); b2 below the rig mask; b3 1.5R from a 0.35 peak
    (dropped); b4 0.5R from a 0.06 peak (not Argon-only)."""
    ys = [0.6, 0.6, a.NADIR_MASK_Y + 0.01, 0.6, 0.6]
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    zs = [(x, y, None) for x, y in zip(xs, ys)]
    peaks = [(0.3 + 1.5 * R, 0.6, 0.05), (0.7 + 1.5 * R, 0.6, 0.35), (0.9 + 0.5 * R, 0.6, 0.06)]
    d = _split([(0.1, 0.6)], peaks=peaks, zs=zs,
               conf=[(0.1 + 0.5 * R, 0.6, 0.8), (0.3 + 3 * R, 0.6, 0.9)],
               yolo=[(0.1 + 1.5 * R, 0.6, 0.06), (0.3 + 3 * R, 0.6, 0.9)])
    nb = {"splits": {"toy": {"available": True,
                             "panos": {"p1": {"argon_support_m": [2.0, 7.0, None, 1.0, 1.0]}}}}}
    hm = {"splits": {"toy": {"panos": {"p1": {
        "box_max_R_2R": [[0.025, 0.03], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "modes_near_boxes": [[0.1 + 1.5 * R, 0.6, 0.02]]}}}}}
    return {"toy": d}, nb, hm


def test_features_drops_and_values(monkeypatch):
    monkeypatch.setattr(a, "SPLITS", ("toy",))
    data, nb, hm = _toy_feature_inputs()
    rows, drops = a.features(data, nb, hm, THR)
    assert drops["toy"] == {"argon_boxes_confirmed": 5, "argon_only": 4, "rig_mask": 1,
                            "near_peak_030": 1, "kept": 2}
    assert [r["box"] for r in rows] == [0, 1]
    b0, b1 = rows
    assert (b0["f1_heatmap_max_R"], b0["f2_support_m"], b0["f3_conf_score"]) == (0.025, 2.0, 0.8)
    assert (b0["f4_yolo_2R"], b0["f5_elev_in_range"]) == (True, True)
    assert b0["f6_snap"] == [0.1 + 1.5 * R, 0.6, 0.02]
    # b1's confidence box and YOLO box are 3R away: outside R and 2R
    assert (b1["f3_conf_score"], b1["f4_yolo_2R"], b1["f6_snap"]) == (None, False, None)
    assert b1["f2_support_m"] == 7.0


def test_features_refuse_a_derived_file_of_the_wrong_length(monkeypatch):
    monkeypatch.setattr(a, "SPLITS", ("toy",))
    data, nb, hm = _toy_feature_inputs()
    hm["splits"]["toy"]["panos"]["p1"]["box_max_R_2R"].pop()
    with pytest.raises(ValueError, match="heatmap_features"):
        a.features(data, nb, hm, THR)
    data, nb, hm = _toy_feature_inputs()
    nb["splits"]["toy"]["panos"]["p1"]["argon_support_m"].pop()
    with pytest.raises(ValueError, match="neighbours"):
        a.features(data, nb, hm, THR)


# ---- leave-one-split-out, end to end -------------------------------------------------- #
TOY = ("A", "B", "C", "D")


def _toy_loso():
    """Four one-pano splits. Every ramp is blind and strict-silent (no peaks). Three kinds
    of box, each passing exactly one test, so only the single-test candidates keep anything
    (no box carries a snap, so each "+snap" variant ties with its plain twin):

    - ``f4_yolo``: 4 TP on A, B, C; 1 FP more on A; 4 FP on D.
    - ``f5_elev``: 1 TP on every split.
    - ``f3>=median``: 7 TP and 2 FP on A only (precision 0.78, just under the 0.80 bar).

    Folds: holding out D, ``f4_yolo`` trains at 12 / 1 (0.92, 12 blind) and beats
    ``f5_elev`` (3 / 0, 1.0, 3 blind) on blind ramps kept, though not on precision; it then
    scores 0 / 4 on D. Holding out A, B or C, ``f4_yolo`` trains under 0.80 and
    ``f3>=median`` is at 0.78, so ``f5_elev`` is chosen and scores 1 / 0. In sample on all
    four splits, the best precision is ``f5_elev`` (4 / 0), not the D fold's choice."""
    data, rows = {}, []
    for s in TOY:
        n_f4 = 4 if s != "D" else 0
        n_f3 = 7 if s == "A" else 0
        gts = [(0.04 + 0.06 * i, 0.6) for i in range(n_f4 + n_f3 + 1)]
        boxes = [(x, y, dict(f4_yolo_2R=True)) for x, y in gts[:n_f4]]
        boxes += [(x, y, dict(f3_conf_score=0.6)) for x, y in gts[n_f4:n_f4 + n_f3]]
        boxes += [(gts[-1][0], 0.6, {})]                                       # f5_elev TP
        fp_f4 = {"A": 1, "D": 4}.get(s, 0)
        boxes += [(0.1 + 0.1 * i, 0.3, dict(f4_yolo_2R=True)) for i in range(fp_f4)]
        boxes += [(0.15 + 0.1 * i, 0.3, dict(f3_conf_score=0.6)) for i in range(2 if s == "A" else 0)]
        for i, (x, y, kw) in enumerate(boxes):
            f5 = not kw                    # only the f5 kind is in the elevation range
            rows.append(dict(_row(x, y, f5_elev_in_range=f5, **kw), split=s, pano="p", box=i))
        data[s] = _split(gts, zs=[(x, y, None) for x, y, _ in boxes], pid="p")
    return data, rows


def test_loso_end_to_end_held_out_choice_differs_from_in_sample_best(monkeypatch):
    monkeypatch.setattr(a, "SPLITS", TOY)
    data, rows = _toy_loso()
    counts = {s: a.base_counts(data[s]) for s in TOY}
    F = a.filters(data, rows, THR, counts)
    sel = {s: F["loso"][s]["selected"] for s in TOY}
    assert sel == {"A": "f5_elev", "B": "f5_elev", "C": "f5_elev", "D": "f4_yolo"}
    assert [F["loso"][s]["n_eligible"] for s in TOY] == [2, 2, 2, 4]   # each with its +snap twin
    tr = F["loso"]["D"]["train"]
    assert (tr["tp"], tr["fp"], tr["blind_kept"]) == (12, 1, 12)
    ho = {s: (F["loso"][s]["held_out"]["tp"], F["loso"][s]["held_out"]["fp"]) for s in TOY}
    assert ho == {"A": (1, 0), "B": (1, 0), "C": (1, 0), "D": (0, 4)}
    po = F["held_out_pooled"]
    assert (po["tp"], po["fp"], po["blind_kept"], po["ss_kept"]) == (3, 4, 3, 3)
    assert F["gates"]["P"]["verdict"] == "kill"
    ins = F["in_sample"]
    assert ins["best"] == "f5_elev"
    assert (ins["best_pooled"]["tp"], ins["best_pooled"]["fp"]) == (4, 0)
    assert ins["n_at_or_above_p_go"] == 2 and ins["p_go_and_y_go"] == []
    # dropping the f2 candidates changes nothing here (no f2 box), through the same helper
    W = a.filters_without_f2(F, THR, counts)
    assert {s: W["loso"][s]["selected"] for s in TOY} == sel
    # no filter: every box kept
    c = a.filter_ceilings(data, rows, THR, counts)
    assert (c["no_filter"]["pooled"]["tp"], c["y_attainable"]) == (23, 23)


def test_selection_tie_breaks_simpler_then_no_snap():
    row = {"boxes": 2, "tp": 2, "fp": 0, "ignored": 0, "blind_kept": 2, "ss_kept": 2,
           "precision": 1.0}
    order = {"a": (1, False), "a & b": (2, False), "a +snap": (1, True), "a & b +snap": (2, True)}
    table = {n: {s: dict(row) for s in a.SPLITS} for n in order}
    loso, _, _ = a.select_loso(table, order, _counts())
    assert {loso[s]["selected"] for s in a.SPLITS} == {"a"}
    table = {n: v for n, v in table.items() if n != "a"}
    loso, _, _ = a.select_loso(table, order, _counts())
    assert {loso[s]["selected"] for s in a.SPLITS} == {"a +snap"}


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


def test_gates_at_their_bars():
    """The plan's bars are inclusive on the go side: precision 0.80, 30 blind ramps, and
    10 strict-silent ramps at 25% read go; one below each does not."""
    loso = _loso([(5, 0.8)] * 4)
    at = {"tp": 80, "fp": 20, "ignored": 0, "precision": 0.80, "blind_kept": 30, "ss_kept": 10}
    g = a.gates(loso, at, _counts(ss_reached=40))
    assert (g["P"]["verdict"], g["Y"]["verdict"], g["S"]["verdict"]) == ("go", "go", "go")
    below = dict(at, tp=79, fp=21, precision=0.79, blind_kept=29, ss_kept=9)
    g = a.gates(loso, below, _counts(ss_reached=20))       # 9 of 20 is 45%: fails on N only
    assert (g["P"]["verdict"], g["Y"]["verdict"], g["S"]["verdict"]) == (
        "in-between", "in-between", "kill")
    assert (a.P_GO, a.P_SPLIT_MIN, a.P_SPLIT_N, a.P_KILL) == (0.80, 0.60, 10, 0.65)
    assert (a.Y_GO, a.Y_KILL, a.S_GO_N, a.S_GO_FRAC) == (30, 20, 10, 0.25)
    assert (a.SELECT_PRECISION, a.LEAK_BAR) == (0.80, 0.05)


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


def test_no_candidate_reaches_the_kill_bar_in_sample_at_R(committed):
    """The doc's claim, at R (the plan's radius): pooled over all four splits, no candidate
    reaches precision 0.65. ``filters.in_sample`` agrees with the candidate table."""
    best = max((a._add([per[s] for s in a.SPLITS])["precision"] or 0.0)
               for per in committed["filters"]["candidates"].values())
    assert best < a.P_KILL
    ins = committed["filters"]["in_sample"]
    assert ins["best_pooled"]["precision"] == pytest.approx(best)
    assert (ins["n_at_or_above_p_kill"], ins["n_at_or_above_p_go"]) == (0, 0)


def test_at_2R_the_stop_rests_on_gate_P_alone(committed):
    """The 2R sensitivity read (not a gate): Y and S read go, P still reads kill, and in
    sample one candidate clears the P and Y go bars together."""
    F = committed["filters_at_2R"]
    assert {k: F["gates"][k]["verdict"] for k in "PYS"} == {"P": "kill", "Y": "go", "S": "go"}
    po = F["held_out_pooled"]
    assert (po["tp"], po["fp"], po["blind_kept"], po["ss_kept"]) == (45, 38, 38, 28)
    ins = F["in_sample"]
    assert ins["n_at_or_above_p_go"] == 8
    assert ins["best_pooled"]["precision"] == pytest.approx(0.875)
    assert ins["p_go_and_y_go"] == ["f2<=5m & f4_yolo"]


def test_ceilings_and_or_read(committed):
    c = committed["filter_ceilings"]
    assert (c["y_attainable"], c["blind_reached_by_argon"]) == (48, 65)
    assert (c["s_attainable"], c["ss_reached_by_argon"]) == (38, 42)
    o = committed["filters_or_pairs"]["in_sample"]
    assert o["best"] == "f3>=q75 | f4_yolo"
    assert (o["best_pooled"]["tp"], o["best_pooled"]["fp"]) == (37, 40)
    assert o["n_at_or_above_p_kill"] == 0
