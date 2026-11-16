"""The #248 default study (``scripts/analysis/seam_nms_default_248.py``): CPU only, committed
files and synthetic inputs, no torch, no panoramas.

Pins the removal-set builders (certain ``C`` / envelope ``E``), the in-memory GT-B merge, the
manual_gold verdict join, the card classification, the deck shuffle, every branch of the
pre-registered rule, the no-removal-pano invariant, and that the committed counts and
calibration re-derive. The full ``--check`` (which also re-derives ``results.json``, about 25 s)
runs in ``scripts/check_all.py`` as ``seam_nms_default_248``.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import seam_nms_default_248 as s  # noqa: E402


def px(col, row, score):
    """A peak at heatmap pixel (col, row) as the caches store it (normalized x, y)."""
    return (col / s.W, row / s.H, score)


# --------------------------------------------------------------------------- C / E builders
def test_straddling_pair_four_px_gives_one_certain_removal():
    peaks = [px(1022, 300, 0.9), px(2, 300, 0.6), px(500, 300, 0.95)]
    c = s.removal_candidates(peaks)
    assert [(r["weak"], r["strong"], r["set"], r["dist"]) for r in c] == [(1, 0, "C", 4)]


def test_stronger_peak_fifteen_px_across_gives_envelope_only():
    peaks = [px(1015, 200, 0.9), px(6, 200, 0.5)]        # wrapped distance 15
    c = s.removal_candidates(peaks)
    assert [(r["weak"], r["set"], r["dist"]) for r in c] == [(1, "E", 15)]


def test_stronger_peak_on_the_same_side_is_neither():
    peaks = [px(20, 200, 0.9), px(5, 200, 0.5)]           # both near x = 0, not across
    assert s.removal_candidates(peaks) == []


def test_stronger_half_is_never_a_candidate_and_raw_scores_above_one_order_correctly():
    peaks = [px(1020, 100, 1.07), px(3, 100, 0.99)]       # raw > 1 clips to 1.0 > 0.99
    c = s.removal_candidates(peaks)
    assert [r["weak"] for r in c] == [1]
    assert all(r["weak"] != 0 for r in c)


def test_equal_clipped_scores_are_reported_as_a_tie_not_a_removal():
    peaks = [px(1020, 100, 1.2), px(3, 100, 1.05)]        # both clip to 1.0
    c = s.removal_candidates(peaks)
    assert all(r["set"] is None for r in c) and sum(len(r["ties"]) for r in c) == 2


def test_peak_outside_the_strip_cannot_be_removed():
    peaks = [px(1020, 100, 0.9), px(12, 100, 0.5)]       # column 12: window does not wrap
    assert s.removal_candidates(peaks) == []


def test_straddling_pairs_counts_pairs_both_above_t():
    peaks = [px(1022, 300, 0.9), px(2, 300, 0.6)]
    assert s.straddling_pairs(peaks, 0.30) == [4]
    assert s.straddling_pairs(peaks, 0.70) == []


# --------------------------------------------------------------------------- GT-B merge
def test_gt_b_merges_ten_one_pairs_to_seam_midpoints():
    gts = s.manual_gold_gt()
    pairs = s.jon_pairs()
    assert sum(len(g.gt_points) for g in gts.values()) == 3919
    merged_gts, merged = s.merge_gt_b(gts, pairs)
    assert sum(len(g.gt_points) for g in merged_gts.values()) == 3909
    assert len(merged) == 10
    for m in merged:              # a naive average of x = 0.002 and 0.999 would land at 0.5
        assert m["to_a_px"] <= s.JOIN_PX and m["to_b_px"] <= s.JOIN_PX
        assert m["to_seam_px"] <= 7
    for pr in pairs:
        if pr["verdict"] == "two":
            assert len(merged_gts[pr["pano"]].gt_points) == len(gts[pr["pano"]].gt_points)


def test_wrapped_midpoint_crosses_the_seam():
    x, y = s.wrapped_midpoint((0.002, 0.5), (0.998, 0.6))
    assert min(x, 1 - x) < 1e-9 and y == pytest.approx(0.55)


# --------------------------------------------------------------------------- verdict join
def test_join_inherits_only_when_each_peak_is_nearest_a_different_mark(monkeypatch):
    pairs = [{"pano": "p", "a": (0.001, 0.5), "b": (0.995, 0.5), "verdict": "one"}]
    both = {"pano": "p", "weak": [0.001, 0.5, 0.4], "strong": [0.996, 0.5, 0.8]}
    assert s.inherit(both, pairs) == 0
    # a tight model pair (4 px) on a `two` pano whose marks are 20 px apart: both peaks are
    # nearest the same mark, so it is not inherited (it would be a false L)
    two = [{"pano": "q", "a": (0.002, 0.5), "b": (0.983, 0.5), "verdict": "two"}]
    tight = {"pano": "q", "weak": [0.003, 0.5, 0.4], "strong": [0.999, 0.5, 0.8]}
    assert s.inherit(tight, two) is None
    far = {"pano": "p", "weak": [0.02, 0.5, 0.4], "strong": [0.996, 0.5, 0.8]}
    assert s.inherit(far, pairs) is None                   # weak peak > 6.5 px from mark a


# --------------------------------------------------------------------------- classification
@pytest.mark.parametrize("pair, weak, strong, cls", [
    ("same", "ramp", "ramp", "D"), ("same", "not_ramp", "ramp", "D"),
    ("same", "ramp", "not_ramp", "N"), ("same", "ramp", "unsure", "N"),
    ("different", "ramp", "ramp", "L"), ("different", "not_ramp", "ramp", "N"),
    ("different", "unsure", "ramp", "U"), ("unsure", "ramp", "ramp", "U"),
    (None, None, None, "U"),
])
def test_classification_table(pair, weak, strong, cls):
    assert s.classify(pair, weak, strong) == cls


def test_jon_verdicts_map_one_two_unsure():
    assert [s.jon_class(v) for v in ("one", "two", "unsure")] == ["D", "L", "U"]


def test_deck_order_is_deterministic_under_the_seed():
    keys = [f"k{i}" for i in range(30)]
    a, b = s.deck_order(keys), s.deck_order(list(reversed(keys)))
    assert a == b and sorted(k for k, _ in a) == sorted(keys)
    assert 5 < sum(f for _, f in a) < 25                     # letters really are mixed


# --------------------------------------------------------------------------- the rule
def card(st, cls, source="ai", cid="x"):
    return {"set": st, "cls": cls, "source": source, "card": cid, "split": "bend", "pano": "p"}


def test_rule_keep_on_one_jon_loss_in_c():
    assert s.decide([card("C", "L", "jon (inherited)", None)], True)[0] == s.KEEP


def test_rule_keep_on_two_ai_losses_in_c():
    assert s.decide([card("C", "L"), card("C", "L", cid="y")], True)[0] == s.KEEP


def test_rule_envelope_only_loss_cannot_keep():
    out, reasons = s.decide([card("E", "L"), card("E", "L", cid="y")], True)
    assert out == s.NEXT and reasons["panoramas_for_exact_tier"] == ["bend:p"]


def test_rule_flip_conditional_and_flip():
    assert s.decide([card("C", "D"), card("E", "U", cid="y")], True)[0] == s.FLIP_COND
    jon = [card("C", "D", "jon (audit)"), card("E", "N", "jon (audit)", "y")]
    assert s.decide(jon, True)[0] == s.FLIP


def test_rule_uncalibrated_ai_gives_next_step_naming_every_card():
    out, reasons = s.decide([card("C", "U", "none", "a"), card("E", "U", "none", "b")], False)
    assert out == s.NEXT and reasons["cards_jon_must_rate"] == ["a", "b"]


def test_rule_two_unresolved_gives_next_step():
    out, reasons = s.decide([card("C", "U", cid="a"), card("E", "U", cid="b"),
                             card("C", "D", cid="c")], True)
    assert out == s.NEXT and reasons["cards_jon_must_rate"] == ["a", "b"]


# --------------------------------------------------------------------------- invariant
def test_no_removal_pano_scores_identically():
    pytest.importorskip("numpy")
    from rampnet.detection_eval import GroundTruth, radius_sq_for
    rsq = radius_sq_for()
    gts = {"a": GroundTruth([(0.001, 0.5)], [], True), "b": GroundTruth([(0.5, 0.5)], [], True)}
    panos = [("a", [px(1023, 256, 0.9), px(3, 256, 0.6)], None),
             ("b", [px(512, 256, 0.8)], None)]
    cands = [{"split": "x", "pano": "a", "set": "C", "weak": list(panos[0][1][1]),
              "strong": list(panos[0][1][0]), "dist": 4}]
    arms, removed = s.arms_for(panos, cands)
    assert len(arms["lo"]["a"]) == 1 and arms["lo"]["b"] == arms["default"]["b"]
    sd = s.scored("x", arms["default"], gts, rsq)
    sa = s.scored("x", arms["lo"], gts, rsq)
    s.assert_unchanged_panos_identical(sd, sa, removed["lo"])


# --------------------------------------------------------------------------- committed outputs
def test_committed_bounds_and_calibration_re_derive():
    b = s.build_bounds()
    b["manual_gold_gt_points"] = 3919
    assert s.check_reviewer_literals(b) == []
    committed = s.read_json(os.path.join(s.OUT_DIR, "bounds.json"))
    assert json.loads(json.dumps(b)) == committed
    assert json.loads(json.dumps(s.build_calibration())) == \
        s.read_json(os.path.join(s.OUT_DIR, "calibration.json"))


def test_committed_results_pin_the_outcome():
    r = s.read_json(os.path.join(s.OUT_DIR, "results.json"))
    assert r["outcome"] == s.NEXT and r["reasons"]["L_pess"] == 4
    assert r["calibration"]["calibrated"] is True and r["tripwire"]["fired"] is False
