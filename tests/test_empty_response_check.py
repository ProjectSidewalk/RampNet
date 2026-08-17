"""Guard on the #120-review finding: gemini-3.7-flash's silent panoramas are real.

The conclusion written into docs/model_comparison.md is that the leg's 31% silent
rate is the model being conservative, not the harness losing responses. That rests
on two measured directions, both of which a re-export could quietly invert. If
either flips, the doc is wrong and someone has to look again — which is what these
assert. Pure: reads only the committed published detections and manual_labels.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import empty_response_check as erc  # noqa: E402


def _data():
    d = erc.collect([erc.SUBJECT, "gemini-3.6-flash"], erc.CITY_SPLITS + ["manual_gold"])
    if not d.get(erc.SUBJECT) or not d.get("gemini-3.6-flash"):
        pytest.skip("published Gemini detections not present in this checkout")
    return d


def test_the_deficit_is_spread_not_concentrated(capsys):
    # A dropped response removes a whole view and leaves the surviving panoramas
    # untouched, so a mechanical loss would show ratio ~1.0 here. Measured 0.79.
    rows = erc.test_all_or_nothing(_data(), erc.CITY_SPLITS)
    capsys.readouterr()
    assert rows, "no model pair had panoramas where both fire"
    for other, both, subj, oth, ratio, _, _ in rows:
        assert both > 100, (other, both)
        assert ratio < 0.95, (
            f"{erc.SUBJECT} now finds as much as {other} on panoramas where both "
            f"fire (ratio {ratio:.2f}). The doc reads the deficit as uniform "
            f"strictness; at ratio ~1.0 it would look like concentrated loss instead.")


def test_silence_lands_where_the_gold_set_is_empty(capsys):
    # The load-bearing one: it also rules out a correlated whole-panorama failure,
    # which the six-view arithmetic cannot. Measured 0.80 vs 4.96 gold ramps.
    v = erc.test_emptiness_vs_ground_truth(_data(), erc.gold_counts())
    capsys.readouterr()
    if erc.SUBJECT not in v:
        pytest.skip("manual_gold leg not published for the subject model")
    s = v[erc.SUBJECT]
    assert s["mean_gt_silent"] < 0.4 * s["mean_gt_firing"], (
        f"silent panoramas now carry {s['mean_gt_silent']:.2f} gold ramps against "
        f"{s['mean_gt_firing']:.2f} where the model fires — silence on ramp-dense "
        f"panoramas is what a lost response looks like, and the doc says it doesn't "
        f"happen here.")
    # And the cost of that caution, which the doc quotes as ~5 recall points.
    assert 0 < s["gt_stranded"] / s["gt_total"] < 0.10, s
