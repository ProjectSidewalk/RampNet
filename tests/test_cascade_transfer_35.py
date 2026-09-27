"""Tests for the #35 cascade transfer read (``cascade_transfer_35.py``).

CPU only, committed inputs only: the bundles' GT, ``analysis_out/op_cache`` and the published
Vistas parity detections. The committed ``transfer.json`` must re-derive byte for byte, the
richmond reference must equal the fixed row ``cascade_cost_35`` already committed, and the
headline rows are pinned so a change to any input shows up as a named failure.
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
import cascade_transfer_35 as ct  # noqa: E402


@pytest.fixture(scope="module")
def committed_text():
    with open(ct.OUT_PATH, encoding="utf-8", newline="") as f:
        return f.read()


@pytest.fixture(scope="module")
def committed(committed_text):
    return json.loads(committed_text)


@pytest.fixture(scope="module")
def fresh():
    return ct.build()


def test_transfer_artifact_reproduces_byte_for_byte(committed_text, fresh):
    assert cc.dumps(fresh) == committed_text


def test_richmond_reference_is_the_committed_fixed_row(committed):
    with open(os.path.join(cc.DEFAULT_OUT, cc.out_name("richmond", ct.CHALLENGER, cc.T_HI)),
              encoding="utf-8") as f:
        fixed = json.load(f)["fixed_setting"]
    r = committed["primary"]["richmond"]["fixed"]
    for k in ("tp", "fp", "fn", "F1", "attributable_dR", "fp_per_attributable_ramp",
              "null_dR_mean", "promoted_fp", "c_min"):
        assert r[k] == fixed[k], k
    m = committed["primary"]["richmond"]["threshold_only_at_matched_recall"]
    assert m["t"] == fixed["threshold_only_at_matched_recall"]["t"] == 0.15
    assert (m["extra_fp"], m["extra_ramps"]) == (48, 12)


def test_every_split_uses_its_own_median_and_every_shift(committed):
    for split in ct.SPLITS:
        p = committed["primary"][split]
        assert p is not None, split
        assert p["fixed"]["c_min"] == p["c_min_quartiles"][1]
        assert p["null"]["shifts"] == p["n_panos"] - 1
        assert p["calibration"]["challengers"] == p["n_panos"] - 1
        assert p["fixed"]["t_lo"] == 0.05 and p["fixed"]["r_gate"] == 0.011


def test_transfers_is_the_conjunction_of_the_three_criteria(committed):
    for split in ct.SPLITS:
        p = committed["primary"][split]
        c = p["criteria"]
        assert c["viable"] == (p["verdict"] == "VIABLE")
        assert c["attr_dR_ci_above_0"] == (p["bootstrap"]["attr_dR_ci95"][0] > 0)
        assert p["transfers"] == all(c.values())
    passes = [s for s in ct.GSV_SPLITS if committed["primary"][s]["transfers"]]
    assert committed["gsv_passes"] == passes
    assert committed["overall"] == ("TRANSFERS" if len(passes) >= ct.MIN_GSV_PASSES
                                    else "DOES NOT TRANSFER")


# Pinned headline rows (filled from the committed run; see docs/cascade_cost_35.md, Transfer).
PINNED = {
    "bend": {"baseline": (269, 22, 58), "cascade": (275, 31, 52), "attr_dR": 0.0167,
             "verdict": "NOT VIABLE", "transfers": False},
    "paterson": {"baseline": (284, 15, 111), "cascade": (288, 31, 107), "attr_dR": 0.0083,
                 "verdict": "NOT VIABLE", "transfers": False},
    "gainesville": {"baseline": (210, 35, 62), "cascade": (220, 45, 52), "attr_dR": 0.0332,
                    "verdict": "VIABLE", "transfers": True},
    "annapolis": {"baseline": (238, 26, 56), "cascade": (247, 34, 47), "attr_dR": 0.0284,
                  "verdict": "VIABLE", "transfers": True},
    "richmond": {"baseline": (257, 28, 53), "cascade": (269, 38, 41), "attr_dR": 0.0357,
                 "verdict": "VIABLE", "transfers": True},
}


def test_overall_verdict_is_pinned(committed):
    assert committed["gsv_passes"] == ["gainesville"]
    assert committed["overall"] == "DOES NOT TRANSFER"
    # the post-seam-fix sensitivity does not rescue it (gainesville fails criterion 3 there)
    ps = committed["sensitivity_post_seam_peaks"]["splits"]
    assert [s for s in ct.GSV_SPLITS if ps[s]["transfers"]] == []


@pytest.mark.parametrize("split", sorted(PINNED))
def test_pinned_rows(committed, split):
    p = committed["primary"][split]
    want = PINNED[split]
    r, b = p["fixed"], p["baseline"]
    assert (b["tp"], b["fp"], b["fn"]) == want["baseline"]
    assert (r["tp"], r["fp"], r["fn"]) == want["cascade"]
    assert round(r["attributable_dR"], 4) == want["attr_dR"]
    assert p["verdict"] == want["verdict"] and p["transfers"] == want["transfers"]
