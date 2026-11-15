"""Tests for the #35 Argon cascade read (``cascade_argon_35.py``).

CPU only, committed inputs only. The full byte-for-byte regeneration is ``--check`` (about
three minutes, in ``check_all.py --all``); here one pair file and one split of
``argon_pairs.json`` are re-derived, and the headline rows are pinned.
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
    """A pair the study already committed is never re-run into this directory, and the
    study's own directory listing (its --check and summary.json) does not see these files."""
    study = set(cc.pair_files(cc.DEFAULT_OUT))
    own = set(ca.own_pair_files(ca.OUT_DIR))
    assert own and not (own & study)
    assert ca.RESULT not in own and "argon" not in study
    for leg in ca.ARGON:      # the study still lists every Argon pair as a gap
        assert cc.out_name(SPLIT, leg, cc.T_HI) in own


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
