"""The matching family's committed report re-derives from committed predictions (#48).

Checks the headline numbers only (medians, fallback rates, aligned counts); the bootstrap
CIs are re-derived by running ``scripts/analysis/crossview_matching_48.py``.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import crossview_align_48 as H  # noqa: E402
import crossview_matching_48 as M  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(M.OUT_JSON), reason="no committed report")


def test_report_headlines_rederive():
    with open(M.OUT_JSON, encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["config"]["pairs_sha256"] == H.PAIRS_SHA256
    pairs = H.read_frozen_pairs()
    auto = H.arm_errors(pairs, H.read_predictions(M.AUTO))
    for name, st in rep["arms"].items():
        e = H.arm_errors(pairs, H.read_predictions(name))
        a = st["all"]
        assert a["median_deg"] == pytest.approx(np.median([r[0] for r in e]), abs=1e-3)
        assert a["fallback_rate"] == pytest.approx(np.mean([r[3] for r in e]), abs=1e-4)
        aligned = [i for i, r in enumerate(e) if not r[3]]
        if aligned:
            assert a["aligned"]["n"] == len(aligned)
            assert a["aligned"]["auto_median_deg"] == pytest.approx(
                np.median([auto[i][0] for i in aligned]), abs=1e-3)


def test_matching_arms_never_see_answers():
    """Every family arm's prediction row carries no answer column."""
    for name in M.DEFAULT_ARMS:
        for r in M.load_raw(name).values():
            assert not set(r) & {"ref_x", "ref_y", "ref_conf", "ref_world_gap_m"}
