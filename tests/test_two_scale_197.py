"""Tests for scripts/analysis/two_scale_197.py (issue #197, two-scale inference).

CPU only, no network: the fusion helpers are tested on synthetic peaks, and the reproduce
test reads the committed #196 caches under analysis_out/input_res_sweep_25/cache/ and the
committed analysis_out/two_scale_197/results.json.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import two_scale_197 as ts  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
RESULTS = os.path.join(REPO, "analysis_out", "two_scale_197", "results.json")


def test_cut_row_matches_flat_range():
    y = ts.y_cut_for(18.0)
    assert abs(y - 0.543929) < 1e-6
    assert ts.is_far(y - 1e-6, 18.0) and not ts.is_far(y + 1e-6, 18.0)
    assert ts.is_far(0.45, 18.0)            # above the horizon counts as far
    assert not ts.is_far(0.9, 18.0)


def test_dedupe_keeps_higher_score_across_passes_only():
    near = [(0.50, 0.60, 0.90), (0.505, 0.60, 0.40)]   # 5 px apart: same pass, both kept
    far = [(0.502, 0.60, 0.80), (0.20, 0.55, 0.50)]
    kept, dn, df = ts.dedupe(near, far, RSQ)
    assert (0.50, 0.60, 0.90) in kept and (0.505, 0.60, 0.40) in kept
    assert (0.502, 0.60, 0.80) not in kept and (0.20, 0.55, 0.50) in kept
    assert (dn, df) == (0, 1)
    assert [p[2] for p in kept] == sorted((p[2] for p in kept), reverse=True)
    # a higher-scoring far peak suppresses the near one instead
    kept, dn, df = ts.dedupe([(0.5, 0.6, 0.4)], [(0.501, 0.6, 0.7)], RSQ)
    assert kept == [(0.501, 0.6, 0.7)] and (dn, df) == (1, 0)


def test_dedupe_wraps_at_the_seam():
    kept, dn, df = ts.dedupe([(0.999, 0.6, 0.9)], [(0.001, 0.6, 0.8)], RSQ)
    assert kept == [(0.999, 0.6, 0.9)] and df == 1


def test_fuse_range_takes_each_side_from_its_pass():
    r = [(0.1, 0.70, 0.9), (0.3, 0.52, 0.9), (0.5, 0.70, 0.2)]   # near hit, far hit, low
    u = [(0.6, 0.70, 0.9), (0.8, 0.52, 0.9), (0.9, 0.52, 0.25)]
    kept, _, _ = ts.fuse_range(r, u, 18.0, RSQ)
    assert sorted(kept) == [(0.1, 0.70, 0.9), (0.8, 0.52, 0.9)]
    kept, _, _ = ts.fuse_range(r, u, 18.0, RSQ, t_u=0.2)
    assert (0.9, 0.52, 0.25) in kept
    kept, _, _ = ts.fuse_union(r, u, RSQ)
    assert len(kept) == 4


def _c(r_lo, f_lo, f_hi):
    return {"recall": {"ci_lo": r_lo}, "f1": {"ci_lo": f_lo, "ci_hi": f_hi}}


def test_verdict_branches():
    beat = {"f1": {"ci_lo": 0.01}}
    lose = {"f1": {"ci_lo": -0.01}}
    assert ts.verdict(_c(0.01, 0.001, 0.02), beat) == "HELPS"
    assert ts.verdict(_c(0.01, -0.01, 0.02), beat) == "RECALL LEVER"
    assert ts.verdict(_c(0.01, -0.01, -0.001), lose) == "NO BETTER THAN A THRESHOLD"
    assert ts.verdict(_c(0.01, 0.001, 0.02), None) == "NO BETTER THAN A THRESHOLD"
    assert ts.verdict(_c(-0.01, -0.03, -0.001), lose) == "HURTS"
    assert ts.verdict(_c(-0.01, -0.03, 0.001), lose) == "NULL"


def test_richmond_rows_reproduce_the_committed_results():
    """Per-split rows that do not depend on the other splits (everything but R3, whose
    leave-one-split-out choice does) re-derive exactly from the committed caches."""
    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)["per_split"]["richmond"]
    got = json.loads(json.dumps(ts.build(cities=("richmond",), n_shifts=2)))
    got = got["per_split"]["richmond"]
    for v in ("r2048", "u4096", "naive_union", "R1", "R2", "R2s"):
        assert got["metrics"][v] == committed["metrics"][v], v
    for v in ("u4096", "naive_union", "R1", "R2", "R2s"):
        assert got["rules"][v] == committed["rules"][v], v
    # the control is #196's r2048 row
    assert committed["metrics"]["r2048"]["F1"] == 0.8614


def test_inference_cost_from_the_ledger():
    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)["inference_cost"]
    got = ts.inference_cost()
    assert got == committed
    assert got["panos"] == {"r2048": 1289, "u4096": 1289}
    assert 4.5 < got["ratio_vs_r2048"] < 5.2


@pytest.mark.parametrize("pool", ["US pool (miss_decomposition.US_SPLITS)"])
def test_committed_verdicts(pool):
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    rules = rep["pooled"][pool]["rules"]
    assert rules["R1"]["verdict"] == "HURTS"
    assert rules["R2"]["verdict"] == "NO BETTER THAN A THRESHOLD"
    assert rules["R3"]["verdict"] == "NO BETTER THAN A THRESHOLD"
    assert all(e["chosen"] == {"D_m": 40.0, "t_u": 0.4} for e in rep["loso"].values())
