"""Tests for Stage 0 of the open-weight cascade confirmation (``cascade_confirm_35.py``, #35).

CPU only, committed inputs only, about 5 s. The synthetic tests check the logic on inputs
whose answer is worked out by hand: the rig mask boundary, the cut test, the top-5 share,
and the leave-one-split-out selection. The data tests check that the reused
``cascade_cost_35`` path reproduces that study's committed rows when the mask is off, and
that ``stage0.json`` regenerates byte for byte.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import cascade_cost_35 as cc  # noqa: E402
import cascade_confirm_35 as s0  # noqa: E402


# ----------------------------------------------------------------- synthetic
def test_rig_mask_boundary_matches_the_labeler():
    """The labeler masks y strictly above 0.5 + 49/180; the boundary itself is kept."""
    assert s0.NADIR_MASK_Y == pytest.approx(0.772222, abs=1e-6)
    assert not s0.on_camera_rig(s0.NADIR_MASK_Y)
    assert s0.on_camera_rig(s0.NADIR_MASK_Y + 1e-9)
    peaks = [(0.1, 0.5, 0.9), (0.2, 0.80, 0.9), (0.3, 0.77, 0.2)]
    assert s0.mask_rig(peaks) == [(0.1, 0.5, 0.9), (0.3, 0.77, 0.2)]


def test_cut_separates():
    assert s0.cut_separates([0.2], [0.0, 0.1])          # Laurens above
    assert s0.cut_separates([0.0], [0.1, 0.2])          # Laurens below
    assert not s0.cut_separates([0.0], [0.0, 0.1])      # a tie cannot be cut
    assert not s0.cut_separates([0.05, 0.3], [0.1])     # interleaved
    assert not s0.cut_separates([], [0.1])


def test_top_share():
    assert s0.top_share([5, 1, 1, 1, 1, 1, 0]) == pytest.approx(0.9)
    assert s0.top_share([1] * 10) == pytest.approx(0.5)
    assert s0.top_share([2, -3, 1]) is None             # total not positive


def _c(tp, fp, ngt):
    """One pano's counts with recall confirmed: (tp, fp, recall tp, recall n_gt)."""
    return [tp, fp, tp, ngt]


def test_loso_selects_on_the_other_splits_only():
    """Three splits, one pano each, two thresholds and two C candidates.

    Threshold 0 wins clearly on B and C and loses on A, so every fold picks it: B + C
    outvote A, and A + C (or A + B) still favour it (14/9 against 14/10). C candidate 1
    wins only on A, so A + C prefers candidate 0 (14/7 against 12/8), and so does B + C.
    """
    N = [np.array([_c(5, 5, 10)]), np.array([_c(5, 5, 10)]), np.array([_c(5, 5, 10)])]
    T = [np.array([[_c(5, 9, 10)], [_c(9, 1, 10)]]),     # A: t1 better
         np.array([[_c(9, 0, 10)], [_c(5, 9, 10)]]),     # B: t0 better
         np.array([[_c(9, 0, 10)], [_c(5, 9, 10)]])]     # C: t0 better
    C = [np.array([[_c(6, 5, 10)], [_c(10, 0, 10)]]),
         np.array([[_c(8, 2, 10)], [_c(2, 8, 10)]]),
         np.array([[_c(8, 2, 10)], [_c(2, 8, 10)]])]
    o = s0.loso(N, T, C)
    assert o["t_idx"][0].tolist() == [0, 0, 0]
    assert o["c_idx"][0].tolist() == [0, 0, 0]
    assert o["t"][0, 0].tolist() == _c(5, 9, 10)          # A scored at the t chosen on B + C
    assert o["c_pool"][0].tolist() == [22, 9, 22, 30]
    d = s0.deltas(o)
    assert d["C_minus_maxNT"][0] == pytest.approx(
        s0.f1_of(o["c_pool"][0]) - max(s0.f1_of(o["n_pool"][0]), s0.f1_of(o["t_pool"][0])))


def test_select_breaks_ties_to_the_first_row():
    rows = np.array([[1, 1, 1, 2], [1, 1, 1, 2], [0, 0, 0, 2]])
    assert int(s0.select(rows)) == 0


def test_loso_weights_scale_like_duplicated_panos():
    """A pano weighted 2 counts exactly as the same pano listed twice."""
    one = [np.array([_c(3, 1, 4), _c(1, 2, 3)])] * 2
    Tm = [np.stack([x, x]) for x in one]
    w = [np.array([[2, 1]]), np.array([[2, 1]])]
    dup = [np.array([_c(3, 1, 4), _c(3, 1, 4), _c(1, 2, 3)])] * 2
    Td = [np.stack([x, x]) for x in dup]
    a, b = s0.loso(one, Tm, Tm, w), s0.loso(dup, Td, Td)
    assert a["n_pool"].tolist() == b["n_pool"].tolist()


# ----------------------------------------------------------------- committed data
@pytest.fixture(scope="module")
def laurens():
    d = s0.load_split("laurens_mapillary")
    d["peaks"] = {pid: d["raw"].get(pid, []) for pid in d["gts"]}     # mask off
    return d


def test_unmasked_path_reproduces_cascade_cost_35(laurens):
    """With the mask off, N and one C candidate equal the study's committed rows (c_min
    0.05 is every published YOLO box, which is the study's c_min 0 row)."""
    with open(os.path.join(cc.DEFAULT_OUT, "laurens_mapillary__y26_pano.json"),
              encoding="utf-8") as f:
        ref = json.load(f)
    N, T, C, _ = s0.split_counts(laurens)
    base = ref["baseline"]
    assert N.sum(0).tolist() == [base["tp"], base["fp"], base["n_gt"] - base["fn"], base["n_gt"]]
    i = s0.CANDIDATES.index(("y26_pano", 0.10, 0.022, 0.05))
    row = next(r for r in ref["grid"] if r["t_lo"] == 0.10 and r["r_gate"] == 0.022
               and r["c_min"] == 0.0)
    assert C[i].sum(0)[:2].tolist() == [row["tp"], row["fp"]]
    ti = s0.T_GRID.index(0.16)
    sweep = ref["threshold_only_best"]
    assert sweep["t"] == 0.16 and T[ti].sum(0)[:2].tolist() == [sweep["tp"], sweep["fp"]]


def test_grid_is_the_plans():
    assert len(s0.CANDIDATES) == 144
    assert s0.T_GRID[0] == 0.10 and s0.T_GRID[-1] == 0.55 and len(s0.T_GRID) == 46
    assert min(c[1] for c in s0.CANDIDATES) >= 0.10


def test_no_lyon_or_kaysville_input():
    assert not any(("lyon" in s or "kaysville" in s) for s in s0.SPLITS)


def test_stage0_regenerates_byte_for_byte():
    with open(os.path.join(s0.OUT_DIR, s0.RESULT), encoding="utf-8", newline="") as f:
        committed = f.read()
    assert cc.dumps(s0.run()) == committed
    p = json.loads(committed)
    assert p["rules"]["4_outcome"] == "N"
    assert p["rules"]["1_drop_T"]["drop_T"] and p["rules"]["2_drop_C"]["drop_C"]
