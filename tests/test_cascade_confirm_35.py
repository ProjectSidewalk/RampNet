"""Tests for Stage 0 of the open-weight cascade confirmation (``cascade_confirm_35.py``, #35).

CPU only, committed inputs only, about 5 s. The synthetic tests check the logic on inputs
whose answer is worked out by hand: the rig mask boundary, the cut test, the top-5 share,
the leave-one-split-out selection, every branch of the four rules and their bars, and the
second pooling reading. The data tests check that the reused ``cascade_cost_35`` path
reproduces that study's committed rows when the mask is off, and that ``stage0.json`` and
``inputs.json`` regenerate byte for byte.
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
    """Three splits A, B, C, one pano each (10 GT), two thresholds and two C candidates.

    Thresholds: t1 is far better on A (F1 1.0 against 0.1), t0 is better on B and C (0.9
    against 0.7). Trained on B + C, fold A picks t0 (0.90 against 0.70); folds B and C
    have A in training and pick t1 (0.85 against 0.50). With the held-out split also in
    training, every fold would pick t1 (0.80 against 0.63 over all three), and on the
    held-out split alone the picks would be [1, 0, 0]. So only the correct rule gives
    [0, 1, 1]. C candidates mirror this with B as the odd split: [1, 0, 1]. N is poor
    (F1 0.2) so that max(N, T) differs from N, pooled and per split.
    """
    N = [np.array([_c(2, 8, 10)]), np.array([_c(2, 8, 10)]), np.array([_c(2, 8, 10)])]
    T = [np.array([[_c(1, 9, 10)], [_c(10, 0, 10)]]),    # A: t1 far better
         np.array([[_c(9, 1, 10)], [_c(7, 3, 10)]]),     # B: t0 better
         np.array([[_c(9, 1, 10)], [_c(7, 3, 10)]])]     # C: t0 better
    C = [np.array([[_c(9, 1, 10)], [_c(7, 3, 10)]]),     # A: c0 better
         np.array([[_c(1, 9, 10)], [_c(10, 0, 10)]]),    # B: c1 far better
         np.array([[_c(9, 1, 10)], [_c(7, 3, 10)]])]     # C: c0 better
    o = s0.loso(N, T, C)
    assert o["t_idx"][0].tolist() == [0, 1, 1]
    assert o["c_idx"][0].tolist() == [1, 0, 1]
    assert o["t"][0, 0].tolist() == _c(1, 9, 10)          # A scored at the t chosen on B + C
    assert o["c"][0, 1].tolist() == _c(1, 9, 10)          # B scored at the C chosen on A + C
    assert o["t_pool"][0].tolist() == [15, 15, 15, 30]
    assert o["c_pool"][0].tolist() == [15, 15, 15, 30]
    d = s0.deltas(o)
    # pooled: N 0.2, T 0.5, C 0.5, so C - max(N, T) is 0 and C - N is 0.3
    assert d["C_minus_maxNT"][0] == pytest.approx(0.0)
    assert d["C_minus_N"][0] == pytest.approx(0.3)
    # per split, C - max(N, T): A 0.7 - 0.2, B 0.1 - 0.7, C 0.7 - 0.7
    assert d["split_C_minus_maxNT"][0].tolist() == pytest.approx([0.5, -0.6, 0.0])
    # per split, C - N: A 0.7 - 0.2, B 0.1 - 0.2, C 0.7 - 0.2
    assert d["split_C_minus_N"][0].tolist() == pytest.approx([0.5, -0.1, 0.5])


# ----------------------------------------------------------------- the rules
def test_bars_are_the_plans():
    """Plan section 3, as literals, so a changed constant fails here and not only in the
    byte-compare."""
    assert s0.T_MIN_LOWER == 0.0
    assert s0.C_MIN_GAIN == 0.03
    assert s0.C_MIN_LOWER == 0.0
    assert s0.C_MIN_WINS == 2
    assert s0.ONE_INSTANCE_GAIN == 0.02
    assert s0.TOP_PANOS == 5
    assert s0.TOP_SHARE_MAX == 0.5
    assert s0.bars() == {"T_min_lower": 0.0, "C_min_gain": 0.03, "C_min_lower": 0.0,
                         "C_min_wins": 2, "one_instance_gain": 0.02, "top_panos": 5,
                         "top_share_max": 0.5}


def test_won_is_strict():
    assert not s0.won(0.0)
    assert s0.won(1e-9)
    assert not s0.won(-1e-9)


def test_rule1_drop_T_branches():
    assert not s0.drop_T(True, 0.001)       # key passes, interval clears 0: keep
    assert s0.drop_T(True, 0.0)             # lower bound at 0
    assert s0.drop_T(True, -0.01)           # lower bound below 0
    assert s0.drop_T(False, 0.05)           # key fails


def test_rule2_drop_C_branches():
    assert not s0.drop_C(0.031, 0.001, 2)   # every condition met
    assert not s0.drop_C(0.03, 0.001, 2)    # exactly at the bar is not below it
    assert not s0.drop_C(0.04, 0.01, 4)
    assert s0.drop_C(0.029, 0.01, 4)        # below +0.03
    assert s0.drop_C(0.05, 0.0, 4)          # lower bound at 0
    assert s0.drop_C(0.05, -0.01, 4)        # lower bound below 0
    assert s0.drop_C(0.05, 0.01, 1)         # one win


def test_rule3_one_instance_branches():
    assert s0.one_instance(0.021, 0.5) == (False, False)   # exactly half is not over half
    assert s0.one_instance(0.05, 0.45) == (False, False)
    assert s0.one_instance(0.019, 0.3) == (True, False)
    assert s0.one_instance(0.10, 0.51) == (False, True)
    assert s0.one_instance(0.10, None) == (False, False)   # nothing to concentrate


def test_rule4_outcome_labels():
    assert s0.outcome(True, True) == "N"
    assert s0.outcome(False, True) == "T only"
    assert s0.outcome(True, False) == "C only"
    assert s0.outcome(False, False) == "T and C"


def _deltas(t_n, c_max, c_n, split_c_max, split_c_n, B=40, spread=0.01, spreads=None):
    """Hand-made ``deltas`` output: point values ``[1]`` / ``[1, S]``, and a bootstrap whose
    95% interval is the point plus or minus about ``spread`` (``spreads`` overrides it per
    key)."""
    spreads = spreads or {}
    dp = {"T_minus_N": np.array([t_n]), "C_minus_maxNT": np.array([c_max]),
          "C_minus_N": np.array([c_n]),
          "split_C_minus_maxNT": np.array([split_c_max]),
          "split_C_minus_N": np.array([split_c_n])}
    db = {}
    for k, v in dp.items():
        jitter = np.linspace(-spreads.get(k, spread), spreads.get(k, spread), B)
        db[k] = v[0] + jitter if v.ndim == 1 else v[0][None, :] + jitter[:, None]
    return dp, db


def test_rules_read_max_N_T_not_N():
    """C minus N clears every bar and C minus max(N, T) does not: Rule 2 must drop C and
    the C-against-N reading must keep it. Wins count only strict gains (one zero split)."""
    dp, db = _deltas(t_n=0.02, c_max=0.02, c_n=0.05,
                     split_c_max=[0.05, 0.0, 0.04, -0.01], split_c_n=[0.06, 0.04, 0.05, 0.02])
    r = s0.rules_from_deltas(dp, db, rig_key_passes=True, worst_loo_gain=0.05, top_share=0.3)
    assert r["2_drop_C"]["C_minus_maxNT"] == pytest.approx(0.02)
    assert r["2_drop_C"]["splits_won"] == 2
    assert r["2_drop_C"]["drop_C"] and r["2_drop_C"]["below_0.03"]
    assert r["2_alt_C_vs_N"]["splits_won"] == 4
    assert not r["2_alt_C_vs_N"]["drop_C"]
    assert not r["1_drop_T"]["drop_T"]                # T - N lower bound ~ +0.01
    assert r["4_outcome"] == "T only"


def test_rules_from_deltas_each_outcome():
    keep_c = dict(c_max=0.05, c_n=0.05, split_c_max=[0.05] * 4, split_c_n=[0.05] * 4)
    drop_c = dict(c_max=0.02, c_n=0.02, split_c_max=[0.05] * 4, split_c_n=[0.05] * 4)
    for rig, t_n, cs, want in ((False, 0.05, drop_c, "N"), (True, 0.05, drop_c, "T only"),
                               (True, -0.05, keep_c, "C only"), (True, 0.05, keep_c, "T and C")):
        dp, db = _deltas(t_n=t_n, **cs)
        r = s0.rules_from_deltas(dp, db, rig_key_passes=rig, worst_loo_gain=0.05, top_share=0.3)
        assert r["4_outcome"] == want, (rig, t_n, want)
    # T - N interval straddles 0 (about [-0.0095, +0.0095]): Rule 1 reads the lower bound
    dp, db = _deltas(t_n=0.0, **keep_c)
    r = s0.rules_from_deltas(dp, db, rig_key_passes=True, worst_loo_gain=0.05, top_share=0.3)
    assert r["1_drop_T"]["T_minus_N_lower"] < 0 and r["4_outcome"] == "C only"
    dp, db = _deltas(t_n=0.05, **keep_c)
    r = s0.rules_from_deltas(dp, db, rig_key_passes=True, worst_loo_gain=0.01, top_share=0.3)
    assert r["3_one_instance"] == {"flag_one_instance": True, "flag_sequence": True,
                                   "flag_top5": False}


def test_rig_key_passes_needs_ecartip():
    """Without ecartip the key cannot pass, even when the other splits can be cut."""
    assert not s0.rig_key_passes([0.2], [0.0, 0.1])
    assert s0.rig_key_passes([0.2], [0.0, 0.1], ecartip=0.3)
    assert not s0.rig_key_passes([0.2], [0.0, 0.1], ecartip=0.05)   # ecartip on the wrong side
    assert not s0.rig_key_passes([0.0], [0.0, 0.1], ecartip=0.3)    # tie with the other side


def test_count_margin():
    """F1 0.7 at (7 TP, 3 FP, 10 GT); the bar asks for 0.72: one more TP (0.762) or one
    fewer FP (0.737) gets there."""
    m = s0.count_margin([7, 3, 7, 10], best_other_f1=0.69, bar=0.03)
    assert m["extra_tp_needed"] == 1 and m["tp_needed_at_same_fp"] == 8
    assert m["fewer_fp_needed"] == 1 and m["fp_allowed_at_same_tp"] == 2
    m = s0.count_margin([8, 2, 8, 10], best_other_f1=0.70, bar=0.03)
    assert m["extra_tp_needed"] == 0 and m["fewer_fp_needed"] == 0


def test_per_split_mean_reading():
    """The mean of the per-split deltas can clear the bar where the summed counts do not."""
    dp, db = _deltas(t_n=-0.02, c_max=0.027, c_n=0.027,
                     split_c_max=[0.09, 0.05, -0.008, -0.008], split_c_n=[0.09, 0.05, 0.0, 0.0])
    r = s0.per_split_mean_reading(dp, db, drop_t=True)
    assert r["C_minus_maxNT"] == pytest.approx(0.031)
    assert r["splits_won"] == 2
    assert not r["drop_C"] and r["outcome"] == "C only"
    dp, db = _deltas(t_n=-0.02, c_max=0.027, c_n=0.027,
                     split_c_max=[0.09, 0.02, -0.008, -0.008], split_c_n=[0.09, 0.02, 0.0, 0.0])
    r = s0.per_split_mean_reading(dp, db, drop_t=True)
    assert r["drop_C"] and r["outcome"] == "N"


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


def test_rule2_reads_the_lower_bound_of_C_minus_max_N_T():
    """Review 2, nit 1. The C - max(N, T) interval straddles 0 (about [-0.057, +0.157])
    while the C - N interval does not (about [+0.04, +0.06]): Rule 2 must read the lower
    bound of C - max(N, T) and drop C."""
    dp, db = _deltas(t_n=0.05, c_max=0.05, c_n=0.05, split_c_max=[0.05] * 4,
                     split_c_n=[0.05] * 4, spreads={"C_minus_maxNT": 0.11})
    r = s0.rules_from_deltas(dp, db, rig_key_passes=True, worst_loo_gain=0.05, top_share=0.3)
    assert r["2_drop_C"]["C_minus_maxNT_lower"] < 0
    assert r["2_drop_C"]["lower_le_0"] and r["2_drop_C"]["drop_C"]
    assert not r["2_alt_C_vs_N"]["drop_C"]
    assert r["4_outcome"] == "T only"


def test_rule3_gain_exactly_at_the_bar_is_not_below_it():
    assert s0.one_instance(0.02, 0.3) == (False, False)


def test_per_split_mean_reading_keeps_rule_1_and_its_own_interval():
    """The outcome follows Rule 1's verdict, and the interval comes from the per-split
    resamples, not the pooled ones (here the pooled interval is wide enough to cross 0)."""
    dp, db = _deltas(t_n=0.05, c_max=0.027, c_n=0.027,
                     split_c_max=[0.09, 0.05, -0.008, -0.008], split_c_n=[0.09, 0.05, 0.0, 0.0],
                     spreads={"C_minus_maxNT": 0.1})
    r = s0.per_split_mean_reading(dp, db, drop_t=False)
    assert r["C_minus_maxNT_ci95"][0] > 0.015
    assert not r["drop_C"] and r["outcome"] == "T and C"


def test_mean_f1_reading():
    """Mean F1 over the four splits: N 0.6, T 0.7, C 0.75, so the gap is 0.05 against
    max(N, T) (0.15 against N alone). The resamples move each row, so the interval has
    width."""
    B = 40
    jitter = np.linspace(-0.01, 0.01, B)[:, None]
    dp = {"split_F1_N": np.full((1, 4), 0.6), "split_F1_T": np.full((1, 4), 0.7),
          "split_F1_C": np.full((1, 4), 0.75), "split_C_minus_maxNT": np.full((1, 4), 0.05)}
    db = {k: np.repeat(v, B, axis=0) + (jitter if k == "split_F1_C" else 0.0)
          for k, v in dp.items()}
    r = s0.mean_f1_reading(dp, db, drop_t=True)
    assert r["C_minus_maxNT"] == pytest.approx(0.05)
    lo, hi = r["C_minus_maxNT_ci95"]
    assert hi - lo > 0.01 and lo > 0.035 and hi < 0.065
    assert r["splits_won"] == 4 and not r["drop_C"] and r["outcome"] == "C only"


def test_selection_sensitivity_on_hand_made_counts():
    """Three splits, one pano each (10 GT). N is F1 0.2 and both thresholds 0.5, so
    max(N, T) is T. Of the 144 candidates, four score well and alike on every split:
    0 (y11l, F1 0.9) > 1 (y11l, 0.8) > 48 (y11x, 0.7) > 96 (y26, 0.6); the rest score 0.1.
    Every fold picks 0 with 1 as its runner-up (0.8 held out, 0.3 above T pooled); one arm
    at a time gives 0.4, 0.2 and 0.1 above T."""
    K = len(s0.CANDIDATES)
    N = [np.array([_c(2, 8, 10)]) for _ in range(3)]
    T = [np.array([[_c(5, 5, 10)], [_c(5, 5, 10)]]) for _ in range(3)]
    C = []
    for _ in range(3):
        c = np.tile(np.array([[_c(1, 9, 10)]]), (K, 1, 1))
        for i, (tp, fp) in {0: (9, 1), 1: (8, 2), 48: (7, 3), 96: (6, 4)}.items():
            c[i, 0] = _c(tp, fp, 10)
        C.append(c)
    assert [s0.CANDIDATES[i][0] for i in (0, 1, 48, 96)] == [
        "y11l_pano", "y11l_pano", "y11x_pano_h200", "y26_pano"]
    point = s0.loso(N, T, C)
    out = s0.selection_sensitivity(N, T, C, point, s0.deltas(point))
    for row in out["runner_up"]:
        assert row["chosen"] == s0._cand(0) and row["runner_up"] == s0._cand(1)
        assert row["runner_up_held_out_F1"] == pytest.approx(0.8)
        assert not row["runner_up_better_held_out"]
    assert out["runner_up_pooled_C_minus_maxNT"] == pytest.approx(0.3)
    arms = out["per_arm"]
    assert arms["y11l_pano"]["C_minus_maxNT"] == pytest.approx(0.4)
    assert arms["y11x_pano_h200"]["C_minus_maxNT"] == pytest.approx(0.2)
    assert arms["y26_pano"]["C_minus_maxNT"] == pytest.approx(0.1)
    assert arms["y26_pano"]["candidates"] == [s0._cand(96)] * 3


def test_check_exits_1_when_either_file_differs(tmp_path, monkeypatch, capsys):
    """``main --check`` against a scratch directory holding a fresh stage0.json (from
    ``run()``, so this tests the exit logic, not the committed file) and a changed
    inputs.json: one DIFF must fail the check."""
    (tmp_path / s0.RESULT).write_bytes(cc.dumps(s0.run()).encode("utf-8"))
    (tmp_path / s0.INPUTS).write_bytes(b"{}\n")
    monkeypatch.setattr(sys, "argv", ["cascade_confirm_35.py", "--check", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        s0.main()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert out.count("same  ") == 1 and out.count("DIFF  ") == 1


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


@pytest.fixture(scope="module")
def stage0():
    with open(os.path.join(s0.OUT_DIR, s0.RESULT), encoding="utf-8", newline="") as f:
        committed = f.read()
    return committed, json.loads(committed)


def test_stage0_regenerates_byte_for_byte(stage0):
    committed, p = stage0
    assert cc.dumps(s0.run()) == committed
    assert p["rules"]["4_outcome"] == "N"
    assert p["rules"]["1_drop_T"]["drop_T"] and p["rules"]["2_drop_C"]["drop_C"]
    assert p["bars"] == s0.bars()


def test_pooling_readings_agree_with_the_folds(stage0):
    """The second reading is the plain mean of the four folds' deltas, and the summed
    reading is the pooled one the rules use; the stop rests on the summed reading."""
    _, p = stage0
    pr = p["pooling_readings"]
    per = [f["C_minus_maxNT"] for f in p["folds"]]
    assert pr["per_split_mean"]["C_minus_maxNT"] == pytest.approx(sum(per) / 4, abs=2e-6)
    assert pr["summed_counts"]["C_minus_maxNT"] == p["pooled"]["C_minus_maxNT"]
    assert pr["summed_counts"]["drop_C"] == p["rules"]["2_drop_C"]["drop_C"]
    assert pr["stop_rests_on"] == "summed_counts"


def test_inputs_json_is_current():
    """``inputs.json`` hashes each input with CRLF read as LF, so a ``core.autocrlf=true``
    checkout gives the same hashes as the committed blobs."""
    with open(os.path.join(s0.OUT_DIR, s0.INPUTS), encoding="utf-8", newline="") as f:
        assert f.read() == cc.dumps(s0.input_hashes())


def test_sha_reads_crlf_as_lf(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b'{"x": 1}\n{"y": 2}\n')
    b.write_bytes(b'{"x": 1}\r\n{"y": 2}\r\n')
    assert s0._sha(str(a)) == s0._sha(str(b))
