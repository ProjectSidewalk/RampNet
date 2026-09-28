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
from rampnet.detection_eval import GroundTruth, radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
RESULTS = os.path.join(REPO, "analysis_out", "two_scale_197", "results.json")
RESULTS_196 = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "results.json")
US = "US pool (miss_decomposition.US_SPLITS)"
GSV = "GSV z5 (bend+paterson+gainesville+sao_paulo+laurens_gsv)"
GSV_NO_PATERSON = "GSV z5 minus paterson (post hoc)"


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
    # the control and the reference are #196's own 0.30 rows, read from #196's results
    with open(RESULTS_196, encoding="utf-8") as f:
        m196 = json.load(f)["per_split"]["richmond"]["metrics"]
    for arm in ("r2048", "u4096"):
        for k in ("P", "R", "F1", "tp", "fp"):
            assert committed["metrics"][arm][k] == m196[arm]["0.30"][k], (arm, k)


def test_inference_cost_from_the_ledger():
    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)["inference_cost"]
    got = ts.inference_cost()
    assert got == committed
    assert got["panos"] == {"r2048": 1289, "u4096": 1289}
    assert 4.5 < got["ratio_vs_r2048"] < 5.2


def test_inference_cost_ignores_later_rows_and_refuses_missing_ones(tmp_path):
    """The Q4 rows are pinned by (label, ts): a later full-size input-res-25 row must not
    move the number, and a ledger without one of the pinned rows must fail loudly."""
    with open(ts.USAGE_LOG, encoding="utf-8") as f:
        lines = [ln for ln in f if '"input-res-25:' in ln]
    extra = json.loads(next(ln for ln in lines if '"input-res-25:u4096"' in ln))
    extra.update(ts="2027-01-01T00:00:00Z", panos_scored=1289, elapsed_s=1.0)
    log = tmp_path / "usage_log.jsonl"
    log.write_text("".join(lines) + json.dumps(extra) + "\n", encoding="utf-8")
    assert ts.inference_cost(str(log)) == ts.inference_cost()
    pinned_r = '"ts": "%s"' % ts.COST_ROWS[0][1]
    log.write_text("".join(ln for ln in lines if pinned_r not in ln), encoding="utf-8")
    with pytest.raises(AssertionError):
        ts.inference_cost(str(log))


def _toy_pairs(u_peaks):
    """Three one-pano splits. Each has a near ramp that r2048 finds and a far ramp
    (y 0.51, ~80 m at the 2.5 m camera) that r2048 misses; ``u_peaks`` is u4096's list."""
    gt = GroundTruth(gt_points=[(0.2, 0.80), (0.6, 0.51)], ignore_points=[],
                     fn_confirmed=True)
    r = [(0.2, 0.80, 0.9)]
    return {c: [(f"{c}_pano", gt, list(r), list(u_peaks))] for c in ("a", "b", "c")}


def test_loso_settings_picks_the_first_strictly_better_setting():
    """A u4096 peak at 0.35 on the far ramp: every setting with t_u <= 0.30 recovers it
    and they tie, so the first on the grid (D 12 m, t_u 0.20) wins; t_u 0.40 does not."""
    chosen, table, margin = ts.loso_settings(_toy_pairs([(0.6, 0.51, 0.35)]), RSQ)
    assert set(chosen.values()) == {(12.0, 0.2)}
    assert all(abs(m - (1.0 - 2 / 3)) < 1e-9 for m in margin.values())   # F1 1 vs 2/3
    grid = {(e["D_m"], e["t_u"]): e["F1_other_splits"] for e in table["a"]}
    assert grid[(None, None)] == round(2 / 3, 4) and grid[(40.0, 0.4)] == round(2 / 3, 4)


def test_loso_settings_keeps_no_fusion_on_a_tie():
    """No u4096 peaks at all: every setting scores the same, and "no fusion" (listed
    first) is kept, with a margin of exactly 0."""
    chosen, _, margin = ts.loso_settings(_toy_pairs([]), RSQ)
    assert set(chosen.values()) == {(None, None)}
    assert set(margin.values()) == {0.0}


def test_report_check_reproduces_the_committed_outputs():
    """Drift guard (about a minute, like #196's verdicts test): every committed number,
    verdict, LOSO choice, control, band CI and diagnostic re-derives byte for byte."""
    assert ts.main(["report", "--check"]) == 0


@pytest.mark.parametrize("pool", ["US pool (miss_decomposition.US_SPLITS)"])
def test_committed_verdicts(pool):
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    rules = rep["pooled"][pool]["rules"]
    assert rules["R1"]["verdict"] == "HURTS"
    assert rules["R2"]["verdict"] == "NO BETTER THAN A THRESHOLD"
    assert rules["R3"]["verdict"] == "NO BETTER THAN A THRESHOLD"
    assert all(e["chosen"] == {"D_m": 40.0, "t_u": 0.4} for e in rep["loso"].values())


def test_gsv_recall_lever_rests_on_paterson():
    """PR #199 review: the GSV pool's R2 RECALL LEVER does not survive dropping paterson
    (post hoc pool), and paterson's R3 HELPS rests on a LOSO margin of ~2e-6 F1."""
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["pooled"][GSV]["rules"]["R2"]["verdict"] == "RECALL LEVER"
    post = rep["pooled_post_hoc"][GSV_NO_PATERSON]
    assert post["members"] == ["bend", "gainesville", "sao_paulo", "laurens_gsv"]
    assert post["rules"]["R2"]["verdict"] == "NO BETTER THAN A THRESHOLD"
    assert post["rules"]["R2"]["matched_recall_baseline"]["fused_minus_matched"]["f1"][
        "ci_hi"] > 0
    assert rep["per_split"]["paterson"]["rules"]["R3"]["verdict"] == "HELPS"
    assert 0 < rep["loso"]["paterson"]["F1_margin_over_no_fusion"] < 1e-5
    others = [e["F1_margin_over_no_fusion"] for c, e in rep["loso"].items() if c != "paterson"]
    assert min(others) > 0.001


def test_194_matched_definition_changes_no_verdict():
    """#194's matched-recall definition (best-F1 threshold reaching the recall) favours the
    baseline at least as much as the plan's; on these data it changes no verdict."""
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    groups = list(rep["per_split"].values()) + list(rep["pooled"].values()) + list(
        rep["pooled_post_hoc"].values())
    for g in groups:
        for v in ("R1", "R2", "R3"):
            r = g["rules"][v]
            m1, m2 = r["matched_recall_baseline"], r["matched_recall_194_definition"]
            assert (m1 is None) == (m2 is None)
            if m1 is not None:
                assert m2["F1"] >= m1["F1"]
                assert m2["verdict_if_used"] == r["verdict"]
