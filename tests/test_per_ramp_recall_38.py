"""Tests for scripts/analysis/per_ramp_recall_38.py (#38).

CPU only, no network: toy checks of the correlation design, the permutation null and the
thinning rule, plus a re-derivation of the committed headline counts from the committed
``analysis_out/multiview_48/captures_R25.csv``.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import per_ramp_recall_38 as pr  # noqa: E402

RESULTS = os.path.join(REPO, "analysis_out", "per_ramp_recall_38", "results.json")


def _cap(pid, dist, conf, src=False, e=0.0, n=0.0, date="2025-01"):
    return {"pano_id": pid, "dist_m": dist, "is_source": src, "world_conf": conf,
            "capture_date": date, "cam_e": e, "cam_n": n}


def _ramp(uid, caps):
    return {"uid": uid, "city": uid.split(":")[0], "captures": caps}


def test_independent_design_predicts_the_observed_count():
    # two strata, each with miss rate 1/2; four ramps covering every miss pattern once,
    # so misses are exactly independent and observed == predicted (1 of 4 all-missed)
    rows = [("c", [(("c", "o", 0), a), (("c", "o", 1), b)])
            for a in (True, False) for b in (True, False)]
    d = pr.Design(rows, [("c", "o", 0), ("c", "o", 1)])
    obs, pred, _ = d.stats()
    assert obs == 1 and pred == pytest.approx(1.0)


def test_perfectly_correlated_misses_give_a_ratio_above_one():
    rows = [("c", [(("c", "o", 0), True), (("c", "o", 0), True)]),
            ("c", [(("c", "o", 0), False), (("c", "o", 0), False)])]
    d = pr.Design(rows, [("c", "o", 0)])
    obs, pred, _ = d.stats()
    assert obs == 1 and pred == pytest.approx(0.5)   # 2 * 0.5^2


def test_bootstrap_weights_reestimate_the_marginals():
    rows = [("c", [(("c", "o", 0), True)]), ("c", [(("c", "o", 0), False)])]
    d = pr.Design(rows, [("c", "o", 0)])
    obs, pred, _ = d.stats(np.array([2.0, 0.0]))     # resample = the missed ramp twice
    assert obs == 2 and pred == pytest.approx(2.0)


def test_permutation_keeps_each_strata_miss_count():
    rng = np.random.default_rng(0)
    rows = [("c", [(("c", "o", 0), True), (("c", "o", 1), False)]),
            ("c", [(("c", "o", 0), False), (("c", "o", 1), True)]),
            ("c", [(("c", "o", 0), True), (("c", "o", 1), True)])]
    d = pr.Design(rows, [("c", "o", 0), ("c", "o", 1)])
    null = d.permutation_null(200, rng)
    # each stratum keeps 2 misses among 3 ramps, so the two miss sets always overlap in 1
    # or 2 ramps; any other count would mean the shuffle changed a stratum's miss count
    assert set(null.tolist()) <= {1.0, 2.0}


def test_grid_thin_keeps_the_newest_pano_per_cell():
    rng = np.random.default_rng(0)
    panos = {"old": (1.0, 1.0, "2020-01"), "new": (2.0, 2.0, "2024-06"),
             "far": (12.0, 1.0, "2019-01")}
    kept = pr.grid_thin(panos, 10.0, (0.0, 0.0), rng)
    assert kept == {"new", "far"}


def test_others_excludes_the_source_view_and_sorts_by_distance():
    r = _ramp("x:1", [_cap("s", 3.0, 0.9, src=True), _cap("b", 9.0, None),
                      _cap("a", 4.0, 0.7), _cap("z", 19.0, 0.8)])
    assert [c["pano_id"] for c in pr.others(r)] == ["a", "b"]


def test_committed_headline_rederives_from_the_capture_table():
    ramps = sorted(pr.load_captures(pr.CAPTURES).values(), key=lambda r: r["uid"])
    rng = np.random.default_rng(0)
    blk, _, _ = pr.correlation_block(ramps, 0.55, rng, n_boot=0, n_perm=0)
    # docs/multiview_48.md section 5
    assert blk["observed_all_missed"] == 153
    assert blk["predicted_independent"] == pytest.approx(76.5197, abs=1e-3)
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    c = res["correlation"]["pooled_055"]
    assert c["observed_all_missed"] == 153 and c["ramps"] == blk["ramps"]
    u, _, _ = pr.correlation_block(ramps, 0.55, rng, include_source=True, n_boot=0, n_perm=0)
    assert res["correlation"]["pooled_055_union_with_source"]["observed_all_missed"] == \
        u["observed_all_missed"]


def test_committed_results_are_lf_pinned():
    with open(RESULTS, "rb") as f:
        assert b"\r\n" not in f.read()


def test_doc_quotes_every_generated_table_row_verbatim():
    # S4 of the PR #231 review: the doc's numbers must come from results.json
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    with open(os.path.join(REPO, "docs", "per_ramp_recall_38.md"), encoding="utf-8") as f:
        doc = f.read()
    missing = [row for rows in pr.doc_tables(res).values() for row in rows if row not in doc]
    assert not missing, missing[:3]


def test_floor_counts_every_view_out_to_25m():
    with open(RESULTS, encoding="utf-8") as f:
        fl = json.load(f)["all_missed_055"]["floor"]
    assert fl["missed_by_every_view_25m"] == 46
    assert fl["missed_by_every_view_18m"] == 55
    assert fl["of_those_18m_found_at_18_25m"] == 9


def test_floor_flag_uses_views_beyond_18m():
    r = _ramp("bend:1", [_cap("s", 10.0, None, src=True), _cap("a", 5.0, None),
                      _cap("b", 9.0, None), _cap("far", 22.0, 0.9)])
    row = pr.all_missed_table([r], 0.55, {})[0]
    assert row["all_other_missed"] and row["other_hit_18_25m"]
    assert not row["all_views_missed_25m"]


def test_cluster_bootstrap_draws_a_cluster_whole():
    # cluster "a" = one missed ramp + one found ramp, cluster "b" = one missed ramp. Two
    # clusters are drawn per resample, and each draw adds exactly one missed ramp, so the
    # all-missed count is always 2; a ramp-level bootstrap would give 0-3.
    rows = [("c", [(("c", "o", 0), True)]), ("c", [(("c", "o", 0), False)]),
            ("c", [(("c", "o", 0), True)])]
    d = pr.Design(rows, [("c", "o", 0)])
    rng = np.random.default_rng(1)
    clustered = {float(o) for o, _ in d.bootstrap(200, rng, clusters=["a", "a", "b"])}
    plain = {float(o) for o, _ in d.bootstrap(200, rng)}
    assert clustered == {2.0}
    assert len(plain) > 1
