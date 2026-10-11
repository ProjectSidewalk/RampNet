"""Stage 1 agreement re-measurement (#172): the conventions, and the committed result.

CPU only, no network: the Stage 1 points of the gold panoramas are committed
(``analysis_out/stage1_agreement_172/stage1_gold_labels.json``), so the whole
measurement re-derives here. The byte-for-byte test catches a stale committed result; the
hard-coded counts catch a matcher change followed by a regeneration, which would otherwise
leave the docs quoting a number the JSON no longer holds.
"""
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_agreement_172 as s1  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

R2 = s1.radius_sq_for(s1.PANO_RADIUS_NORMALIZED)   # (0.022 * 1024)^2 = 22.528^2


def test_published_and_shared_conventions_differ_on_the_two_documented_cases():
    # Two ramps 20 px apart in x (both within radius of the first prediction); the first
    # prediction sits nearer the SECOND ramp; the second prediction is in range of the first only.
    gt = [(0.5, 0.5), (0.5 + 20 / 1024, 0.5)]
    pred = [(0.5 + 15 / 1024, 0.5), (0.5 - 10 / 1024, 0.5)]
    # v1.0: pred 1 claims gt 0 (first in file order), pred 2 sees only a claimed ramp -> ignored.
    assert s1.score_pano(pred, gt, "published", R2) == (1, 0, 1, 2)
    # #18 half one: the ignored point is a false positive.
    assert s1.score_pano(pred, gt, "published_redund_fp", R2) == (1, 1, 0, 2)
    # #18 half two: nearest unclaimed -> pred 1 claims gt 1, pred 2 claims gt 0: two TPs.
    assert s1.score_pano(pred, gt, "shared_nowrap", R2) == (2, 0, 0, 2)
    assert s1.score_pano(pred, gt, "shared", R2) == (2, 0, 0, 2)


def test_seam_wrap_is_the_only_difference_between_the_two_shared_conventions():
    gt = [(0.002, 0.5)]
    pred = [(0.998, 0.5)]           # ~4 px away across the seam, ~1020 px the other way
    assert s1.score_pano(pred, gt, "shared_nowrap", R2) == (0, 1, 0, 1)
    assert s1.score_pano(pred, gt, "shared", R2) == (1, 0, 0, 1)
    assert s1.score_pano(pred, gt, "published", R2) == (0, 1, 0, 1)


def test_match_first_in_order_is_first_not_nearest():
    gt = [(0.5, 0.5), (0.5 + 5 / 1024, 0.5)]
    pred = [(0.5 + 6 / 1024, 0.5)]  # nearest is gt 1, but v1.0 takes gt 0
    assert s1.match_first_in_order(pred, gt, R2) == (1, 0, 0)
    assert s1.score_pano(pred, gt, "shared", R2) == (1, 0, 0, 2)
    assert greedy_match(pred, gt, R2, 1024, 512)[0][0] == 1


def test_max_matching_beats_a_bad_order_and_bounds_the_shared_matcher():
    # Pred 0 is in range of both ramps and nearest gt 0; pred 1 reaches only gt 0. In this
    # order the shared matcher gives 1 TP; the other order, and the max matching, give 2.
    gt = [(0.5, 0.5), (0.5 + 18 / 1024, 0.5)]
    pred = [(0.5 + 8 / 1024, 0.5), (0.5 - 10 / 1024, 0.5)]
    assert s1.score_pano(pred, gt, "shared", R2)[0] == 1
    assert s1.score_pano(pred[::-1], gt, "shared", R2)[0] == 2
    assert s1.max_matching_tp(pred, gt, R2) == 2
    # Seam: the max matching folds x the way the shared matcher does.
    assert s1.max_matching_tp([(0.998, 0.5)], [(0.002, 0.5)], R2) == 1
    assert s1.max_matching_tp([(0.998, 0.5)], [(0.002, 0.5)], R2, wrap_x=False) == 0


def test_aggregate_and_prf():
    agg = s1.aggregate([(3, 1, 2, 4), (0, 0, 0, 0), (1, 1, 0, 2)])
    assert (agg["tp"], agg["fp"], agg["ignored"], agg["fn"], agg["n_gt"], agg["n_pred"]) == (4, 2, 2, 2, 6, 8)
    assert agg["precision"] == pytest.approx(4 / 6)
    assert agg["recall"] == pytest.approx(4 / 6)


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    per_pano = [(3, 1, 0, 4), (2, 0, 0, 2), (0, 2, 0, 1), (4, 0, 0, 4)] * 10
    a = s1.bootstrap_ci(per_pano, n=200, seed=1)
    b = s1.bootstrap_ci(per_pano, n=200, seed=1)
    assert a == b
    agg = s1.aggregate(per_pano)
    assert a["precision"][0] <= agg["precision"] <= a["precision"][1]
    assert a["recall"][0] <= agg["recall"] <= a["recall"][1]


def test_committed_result_rederives_byte_for_byte():
    """The number of record, from the committed inputs, with nothing else on disk."""
    for p in (s1.LABELS_JSON, s1.RESULT_JSON, s1.SUMMARY_MD):
        assert os.path.exists(p), p
    with tempfile.TemporaryDirectory() as tmp:
        out_json = os.path.join(tmp, "result.json")
        out_md = os.path.join(tmp, "summary.md")
        s1.main(["score", "--json-out", out_json, "--summary-out", out_md])
        for got, want in ((out_json, s1.RESULT_JSON), (out_md, s1.SUMMARY_MD)):
            with open(got, "rb") as f1, open(want, "rb") as f2:
                assert f1.read() == f2.read(), f"{want} is stale: re-run `stage1_agreement_172.py score`"


def test_committed_result_reproduces_the_paper_under_its_own_convention():
    """The v1.0 convention on the committed inputs gives the paper's counts exactly, which is
    what says the fetched labels are the ones the paper scored."""
    import json
    with open(s1.RESULT_JSON, encoding="utf-8") as fh:
        r = json.load(fh)
    pub = r["conventions"]["published"]
    assert (pub["tp"], pub["fp"], pub["ignored"], pub["n_gt"]) == (3623, 230, 119, 3919)
    assert pub["precision"] == pytest.approx(0.9403, abs=5e-5)
    assert pub["recall"] == pytest.approx(0.9245, abs=5e-5)
    assert r["conventions"]["published_redund_fp"]["precision"] == pytest.approx(0.9121, abs=5e-5)
    assert r["n_panos"] == 1000


def test_committed_result_holds_the_corrected_counts_the_docs_quote():
    """Hard-coded, so regenerating the JSON after a matcher change cannot pass silently: the
    README, rampnet1_findings.md and stage1_generation_cost.md quote these numbers."""
    import json
    with open(s1.RESULT_JSON, encoding="utf-8") as fh:
        r = json.load(fh)
    for conv in ("shared", "shared_nowrap"):
        c = r["conventions"][conv]
        assert (c["tp"], c["fp"], c["ignored"], c["fn"], c["n_gt"]) == (3635, 337, 0, 284, 3919), conv
        assert c["precision"] == pytest.approx(0.9152, abs=5e-5), conv
        assert c["recall"] == pytest.approx(0.9275, abs=5e-5), conv
    ci = r["conventions"]["shared"]["ci95"]
    assert [round(v, 3) for v in ci["precision"]] == [0.904, 0.925]
    assert [round(v, 3) for v in ci["recall"]] == [0.918, 0.937]
    assert r["matching_change"] == {"n_panos_changed": 14, "tp_gained": 13, "tp_lost": 1, "tp_net": 12}
    o = r["order_sensitivity"]
    assert (o["tp_stored_order"], o["tp_shuffle_min"], o["tp_shuffle_max"], o["tp_max_matching"]) ==         (3635, 3635, 3637, 3637)
    assert r["n_dropped_out_of_range"] == 0
