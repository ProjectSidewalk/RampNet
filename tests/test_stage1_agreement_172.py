"""Stage 1 agreement re-measurement (#172): the conventions, and the committed result.

CPU only, no network: the Stage 1 points of the gold panoramas are committed
(``analysis_out/stage1_agreement_172/stage1_gold_labels.json``), so the whole
measurement re-derives here, and the byte-for-byte test means a change to the shared
matcher that moves the corrected number fails CI rather than silently drifting the doc.
"""
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_agreement_172 as s1  # noqa: E402

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
    assert s1.greedy_match(pred, gt, R2, 1024, 512)[0][0] == 1


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
