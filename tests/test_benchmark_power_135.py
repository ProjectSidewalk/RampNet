"""Unit tests for the Run B power analysis (#135).

The arithmetic here is a bootstrap, so most of it is self-evidently right or
self-evidently wrong. The places a *silent* wrong answer is possible are three:

- **The observed path must agree with the repo's own scorer.** Everything in the
  analysis is a weighted sum over panoramas, and the all-ones weight row has to
  reproduce ``rampnet.detection_eval.aggregate`` exactly. If it does not, every
  standard error is a standard error of the wrong statistic — and it would still
  look plausible, because the numbers would be close.
- **The matcher detail must agree with the matcher.** ``match_detail`` keeps the
  ground-truth index that ``score_pano`` throws away, which the McNemar
  decomposition needs. The script asserts agreement at runtime on every pano; this
  pins that the assertion is real by checking a case with duplicates and ignores.
- **Stratified resampling must preserve split sizes.** A pooled bootstrap that let
  one split's panorama count drift would silently reweight the pooled statistic.

The greedy matching geometry itself lives in ``rampnet/metrics.py`` and is covered by
tests/test_metrics.py — deliberately not re-tested here.
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import benchmark_power_135 as bp  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    GroundTruth, aggregate, prediction_confidence, radius_sq_for, score_pano,
)

RSQ = radius_sq_for()

# Small splits only: the point is agreement with the reference scorer, which does not
# depend on split size, and the suite is meant to stay quick. One anchored-GT bundle
# scored against RampNet's own detections, one against a published YOLO export, so
# both ground-truth paths and both detection sources are exercised.
CASES = [("clovis", "rampnet"), ("paterson", "y11l_pano"), ("annapolis", "y11x_pano_h200")]


@pytest.mark.parametrize("split,model", CASES)
def test_observed_metrics_match_aggregate(split, model):
    """All-ones weights must reproduce the repo's own scorer to machine precision."""
    records, gts = bp.load_split(REPO, split)
    s = bp.score_model(REPO, split, model, records, gts, RSQ)
    dets = bp.detections_for(REPO, split, model, records)
    thr = bp.protocol_threshold(model)

    scores = [score_pano([p for p in dets.get(pid, [])
                          if (prediction_confidence(p) or -1e9) >= thr],
                         gts[pid], radius_sq=RSQ)
              for pid in sorted(gts)]
    ref = aggregate(scores)

    p, r, f1, mx = (float(v) for v in
                    np.array(bp.metrics(s, np.ones((1, len(s.pids))), thr)).ravel())
    assert p == pytest.approx(ref.precision, abs=1e-12)
    assert r == pytest.approx(ref.recall, abs=1e-12)
    assert f1 == pytest.approx(ref.f1, abs=1e-12)
    # max-F1 sweeps the whole curve, so it can never be below F1 at a fixed point.
    assert mx >= f1 - 1e-12


def test_match_detail_agrees_with_score_pano_on_duplicates_and_ignores():
    """The case the runtime assertion exists to catch.

    Two predictions on one ramp (the second is a false positive, not a second true
    positive), one prediction on an ``unsure`` mark (neither), one with nothing near
    it (a false positive). If ``match_detail`` diverged from ``score_pano`` on any of
    those, the McNemar counts would be built on a different matching than the
    precision/recall they are reported beside.
    """
    gt = GroundTruth(gt_points=[(0.5, 0.5)], ignore_points=[(0.2, 0.5)], fn_confirmed=True)
    preds = [(0.5, 0.5, 0.9), (0.5005, 0.5, 0.8), (0.2, 0.5, 0.7), (0.8, 0.5, 0.6)]

    detail = bp.match_detail(preds, gt, RSQ)
    ref = score_pano(preds, gt, radius_sq=RSQ)

    assert sum(1 for _, tp, _ in detail if tp) == ref.tp == 1
    assert sum(1 for _, tp, _ in detail if not tp) == ref.fp == 2
    assert len(detail) == ref.tp + ref.fp        # the ignored prediction is dropped
    assert [k for _, tp, k in detail if tp] == [0]


def test_bootstrap_weights_preserve_split_sizes():
    """Stratified resampling: each split contributes exactly its own panorama count."""
    rng = np.random.default_rng(0)
    sizes = [125, 110, 1000]
    w = bp.bootstrap_weights(rng, sizes, 32)
    assert w.shape == (32, sum(sizes))
    start = 0
    for n in sizes:
        assert np.all(w[:, start:start + n].sum(axis=1) == n)
        start += n
    assert w.sum(axis=1).min() == sum(sizes)


def test_required_discordance_inverts():
    """The inversion must round-trip against the McNemar s.e. it is derived from."""
    n, deff, z = 3919, 1.15, bp.Z_CI
    need, rate = bp.required_discordance(0.0171, n, deff, z=z)
    assert rate == pytest.approx(need / n)
    # At exactly the required discordance the gap sits on the significance bar.
    se = (deff * need) ** 0.5 / n
    assert 0.0171 / se == pytest.approx(z)
    # A zero gap has no required discordance -- it is unreadable at any n, so the
    # inversion must say "undefined" rather than return 0 and read as "resolvable".
    assert math.isnan(bp.required_discordance(0.0, n, deff)[0])


def test_seam_enrichment_flags_only_the_seam():
    """A pair disagreeing away from the seam must read the baseline, not above it."""
    class Fake:
        pass

    a, b = Fake(), Fake()
    # 100 instances, 4 of them at the seam. A and B disagree on 10, none at the seam.
    a.gt_x = np.concatenate([np.full(4, 0.005), np.linspace(0.2, 0.8, 96)])
    b.gt_x = a.gt_x
    a.hit_gt = np.full(100, 1.0)
    b.hit_gt = np.full(100, 1.0)
    b.hit_gt[10:20] = -np.inf

    n_disc, n_seam, baseline = bp.seam_enrichment(a, b, 0.5, 0.5)
    assert (n_disc, n_seam) == (10, 0)
    assert baseline == pytest.approx(0.04)

    # Now move the disagreement onto the seam instances.
    b.hit_gt = np.full(100, 1.0)
    b.hit_gt[:4] = -np.inf
    n_disc, n_seam, _ = bp.seam_enrichment(a, b, 0.5, 0.5)
    assert (n_disc, n_seam) == (4, 4)


def test_committed_json_matches_the_doc():
    """The headline numbers quoted in docs/stage2_run_b_power_135.md are the run's.

    Guards the failure this repo has hit before: a committed table drifting away from
    the artifact it was read out of, with nothing to notice.
    """
    import json
    path = os.path.join(REPO, "docs", "data", "benchmark_power_135.json")
    if not os.path.exists(path):
        pytest.skip("benchmark_power_135.json not committed")
    with open(path, encoding="utf-8") as fh:
        out = json.load(fh)

    assert out["inventory"]["manual_gold"]["n_gt_instances"] == 3919
    assert out["inventory"]["manual_gold"]["gt_source"] == "manual"
    assert sum(v["n_gt_instances"] for v in out["inventory"].values()) == 6560
    # Pooling all ten splits does not meaningfully beat manual_gold alone -- the
    # finding the doc rests on, pinned so a re-run that overturned it would fail here.
    mg = out["unpaired"]["manual_gold"]["f1"]["se"]
    pooled = out["unpaired"]["POOLED all"]["f1"]["se"]
    assert pooled > 0.85 * mg
