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


def _conf_or_floor(p):
    """A prediction's confidence, or -inf when it genuinely has none."""
    c = prediction_confidence(p)
    return -math.inf if c is None else c


@pytest.mark.parametrize("split,model", CASES)
def test_observed_metrics_match_aggregate(split, model):
    """All-ones weights must reproduce the repo's own scorer to machine precision."""
    records, gts = bp.load_split(REPO, split)
    s = bp.score_model(REPO, split, model, records, gts, RSQ)
    dets = bp.detections_for(REPO, split, model, records)
    thr = bp.protocol_threshold(model)

    # `if c is None`, not `or`: a legitimate confidence of exactly 0.0 is falsy, so
    # `or -1e9` would drop a real detection rather than an unscored one. Inert at the
    # 0.25/0.30 protocol points, wrong the moment this is reused at a 0.0 floor.
    scores = [score_pano([p for p in dets.get(pid, [])
                          if _conf_or_floor(p) >= thr],
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


def test_rampnet_family_reads_at_the_protocol_point():
    """Every RampNet arm must get 0.30, including epoch dumps that do not exist yet.

    The hazard this guards is silent: a `run_a_epoch_9` falling through to the YOLO
    0.25 would not error, it would report a large capability gap that is really an
    operating-point gap — exactly the confound the #84 amendment added the
    calibration-free column to avoid.
    """
    for model in ("rampnet", "rampnet_1pass", "run_a_epoch_1", "run_a_epoch_8",
                  "run_a_epoch_99"):
        assert bp.protocol_threshold(model) == 0.30, model
    for model in ("y11l_pano", "y11x_pano_h200", "y26_pano"):
        assert bp.protocol_threshold(model) == 0.25, model


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

    # The measured epoch-pair matrix: 28 pairs from 8 checkpoints, and a paired s.e.
    # comfortably below the 0.01 unpaired tie bar. Pinned because the whole
    # recommendation rests on it.
    m = out["measured_epoch_pairs"]
    assert len(m["pairs"]) == 28
    assert m["se_max_f1_max"] < 0.005
    assert 0.02 < m["discordance_min"] < m["discordance_max"] < 0.10
    # s.e. must grow with epoch separation -- the trend the Run B planning number
    # (~0.003 at larger separation) is extrapolated from.
    gap1 = [v["delta"]["max_f1"]["se"] for v in m["pairs"].values() if v["gap"] == 1]
    far = [v["delta"]["max_f1"]["se"] for v in m["pairs"].values() if v["gap"] >= 5]
    assert sum(far) / len(far) > sum(gap1) / len(gap1)


#: Every number the doc's headline tables and the PR body quote, to the precision they
#: are quoted at. The gap this closes: the previous version of the test below pinned
#: three inventory integers and a handful of inequalities, so the doc could -- and did
#: -- drift away from the artifact in the fourth decimal without anything noticing.
#: Keyed by the doc's own wording so a failure says which sentence to fix.
DOC_HEADLINE = {
    # "The answer, up front" -- all four rows are max-F1, matching the header.
    "unpaired manual_gold max-F1 s.e.": 0.0041,
    "unpaired POOLED all max-F1 s.e.": 0.0039,
    # The unpaired F1 column, which the same table used to print under a max-F1 header.
    "unpaired manual_gold F1 s.e.": 0.0042,
    "unpaired POOLED all F1 s.e.": 0.0039,
    # The measured paired bracket -- the recommendation is stated in these.
    "paired s.e. min": 0.0016,
    "paired s.e. max": 0.0029,
    "paired s.e. median": 0.0021,
    "paired s.e. median, gap >= 3": 0.0022,
}


def test_committed_json_matches_the_doc_headline_numbers():
    """Every value the doc quotes in a headline table, pinned to 4 decimals.

    Prior review finding 7e: the test below guards the artifact's *shape* and the
    findings' *direction*, but pinned none of the numbers the doc actually prints, so
    a regeneration that moved them left the prose silently stale. Four of them were
    in fact stale when this was added.
    """
    import json
    path = os.path.join(REPO, "docs", "data", "benchmark_power_135.json")
    if not os.path.exists(path):
        pytest.skip("benchmark_power_135.json not committed")
    with open(path, encoding="utf-8") as fh:
        out = json.load(fh)

    m = out["measured_epoch_pairs"]
    actual = {
        "unpaired manual_gold max-F1 s.e.": out["unpaired"]["manual_gold"]["max_f1"]["se"],
        "unpaired POOLED all max-F1 s.e.": out["unpaired"]["POOLED all"]["max_f1"]["se"],
        "unpaired manual_gold F1 s.e.": out["unpaired"]["manual_gold"]["f1"]["se"],
        "unpaired POOLED all F1 s.e.": out["unpaired"]["POOLED all"]["f1"]["se"],
        "paired s.e. min": m["se_max_f1_min"],
        "paired s.e. max": m["se_max_f1_max"],
        "paired s.e. median": m["se_max_f1_median"],
        "paired s.e. median, gap >= 3": m["se_max_f1_median_gap_ge_3"],
    }
    for name, quoted in DOC_HEADLINE.items():
        assert round(actual[name], 4) == quoted, (
            f"{name}: artifact has {actual[name]:.6f} (rounds to "
            f"{round(actual[name], 4)}), the doc quotes {quoted}")

    # The MDE column beside them is Z_MDE x s.e., and the doc prints it rounded.
    assert round(bp.Z_MDE * actual["unpaired manual_gold max-F1 s.e."], 4) == 0.0114
    assert round(bp.Z_MDE * actual["unpaired POOLED all max-F1 s.e."], 4) == 0.0108
    assert round(bp.Z_MDE * actual["paired s.e. median, gap >= 3"], 4) == 0.0063
    assert round(bp.Z_MDE * actual["paired s.e. min"], 4) == 0.0045
    assert round(bp.Z_MDE * actual["paired s.e. max"], 4) == 0.0082


def test_committed_json_records_the_bundle_truncation():
    """The nine city bundles stop at 0.55, so the named 0.30 point is a no-op there.

    Prior review finding 1. This was true, load-bearing for the pooling claim, and
    only *indirectly* visible in the artifact (max_f1 == f1 exactly, self_pair b = c =
    0) for two review passes. The inventory now records it directly, and this pins
    that it keeps doing so -- an artifact that stopped saying which rows are truncated
    would let the caveat quietly fall out of the doc again.
    """
    import json
    path = os.path.join(REPO, "docs", "data", "benchmark_power_135.json")
    if not os.path.exists(path):
        pytest.skip("benchmark_power_135.json not committed")
    with open(path, encoding="utf-8") as fh:
        out = json.load(fh)
    if out.get("reference") != "rampnet":
        pytest.skip("truncation is a property of the published bundles, not op_cache")

    inv = out["inventory"]
    assert inv["manual_gold"]["protocol_threshold_binds"] is True
    assert inv["manual_gold"]["reference_min_confidence"] < 0.10
    cities = [k for k, v in inv.items() if v["gt_source"] == "anchored"]
    assert len(cities) == 9
    for c in cities:
        assert inv[c]["protocol_threshold_binds"] is False, (
            f"{c} now holds detections below 0.30 -- the truncation caveat in "
            f"docs/stage2_run_b_power_135.md needs rewriting, not deleting")
        assert inv[c]["reference_min_confidence"] > 0.55 - 1e-2

    # Both pooled rows inherit it, and the artifact says which members.
    assert set(out["unpaired"]["POOLED cities"]["truncated_members"]) == set(cities)
    assert set(out["unpaired"]["POOLED all"]["truncated_members"]) == set(cities)
    assert out["unpaired"]["manual_gold"]["truncated_members"] == []

    # And the self-pair block no longer reports a degenerate max-F1 as a measurement.
    for key, row in out["self_pair"].items():
        assert row["max_f1"] is None, f"{key}: max-F1 is not a function of the shift"
        assert "max_f1_note" in row


#: The one Run A max-F1 that #140 moved. ``summary.csv`` was written on 2026-08-18
#: 14:24, about two hours before the seam wrap merged (bf64451, 16:30), so it records
#: the curve under the non-wrapping matcher. Re-scored with wrapping, exactly one of
#: the eight epochs changes -- epoch 7, by 0.000264, because one prediction there now
#: claims a ground-truth ramp across the 360 seam. Pinned rather than tolerated: a
#: second epoch starting to move would be a real change and has to fail the build.
SEAM_FIXED_MAX_F1 = {7: 0.910745}


def _epoch_max_f1(records, gts, epoch, wrap_x):
    """max-F1 for one committed epoch dump, under the chosen matcher."""
    label = bp.RUN_A_EPOCH.format(epoch)
    if bp.detections_for(REPO, "manual_gold", label, records) is None:
        return None
    s = bp.score_model(REPO, "manual_gold", label, records, gts, RSQ, wrap_x=wrap_x)
    return float(np.array(bp.metrics(s, np.ones((1, len(s.pids))), 0.30)).ravel()[3])


def test_run_a_epoch_dumps_are_the_committed_curve():
    """Every epoch dump must reproduce summary.csv's max-F1, under its own matcher.

    dump_peaks_from_cache.py checks this at write time, but the check has to survive
    into the repo: these files are committed precisely because the 13 GB heatmap cache
    they came from cannot be, so nothing else can re-derive them.

    The matcher is passed explicitly because summary.csv predates #140. Checking the
    historical file against the historical matcher keeps this exact to 1e-5 -- a
    tolerance loose enough to absorb the seam would also absorb a real regression.
    """
    import csv
    summary_path = os.path.join(REPO, "docs", "data", "run_a_84_manual_gold", "summary.csv")
    with open(summary_path, encoding="utf-8") as fh:
        rows = {int(r["epoch"]): r for r in csv.DictReader(fh)}

    records, gts = bp.load_split(REPO, "manual_gold")
    checked = 0
    for epoch, row in rows.items():
        max_f1 = _epoch_max_f1(records, gts, epoch, wrap_x=False)
        if max_f1 is None:
            continue
        assert max_f1 == pytest.approx(float(row["max_f1"]), abs=1e-5), epoch
        checked += 1
    assert checked == 8, f"expected 8 committed epoch dumps, found {checked}"


def test_seam_wrap_moves_exactly_one_epoch_of_the_run_a_curve():
    """What #140 did to the curve this analysis is built on, measured and pinned.

    The analysis runs at ``bp.WRAP_X``; the committed file does not. That difference
    is a fact about the data, so it is asserted rather than described: every epoch but
    one is unchanged within summary.csv's own rounding, and epoch 7 moves by the
    amount recorded above. The
    conclusion in docs/stage2_run_b_power_135.md is unaffected -- 0.000264 is 4% of
    the 0.0063 paired MDE it rests on, and it moves epoch 7 further down, not up.
    """
    import csv
    summary_path = os.path.join(REPO, "docs", "data", "run_a_84_manual_gold", "summary.csv")
    with open(summary_path, encoding="utf-8") as fh:
        rows = {int(r["epoch"]): r for r in csv.DictReader(fh)}

    records, gts = bp.load_split(REPO, "manual_gold")
    moved = {}
    for epoch, row in rows.items():
        wrapped = _epoch_max_f1(records, gts, epoch, wrap_x=bp.WRAP_X)
        if wrapped is None:
            continue
        historical = float(row["max_f1"])
        # summary.csv is written to 6 decimals, so 5e-7 is pure rounding; 1e-6
        # sits above that and 264x below the epoch-7 move, a clean separation.
        if wrapped != pytest.approx(historical, abs=1e-6):
            moved[epoch] = wrapped

    assert sorted(moved) == sorted(SEAM_FIXED_MAX_F1), moved
    for epoch, expected in SEAM_FIXED_MAX_F1.items():
        assert moved[epoch] == pytest.approx(expected, abs=1e-6), epoch
        # Downward: the wrap gives a seam ramp to a prediction that had been a false
        # positive on one side and a miss on the other, which cannot raise max-F1 here.
        assert moved[epoch] < float(rows[epoch]["max_f1"])
