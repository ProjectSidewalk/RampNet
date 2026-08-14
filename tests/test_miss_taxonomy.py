"""Unit tests for the miss taxonomy (#46).

Pure logic only — no cache on disk, no network, no imagery. The load-bearing
guarantees are the ones the headline number rests on:

* the cascade is **exhaustive and mutually exclusive** — every miss gets exactly
  one cause, so the buckets can be summed against #59's near-field figure;
* ``localization`` really does require an *unclaimed* prediction, because counting
  a neighbour's true positive as evidence about this ramp is what inflated that
  bucket six-fold in the first draft;
* ``optimal_hits`` is a genuine upper bound on the greedy matcher, which is what
  licenses the "the matcher is not manufacturing misses" claim;
* the null wraps at the panorama seam, without which it under-counts near x=0/1
  and flatters every in-radius bucket.
"""
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import miss_taxonomy as mt  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
R_NORM = RSQ ** 0.5 / PANO_SCALE_X      # match radius back in normalized x units
GT = (0.5, 0.6)


def _near(dx_frac, conf):
    """A prediction ``dx_frac`` of the match radius away from GT in x."""
    return (GT[0] + dx_frac * R_NORM, GT[1], conf)


# --------------------------------------------------------------------------- #
# classify_miss — the cascade
# --------------------------------------------------------------------------- #
def test_supra_threshold_peak_in_radius_is_merged():
    p = _near(0.5, 0.9)
    assert mt.classify_miss(GT, [p], [p], {0: 3}, RSQ, 0.30) == "merged"


def test_sub_threshold_peak_in_radius_is_sub_threshold():
    p = _near(0.5, 0.12)
    assert mt.classify_miss(GT, [p], [], {}, RSQ, 0.30) == "sub_threshold"


def test_unclaimed_peak_in_the_annulus_is_localization():
    p = _near(1.5, 0.9)                       # outside R, inside 2R
    assert mt.classify_miss(GT, [p], [p], {0: -1}, RSQ, 0.30) == "localization"


def test_annulus_peak_owned_by_a_neighbour_is_silent_not_localization():
    # The exact confound that inflated this bucket 6x: a prediction near a missed
    # ramp is evidence about that ramp only if no other ramp already claimed it.
    p = _near(1.5, 0.9)
    assert mt.classify_miss(GT, [p], [p], {0: 7}, RSQ, 0.30) == "silent"


def test_nothing_anywhere_is_silent():
    assert mt.classify_miss(GT, [], [], {}, RSQ, 0.30) == "silent"


def test_peak_beyond_the_annulus_is_silent():
    p = _near(2.5, 0.9)
    assert mt.classify_miss(GT, [p], [p], {0: -1}, RSQ, 0.30) == "silent"


def test_merged_outranks_sub_threshold_when_both_are_present():
    # Ordered cascade. Pooled this never fires (0 of 124 merged misses carry their
    # own sub-threshold peak), but the order must still be deterministic.
    hi, lo = _near(0.4, 0.9), _near(0.6, 0.10)
    assert mt.classify_miss(GT, [hi, lo], [hi], {0: 3}, RSQ, 0.30) == "merged"


def test_threshold_moves_a_peak_between_merged_and_sub_threshold():
    p = _near(0.5, 0.40)
    assert mt.classify_miss(GT, [p], [p], {0: 3}, RSQ, 0.30) == "merged"
    assert mt.classify_miss(GT, [p], [], {}, RSQ, 0.55) == "sub_threshold"


def test_radius_is_exclusive_at_its_edge():
    # greedy_match uses a strict <, so a peak exactly at R is NOT in radius; it
    # falls through to the annulus. Matching that here keeps the taxonomy's
    # population identical to the scorer's.
    p = _near(1.0, 0.9)
    assert mt.classify_miss(GT, [p], [p], {0: -1}, RSQ, 0.30) == "localization"


def test_every_bucket_name_is_declared():
    for claimed, expected in ((3, "merged"), (-1, "localization")):
        p = _near(0.5 if expected == "merged" else 1.5, 0.9)
        assert mt.classify_miss(GT, [p], [p], {0: claimed}, RSQ, 0.30) in mt.BUCKETS


# --------------------------------------------------------------------------- #
# optimal_hits — the matcher upper bound
# --------------------------------------------------------------------------- #
def test_optimal_matching_is_never_worse_than_greedy():
    from rampnet.metrics import greedy_match
    from rampnet.detection_eval import PANO_SCALE_Y
    rng = random.Random(7)
    for _ in range(200):
        gts = [(rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.7)) for _ in range(4)]
        preds = [(rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.7), rng.random())
                 for _ in range(4)]
        preds.sort(key=lambda p: p[2], reverse=True)
        greedy = {gi for gi, _ in greedy_match([(p[0], p[1]) for p in preds], gts,
                                               RSQ, PANO_SCALE_X, PANO_SCALE_Y)
                  if gi >= 0}
        assert len(mt.optimal_hits(preds, gts, RSQ)) >= len(greedy)


def test_optimal_matching_pairs_each_prediction_at_most_once():
    p = _near(0.2, 0.9)
    gts = [GT, (GT[0] + 0.3 * R_NORM, GT[1])]     # both within R of the one peak
    assert len(mt.optimal_hits([p], gts, RSQ)) == 1


def test_optimal_matching_finds_the_assignment_greedy_can_miss():
    # Two peaks, two ramps, but the higher-confidence peak is in range of both.
    # A cardinality-2 assignment exists and optimal_hits must find it.
    a, b = GT, (GT[0] + 1.2 * R_NORM, GT[1])
    p_hi = (GT[0] + 0.6 * R_NORM, GT[1], 0.9)     # in range of a and b
    p_lo = (GT[0] - 0.1 * R_NORM, GT[1], 0.4)     # in range of a only
    assert len(mt.optimal_hits([p_hi, p_lo], [a, b], RSQ)) == 2


def test_no_predictions_hits_nothing():
    assert mt.optimal_hits([], [GT], RSQ) == set()


# --------------------------------------------------------------------------- #
# wrapped_d2 — the panorama seam
# --------------------------------------------------------------------------- #
def test_distance_wraps_across_the_panorama_seam():
    left, right = (2.0, 100.0), (PANO_SCALE_X - 2.0, 100.0)
    assert mt.wrapped_d2(left, right) == 16.0          # 4 px apart, not 1020
    assert mt.wrapped_d2(left, right) < mt._d2(left, right)


def test_wrapping_never_exceeds_half_the_panorama():
    for x in (0.0, 100.0, 512.0, 900.0, PANO_SCALE_X - 1.0):
        d = mt.wrapped_d2((0.0, 0.0), (x, 0.0)) ** 0.5
        assert d <= PANO_SCALE_X / 2 + 1e-9


# --------------------------------------------------------------------------- #
# null_in_radius — the density control
# --------------------------------------------------------------------------- #
def test_null_is_zero_when_the_pano_has_no_peaks():
    supra, sub = mt.null_in_radius(GT, [], RSQ, 0.30, random.Random(1), 50)
    assert supra == 0.0 and sub == 0.0


def test_null_saturates_when_peaks_blanket_the_row():
    # A peak every few pixels along the ramp's elevation: chance placement always
    # lands in radius, so the null must report ~1.0 and the bucket would be worthless.
    peaks = [(i / 200.0, GT[1], 0.10) for i in range(200)]
    supra, sub = mt.null_in_radius(GT, peaks, RSQ, 0.30, random.Random(1), 50)
    assert sub > 0.95 and supra == 0.0


def test_null_separates_supra_from_sub_threshold():
    peaks = [(i / 200.0, GT[1], 0.90) for i in range(200)]
    supra, sub = mt.null_in_radius(GT, peaks, RSQ, 0.30, random.Random(1), 50)
    assert supra > 0.95 and sub == 0.0


def test_null_is_reproducible_for_a_fixed_seed():
    peaks = [(0.31, GT[1], 0.10), (0.72, GT[1], 0.90)]
    a = mt.null_in_radius(GT, peaks, RSQ, 0.30, random.Random(mt.NULL_SEED), 100)
    b = mt.null_in_radius(GT, peaks, RSQ, 0.30, random.Random(mt.NULL_SEED), 100)
    assert a == b


# --------------------------------------------------------------------------- #
# merged_separation — the extractor-vs-target diagnostic
# --------------------------------------------------------------------------- #
def test_separation_is_measured_to_the_ramp_that_won_the_peak():
    partner = (GT[0] + 0.8 * R_NORM, GT[1])
    p = _near(0.4, 0.9)
    cheb, euc = mt.merged_separation(GT, [GT, partner], [p], {0: 1}, RSQ)
    expected = 0.8 * R_NORM * PANO_SCALE_X
    assert abs(euc - expected) < 1e-6 and abs(cheb - expected) < 1e-6


def test_separation_is_none_when_the_peak_belongs_to_nobody():
    p = _near(0.4, 0.9)
    assert mt.merged_separation(GT, [GT], [p], {0: -1}, RSQ) is None


def test_separation_is_none_without_a_peak_in_radius():
    assert mt.merged_separation(GT, [GT], [], {}, RSQ) is None


# --------------------------------------------------------------------------- #
# summarize — the partition must reconcile with #59's figures
# --------------------------------------------------------------------------- #
def _row(hit, bucket=None, field="near"):
    return {"city": "bend", "x": 0.5, "y": 0.6, "dist": 5.0, "px": 100.0,
            "hit": hit, "bucket": bucket, "tier": "gsv", "field": field,
            "sep_cheb": None, "sep_euc": None, "null_supra": None, "null_sub": None}


def test_buckets_partition_every_miss():
    rows = [_row(False, "merged"), _row(False, "sub_threshold"),
            _row(False, "silent"), _row(True)]
    s = mt.summarize(rows)
    assert sum(s["counts"].values()) == s["n_miss"] == 3
    assert abs(sum(s["shares"].values()) - 1.0) < 1e-9


def test_recall_points_sum_to_the_missing_recall():
    rows = [_row(False, "silent"), _row(False, "merged")] + [_row(True)] * 8
    s = mt.summarize(rows)
    assert abs(sum(s["recall_pts"].values()) - (1 - s["recall"])) < 1e-9


def test_hits_are_never_bucketed():
    s = mt.summarize([_row(True)] * 5)
    assert s["n_miss"] == 0 and all(v == 0 for v in s["counts"].values())


def test_empty_population_does_not_divide_by_zero():
    s = mt.summarize([])
    assert s["n_gt"] == 0 and s["n_miss"] == 0


# --------------------------------------------------------------------------- #
# bookkeeping shared with #59 — the two scripts must partition one population
# --------------------------------------------------------------------------- #
def test_split_lists_are_inherited_from_the_decomposition_not_redeclared():
    import miss_decomposition as md
    assert mt.US_SPLITS is md.US_SPLITS
    assert mt.ALL_SPLITS is md.ALL_SPLITS
    assert mt.FAR_BOUNDARY_M == md.FAR_BOUNDARY_M
    assert mt.DEFAULT_THRESHOLD == md.DEFAULT_THRESHOLD


def test_extraction_constants_match_the_deployed_pipeline():
    # The merged bucket's interpretation turns on these two being the real values
    # from stage_two/{evaluate,train}.py; a silent drift would invert the
    # "extractor was free to emit two peaks" reading.
    assert mt.PEAK_MIN_DISTANCE == 10
    assert mt.TARGET_SIGMA == 10
