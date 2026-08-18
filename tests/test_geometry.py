"""The 360 seam: that cyclic geometry wraps where it should, and NOT where it shouldn't.

Issue #132. Two halves, and the second matters as much as the first — an unconditional
wrap would fix the panorama scorer and silently break the crop-model evaluator, which
matches in a coordinate space whose x axis has two genuinely different ends.
"""
import pytest

from rampnet.detection_eval import (
    GroundTruth, PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for, score_pano)
from rampnet.geometry import (
    crop_left, dist_sq, dist_to_seam, fold, merge_seam_duplicates, wrapped_delta_x)
from rampnet.metrics import greedy_match, match_predictions

RSQ = radius_sq_for()
R = RSQ ** 0.5                      # 22.53 px on the 1024-wide matcher axis
PANO = dict(scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y)


# --- fold: the one primitive ---------------------------------------------------------

def test_fold_takes_the_shorter_way_round():
    assert fold(0.25, 1.0) == pytest.approx(0.25)
    assert fold(0.75, 1.0) == pytest.approx(0.25)


def test_fold_is_symmetric_in_sign():
    assert fold(-0.75, 1.0) == pytest.approx(fold(0.75, 1.0))


def test_fold_never_returns_a_negative_distance_beyond_one_period():
    """The naive ``min(dx, period - dx)`` goes negative once dx > period.

    Reachable from the unit-scale synthetic coordinates in the other test modules, and
    from any caller that has not normalized. A negative distance squares back to a
    plausible small number instead of raising, so this is a silent-corruption guard.
    """
    for dx in (1.5, 2.0, 5.0, 17.3):
        assert fold(dx, 1.0) >= 0.0
    assert fold(5.0, 1.0) == pytest.approx(0.0)      # 5 whole periods


def test_fold_of_half_a_period_is_the_maximum():
    assert fold(0.5, 1.0) == pytest.approx(0.5)
    assert fold(0.49, 1.0) < 0.5 and fold(0.51, 1.0) < 0.5


# --- distances -----------------------------------------------------------------------

def test_wrapped_delta_x_sees_a_seam_pair_as_adjacent():
    # The real geometry from #130: x=0.9897 and x=0.0021 are ~13 px apart, not ~1010.
    assert wrapped_delta_x(0.9897, 0.0021, PANO_SCALE_X) == pytest.approx(12.6, abs=0.5)


def test_dist_sq_wrap_flag_changes_a_seam_pair_from_far_to_near():
    far = dist_sq(0.999, 0.5, 0.001, 0.5, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=False)
    near = dist_sq(0.999, 0.5, 0.001, 0.5, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    assert far > RSQ                      # unwrapped: nowhere near each other
    assert near < RSQ                     # wrapped: the same ramp


def test_dist_sq_leaves_y_alone_because_the_poles_are_not_identified():
    """A panorama is a cylinder in this space, not a torus."""
    a = dist_sq(0.5, 0.01, 0.5, 0.99, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    b = dist_sq(0.5, 0.01, 0.5, 0.99, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=False)
    assert a == pytest.approx(b)
    assert a > RSQ


def test_dist_to_seam_measures_to_the_nearer_edge():
    assert dist_to_seam(0.002, PANO_SCALE_X) == pytest.approx(2.048)
    assert dist_to_seam(0.998, PANO_SCALE_X) == pytest.approx(2.048)
    assert dist_to_seam(0.5, PANO_SCALE_X) == pytest.approx(512.0)


# --- the matcher ---------------------------------------------------------------------

def test_greedy_match_wraps_a_prediction_onto_ground_truth_across_the_seam():
    assignments = greedy_match([(0.001, 0.5)], [(0.999, 0.5)], RSQ,
                               PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    assert assignments == [(0, True)]


def test_greedy_match_without_wrap_misses_the_same_pair():
    """The pre-#132 behaviour, pinned so the flag's effect is unambiguous."""
    assignments = greedy_match([(0.001, 0.5)], [(0.999, 0.5)], RSQ,
                               PANO_SCALE_X, PANO_SCALE_Y, wrap_x=False)
    assert assignments == [(-1, False)]


def test_score_pano_wraps_by_default():
    """score_pano is the panorama scorer; its default coordinate space is cyclic."""
    gt = GroundTruth([(0.999, 0.5)], [], True)
    assert score_pano([(0.001, 0.5)], gt)[:3] == (1, 0, 0)


def test_score_pano_ignore_points_wrap_too():
    """The ignore fallback had its own inline distance and did not wrap (#132 section 4)."""
    gt = GroundTruth([], [(0.999, 0.5)], True)
    tp, fp, ignored = score_pano([(0.001, 0.5)], gt)[:3]
    assert (tp, fp, ignored) == (0, 0, 1)


# --- the half that must NOT wrap -----------------------------------------------------

def test_match_predictions_does_not_wrap_by_default():
    """CROP space regression guard (#132 section 5).

    ``stage_one/crop_model/ps_and_manual_model/evaluate.py`` matches crop-model output
    with ``scale_x=341/4``. A crop is not a panorama: its left and right edges are
    different places, so a detection at one edge must NOT claim ground truth at the
    other. This pins the default, because the tempting "make it consistent" cleanup
    would turn the #132 fix into a fresh bug in the crop model's numbers.
    """
    scale_x, scale_y = 341 / 4, 1024 / 4
    radius_sq = 10.0 ** 2
    details = match_predictions([(0.01, 0.5, 0.9)], [(0.99, 0.5)],
                                radius_sq, scale_x, scale_y)
    assert details == [(0.9, False)]                     # far apart, as it should be
    # ... and the same call in a cyclic space would wrongly call it a hit:
    wrapped = match_predictions([(0.01, 0.5, 0.9)], [(0.99, 0.5)],
                                radius_sq, scale_x, scale_y, wrap_x=True)
    assert wrapped == [(0.9, True)]


# --- seam-duplicate merging ----------------------------------------------------------

def test_merge_seam_duplicates_collapses_a_pair_across_the_seam():
    pts = [(0.9897, 0.5601), (0.0021, 0.5602)]
    assert len(merge_seam_duplicates(pts, RSQ, **PANO)) == 1


def test_merge_seam_duplicates_leaves_genuine_adjacent_ramps_alone():
    """Non-seam pairs inside the radius are common and overwhelmingly real (#130).

    ``manual_gold`` holds 234 of them away from the seam. Merging those would delete
    real ramps in the direction that flatters recall, which is why this is
    seam-crossing only rather than a general dedup.
    """
    pts = [(0.500, 0.54), (0.515, 0.54)]        # ~15 px apart, inside the radius
    assert len(merge_seam_duplicates(pts, RSQ, **PANO)) == 2


def test_merge_seam_duplicates_is_order_independent_under_a_wrapping_matcher():
    """#130's argument: with the wrap, which member survives cannot change the score."""
    a, b = (0.9897, 0.5601), (0.0021, 0.5602)
    pred = [(0.9950, 0.5601)]
    for pts in ([a, b], [b, a]):
        gt = merge_seam_duplicates(pts, RSQ, **PANO)
        assert score_pano(pred, GroundTruth(gt, [], True))[:3] == (1, 0, 0)


# --- crop windows --------------------------------------------------------------------

def test_crop_left_wraps_instead_of_clamping():
    """``gt_gallery.py`` clamped here, which is why the #130 duplicates survived review."""
    assert crop_left(10, 4096, 512) == 4096 - 246       # window starts before column 0
    assert crop_left(2048, 4096, 512) == 1792           # interior window is untouched
