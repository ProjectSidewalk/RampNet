"""Unit tests for the silent-miss witness analysis (#46, #35).

Pure logic — no ``.model_cache``, no network, no GPU.

The claim this file protects is "another model saw this ramp, so RampNet's failure is
its own". That claim is only worth anything if the chance baseline is right: OWLv2
emits 55-88 boxes per panorama and witnesses almost everything by accident. So the
tests concentrate on :func:`hit_chance` behaving like a probability, rising with
density, and vanishing when a prediction cannot reach the ramp at all.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import silent_witness as sw  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
R = RSQ ** 0.5
GT = (0.5, 0.6)


def _at(dx_frac=0.0, dy_px=0.0):
    """A prediction offset from GT by a fraction of R in x and dy_px in y."""
    return (GT[0] + dx_frac * R / PANO_SCALE_X,
            GT[1] + dy_px / PANO_SCALE_Y)


# --------------------------------------------------------------------------- #
# witnessed — did a prediction actually land on the ramp?
# --------------------------------------------------------------------------- #
def test_a_prediction_inside_the_radius_witnesses():
    assert sw.witnessed(GT, [_at(0.5)], RSQ)


def test_a_prediction_outside_the_radius_does_not():
    assert not sw.witnessed(GT, [_at(1.5)], RSQ)


def test_no_predictions_witness_nothing():
    assert not sw.witnessed(GT, [], RSQ)


def test_one_hit_among_many_misses_still_witnesses():
    preds = [_at(3.0), _at(5.0), _at(0.2), _at(9.0)]
    assert sw.witnessed(GT, preds, RSQ)


def test_the_radius_is_exclusive_at_its_edge():
    # Matches greedy_match's strict <, so 'witnessed' and 'hit' mean the same thing.
    assert not sw.witnessed(GT, [_at(1.0)], RSQ)


# --------------------------------------------------------------------------- #
# hit_chance — the density control
# --------------------------------------------------------------------------- #
def test_no_predictions_means_no_chance():
    assert sw.hit_chance(GT, [], RSQ) == 0.0


def test_one_prediction_at_the_same_height_spans_one_diameter():
    # Reachable azimuths are +/- R of 1024 px around.
    c = sw.hit_chance(GT, [_at(7.0)], RSQ)
    assert abs(c - 2 * R / PANO_SCALE_X) < 1e-9


def test_a_prediction_out_of_vertical_reach_contributes_nothing():
    assert sw.hit_chance(GT, [_at(0.0, dy_px=R + 1.0)], RSQ) == 0.0


def test_chance_rises_with_density():
    one = sw.hit_chance(GT, [_at(7.0)], RSQ)
    many = sw.hit_chance(GT, [_at(7.0)] * 20, RSQ)
    assert many > one
    # ...and combines as independent trials, not as a sum.
    assert abs(many - (1 - (1 - one) ** 20)) < 1e-9
    assert many < 20 * one


def test_chance_is_always_a_probability():
    for k in (1, 10, 100, 1000):
        c = sw.hit_chance(GT, [_at(7.0)] * k, RSQ)
        assert 0.0 <= c <= 1.0


def test_a_dense_detector_approaches_certainty():
    # This is the whole reason the correction exists: at OWLv2's density, 'witnessed'
    # is nearly free, so the raw count is not evidence.
    assert sw.hit_chance(GT, [_at(7.0)] * 500, RSQ) > 0.95


def test_chance_ignores_where_the_prediction_actually_is_in_x():
    # Azimuth is randomized, so only the height matters. A prediction sitting exactly
    # on the ramp and one on the far side of the pano have identical chance.
    assert abs(sw.hit_chance(GT, [_at(0.0)], RSQ)
               - sw.hit_chance(GT, [_at(20.0)], RSQ)) < 1e-12


# --------------------------------------------------------------------------- #
# summarize — per-model rows and the union
# --------------------------------------------------------------------------- #
def _rec(**by_model):
    return {"by_model": by_model}


def test_per_model_counts_and_expectations():
    recs = [_rec(a=(True, 0.1), b=(False, 0.2)),
            _rec(a=(False, 0.1), b=(False, 0.2))]
    s = sw.summarize(recs, ["a", "b"])
    assert s["a"]["witnessed"] == 1 and abs(s["a"]["expected"] - 0.2) < 1e-9
    assert s["b"]["witnessed"] == 0 and abs(s["b"]["expected"] - 0.4) < 1e-9


def test_the_union_needs_only_one_model_to_have_seen_it():
    recs = [_rec(a=(True, 0.1), b=(False, 0.1)), _rec(a=(False, 0.1), b=(True, 0.1))]
    assert sw.summarize(recs, ["a", "b"])["__union__"]["witnessed"] == 2


def test_the_union_null_is_not_the_sum_of_the_model_nulls():
    # Two models each with a 0.5 chance do not give a union null of 1.0.
    recs = [_rec(a=(False, 0.5), b=(False, 0.5))]
    u = sw.summarize(recs, ["a", "b"])["__union__"]
    assert abs(u["expected"] - 0.75) < 1e-9


def test_excess_is_observed_minus_chance():
    recs = [_rec(a=(True, 0.25))] * 4
    s = sw.summarize(recs, ["a"])
    assert s["a"]["witnessed"] == 4 and abs(s["a"]["excess"] - 3.0) < 1e-9


def test_a_model_missing_from_a_record_is_not_counted_against_it():
    # A pano with no cache entry for a model must not read as 'that model missed'.
    recs = [_rec(a=(True, 0.1)), _rec(b=(True, 0.1))]
    s = sw.summarize(recs, ["a"])
    assert s["a"]["witnessed"] == 1 and abs(s["a"]["expected"] - 0.1) < 1e-9


def test_union_over_an_empty_model_list_is_empty():
    u = sw.summarize([_rec(a=(True, 0.1))], [])["__union__"]
    assert u["witnessed"] == 0 and u["expected"] == 0.0


def test_summarize_reports_the_population_size():
    assert sw.summarize([_rec(a=(True, 0.1))] * 7, ["a"])["__union__"]["n"] == 7


# --------------------------------------------------------------------------- #
# bookkeeping — the sparse/dense split is load-bearing
# --------------------------------------------------------------------------- #
def test_sparse_and_dense_partition_the_roster():
    from fp_taxonomy import CHALLENGERS
    assert set(sw.SPARSE) | set(sw.DENSE) == set(CHALLENGERS)
    assert set(sw.SPARSE) & set(sw.DENSE) == set()


def test_the_open_vocabulary_detectors_are_the_dense_ones():
    # They are the models docs/model_comparison.md measures at 55-88 boxes/pano, and
    # the headline deliberately excludes them: at that density a witness is free.
    assert set(sw.DENSE) == {"owlv2", "gdino"}
