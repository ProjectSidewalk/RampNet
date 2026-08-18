"""Unit tests for the false-positive taxonomy (#46).

Pure logic plus one drift guard — no GPU, no ``.model_cache``, no imagery.

The load-bearing guarantees:

* :func:`arc_union_fraction` is the density control the whole "is this model nearly
  right, or just dense?" reading depends on. It is a closed form, so it is testable
  against hand-computed values rather than merely self-consistent.
* the FP buckets partition every prediction exactly once;
* ``duplicate`` requires the ramp to be **already claimed**, which is what stops it
  from silently absorbing the ``near_gt`` bucket;
* the cache-signature defaults still match ``compare.py``. A drift there does not
  raise — it changes the cache key, every lookup misses, and the script reports a
  model with zero detections. That failure is invisible without this test.
"""
import ast
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import fp_taxonomy as fx  # noqa: E402
from rampnet import roster  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
R = RSQ ** 0.5
R_NORM = R / PANO_SCALE_X


# --------------------------------------------------------------------------- #
# arc_union_fraction — the exact density control
# --------------------------------------------------------------------------- #
def test_no_ground_truth_means_no_chance_of_landing_near_it():
    assert fx.arc_union_fraction(300.0, [], RSQ) == 0.0


def test_a_single_ramp_at_the_same_height_spans_one_diameter():
    # Same y: the reachable azimuths are exactly +/- R around it, i.e. 2R of the
    # 1024-px circumference.
    gy = 0.6
    frac = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.5, gy)], RSQ)
    assert abs(frac - 2 * R / PANO_SCALE_X) < 1e-9


def test_a_ramp_further_away_in_y_spans_a_shorter_arc():
    gy = 0.6
    dy = 0.5 * R                                    # half a radius up, in scaled px
    here = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.5, gy)], RSQ)
    above = fx.arc_union_fraction(gy * PANO_SCALE_Y - dy, [(0.5, gy)], RSQ)
    assert above < here
    expected = 2 * math.sqrt(RSQ - dy ** 2) / PANO_SCALE_X
    assert abs(above - expected) < 1e-9


def test_a_ramp_out_of_vertical_reach_contributes_nothing():
    gy = 0.6
    py = gy * PANO_SCALE_Y - (R + 1.0)
    assert fx.arc_union_fraction(py, [(0.5, gy)], RSQ) == 0.0


def test_two_distant_ramps_add_their_arcs():
    gy = 0.6
    one = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.2, gy)], RSQ)
    two = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.2, gy), (0.8, gy)], RSQ)
    assert abs(two - 2 * one) < 1e-9


def test_overlapping_ramps_are_unioned_not_double_counted():
    # Two ramps half a radius apart: the union is 2.5R wide, not 4R.
    gy = 0.6
    pts = [(0.5, gy), (0.5 + 0.5 * R_NORM, gy)]
    frac = fx.arc_union_fraction(gy * PANO_SCALE_Y, pts, RSQ)
    assert abs(frac - 2.5 * R / PANO_SCALE_X) < 1e-9


def test_an_arc_crossing_the_seam_is_not_lost():
    # A ramp at x=0 reaches azimuths on both sides of the wrap; the total must
    # still be one diameter, not half of one.
    gy = 0.6
    frac = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.0, gy)], RSQ)
    assert abs(frac - 2 * R / PANO_SCALE_X) < 1e-9


def test_seam_arcs_from_both_sides_are_unioned():
    gy = 0.6
    left = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.001, gy)], RSQ)
    right = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.999, gy)], RSQ)
    both = fx.arc_union_fraction(gy * PANO_SCALE_Y, [(0.001, gy), (0.999, gy)], RSQ)
    assert both < left + right           # they overlap across the seam
    assert both > max(left, right)


def test_a_blanket_of_ramps_saturates_at_one():
    gy = 0.6
    pts = [(i / 200.0, gy) for i in range(200)]
    assert fx.arc_union_fraction(gy * PANO_SCALE_Y, pts, RSQ) == 1.0


def test_the_fraction_is_always_a_probability():
    gy = 0.6
    for n in (1, 5, 50, 500):
        pts = [(i / n, gy) for i in range(n)]
        f = fx.arc_union_fraction(gy * PANO_SCALE_Y, pts, RSQ)
        assert 0.0 <= f <= 1.0


# --------------------------------------------------------------------------- #
# classify_fp — the buckets
# --------------------------------------------------------------------------- #
GT = (0.5, 0.6)


def _at(dx_frac, y=GT[1]):
    return (GT[0] + dx_frac * R_NORM, y)


def test_second_hit_on_an_already_claimed_ramp_is_a_duplicate():
    assert fx.classify_fp(_at(0.5), [GT], {0}, RSQ) == "duplicate"


def test_inside_the_radius_of_an_unclaimed_ramp_is_not_a_duplicate():
    # The matcher would have assigned it, so this cannot arise in practice; the
    # bucket must not claim it regardless, or 'duplicate' stops meaning 'redundant'.
    assert fx.classify_fp(_at(0.5), [GT], set(), RSQ) == "near_gt"


def test_between_one_and_two_radii_is_near_gt():
    assert fx.classify_fp(_at(1.5), [GT], {0}, RSQ) == "near_gt"


def test_beyond_two_radii_is_not_near_gt():
    assert fx.classify_fp(_at(2.5), [GT], {0}, RSQ) == "isolated"


def test_below_the_vehicle_line_is_hood():
    assert fx.classify_fp((0.1, 0.80), [GT], {0}, RSQ) == "hood"
    assert fx.classify_fp((0.1, 0.74), [GT], {0}, RSQ) == "isolated"


def test_the_hood_boundary_is_inclusive_and_configurable():
    assert fx.classify_fp((0.1, 0.75), [GT], {0}, RSQ, hood_y=0.75) == "hood"
    assert fx.classify_fp((0.1, 0.75), [GT], {0}, RSQ, hood_y=0.80) == "isolated"


def test_proximity_to_a_ramp_outranks_the_hood_band():
    # A ramp genuinely at the vehicle line: the detection is about the ramp, not
    # the car, so it must not be written off as hood.
    low_gt = (0.5, 0.78)
    p = (low_gt[0] + 0.5 * R_NORM, 0.78)
    assert fx.classify_fp(p, [low_gt], {0}, RSQ) == "duplicate"


def test_nothing_nearby_and_above_the_hood_is_isolated():
    assert fx.classify_fp((0.1, 0.55), [GT], {0}, RSQ) == "isolated"


def test_every_outcome_is_a_declared_bucket():
    cases = [(_at(0.5), {0}), (_at(1.5), {0}), ((0.1, 0.9), set()), ((0.1, 0.5), set())]
    for point, claimed in cases:
        assert fx.classify_fp(point, [GT], claimed, RSQ) in fx.BUCKETS


# --------------------------------------------------------------------------- #
# summarize_fp — the partition
# --------------------------------------------------------------------------- #
def _row(bucket, null_1r=0.0, null_2r=0.0, city="bend"):
    return {"city": city, "pano": "p", "x": 0.5, "y": 0.6, "bucket": bucket,
            "null_1r": null_1r, "null_2r": null_2r}


def test_buckets_partition_every_false_positive():
    rows = [_row("duplicate"), _row("near_gt"), _row("hood"), _row("isolated")]
    s = fx.summarize_fp(rows)
    assert sum(s["counts"].values()) == s["n_fp"] == 4
    assert abs(sum(s["shares"].values()) - 1.0) < 1e-9


def test_null_totals_are_summed_expectations_not_rates():
    rows = [_row("near_gt", null_2r=0.10), _row("isolated", null_2r=0.30)]
    assert abs(fx.summarize_fp(rows)["null_near_gt"] - 0.40) < 1e-9


def test_empty_model_does_not_divide_by_zero():
    s = fx.summarize_fp([])
    assert s["n_fp"] == 0 and all(v == 0 for v in s["counts"].values())


# --------------------------------------------------------------------------- #
# drift guard — the silent failure mode
# --------------------------------------------------------------------------- #
def _compare_parser_defaults():
    """``{dest: default}`` for every statically-resolvable flag default in compare.py.

    Parsed from the source with ``ast`` rather than by importing: compare.py builds
    its parser inside ``main`` and pulls in the detector stack, which needs torch.

    Two default shapes resolve. A plain literal, and ``_D["key"]`` — the per-provider
    defaults, which since #122 come from ``rampnet.roster.PROVIDER_DEFAULTS`` instead
    of being spelled out here. Resolving the subscript rather than skipping it is the
    point: it keeps this drift check covering the provider flags, and it fails if
    compare.py ever stops reading the registry.
    """
    path = os.path.join(REPO, "scripts", "model_comparison", "compare.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        long = next((f for f in flags if f.startswith("--")), None)
        if not long:
            continue
        dest = long[2:].replace("-", "_")
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            if isinstance(kw.value, ast.Constant):
                out[dest] = kw.value.value
            elif (isinstance(kw.value, ast.Subscript)
                    and isinstance(kw.value.value, ast.Name)
                    and kw.value.value.id == "_D"
                    and isinstance(kw.value.slice, ast.Constant)):
                out[dest] = roster.PROVIDER_DEFAULTS[kw.value.slice.value]
    return out


def test_signature_defaults_still_match_compare_py():
    # A drift here does not raise anywhere: it changes the cache key, so every
    # lookup misses and the taxonomy silently reports a model with no detections.
    ours = vars(fx._compare_args("/tmp/cache"))
    theirs = _compare_parser_defaults()
    checked = 0
    for dest, default in theirs.items():
        if dest in ("cache_dir",) or dest not in ours:
            continue
        assert ours[dest] == default, (
            f"--{dest.replace('_', '-')} default drifted: compare.py has "
            f"{default!r}, fp_taxonomy has {ours[dest]!r}")
        checked += 1
    assert checked >= 10, f"only {checked} defaults cross-checked; parser shape changed?"


def test_the_flags_that_feed_the_signature_are_all_present():
    # These are the ones detectors.signature() reads. A missing attribute would
    # raise inside build_detector rather than drift, but listing them here documents
    # the coupling.
    ours = vars(fx._compare_args("/tmp/cache"))
    for k in ("gemini_model", "qwen_model", "qwen_coord_space", "owlv2_model",
              "gdino_model", "molmo_model", "owlv2_query", "gdino_query",
              "gdino_text_threshold", "score_threshold", "molmo_coord_scale",
              "claude_model", "claude_effort", "claude_tool_choice",
              "claude_image_format", "claude_temperature",
              "tiling", "radius"):
        assert k in ours, k


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_the_hood_line_sits_below_every_plausible_ramp():
    # 0.75 was chosen from the data: the 99.5th percentile of pooled GT height is
    # 0.725. Flat-ground geometry puts the line at ~2.5 m, i.e. inside the car.
    from stage1_label_recall import geom
    assert 0.70 < fx.HOOD_Y < 0.80
    assert geom(fx.HOOD_Y)[0] < 3.0


def test_the_annulus_matches_the_miss_taxonomy():
    # The FN and FP sides of one loose box must be attributed on the same radius.
    import miss_taxonomy as mt
    assert fx.ANNULUS_FACTOR == mt.ANNULUS_FACTOR


def test_the_challenger_roster_is_the_documented_one():
    assert len(fx.CHALLENGERS) == 7
    assert any("owlv2" in s for s in fx.CHALLENGERS)
    assert any("gdino" in s for s in fx.CHALLENGERS)
