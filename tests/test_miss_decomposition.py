"""Unit tests for the miss decomposition (#59, #38, #48).

Pure logic only — no cache on disk, no network. The load-bearing guarantees: the
far/near partition is exhaustive (every miss is attributed to exactly one fixable
population), the multi-view ceiling stays an *optimistic bound* rather than a
forecast, and the above-horizon counter actually catches the geometry failure it
exists to flag.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import miss_decomposition as md  # noqa: E402


def _row(dist, hit, y=0.7, city="bend"):
    return {"city": city, "x": 0.5, "y": y, "dist": dist, "px": 100.0, "hit": hit}


# --------------------------------------------------------------------------- #
# split_misses — the partition must be exhaustive and hit-only
# --------------------------------------------------------------------------- #
def test_every_miss_lands_in_exactly_one_population():
    rows = [_row(5, False), _row(30, False), _row(5, True), _row(30, True)]
    s = md.split_misses(rows, boundary=18.0)
    assert s["n_miss"] == 2
    assert s["n_far_miss"] + s["n_near_miss"] == s["n_miss"]
    assert abs(s["far_share"] + s["near_share"] - 1.0) < 1e-9


def test_hits_never_count_as_misses_however_far():
    rows = [_row(200, True)] * 10
    s = md.split_misses(rows)
    assert s["n_miss"] == 0 and s["recall"] == 1.0
    assert math.isnan(s["far_share"])          # no misses -> no share to report


def test_recall_points_sum_to_the_missing_recall():
    rows = [_row(5, False), _row(30, False)] + [_row(5, True)] * 8
    s = md.split_misses(rows, boundary=18.0)
    assert abs((s["far_miss_pts"] + s["near_miss_pts"]) - (1 - s["recall"])) < 1e-9


def test_boundary_is_inclusive_on_the_far_side():
    assert md.split_misses([_row(18.0, False)], boundary=18.0)["n_far_miss"] == 1
    assert md.split_misses([_row(17.99, False)], boundary=18.0)["n_near_miss"] == 1


def test_moving_the_boundary_moves_the_split():
    rows = [_row(20, False), _row(5, False)]
    assert md.split_misses(rows, boundary=18.0)["n_far_miss"] == 1
    assert md.split_misses(rows, boundary=50.0)["n_far_miss"] == 0
    assert md.split_misses(rows, boundary=1.0)["n_far_miss"] == 2


def test_empty_input_does_not_divide_by_zero():
    s = md.split_misses([])
    assert s["n_gt"] == 0 and math.isnan(s["recall"])


# --------------------------------------------------------------------------- #
# above_horizon — the geometry-failure tell
# --------------------------------------------------------------------------- #
def test_above_horizon_catches_impossible_ground_ramps():
    # y <= 0.5 is at or above the horizon: impossible for a ramp on the ground,
    # so it signals an unleveled rig or a hill. geom() clamps these to 150 m,
    # which would silently inflate the far-field bucket.
    rows = [_row(5, True, y=0.7), _row(150, False, y=0.45), _row(150, False, y=0.5)]
    assert md.above_horizon(rows) == 2


def test_above_horizon_is_reported_alongside_the_split():
    s = md.split_misses([_row(150, False, y=0.4), _row(5, True)])
    assert s["n_above_horizon"] == 1


# --------------------------------------------------------------------------- #
# multiview_ceiling — must stay an upper bound
# --------------------------------------------------------------------------- #
def test_ceiling_never_falls_below_current_recall():
    rows = [_row(5, True)] * 9 + [_row(30, False)]
    s = md.split_misses(rows, boundary=18.0)
    assert md.multiview_ceiling(rows, 18.0) >= s["recall"]


def test_ceiling_lifts_far_ramps_to_the_near_field_rate():
    # 8/10 near hit (0.8); 10 far ramps all missed. Re-observing them at the near
    # rate gives (8 + 0.8*10) / 20 = 0.8.
    rows = [_row(5, True)] * 8 + [_row(5, False)] * 2 + [_row(30, False)] * 10
    assert abs(md.multiview_ceiling(rows, 18.0) - 0.8) < 1e-9


def test_perfect_near_field_ceiling_is_one():
    rows = [_row(5, True)] * 5 + [_row(30, False)] * 5
    assert abs(md.multiview_ceiling(rows, 18.0) - 1.0) < 1e-9


def test_ceiling_is_nan_without_a_near_field_population():
    # Nothing to extrapolate the re-observation rate from.
    assert math.isnan(md.multiview_ceiling([_row(30, False)], 18.0))


def test_ceiling_equals_recall_when_nothing_is_far():
    rows = [_row(5, True)] * 7 + [_row(5, False)] * 3
    s = md.split_misses(rows, boundary=18.0)
    assert abs(md.multiview_ceiling(rows, 18.0) - s["recall"]) < 1e-9


# --------------------------------------------------------------------------- #
# split/tier bookkeeping
# --------------------------------------------------------------------------- #
def test_every_pooled_split_has_a_declared_imagery_tier():
    # The tier split is how the geometry caveat is audited; a missing entry would
    # silently drop a split out of both tier rows.
    for city in md.US_SPLITS:
        assert md.TIER[city] in ("gsv", "mapillary"), city


def test_held_out_splits_are_not_in_the_pooled_basis():
    assert set(md.HELD_OUT) & set(md.US_SPLITS) == set()
    assert set(md.ALL_SPLITS) == set(md.US_SPLITS) | set(md.HELD_OUT)


def test_a_city_contributes_at_most_one_pooled_split():
    """One city, one pooled row — the rule docs/adding_a_benchmark_city.md states.

    Laurens is the only city run on two imagery rigs, and the two arms sample one
    1.91 km2 town: 59% of `laurens_gsv`'s panos sit within 20 m of a
    `laurens_mapillary` one (median nearest neighbour 17.2 m), so they largely see
    the same physical curb ramps. Pooling both would count those ramps twice and
    break the independence the Wilson intervals assume — and it would do it
    silently, by moving every pooled number a little.

    So the second arm is held out, and this is what enforces it rather than
    anyone remembering. Splits with no `CITY_OF` entry are their own city, which
    keeps the registry to the one line per genuine ambiguity.
    """
    import low_floor_sweep as lfs
    seen = {}
    for split in lfs.US_SPLITS:
        city = lfs.CITY_OF.get(split, split)
        assert city not in seen, (
            f"{split} and {seen[city]} are both pooled and are both the city "
            f"{city!r}. A second imagery arm must go in HELD_OUT, not US_SPLITS.")
        seen[city] = split


def test_every_multi_arm_city_is_declared_in_city_of():
    """A second arm that nobody declared reads as an unrelated city, and the guard
    above then passes vacuously — which is the failure mode that matters, because
    it looks identical to having no second arm at all."""
    import low_floor_sweep as lfs
    for split in lfs.ALL_SPLITS:
        base = split.rsplit("_", 1)[0]
        siblings = [s for s in lfs.ALL_SPLITS if s != split and s.startswith(base + "_")]
        if siblings:
            assert split in lfs.CITY_OF, (
                f"{split} shares the prefix {base!r} with {siblings} but has no "
                f"CITY_OF entry, so the one-pooled-split-per-city guard cannot see "
                f"the pairing.")


def test_registries_agree_with_low_floor_sweep():
    """The two split registries must cover the same splits.

    They are separate modules by history, not by design, and this family's CLI
    defaults come off *this* one (export_model_cache, imagery_manifest, the
    galleries, the taxonomies). A split registered in only one of them is not an
    error anywhere — it is silently skipped, which reads as a result nobody ran.
    sao_paulo was added to low_floor_sweep and missed here (PR #100 review).
    """
    import low_floor_sweep as lfs
    assert set(md.US_SPLITS) == set(lfs.US_SPLITS)
    assert set(md.ALL_SPLITS) == set(lfs.ALL_SPLITS)
    assert set(md.HELD_OUT) == set(lfs.ALL_SPLITS) - set(lfs.US_SPLITS)
