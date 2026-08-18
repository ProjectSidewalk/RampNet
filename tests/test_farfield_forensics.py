"""Tests for the far-field anomaly's Phase 0 sample forensics (#46).

Two layers, matching the script:

* pure-function tests — the AUC, the per-split floor translation, the band
  classifier (including the above-horizon clamp), and the matched-size rate,
  since each one carries a headline claim;
* integration tests against the **committed** inputs (`analysis_out/op_cache`,
  `analysis_out/silent_witness.json`, `benchmark/miss_taxonomy_46/`,
  `benchmark/<city>/imagery_manifest.json`) pinning the population arithmetic the
  write-up quotes: 83 far-field silent misses = 37 rated + 9 below the floor +
  37 witnessed. If a cache or manifest changes, these numbers must be re-derived,
  not assumed — that is exactly the failure this file exists to catch.

CPU-only, no network, no imagery, no GPU.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import farfield_forensics as ff  # noqa: E402


# --------------------------------------------------------------------------- #
# auc — the survivorship and discrimination statistic
# --------------------------------------------------------------------------- #
def test_auc_is_half_for_identical_samples():
    assert ff.auc([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_auc_is_one_when_a_dominates():
    assert ff.auc([10, 11], [1, 2]) == 1.0


def test_auc_is_zero_when_b_dominates():
    assert ff.auc([1, 2], [10, 11]) == 0.0


def test_auc_counts_ties_half():
    # a = [1], b = [1]: one comparison, tied.
    assert ff.auc([1], [1]) == pytest.approx(0.5)


def test_auc_empty_is_nan():
    assert ff.auc([], [1]) != ff.auc([], [1])  # NaN


# --------------------------------------------------------------------------- #
# effective_floor_model_px — the per-split floor translation
# --------------------------------------------------------------------------- #
def test_floor_on_a_16384_split_is_seven_and_a_half_model_px():
    assert ff.effective_floor_model_px(16384) == pytest.approx(7.5)


def test_floor_at_parity_is_the_floor_itself():
    assert ff.effective_floor_model_px(4096) == pytest.approx(30.0)


def test_floor_scales_inversely_with_stored_width():
    assert ff.effective_floor_model_px(8000) == pytest.approx(30.0 * 4096 / 8000)


# --------------------------------------------------------------------------- #
# band_of — the far bands and the clamp
# --------------------------------------------------------------------------- #
def _row(dist, y=0.7):
    return {"dist": dist, "y": y}


def test_band_edges_are_half_open():
    assert ff.band_of(_row(18.0)) == (18.0, 25.0)
    assert ff.band_of(_row(25.0)) == (25.0, 40.0)
    assert ff.band_of(_row(40.0)) == (40.0, 150.0)


def test_above_horizon_is_the_clamp_band_regardless_of_distance():
    # geom() sends y <= 0.5 to 150 m; the y is the tell, not the distance.
    assert ff.band_of(_row(150.0, y=0.5)) == "clamp"
    assert ff.band_of(_row(150.0, y=0.4)) == "clamp"


def test_the_top_band_is_closed_at_its_upper_edge():
    # geom() reaches 150 m by a SECOND route: min(d, 150.0) on a row that is below
    # the horizon (y = 0.502 is ~406 m before the clamp). A half-open [40, 150)
    # dropped those from every band while y > 0.5 kept them out of `clamp`, so two
    # pooled far-field GT rows -- one of them a silent miss -- were reported nowhere.
    assert ff.band_of(_row(150.0, y=0.502)) == (40.0, 150.0)


def test_near_field_rows_fall_in_no_band():
    assert ff.band_of(_row(10.0)) is None


# --------------------------------------------------------------------------- #
# matched_rate — "how often does the model find OTHER ramps this size?"
# --------------------------------------------------------------------------- #
def _sized(px, hit):
    return {"px": px, "hit": hit}


def test_matched_rate_counts_only_rows_within_tolerance():
    rows = [_sized(20, True), _sized(24, False), _sized(50, True)]
    hits, n = ff.matched_rate(rows, 20.0, tol=0.20)
    assert (hits, n) == (1, 2)  # the 50-px row is out of band


def test_matched_rate_tolerance_is_symmetric_and_inclusive():
    rows = [_sized(16.0, True), _sized(24.0, False)]
    hits, n = ff.matched_rate(rows, 20.0, tol=0.20)
    assert (hits, n) == (1, 2)


def test_matched_rate_with_no_neighbours_is_zero_of_zero():
    assert ff.matched_rate([_sized(100, True)], 20.0) == (0, 0)


# --------------------------------------------------------------------------- #
# percentile_rank / quartiles
# --------------------------------------------------------------------------- #
def test_percentile_rank_midpoint():
    assert ff.percentile_rank([1, 2, 3, 4], 2.5) == pytest.approx(0.5)


def test_percentile_rank_ties_count_half():
    assert ff.percentile_rank([1, 2, 2, 3], 2) == pytest.approx(0.5)


def test_quartiles_are_ordered():
    q1, med, q3 = ff.quartiles(list(range(100)))
    assert q1 < med < q3


# --------------------------------------------------------------------------- #
# Integration — the committed populations the write-up quotes
# --------------------------------------------------------------------------- #
WITNESS = os.path.join(REPO, "analysis_out", "silent_witness.json")
GALLERY = os.path.join(REPO, "benchmark", "miss_taxonomy_46")

needs_committed = pytest.mark.skipif(
    not (os.path.exists(WITNESS) and os.path.exists(ff.verdicts_path(GALLERY))),
    reason="committed witness/gallery files not present")


def test_verdicts_path_is_per_rater():
    # A second rater is the top open follow-up on #46; the file name is the knob,
    # so neither script nor test may hardcode one pass.
    assert ff.verdicts_path("g", "jonf").endswith("silent__jonf.json")
    assert ff.verdicts_path("g", "rater2").endswith("silent__rater2.json")


@needs_committed
def test_every_returned_item_carries_a_verdict():
    # An item queued into the gallery but not yet rated must NOT count as rated:
    # mid-pass with a second rater it would inflate rated_rows, the per-split table
    # and the survivorship AUC, all silently.
    for v in ff.load_rated(GALLERY, field=None).values():
        assert isinstance(v["verdict"], str) and v["verdict"]


@pytest.fixture(scope="module")
def far_populations():
    import miss_taxonomy as mt
    from miss_decomposition import DEFAULT_THRESHOLD, US_SPLITS
    pooled = []
    for city in US_SPLITS:
        loaded = mt.load_rows(city, DEFAULT_THRESHOLD, rng=None)
        if loaded is not None:
            pooled.extend(loaded[0])
    far_silent = [r for r in pooled if r["field"] == "far" and not r["hit"]
                  and r["bucket"] == "silent"]
    from miss_gallery import load_queue
    queue = load_queue(WITNESS)
    unw = [r for r in far_silent if ff.row_key(r) in queue]
    rated = ff.load_rated(GALLERY)
    rated_keys = {(v["city"], v["pano"], round(float(v["x"]), 6),
                   round(float(v["y"]), 6)) for v in rated.values()}
    return far_silent, unw, rated, rated_keys


@needs_committed
def test_the_population_arithmetic_the_writeup_quotes(far_populations):
    far_silent, unw, rated, rated_keys = far_populations
    assert len(far_silent) == 83
    assert len(unw) == 46
    assert len(rated) == 37
    below = [r for r in unw if ff.row_key(r) not in rated_keys]
    assert len(below) == 9


@needs_committed
def test_every_rated_item_is_an_unwitnessed_far_silent_miss(far_populations):
    far_silent, unw, rated, rated_keys = far_populations
    unw_keys = {ff.row_key(r) for r in unw}
    assert rated_keys <= unw_keys


@needs_committed
def test_the_far_verdict_tally_matches_the_committed_pass(far_populations):
    _, _, rated, _ = far_populations
    tally = {}
    for v in rated.values():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    assert tally == {"visible": 34, "context-only": 2, "unclear": 1}


@needs_committed
def test_the_bands_partition_the_far_field(far_populations):
    # The write-up's decisive table is a decomposition of the far field, so its
    # rows must sum to the far field. They did not: rows clamped to exactly 150 m
    # below the horizon fell through every band, taking 2 GT and 1 silent miss out
    # of a table printed directly beneath the population totals they contradict.
    import miss_taxonomy as mt
    from miss_decomposition import DEFAULT_THRESHOLD, US_SPLITS
    pooled = []
    for city in US_SPLITS:
        loaded = mt.load_rows(city, DEFAULT_THRESHOLD, rng=None)
        if loaded is not None:
            pooled.extend(loaded[0])
    far = [r for r in pooled if r["field"] == "far"]
    far_silent, _, _, _ = far_populations
    bands = list(ff.FAR_BANDS) + ["clamp"]
    assert sum(1 for r in far if ff.band_of(r) is None) == 0
    assert sum(len([r for r in far if ff.band_of(r) == b]) for b in bands) == len(far)
    assert sum(len([r for r in far_silent if ff.band_of(r) == b])
               for b in bands) == len(far_silent) == 83


@needs_committed
def test_imagery_manifests_cover_every_rated_pano():
    # The floor table translates the source-px floor through the committed widths;
    # a rated pano missing from its manifest would silently drop from that table.
    rated = ff.load_rated(GALLERY)
    for v in rated.values():
        widths = ff.stored_widths(v["city"])
        assert v["pano"] in widths, (v["city"], v["pano"])
        assert widths[v["pano"]] == v["source_width"]
