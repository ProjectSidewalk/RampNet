"""Unit tests for the silent-miss activation forensics (#46, Phase 1).

Pure core only — no torch, no imagery, no GPU. The heavy path (model inference)
is exercised by running the script itself; what these protect is the geometry:
``radius_max`` must read the heatmap through exactly the matcher's coordinate
convention, or the activation numbers describe the wrong locations.
"""
import collections
import json
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import silent_activation as sa  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402

RSQ = radius_sq_for()
R = RSQ ** 0.5  # 22.5 heatmap px


def _heat(value=0.0):
    return [[value] * 1024 for _ in range(512)]


# --------------------------------------------------------------------------- #
# radius_max — the matcher's window, applied to the heatmap
# --------------------------------------------------------------------------- #
def test_reads_a_peak_at_the_site():
    h = _heat()
    h[256][512] = 0.7
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == pytest.approx(0.7)


def test_ignores_a_peak_outside_the_radius():
    h = _heat()
    h[256][512 + int(R) + 2] = 0.9
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == 0.0


def test_sees_a_peak_just_inside_the_radius():
    h = _heat()
    h[256][512 + int(R) - 1] = 0.9
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == pytest.approx(0.9)


def test_columns_wrap_at_the_seam():
    # A site at x~0 must see a peak stored at the right edge of the heatmap.
    h = _heat()
    h[256][1023] = 0.8
    assert sa.radius_max(h, 2 / 1024, 256 / 512) == pytest.approx(0.8)


def test_rows_clamp_at_the_top():
    # A site near the top row must not crash reaching above the panorama.
    h = _heat()
    h[0][512] = 0.6
    assert sa.radius_max(h, 512 / 1024, 0.0) == pytest.approx(0.6)


def test_values_clip_to_one_like_peak_extraction():
    h = _heat()
    h[256][512] = 1.7
    assert sa.radius_max(h, 512 / 1024, 256 / 512) == 1.0


# --------------------------------------------------------------------------- #
# null_percentile — signal against the pano's own noise floor
# --------------------------------------------------------------------------- #
def test_flat_heatmap_reads_as_chance():
    act, pct, med, p95 = sa.null_percentile(_heat(0.003), 0.5, 0.5,
                                            random.Random(0), trials=50)
    assert act == pytest.approx(0.003)
    assert pct == pytest.approx(0.5)  # every draw ties the site
    assert med == p95 == pytest.approx(0.003)


def test_a_lone_bump_at_the_site_beats_its_null():
    h = _heat()
    h[256][512] = 0.04
    act, pct, _, p95 = sa.null_percentile(h, 512 / 1024, 256 / 512,
                                          random.Random(0), trials=100)
    assert act == pytest.approx(0.04)
    assert pct > 0.9
    assert act > p95


def test_a_site_no_better_than_the_horizon_band_fails_the_test():
    # Strong response everywhere along the site's row: the site is nothing special.
    h = _heat()
    for c in range(0, 1024, 8):
        h[256][c] = 0.5
    act, pct, _, p95 = sa.null_percentile(h, 512 / 1024, 256 / 512,
                                          random.Random(0), trials=100)
    assert act == pytest.approx(0.5)
    assert not act > p95


# --------------------------------------------------------------------------- #
# site_profile / nearest_peak — separating a site response from a neighbour's tail
# --------------------------------------------------------------------------- #
def test_site_profile_centred_bump_has_zero_offset():
    h = _heat()
    h[256][512] = 0.4
    act, off, center = sa.site_profile(h, 512 / 1024, 256 / 512)
    assert act == pytest.approx(0.4)
    assert off == pytest.approx(0.0)
    assert center == pytest.approx(0.4)


def test_site_profile_offset_bump_reports_its_distance():
    h = _heat()
    h[256][512 + 15] = 0.4
    act, off, center = sa.site_profile(h, 512 / 1024, 256 / 512)
    assert act == pytest.approx(0.4)
    assert off == pytest.approx(15.0)
    assert center == 0.0  # nothing at the ramp itself


def test_nearest_peak_measures_in_matcher_units_and_wraps():
    # A peak across the seam: x=0.999 vs site x=0.001 is ~2 px away, not ~1022.
    d, score = sa.nearest_peak([(0.999, 0.5, 0.7)], 0.001, 0.5)
    assert d == pytest.approx(0.002 * 1024, abs=0.01)
    assert score == 0.7


def test_nearest_peak_with_no_peaks_is_infinite():
    d, score = sa.nearest_peak([], 0.5, 0.5)
    assert d == float("inf") and score is None


# --------------------------------------------------------------------------- #
# group_of — Phase 0's partition, reused
# --------------------------------------------------------------------------- #
def _row(city="bend", pano="p1", x=0.25, y=0.6):
    return {"city": city, "pano": pano, "x": x, "y": y}


def test_group_partition():
    r = _row()
    key = sa.row_key(r)
    assert sa.group_of(r, set(), {}) == "witnessed"
    assert sa.group_of(r, {key}, {}) == "below_floor"
    assert sa.group_of(r, {key}, {key: {}}) == "rated"


# --------------------------------------------------------------------------- #
# seam_of — where this window and the matcher's disagree
# --------------------------------------------------------------------------- #
def test_seam_flag_is_true_within_a_radius_of_either_edge():
    # radius_max wraps columns; rampnet.metrics.greedy_match, which produced the
    # `silent` label, takes a plain x difference. Inside R of x=0 or x=1 the two
    # therefore read different windows, and that has to be visible in the output
    # rather than inferred later from the coordinates.
    assert sa.seam_of(0.001) and sa.seam_of(0.999)
    assert sa.seam_of(R / 1024 - 1e-6)


def test_seam_flag_is_false_away_from_the_edges():
    assert not sa.seam_of(0.5)
    assert not sa.seam_of(R / 1024 + 1e-6)
    assert not sa.seam_of(1.0 - (R / 1024 + 1e-6))


# --------------------------------------------------------------------------- #
# class_of — the decomposition's only two cutoffs
# --------------------------------------------------------------------------- #
def test_class_cutoffs():
    assert sa.class_of(0.0) == "absent"
    assert sa.class_of(sa.ABSENT_MAX - 1e-9) == "absent"
    assert sa.class_of(sa.ABSENT_MAX) == "faint_local"
    assert sa.class_of(sa.PEAK_FLOOR - 1e-9) == "faint_local"
    assert sa.class_of(sa.PEAK_FLOOR) == "tail"
    assert sa.class_of(1.0) == "tail"


def test_the_class_floor_is_the_extractors_own_floor():
    # `tail` means "at or above the score floor the caches were extracted at", which
    # is what licenses reading it as an outside mode: a silent miss has no floor peak
    # inside the radius by definition. Drifting these apart would break the reading
    # without breaking anything visible.
    assert sa.PEAK_FLOOR == 0.05


# --------------------------------------------------------------------------- #
# build_payload — the result file records what it is a result OF
# --------------------------------------------------------------------------- #
def test_payload_records_the_run_scope():
    pay = sa.build_payload([{"act": 0.1}], 0.30, ["bend", "clovis"], 7, 2)
    assert pay["cities"] == ["bend", "clovis"] and pay["panos"] == 7
    assert pay["n"] == 1 and pay["skipped_no_imagery"] == 2
    assert pay["null_seed"] == sa.NULL_SEED and pay["tta"] is False


def test_json_out_refuses_a_truncated_run():
    # analysis_out/silent_activation.json is a committed artifact and every number
    # in 0c derives from it, so a smoke-test run must not be able to overwrite it
    # with something that looks complete.
    with pytest.raises(SystemExit):
        sa.main(["--limit", "3", "--json-out", "x.json"])


def test_json_out_refuses_a_city_subset_without_allow_partial():
    with pytest.raises(SystemExit):
        sa.main(["--cities", "bend", "--json-out", "x.json"])


# --------------------------------------------------------------------------- #
# null_azimuths — the self-exclusion, which is the whole point of the null
#
# A draw whose window overlaps the site's own would read the site's own bump back
# as "chance", flattening exactly the sparse-heatmap case this analysis exists to
# detect. Until now nothing tested it: the p95 is so high that the pooled numbers
# look the same either way, so the bug would have been silent.
# --------------------------------------------------------------------------- #
def _wrapped(a, b):
    d = abs(a - b)
    return min(d, 1.0 - d)


def test_null_draws_never_land_inside_the_excluded_zone():
    xs = sa.null_azimuths(0.5, random.Random(1), 300, 0.044)
    assert len(xs) == 300
    assert all(_wrapped(nx, 0.5) >= 0.044 for nx in xs)


def test_null_exclusion_wraps_around_the_seam():
    # A site at x=0.01 must also reject draws at x=0.99, which are 0.02 away.
    xs = sa.null_azimuths(0.01, random.Random(2), 300, 0.044)
    assert all(_wrapped(nx, 0.01) >= 0.044 for nx in xs)
    assert not any(nx > 0.975 for nx in xs)


def test_a_lone_bump_at_the_site_is_never_read_back_as_chance():
    # The decisive behavioural test: with the site the ONLY signal in the pano, a
    # correct null is flat at zero. Without self-exclusion some draws would pick
    # the site's own bump up and the null would no longer be a null.
    h = _heat()
    h[256][512] = 0.9
    act, pct, med, p95 = sa.null_percentile(h, 512 / 1024, 256 / 512,
                                            random.Random(3), trials=200)
    assert act == pytest.approx(0.9)
    assert med == 0.0 and p95 == 0.0
    assert pct == 1.0


def test_the_exclusion_zone_is_two_match_radii():
    assert sa.NULL_EXCLUDE_RADII == 2.0


def test_null_percentile_reports_the_median_and_p95_of_its_draws():
    # A ramp of distinct values along the row: the null's med/p95 must be order
    # statistics of the draws, not of the whole heatmap.
    h = _heat()
    for c in range(1024):
        h[256][c] = c / 1024.0
    act, pct, med, p95 = sa.null_percentile(h, 0.5, 256 / 512,
                                            random.Random(4), trials=200)
    assert 0.0 <= med <= p95 <= 1.0
    assert 0.0 <= pct <= 1.0


# --------------------------------------------------------------------------- #
# site_profile / nearest_peak — the remaining geometry
# --------------------------------------------------------------------------- #
def test_rows_clamp_at_the_bottom():
    h = _heat()
    h[511][512] = 0.5
    assert sa.radius_max(h, 512 / 1024, 1.0) == pytest.approx(0.5)


def test_negative_heatmap_values_read_as_no_response():
    # The model output is not bounded below; peak extraction clips at 0, so a
    # negative trough must not become an `act` of -0.3 and land in `absent` for
    # the wrong reason.
    h = _heat(-0.3)
    act, _, center = sa.site_profile(h, 0.5, 0.5)
    assert act == 0.0 and center == 0.0


def test_nearest_peak_picks_the_closest_of_several_and_returns_its_score():
    peaks = [(0.50, 0.50, 0.9), (0.60, 0.50, 0.8), (0.505, 0.50, 0.4)]
    d, score = sa.nearest_peak(peaks, 0.5, 0.5)
    assert d == pytest.approx(0.0) and score == 0.9
    d, score = sa.nearest_peak(peaks, 0.52, 0.50)
    assert score == 0.4 and d == pytest.approx(0.015 * 1024, abs=0.01)


def test_nearest_peak_measures_rows_in_the_scaled_space_not_normalized():
    # y is scaled by 512 and x by 1024, so equal normalized offsets are NOT equal
    # distances. Getting this backwards would silently halve or double every
    # "nearest floor peak" figure in the write-up.
    dx, _ = sa.nearest_peak([(0.51, 0.50, 0.5)], 0.50, 0.50)
    dy, _ = sa.nearest_peak([(0.50, 0.51, 0.5)], 0.50, 0.50)
    assert dx == pytest.approx(0.01 * 1024, abs=0.01)
    assert dy == pytest.approx(0.01 * 512, abs=0.01)


# --------------------------------------------------------------------------- #
# The committed result JSON, and the numbers §0c quotes out of it
#
# Phase 1 needs a GPU and the native-res panoramas, so unlike Phase 0 it cannot be
# re-run here. What CAN be checked without either is that the artifact is
# self-consistent with the pure functions that ship beside it, and that every
# figure the write-up quotes is still the figure in the file. That is what makes a
# derived-field regeneration provable rather than asserted.
# --------------------------------------------------------------------------- #
RESULT_JSON = os.path.join(REPO, "analysis_out", "silent_activation.json")

needs_result = pytest.mark.skipif(not os.path.exists(RESULT_JSON),
                                  reason="committed result JSON not present")


@pytest.fixture(scope="module")
def result():
    with open(RESULT_JSON, encoding="utf-8") as fh:
        return json.load(fh)


@needs_result
def test_the_header_describes_the_run_it_came_from(result):
    assert result["threshold"] == 0.30
    assert result["null_trials"] == sa.NULL_TRIALS == 200
    assert result["null_seed"] == sa.NULL_SEED
    assert result["model"] == "projectsidewalk/rampnet-model" and result["tta"] is False
    assert result["skipped_no_imagery"] == 0, "a skipped pano means missing imagery"
    assert result["n"] == len(result["results"]) == 128


@needs_result
def test_the_recorded_scope_is_the_pooled_population(result):
    # A subset run is not what 0c quotes; the payload has to say which it is.
    # The frozen published population, not the live registry (see the module comment
    # in scripts/analysis/silent_activation.py).
    from silent_activation import US_SPLITS as PUBLISHED_SPLITS
    assert sorted(result["cities"]) == sorted(PUBLISHED_SPLITS)
    assert result["panos"] == len({(r["city"], r["pano"]) for r in result["results"]})
    assert result["panos"] == 108


@needs_result
def test_every_derived_field_regenerates_from_the_measured_ones(result):
    # `seam` was added to the artifact without re-running the model. This is what
    # licenses that: it is a pure function of `x`, and the shipped function still
    # reproduces every stored value.
    for r in result["results"]:
        assert r["seam"] == sa.seam_of(r["x"])
        assert r["above_own_null_p95"] == (r["act"] > r["null_p95"])


@needs_result
def test_the_class_decomposition_0c_quotes(result):
    counts = collections.Counter(sa.class_of(r["act"]) for r in result["results"])
    assert counts == {"tail": 79, "faint_local": 39, "absent": 10}
    # 8% genuinely flat, 92% responding: the headline of the section.
    assert sum(1 for r in result["results"] if r["act"] >= sa.ABSENT_MAX) == 118


@needs_result
def test_the_null_percentiles_0c_quotes(result):
    from farfield_forensics import quartiles
    by = collections.defaultdict(list)
    for r in result["results"]:
        by[sa.class_of(r["act"])].append(r["null_pct"])
    med = {k: quartiles(v)[1] for k, v in by.items()}
    # `absent` at chance is the check on the whole decomposition: a flat heatmap
    # should read 0.5, and it does. The other two sit well above it.
    assert med["absent"] == pytest.approx(0.495, abs=5e-4)
    assert med["faint_local"] == pytest.approx(0.780, abs=5e-4)
    assert med["tail"] == pytest.approx(0.915, abs=5e-4)
    assert med["absent"] < med["faint_local"] < med["tail"]


@needs_result
def test_the_p95_count_is_low_for_the_reason_0c_gives(result):
    # 31/128 looks like it undercuts `faint_local` until you see that the p95 is
    # the band maximum, not a noise floor. Pin both so the explanation stays true.
    from farfield_forensics import quartiles
    assert sum(1 for r in result["results"] if r["above_own_null_p95"]) == 31
    assert quartiles([r["null_p95"] for r in result["results"]])[1] == \
        pytest.approx(0.595, abs=5e-3)
    assert quartiles([r["null_med"] for r in result["results"]])[1] == \
        pytest.approx(0.003, abs=5e-4)


@needs_result
def test_the_tail_evidence_0c_quotes(result):
    tail = [r for r in result["results"] if sa.class_of(r["act"]) == "tail"]
    r_px = RSQ ** 0.5
    assert sum(1 for r in tail if r["argmax_off_px"] > 0.75 * r_px) == 75
    assert sum(1 for r in tail if r["nearest_peak_px"]
               and r["nearest_peak_px"] < 2 * r_px) == 70


@needs_result
def test_the_faint_local_evidence_0c_quotes(result):
    faint = [r for r in result["results"] if sa.class_of(r["act"]) == "faint_local"]
    assert sum(1 for r in faint if r["act_at_site"] >= sa.ABSENT_MAX) == 30


@needs_result
def test_the_two_populations_0c_reads_the_result_through(result):
    rows = result["results"]
    far_visible = [r for r in rows if r["field"] == "far" and r["verdict"] == "visible"]
    assert len(far_visible) == 34
    assert collections.Counter(sa.class_of(r["act"]) for r in far_visible) == {
        "tail": 19, "faint_local": 12, "absent": 3}
    near = [r for r in rows if r["field"] == "near"]
    assert len(near) == 45
    # The 0.013 sourcing estimate rests on these: only 6 are heatmap-absent.
    assert sum(1 for r in near if sa.class_of(r["act"]) == "absent") == 6


@needs_result
def test_no_seam_row_would_change_bucket_under_a_wrapping_matcher(result):
    # The claim that the wrap/no-wrap divergence moves no number. A floor peak
    # inside the radius would mean a wrapping matcher called the miss `merged`,
    # not `silent` — so this is the count that would falsify it.
    seam = [r for r in result["results"] if r["seam"]]
    assert len(seam) == 9
    assert [r for r in seam if r["nearest_peak_px"]
            and r["nearest_peak_px"] < RSQ ** 0.5] == []


@needs_result
def test_the_strata_partition_the_bucket(result):
    groups = collections.Counter(r["group"] for r in result["results"])
    assert groups == {"witnessed": 69, "rated": 50, "below_floor": 9}
    # Phase 0's far-field partition, seen from Phase 1's side.
    far = [r for r in result["results"] if r["field"] == "far"]
    assert collections.Counter(r["group"] for r in far) == {
        "witnessed": 37, "rated": 37, "below_floor": 9}
