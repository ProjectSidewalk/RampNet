"""Unit tests for the aerial review sheet's tile math and sampling (issues #96, #59).

No network and no PIL work — only the pure half. The load-bearing guarantees: the
Web Mercator math puts the crosshair on the coordinate being judged (an offset
here would corrupt every verdict in a way no reviewer could detect), the blank
placeholder that services return past their deepest level is recognised rather
than presented as evidence, and the default sample is record-weighted so the
resulting distribution estimates label accuracy rather than area coverage.
"""
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_sheet as irs  # noqa: E402


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def test_pixel_origin_is_the_top_left_of_the_world():
    x, y = irs.lonlat_to_pixel(-180.0, 85.05112878, 0)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-3)


def test_null_island_is_the_centre_of_the_world():
    x, y = irs.lonlat_to_pixel(0.0, 0.0, 0)
    assert math.isclose(x, 128.0) and math.isclose(y, 128.0)


def test_resolution_matches_the_published_web_mercator_table():
    # z=0 at the equator is 156543.03 m/px, and each level halves it.
    assert math.isclose(irs.metres_per_pixel(0.0, 0), 156543.034, rel_tol=1e-6)
    assert math.isclose(irs.metres_per_pixel(0.0, 19), 156543.034 / 2 ** 19, rel_tol=1e-6)


def test_resolution_shrinks_with_latitude():
    """Denver's 0.23 m/px at z19 is the equatorial 0.2986 times cos(39.74)."""
    got = irs.metres_per_pixel(39.74, 19)
    assert math.isclose(got, 0.2986 * math.cos(math.radians(39.74)), rel_tol=1e-3)


def test_tile_range_centres_the_crop_on_the_coordinate():
    lon, lat, z = -104.9911615, 39.7461177, 19
    span_px = 174
    x0, y0, x1, y1, ox, oy = irs.tile_range(lon, lat, z, span_px)
    px, py = irs.lonlat_to_pixel(lon, lat, z)
    # the crosshair sits at the crop's centre, to within a pixel
    assert math.isclose(ox + span_px / 2.0, px, abs_tol=1.0)
    assert math.isclose(oy + span_px / 2.0, py, abs_tol=1.0)
    # and the tile block actually covers the crop
    assert x0 * irs.TILE_PX <= ox and (x1 + 1) * irs.TILE_PX >= ox + span_px
    assert y0 * irs.TILE_PX <= oy and (y1 + 1) * irs.TILE_PX >= oy + span_px


def test_tile_range_spans_multiple_tiles_when_the_crop_straddles_a_seam():
    # z=1 is a 2x2 world; (0, 0) sits exactly on the seam, so any crop straddles.
    x0, y0, x1, y1, _, _ = irs.tile_range(0.0, 0.0, 1, 100)
    assert (x1 - x0 + 1) * (y1 - y0 + 1) == 4


# --------------------------------------------------------------------------- #
# blank-tile detection
# --------------------------------------------------------------------------- #
def test_placeholder_tile_is_recognised():
    """'Map data not yet available' is flat mid-grey."""
    assert irs.looks_blank((200.0, 1.5)) is True


def test_real_imagery_is_not_discarded():
    assert irs.looks_blank((110.0, 42.0)) is False
    assert irs.looks_blank((200.0, 30.0)) is False


def test_a_dark_flat_chip_is_not_called_a_placeholder():
    """Deep building shadow is flat but dark — that is unreadable-for-the-reviewer,
    not a missing tile, and the two have different remedies."""
    assert irs.looks_blank((20.0, 2.0)) is False


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_uniform_sample_is_deterministic_and_sized():
    a = irs.uniform_sample(1000, 60, seed=20260731)
    b = irs.uniform_sample(1000, 60, seed=20260731)
    assert a == b and len(a) == 60 and len(set(a)) == 60


def test_uniform_sample_changes_with_the_seed():
    assert irs.uniform_sample(1000, 60, 1) != irs.uniform_sample(1000, 60, 2)


def test_uniform_sample_cannot_over_draw():
    assert len(irs.uniform_sample(10, 60, seed=1)) == 10


def test_uniform_sample_follows_record_density():
    """The point of the default: a city with 90% of its records downtown should
    yield a sample that is ~90% downtown, because that is where the labels are."""
    n = 1000
    picked = irs.uniform_sample(n, 200, seed=7)
    dense = sum(1 for i in picked if i < 900)
    assert 0.85 < dense / 200.0 < 0.95


def test_stratified_sample_spreads_across_cells():
    """The diagnostic option: 999 clustered points and 1 outlier, and the outlier
    still gets picked — which is precisely why it is not the default."""
    pts = [(0.0 + i * 1e-6, 0.0) for i in range(999)] + [(1.0, 1.0)]
    got = irs.stratified_sample(pts, 2, seed=3, grid=8)
    assert 999 in got


def test_stratified_sample_is_deterministic():
    pts = [(i * 0.01, i * 0.01) for i in range(200)]
    assert irs.stratified_sample(pts, 20, 5) == irs.stratified_sample(pts, 20, 5)


def test_stratified_sample_of_nothing_is_empty():
    assert irs.stratified_sample([], 10, 1) == []
    assert irs.stratified_sample([(0.0, 0.0)], 0, 1) == []


# --------------------------------------------------------------------------- #
# sheet assembly
# --------------------------------------------------------------------------- #
def _meta():
    return {"city": "denver-co", "inventory": "denver-co-2026-07-31.jsonl.gz",
            "sampling": "uniform", "seed": 20260731, "tile_source": "denver-2016",
            "zoom": 21, "mpp": 0.0573, "span_px": 698,
            "attribution": "City and County of Denver", "note": "leaf-off 3-inch"}


def _chips(n=2):
    return [{"uri": "data:image/jpeg;base64,AAAA", "id": str(1000 + i),
             "lon": -105.0, "lat": 39.7, "tiles": ["21/1/2"]} for i in range(n)]


def test_sheet_leaves_no_unsubstituted_placeholders():
    html = irs.build_sheet(_meta(), _chips(), {"city": "denver-co"})
    assert "__" not in html.replace("__proto__", "")
    assert "{{" not in html


def test_sheet_embeds_parseable_meta_and_chips():
    """The page is driven entirely by these two blobs; a malformed one is a blank
    screen with no error the reviewer can act on."""
    html = irs.build_sheet(_meta(), _chips(3), {"city": "denver-co", "seed": 1})
    meta = json.loads(re.search(r"const META = (\{.*?\});\n", html, re.S).group(1))
    chips = json.loads(re.search(r"const CHIPS = (\[.*?\]);\n", html, re.S).group(1))
    assert meta["span_px"] == 698 and meta["rings"] == list(irs.RING_RADII_M)
    assert meta["manifest"]["city"] == "denver-co"
    assert len(chips) == 3 and chips[0]["id"] == "1000"


def test_sheet_scales_the_overlay_to_the_chip_rather_than_hardcoding_pixels():
    """The overlay must use the chip's own viewBox, or rings drawn for a 174 px
    chip land in the wrong place on a 698 px one."""
    html = irs.build_sheet(_meta(), _chips(4), {})
    assert '<svg viewBox="0 0 ${S} ${S}"' in html
    assert "const S = META.span_px" in html


def test_sheet_carries_the_imagery_provenance_into_the_page():
    """A verdict must never be readable without knowing what it was made against."""
    html = irs.build_sheet(_meta(), _chips(), {})
    assert "City and County of Denver" in html
    assert "leaf-off 3-inch" in html
    assert "denver-2016" in html


def test_sheet_does_not_bake_annotations_into_the_image():
    """Regression guard for the thing that made the first sheet unusable."""
    html = irs.build_sheet(_meta(), _chips(), {})
    assert "overlayInner" in html       # rings are markup, not pixels
    assert "nooverlay" in html          # and they can be switched off


# --------------------------------------------------------------------------- #
# the rubric
#
# These read like pedantry and are not. Every clause the rubric carries was
# written after a real chip got called two ways in one sitting, and the failure
# mode is silent: a sheet that quietly stops stating its rules still produces
# confident-looking numbers that mean something different from the last city's.
# --------------------------------------------------------------------------- #
def test_rubric_travels_into_the_exported_manifest():
    """An offset is uninterpretable without the rule saying what it is offset from.

    The sheet reads the rubric out of ``META.manifest``, which is the same object
    written to ``verdicts.json`` — so this also pins the two to one source.
    """
    html = irs.build_sheet(_meta(), _chips(), {"city": "denver-co", "rubric": irs.RUBRIC})
    meta = json.loads(re.search(r"const META = (\{.*?\});\n", html, re.S).group(1))
    assert meta["manifest"]["rubric"]["click_target"] == irs.RUBRIC["click_target"]
    assert "META.manifest.rubric" in html


def test_rubric_covers_every_field_a_reviewer_can_set():
    for key in ("click_target", "always_click", "ramps_visible", "on_corner",
                "no_ramp", "unjudgeable", "resolution_floor", "published_nearby"):
        assert key in irs.RUBRIC, key
        assert len(irs.RUBRIC[key]) > 80, key


def test_click_target_rubric_warns_off_the_detectable_warning_pad():
    """The pad is the most visible thing in the frame and sits 0.6-0.9 m down-slope
    of the ramp centre, so clicking it biases every record in one direction."""
    text = irs.RUBRIC["click_target"].lower()
    assert "not the detectable-warning pad" in text
    assert "parallel" in text and "landing" in text


def test_ramps_visible_rubric_states_the_containment_rule_not_the_crossing_rule():
    """'One ramp per crossing' is wrong for a median island: two cut-through ends
    serve a single crossing. Containment is the rule that survives every case."""
    text = irs.RUBRIC["ramps_visible"].lower()
    assert "without crossing a roadway" in text
    assert "per-corner, not per-chip" in text


# --------------------------------------------------------------------------- #
# terminal verdict states
# --------------------------------------------------------------------------- #
def test_a_readable_corner_with_no_ramp_can_be_completed():
    """Before ``no_ramp`` this chip was uncompletable: nothing to click, so the
    offset stayed null, so ``done()`` was never true and 'next unreviewed' walked
    straight back to it. It is also the phantom rate, which is a reported number.
    """
    html = irs.build_sheet(_meta(), _chips(), {})
    assert "v.unreadable || v.no_ramp || v.offset_m != null" in html
    assert "no_ramp: !!v.no_ramp" in html


def test_no_ramp_and_unjudgeable_are_mutually_exclusive():
    """'I can see, and it is not there' and 'I cannot see' are different claims;
    a chip asserting both would corrupt the phantom rate and the unreadable rate
    at once."""
    html = irs.build_sheet(_meta(), _chips(), {})
    assert "if (x) v.no_ramp = false;" in html
    assert "v.unreadable = false; v.offset_m = null;" in html


# --------------------------------------------------------------------------- #
# published-neighbour counts
# --------------------------------------------------------------------------- #
def test_neighbour_count_includes_the_record_itself():
    """So the number is directly comparable to a reviewer's per-corner count
    rather than off by one against it."""
    pts = [(-105.0, 39.7)]
    assert irs.count_neighbours(pts, [(-105.0, 39.7)], (6.0,)) == [[1]]


def test_neighbour_count_separates_radii():
    """A metre east is 1/(111320*cos(39.7)) degrees; place ramps at ~4 m and ~8 m."""
    deg = 1.0 / (111320.0 * math.cos(math.radians(39.7)))
    pts = [(-105.0, 39.7), (-105.0 + 4 * deg, 39.7), (-105.0 + 8 * deg, 39.7)]
    assert irs.count_neighbours(pts, [(-105.0, 39.7)], (6.0, 10.0)) == [[2, 3]]


def test_neighbour_count_finds_points_across_grid_cell_seams():
    """The bucketing is an optimisation; a ramp must not vanish because it fell in
    the next cell. Sweeps a full circle of bearings at just under the radius."""
    lat, lon = 39.7, -105.0
    mlon = 111320.0 * math.cos(math.radians(lat))
    for bearing in range(0, 360, 15):
        r = 5.5
        dx = r * math.sin(math.radians(bearing)) / mlon
        dy = r * math.cos(math.radians(bearing)) / 111132.0
        got = irs.count_neighbours([(lon + dx, lat + dy)], [(lon, lat)], (6.0,))
        assert got == [[1]], bearing


def test_neighbour_count_excludes_beyond_the_largest_radius():
    deg = 1.0 / (111320.0 * math.cos(math.radians(39.7)))
    pts = [(-105.0 + 40 * deg, 39.7)]
    assert irs.count_neighbours(pts, [(-105.0, 39.7)], (6.0, 10.0)) == [[0, 0]]


def test_build_id_is_in_the_controls_row_not_the_fine_print():
    """It answers "is this page stale?", which is useless if it is buried. The
    first version sat at the end of five wrapped lines of grey text and was not
    found. It must sit in the header's control row, alongside the buttons."""
    html = irs.build_sheet(_meta(), _chips(), {})
    header = re.search(r"<header>(.*?)</header>", html, re.S).group(1)
    controls = header.split('<div class="sub">')[0]
    assert 'class="build"' in controls
    assert irs.sheet_build_id() in controls


def test_build_id_changes_when_the_rubric_or_page_logic_changes(monkeypatch):
    """A stamp that does not move when the instrument moves is worse than none —
    it would certify a stale page as current."""
    before = irs.sheet_build_id()
    monkeypatch.setitem(irs.RUBRIC, "click_target", "something else entirely")
    assert irs.sheet_build_id() != before


def test_build_id_travels_into_the_manifest():
    """So a verdict can be traced to the exact instrument that produced it."""
    html = irs.build_sheet(_meta(), _chips(), {"sheet_build": irs.sheet_build_id()})
    meta = json.loads(re.search(r"const META = (\{.*?\});\n", html, re.S).group(1))
    assert meta["manifest"]["sheet_build"] == irs.sheet_build_id()


def test_neighbour_pixel_offsets_land_where_the_record_is():
    """A marker drawn in the wrong place is worse than no marker — it would look
    like evidence. A record N metres east must sit N/mpp pixels right of centre
    and level with it, in the chip's own projection."""
    lat, lon, zoom = 39.7, -105.0, 21
    mpp = irs.metres_per_pixel(lat, zoom)
    east_m = 6.0
    lon_e = lon + east_m / (111320.0 * math.cos(math.radians(lat)))
    got = irs.find_neighbours([(lon_e, lat)], [(lon, lat)], 20.0, zoom=zoom)[0]
    assert len(got) == 1
    assert math.isclose(got[0]["dx_px"], east_m / mpp, rel_tol=2e-3)
    assert abs(got[0]["dy_px"]) < 0.5


def test_neighbour_pixel_offsets_point_north_up():
    """Screen y grows downward, so a record to the NORTH must have a NEGATIVE
    dy — getting this backwards would mirror every marker about the crosshair."""
    lat, lon, zoom = 39.7, -105.0, 21
    lat_n = lat + 6.0 / 111132.0
    got = irs.find_neighbours([(lon, lat_n)], [(lon, lat)], 20.0, zoom=zoom)[0]
    assert got[0]["dy_px"] < 0


def test_counts_and_markers_cannot_disagree():
    """Both are projections of one search, so a record can never be counted but
    not drawn, or drawn but not counted."""
    deg = 1.0 / (111320.0 * math.cos(math.radians(39.7)))
    pts = [(-105.0, 39.7), (-105.0 + 4 * deg, 39.7), (-105.0 + 8 * deg, 39.7)]
    counts = irs.count_neighbours(pts, [(-105.0, 39.7)], (6.0, 10.0))[0]
    found = irs.find_neighbours(pts, [(-105.0, 39.7)], 10.0)[0]
    assert counts == [2, 3]
    assert [sum(1 for n in found if n["d_m"] <= r) for r in (6.0, 10.0)] == counts


def test_published_data_is_held_until_the_chip_is_fully_recorded():
    """Anti-anchoring, and it is the whole value of the comparison: the imagery
    evidence and the published data have to be reached independently or their
    difference measures nothing.

    The gate needs BOTH of the chip's own numbers. Gating on the count alone left
    ``offset_m`` — the headline number — exposed, because a click made with the
    markers already on screen drifts toward one. Behaviour is exercised for real
    in ``test_review_sheet_page_logic.py``; this pins the rule in one place so it
    cannot be widened by accident.
    """
    html = irs.build_sheet(_meta(), _chips(), {})
    assert "function revealed(v) { return done(v) && v.ramps_visible != null; }" in html
    assert "if (!c.pub || !revealed(v)) return \"\";" in html


# --------------------------------------------------------------------------- #
# basemap registry
# --------------------------------------------------------------------------- #
def test_every_source_declares_provenance_and_a_depth_limit():
    for name, src in irs.TILE_SOURCES.items():
        assert src["attribution"] and src["note"], name
        assert src["max_zoom"] >= 1, name
        assert "{z}" in src["url"] and "{x}" in src["url"] and "{y}" in src["url"], name
