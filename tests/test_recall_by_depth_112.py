"""Tests for the depth-axis re-measure (#112).

CPU only. The geometry helpers are checked on closed forms; the committed
``analysis_out/recall_by_depth_112.json`` is checked to re-derive its own tables from its
per-point rows (the replication guard: the payloads are unpublished, the rows are not) and
to reproduce the doc's population, and the doc's §0 tables are pinned to it. The geometry is
checked on stub payloads (no checkout needed, so it runs in CI), including a tilted plane with an
explicit image-column <-> raw-column mapping; one parity test against the labeler's parser is
skipped when no labeler checkout is available.
"""
import json
import math
import os
import sys

from collections import namedtuple

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import recall_by_depth_112 as rbd  # noqa: E402

COMMITTED = os.path.join(REPO, "analysis_out", "recall_by_depth_112.json")


# ---------------------------------------------------------------------------
# closed forms

def test_flat_range_is_cotangent_and_none_above_horizon():
    # 45 degrees of depression: range equals the camera height
    assert rbd.flat_range(0.75, 2.5) == pytest.approx(2.5)
    assert rbd.flat_range(0.75, 1.7) == pytest.approx(1.7)
    # the same point at a lower camera is nearer, in proportion
    assert rbd.flat_range(0.6, 1.7) / rbd.flat_range(0.6, 2.5) == pytest.approx(1.7 / 2.5)
    assert rbd.flat_range(0.5, 2.5) is None
    assert rbd.flat_range(0.4, 2.5) is None


def test_apparent_px_uses_the_4096_px_input():
    # a 1.2 m ramp at 10 m subtends 0.12 rad -> 0.12 * 4096 / 2pi px
    assert rbd.apparent_px(10.0) == pytest.approx(0.12 * 4096 / (2 * math.pi))
    # nearer is larger, in inverse proportion
    assert rbd.apparent_px(5.0) == pytest.approx(2 * rbd.apparent_px(10.0))


def test_recall_table_buckets_and_all_row():
    pts = [{"d": 3.0, "hit": True}, {"d": 9.0, "hit": False}, {"d": 9.5, "hit": True},
           {"d": None, "hit": True}]
    rows = rbd.recall_table(pts, "d", [(0, 8), (8, 12)], "m")
    assert [r["bucket"] for r in rows] == ["0-8 m", "8-12 m", "all"]
    assert rows[0]["n"] == 1 and rows[0]["recall"] == 1.0
    assert rows[1]["n"] == 2 and rows[1]["recall"] == 0.5
    assert rows[2]["n"] == 3   # the point without the axis is not counted anywhere


def test_deflation_and_thresholds():
    pts = [{"flat_2p5": 12.0, "depth_range": 10.0, "hit": True},
           {"flat_2p5": 24.0, "depth_range": 20.0, "hit": False},
           {"flat_2p5": 6.0, "depth_range": 6.0, "hit": True}]
    d = rbd.deflation(pts)
    assert d["n"] == 3
    assert d["median_ratio"] == pytest.approx(1.2)
    assert rbd.deflated_thresholds(1.2) == [15.0, 20.8]
    assert rbd.deflation(pts[:2]) is None   # fewer than three pairs: no estimate


def test_resolution_forecast_moves_points_up_the_size_curve():
    # two buckets: small (recall 0) and large (recall 1); doubling size lifts the small ones
    pts = ([{"px": 25.0, "hit": False}] * 4) + ([{"px": 60.0, "hit": True}] * 4)
    f = rbd.resolution_forecast(pts, "px", factors=(2.0,))
    assert f[0]["base"] == 0.5 and f[0]["forecast"] == 1.0 and f[0]["gain"] == 0.5


# ---------------------------------------------------------------------------
# the committed artifact

@pytest.fixture(scope="module")
def committed():
    with open(COMMITTED, encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_tables_rederive_from_the_committed_rows(committed):
    assert rbd.tables(committed) == committed["tables"]


def test_committed_artifact_reproduces_the_docs_population(committed):
    # docs/detection_recall_analysis.md: 637 reviewer-confirmed ramps, recall 0.765 at 0.55
    pr = committed["tables"]["published_reproduction"]
    assert pr["n"] == 637 and pr["hit"] == 487
    assert pr["per_city"] == {"richmond": {"n": 310, "hit": 238}, "bend": {"n": 327, "hit": 249}}


def test_committed_artifact_is_lf_and_rounded(committed):
    with open(COMMITTED, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw
    # every float in every row carries at most ND decimals, except the coordinates, which are
    # stored exactly (review N2: rounding them moved points across a payload row)
    for row in committed["points"] + committed["detections"]:
        for key, v in row.items():
            if isinstance(v, float) and key not in ("x", "y", "confidence"):
                assert round(v, rbd.ND) == v, (key, v)
    # no machine-specific path in the artifact (review N3), so a derive elsewhere is byte-identical
    assert "labeler_root" not in committed
    assert isinstance(committed["labeler_commit"], str) and len(committed["labeler_commit"]) == 40


def test_the_doc_section_0_tables_are_the_committed_tables(committed):
    """Every table in docs/detection_recall_analysis.md §0 is pasted from ``doc_tables`` and
    must appear there verbatim (as test_scoreboard pins model_comparison.md)."""
    with open(os.path.join(REPO, "docs", "detection_recall_analysis.md"), encoding="utf-8") as fh:
        doc = fh.read().replace("\r\n", "\n")
    tabs = rbd.doc_tables(committed, rbd.tables(committed))
    assert len(tabs) == 9
    for name, tab in tabs.items():
        assert tab in doc, f"§0 table {name!r} in the doc does not match the committed rows"


def test_the_alignment_evidence_favours_the_mapping_the_script_uses():
    """analysis_out/depth_image_alignment_112.json: image column = raw column, raw-space ray."""
    with open(os.path.join(REPO, "analysis_out", "depth_image_alignment_112.json"), encoding="utf-8") as fh:
        al = json.load(fh)
    sky = al["A_sky"]["pooled"]
    assert sky["best_raw_within_2"] > 10 * max(1, sky["best_flip_within_2"])
    gt = al["B_ground_under_point"]["gt_points"]
    assert gt["raw"] > gt["flip"]
    edges = al["C_edges"]["pooled"]
    assert edges["raw_beats_flip_at_zero_shift"] > edges["panos"] / 2
    seam = al["D_seam_continuity"]
    assert seam["raw_formula"]["median_abs_log_ratio"] < seam["mirrored_formula"]["median_abs_log_ratio"] / 5


def test_depth_axis_never_backfills_a_non_measured_pano(committed):
    for p in committed["points"]:
        if p["camera_height_status"] != "measured":
            assert p["depth_range"] is None and p["depth_source"] is None
        else:
            assert p["depth_source"] is not None


def test_every_payload_read_is_hash_verified(committed):
    read = [p for p in committed["panos"] if p["sha256"]]
    assert read and all(p["sha256_matches_index"] for p in read)
    assert all(committed["index_sha256"][c] for c in committed["constants"]["depth_splits"])


# ---------------------------------------------------------------------------
# the geometry on synthetic payloads: stubs, no labeler checkout, so these run in CI

Plane = namedtuple("Plane", "nx ny nz d")
Payload = namedtuple("Payload", "width height planes indices")
NO_PLANE = Plane(0.0, 0.0, 0.0, 0.0)


def _payload(w, h, planes, index_of):
    """Raw index array from index_of(row, raw_col) -- raw column order, as the payload stores it."""
    return Payload(w, h, [NO_PLANE] + planes, bytes(index_of(r, c) for r in range(h) for c in range(w)))


def _raw_ray(w, h, row, raw_col):
    """The labeler's raw-column ray (depth._direction; streetlevel's convention), written out
    here independently: theta from the zenith, +z down, azimuth counted down from width."""
    theta = (h - row - 0.5) / h * math.pi
    phi = (w - raw_col - 0.5) / w * 2 * math.pi + math.pi / 2
    return math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)


def test_raw_column_is_the_identity_and_wraps():
    assert rbd.raw_column(0.0, 512) == 0
    assert rbd.raw_column(0.75, 512) == 384
    assert rbd.raw_column(0.999999, 512) == 511
    assert rbd.raw_column(1.0, 512) == 0          # the seam wraps
    assert rbd.raw_column(138 / 512, 512) == 138  # an exact column boundary belongs to the right


def test_depth_ranges_on_a_level_plane_equal_flat_ground_at_the_measured_height():
    w, h, cam_h = 512, 256, 1.8
    p = _payload(w, h, [Plane(0.0, 0.0, -1.0, cam_h)], lambda r, c: 0 if r < h // 2 else 1)
    for x, y in ((0.1, 0.6), (0.5, 0.75), (0.9, 0.55)):
        rng, ray, src = rbd.depth_ranges(p, cam_h, x, y)
        assert src == "pixel_plane"
        assert rng == pytest.approx(rbd.flat_range(y, cam_h), rel=1e-9)
        assert ray == pytest.approx(math.hypot(rng, cam_h), rel=1e-9)
    rng, ray, src = rbd.depth_ranges(p, cam_h, 0.5, 0.45)   # sky, above the horizon
    assert src == "fallback_sky_none" and rng is None


@pytest.mark.parametrize("raw_col", [40, 128, 200, 300, 384, 470])
def test_tilted_plane_known_answer_with_an_explicit_column_mapping(raw_col):
    """A 3-degree roll (nx != 0) is the case a mirrored azimuth gets wrong; a level plane
    cannot tell the two apart. Image column c is raw column c; the ray is the raw-column ray."""
    w, h, cam_h, roll = 512, 256, 2.0, math.radians(3.0)
    tilted = Plane(math.sin(roll), 0.0, -math.cos(roll), cam_h)
    p = _payload(w, h, [tilted], lambda r, c: 0 if r < h // 2 else 1)
    row = 150
    x, y = (raw_col + 0.5) / w, (row + 0.5) / h            # a pixel centre in the image
    v = _raw_ray(w, h, row, raw_col)                        # the same pixel's raw-column ray
    expected_ray = cam_h / abs(v[0] * tilted.nx + v[1] * tilted.ny + v[2] * tilted.nz)
    expected_rng = expected_ray * math.cos((0.5 - y) * math.pi)
    rng, ray, src = rbd.depth_ranges(p, cam_h, x, y)
    assert src == "pixel_plane"
    assert ray == pytest.approx(expected_ray, rel=1e-9)
    assert rng == pytest.approx(expected_rng, rel=1e-9)
    # and the mirrored answer (the labeler's stored-column mapping) is measurably different,
    # except where the roll is along the line of sight's normal (cos(phi) == 0)
    vm = _raw_ray(w, h, row, w - 1 - raw_col)
    mirrored = cam_h / abs(vm[0] * tilted.nx + vm[1] * tilted.ny + vm[2] * tilted.nz)
    if abs(v[0]) > 0.2:
        assert abs(mirrored / expected_ray - 1) > 0.01


def test_tilted_plane_is_nearer_on_the_side_it_rises_toward():
    """Physical sense, both sides: with the ground normal tipped toward +x the ground is nearer
    where the ray has vx < 0. Under the raw-column ray vx < 0 is image x in (0.5, 1)."""
    w, h, cam_h, roll = 512, 256, 2.0, math.radians(3.0)
    tilted = Plane(math.sin(roll), 0.0, -math.cos(roll), cam_h)
    p = _payload(w, h, [tilted], lambda r, c: 0 if r < h // 2 else 1)
    y = 0.6
    left, _, _ = rbd.depth_ranges(p, cam_h, 0.25, y)    # vx > 0: ground falls away
    right, _, _ = rbd.depth_ranges(p, cam_h, 0.75, y)   # vx < 0: ground rises toward the camera
    level = rbd.flat_range(y, cam_h)
    assert right < level < left


def test_depth_ranges_falls_back_under_a_wall_in_raw_columns():
    w, h, cam_h = 512, 256, 2.0
    ground, wall = Plane(0.0, 0.0, -1.0, cam_h), Plane(0.0, 1.0, 0.0, 6.0)

    def index_of(r, c):
        if r < h // 2:
            return 0
        # the wall occupies the near-horizon rows of RAW columns w/2 .. w-1
        return 2 if (r < h // 2 + 8 and c >= w // 2) else 1
    p = _payload(w, h, [ground, wall], index_of)
    y = (h // 2 + 2 + 0.5) / h
    # image column = raw column, so the wall is under image x >= 0.5
    rng, _, src = rbd.depth_ranges(p, cam_h, 0.75, y)
    assert src == "fallback_wall" and rng == pytest.approx(rbd.flat_range(y, cam_h))
    rng, _, src = rbd.depth_ranges(p, cam_h, 0.25, y)
    assert src == "pixel_plane" and rng == pytest.approx(rbd.flat_range(y, cam_h))


def test_window_threshold_uses_only_the_points_near_the_threshold():
    near = [{"flat_2p5": 18.0, "depth_range": 18.0 / 1.5}] * 12       # stretched 1.5x near 18 m
    far = [{"flat_2p5": 40.0, "depth_range": 40.0}] * 50               # unstretched elsewhere
    w = rbd.window_threshold(near + far, 18.0)
    assert w["n"] == 12 and w["median_ratio"] == 1.5 and w["deflated_m"] == 12.0
    assert rbd.window_threshold(far, 18.0)["deflated_m"] is None      # fewer than 10: no estimate


# ---------------------------------------------------------------------------
# parity with the labeler's parser (needs the checkout; skipped in CI)

LABELER = os.environ.get("LABELER_ROOT", r"D:\Git\sidewalk-auto-labeler")


@pytest.mark.skipif(not os.path.exists(os.path.join(LABELER, "depth.py")),
                    reason="no sidewalk-auto-labeler checkout (LABELER_ROOT)")
def test_image_x_is_the_labelers_image_x():
    """Both lookups answer in the panorama JPEG's frame: depth_ranges at image x equals
    depth.ray_depth_at at the same x, on a tilted plane, away from column edges.

    Until sidewalk-auto-labeler#84 (merged 2026-09-26) the labeler answered range queries in
    streetlevel's mirrored raster frame, and this test called it at 1 - x on purpose
    (sidewalk-auto-labeler#80, RampNet #191). On a plane with nx != 0 the two frames give
    different ranges, so the second assertion fails against a labeler checkout older than #84
    instead of letting the two repos disagree silently."""
    depthlib = rbd.load_depthlib(LABELER)
    w, h, cam_h, roll = 512, 256, 2.0, math.radians(3.0)
    tilted = depthlib.Plane(math.sin(roll), 0.0, -math.cos(roll), cam_h)
    idx = bytes(0 if r < h // 2 else 1 for r in range(h) for c in range(w))
    payload = depthlib.DepthPayload(w, h, [depthlib.Plane(0.0, 0.0, 0.0, 0.0), tilted], idx)
    for x in (0.1234, 0.3791, 0.6602, 0.9013):
        _, ray, _ = rbd.depth_ranges(payload, cam_h, x, 0.61)
        assert ray == pytest.approx(depthlib.ray_depth_at(payload, x, 0.61), rel=1e-12)
        assert abs(depthlib.ray_depth_at(payload, 1.0 - x, 0.61) / ray - 1) > 0.01, (
            "the labeler still answers in the mirrored raster frame (older than "
            "sidewalk-auto-labeler#84)")
