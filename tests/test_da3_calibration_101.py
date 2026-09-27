"""Tests for the DA3-vs-GSV-depth calibration (#101).

CPU only, no network, no DA3. The geometry is checked on a synthetic scene whose z-depth is
known in closed form (level or tilted ground seen through the six extraction views); the
statistics on hand-made numbers; and the committed artifacts are checked to re-derive: rows
from the committed raw DA3 files + bundles + the #112 JSON, tables from rows, markdown from
tables, and every file against the committed SHA256SUMS (the ``--check`` path).
"""
import json
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import da3_calibration_101 as dc  # noqa: E402
from equirect_tiling import default_views, _camera_basis  # noqa: E402

HAVE_ARTIFACTS = os.path.exists(dc.TABLES_JSON)
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="committed #101 artifacts not present")


def synthetic_zdepth(view, h, normal=(0.0, 1.0, 0.0), size=64):
    """Planar z-depth of the plane n.p = -h seen through ``view`` (inf where the ray misses)."""
    f, r, up = (np.array(t) for t in _camera_basis(view.yaw_deg, view.pitch_deg))
    th = math.tan(math.radians(view.fov_h_deg) / 2)
    tv = math.tan(math.radians(view.fov_v_deg) / 2)
    u = (np.arange(size) + 0.5) / size
    v = (np.arange(size) + 0.5) / size
    a = ((2 * u - 1) * th)[None, :, None]
    b = ((1 - 2 * v) * tv)[:, None, None]
    D = f + a * r + b * up
    n = np.array(normal) / np.linalg.norm(normal)
    denom = D @ n
    with np.errstate(divide="ignore"):
        z = np.where(denom < -1e-9, -h / denom, np.inf)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# geometry

def test_plane_range_level_ground_is_the_flat_range():
    # 45 deg below the horizon on level ground at 2 m: range 2 m
    assert dc.plane_range((0.0, 1.0, 0.0), 2.0, 0.3, 0.75) == pytest.approx(2.0)
    for y in (0.55, 0.6, 0.7, 0.9):
        assert dc.plane_range((0.0, 1.0, 0.0), 1.7, 0.1, y) == pytest.approx(dc.rbd.flat_range(y, 1.7))
    # at or above the horizon the ray never meets the ground
    assert dc.plane_range((0.0, 1.0, 0.0), 2.0, 0.3, 0.5) is None
    assert dc.plane_range((0.0, 1.0, 0.0), 2.0, 0.3, 0.4) is None


def test_ray_from_z_value_recovers_the_flat_range():
    views = default_views()
    h = 2.2
    for x, y in ((0.5, 0.62), (0.13, 0.7), (0.81, 0.58), (0.99, 0.66)):
        _, vi, (u, v) = dc.best_view(x, y, views)
        vw = views[vi]
        d = dc.view_dir_unnormalized(vw, u, v)
        z = -h / d[1]   # z-depth of the level-ground point on that ray
        ray = dc.ray_from_value(z, vw, u, v, convention="z")
        assert dc.horizontal_range(ray, y) == pytest.approx(dc.rbd.flat_range(y, h), rel=1e-6)
        # read as ray depth, the same number would be short by |D|
        assert dc.ray_from_value(z, vw, u, v, convention="ray") == pytest.approx(z)


@pytest.mark.parametrize("h,normal", [(2.0, (0.0, 1.0, 0.0)), (1.6, (0.05, 1.0, -0.08))])
def test_ground_fit_recovers_height_and_tilt_from_synthetic_views(h, normal):
    views = default_views()
    P = np.concatenate([dc.band_points(synthetic_zdepth(vw, h, normal), vw, 20, 45, "z", stride=2,
                                       az_half=30.0) for vw in views])
    fit = dc.fit_ground_plane(P, seed=1)
    n = np.array(normal) / np.linalg.norm(normal)
    assert fit["h"] == pytest.approx(h, abs=1e-6)
    assert fit["tilt_deg"] == pytest.approx(math.degrees(math.acos(n[1])), abs=1e-4)
    assert fit["inlier_share"] == pytest.approx(1.0)
    # the band really is the band: every kept point is 20-45 deg below the horizon
    dep = np.degrees(np.arcsin(-P[:, 1] / np.linalg.norm(P, axis=1)))
    if normal == (0.0, 1.0, 0.0):
        assert dep.min() >= 20 - 1e-6 and dep.max() <= 45 + 1e-6


def test_ground_fit_reading_z_as_ray_bends_the_road():
    views = default_views()
    P_z = np.concatenate([dc.band_points(synthetic_zdepth(vw, 2.0), vw, 20, 45, "z", stride=2, az_half=30.0)
                          for vw in views])
    P_r = np.concatenate([dc.band_points(synthetic_zdepth(vw, 2.0), vw, 20, 45, "ray", stride=2, az_half=30.0)
                          for vw in views])
    fz, fr = dc.fit_ground_plane(P_z, seed=1), dc.fit_ground_plane(P_r, seed=1)
    assert fz["inlier_share"] == pytest.approx(1.0)
    assert fr["inlier_share"] < 0.9   # the wrong reading does not lie on one plane


def test_ground_fit_ignores_an_obstacle():
    views = default_views()
    P = np.concatenate([dc.band_points(synthetic_zdepth(vw, 2.0), vw, 20, 45, "z", stride=2, az_half=30.0)
                        for vw in views])
    rng = np.random.default_rng(0)
    clutter = rng.uniform([-3, -1.5, 2], [3, 0, 6], size=(len(P) // 4, 3))   # a car-sized box above ground
    fit = dc.fit_ground_plane(np.concatenate([P, clutter]), seed=3)
    assert fit["h"] == pytest.approx(2.0, abs=1e-3)
    assert fit["inlier_share"] == pytest.approx(len(P) / (len(P) + len(clutter)), abs=0.02)


def test_rig_key_normalizes_like_the_labeler():
    assert dc.rig_key("GoPro", "GoPro Fusion FS1.04.01.80.00") == "gopro/fusion"
    assert dc.rig_key("GoPro", "GoPro Max") == "gopro/max"
    assert dc.rig_key("Trimble", "Trimble mx7") == "trimble/mx7"
    assert dc.rig_key("none", "none") == "unknown"
    assert dc.rig_key(None, None) == "unknown"


# ---------------------------------------------------------------------------
# statistics

def test_ols_and_pearson_closed_forms():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]
    assert dc.ols(xs, ys) == pytest.approx((2.0, 1.0))
    assert dc.pearson(xs, ys) == pytest.approx(1.0)
    assert dc.ols(xs[:2], ys[:2]) == (None, None)


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    items = [{"c": i // 3, "a": 1.0 + 0.01 * i, "b": 1.0} for i in range(60)]
    stat = lambda s: dc._median_ratio([(q["a"], q["b"]) for q in s])  # noqa: E731
    ci1 = dc.cluster_bootstrap(items, lambda q: q["c"], stat)
    ci2 = dc.cluster_bootstrap(items, lambda q: q["c"], stat)
    assert ci1 == ci2
    est = stat(items)
    assert ci1[0] <= est <= ci1[1]


def test_agreement_counts_within_ten_percent():
    a = dc.agreement([(10.0, 10.0), (10.5, 10.0), (12.0, 10.0), (8.0, 10.0)])
    assert a["n"] == 4
    assert a["share_within_10pct"] == 0.5
    assert a["median_ratio"] == pytest.approx(1.025)


# ---------------------------------------------------------------------------
# the committed artifacts (--check)

@needs_artifacts
def test_committed_artifacts_re_derive():
    assert dc.check() == 0


@needs_artifacts
def test_committed_files_are_lf_and_rounded():
    for path in dc.committed_files():
        with open(path, "rb") as fh:
            data = fh.read()
        assert b"\r" not in data, path
    with open(dc.ROWS_POINTS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for k in ("da3_range", "da3_ray", "flat_2p5", "da3_value"):
                v = row[k]
                assert v is None or round(v, dc.ND) == v, (k, v)


@needs_artifacts
def test_headline_numbers_are_pinned():
    with open(dc.TABLES_JSON, encoding="utf-8") as fh:
        t = json.load(fh)["tables"]
    inv = {r["split"]: r for r in t["inventory"]}
    assert set(inv) == set(dc.ALL_SPLITS)
    # every verdict-reviewed pano was extracted and verified against its manifest
    for s, r in inv.items():
        assert r["extracted"] == r["panos"], s
    pooled = next(r for r in t["point_calibration"] if r["group"] == "gsv_pooled")
    assert pooled["n"] > 1000
    assert t["headline_axis"] in dc.DA3_AXES
    assert t["constants"]["depth_convention"] == dc.DEPTH_CONVENTION
