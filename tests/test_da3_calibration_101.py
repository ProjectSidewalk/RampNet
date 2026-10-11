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


def test_lowest_plane_skips_a_vehicle_roof():
    # a car-roof rig: most of the band is the roof 0.75 m under the camera, the rest is road
    # 2.1 m down. The single dominant plane is the roof; the ground is the lowest plane.
    rng = np.random.default_rng(5)
    xz_roof = rng.uniform(-1.2, 1.2, size=(3000, 2))
    xz_road = rng.uniform(-8, 8, size=(1500, 2))
    roof = np.column_stack([xz_roof[:, 0], np.full(3000, -0.75), xz_roof[:, 1]])
    road = np.column_stack([xz_road[:, 0], np.full(1500, -2.1), xz_road[:, 1]])
    P = np.concatenate([roof, road])
    assert dc.fit_ground_plane(P, seed=2)["h"] == pytest.approx(0.75, abs=1e-6)
    fit = dc.fit_lowest_plane(P, seed=2)
    assert fit["h"] == pytest.approx(2.1, abs=1e-6)
    assert fit["dominant_h"] == pytest.approx(0.75, abs=1e-6)
    assert fit["inlier_share"] == pytest.approx(1500 / 4500, abs=1e-6)
    assert [round(h, 2) for h, _ in fit["planes"]] == [0.75, 2.1]


def test_lowest_plane_ignores_a_minor_low_plane():
    # a sliver of a lower surface (5% of the band) is below the support floor and is not chosen
    rng = np.random.default_rng(6)
    road = np.column_stack([rng.uniform(-8, 8, 2000), np.full(2000, -2.0), rng.uniform(-8, 8, 2000)])
    pit = np.column_stack([rng.uniform(-1, 1, 100), np.full(100, -2.6), rng.uniform(2, 3, 100)])
    fit = dc.fit_lowest_plane(np.concatenate([road, pit]), seed=4)
    assert fit["h"] == pytest.approx(2.0, abs=1e-6)


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
    # the numbers docs/da3_calibration_101.md leads with
    assert pooled["median_ratio"] == 1.1059
    assert t["constants"]["k_height"] == 1.038
    lc = {r["split"]: r["da3_over_google_scaled"] for r in t["vs_labeler_frame_scale"]}
    assert lc == {"bend": 0.9747, "paterson": 1.027, "gainesville": 0.995, "sao_paulo": 1.0176}
    v = t["validation"]["pooled_common"]
    assert (v["flat_2p5"]["share_within_10pct"], v["da3_point"]["share_within_10pct"]) == (0.4442, 0.6402)
    rep = {r["split"]: r["da3_value"] for r in t["published_reproduction"]}
    # detection_recall_analysis.md: "agree to within 6.5-8.5% (Spearman 0.95 Bend / 0.81 Richmond)"
    assert (rep["bend"]["median_flat_over_da3"], rep["richmond"]["median_flat_over_da3"]) == (1.0645, 1.0849)
    unusable = {r["split"]: (r["n_at_or_above_horizon"], r["n_unusable_flat"]) for r in t["published_reproduction"]}
    assert unusable["richmond"] == (3, 4)   # the published "4 Richmond ramps": 3 above the horizon + 1 at >= 150 m
    # review M1: the two parts of the ground-fit change, attributed from committed rows
    gfc = {r["group"]: r for r in t["ground_fit_change"]}
    assert (gfc["morgantown"]["now_pass"], gfc["morgantown"]["now_threshold_only"]) == (92, 62)
    assert (gfc["clovis"]["now_pass"], gfc["clovis"]["now_threshold_only"]) == (112, 55)
    assert (gfc["gsv_google_depth"]["now_switched_plane"], gfc["gsv_google_depth"]["now_pass"]) == (76, 403)
    # review M4: the downstream method on GSV, DA3 axis vs Google's
    th = {r["group"]: r for r in t["gsv_method_validation"]["thresholds"]}
    assert (th["gsv_pooled"]["google"], th["gsv_pooled"]["da3_loso"]) == ([16.2, 21.7], [15.9, 21.3])
    # review M2: the labeler input is pinned
    lc = t["laurens_cross_read"]
    assert lc["labeler_commit"] == dc.LABELER_COMMIT and lc["labeler_files"] == dc.LABELER_FILES
    assert lc["labeler_rig"]["b_validated"] is False


# ---------------------------------------------------------------------------
# the held-out split (laurens_gsv): helpers on synthetic data, then the committed numbers

def _pano_row(split, pano, h, share=0.6, npts=500):
    k = f"{dc.DEPTH_CONVENTION}_{dc.PRIMARY_BAND}"
    return {"split": split, "pano": pano, "status": "ok", f"{k}_h": h, f"{k}_inlier_share": share,
            f"{k}_n_points": npts, f"{k}_h_sin_median": h}


def test_held_out_join_counts_every_step_and_dedupes_locations():
    google = {"panos": {"a": {"camera_height_status": "measured", "camera_height_m": 2.4},
                        "b": {"camera_height_status": "synthetic_ground", "camera_height_m": None},
                        "c": {"camera_height_status": "measured", "camera_height_m": 2.0}},
              "gt": {("a", 0.1, 0.6): {"camera_height_status": "measured", "depth_source": "pixel_plane", "depth_range": 10.0},
                     ("b", 0.2, 0.6): {"camera_height_status": "synthetic_ground", "depth_source": None, "depth_range": None}},
              "det": {("a", 0.1, 0.6): {"camera_height_status": "measured", "depth_source": "pixel_plane", "depth_range": 10.0}}}
    pts = [{"split": "laurens_gsv", "pano": "a", "kind": "gt", "x": 0.1, "y": 0.6, "da3_range": 11.0, "flat_2p5": 7.7},
           {"split": "laurens_gsv", "pano": "a", "kind": "det", "x": 0.1, "y": 0.6, "da3_range": 11.0, "flat_2p5": 7.7},
           {"split": "laurens_gsv", "pano": "b", "kind": "gt", "x": 0.2, "y": 0.6, "da3_range": 5.0, "flat_2p5": 7.7},
           {"split": "laurens_gsv", "pano": "a", "kind": "det", "x": 0.9, "y": 0.6, "da3_range": 5.0, "flat_2p5": 7.7},
           {"split": "bend", "pano": "a", "kind": "gt", "x": 0.1, "y": 0.6, "da3_range": 11.0, "flat_2p5": 7.7}]
    locs, n = dc.held_out_locations(pts, google)
    # a TP detection and its GT point are one location; a detection with no Google row is not joined;
    # a synthetic-ground location is joined but filtered; another split is never read
    assert n == {"point_rows": 4, "point_rows_joined": 3, "unique_locations": 2,
                 "locations_measured_pixel_plane": 1, "location_panos": 1}
    assert [(q["pano"], q["da3"], q["google"]) for q in locs] == [("a", 11.0, 10.0)]
    panos = [_pano_row("laurens_gsv", "a", 2.2), _pano_row("laurens_gsv", "b", 2.1),
             _pano_row("laurens_gsv", "c", 2.1, share=0.1), _pano_row("bend", "a", 2.0)]
    hp = dc.held_out_height_pairs(panos, google)
    # the intersection only: a is fit-ok and measured; b is not measured; c fails the fit
    assert [(q["pano"], q["da3"], q["google"]) for q in hp] == [("a", 2.2, 2.4)]


def test_two_sample_ratio_ci_finds_a_shift_and_not_its_absence():
    stat = lambda s: dc._median_ratio([(q["da3"], q["google"]) for q in s])  # noqa: E731
    rng = np.random.default_rng(0)

    def sample(scale, n_panos, tag):
        return [{"pano": f"{tag}{i}", "da3": scale * (1 + e), "google": 1.0}
                for i in range(n_panos) for e in rng.normal(0, 0.03, 3)]
    pooled, same, low = sample(1.10, 80, "p"), sample(1.10, 40, "s"), sample(0.99, 40, "l")
    key = lambda q: q["pano"]  # noqa: E731
    ci_same = dc.two_sample_ratio_ci(same, key, pooled, key, stat)
    ci_low = dc.two_sample_ratio_ci(low, key, pooled, key, stat)
    assert ci_same[0] <= 1.0 <= ci_same[1]
    assert ci_low[1] < 1.0 and ci_low[0] <= 0.9 <= ci_low[1]
    assert dc.two_sample_ratio_ci(low, key, pooled, key, stat) == ci_low   # seeded
    assert dc.two_sample_ratio_ci(low[:12], key, pooled, key, stat) is None   # < 5 clusters


def test_prediction_read_flags():
    ref = {"a": 0.96, "b": 1.03, "c": 1.05, "d": 1.11}
    inside = dc.prediction_read(1.04, [1.02, 1.06], 1.038, [1.026, 1.056], ref)
    assert inside["inside_predicted_ci"] and inside["cis_overlap"] and inside["inside_fitted_split_range"]
    out = dc.prediction_read(0.925, [0.92, 0.94], 1.038, [1.026, 1.056], ref)
    assert not out["inside_predicted_ci"] and not out["cis_overlap"] and not out["inside_fitted_split_range"]
    assert out["fitted_split_range"] == [0.96, 1.11]
    assert out["observed_over_predicted"] == pytest.approx(0.8911, abs=1e-4)


@needs_artifacts
def test_held_out_numbers_are_pinned():
    with open(dc.TABLES_JSON, encoding="utf-8") as fh:
        t = json.load(fh)["tables"]
    b = t["laurens_gsv_held_out"]
    assert b["n"] == {"panos": 86, "da3_fit_ok": 78, "google_measured": 60, "fit_ok_and_measured": 53,
                      "point_rows": 341, "point_rows_joined": 341, "unique_locations": 230,
                      "locations_measured_pixel_plane": 156, "location_panos": 45}
    assert b["rig_group"] == "older US vintages" and b["capture_years"] == ["2024"]
    # the unpaired read the #101 flag reported
    assert (b["unpaired"]["da3_calibrated_median_h_m"], b["unpaired"]["google_median_h_m"]) == (2.1443, 2.4094)
    p, h = b["prediction"]["point"], b["prediction"]["height"]
    assert (p["observed"], p["observed_ci"], p["predicted"]) == (1.0152, [0.997, 1.046], 1.1059)
    assert (p["observed_over_predicted"], p["held_out_over_pooled_ci"]) == (0.918, [0.8974, 0.9466])
    assert (h["observed"], h["observed_ci"], h["predicted"]) == (0.9252, [0.9197, 0.9416], 1.038)
    assert (h["observed_over_predicted"], h["held_out_over_pooled_ci"]) == (0.8913, [0.8748, 0.9116])
    for r in (p, h):
        assert not r["inside_predicted_ci"] and not r["cis_overlap"] and not r["inside_fitted_split_range"]
    assert (b["point"]["loglog_exponent"], b["point"]["loglog_exponent_ci"]) == (1.0147, [0.9818, 1.0415])
    ag = b["axis_agreement"]["common"]
    assert (ag["flat_2p5"]["share_within_10pct"], ag["da3_point"]["share_within_10pct"]) == (0.7424, 0.5909)
    # held out means held out: the pooled constants and LOSO set are the four fitted splits'
    assert set(t["constants"]["loso"]) == set(dc.GSV_DEPTH_SPLITS)
    assert all(r["group"] != "laurens_gsv" for r in t["point_calibration"] + t["height_calibration"])


def test_held_out_join_takes_the_first_passing_row_at_a_location():
    # review of #208, N4: as calib_locations does, a failing GT row does not block a passing
    # detection at the same coordinates
    google = {"panos": {}, "gt": {("a", 0.1, 0.6): {"camera_height_status": "synthetic_ground",
                                                     "depth_source": None, "depth_range": None}},
              "det": {("a", 0.1, 0.6): {"camera_height_status": "measured", "depth_source": "pixel_plane",
                                         "depth_range": 10.0}}}
    pts = [{"split": "laurens_gsv", "pano": "a", "kind": "gt", "x": 0.1, "y": 0.6, "da3_range": 11.0, "flat_2p5": 7.7},
           {"split": "laurens_gsv", "pano": "a", "kind": "det", "x": 0.1, "y": 0.6, "da3_range": 11.0, "flat_2p5": 7.7}]
    locs, n = dc.held_out_locations(pts, google)
    assert [(q["da3"], q["google"]) for q in locs] == [(11.0, 10.0)]
    assert (n["unique_locations"], n["locations_measured_pixel_plane"]) == (1, 1)


@needs_artifacts
def test_held_out_review_numbers_are_pinned():
    # review of #208, N5 / M1 / M2 / N3: the values the doc's reading rests on
    with open(dc.TABLES_JSON, encoding="utf-8") as fh:
        b = json.load(fh)["tables"]["laurens_gsv_held_out"]
    p, h = b["prediction"]["point"], b["prediction"]["height"]
    assert p["fitted_split_range"] == [1.0332, 1.1804] and h["fitted_split_range"] == [0.9587, 1.1085]
    assert p["fitted_split_range_over_predicted"] == [0.9343, 1.0674]
    assert h["fitted_split_range_over_predicted"] == [0.9236, 1.0679]
    # the pooled-CI test fails in sample too
    assert p["fitted_splits_outside_predicted_ci"] == ["bend", "gainesville", "sao_paulo"]
    assert h["fitted_splits_outside_predicted_ci"] == ["bend", "sao_paulo"]
    # medians behind the unpaired 0.890 and the paired 0.891
    assert (b["unpaired"]["da3_calibrated_median_h_m"], b["unpaired"]["google_median_h_m"],
            b["unpaired"]["calibrated_over_google"]) == (2.1443, 2.4094, 0.89)
    assert (b["height"]["median_da3_calibrated_h_m"], b["height"]["median_google_h_m"]) == (2.1445, 2.4141)
    # against bend: within noise at points, not in height
    vb = b["vs_bend"]
    assert vb["point"]["ratio_ci"] == [0.9537, 1.016] and vb["height"]["ratio_ci"] == [0.9448, 0.9872]
    # frame-corrected bands of the fitted splits contain laurens_gsv's uncorrected ratios
    fc = b["frame_corrected"]
    assert fc["point"]["fitted_splits"] == {"bend": 0.9747, "paterson": 1.027, "gainesville": 0.995, "sao_paulo": 1.0176}
    assert fc["height"]["fitted_splits"] == {"bend": 0.9044, "paterson": 0.9551, "gainesville": 0.9629, "sao_paulo": 0.9556}
    assert fc["point"]["held_out_inside_range"] and fc["height"]["held_out_inside_range"]
    assert fc["held_out_frame_scale"] is None
    # the one hand-carried cell in the doc's section 5.1 table matches the held-out block
    with open(os.path.join(REPO, "docs", "da3_calibration_101.md"), encoding="utf-8") as fh:
        row = next(line for line in fh if line.startswith("| laurens_gsv | 86 | 78 |"))
    assert row.rstrip().endswith(f"| {b['unpaired']['google_median_h_m']:.2f}¹ |")
