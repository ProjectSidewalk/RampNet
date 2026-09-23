"""Unit tests for the SAM2 extent experiment's geometry (#83, plan item 7).

Synthetic data only: no SAM2, no torch, no checkpoint, no panos, no network. What is
pinned is what fails silently -- a gnomonic view that does not round-trip, a mask box
that loses the seam, an IoU that compares a box against the wrong side of x = 0, and
a distance band read off the wrong row.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import sam2_extent_83 as se  # noqa: E402
from equirect_tiling import equirect_point_to_perspective, perspective_point_to_equirect  # noqa: E402


# ---------------------------------------------------------------------------
# gnomonic forward / back

@pytest.mark.parametrize("x,y", [(0.3, 0.6), (0.999, 0.55), (0.001, 0.7), (0.5, 0.5)])
def test_view_is_centered_on_the_point(x, y):
    view = se.view_for_point(x, y, 90, 512)
    X, Y = se.view_uv_to_equirect(np.array([0.5]), np.array([0.5]), view)
    assert float(se.unwrap_x(X[0], x)) == pytest.approx(0.0, abs=1e-9)
    assert Y[0] == pytest.approx(y, abs=1e-9)


@pytest.mark.parametrize("fov", [90, 60])
def test_vectorized_matches_scalar_and_round_trips(fov):
    view = se.view_for_point(0.02, 0.62, fov, 1000)       # pitched down, near the seam
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 1, 200), rng.uniform(0, 1, 200)
    X, Y = se.view_uv_to_equirect(u, v, view)
    for i in range(0, 200, 23):                            # agrees with the in-repo scalar
        sx, sy = perspective_point_to_equirect(u[i], v[i], view)
        assert float(se.unwrap_x(X[i], sx)) == pytest.approx(0.0, abs=1e-12)
        assert Y[i] == pytest.approx(sy, abs=1e-12)
        su, sv = equirect_point_to_perspective(X[i], Y[i], view)
        assert (su, sv) == pytest.approx((u[i], v[i]), abs=1e-9)
    u2, v2, ok = se.equirect_to_view_uv(X, Y, view)
    assert ok.all()
    np.testing.assert_allclose(u2, u, atol=1e-9)
    np.testing.assert_allclose(v2, v, atol=1e-9)


def test_behind_camera_is_flagged():
    view = se.view_for_point(0.25, 0.5, 90, 100)
    _, _, ok = se.equirect_to_view_uv(np.array([0.75]), np.array([0.5]), view)
    assert not ok[0]


def test_render_samples_the_right_place():
    # A pano whose red channel encodes column and green encodes row: the rendered
    # view's center pixel must read back the prompt's own pano pixel.
    H, W = 256, 512
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    src = np.stack([(cols % 256), rows, np.zeros_like(rows)], axis=-1).astype(np.uint8)
    x, y = 100.5 / W, 150.5 / H                           # a pixel center
    view = se.view_for_point(x, y, 60, 101)
    img = se.render_gnomonic(src, view, chunk_rows=17)     # chunking must not matter
    assert tuple(img[50, 50, :2]) == (100, 150)


# ---------------------------------------------------------------------------
# seam-aware boxes

def test_seam_iou_wraps():
    a = (0.998, 0.6, 0.01, 0.02)                          # spans [0.993, 1.003]
    b = (0.003 - 1e-3, 0.6, 0.01, 0.02)                   # same box written past 0
    assert se.seam_iou(a, a) == pytest.approx(1.0)
    iou = se.seam_iou(a, b)
    # x overlap: a [0.993, 1.003], b [-0.003, 0.007] -> [0.997, 1.007] after unwrap
    assert iou == pytest.approx((0.006 * 0.02) / (2 * 0.01 * 0.02 - 0.006 * 0.02), rel=1e-6)
    assert se.seam_iou(a, b) == pytest.approx(se.seam_iou(b, a), rel=1e-9)
    assert se.seam_iou(a, (0.5, 0.6, 0.01, 0.02)) == 0.0
    assert se.seam_iou(a, None) == 0.0


def test_seam_bbox_does_not_span_the_pano():
    xs = np.array([0.995, 0.999, 0.002, 0.004])
    ys = np.array([0.6, 0.61, 0.62, 0.6])
    cx, cy, w, h = se.seam_bbox(xs, ys, ref_x=0.0)
    assert w == pytest.approx(0.009)
    assert cx == pytest.approx(0.9995)
    assert (cy, h) == pytest.approx((0.61, 0.02))


def test_points_in_box_across_seam():
    box = (0.999, 0.5, 0.01, 0.1)
    inside = se.points_in_box(np.array([0.996, 0.003, 0.01, 0.999]),
                              np.array([0.5, 0.5, 0.5, 0.56]), box)
    assert inside.tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# mask -> equirect bbox

def _pano_with_box(W, H, box):
    """A black pano with the gold box painted white (seam-aware)."""
    src = np.zeros((H, W, 3), np.uint8)
    x0, x1 = se.box_x_interval(box, box[0])
    c0 = int(round((box[0] + x0) * W))
    c1 = int(round((box[0] + x1) * W))
    r0 = int(round((box[1] - box[3] / 2) * H))
    r1 = int(round((box[1] + box[3] / 2) * H))
    src[r0:r1, np.arange(c0, c1) % W] = 255
    return src, (c0, c1, r0, r1)


@pytest.mark.parametrize("cx", [0.4, 0.999])
def test_gnomonic_mask_maps_back_to_the_gold_box(cx):
    W, H = 4096, 2048
    box = (cx, 0.58, 0.02, 0.01)
    src, (c0, c1, r0, r1) = _pano_with_box(W, H, box)
    exact = (((c0 + c1) / 2 / W) % 1.0, (r0 + r1) / 2 / H, (c1 - c0) / W, (r1 - r0) / H)
    view = se.view_for_point(box[0], box[1], 90, 1024)
    mask = se.render_gnomonic(src, view)[..., 0] > 127
    got = se.gnomonic_mask_to_equirect_bbox(mask, view)
    # A view pixel here is ~1.3 pano pixels, so edges agree within ~2 pano px.
    assert se.seam_iou(exact, got) > 0.9
    assert abs(float(se.unwrap_x(got[0], exact[0]))) * W < 2
    assert abs(got[2] - exact[2]) * W < 4


def test_empty_mask_is_none():
    view = se.view_for_point(0.5, 0.6, 90, 64)
    assert se.gnomonic_mask_to_equirect_bbox(np.zeros((64, 64), bool), view) is None
    assert se.crop_mask_to_equirect_bbox(np.zeros((8, 8), bool), 0, 0, 100, 50) is None


def test_crop_mask_bbox_wraps_the_seam():
    W, H, side = 1000, 500, 100
    left, top = se.crop_rect(0.999, 0.5, W, H, side)
    assert left == 949
    mask = np.zeros((side, side), bool)
    mask[40:60, 45:56] = True                              # cols 994..1004 -> wraps
    cx, cy, w, h = se.crop_mask_to_equirect_bbox(mask, left, top, W, H)
    assert w == pytest.approx(11 / W)
    assert cx == pytest.approx(((949 + 50.5) / W) % 1.0)
    assert (cy, h) == pytest.approx(((top + 50) / H, 20 / H))


def test_cut_crop_array_wraps():
    src = np.arange(10)[None, :, None].repeat(3, axis=0)
    assert se.cut_crop_array(src, 8, 0, 3)[0, :, 0].tolist() == [8, 9, 0]


def test_score_mask_perfect_equirect_mask():
    W, H, side = 2000, 1000, 400
    item = {"pano_w": W, "pano_h": H, "gold": (0.5005, 0.6005, 0.021, 0.011)}
    left, top = se.crop_rect(0.5, 0.6, W, H, side)
    mask = np.zeros((side, side), bool)
    c0 = int(round((0.5005 - 0.0105) * W)) - left
    r0 = int(round((0.6005 - 0.0055) * H)) - top
    mask[r0:r0 + 11, c0:c0 + 42] = True
    s = se.score_mask(mask, ("equirect", left, top, side), item, 0.9)
    assert s["iou"] == pytest.approx(1.0, abs=1e-6)
    assert s["mask_frac_in_gold"] == 1.0
    assert s["gold_frac_covered"] == pytest.approx(1.0, abs=1e-6)
    assert s["mask_touches_edge"] is False


# ---------------------------------------------------------------------------
# bands, prior, stats

@pytest.mark.parametrize("dep,label", [(2, ">36 m / horizon"), (6, "18-36 m"), (10, "9-18 m"),
                                       (20, "5-9 m"), (40, "<5 m"), (-3, ">36 m / horizon")])
def test_band_from_box_row(dep, label):
    assert se.band_of(0.5 + dep / 180.0) == label


def test_prior_grows_toward_the_camera():
    far = se.prior_box(0.3, 0.5 + 5 / 180)
    near = se.prior_box(0.3, 0.5 + 30 / 180)
    assert near[2] > far[2] and near[3] > far[3]
    assert far[1] == pytest.approx(0.5 + 5 / 180, abs=0.01)
    view = se.view_for_point(0.3, 0.5 + 30 / 180, 90, 1000)
    x0, y0, x1, y1 = se.box_to_view_rect(near, view)
    assert x0 < 500 < x1 and y0 < 500 < y1


def test_box_to_crop_rect_across_seam():
    W, H, side = 1000, 500, 100
    left, top = se.crop_rect(0.999, 0.5, W, H, side)
    x0, y0, x1, y1 = se.box_to_crop_rect((0.001, 0.5, 0.01, 0.02), left, top, W, H, side)
    assert (x0, x1) == pytest.approx((47.0, 57.0))
    assert (y0, y1) == pytest.approx((45.0, 55.0))


def test_paired_delta_is_clustered_and_deterministic():
    rows = []
    for p in range(6):
        for k in range(3):
            for arm, iou in (("boxcenter_gnomonic", 0.6 + 0.01 * p), ("boxcenter_equirect", 0.5)):
                rows.append({"pano_id": f"p{p}", "key": f"det:{k}", "arm": arm, "fov": 90,
                             "variant": "pt_multi", "iou": iou})
    a = {"arm": "boxcenter_gnomonic", "fov": 90, "variant": "pt_multi"}
    b = {"arm": "boxcenter_equirect", "fov": 90, "variant": "pt_multi"}
    r1 = se.paired_delta(rows, a, b, reps=500)
    r2 = se.paired_delta(rows, a, b, reps=500)
    assert r1 == r2
    assert r1["n"] == 18 and r1["n_panos"] == 6
    assert r1["mean_delta"] == pytest.approx(0.125)
    assert 0.1 <= r1["ci95"][0] <= r1["ci95"][1] <= 0.15
    assert r1["a_better"] == 18


def test_csv_is_lf_and_rounded(tmp_path):
    row = {k: "" for k in se.CSV_FIELDS}
    row.update({"iou": se._r(1 / 3), "pano_id": "x"})
    path = tmp_path / "r.csv"
    se.write_csv(path, [row])
    data = path.read_bytes()
    assert b"\r\n" not in data
    assert b"0.33333," in data


# ---------------------------------------------------------------------------
# committed gold

@pytest.mark.parametrize("city,n_boxed,n_det", [("richmond", 299, 227), ("annapolis", 131, 94),
                                                ("sao_paulo", 119, 75), ("paterson", 109, 86)])
def test_items_load_and_det_points_are_the_detections(city, n_boxed, n_det):
    items, meta = se.load_items(os.path.join(REPO, "benchmark", city))
    assert len(items) == n_boxed == meta["n_boxed"]
    assert sum(it["kind"] == "det" for it in items) == n_det
    assert all(it["pano_w"] == 2 * it["pano_h"] for it in items)
