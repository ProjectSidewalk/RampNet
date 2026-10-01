"""Geometry of the #217 width estimator on synthetic label maps with a known width.

CPU only, no network, no checkpoint: a flat sidewalk of known width is rendered through
the same pinhole model the estimator inverts, and the estimator must recover it."""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import sidewalk_width_217 as S  # noqa: E402

SIDEWALK, ROAD, SKY, POLE, PERSON = 15, 13, 27, 45, 19


def render(width_m, yaw_deg=0.0, offset_m=0.0, pitch_deg=0.0, h=1.0, f=1000.0,
           shape=(1080, 1440), obstacle=None, person=None):
    """Label map of a straight sidewalk ``width_m`` wide, centred ``offset_m`` right of the
    camera, running at ``yaw_deg``; road outside, sky above the horizon. ``obstacle`` /
    ``person`` = (s0, s1, z0, z1) footprints (lateral, forward) painted flat on the ground."""
    H, W = shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    X, Z = S.backproject(uu, vv, f, cx, cy, h, math.radians(pitch_deg))
    psi = math.radians(yaw_deg)
    s = X * math.cos(psi) - Z * math.sin(psi)          # lateral, across the path
    t = X * math.sin(psi) + Z * math.cos(psi)          # along the path
    lab = np.full(shape, SKY, np.uint8)
    ground = np.isfinite(X)
    lab[ground] = ROAD
    lab[ground & (np.abs(s - offset_m) <= width_m / 2)] = SIDEWALK
    for fp, cls in ((obstacle, POLE), (person, PERSON)):
        if fp:
            s0, s1, z0, z1 = fp
            lab[ground & (s >= s0) & (s <= s1) & (t >= z0) & (t <= z1)] = cls
    return lab, f, cx, cy


def estimate(lab, f, cx, cy, pitch=0.0, passable=(), zmin=2.0, band=3.0, stat="median"):
    l, r = S.row_spans(lab, cx, S.WALK_SETS["base"], passable)
    l, r = S.exclude_border(l, r, lab.shape[1])
    Z, Wd, yaw = S.ground_widths(l, r, f, cx, cy, 1.0, pitch)
    return S.band_stat(Z, Wd, zmin, band, stat), yaw


def test_backproject_level_example():
    X, Z = S.backproject(np.array([600.0]), np.array([700.0]), 1000.0, 500.0, 500.0, 1.0, 0.0)
    assert X[0] == pytest.approx(0.5) and Z[0] == pytest.approx(5.0)


def test_backproject_above_horizon_is_nan():
    X, Z = S.backproject(np.array([500.0]), np.array([400.0]), 1000.0, 500.0, 500.0, 1.0, 0.0)
    assert np.isnan(X[0]) and np.isnan(Z[0])


def test_backproject_pitch_roundtrip():
    # a ground point 4 m ahead, pitch 5 deg: project it, then back-project it
    p = math.radians(5)
    f, cx, cy, h = 1000.0, 500.0, 500.0, 1.0
    Zw, Xw = 4.0, 0.7
    # camera coords of the ground point
    yc = h * math.cos(p) - Zw * math.sin(p)
    zc = h * math.sin(p) + Zw * math.cos(p)
    u, v = cx + f * Xw / zc, cy + f * yc / zc
    X, Z = S.backproject(np.array([u]), np.array([v]), f, cx, cy, h, p)
    assert X[0] == pytest.approx(Xw, abs=1e-9) and Z[0] == pytest.approx(Zw, abs=1e-9)


def test_focal_px_iphone():
    assert S.focal_px(26, 5712, 4284) == pytest.approx(4290.6, abs=0.1)


@pytest.mark.parametrize("width", [1.0, 1.8, 3.0])
def test_level_straight_width_recovered(width):
    lab, f, cx, cy = render(width)
    w, yaw = estimate(lab, f, cx, cy)
    assert w == pytest.approx(width, rel=0.02)
    assert abs(yaw) < 0.01


def test_yaw_is_corrected():
    lab, f, cx, cy = render(2.0, yaw_deg=10.0, offset_m=0.2)
    w, yaw = estimate(lab, f, cx, cy)
    assert math.degrees(yaw) == pytest.approx(10.0, abs=0.5)
    assert w == pytest.approx(2.0, rel=0.02)


def test_wrong_pitch_scales_width():
    # rendered at 2 deg down, measured as level: overstated, as the doc's sensitivity says
    lab, f, cx, cy = render(2.0, pitch_deg=2.0)
    w_level, _ = estimate(lab, f, cx, cy, pitch=0.0)
    w_true, _ = estimate(lab, f, cx, cy, pitch=math.radians(2.0))
    assert w_true == pytest.approx(2.0, rel=0.02)
    assert w_level > 2.1


def test_vanishing_point_recovers_pitch():
    lab, f, cx, cy = render(2.0, pitch_deg=3.0, yaw_deg=4.0)
    l, r = S.row_spans(lab, cx, S.WALK_SETS["base"], ())
    l, r = S.exclude_border(l, r, lab.shape[1])
    rows = np.flatnonzero(l >= 0)
    p = S.vp_pitch(l, r, rows, f, cy)
    assert p is not None and math.degrees(p) == pytest.approx(3.0, abs=0.2)


def test_wide_sidewalk_truncated_rows_are_dropped():
    # 6 m wide: the near rows run off the frame and must not count as narrow
    lab, f, cx, cy = render(6.0)
    w, _ = estimate(lab, f, cx, cy, zmin=1.0, band=2.0)
    assert w == pytest.approx(6.0, rel=0.03)


def test_obstacle_blocks_clear_not_total():
    # 3 m sidewalk, a fixed obstacle from 0.6 to 0.9 m right of centre, 2-6 m ahead:
    # total width 3.0; clear run containing the walking line ends at 0.6 -> 1.5 + 0.6
    lab, f, cx, cy = render(3.0, obstacle=(0.6, 0.9, 2.0, 6.0))
    total, _ = estimate(lab, f, cx, cy, passable=(POLE,), zmin=3.0, band=2.0)
    clear, _ = estimate(lab, f, cx, cy, passable=(), zmin=3.0, band=2.0)
    assert total == pytest.approx(3.0, rel=0.03)
    assert clear == pytest.approx(2.1, rel=0.04)


def test_person_is_passable_for_clear():
    lab, f, cx, cy = render(3.0, person=(-0.2, 0.2, 2.0, 6.0))
    clear, _ = estimate(lab, f, cx, cy, passable=S.TRANSIENT, zmin=3.0, band=2.0)
    assert clear == pytest.approx(3.0, rel=0.03)


def test_split_is_grouped_and_deterministic():
    gt = S._read_gt()
    h1, g1 = S.split_groups(gt)
    h2, _ = S.split_groups(gt)
    assert h1 == h2 and len(h1) == 514
    for grp in set(g1.values()):
        assert len({h1[n] for n in g1 if g1[n] == grp}) == 1
    assert 150 < sum(v == "A" for v in h1.values()) < 364


def test_metrics_flags_and_failures():
    est = np.array([1.0, 1.1, 2.0, np.nan])
    gt = np.array([1.0, 1.5, 1.1, 1.0])
    m = S.metrics(est, gt)
    assert m["n_estimated"] == 3 and m["n_narrow_gt"] == 3  # the NaN one is narrow: a miss
    assert m["tp"] == 1 and m["n_flagged"] == 2
    assert m["recall_lt_1_2"] == pytest.approx(1 / 3) and m["precision_lt_1_2"] == 0.5


def test_committed_headline_recomputes_from_per_image():
    """results.json's half-B headline is what its own per-image estimates give against
    the committed GT table (a cheap check; the full re-score is in the doc)."""
    import json
    with open(os.path.join(REPO, "analysis_out", "sidewalk_width_217", "results.json"),
              encoding="utf-8") as f:
        res = json.load(f)
    gt = S._read_gt()
    for meas in ("clear", "total"):
        r = res["measures"][meas]
        names = sorted(n for n in gt if res["half"][n] == "B")
        est = np.array([np.nan if r["per_image"][n] is None else r["per_image"][n]
                        for n in names])
        g = np.array([float(gt[n]["width"]) for n in names])
        m = S.metrics(est, g)
        assert m["mae"] == pytest.approx(r["metrics_B"]["mae"], abs=1e-3)
        assert m["tp"] == r["metrics_B"]["tp"]
        assert m["n_flagged"] == r["metrics_B"]["n_flagged"]


def test_every_class_id_used_is_checked():
    """#225 review N4: every Vistas id in a class group is in EXPECTED_LABELS, so a
    checkpoint whose ids differ fails at load time."""
    used = set(S.TRANSIENT)
    for group in (S.WALK_SETS, S.OBSTACLE_SETS, S.MARK_SETS):
        for ids in group.values():
            used |= set(ids)
    assert used <= set(S.EXPECTED_LABELS), sorted(used - set(S.EXPECTED_LABELS))


def test_committed_gt_csv_is_the_zenodo_file():
    """#225 review N5: the committed GT table's md5 is Zenodo's, as recorded in the manifest."""
    import seoul_fetch_217 as F
    ok, got, want = F.verify_committed_csv()
    assert ok, (got, want)


def test_score_reproduces_committed_results(tmp_path):
    """#225 review N7: re-scoring the committed widths CSV picks the committed configurations
    and reproduces the committed half-B metrics and sensitivity reads (bootstrap draws cut to
    200: the point metrics and the sensitivity block do not depend on them)."""
    import argparse
    import json
    d = os.path.join(REPO, "analysis_out", "sidewalk_width_217")
    out, sens = tmp_path / "results.json", tmp_path / "sensitivity.json"
    S.score(argparse.Namespace(widths=os.path.join(d, "widths.csv.gz"), out=str(out),
                               n_boot=200, min_coverage=0.9, sensitivity_out=str(sens)))
    with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
        want = json.load(f)
    got = json.loads(out.read_text(encoding="utf-8"))
    for meas in ("clear", "total"):
        assert got["measures"][meas]["config"] == want["measures"][meas]["config"]
        assert got["measures"][meas]["metrics_B"] == want["measures"][meas]["metrics_B"]
        assert got["measures"][meas]["per_image"] == want["measures"][meas]["per_image"]
    with open(os.path.join(d, "sensitivity.json"), encoding="utf-8") as f:
        _assert_same(json.loads(sens.read_text(encoding="utf-8")), json.load(f))


def _assert_same(got, want, path="", boot=False):
    """Exact equality, except bootstrap intervals (keys ending ``_ci95``), which may move in
    the last rounded digit between numpy builds: those match to 5e-4."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), path
        for k in want:
            _assert_same(got[k], want[k], f"{path}/{k}", boot or k.endswith("_ci95"))
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_same(g, w, f"{path}[{i}]", boot)
    elif boot and isinstance(want, float):
        assert got == pytest.approx(want, abs=5e-4), path
    else:
        assert got == want, path


@pytest.mark.parametrize("width,pitch_deg", [(1.2, 0.0), (1.2, -3.4), (3.0, -3.4), (3.0, 3.0)])
def test_focal_error_barely_moves_width(width, pitch_deg):
    """#225 review S2: with the horizon row taken from the image (the edges' vanishing point,
    recomputed with the WRONG focal length, as the pipeline would), a +-10% focal-length
    error moves width by well under 1%. Pitch, not focal length, is the dominant term."""
    lab, f, cx, cy = render(width, pitch_deg=pitch_deg, yaw_deg=5.0)
    l, r = S.row_spans(lab, cx, S.WALK_SETS["base"], ())
    l, r = S.exclude_border(l, r, lab.shape[1])
    rows = np.flatnonzero(l >= 0)
    out = {}
    for k in (0.9, 1.0, 1.1):
        fk = f * k
        p = S.vp_pitch(l, r, rows, fk, cy)
        assert p is not None
        Z, Wd, _ = S.ground_widths(l, r, fk, cx, cy, 1.0, p)
        out[k] = S.band_stat(Z, Wd, 1.5, 1.0, "median")
    assert out[1.0] == pytest.approx(width, rel=0.03)
    for k in (0.9, 1.1):
        assert abs(out[k] / out[1.0] - 1) < 0.005, (k, out)


def test_example_selection_is_pinned():
    """The eight contact-sheet photos follow from committed files alone; a change to the
    selection rule or to results.json that moves them should be a deliberate edit."""
    import sidewalk_width_217_figures as F
    names, half, G, C, _, cfg = F.load()
    picks = F.select_examples(names, half, G, C)
    assert picks == [
        ("best", "IMG_4299.HEIC"), ("median", "IMG_6417.HEIC"),
        ("worst over", "IMG_6839.HEIC"), ("worst under", "IMG_6398.HEIC"),
        ("true <1.2 m", "IMG_4699.HEIC"), ("false <1.2 m", "IMG_6694.HEIC"),
        ("no estimate", "IMG_4500.HEIC"), ("2nd worst over", "IMG_6451.HEIC")]
    assert all(half[n] == "B" for _, n in picks)
    assert cfg["clear"]["horizon"] == "vp"   # overlay_geometry draws only this horizon


def test_pitch_label_has_no_negative_zero():
    import sidewalk_width_217_figures as F
    assert f"{F.pitch_deg(math.radians(-0.04)):+.1f}" == "+0.0"
    assert f"{F.pitch_deg(math.radians(-6.9)):+.1f}" == "-6.9"
