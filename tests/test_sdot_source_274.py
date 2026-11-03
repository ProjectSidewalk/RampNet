"""SDOT marked crosswalks as a Stage 1 location source, and the pilot's B1 projection (#274). No network."""
import importlib.util
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "stage_one", "dataset_generation"))

import sdot_source as ss  # noqa: E402


def feat(oid, lat, lng, status="INSVC", **p):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {"OBJECTID": oid, "CURRENT_STATUS": status, **p}}


def test_status_filter_sort_and_area():
    fs = [feat(5, 47.6, -122.3, MARKING_TYPE="PARSL"),
          feat(2, 47.6, -122.3, status="PLANNED"),
          feat(3, 47.6, -122.3, status=None),            # empty status counts as current (layer's own filter)
          feat(4, 47.6, -122.3, status="OUTSVC"),
          {"type": "Feature", "geometry": None, "properties": {"OBJECTID": 1, "CURRENT_STATUS": "INSVC"}},
          feat(6, 48.0, -122.3)]
    rows = ss.sdot_locations(fs, keep=lambda lat, lng: lat < 47.9)
    assert [r["source_id"] for r in rows] == ["sdot:3", "sdot:5"]
    assert rows[1]["marking"] == "PARSL" and rows[0]["date"] == ""
    assert list(rows[0]) == ss.FIELDS


def test_schema_prefix_matches_curb_ramp_locations():
    src = open(os.path.join(REPO, "stage_one", "dataset_generation", "combine_location_data.py"),
               encoding="utf-8").read()
    for col in ("latitude", "longitude", "date"):
        assert f'"{col}"' in src or f"'{col}'" in src
    assert ss.FIELDS[:3] == ["latitude", "longitude", "date"]


def _pilot():
    pytest.importorskip("pyproj")
    path = os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274", "pilot.py")
    spec = importlib.util.spec_from_file_location("pilot274", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b1_bearing_is_download_datasets_step():
    pilot = _pilot()
    from pyproj import Geod
    sys.path.insert(0, REPO)
    from rampnet.gsv import heading_to_azimuth
    pano, pt, heading = (47.6100, -122.3300), (47.61005, -122.32985), 250.0
    az, _, d = Geod(ellps="WGS84").inv(pano[1], pano[0], pt[1], pt[0])
    rel = az - heading_to_azimuth(heading)
    x, y, dist, rel_out = pilot.project(pano[0], pano[1], heading, pt[0], pt[1])
    assert rel_out == pytest.approx(rel)
    assert dist == pytest.approx(d)
    assert x == pytest.approx((0.5 + rel / 360.0) % 1.0)
    assert y == pytest.approx(0.5 + math.degrees(math.atan2(pilot.CAMERA_HEIGHT_M, d)) / 180.0)


def test_b1_x_matches_the_production_renderer():
    """A bright column at B1's x must come out at the centre of download_dataset's perspective render."""
    pilot = _pilot()
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    sys.path.insert(0, REPO)
    from rampnet.gsv import equirectangular_to_perspective
    for rel in (-120.0, -30.0, 0.0, 45.0, 170.0):
        x = (0.5 + rel / 360.0) % 1.0
        equi = np.zeros((256, 512, 3), dtype=np.uint8)
        c = int(round(x * 511))
        equi[:, max(0, c - 1):c + 2] = 255
        persp = equirectangular_to_perspective(equi, 90, rel, -30, 64, 64)
        cols = np.where(persp[32, :, 0] > 128)[0]
        assert len(cols) and abs(cols.mean() - 31.5) <= 2.0, (rel, cols)


def test_crowd_arm_explained_rule():
    """A Vistas component is explained by a mark near its centroid or inside its (seam-aware) horizontal extent."""
    pytest.importorskip("pyproj")
    path = os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274", "crowd_arm.py")
    spec = importlib.util.spec_from_file_location("crowd_arm274", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    long_near = {"x": 0.50, "y": 0.80, "x_left": 0.35, "x_right": 0.65}
    assert mod.explained(long_near, [(0.38, 0.81)])            # click at one end of a long near crosswalk
    assert not mod.explained(long_near, [(0.38, 0.60)])        # same column, far above it
    assert not mod.explained(long_near, [(0.10, 0.80)])        # elsewhere
    seam = {"x": 0.99, "y": 0.6, "x_left": 0.95, "x_right": 0.03}
    assert mod.explained(seam, [(0.02, 0.61)])                 # extent wraps the seam
    assert mod.explained({"x": 0.2, "y": 0.6, "x_left": 0.2, "x_right": 0.2}, [(0.22, 0.6)])   # near centroid
