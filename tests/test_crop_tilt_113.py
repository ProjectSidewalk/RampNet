"""Tests for the #113 crop-tilt measurement (scripts/analysis/crop_tilt_113.py, rampnet/stage1_geometry.py).

CPU only, no network, no checkpoint, no sibling checkout: the vendored tilt functions are pinned to
values computed by the original sidewalk-panorama-tools module, the float projection to the paper-era
integer one (extracted from download_data.py by AST, so its network-reading module body never runs),
and --check re-derives the committed outputs.
"""

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from rampnet import stage1_geometry as sg

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_out" / "crop_tilt_113"


def _load_script():
    spec = importlib.util.spec_from_file_location("crop_tilt_113", REPO / "scripts/analysis/crop_tilt_113.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct = _load_script()


def _paper_point_projection():
    """download_data.py's equirectangular_point_to_perspective, compiled from its source alone."""
    src = (REPO / "stage_one/crop_model/ps_model/data/download_data.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "equirectangular_point_to_perspective")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "download_data.py", "exec"), ns)
    return ns["equirectangular_point_to_perspective"]


class TestVendoredTilt:
    # (x, y, w, h, pitch, roll) -> (rig_x, rig_y, T) from sidewalk-panorama-tools
    # reports/scripts/tilt_geometry.py @ 21d10aa, computed 2026-10-05 by importing that module.
    PINS = [
        ((14836.0, 5233.0, 16384, 8192, 1.7, -0.9), (14832.037262500999, 5320.034047812806, -1.9125886357853927)),
        ((3000.5, 4600.0, 16384, 8192, -2.4, 3.1), (2969.944647388512, 4683.362088956001, -1.8515394539741021)),
        ((9000.0, 4200.0, 13312, 6656, 0.6, 4.2), (9018.596229032319, 4050.987430559893, 4.023855055944538)),
    ]

    @pytest.mark.parametrize("args, want", PINS)
    def test_matches_pano_tools(self, args, want):
        x, y = sg.rig_pixel_from_gravity_pixel(*args)
        b = args[0] / args[2] * 360 - 180
        assert float(x) == pytest.approx(want[0], abs=1e-6)
        assert float(y) == pytest.approx(want[1], abs=1e-6)
        assert float(sg.tilt_term_deg(b, args[4], args[5])) == pytest.approx(want[2], abs=1e-9)

    def test_first_order_sign(self):
        """Nose down (pitch > 0) at the forward bearing: T > 0, the rig pixel sits ABOVE (smaller y)."""
        x, y = sg.rig_pixel_from_gravity_pixel(8192.0, 5000.0, 16384, 8192, 2.0, 0.0)
        assert float(y) == pytest.approx(5000.0 - 2.0 * 8192 / 180, abs=1.0)

    def test_xml_conversion(self):
        # pano-tools test_planning_example and a pool pano (-AjjQP44uRC5VTitfWFFNQ, Amsterdam)
        p, r = sg.xml_tilt_to_pitch_roll(269.22998, 110.909996, 4.73)
        assert (float(p), float(r)) == pytest.approx((4.395406789660389, 1.7473692092420987), abs=1e-9)
        p, r = sg.xml_tilt_to_pitch_roll(28.539999, 28.619999, 2.75)
        assert (float(p), float(r)) == pytest.approx((-2.7499973193671416, -0.0038397231067641775), abs=1e-9)


class TestProjection:
    def test_float_matches_paper_int(self):
        paper = _paper_point_projection()
        rng = np.random.default_rng(113)
        n_checked = 0
        for _ in range(50):
            theta = int(rng.choice(np.arange(-180, 181, 30)))
            b = theta + rng.uniform(-15, 15)
            x = (b + 180) / 360 * 8192
            y = rng.uniform(2100, 3600)            # elevations -2 to -68 degrees
            want = paper(x, y, 8192, 4096, 90, theta, -30, 2048, 2048)
            got = sg.equirect_point_to_perspective_float(x, y, 8192, 4096, 90, theta, -30, 2048, 2048)
            assert (want is None) == (got is None)
            if want is not None:
                assert (int(got[0]), int(got[1])) == want
                n_checked += 1
        assert n_checked == 50

    def test_strip_offset(self):
        sx, sy = sg.equirect_point_to_strip(0.5, 0.5 + 30 / 180, 0)   # straight ahead, 30 deg down
        assert sx == pytest.approx(1024 - 2048 / 3, abs=1e-6)
        assert sy == pytest.approx(1024, abs=1e-6)

    def test_renderer_agrees_with_point_projection(self):
        """A dot painted on an equirect lands, in the real renderer, where the point projection says."""
        pytest.importorskip("cv2")
        pytest.importorskip("torch")
        from rampnet.gsv import equirectangular_to_perspective
        W, H = 4096, 2048
        img = np.zeros((H, W, 3), np.uint8)
        b, el, theta = 40.0, -38.0, 30
        x = (b + 180) / 360 * W
        y = (90 - el) / 180 * H
        yy, xx = np.mgrid[0:H, 0:W]
        img[(xx - x) ** 2 + (yy - y) ** 2 <= 9] = 255
        persp = equirectangular_to_perspective(img, 90, theta, -30, 2048, 2048)
        ys, xs = np.nonzero(persp[..., 0] > 128)
        px, py = sg.equirect_point_to_perspective_float(x, y, W, H, 90, theta, -30, 2048, 2048)
        assert xs.mean() == pytest.approx(px, abs=3)
        assert ys.mean() == pytest.approx(py, abs=3)


class TestSinusoid:
    def test_pure_pitch_gives_cosine(self):
        """24 headings, pitch 2, roll 0: d_y = -k(b) * 2 cos b, k the local render px per degree."""
        ds, preds = [], []
        for i in range(24):
            b = -180 + 15 * i + 7.5
            g = ct.label_geometry((b + 180) / 360 * 16384, 8192 * (0.5 + 25 / 180), 16384, 8192, 2.0, 0.0)
            ds.append(g["d_y_b100"])
            preds.append(-g["px_per_deg_y"] * 2.0 * math.cos(math.radians(b)))
            assert g["T_deg"] == pytest.approx(2.0 * math.cos(math.radians(b)))
        # first order: the pitch also moves the point in x, and off the strip's centre column an x move
        # changes strip y a little, so allow ~3% of the 36 px amplitude
        np.testing.assert_allclose(ds, preds, atol=1.5)
        assert np.corrcoef(ds, preds)[0, 1] > 0.998
        assert max(ds) > 30 and min(ds) < -30                # ~ 19 px/deg * 2 deg at the extremes

    def test_beta_scales(self):
        g = ct.label_geometry(5000.0, 5200.0, 16384, 8192, 1.5, -2.0)
        assert g["d_y_b090"] == pytest.approx(0.9 * g["d_y_b100"], rel=0.01)

    def test_sigma_is_96_render_px(self):
        assert ct.SIGMA_RENDER == 96.0


@pytest.mark.skipif(not (OUT / "labels.csv").exists(), reason="outputs not committed")
def test_check_reproduces_committed_outputs():
    ct.main(["--check", "--out", str(OUT)])
