"""#82 Step 1 probe: pure helpers and the arm table derived from the committed stats.

CPU only; reads analysis_out/aug_transfer_82/stats.json (committed). No panos, no GPU.
"""
import io
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
P = pytest.importorskip("aug_probe_82")

STATS = os.path.join(REPO, "analysis_out", "aug_transfer_82", "stats.json")


def _stats():
    with open(STATS, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("q", [50, 75, 95])
def test_jpeg_quality_estimate_recovers_the_encoder_setting(q):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                    ).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    est = P.jpeg_quality_estimate(Image.open(buf).quantization[0])
    assert abs(est - q) < 1.5


def test_invert_curve_interpolates_and_clamps():
    curve = [(0.0, 10.0), (1.0, 5.0), (2.0, 1.0)]
    assert P.invert_curve(curve, 7.5) == (0.5, False)
    assert P.invert_curve(curve, 100.0) == (0.0, True)
    assert P.invert_curve(curve, 0.1) == (2.0, True)


def test_sharpness_drops_with_blur():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 512), dtype=np.uint8))
    g0 = np.asarray(img, dtype=np.float64)
    g1 = np.asarray(P.A.blur(img.convert("RGB"), 1.0).convert("L"), dtype=np.float64)
    assert P.sharpness(g1)[0] < P.sharpness(g0)[0] / 5


def test_levels_derived_from_committed_stats():
    """Pins the probe's levels to the committed stats; a change to either moves these."""
    lv = P.derive_levels(_stats())
    assert lv["blur"]["levels"] == (0.51, 0.719, 1.153)
    assert lv["downscale"]["levels"] == (0.929, 0.854, 0.504)
    assert lv["brightness"]["levels"] == (0.823, 0.677, 0.557)
    assert lv["saturation"]["levels"] == (1.219, 1.486, 1.811)
    assert lv["contrast"]["levels"] == (1.074, 1.154, 1.239)
    assert lv["wb"]["levels"] == (-0.168, -0.337, -0.505)
    assert lv["jpeg"]["levels"] == (89.8, 74.8, 49.8)
    assert "clovis" in lv["blur"]["note"]
    assert "negligible" in lv["noise"]["note"]


def test_arm_table_shape():
    arms, _, _ = P.build_arms(_stats())
    assert arms[P.CONTROL]["splits"] == P.PROBE_SPLITS and arms[P.CONTROL]["ops"] == []
    assert all(set(d["splits"]) <= set(P.GSV_SPLITS) for a, d in arms.items()
               if "@" in a and d["axis"] in P.derive_levels(_stats()))
    assert arms["blur@gopro"]["splits"] == P.GSV_SPLITS
    assert len(arms) == 40 and sum(len(d["splits"]) for d in arms.values()) == 103


def test_stats_cover_the_eleven_splits_with_panos():
    s = _stats()
    assert sorted(s["summary"]) == sorted(c for c in P.STATS_SPLITS if c != "manual_gold")
