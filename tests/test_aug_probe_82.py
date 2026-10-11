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


RESULTS = os.path.join(REPO, "analysis_out", "aug_transfer_82", "probe_results.json")


def _results():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("split,arm", [("laurens_gsv", "all_brightness@gopro"),
                                       ("laurens_gsv", "photo_gamma@gopro"),
                                       ("bend", "res_all@gopro"),
                                       ("laurens_mapillary", "clahe@0.02")])
def test_doc_contrasts_rederive_from_committed_caches(split, arm):
    """The contrasts docs/aug_transfer_82.md quotes, recomputed from the committed caches
    (the full `report --check` takes ~40 s, so it is not in the default suite)."""
    from rampnet.detection_eval import radius_sq_for
    rsq = radius_sq_for()
    gts = P.bundle_ground_truths(split)[0]
    sc = {}
    for a in (P.CONTROL, arm):
        preds, _ = P.read_cache(P.cache_path(P.PROBE_ROOT, a, split))
        sc[a] = P._scored(split, P._panos(preds, gts), rsq)
    for thr in ("0.30", "0.55"):
        got = P._contrast(sc[arm], sc[P.CONTROL], [len(sc[arm].pids)], float(thr))
        assert got == _results()["per_split"][split]["vs_none"][arm][thr]


def test_headline_numbers():
    r = _results()
    g = r["laurens_gap"]["0.30"]
    assert (g["recall_gsv"], g["recall_gopro"], g["gap"]) == (0.6545, 0.5261, -0.1284)
    lg = r["per_split"]["laurens_gsv"]["vs_none"]
    assert lg["all_brightness@gopro"]["0.30"]["recall"]["observed"] == -0.0636
    assert lg["photo_brightness@gopro"]["0.30"]["recall"]["observed"] == -0.0091
    assert lg["photo_gamma@gopro"]["0.30"]["recall"]["observed"] == -0.0818
    pool = r["pooled"]["GSV pool (degrade)"]["vs_none"]
    assert pool["jpeg@gopro"]["0.30"]["recall"]["observed"] == -0.0181
    assert pool["all@gopro"]["0.30"]["recall"]["observed"] == -0.2092
    rep = r["pooled"]["GoPro pool (repair)"]["vs_none"]
    assert all(v["0.30"]["recall"]["observed"] <= 0 for v in rep.values())


def test_committed_markdown_matches_the_json():
    with open(STATS, encoding="utf-8") as f:
        stats = json.load(f)
    with open(P.RESULTS_MD, encoding="utf-8", newline="") as f:
        assert f.read() == P.markdown(_results(), stats)
