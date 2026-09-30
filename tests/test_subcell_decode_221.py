"""The #221 report re-derives from the committed per-peak neighbourhoods (CPU, no model)."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
import subcell_decode_221 as sd  # noqa: E402

DETS = os.path.join(REPO, "analysis_out", "subcell_decode_221", "detections.json")
RESULTS = os.path.join(REPO, "analysis_out", "subcell_decode_221", "results.json")
needs_outputs = pytest.mark.skipif(not (os.path.exists(DETS) and os.path.exists(RESULTS)),
                                   reason="committed #221 outputs absent")


@needs_outputs
def test_results_name_the_committed_detections():
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["inputs"]["detections_sha256"] == sd.sha256_file(DETS)


@needs_outputs
@pytest.mark.parametrize("split", ["manual_gold", "richmond"])
def test_point_estimates_rederive(split):
    """Pairs, per-decode residual stats and the mechanism counts, without the bootstrap."""
    with open(DETS, encoding="utf-8") as f:
        recs = json.load(f)["panos"][split]
    with open(RESULTS, encoding="utf-8") as f:
        rep = json.load(f)
    rows = sd.build_pairs(recs, sd.ground_truth(split), sd.radius_sq_for())
    got = rep["splits"][split]
    assert len(rows) == got["pairs"]
    for m in ("argmax", "gaussian", "dark"):
        st = sd.rnd(sd.stats(*sd.residuals(rows, m, False)))
        for k, v in st.items():
            assert v == pytest.approx(got["methods"][m][k], abs=2e-4), (m, k)
    alld = [d for r in recs.values() for d in r["dets"]]
    assert rep["mechanism"][split]["col_mod8_in_34"] == sum(d[1] % 8 in (3, 4) for d in alld)


@needs_outputs
def test_argmax_decode_is_the_stored_pixel():
    with open(DETS, encoding="utf-8") as f:
        d = json.load(f)["panos"]["manual_gold"]
    det = next(r["dets"][0] for r in d.values() if r["dets"])
    assert sd.decode(det, "argmax", False) == (det[1] / 1024, det[0] / 512)
    # the refined position stays within half a coarse cell (4 px) of the coarse centre
    x, y = sd.decode(det, "gaussian", False)
    assert abs(x * 1024 - (8 * det[4] + 3.5)) <= 4 + 1e-9
    assert abs(y * 512 - (8 * det[3] + 3.5)) <= 4 + 1e-9


def test_border_band():
    assert sd.in_border_band(0, 500) and sd.in_border_band(300, 1015)
    assert not sd.in_border_band(300, 500)


def test_great_circle_wraps_the_seam():
    assert sd.great_circle_deg(np.array([0.999]), np.array([0.5]),
                               np.array([0.001]), np.array([0.5]))[0] == pytest.approx(0.72)
