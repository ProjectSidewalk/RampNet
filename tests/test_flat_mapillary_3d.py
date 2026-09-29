"""#214: the flat-Mapillary 3D arms re-derive from committed files (CPU, no network)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis", "flat3d"))

import crossview_align_48 as H  # noqa: E402
import flat_mapillary_48 as F  # noqa: E402

ARMS = ["flat_sfm", "flat_gs", "flat_gsmed", "flat_mvs", "noflat_sfm", "noflat_gs",
        "noflat_gsmed", "noflat_mvs", "mlypano_sfm", "mlypano_gs", "mlypano_gsmed",
        "mlypano_mvs"]

pytestmark = pytest.mark.skipif(not os.path.exists(F.RESULTS),
                                reason="flat_mapillary_3d outputs not present")


@pytest.mark.parametrize("arm", ARMS)
def test_committed_predictions_rederive_from_corner_json(arm):
    registry = H.load_arms()
    pairs = H.read_frozen_pairs()
    rows, _ = H.run_arm(registry[arm], pairs, H.Context(None, pairs))
    got = {r["pair_id"]: r for r in rows}
    want, _ = F.read_preds_any(arm)
    assert set(got) == set(want)
    for pid, w in want.items():
        g = got[pid]
        if w["x"] is None:
            assert g["x"] is None, pid
        else:
            assert g["x"] == pytest.approx(w["x"], abs=1e-5), pid
            assert g["y"] == pytest.approx(w["y"], abs=1e-5), pid


def test_gsv_pairs_always_fall_back():
    registry = H.load_arms()
    pairs = [p for p in H.read_frozen_pairs() if p["city"] != "richmond"][:5]
    rows, _ = H.run_arm(registry["flat_sfm"], pairs, H.Context(None, pairs))
    assert all(r["x"] is None and r["reason"] == "not_mapillary" for r in rows)


def test_richmond_results_rederive():
    res = json.load(open(F.RESULTS, encoding="utf-8"))
    pairs = [p for p in H.read_frozen_pairs() if p["city"] == "richmond"]
    for arm in ARMS:
        preds, _ = F.read_preds_any(arm)
        errs = H.arm_errors(pairs, preds)
        assert round(float(np.median([e[0] for e in errs])), 4) == \
            pytest.approx(res["arms"][arm]["all"]["median_deg"], abs=1e-4), arm


def test_median_depth_of_a_single_opaque_gaussian():
    import gs_median_depth as G
    mu = np.array([[0.0, 0.0, 5.0]])
    s = np.array([[0.05, 0.05, 0.05]])
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    op = np.array([0.99])
    t, n = G.median_depth(np.zeros(3), np.array([0.0, 0.0, 1.0]), mu, s, q, op)
    assert n == 1 and t == pytest.approx(5.0, abs=1e-6)
