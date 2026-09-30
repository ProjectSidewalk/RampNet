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


def test_quantization_variance_models():
    """The argmax model (review S2): E[(u - 0.5 sign u)^2] for u ~ U[-4, 4], by quadrature."""
    u = np.linspace(-4, 4, 800001)
    assert np.mean((u - 0.5 * np.sign(u)) ** 2) == pytest.approx(sd.QVAR_ARGMAX, abs=1e-4)
    assert np.mean(u ** 2) == pytest.approx(sd.QVAR_CENTRE, abs=1e-4)
    m = {"argmax": {"sd_x_px": 2.0, "sd_y_px": 1.0}, "centre": {"sd_x_px": 2.5, "sd_y_px": 1.5},
         "gaussian": {"sd_x_px": 1.0, "sd_y_px": 0.5}}
    q = sd.quantization_variance(m)
    assert q["x"]["removed_vs_argmax_px2"] == pytest.approx(3.0)
    assert q["x"]["removed_frac_of_argmax_model"] == pytest.approx(3.0 / sd.QVAR_ARGMAX)
    assert q["y"]["removed_vs_centre_px2"] == pytest.approx(2.25 - 0.25)


@needs_outputs
def test_review_counts_rederive():
    """S1 (non-zero fp32 reconstruction), S4 (climb census), N3 (198 boxed panos)."""
    with open(DETS, encoding="utf-8") as f:
        D = json.load(f)["panos"]
    for split, recs in D.items():
        vals = [r["recon_max_abs"] for r in recs.values()]
        assert 0 < min(vals) and max(vals) < 1e-6, split
        dets = [d for r in recs.values() for d in r["dets"]]
        assert not any(d[5] > 0 and d[2] > 1 for d in dets), split
        assert all(d[1] % 8 in (3, 4) and d[0] % 8 in (3, 4) for d in dets if d[5] > 0)
    boxed = 0
    for split in sd.BOX_SPLITS:
        pts = sd.ground_truth(split)
        boxed += sum(any(s) for _, s in pts.values())
    assert boxed == 198


@needs_outputs
def test_y_profile_bands_partition_the_pairs():
    with open(DETS, encoding="utf-8") as f:
        recs = json.load(f)["panos"]["manual_gold"]
    rows = sd.build_pairs(recs, sd.ground_truth("manual_gold"), sd.radius_sq_for())
    yp = sd.y_profile(rows, sd.Y_BANDS, False)
    assert sum(b["pairs"] for b in yp["bands"]) == len(rows)
    assert sum(c["pairs"] for c in yp["calibration"]) == len(rows)
    with open(RESULTS, encoding="utf-8") as f:
        got = json.load(f)["splits"]["manual_gold"]["y_profile"]
    assert [b["pairs"] for b in got["bands"]] == [b["pairs"] for b in yp["bands"]]


def test_verify_imagery_flags_changed_and_missing(tmp_path, monkeypatch):
    import imagery_manifest as im
    pids = sd.split_pano_ids("annapolis")[:3]
    pdir = tmp_path / "benchmark" / "annapolis" / "panos"
    pdir.mkdir(parents=True)
    for k, pid in enumerate(pids[:2]):
        (pdir / f"{pid}.jpg").write_bytes(b"pano %d" % k)
    good = {pids[0]: {"sha256": sd.sha256_file(str(pdir / f"{pids[0]}.jpg"))},
            pids[1]: {"sha256": "0" * 64}, pids[2]: {"sha256": "1" * 64}}
    monkeypatch.setattr(im, "load", lambda city: {"panos": good, "digest": "d", "n": 3})
    r = sd.verify_imagery(str(tmp_path), ["annapolis"], limit=3)
    s = r["splits"]["annapolis"]
    assert r["status"] == "MISMATCH"
    assert (s["match"], s["changed"], s["missing"]) == (1, [pids[1]], [pids[2]])
    r = sd.verify_imagery(str(tmp_path), ["annapolis"], limit=1)
    assert r["status"] == "ok"


def test_border_band():
    assert sd.in_border_band(0, 500) and sd.in_border_band(300, 1015)
    assert not sd.in_border_band(300, 500)


def test_great_circle_wraps_the_seam():
    assert sd.great_circle_deg(np.array([0.999]), np.array([0.5]),
                               np.array([0.001]), np.array([0.5]))[0] == pytest.approx(0.72)
