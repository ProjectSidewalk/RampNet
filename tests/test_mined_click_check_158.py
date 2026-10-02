"""mined_click_check_158 (#158 step 4, click pass): click-to-pano geometry, the Yes-card
selection from the committed pass-1 / pass-2 files, the click-file checks and the offset
read. Reads only committed files."""
import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import mined_click_check_158 as C  # noqa: E402
import mined_label_check_158 as M  # noqa: E402
import multiview_evidence_48 as mv  # noqa: E402

A = os.path.join(M.OUT, "")
P1 = os.path.join(M.OUT, "mined_label_check__jonf.json")
P2 = os.path.join(M.OUT, "mined_label_check__jonf-p2.json")


def _ring_frac(x, y):
    return 0.5, (y * 180.0 - C.window_top_frac(y) * 180.0) / mv.CROP_FOV_V


@pytest.mark.parametrize("x,y", [(0.3, 0.55), (0.01, 0.5), (0.99, 0.6), (0.5, 0.02), (0.5, 0.98)])
def test_a_click_on_the_ring_is_the_miners_point(x, y):
    fx, fy = _ring_frac(x, y)
    px, py = C.click_to_pano(x, y, fx, fy)
    assert C.offset_px(x, y, px, py)[2] == pytest.approx(0, abs=1e-9)


def test_click_offsets_are_in_heatmap_pixels_and_wrap_at_the_seam():
    # A click at the crop's right edge is half the window (18 degrees) to the right.
    px, _ = C.click_to_pano(0.99, 0.5, 1.0, 0.5)
    dx, dy, d = C.offset_px(0.99, 0.5, px, 0.5)
    assert px < 0.1                                    # wrapped past the seam
    assert dx == pytest.approx(18 / 360 * C.HEATMAP_W) and dy == 0
    # Vertically, the full 24-degree window spans 24/180 of the 512 rows.
    _, top = C.click_to_pano(0.5, 0.5, 0.5, 0.0)
    _, bot = C.click_to_pano(0.5, 0.5, 0.5, 1.0)
    assert (bot - top) * C.HEATMAP_H == pytest.approx(24 / 180 * C.HEATMAP_H)
    assert C.EVAL_RADIUS_PX == pytest.approx(22.528)


def test_yes_cards_are_the_combined_sample_yeses():
    ids, ref, p1, p2 = C.yes_cards(P1, P2)
    assert len(ids) == 71
    assert not any(ref["cards"][u]["instrument"] for u in ids)
    assert set(ids) >= {u for u in p2["items"]
                        if (p2["verdicts"].get(u) or {}).get("answer") == "yes"
                        and not ref["cards"][u]["instrument"]}
    plan = json.load(open(C.ITEMS_PATH, encoding="utf-8"))
    assert plan["cards"] == ids and all(it["ring"] is False for it in plan["items"])
    assert plan["from"]["pass2_sha256"] == M.R.sha256_file(P2)


def _ref_and_empty():
    ref = C.committed_reference()
    d = json.load(open(C.verdicts_path("jonf"), encoding="utf-8"))
    return ref, d


def test_committed_click_gallery_re_derives_and_the_empty_file_loads(tmp_path):
    ref, d = _ref_and_empty()
    assert d["n_answered"] == 0 and d["items"] == ref["items"]
    assert C.load_verdicts(C.verdicts_path("jonf"), ref)["rater"] == "jonf"


def _write(tmp_path, d):
    p = tmp_path / (C.EXPORT_PREFIX + d["rater"] + C.EXPORT_SUFFIX)
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_click_file_checks(tmp_path):
    ref, d = _ref_and_empty()
    u = ref["items"][0]
    bad = copy.deepcopy(d)
    bad["verdicts"] = {u: {"status": "placed", "click": None}}
    with pytest.raises(ValueError, match="status"):
        C.load_verdicts(_write(tmp_path, bad), ref)
    bad["verdicts"] = {u: {"status": "multi", "click": {"fx": 0.5, "fy": 0.5}}}
    with pytest.raises(ValueError, match="status"):
        C.load_verdicts(_write(tmp_path, bad), ref)
    bad["verdicts"] = {u: {"status": "placed", "click": {"fx": 1.5, "fy": 0.5}}}
    with pytest.raises(ValueError, match="outside"):
        C.load_verdicts(_write(tmp_path, bad), ref)
    bad = copy.deepcopy(d)
    bad["manifest_digest"] = "0" * 16
    with pytest.raises(ValueError, match="gallery"):
        C.load_verdicts(_write(tmp_path, bad), ref)


def test_score_offsets_and_precision_at_radius(tmp_path):
    ref, d = _ref_and_empty()
    a, b, c = ref["items"][:3]
    ia = ref["items_by_uid"][a]
    fx, fy = _ring_frac(ia["x"], ia["y"])
    # a: on the miner's point; b: 30 heatmap px to the right; c: two ramps.
    ib = ref["items_by_uid"][b]
    fxb, fyb = _ring_frac(ib["x"], ib["y"])
    fxb += 30 / C.HEATMAP_W * 360 / mv.CROP_FOV_H
    d = copy.deepcopy(d)
    d["verdicts"] = {a: {"status": "placed", "click": {"fx": fx, "fy": fy}},
                     b: {"status": "placed", "click": {"fx": fxb, "fy": fyb}},
                     c: {"status": "multi", "click": None}}
    d = C.load_verdicts(_write(tmp_path, d), ref)
    s = C.score(d, ref, n_decided=83)
    assert s["status"]["placed"] == 2 and s["status"]["multi"] == 1
    dist = sorted(r["dist_px"] for r in s["offsets"])
    assert dist[0] == pytest.approx(0, abs=1e-6) and dist[1] == pytest.approx(30, abs=1e-6)
    assert s["pooled"]["within_eval_radius_22_5"]["count"] == 1
    assert s["precision_at_eval_radius_22_5"]["yes_within"] == 1
    # Agreement with itself is exact.
    ag = C.agreement(d, d, ref)
    assert ag["both_placed"] == 2 and ag["between_raters"]["max_px"] == pytest.approx(0)
