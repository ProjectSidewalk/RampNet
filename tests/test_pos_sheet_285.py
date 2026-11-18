"""Tests for pos_sheet.py (#285, A4 follow-up 2): the seeded draw, the blind cell order, the
pre-registered verdict rule and the training-disc annotation.

Pure logic on synthetic inputs; no imagery or roots on disk.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274"))

import pos_sheet as ps  # noqa: E402


def _inputs():
    auto = [{"label_uid": f"osm:c{i % 3}:{i}@p{i}", "city": f"c{i % 3}", "pano_id": f"p{i}",
             "x": "0.5", "y": "0.55"} for i in range(60)]
    v4 = [{"uid": f"h{i}@q{i}", "source": src, "city": "s", "pano_id": f"q{i}", "x": 0.5, "y": 0.55}
          for i, src in enumerate(["crowd"] * 20 + ["ps_crowd_jon_confirmed"] * 5 + ["sdot_b2"] * 15)]
    return auto, v4


def test_draw_is_stratified_seeded_and_input_order_free():
    auto, v4 = _inputs()
    rows = ps.draw(auto, v4, per_city=5, n_human=8, n_sdot=4)
    assert sum(r["stratum"] == "auto" for r in rows) == 15
    assert {r["source"] for r in rows if r["stratum"] == "human"} <= set(ps.HUMAN_SOURCES)
    assert all(r["source"] == "sdot_b2" for r in rows if r["stratum"] == "sdot_b2")
    again = ps.draw(list(reversed(auto)), list(reversed(v4)), per_city=5, n_human=8, n_sdot=4)
    assert [(r["cell"], r["uid"]) for r in rows] == [(r["cell"], r["uid"]) for r in again]


def test_blind_order_mixes_strata():
    auto, v4 = _inputs()
    rows = ps.draw(auto, v4, per_city=5, n_human=8, n_sdot=4)
    first = [r["stratum"] for r in rows[:12]]
    assert len(set(first)) > 1                       # not grouped by stratum


def _rows_tags(n_none_auto, n_none_human):
    rows = ([{"cell": i, "stratum": "auto", "city": "c1"} for i in range(1, 31)]
            + [{"cell": i, "stratum": "auto", "city": "c2"} for i in range(31, 61)]
            + [{"cell": i, "stratum": "human", "city": "h"} for i in range(61, 81)]
            + [{"cell": i, "stratum": "sdot_b2", "city": "s"} for i in range(81, 91)])
    tags = {str(r["cell"]): {"paint": "clear", "point": "on_paint"} for r in rows}
    for c in list(range(1, 1 + n_none_auto // 2)) + list(range(31, 31 + n_none_auto - n_none_auto // 2)):
        tags[str(c)] = {"paint": "none", "point": "off"}
    for c in range(61, 61 + n_none_human):
        tags[str(c)] = {"paint": "none", "point": "off"}
    return rows, tags


@pytest.mark.parametrize("n_auto, n_human, verdict", [
    (0, 0, "not_supported"),        # nothing without paint
    (12, 0, "supported"),           # 20% of auto, none of human
    (12, 4, "not_supported"),       # 20% of auto, but human is just as high: not specific to auto
    (5, 0, "inconclusive"),         # 8%: under the bar but the interval reaches it
])
def test_preregistered_verdict(n_auto, n_human, verdict):
    rows, tags = _rows_tags(n_auto, n_human)
    assert ps.tally(rows, tags)["verdict"] == verdict


def test_cant_tell_leaves_the_denominator():
    rows, tags = _rows_tags(0, 0)
    tags["1"] = {"paint": "cant_tell", "point": "na"}
    b = ps.tally(rows, tags)["strata"]["auto"]
    assert b["paint"]["cant_tell"] == 1 and b["no_paint_share"] == 0.0
    assert b["no_paint_or_cant_tell_share"] == pytest.approx(1 / 60)


def test_tight_crop_holds_the_training_disc():
    # the 3-sigma disc (30 heatmap px of 1024) must fit inside the tight crop's height
    assert 2 * ps.DISC_W < ps.TIGHT_W * ps.TIGHT_ASPECT


def test_render_marks_the_point_across_the_seam():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (4096, 2048), (90, 90, 90))
    tight, ctx = ps.render(img, 0.999, 0.56)
    px = tight.load()
    magenta = sum(1 for x in range(tight.size[0]) for y in range(tight.size[1])
                  if px[x, y][0] > 200 and px[x, y][2] > 200 and px[x, y][1] < 80)
    assert magenta > 20                                # the cross and the 1-sigma circle were drawn
    cw = int(ps.TIGHT_W * 4096)
    top = int(0.56 * 2048) - int(cw * ps.TIGHT_ASPECT) // 2
    s = tight.size[0] / cw
    cx, cy = int(((0.999 * 4096 - (int(0.999 * 4096) - cw // 2)) % 4096) * s), int((0.56 * 2048 - top) * s)
    assert px[cx, cy][:3] == (90, 90, 90)              # the gap keeps the pixel under the point clean


def test_committed_summary_reproduces_from_committed_tags():
    """summary.json is rederived from sample.csv + the committed tags file (no imagery needed)."""
    import json
    out = ps.OUT
    paths = [os.path.join(out, f) for f in ("sample.csv", "summary.json")]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("pos_sheet outputs not committed")
    saved = json.load(open(paths[1], encoding="utf-8"))
    tags = json.load(open(os.path.join(out, saved["tags_file"]), encoding="utf-8"))
    assert tags["not_ground_truth"] is True
    rows = ps.read_csv(paths[0])
    for r in rows:
        r["cell"] = int(r["cell"])
    res = ps.tally(rows, tags["tags"])
    assert res["verdict"] == saved["verdict"] and res["point_flag"] == saved["point_flag"]
    assert res["strata"] == saved["strata"] and res["auto_cities"] == saved["auto_cities"]
    assert ps.digest([r["uid"] for r in rows]) == saved["sample_digest"] == tags["sample_digest"]
