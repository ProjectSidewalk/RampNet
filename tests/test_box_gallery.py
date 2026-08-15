"""Guards for the box annotator's Python contracts (scripts/box_gallery.py, issue #116).

What matters here: which adjudicated ramps become annotation items (mirrors
rampnet.validation.collect's semantics), the seam-wrapping crop geometry the viewer's
box math is anchored to, and the prefill reconciliation that keeps a revised
verdicts.json from silently re-attaching a box to a different ramp.

    pytest tests/test_box_gallery.py -v
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from box_gallery import (  # noqa: E402
    BOX_RULE, BOX_RULE_VERSION, build_html, crop_rect, crop_side, cut_crop,
    entry_meta, enumerate_items, items_from_manual_labels, reconcile_initial,
    render_pano_items, resolution_note, STATE_BOOTSTRAP_JS)


def _record(pid, dets, width=4096, height=2048):
    return {"pano": {"panorama_id": pid, "width": width, "height": height,
                     "source": "gsv", "capture_date": "2025-01"},
            "detections": [{"x_normalized": x, "y_normalized": y, "confidence": c}
                           for x, y, c in dets]}


# --- Item enumeration ----------------------------------------------------------------

def test_enumerate_items_mirrors_collect_semantics():
    records = {"P1": _record("P1", [(0.1, 0.5, 0.9), (0.2, 0.5, 0.8),
                                    (0.3, 0.5, 0.7), (0.4, 0.5, 0.6)])}
    verdicts = {"P1": {"dets": [True, False, "duplicate", "unsure"],
                       "missed": [{"x": 0.6, "y": 0.5},
                                  {"x": 0.7, "y": 0.5, "unsure": True}],
                       "no_missed": False}}
    items, warnings = enumerate_items(verdicts, records)
    assert warnings == []
    # Only the True det and the sure missed mark are ramps; False isn't one,
    # 'duplicate' is a ramp another item already covers, 'unsure' abstains.
    assert [(it["key"], it["x"]) for it in items] == [("det:0", 0.1), ("missed:0", 0.6)]
    assert items[0]["conf"] == 0.9 and items[1]["conf"] is None
    assert [it["seq"] for it in items] == [0, 1]


def test_enumerate_items_skips_mismatched_pano_with_warning():
    records = {"P1": _record("P1", [(0.1, 0.5, 0.9)])}
    verdicts = {"P1": {"dets": [True, True], "missed": [], "no_missed": True}}
    items, warnings = enumerate_items(verdicts, records)
    assert items == []
    assert len(warnings) == 1 and "P1" in warnings[0]


def test_enumerate_items_skips_partially_judged_pano():
    # collect() drops a pano with any None verdict ("unusable for either metric"), so
    # this must too: its missed marks in particular come from a pano nobody finished
    # scanning, and boxing ramps collect() never counts splits the two populations.
    records = {"P1": _record("P1", [(0.1, 0.5, 0.9), (0.2, 0.5, 0.8)])}
    verdicts = {"P1": {"dets": [True, None], "missed": [{"x": 0.6, "y": 0.5}],
                       "no_missed": False}}
    items, warnings = enumerate_items(verdicts, records)
    assert items == []
    assert len(warnings) == 1 and "partially judged" in warnings[0]


def test_items_from_manual_labels_uses_centers_as_prompts(tmp_path):
    (tmp_path / "PANO.txt").write_text(
        "0 0.25 0.5 0.003 0.002\n\n0 0.75 0.6 0.004 0.001\n", encoding="utf-8")
    items = items_from_manual_labels(tmp_path, ["PANO", "ABSENT"])
    assert [(it["key"], it["x"], it["y"]) for it in items] == [
        ("gold:0", 0.25, 0.5), ("gold:1", 0.75, 0.6)]


def test_items_from_manual_labels_raises_on_malformed_line(tmp_path):
    (tmp_path / "BAD.txt").write_text("0 0.25 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BAD.txt:1"):
        items_from_manual_labels(tmp_path, ["BAD"])


# --- Crop geometry -------------------------------------------------------------------

def test_crop_side_is_fov_fraction_capped_at_height():
    assert crop_side(4096, 2048, 90) == 1024
    assert crop_side(4096, 2048, 360) == 2048  # cap: a 2:1 pano's height


def test_crop_rect_wraps_x_and_clamps_y():
    # Point at the seam: the crop's left edge sits in the right half of the pano.
    left, top = crop_rect(0.01, 0.5, 4096, 2048, 1024)
    assert left == (int(round(0.01 * 4096 - 512))) % 4096 == 3625
    # y clamps by shifting, never wraps: near the zenith the crop pins to the top...
    assert crop_rect(0.5, 0.0, 4096, 2048, 1024)[1] == 0
    # ...and near the nadir to the bottom.
    assert crop_rect(0.5, 1.0, 4096, 2048, 1024)[1] == 2048 - 1024


def test_cut_crop_stitches_across_the_seam():
    # 8x4 image whose pixel value encodes its column: wrap must be column-exact.
    img = Image.new("RGB", (8, 4))
    img.putdata([(x * 30, 0, 0) for _ in range(4) for x in range(8)])
    crop = cut_crop(img, 6, 0, 4)  # columns 6, 7, 0, 1
    assert [crop.getpixel((i, 0))[0] for i in range(4)] == [180, 210, 0, 30]
    # The non-wrapping path is a plain crop.
    crop = cut_crop(img, 2, 0, 4)
    assert [crop.getpixel((i, 0))[0] for i in range(4)] == [60, 90, 120, 150]


def test_resolution_note_flags_a_model_resolution_bundle():
    # benchmark/manual_gold is 4096x2048 for all 1000 panos: 1024 px crops, and "tight
    # at native zoom" is a weaker instrument there than on a 12288-wide native archive.
    recs = {"P1": _record("P1", [], width=4096, height=2048)}
    crop_px, msg = resolution_note(recs, ["P1"], 90)
    assert crop_px == {"4096x2048": 1024}
    assert "MODEL-RESOLUTION" in msg

    recs = {"P1": _record("P1", [], width=12288, height=6144)}
    crop_px, msg = resolution_note(recs, ["P1"], 90)
    assert crop_px == {"12288x6144": 3072}
    assert "MODEL-RESOLUTION" not in msg


def test_entry_meta_carries_the_geometry_the_viewer_needs():
    rec = _record("P1", [])
    it = {"pid": "P1", "key": "det:3", "seq": 0, "x": 0.5, "y": 0.5, "conf": 0.61}
    meta = entry_meta(it, 4096, 2048, 1024, 1536, 512, rec)
    assert meta["img"] == "P1_det_3.jpg"          # ':' never reaches a filename
    assert (meta["pw"], meta["ph"], meta["cl"], meta["ct"], meta["cs"]) == \
        (4096, 2048, 1536, 512, 1024)


# --- Rendering -----------------------------------------------------------------------

def _pano_on_disk(tmp_path, pid, size):
    panos = tmp_path / "panos"
    panos.mkdir(exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(panos / f"{pid}.jpg")
    return panos


def test_render_pano_items_cuts_crops_from_record_geometry(tmp_path):
    panos = _pano_on_disk(tmp_path, "P1", (256, 128))
    images = tmp_path / "images"
    images.mkdir()
    rec = _record("P1", [(0.5, 0.5, 0.9)], width=256, height=128)
    items = [{"pid": "P1", "key": "det:0", "seq": 0, "x": 0.5, "y": 0.5, "conf": 0.9}]
    entries = render_pano_items("P1", items, rec, panos, images, 90, 0)
    assert len(entries) == 1
    side = crop_side(256, 128, 90)
    assert entries[0]["cs"] == side == 64
    with Image.open(images / entries[0]["img"]) as crop:
        assert crop.size == (side, side)


def test_render_pano_items_refuses_non_native_image(tmp_path):
    # The archive is verified 1:1 native; a dimension mismatch is wrong pixels, and the
    # pano must be skipped loudly rather than silently annotated at the wrong scale.
    panos = _pano_on_disk(tmp_path, "P1", (128, 64))
    images = tmp_path / "images"
    images.mkdir()
    rec = _record("P1", [], width=256, height=128)
    items = [{"pid": "P1", "key": "missed:0", "seq": 0, "x": 0.5, "y": 0.5, "conf": None}]
    with pytest.raises(ValueError, match="native archive expected"):
        render_pano_items("P1", items, rec, panos, images, 90, 0)


# --- Prefill reconciliation ----------------------------------------------------------

ITEMS = [{"pid": "P1", "key": "missed:0", "seq": 0, "x": 0.6, "y": 0.5, "conf": None}]


def test_reconcile_drops_boxes_whose_point_moved():
    # A re-reviewed verdicts.json renumbered the missed marks: the stored point no
    # longer matches the item behind the key, so the box must not re-attach.
    initial = {"P1": {"missed:0": {"point": {"x": 0.3, "y": 0.5}, "status": "boxed",
                                   "cx": 0.3, "cy": 0.5, "w": 0.01, "h": 0.01}}}
    clean, stale = reconcile_initial(initial, ITEMS)
    assert clean == {} and stale == ["P1 missed:0"]


def test_reconcile_keeps_matching_and_unrendered_entries():
    match = {"point": {"x": 0.6, "y": 0.5}, "status": "cant"}
    other = {"point": {"x": 0.1, "y": 0.1}, "status": "boxed",
             "cx": 0.1, "cy": 0.1, "w": 0.02, "h": 0.02}
    clean, stale = reconcile_initial(
        {"P1": {"missed:0": match}, "P2": {"det:0": other}}, ITEMS)
    assert stale == []
    assert clean["P1"]["missed:0"] == match
    assert clean["P2"]["det:0"] == other  # not in this session: round-trips verbatim


# --- Viewer state bootstrap (run under node: the one JS path that can destroy work) ---

def _run_bootstrap(tmp_path, initial, local, entries):
    """Call the viewer's bootstrapState under node and return its result."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = tmp_path / "boot.js"
    script.write_text(
        STATE_BOOTSTRAP_JS
        + "\nconst out = bootstrapState(%s, %s, %s);\n"
          "console.log(JSON.stringify(out));\n"
          % (json.dumps(initial), json.dumps(local), json.dumps(entries)),
        encoding="utf-8")
    proc = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_state_bootstrap_merges_prefill_per_key_not_per_pano(tmp_path):
    # The regression: local state holding ONE key for a pano used to suppress the whole
    # pano's prefill, and Export then wrote boxes.json without the suppressed boxes.
    initial = {"P1": {"det:0": {"point": {"x": 0.1, "y": 0.5}, "status": "boxed",
                                "cx": 0.1, "cy": 0.5, "w": 0.02, "h": 0.02},
                      "det:1": {"point": {"x": 0.3, "y": 0.5}, "status": "cant"}}}
    local = {"P1": {"det:0": {"status": "boxed", "px": 0.1, "py": 0.5,
                              "cx": 0.11, "cy": 0.5, "w": 0.03, "h": 0.03}}}
    entries = [{"pid": "P1", "key": "det:0", "x": 0.1, "y": 0.5},
               {"pid": "P1", "key": "det:1", "x": 0.3, "y": 0.5}]
    out = _run_bootstrap(tmp_path, initial, local, entries)
    assert set(out["state"]["P1"]) == {"det:0", "det:1"}   # prefill survives
    assert out["state"]["P1"]["det:0"]["w"] == 0.03        # local edit wins on its key
    assert out["state"]["P1"]["det:1"]["status"] == "cant"
    assert out["state"]["P1"]["det:1"]["px"] == 0.3        # point -> px/py on merge
    assert "point" not in out["state"]["P1"]["det:1"]


def test_state_bootstrap_drops_stale_local_annotations(tmp_path):
    # A re-reviewed verdicts.json renumbered missed:0 onto a different ramp. The old
    # guard only filtered the prefill, so a box already in localStorage re-attached.
    local = {"P1": {"missed:0": {"status": "boxed", "px": 0.6, "py": 0.5,
                                 "cx": 0.6, "cy": 0.5, "w": 0.02, "h": 0.02}}}
    entries = [{"pid": "P1", "key": "missed:0", "x": 0.2, "y": 0.5}]
    out = _run_bootstrap(tmp_path, {}, local, entries)
    assert out["state"]["P1"] == {} and out["staleDropped"] == 1


def test_state_bootstrap_adopts_annotations_predating_the_guard(tmp_path):
    local = {"P1": {"det:0": {"status": "boxed", "cx": 0.5, "cy": 0.5,
                              "w": 0.02, "h": 0.02}}}
    entries = [{"pid": "P1", "key": "det:0", "x": 0.5, "y": 0.5}]
    out = _run_bootstrap(tmp_path, {}, local, entries)
    assert out["adopted"] == 1 and out["staleDropped"] == 0
    assert out["state"]["P1"]["det:0"]["px"] == 0.5


# --- Viewer HTML ---------------------------------------------------------------------

def test_build_html_embeds_rule_entries_and_prefill():
    entries = [entry_meta({"pid": "P1", "key": "det:0", "seq": 0, "x": 0.5, "y": 0.5,
                           "conf": 0.9}, 4096, 2048, 1024, 1536, 512, _record("P1", []))]
    initial = {"P1": {"det:0": {"point": {"x": 0.5, "y": 0.5}, "status": "boxed",
                                "cx": 0.5, "cy": 0.51, "w": 0.02, "h": 0.03}}}
    html = build_html(entries, initial, {"name": "jonf"}, "richmond", "richmond",
                      90, "benchmark/richmond/boxes.json", {"4096x2048": 1024})
    assert json.dumps({"version": BOX_RULE_VERSION, "text": BOX_RULE}) in html
    assert json.dumps(entries) in html
    assert json.dumps(initial) in html
    assert '{"4096x2048": 1024}' in html   # the resolution the gold was drawn at
    assert "function bootstrapState" in html
    assert "boxannotator:" in html  # annotator block persists per bundle
    for placeholder in ("__ENTRIES__", "__BOX_RULE__", "__CROP_PX__",
                        "__STATE_BOOTSTRAP__"):
        assert placeholder not in html
