"""Unit tests for the false-positive gallery (#46).

Pure logic — no ``.model_cache``, no imagery.

The gallery's job is to show a reviewer the *worst* cases and to be honest about the
ones it left out. So the guarantees are about ordering and about sampling being
visible: a silent truncation would read as "these are the isolated false positives"
when it is showing 12 of 41,668.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import fp_gallery as fg  # noqa: E402


def _row(conf=None, city="bend", pano="p", x=0.5, y=0.6):
    return {"city": city, "pano": pano, "x": x, "y": y, "confidence": conf,
            "bucket": "isolated"}


# --------------------------------------------------------------------------- #
# rank_key — worst first, deterministically
# --------------------------------------------------------------------------- #
def test_higher_confidence_sorts_first():
    rows = sorted([_row(0.2), _row(0.9), _row(0.5)], key=fg.rank_key)
    assert [r["confidence"] for r in rows] == [0.9, 0.5, 0.2]


def test_scoreless_predictions_sort_after_scored_ones():
    # A chat VLM emits no score; it must not displace a scored model's genuinely
    # worst cases when several models are sampled together.
    rows = sorted([_row(None), _row(0.1)], key=fg.rank_key)
    assert rows[0]["confidence"] == 0.1 and rows[1]["confidence"] is None


def test_ties_break_stably_without_a_seed():
    a = _row(0.5, pano="a")
    b = _row(0.5, pano="b")
    assert sorted([b, a], key=fg.rank_key) == [a, b]
    assert sorted([a, b], key=fg.rank_key) == [a, b]


def test_a_zero_confidence_is_not_treated_as_missing():
    rows = sorted([_row(None), _row(0.0)], key=fg.rank_key)
    assert rows[0]["confidence"] == 0.0


# --------------------------------------------------------------------------- #
# to_gallery_items — imagery facts, and honest sampling
# --------------------------------------------------------------------------- #
def _patch_width(monkeypatch, width):
    monkeypatch.setattr(fg.mg, "pano_width", lambda c, p, root=None: width)


def test_items_gain_the_fields_the_renderer_needs(monkeypatch):
    _patch_width(monkeypatch, 16384)
    it = fg.to_gallery_items([_row(0.9)], ".")[0]
    for k in ("dist", "px", "source_width", "source_px", "parity", "judgeable"):
        assert k in it, k


def test_size_comes_from_the_points_own_geometry(monkeypatch):
    # A false positive has no true size; the proxy is the size a ramp WOULD have at
    # that distance, which is what governs whether a reviewer can tell what the model
    # latched onto.
    from stage1_label_recall import geom
    _patch_width(monkeypatch, 4096)
    it = fg.to_gallery_items([_row(0.9, y=0.62)], ".")[0]
    assert abs(it["px"] - geom(0.62)[1]) < 1e-9
    assert abs(it["dist"] - geom(0.62)[0]) < 1e-9


def test_source_pixels_scale_with_the_stored_pano(monkeypatch):
    _patch_width(monkeypatch, 4096)
    small = fg.to_gallery_items([_row(0.9)], ".")[0]["source_px"]
    _patch_width(monkeypatch, 16384)
    big = fg.to_gallery_items([_row(0.9)], ".")[0]["source_px"]
    assert abs(big - 4 * small) < 1e-6


def test_sampling_takes_the_worst_n_not_an_arbitrary_n(monkeypatch):
    _patch_width(monkeypatch, 8192)
    rows = [_row(c) for c in (0.1, 0.95, 0.4, 0.99, 0.6)]
    got = fg.to_gallery_items(rows, ".", sample=2)
    assert [r["confidence"] for r in got] == [0.99, 0.95]


def test_no_sample_keeps_everything(monkeypatch):
    _patch_width(monkeypatch, 8192)
    rows = [_row(c) for c in (0.1, 0.2, 0.3)]
    assert len(fg.to_gallery_items(rows, ".", sample=None)) == 3


def test_a_pano_missing_from_disk_is_dropped_not_faked(monkeypatch):
    # Rendering a crop for a pano we do not have would either crash or invent one;
    # dropping it keeps the sample honest.
    monkeypatch.setattr(fg.mg, "pano_width", lambda c, p, root=None: None)
    assert fg.to_gallery_items([_row(0.9)], ".") == []


def test_sampling_counts_only_renderable_items(monkeypatch):
    # A missing pano must not consume a slot in the worst-N budget.
    widths = {"a": None, "b": 8192, "c": 8192}
    monkeypatch.setattr(fg.mg, "pano_width", lambda c, p, root=None: widths[p])
    rows = [_row(0.9, pano="a"), _row(0.8, pano="b"), _row(0.7, pano="c")]
    got = fg.to_gallery_items(rows, ".", sample=2)
    assert [r["pano"] for r in got] == ["b", "c"]


# --------------------------------------------------------------------------- #
# bookkeeping — one instrument, shared with the miss gallery
# --------------------------------------------------------------------------- #
def test_the_fp_gallery_reuses_the_miss_gallery_instrument():
    import miss_gallery as mg
    assert fg.mg is mg
    # Parity, the pixel floor and the view geometry must not fork between the two
    # galleries, or FP and miss verdicts stop being comparable.
    assert fg.mg.JUDGEABLE_SOURCE_PX == mg.JUDGEABLE_SOURCE_PX
    assert fg.mg.MODEL_WIDTH == mg.MODEL_WIDTH


def test_the_buckets_offered_are_the_fp_taxonomy_s_own():
    import fp_taxonomy as fx
    assert fg.fx.BUCKETS is fx.BUCKETS
    assert "isolated" in fx.BUCKETS
