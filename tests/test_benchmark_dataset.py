"""Guards for the #21 benchmark dataset builder (scripts/build_benchmark_dataset.py).

Covers the parts that must not drift without touching the heavy image packing:
the verdict encode/decode round-trip (the mixed bool / "unsure" / "duplicate"
values must survive the parquet string column) and the training-overlap drop.
Uses the committed image-free bundles (records.jsonl + verdicts.json).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from build_benchmark_dataset import (  # noqa: E402
    LEAKED_BEND_IDS, build_rows, encode_verdict,
)


def _decode(v):
    return {"true": True, "false": False}.get(v, v)


def test_verdict_encode_decode_roundtrip():
    for original in (True, False, "unsure", "duplicate"):
        assert _decode(encode_verdict(original)) == original


def test_encoding_is_homogeneous_strings():
    # Parquet list column can't mix bool and str; every encoded value is a str.
    for original in (True, False, "unsure", "duplicate"):
        assert isinstance(encode_verdict(original), str)


def test_bend_flags_overlap_but_drops_nothing():
    rows, flagged = build_rows("bend", read_images=False)
    assert set(flagged) == LEAKED_BEND_IDS
    # Nothing dropped: the overlapping panos are kept, just flagged.
    flagged_rows = {r["pano_id"] for r in rows if r["train_overlap"]}
    assert flagged_rows == LEAKED_BEND_IDS
    assert {r["pano_id"] for r in rows} >= LEAKED_BEND_IDS
    assert all(r["imagery_source"] == "gsv" for r in rows)


def test_richmond_has_no_overlap_and_is_mapillary():
    rows, flagged = build_rows("richmond", read_images=False)
    assert flagged == []
    assert all(r["train_overlap"] is False for r in rows)
    assert all(r["imagery_source"] == "mapillary" for r in rows)


def test_det_verdicts_align_with_detections():
    for city in ("bend", "richmond"):
        rows, _ = build_rows(city, read_images=False)
        for r in rows:
            assert len(r["det_verdicts"]) == len(r["detections"])
