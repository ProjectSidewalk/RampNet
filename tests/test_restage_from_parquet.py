"""Tests for scripts/restage_from_parquet.py (issue #263).

CPU-only, no network, no pyarrow: the shard reader is imported lazily inside
``process_shard`` so the helpers under test load without it.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restage_from_parquet.py"


def _load():
    spec = importlib.util.spec_from_file_location("restage_from_parquet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_expected_counts_are_the_published_recipe():
    m = _load()
    assert m.EXPECTED == {"train": 150063, "val": 42875, "test": 21438}
    assert m.SPLIT_DIRS["validation"] == "val" and m.SPLIT_DIRS["val"] == "val"


def test_is_jpeg_accepts_soi_and_rejects_the_hub_placeholder():
    m = _load()
    assert m.is_jpeg(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    # The Hub's file_1..file_3 rows: a 19-byte blob beginning "dumm".
    assert not m.is_jpeg(b"dummy image bytes!!")
    assert not m.is_jpeg(None)
    assert not m.is_jpeg(b"")
    assert not m.is_jpeg(b"\xff\xd8")  # too short to be an image


def test_count_pairs_reports_strays(tmp_path):
    m = _load()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.jpg").write_bytes(b"x")  # jpg without json
    (tmp_path / "c.json").write_text("{}")  # json without jpg
    assert m.count_pairs(tmp_path) == (1, 1, 1)


def test_quarantine_manifest_is_sorted_relative_lf(tmp_path):
    m = _load()
    q = tmp_path / "_quarantine"
    for split, pid in (("val", "file_2"), ("train", "file_1")):
        (q / split).mkdir(parents=True)
        (q / split / f"{pid}.jpg").write_bytes(b"dumm")
        (q / split / f"{pid}.json").write_text(json.dumps({"pano_id": pid}))
    assert m.write_quarantine_manifest(tmp_path) == 4
    raw = (q / "MANIFEST.txt").read_bytes()
    assert b"\r" not in raw
    assert raw.decode().splitlines() == [
        "train/file_1.jpg",
        "train/file_1.json",
        "val/file_2.jpg",
        "val/file_2.json",
    ]


def test_quarantine_manifest_without_quarantine_dir(tmp_path):
    m = _load()
    assert m.write_quarantine_manifest(tmp_path) == 0
    assert not (tmp_path / "_quarantine").exists()


def test_write_atomic_leaves_no_tmp(tmp_path):
    m = _load()
    target = tmp_path / "p.jpg"
    m.write_atomic(target, b"\xff\xd8\xff")
    assert target.read_bytes() == b"\xff\xd8\xff"
    assert list(tmp_path.iterdir()) == [target]
