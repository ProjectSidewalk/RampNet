"""Tests for scripts/restage_from_parquet.py (issue #263).

CPU-only and offline. The shard-level tests build a tiny parquet with pyarrow (in
requirements-dev); the helper tests need no pyarrow.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restage_from_parquet.py"

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(64))
DUMMY = b"dummy image bytes!!"  # 19 bytes, like the Hub's file_1..file_3 rows


def _load():
    spec = importlib.util.spec_from_file_location("restage_from_parquet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(pano_id, image):
    return {"image": {"bytes": image, "path": None}, "pano_id": pano_id,
            "record_creation_time": 1, "curb_ramp_points_normalized": [[0.25, 0.5]],
            "pano_coord": [40.0, -73.0], "curb_ramp_coords": [[40.0, -73.0]], "pano_azimuth": 1.5}


def _write_shard(path, rows):
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pyarrow.Table.from_pylist(rows)
    pq.write_table(table, path)


def _snapshot(tmp_path, per_split):
    """A fake HF snapshot: one shard per split, rows as given; returns its dir."""
    snap = tmp_path / "snap"
    for split, rows in per_split.items():
        (snap / split).mkdir(parents=True)
        _write_shard(snap / split / "data-00000-of-00001.parquet", rows)
    return snap


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


# ----- helpers -------------------------------------------------------------------------

def test_expected_counts_are_the_published_recipe():
    m = _load()
    assert m.EXPECTED == {"train": 150063, "val": 42875, "test": 21438}
    assert m.SPLIT_DIRS["validation"] == "val" and m.SPLIT_DIRS["val"] == "val"


def test_is_jpeg_accepts_soi_and_rejects_the_hub_placeholder():
    m = _load()
    assert m.is_jpeg(JPEG)
    assert not m.is_jpeg(DUMMY)
    assert not m.is_jpeg(None)
    assert not m.is_jpeg(b"")
    assert not m.is_jpeg(b"\xff\xd8")  # too short to be an image


def test_split_state_reports_strays_tmp_and_id_hash(tmp_path):
    m = _load()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.jpg").write_bytes(b"x")  # jpg without json
    (tmp_path / "c.json").write_text("{}")  # json without jpg
    (tmp_path / "d.jpg.tmp").write_bytes(b"x")
    pairs, jpg_only, json_only, tmps, digest = m.split_state(tmp_path)
    assert (pairs, jpg_only, json_only, tmps) == (1, 1, 1, 1)
    assert digest == hashlib.sha256(b"a").hexdigest()
    assert m.count_pairs(tmp_path) == (1, 1, 1)


def test_quarantine_manifest_is_sorted_relative_lf(tmp_path):
    m = _load()
    q = tmp_path / "_quarantine"
    for split, pid in (("val", "file_2"), ("train", "file_1")):
        (q / split).mkdir(parents=True)
        (q / split / f"{pid}.jpg").write_bytes(DUMMY)
        (q / split / f"{pid}.json").write_text(json.dumps({"pano_id": pid}))
    assert m.write_quarantine_manifest(tmp_path) == 4
    raw = (q / "MANIFEST.txt").read_bytes()
    assert b"\r" not in raw
    assert raw.decode().splitlines() == ["train/file_1.jpg", "train/file_1.json",
                                         "val/file_2.jpg", "val/file_2.json"]
    assert m.check_quarantine_manifest(tmp_path)
    (q / "train" / "file_3.jpg").write_bytes(DUMMY)
    assert not m.check_quarantine_manifest(tmp_path)


def test_quarantine_manifest_without_quarantine_dir(tmp_path):
    m = _load()
    assert m.write_quarantine_manifest(tmp_path) == 0
    assert not (tmp_path / "_quarantine").exists()
    assert m.check_quarantine_manifest(tmp_path)


def test_write_atomic_leaves_no_tmp(tmp_path):
    m = _load()
    target = tmp_path / "p.jpg"
    m.write_atomic(target, JPEG)
    assert target.read_bytes() == JPEG
    assert list(tmp_path.iterdir()) == [target]


# ----- shard level, through the real process_shard -----------------------------------------

def test_process_shard_writes_valid_rows_and_quarantines_placeholders(tmp_path):
    m = _load()
    snap = _snapshot(tmp_path, {"train": [_row("p1", JPEG), _row("file_1", DUMMY)]})
    out, q = tmp_path / "out" / "train", tmp_path / "out" / "_quarantine" / "train"
    out.mkdir(parents=True)
    shard = str(snap / "train" / "data-00000-of-00001.parquet")
    name, written, skipped, quarantined = m.process_shard((shard, str(out), str(q)))
    assert (written, skipped, quarantined) == (1, 0, 1)
    assert (out / "p1.jpg").read_bytes() == JPEG  # the parquet's own bytes, no re-encode
    meta = json.loads((out / "p1.json").read_text())
    assert "image" not in meta and meta["pano_id"] == "p1" and meta["pano_azimuth"] == 1.5
    assert (q / "file_1.jpg").read_bytes() == DUMMY and (q / "file_1.json").is_file()
    # A second pass skips the written pano and does not duplicate the quarantined one.
    _, written2, skipped2, quarantined2 = m.process_shard((shard, str(out), str(q)))
    assert (written2, skipped2, quarantined2) == (0, 1, 1)
    assert sorted(p.name for p in q.iterdir()) == ["file_1.jpg", "file_1.json"]


def test_process_shard_removes_a_placeholder_written_by_the_old_version(tmp_path):
    m = _load()
    snap = _snapshot(tmp_path, {"train": [_row("file_2", DUMMY)]})
    out, q = tmp_path / "out" / "train", tmp_path / "out" / "_quarantine" / "train"
    out.mkdir(parents=True)
    (out / "file_2.jpg").write_bytes(DUMMY)  # left by the pre-quarantine script
    (out / "file_2.json").write_text("{}")
    m.process_shard((str(snap / "train" / "data-00000-of-00001.parquet"), str(out), str(q)))
    assert not (out / "file_2.jpg").exists() and not (out / "file_2.json").exists()
    assert (q / "file_2.jpg").is_file()


# ----- the gate, through the CLI -------------------------------------------------------

def test_check_fails_when_a_split_directory_is_missing(tmp_path):
    out = tmp_path / "out"
    (out / "train").mkdir(parents=True)
    r = _run("--out", out, "--check")
    assert r.returncode == 1
    assert "MISSING directory" in r.stdout and "[val]" in r.stdout and "[test]" in r.stdout


def test_check_fails_on_tmp_files_and_counts_mismatch(tmp_path):
    out = tmp_path / "out"
    for s in ("train", "val", "test"):
        (out / s).mkdir(parents=True)
    (out / "train" / "x.jpg.tmp").write_bytes(b"x")
    r = _run("--out", out, "--check")
    assert r.returncode == 1
    assert "tmp 1" in r.stdout and "MISMATCH" in r.stdout


def test_check_record_then_check_detects_an_id_change(tmp_path, monkeypatch):
    """With EXPECTED shrunk to the fixture, --check --record writes COUNTS.json and a later
    --check fails when a pano id changes even though the count does not."""
    m = _load()
    out = tmp_path / "out"
    for s in ("train", "val", "test"):
        (out / s).mkdir(parents=True)
        (out / s / "a.jpg").write_bytes(JPEG)
        (out / s / "a.json").write_text("{}")
    monkeypatch.setattr(m, "EXPECTED", {"train": 1, "val": 1, "test": 1})
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--check", "--record"])
    assert m.main() == 0
    counts = json.loads((out / m.COUNTS_FILE).read_text())
    assert set(counts) == {"train", "val", "test"} and counts["train"]["pairs"] == 1
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--check"])
    assert m.main() == 0
    (out / "val" / "a.jpg").rename(out / "val" / "b.jpg")
    (out / "val" / "a.json").rename(out / "val" / "b.json")
    assert m.main() == 1  # same count, different id set


def test_write_mode_refuses_a_snapshot_missing_a_split(tmp_path):
    snap = _snapshot(tmp_path, {"train": [_row("p1", JPEG)]})
    r = _run("--snapshot", snap, "--out", tmp_path / "out", "--workers", "1")
    assert r.returncode == 1 and "lacks split" in r.stdout


def test_write_mode_end_to_end_with_shrunk_expected(tmp_path, monkeypatch):
    m = _load()
    snap = _snapshot(tmp_path, {"train": [_row("p1", JPEG), _row("file_1", DUMMY)],
                                "val": [_row("v1", JPEG)],
                                "test": [_row("t1", JPEG)]})
    out = tmp_path / "out"
    (out / "train").mkdir(parents=True)
    (out / "train" / "stale.jpg.tmp").write_bytes(b"x")  # from a pretend preemption
    monkeypatch.setattr(m, "EXPECTED", {"train": 1, "val": 1, "test": 1})
    monkeypatch.setattr(sys, "argv", ["x", "--snapshot", str(snap), "--out", str(out), "--workers", "1"])
    assert m.main() == 0
    assert not list((out / "train").glob("*.tmp"))
    assert (out / m.COUNTS_FILE).is_file()
    assert (out / "_quarantine" / "MANIFEST.txt").read_text().splitlines() == ["train/file_1.jpg",
                                                                              "train/file_1.json"]
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--check"])
    assert m.main() == 0
