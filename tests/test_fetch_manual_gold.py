"""Unit tests for fetch_manual_gold's --images-only path (issue #58).

No network: everything runs through --source local against tmp_path fixtures. The
load-bearing guarantees: an images-only fetch never rewrites the committed
records.jsonl / bundle_meta.json (the records carry exported detections that a
rebuild would discard), the guard steers to --images-only rather than --force,
and a fetch whose --source contradicts the bundle's recorded source is refused
before anything downloads (byte fidelity: the two sources are not identical).
"""
import json
import os
import sys

import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import fetch_manual_gold as fmg  # noqa: E402

IDS = ["111", "222"]


def make_jpg(path, size=(4, 2)):
    Image.new("RGB", size, (120, 60, 30)).save(path, "JPEG")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A miniature gold bundle + local imagery source, wired into the module."""
    labels = tmp_path / "manual_labels"
    labels.mkdir()
    bundle_dir = tmp_path / "manual_gold"
    bundle_dir.mkdir()
    local = tmp_path / "dataset_test"
    local.mkdir()

    for pid in IDS:
        (labels / f"{pid}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        make_jpg(local / f"{pid}.jpg")
        (local / f"{pid}.json").write_text(
            json.dumps({"pano_coord": [0.0, 0.0], "pano_azimuth": 12.0}), encoding="utf-8")

    records = "\n".join(
        json.dumps({"pano": {"panorama_id": pid, "width": 4, "height": 2},
                    "detections": [{"x": 0.5, "y": 0.5, "score": 0.9}]})
        for pid in IDS) + "\n"
    (bundle_dir / "records.jsonl").write_text(records, encoding="utf-8")
    (bundle_dir / "bundle_meta.json").write_text(
        json.dumps({"built": "2026-07-25", "source": "local", "n_panos": len(IDS)}),
        encoding="utf-8")

    monkeypatch.setattr(fmg, "LABELS_DIR", str(labels))
    monkeypatch.setattr(fmg, "BUNDLE_DIR", str(bundle_dir))
    return bundle_dir, local


def run_images_only(local, extra=()):
    fmg.main(["--images-only", "--source", "local", "--local-dataset", str(local), *extra])


def test_images_only_fetches_imagery_and_touches_nothing_committed(bundle):
    bundle_dir, local = bundle
    records_before = (bundle_dir / "records.jsonl").read_bytes()
    meta_before = (bundle_dir / "bundle_meta.json").read_bytes()

    run_images_only(local)

    for pid in IDS:
        got = (bundle_dir / "panos" / f"{pid}.jpg").read_bytes()
        assert got == (local / f"{pid}.jpg").read_bytes()  # byte-for-byte copy
    assert (bundle_dir / "records.jsonl").read_bytes() == records_before
    assert (bundle_dir / "bundle_meta.json").read_bytes() == meta_before


def test_guard_steers_to_images_only_not_force(bundle):
    bundle_dir, local = bundle
    with pytest.raises(SystemExit, match="images-only"):
        fmg.main(["--source", "local", "--local-dataset", str(local)])
    assert not (bundle_dir / "panos").exists()  # refused before any fetch


def test_source_contradicting_bundle_meta_is_refused(bundle):
    bundle_dir, local = bundle
    meta = json.loads((bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
    meta["source"] = "hf"
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="bundle_meta.json records source='hf'"):
        run_images_only(local)
    assert not (bundle_dir / "panos").exists()


def test_images_only_and_force_contradict(bundle):
    _, local = bundle
    with pytest.raises(SystemExit, match="contradict"):
        run_images_only(local, extra=["--force"])


def test_images_only_needs_existing_records(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "records.jsonl").unlink()
    with pytest.raises(SystemExit, match="existing records.jsonl"):
        run_images_only(local)


def test_label_record_drift_demands_full_rebuild(bundle):
    bundle_dir, local = bundle
    with open(bundle_dir / "records.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"pano": {"panorama_id": "333", "width": 4, "height": 2}}) + "\n")
    with pytest.raises(SystemExit, match="disagree"):
        run_images_only(local)


def test_incomplete_source_errors_but_leaves_records_alone(bundle):
    bundle_dir, local = bundle
    (local / "222.jpg").unlink()
    records_before = (bundle_dir / "records.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="NOT found"):
        run_images_only(local)
    assert (bundle_dir / "panos" / "111.jpg").exists()  # partial imagery is visible
    assert (bundle_dir / "records.jsonl").read_bytes() == records_before
