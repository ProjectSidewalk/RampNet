"""Unit tests for fetch_manual_gold (issue #58).

No network: everything runs through --source local against tmp_path fixtures. The
load-bearing guarantees, in the order they matter:

* an --images-only fetch never rewrites the committed records.jsonl / bundle_meta.json
  (the records carry exported detections that a rebuild would discard), on the happy
  path *and* on every error path;
* the guard steers to --images-only rather than --force;
* imagery that cannot be shown to belong under the committed records is refused
  before it can be scored — a --source contradicting bundle_meta.json, a
  bundle_meta.json that does not say, a pano whose pixel size disagrees with
  records.jsonl, or bytes that miss the committed sha256 manifest;
* panos already on disk are skipped, so a preempted fetch resumes;
* the full-build and --force paths still write what they claim to. Those are not
  reachable through --images-only, and the early `return` that implements
  --images-only sits directly above the code that writes them.
"""
import hashlib
import json
import os
import sys

import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import fetch_manual_gold as fmg  # noqa: E402

IDS = ["111", "222"]


def make_jpg(path, size=(4, 2), color=(120, 60, 30)):
    Image.new("RGB", size, color).save(path, "JPEG")


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


def run_full(local, extra=()):
    fmg.main(["--source", "local", "--local-dataset", str(local), *extra])


def committed_bytes(bundle_dir):
    """The two committed files, as bytes — what --images-only promises not to touch."""
    return ((bundle_dir / "records.jsonl").read_bytes(),
            (bundle_dir / "bundle_meta.json").read_bytes())


def write_manifest(bundle_dir, panos_dir, mutate=None):
    """A committed-style imagery_manifest.json for whatever is in panos_dir."""
    panos = {}
    for name in sorted(os.listdir(panos_dir)):
        stem = os.path.splitext(name)[0]
        raw = (panos_dir / name).read_bytes()
        panos[stem] = {"sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "file": name}
    if mutate:
        mutate(panos)
    (bundle_dir / "imagery_manifest.json").write_text(
        json.dumps({"city": "manual_gold", "n": len(panos), "panos": panos}),
        encoding="utf-8")


# --------------------------------------------------------------------------------
# --images-only: the happy path and the committed-files guarantee
# --------------------------------------------------------------------------------

def test_images_only_fetches_imagery_and_touches_nothing_committed(bundle):
    bundle_dir, local = bundle
    before = committed_bytes(bundle_dir)

    run_images_only(local)

    for pid in IDS:
        got = (bundle_dir / "panos" / f"{pid}.jpg").read_bytes()
        assert got == (local / f"{pid}.jpg").read_bytes()  # byte-for-byte copy
    assert committed_bytes(bundle_dir) == before


def test_guard_steers_to_images_only_not_force(bundle):
    bundle_dir, local = bundle
    with pytest.raises(SystemExit, match="images-only"):
        run_full(local)
    assert not (bundle_dir / "panos").exists()  # refused before any fetch


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


def test_incomplete_source_errors_but_leaves_committed_files_alone(bundle):
    bundle_dir, local = bundle
    (local / "222.jpg").unlink()
    before = committed_bytes(bundle_dir)
    with pytest.raises(SystemExit, match="NOT found"):
        run_images_only(local)
    assert (bundle_dir / "panos" / "111.jpg").exists()  # partial imagery is visible
    assert committed_bytes(bundle_dir) == before        # both files, not just records


# --------------------------------------------------------------------------------
# Refusing imagery that cannot be shown to belong under the committed records
# --------------------------------------------------------------------------------

def test_source_contradicting_bundle_meta_is_refused(bundle):
    bundle_dir, local = bundle
    meta = json.loads((bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
    meta["source"] = "hf"
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="bundle_meta.json records source='hf'"):
        run_images_only(local)
    assert not (bundle_dir / "panos").exists()


def test_missing_bundle_meta_is_refused(bundle):
    """The guard must not no-op precisely when the built source is unknown."""
    bundle_dir, local = bundle
    (bundle_dir / "bundle_meta.json").unlink()
    with pytest.raises(SystemExit, match="does not record which source"):
        run_images_only(local)
    assert not (bundle_dir / "panos").exists()


def test_sourceless_bundle_meta_is_refused(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "bundle_meta.json").write_text(
        json.dumps({"built": "2026-07-25", "n_panos": 2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not record which source"):
        run_images_only(local)
    assert not (bundle_dir / "panos").exists()


def test_unparseable_bundle_meta_is_refused(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "bundle_meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        run_images_only(local)
    assert not (bundle_dir / "panos").exists()


def test_pano_size_disagreeing_with_records_is_refused(bundle):
    """Different pixels under the same ids: the records already say how big each is."""
    bundle_dir, local = bundle
    records = "\n".join(
        json.dumps({"pano": {"panorama_id": pid, "width": 4096, "height": 2048}})
        for pid in IDS) + "\n"
    (bundle_dir / "records.jsonl").write_text(records, encoding="utf-8")
    before = committed_bytes(bundle_dir)
    with pytest.raises(SystemExit, match="not the size the committed records claim"):
        run_images_only(local)
    assert committed_bytes(bundle_dir) == before


def test_records_without_dimensions_are_not_cross_checked(bundle, capsys):
    """Legacy records that omit width/height must still fetch — and say they weren't checked."""
    bundle_dir, local = bundle
    records = "\n".join(json.dumps({"pano": {"panorama_id": pid}}) for pid in IDS) + "\n"
    (bundle_dir / "records.jsonl").write_text(records, encoding="utf-8")
    run_images_only(local)
    assert "cannot be cross-checked" in capsys.readouterr().out
    assert (bundle_dir / "panos" / "111.jpg").exists()


def test_malformed_records_line_exits_with_guidance(bundle):
    bundle_dir, local = bundle
    with open(bundle_dir / "records.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"detections": []}) + "\n")
    with pytest.raises(SystemExit, match="not a gold record"):
        run_images_only(local)


def test_unparseable_records_line_exits_with_guidance(bundle):
    bundle_dir, local = bundle
    with open(bundle_dir / "records.jsonl", "a", encoding="utf-8") as f:
        f.write("{oh no\n")
    with pytest.raises(SystemExit, match="not a gold record"):
        run_images_only(local)


def test_duplicate_record_ids_are_refused(bundle):
    """A set comprehension would silently collapse these and pass the drift check."""
    bundle_dir, local = bundle
    with open(bundle_dir / "records.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"pano": {"panorama_id": "111", "width": 4, "height": 2}}) + "\n")
    with pytest.raises(SystemExit, match="repeats panorama_id"):
        run_images_only(local)


# --------------------------------------------------------------------------------
# Resume: panos already on disk
# --------------------------------------------------------------------------------

def test_second_run_skips_panos_already_present(bundle, monkeypatch):
    _, local = bundle
    run_images_only(local)

    copies = []
    real = fmg.shutil.copyfile
    monkeypatch.setattr(fmg.shutil, "copyfile",
                        lambda s, d: (copies.append(d), real(s, d))[1])
    run_images_only(local)
    assert copies == []


def test_refetch_overrides_the_skip(bundle, monkeypatch):
    _, local = bundle
    run_images_only(local)

    copies = []
    real = fmg.shutil.copyfile
    monkeypatch.setattr(fmg.shutil, "copyfile",
                        lambda s, d: (copies.append(d), real(s, d))[1])
    run_images_only(local, extra=["--refetch"])
    assert len(copies) == len(IDS)


def test_refetch_requires_images_only(bundle):
    _, local = bundle
    with pytest.raises(SystemExit, match="only means anything with --images-only"):
        run_full(local, extra=["--refetch"])


def test_wrong_sized_pano_on_disk_is_refetched(bundle):
    """A truncated or wrong-source leftover must not be mistaken for a completed fetch."""
    bundle_dir, local = bundle
    panos = bundle_dir / "panos"
    panos.mkdir()
    make_jpg(panos / "111.jpg", size=(8, 4))  # records say 4x2

    run_images_only(local)

    with Image.open(panos / "111.jpg") as im:
        assert im.size == (4, 2)


def test_unreadable_pano_on_disk_is_refetched(bundle):
    bundle_dir, local = bundle
    panos = bundle_dir / "panos"
    panos.mkdir()
    (panos / "111.jpg").write_bytes(b"not a jpeg at all")

    run_images_only(local)

    assert (panos / "111.jpg").read_bytes() == (local / "111.jpg").read_bytes()


def test_fully_present_bundle_fetches_nothing(bundle, monkeypatch, capsys):
    """The hf path costs hours, so a complete bundle must not re-enter the fetcher."""
    _, local = bundle
    run_images_only(local)

    def boom(*a, **k):
        raise AssertionError("fetcher entered with nothing to fetch")

    monkeypatch.setattr(fmg, "fetch_local", boom)
    run_images_only(local)
    assert "nothing to fetch" in capsys.readouterr().out


# --------------------------------------------------------------------------------
# The sha256 imagery manifest (the check the nine city splits already have)
# --------------------------------------------------------------------------------

def test_matching_manifest_is_reported(bundle, capsys):
    bundle_dir, local = bundle
    run_images_only(local)
    write_manifest(bundle_dir, bundle_dir / "panos")

    run_images_only(local, extra=["--refetch"])
    assert "match the committed hashes" in capsys.readouterr().out


def test_manifest_mismatch_is_refused(bundle):
    bundle_dir, local = bundle
    run_images_only(local)

    def flip(panos):
        panos["111"]["sha256"] = "0" * 64

    write_manifest(bundle_dir, bundle_dir / "panos", mutate=flip)
    before = committed_bytes(bundle_dir)
    with pytest.raises(SystemExit, match="different bytes"):
        run_images_only(local, extra=["--refetch"])
    assert committed_bytes(bundle_dir) == before


def test_no_manifest_check_skips_verification(bundle):
    bundle_dir, local = bundle
    run_images_only(local)

    def flip(panos):
        panos["111"]["sha256"] = "0" * 64

    write_manifest(bundle_dir, bundle_dir / "panos", mutate=flip)
    run_images_only(local, extra=["--refetch", "--no-manifest-check"])  # no raise


def test_absent_manifest_names_the_command_that_writes_it(bundle, capsys):
    """manual_gold is the one split with no committed content hash — say so, don't hide it."""
    _, local = bundle
    run_images_only(local)
    out = capsys.readouterr().out
    assert "imagery_manifest.json does not exist" in out
    assert "imagery_manifest.py --write --cities manual_gold" in out


# --------------------------------------------------------------------------------
# Flag combinations
# --------------------------------------------------------------------------------

def test_audit_with_images_only_is_refused_not_ignored(bundle, monkeypatch):
    _, local = bundle
    monkeypatch.setattr(fmg, "audit", lambda ids: pytest.fail("audit ran anyway"))
    with pytest.raises(SystemExit, match="cannot also fetch"):
        fmg.main(["--audit", "--images-only", "--source", "local",
                  "--local-dataset", str(local)])


def test_audit_with_force_is_refused_not_ignored(bundle, monkeypatch):
    _, local = bundle
    monkeypatch.setattr(fmg, "audit", lambda ids: pytest.fail("audit ran anyway"))
    with pytest.raises(SystemExit, match="cannot also fetch"):
        fmg.main(["--audit", "--force", "--source", "local",
                  "--local-dataset", str(local)])


# --------------------------------------------------------------------------------
# Sidecar metadata: needed by a full build, never read by --images-only
# --------------------------------------------------------------------------------

def test_images_only_does_not_need_sidecar_json(bundle):
    """Imagery rsync'd without the sidecars still fetches — images-only discards them."""
    bundle_dir, local = bundle
    for pid in IDS:
        (local / f"{pid}.json").unlink()

    run_images_only(local)

    for pid in IDS:
        assert (bundle_dir / "panos" / f"{pid}.jpg").exists()


def test_full_build_missing_sidecar_exits_with_guidance(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "records.jsonl").unlink()
    (local / "222.json").unlink()
    with pytest.raises(SystemExit, match="needs every pano's sidecar"):
        run_full(local)


def test_full_build_unparseable_sidecar_exits_with_guidance(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "records.jsonl").unlink()
    (local / "222.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        run_full(local)


# --------------------------------------------------------------------------------
# The full build and --force rebuild — the paths the --images-only early return
# sits directly above
# --------------------------------------------------------------------------------

def test_full_build_writes_records_and_bundle_meta(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "records.jsonl").unlink()
    (bundle_dir / "bundle_meta.json").unlink()

    run_full(local)

    lines = (bundle_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    assert [r["pano"]["panorama_id"] for r in recs] == sorted(IDS)  # written sorted
    for r in recs:
        assert (r["pano"]["width"], r["pano"]["height"]) == (4, 2)
        assert r["pano"]["pano_azimuth"] == 12.0     # sidecar metadata carried through
        assert "detections" not in r                 # a build never invents detections
        assert "curb_ramp_points_normalized" not in r["pano"]  # nor auto labels

    meta = json.loads((bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "local"
    assert meta["n_panos"] == len(IDS)
    assert meta["hf_split"] == "test"
    for pid in IDS:
        assert (bundle_dir / "panos" / f"{pid}.jpg").exists()


def test_force_rebuild_discards_detections(bundle):
    """--force is destructive by design; pin that so the help text stays honest."""
    bundle_dir, local = bundle
    assert "detections" in (bundle_dir / "records.jsonl").read_text(encoding="utf-8")

    run_full(local, extra=["--force"])

    text = (bundle_dir / "records.jsonl").read_text(encoding="utf-8")
    assert "detections" not in text
    meta = json.loads((bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "local" and meta["n_panos"] == len(IDS)


def test_full_build_incomplete_source_still_reports_missing(bundle):
    bundle_dir, local = bundle
    (bundle_dir / "records.jsonl").unlink()
    (local / "222.jpg").unlink()
    with pytest.raises(SystemExit, match="NOT found"):
        run_full(local)
    # The records that *were* built are kept — the exit is a report, not a rollback.
    assert (bundle_dir / "records.jsonl").exists()
