"""Unit tests for the imagery integrity manifests (#46 replication, #21).

Pure logic plus tiny temp files — no panoramas.

What these protect: a committed verdict describes specific pixels. When the panoramas
move to Hugging Face and come back down, "same filename, same pano id" is not evidence
that the bytes are the ones the reviewer judged — a re-fetch can return re-stitched or
re-compressed imagery. ``compare`` is what turns that from a hope into a check, so its
job is to be **loud about changed bytes** and never to call a drifted set OK.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import imagery_manifest as im  # noqa: E402


def _entry(h, name="p.jpg"):
    return {"sha256": h, "bytes": 10, "file": name}


# --------------------------------------------------------------------------- #
# compare — the integrity check
# --------------------------------------------------------------------------- #
def test_identical_sets_are_ok():
    rec = {"a": _entry("h1"), "b": _entry("h2")}
    ok, missing, extra, changed = im.compare(dict(rec), rec)
    assert ok and not missing and not extra and not changed


def test_changed_bytes_are_caught():
    # The failure this exists for: same id, different pixels.
    rec = {"a": _entry("h1")}
    ok, _, _, changed = im.compare({"a": _entry("DIFFERENT")}, rec)
    assert not ok and changed == ["a"]


def test_a_missing_pano_fails_verification():
    ok, missing, _, _ = im.compare({}, {"a": _entry("h1")})
    assert not ok and missing == ["a"]


def test_extra_local_panos_do_not_fail_verification():
    # Having MORE imagery than was reviewed is not a threat to the verdicts; it is
    # reported so it is visible, but it must not mark the set as drifted.
    ok, _, extra, _ = im.compare({"a": _entry("h1"), "z": _entry("h9")},
                                 {"a": _entry("h1")})
    assert ok and extra == ["z"]


def test_changed_and_missing_are_reported_separately():
    rec = {"a": _entry("h1"), "b": _entry("h2")}
    _, missing, _, changed = im.compare({"a": _entry("nope")}, rec)
    assert missing == ["b"] and changed == ["a"]


# --------------------------------------------------------------------------- #
# digest_of — one comparable value per split
# --------------------------------------------------------------------------- #
def test_the_digest_is_order_independent():
    a = {"x": _entry("h1"), "y": _entry("h2")}
    b = {"y": _entry("h2"), "x": _entry("h1")}
    assert im.digest_of(a) == im.digest_of(b)


def test_the_digest_moves_when_any_pano_changes():
    a = {"x": _entry("h1")}
    assert im.digest_of(a) != im.digest_of({"x": _entry("h2")})


def test_the_digest_moves_when_a_pano_is_added():
    a = {"x": _entry("h1")}
    assert im.digest_of(a) != im.digest_of({"x": _entry("h1"), "y": _entry("h2")})


def test_the_digest_ignores_metadata_that_is_not_content():
    # bytes/width/height are convenience fields; the hash is the identity.
    a = {"x": {"sha256": "h1", "bytes": 10}}
    b = {"x": {"sha256": "h1", "bytes": 999, "width": 4096}}
    assert im.digest_of(a) == im.digest_of(b)


# --------------------------------------------------------------------------- #
# scan / sha256 — real files, tiny ones
# --------------------------------------------------------------------------- #
def test_hashing_is_content_addressed(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert im.sha256_file(str(a)) == im.sha256_file(str(b))
    b.write_bytes(b"hellp")
    assert im.sha256_file(str(a)) != im.sha256_file(str(b))


def test_scan_keys_by_pano_id_not_filename(tmp_path):
    (tmp_path / "SOME_PANO_ID.jpg").write_bytes(b"x")
    got = im.scan(str(tmp_path), with_size=False)
    assert list(got) == ["SOME_PANO_ID"]
    assert got["SOME_PANO_ID"]["file"] == "SOME_PANO_ID.jpg"


def test_scan_ignores_non_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "manifest.json").write_bytes(b"{}")
    assert list(im.scan(str(tmp_path), with_size=False)) == ["a"]


def test_a_missing_directory_scans_empty_rather_than_raising(tmp_path):
    assert im.scan(str(tmp_path / "nope")) == {}


# --------------------------------------------------------------------------- #
# the committed manifests themselves
# --------------------------------------------------------------------------- #
def test_every_reviewed_split_has_a_committed_imagery_manifest():
    # A split with human verdicts but no imagery manifest is one whose judgments
    # cannot be re-paired with imagery after the HF round trip.
    import glob
    reviewed = [os.path.basename(os.path.dirname(p))
                for p in glob.glob(os.path.join(REPO, "benchmark", "*", "verdicts.json"))]
    assert reviewed, "no reviewed splits found"
    for city in reviewed:
        assert im.load(city) is not None, f"{city} has verdicts but no {im.MANIFEST_NAME}"


def test_committed_manifests_record_a_digest_and_per_pano_hashes():
    import glob
    for path in glob.glob(os.path.join(REPO, "benchmark", "*", im.MANIFEST_NAME)):
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
        assert p["digest"] and p["n"] == len(p["panos"]), path
        for pid, rec in list(p["panos"].items())[:3]:
            assert len(rec["sha256"]) == 64, (path, pid)
