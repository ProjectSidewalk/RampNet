"""#111 benchmark depth harvester: decoder, status mapping, manifest round-trip, --check.

Offline and CPU-only: no streetlevel import, no network. The --check test reads only the
committed manifests (and the local archive when one happens to exist).
"""
import base64
import json
import os
import struct
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
H = pytest.importorskip("harvest_depth_111")


def _blob(width=8, height=4, ground_d=2.5, ground_n=(0.0, 0.0, -1.0)):
    """A payload built like the labeler's tests/test_harvest_depth.py::_blob: sky on the
    top row, a wall on the second, a ground plane on the bottom half."""
    planes = [(0.0, 0.0, 0.0, 0.0), (*ground_n, ground_d), (1.0, 0.0, 0.0, 6.0)]
    idx = [0] * width + [2] * width + [1] * (width * (height - 2))
    raw = struct.pack("<BHHHB", 8, len(planes), width, height, 8) + bytes(idx)
    raw += b"".join(struct.pack("<ffff", *p) for p in planes)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_decoder_level_ground():
    g = H.ground_summary(_blob())
    assert g["n_planes"] == 3
    assert g["degenerate"] is False
    assert g["camera_height_m"] == pytest.approx(2.5)
    assert g["ground_tilt_deg"] == pytest.approx(0.0)
    assert g["exactly_level"] is True


def test_decoder_tilted_ground_and_wall_rejected():
    nz = -0.99756405  # 4 degrees off vertical
    g = H.ground_summary(_blob(ground_d=2.3, ground_n=(0.0697565, 0.0, nz)))
    assert g["camera_height_m"] == pytest.approx(2.3, abs=1e-4)
    assert g["ground_tilt_deg"] == pytest.approx(4.0, abs=1e-3)
    assert g["exactly_level"] is False


def test_decoder_rejects_bad_header():
    raw = base64.urlsafe_b64decode(_blob() + "==")
    bad = base64.urlsafe_b64encode(bytes([9]) + raw[1:]).decode()
    with pytest.raises(ValueError, match="header size"):
        H.parse_payload(bad)


def _response(code, blob=None):
    msg = [[code], None, None, None, None,
           [[None, [None, None, [297.2, 88.9, 0.77]], None, None, None,
             [None, [None, None, blob]]]],
           [None] * 7 + [[2019, 11]]]
    return [None, [msg]]


@pytest.mark.parametrize("code", [1, 3])
def test_status_saved(code):
    status, blob, meta = H.classify_response(_response(code, _blob()))
    assert status == H.SAVED and blob == _blob()
    assert meta == {"heading_deg": 297.2, "pitch_deg": 1.1, "roll_deg": 0.77,
                    "capture_ym": "2019-11"}


def test_status_gone_nodepth_error():
    assert H.classify_response(_response(2))[0] == H.GONE
    assert H.classify_response(_response(1, None))[0] == H.NO_DEPTH
    assert H.classify_response({"nope": 1})[0] == H.ERROR


def test_is_block():
    assert H.is_block(403, "https://www.google.com/x", "")
    assert H.is_block(200, "https://www.google.com/sorry/index?continue=x", "")
    assert H.is_block(200, "https://consent.google.com/ml?x", "")
    assert H.is_block(200, "https://www.google.com/x", "<html>Our systems have detected unusual traffic")
    # a real JSON body is never read for block markers
    assert not H.is_block(200, "https://www.google.com/x", ")]}'\n[\"/sorry/ street\"]")


def test_pacer_backoff_and_recovery():
    p = H.Pacer(sleep=lambda s: None)
    for _ in range(H.PACE_RECOVER_AFTER - 1):
        p.ok()
    assert p.interval == H.PACE_START_S
    p.ok()
    assert p.interval == pytest.approx(H.PACE_START_S * H.PACE_RECOVER_FACTOR)
    p.push_back()
    assert p.interval == pytest.approx(H.PACE_START_S * H.PACE_RECOVER_FACTOR * 2)
    for _ in range(20):
        p.push_back()
    assert p.interval == H.PACE_CEIL_S


@pytest.fixture
def fake_repo(tmp_path):
    """A one-split repo: three panos (one id leading with '-'), two saved, one gone."""
    split = "toy"
    d = tmp_path / "benchmark" / split
    d.mkdir(parents=True)
    ids = ["-leadingHyphen", "abc", "gone1"]
    with open(d / "records.jsonl", "w", encoding="utf-8") as fh:
        for pid in ids:
            fh.write(json.dumps({"pano": {"panorama_id": pid}}) + "\n")
    (d / "imagery_manifest.json").write_text(json.dumps({"panos": {p: {} for p in ids}}))
    (d / "depth").mkdir()
    for pid in ids[:2]:
        H.write_archive_file(H.archive_path(split, pid, str(tmp_path)), pid, _blob(),
                             {"heading_deg": 1.0}, "2026-10-05T00:00:00+00:00")
    (d / "depth" / "gone.txt").write_text("gone1\n")
    return str(tmp_path), split


def test_hyphen_id_is_a_path(fake_repo):
    repo, split = fake_repo
    path = H.archive_path(split, "-leadingHyphen", repo)
    assert os.path.basename(path) == "-leadingHyphen.json.gz" and os.path.exists(path)


def test_manifest_round_trip_and_tamper(fake_repo):
    repo, split = fake_repo
    ids = H.split_pano_ids(split, repo)
    man = H.build_manifest(split, ids, repo=repo)
    assert (man["n_saved"], man["n_gone"], man["n_not_fetched"]) == (2, 1, 0)
    assert man["panos"]["abc"]["sha256"] == H.payload_sha256(_blob())
    H.dump_json(os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME), man)
    raw = open(os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME), "rb").read()
    assert b"\r\n" not in raw
    assert H.verify(split, repo) == ([], "archive verified")

    # flip one payload byte: verify names the pano
    H.write_archive_file(H.archive_path(split, "abc", repo), "abc", _blob(ground_d=2.4),
                         {}, "2026-10-06T00:00:00+00:00")
    problems, _ = H.verify(split, repo)
    assert problems == [f"{split}/abc: sha256 drift"]


def test_hash_independent_of_fetch_time(fake_repo):
    repo, split = fake_repo
    ids = H.split_pano_ids(split, repo)
    a = H.build_manifest(split, ids, repo=repo)
    H.write_archive_file(H.archive_path(split, "abc", repo), "abc", _blob(), {},
                         "2030-01-01T00:00:00+00:00")
    b = H.build_manifest(split, ids, repo=repo)
    assert a["digest"] == b["digest"]


def test_archive_absent_is_not_a_failure(fake_repo):
    repo, split = fake_repo
    man = H.build_manifest(split, H.split_pano_ids(split, repo), repo=repo)
    H.dump_json(os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME), man)
    ddir = H.depth_dir(split, repo)
    for f in os.listdir(ddir):
        os.remove(os.path.join(ddir, f))
    os.rmdir(ddir)
    assert H.verify(split, repo) == ([], "archive absent, manifest self-consistent")


def test_aborted_manifest_keeps_status(fake_repo):
    repo, split = fake_repo
    os.remove(H.archive_path(split, "abc", repo))
    man = H.build_manifest(split, H.split_pano_ids(split, repo),
                           errors={}, aborted_at={"pano_id": "abc", "index": 1, "reason": "blocked"},
                           repo=repo)
    assert man["panos"]["abc"]["status"] == H.NOT_FETCHED
    assert man["aborted_at"]["reason"] == "blocked"


def test_committed_manifests_check():
    """--check over the committed manifests: self-consistent, and where the labeler
    comparison is committed, camera heights on identical payloads agree to 1e-3."""
    have = [s for s in H.HARVEST_SPLITS if H.load_manifest(s) is not None]
    assert "manual_gold" in have
    for split in have:
        problems, note = H.verify(split)
        assert problems == [], problems[:5]
        assert note in ("archive verified", "archive absent, manifest self-consistent")
        assert H.check_heights_against_labeler(split) == []
    assert H.run_check() == 0


def test_manual_gold_manifest_covers_every_pano():
    man = H.load_manifest("manual_gold")
    assert man["n_requested"] == 1000 == len(man["panos"])
    assert sum(man[f"n_{s}"] for s in H.STATUSES) == 1000
