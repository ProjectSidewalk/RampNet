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
    assert problems == [f"{split}/abc: payload sha256 drift"]


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


# ------------------------------------------------------------ verify catches more than hashes

def _commit_manifest(repo, split):
    man = H.build_manifest(split, H.split_pano_ids(split, repo), repo=repo)
    H.dump_json(os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME), man)
    return man


def test_verify_rebuilds_entries_and_catches_a_hand_edit(fake_repo):
    repo, split = fake_repo
    man = _commit_manifest(repo, split)
    man["panos"]["abc"]["camera_height_m"] = 2.4      # hand edit, payload untouched
    H.dump_json(os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME), man)
    problems, _ = H.verify(split, repo)
    assert problems == [f"{split}/abc: fields differ from the archive: ['camera_height_m']"]


def test_verify_flags_an_unlisted_archive_file(fake_repo):
    repo, split = fake_repo
    _commit_manifest(repo, split)
    H.write_archive_file(H.archive_path(split, "gone1", repo), "gone1", _blob(), {}, "t")
    problems, _ = H.verify(split, repo)
    assert problems == [f"{split}/gone1: archived but manifest says gone"]


def test_verify_archive_dir_checks_a_copy(fake_repo, tmp_path):
    repo, split = fake_repo
    _commit_manifest(repo, split)
    copy = tmp_path / "copy"
    copy.mkdir()
    for pid in ("-leadingHyphen", "abc"):
        H.write_archive_file(str(copy / (pid + ".json.gz")), pid, _blob(ground_d=2.4), {}, "t")
    problems, _ = H.verify(split, repo, archive_dir=str(copy))
    assert sorted(problems) == [f"{split}/-leadingHyphen: payload sha256 drift",
                                f"{split}/abc: payload sha256 drift"]


def test_manifest_records_both_hashes(fake_repo):
    repo, split = fake_repo
    e = _commit_manifest(repo, split)["panos"]["abc"]
    path = H.archive_path(split, "abc", repo)
    assert e["file_sha256"] == H.file_sha256(path) and e["file_bytes"] == os.path.getsize(path)
    assert e["payload_b64_chars"] == len(_blob()) and e["sha256"] == H.payload_sha256(_blob())


def test_prior_error_is_kept_when_not_reached(fake_repo):
    repo, split = fake_repo
    os.remove(H.archive_path(split, "abc", repo))
    prior = {"panos": {"abc": {"status": H.ERROR, "error": "HTTP 503"}}}
    man = H.build_manifest(split, H.split_pano_ids(split, repo), repo=repo, prior=prior)
    assert man["panos"]["abc"] == {"status": H.ERROR, "error": "HTTP 503"}


def test_gone_codes_round_trip(tmp_path):
    path = str(tmp_path / "gone.txt")
    H._write_ids(path, {"a": 2, "-b": None})
    assert open(path, encoding="utf-8").read() == "-b\na\t2\n"
    assert H._load_codes(path) == {"-b": None, "a": 2}


def test_city_of_and_height_status():
    assert H.city_of({"pano_coord": [40.7, None]}) == "other"
    assert H.city_of({"lat": 45.5, "lng": -122.6}) == "portland"
    both = {"degenerate": True, "camera_height_m": 2.5, "exactly_level": True}
    assert H.classify_height(both) == "degenerate"
    assert H.classify_height({"degenerate": False, "camera_height_m": 0.3,
                              "exactly_level": False}) == "implausible"


# ------------------------------------------------------------------- harvest, offline

JSON_PREFIX = ")]}'" + "\n"


class _Resp:
    def __init__(self, status_code=200, body=None, url="https://www.google.com/maps/photometa/v1"):
        self.status_code, self.url = status_code, url
        self.text = JSON_PREFIX + json.dumps(body) if body is not None else ""


class _Session:
    """Replays a list of responses (or a callable url -> response); counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        r = self.script(url) if callable(self.script) else self.script[len(self.calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r


def _harvest(repo, split, session, **kw):
    return H.harvest(split, repo=repo, session=session, url_builder=lambda pid: "u/" + pid,
                     pacer=H.Pacer(sleep=lambda s: None), **kw)


@pytest.fixture
def empty_repo(tmp_path):
    """A split of n panos with no archive and no manifest."""
    def make(n, split="toy"):
        d = tmp_path / "benchmark" / split
        d.mkdir(parents=True)
        ids = [f"p{i:03d}" for i in range(n)]
        with open(d / "records.jsonl", "w", encoding="utf-8") as fh:
            for pid in ids:
                fh.write(json.dumps({"pano": {"panorama_id": pid}}) + "\n")
        return str(tmp_path), split, ids
    return make


def test_403_stops_after_one_request(empty_repo):
    repo, split, _ = empty_repo(3)
    s = _Session([_Resp(403)] * 3)
    man, log, aborted = _harvest(repo, split, s)
    assert len(s.calls) == 1 and aborted["reason"].startswith("blocked")
    assert man["n_not_fetched"] == 3 and man["aborted_at"]["index"] == 0


@pytest.mark.parametrize("url", ["https://www.google.com/sorry/index?x",
                                 "https://consent.google.com/ml"])
def test_sorry_and_consent_stop(empty_repo, url):
    repo, split, _ = empty_repo(2)
    s = _Session([_Resp(200, body=[], url=url)] * 2)
    _, _, aborted = _harvest(repo, split, s)
    assert len(s.calls) == 1 and aborted is not None


def test_non_json_body_stops(empty_repo):
    repo, split, _ = empty_repo(2)
    r = _Resp(200)
    r.text = "<html>hello</html>"
    s = _Session([r, r])
    _, _, aborted = _harvest(repo, split, s)
    assert len(s.calls) == 1 and "prefix" in aborted["reason"]


def test_429_retries_at_most_three_times(empty_repo):
    repo, split, _ = empty_repo(2)
    s = _Session([_Resp(429)] * 3 + [_Resp(200, body=_response(1, _blob()))])
    man, log, aborted = _harvest(repo, split, s)
    assert aborted is None and len(s.calls) == 4 and log["push_backs"] == 3
    assert man["panos"]["p000"]["status"] == H.ERROR and man["panos"]["p001"]["status"] == H.SAVED


def test_network_errors_count_as_push_backs(empty_repo):
    repo, split, _ = empty_repo(1)
    s = _Session([OSError("reset")] * 3)
    man, log, _ = _harvest(repo, split, s)
    assert log["push_backs"] == 3 and man["panos"]["p000"]["error"] == "network: OSError"


def test_consecutive_failures_abort(empty_repo):
    repo, split, _ = empty_repo(30)
    s = _Session(lambda url: _Resp(404))
    man, log, aborted = _harvest(repo, split, s)
    assert log["attempted"] == H.MAX_CONSECUTIVE_FAILURES and "consecutive" in aborted["reason"]
    assert man["n_error"] == H.MAX_CONSECUTIVE_FAILURES and man["n_not_fetched"] == 5


def test_no_depth_alarm(empty_repo):
    repo, split, _ = empty_repo(120)
    # ids ending in 8 or 9 serve no payload: 20% no_depth, well over the 5% alarm
    s = _Session(lambda url: _Resp(200, body=_response(1, None if url[-1] in "89" else _blob())))
    man, log, aborted = _harvest(repo, split, s)
    assert aborted is not None and "no_depth alarm" in aborted["reason"]
    assert log["attempted"] == H.NO_DEPTH_ALARM_AFTER


def test_corrupt_payload_is_an_error_not_a_crash(empty_repo):
    repo, split, _ = empty_repo(2)
    bad = base64.urlsafe_b64encode(b"\x09" + b"\x00" * 20).decode()
    s = _Session([_Resp(200, body=_response(1, bad)), _Resp(200, body=_response(1, _blob()))])
    man, log, aborted = _harvest(repo, split, s)
    assert aborted is None
    assert man["panos"]["p000"]["status"] == H.ERROR
    assert "corrupt payload" in man["panos"]["p000"]["error"]
    assert man["panos"]["p001"]["status"] == H.SAVED


def test_gone_code_is_recorded(empty_repo):
    repo, split, _ = empty_repo(1)
    man, _, _ = _harvest(repo, split, _Session([_Resp(200, body=_response(2))]))
    assert man["panos"]["p000"] == {"status": H.GONE, "code": 2}
    assert H._load_codes(os.path.join(H.depth_dir(split, repo), "gone.txt")) == {"p000": 2}


def test_harvest_never_overwrites_a_resolved_record(empty_repo):
    """A partial rebuild in a clean clone (archive absent) must not rewrite the record."""
    repo, split, _ = empty_repo(2)
    _harvest(repo, split, _Session(lambda url: _Resp(200, body=_response(1, _blob()))))
    record = os.path.join(H.split_dir(split, repo), H.MANIFEST_NAME)
    before = open(record, "rb").read()
    ddir = H.depth_dir(split, repo)
    for f in os.listdir(ddir):
        os.remove(os.path.join(ddir, f))
    s2 = _Session(lambda url: _Resp(200, body=_response(1, _blob(ground_d=2.4))))
    man, log, _ = _harvest(repo, split, s2, limit=1)
    assert open(record, "rb").read() == before
    assert os.path.basename(log["manifest_path"]).startswith("depth_manifest.refetch-")
    assert man["n_saved"] == 1 and man["n_not_fetched"] == 1
    problems, _ = H.verify(split, repo)
    assert f"{split}/p000: payload sha256 drift" in problems
    assert f"{split}/p001: missing from the archive" in problems


def test_resume_into_an_unresolved_record_writes_it(empty_repo):
    repo, split, _ = empty_repo(2)
    s = _Session(lambda url: _Resp(200, body=_response(1, _blob())))
    _harvest(repo, split, s, limit=1)
    man, log, _ = _harvest(repo, split, s, resume=True)
    assert log["manifest_path"].endswith(H.MANIFEST_NAME) and man["n_saved"] == 2


def test_summarize_survives_no_comparable_revisions(tmp_path):
    split = "bend"
    d = tmp_path / "benchmark" / split
    (d / "depth").mkdir(parents=True)
    (d / "records.jsonl").write_text(json.dumps({"pano": {"panorama_id": "a", "lat": 44.0,
                                                          "lng": -121.3}}) + "\n")
    H.write_archive_file(str(d / "depth" / "a.json.gz"), "a", _blob(), {}, "t")
    man = H.build_manifest(split, ["a"], repo=str(tmp_path))
    H.dump_json(str(d / H.MANIFEST_NAME), man)
    H.dump_json(str(d / H.COMPARE_NAME), {"tally": {"differs": 1}, "panos": {
        "a": {"identical": False, "labeler_n_planes": 3, "labeler_exactly_level": True,
              "index_agreement": None}}})
    lines = []
    H.summarize(repo=str(tmp_path), out=lines.append)
    text = "\n".join(lines)
    assert "not comparable" in text and "no comparable camera heights" in text
