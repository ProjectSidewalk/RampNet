"""scripts/build_sparse_labels.py (#263) on a synthetic labeler run.

The run is laid out in metres, and each pano exercises one rule, so a wrong rule changes
exactly one assertion:

    origin   S1  sites.jsonl site, members P1 P2 P3 P7 at 0.7 (in_refit) and L1 at 0.2
                 (support, in_refit false)
                   P1, P2, P3  -> sparse, strong_site (P1 also carries an unverified peak
                                  -> one ignore point)
                   P7          -> missing_image (no native file)
                   L1          -> skipped (support member below 0.55: not a positive)
             P5  mined row of miner site 1, emit true             -> sparse, mined
             E2  zero peaks, 14 m from S1                          -> never verified_empty
    100 m    S3  file says 3 operational panos, but the third member is 0.3: a recount
                 gives 2 -> not strong; Q1, Q2 skipped
    200 m    S2  weak site (2 panos); P4 its member; P6 peaks, a mined row with emit false
             E1  zero peaks, 5 m from weak site S2                 -> never verified_empty
    300 m    S4  strong, but member B1 is a benchmark pano -> dropped whole; R1, R2 skipped
    400 m    E3  zero peaks, one peak-carrying neighbour (P8)      -> skipped
    500 m    S5  strong, but member X1 is in --exclude-ids -> dropped whole; T1, T2 skipped
             miner site 3: Z1 emit true, X1 row -> dropped whole; Z1 skipped
    600 m    E4  zero peaks, N1 and N2 carry peaks, no site within 25 m
                 -> verified_empty, only with --emit-verified-empty

CPU only, no network; native images are small synthetic JPEGs.
"""
import csv
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import pytest

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import build_sparse_labels as bsl  # noqa: E402

LAT0, LNG0 = 45.75, 4.85
M_LAT = 1 / 111_195.0
M_LNG = 1 / (111_195.0 * math.cos(math.radians(LAT0)))


def ll(east_m, north_m):
    return LAT0 + north_m * M_LAT, LNG0 + east_m * M_LNG


PK = (0.5, 0.6)          # the default single stored peak
LAYOUT = {   # pano: (east m, north m, stored peaks [(x, y)])
    "P1": (0, 5, [(0.25, 0.6), (0.7, 0.6)]), "P2": (0, -5, [(0.75, 0.62)]),
    "P3": (5, 0, [(0.5, 0.58)]), "P7": (3, 3, [(0.4, 0.6)]), "L1": (-3, 3, [(0.33, 0.6)]),
    "P5": (-8, 0, [(0.9, 0.61)]), "E2": (10, 10, []),
    "Q1": (100, 0, [PK]), "Q2": (100, 5, [PK]), "Q3": (105, 0, [PK]),
    "P4": (200, 5, [PK]), "P6": (205, 0, [PK, (0.2, 0.6)]), "E1": (200, -5, []),
    "B1": (300, 0, [PK]), "R1": (300, 5, [PK]), "R2": (305, 0, [PK]),
    "E3": (400, 0, []), "P8": (405, 0, [PK]),
    "X1": (500, 0, [PK]), "T1": (500, 5, [PK]), "T2": (505, 0, [PK]), "Z1": (510, 0, [PK]),
    "E4": (600, 0, []), "N1": (600, 8, [PK]), "N2": (605, 0, [PK]),
}
DEFAULT_EMIT = ["P1", "P2", "P3", "P5"]


def _member(pid, x, y, conf=0.7, in_refit=True):
    e, n, _ = LAYOUT[pid]
    lat, lng = ll(e, n)
    return {"pano_id": pid, "det_index": 0, "x_normalized": x, "y_normalized": y,
            "confidence": conf, "lat": lat, "lng": lng, "range_m": 5.0, "bearing_deg": 0.0,
            "in_refit": in_refit}


def _site(sid, east, members, n_op):
    lat, lng = ll(east, 0)
    return {"site_id": sid, "lat": lat, "lng": lng, "n_members": len(members),
            "n_panos": len({m["pano_id"] for m in members}), "n_operational": n_op,
            "n_operational_panos": n_op, "best_confidence": 0.9, "members": members}


SITES = [
    _site(1, 0, [_member("P1", 0.25, 0.6), _member("P2", 0.75, 0.62),
                 _member("P3", 0.5, 0.58), _member("P7", 0.4, 0.6),
                 _member("L1", 0.33, 0.6, conf=0.2, in_refit=False)], 4),
    _site(2, 200, [_member("P4", *PK), _member("P6", *PK)], 2),
    _site(3, 100, [_member("Q1", *PK), _member("Q2", *PK), _member("Q3", *PK, conf=0.3)], 3),
    _site(4, 300, [_member("B1", *PK), _member("R1", *PK), _member("R2", *PK)], 3),
    _site(5, 500, [_member("X1", *PK), _member("T1", *PK), _member("T2", *PK)], 3),
]
MINED = [
    {"pano_id": "P5", "site_id": 1, "status": "emit", "emit": True, "x": 0.9, "y": 0.61,
     "peak_conf": 0.3, "band": "8-12 m", "benchmark_pano": False},
    {"pano_id": "P6", "site_id": 2, "status": "operational_present", "emit": False,
     "x": None, "y": None, "peak_conf": None, "band": "0-8 m", "benchmark_pano": False},
    {"pano_id": "Z1", "site_id": 3, "status": "emit", "emit": True, "x": 0.5, "y": 0.6,
     "peak_conf": 0.2, "band": "0-8 m", "benchmark_pano": False},
    {"pano_id": "X1", "site_id": 3, "status": "no_peak", "emit": False, "x": None,
     "y": None, "peak_conf": None, "band": "0-8 m", "benchmark_pano": False},
]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _results_rows():
    rows = []
    for pid, (e, n, peaks) in LAYOUT.items():
        lat, lng = ll(e, n)
        rows.append({
            "detections": [{"x_normalized": x, "y_normalized": y, "confidence": 0.7}
                           for x, y in peaks],
            "pano": {"panorama_id": pid, "width": 400, "height": 200, "lat": lat,
                     "lng": lng, "camera_heading": 270.0, "capture_date": "2024-05",
                     "source": "mapillary", "camera_make": "GoPro",
                     "camera_model": "GoPro Max"}})
    return rows


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(bsl, "BENCHMARK_ROOT", tmp_path / "no_benchmark_here")
    d = tmp_path / "run"
    (d / "panos").mkdir(parents=True)
    _write_jsonl(d / "results.jsonl", _results_rows())
    _write_jsonl(d / "sites.jsonl", SITES)
    _write_jsonl(d / "labels.jsonl", MINED)
    (d / "manifest.json").write_text(json.dumps({"detection_storage_floor": 0.1}),
                                     encoding="utf-8")
    (d / "sites_meta.json").write_text(json.dumps(
        {"params": {"floor": 0.1, "min_confidence": 0.3, "max_range_m": 25.0}}),
        encoding="utf-8")
    rng = np.random.default_rng(0)
    for pid in LAYOUT:
        if pid == "P7":
            continue                                   # no native image
        Image.fromarray(rng.integers(0, 256, (200, 400, 3), dtype=np.uint8)).save(
            d / "panos" / f"{pid}.jpg", quality=90)
    (d / "bench.txt").write_text("# benchmark\nB1\n", encoding="utf-8")
    (d / "exclude.txt").write_text("X1\n", encoding="utf-8")
    return d


def _argv(run, out, *extra):
    return ["--results", str(run / "results.jsonl"), "--sites", str(run / "sites.jsonl"),
            "--mined", str(run / "labels.jsonl"), "--panos", str(run / "panos"),
            "--benchmark-ids", str(run / "bench.txt"),
            "--exclude-ids", str(run / "exclude.txt"), "--city", "lyon",
            "--out", str(out), "--workers", "2", *extra]


def _emitted(out):
    return sorted(p.stem for p in out.glob("*.json") if p.name != bsl.MANIFEST)


def _load(out, pid):
    return json.loads((out / f"{pid}.json").read_text(encoding="utf-8"))


def _manifest(out):
    return json.loads((out / bsl.MANIFEST).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the rules

def test_every_rule_lands_on_its_one_pano(run, tmp_path):
    out = tmp_path / "out"
    assert bsl.main(_argv(run, out)) == 0
    assert _emitted(out) == DEFAULT_EMIT
    for pid in ("P1", "P2", "P3"):
        sup = _load(out, pid)["supervision"]
        assert sup["mode"] == "sparse" and sup["sources"] == ["strong_site"]
        assert sup["site_ids"] == ["lyon:1"]
    assert _load(out, "P1")["curb_ramp_points_normalized"] == [[0.25, 0.6]]
    p5 = _load(out, "P5")
    assert p5["curb_ramp_points_normalized"] == [[0.9, 0.61]]
    assert p5["supervision"]["sources"] == ["mined"]
    assert p5["supervision"]["site_ids"] == ["lyon:mined:1"]   # the miner's own serial


def test_verified_empty_is_opt_in_and_vetoed_by_any_site(run, tmp_path):
    """Review H1: E1 sits 5 m from a weak (2-pano) site and must never be emitted; only E4,
    with no site of any tier within 25 m, qualifies -- and only behind the flag."""
    out = tmp_path / "out"
    bsl.main(_argv(run, out, "--emit-verified-empty"))
    assert _emitted(out) == sorted(DEFAULT_EMIT + ["E4"])
    e4 = _load(out, "E4")
    assert e4["curb_ramp_points_normalized"] == []
    assert e4["supervision"] == {"mode": "verified_empty", "sources": [], "site_ids": [],
                                 "ignore_points": [], "city": "lyon",
                                 "run_sha": bsl.sha256_file(run / "results.jsonl")}
    m = _manifest(out)
    assert m["params"]["verified_empty_rule"] == "proposed, unreviewed"
    assert m["params"]["emit_verified_empty"] is True
    assert not _manifest_flag_default(run, tmp_path)


def _manifest_flag_default(run, tmp_path):
    out = tmp_path / "default"
    bsl.main(_argv(run, out))
    return _manifest(out)["params"]["emit_verified_empty"]


def test_a_members_own_ground_point_vetoes_verified_empty():
    """A site whose centroid is 40 m away but which has a member ground point 10 m from the
    pano still vetoes it."""
    lat, lng = ll(0, 0)
    far = ll(40, 0)
    near = ll(10, 0)
    panos = {"E": {"pano_id": "E", "lat": lat, "lng": lng, "peaks": (), "n_peaks": 0}}
    for i, east in enumerate((3, -3)):
        la, ln = ll(east, 8)
        panos[f"N{i}"] = {"pano_id": f"N{i}", "lat": la, "lng": ln,
                          "peaks": ((0.5, 0.6, 0.7),), "n_peaks": 1}
    site = {"site_id": 9, "lat": far[0], "lng": far[1], "n_operational_panos": 1,
            "members": [{"pano_id": "N0", "x_normalized": 0.5, "y_normalized": 0.6,
                         "confidence": 0.2, "in_refit": False,
                         "lat": near[0], "lng": near[1]}]}
    chosen, _ = bsl.select(panos, [site], [], "x", emit_empty=True)
    assert "E" not in chosen
    site["members"][0]["lat"], site["members"][0]["lng"] = far
    chosen, _ = bsl.select(panos, [site], [], "x", emit_empty=True)
    assert chosen["E"] == ("verified_empty", [], [])


def test_low_confidence_and_support_members_are_not_positives():
    """Review H2: the reviewer's case, a 0.12 support member on a pano with no peaks, plus
    a 0.9 member that fuse_sites marked in_refit false."""
    lat, lng = ll(0, 0)
    panos = {p: {"pano_id": p, "lat": lat, "lng": lng, "peaks": (), "n_peaks": 0}
             for p in ("A", "B", "C", "D", "E")}
    members = [{"pano_id": p, "x_normalized": 0.5, "y_normalized": 0.6, "confidence": 0.7,
                "in_refit": True} for p in ("A", "B", "C")]
    members += [{"pano_id": "D", "x_normalized": 0.33, "y_normalized": 0.6,
                 "confidence": 0.12, "in_refit": False},
                {"pano_id": "E", "x_normalized": 0.33, "y_normalized": 0.6,
                 "confidence": 0.9, "in_refit": False}]
    chosen, stats = bsl.select(panos, [{"site_id": 1, "lat": lat, "lng": lng,
                                        "n_operational_panos": 5, "members": members}],
                               [], "x")
    assert sorted(chosen) == ["A", "B", "C"] and stats["strong_sites"] == 1
    # a member file without in_refit (an older fuse) is taken at its confidence
    for m in members:
        del m["in_refit"]
    chosen, _ = bsl.select(panos, [{"site_id": 1, "lat": lat, "lng": lng,
                                    "n_operational_panos": 5, "members": members}], [], "x")
    assert sorted(chosen) == ["A", "B", "C", "E"]


def test_strong_is_recounted_from_the_qualifying_members(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert not {"Q1", "Q2", "Q3", "L1"} & set(_emitted(out))
    assert _manifest(out)["counts"]["strong_sites"] == 1             # S1 only
    out2 = tmp_path / "out2"
    bsl.main(_argv(run, out2, "--member-min-confidence", "0.3"))
    assert {"Q1", "Q2", "Q3"} <= set(_emitted(out2))
    assert _manifest(out2)["params"]["member_min_confidence"] == 0.3


def test_ignore_points_are_the_panos_unverified_peaks(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert _load(out, "P1")["supervision"]["ignore_points"] == [[0.7, 0.6]]
    assert _load(out, "P2")["supervision"]["ignore_points"] == []
    assert _manifest(out)["counts"]["ignore_points"] == 1


def test_a_peak_within_two_heatmap_pixels_is_the_positive():
    peaks = ((0.5 + 1.5 / 1024, 0.6, 0.7), (0.5 + 3 / 1024, 0.6, 0.7))
    assert bsl.ignore_points(peaks, [(0.5, 0.6, "mined", "x:mined:1")]) == [
        (round(0.5 + 3 / 1024, 6), 0.6)]


# ------------------------------------------------------------ input guards

def test_a_run_without_the_storage_floor_is_refused(run, tmp_path):
    for content in ({"detection_storage_floor": 0.55}, {}):
        (run / "manifest.json").write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(SystemExit, match="detection_storage_floor"):
            bsl.main(_argv(run, tmp_path / "out"))
    (run / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest"):
        bsl.main(_argv(run, tmp_path / "out"))


def test_sites_meta_is_required_recorded_and_checked(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert _manifest(out)["fuse_params"]["min_confidence"] == 0.3
    (run / "sites_meta.json").write_text(json.dumps({"params": {"min_confidence": 0.6}}),
                                         encoding="utf-8")
    with pytest.raises(SystemExit, match="min_confidence 0.6"):
        bsl.main(_argv(run, tmp_path / "o2"))
    (run / "sites_meta.json").unlink()
    with pytest.raises(SystemExit, match="sites meta"):
        bsl.main(_argv(run, tmp_path / "o3"))


def test_benchmark_ids_are_required_when_the_city_has_a_bundle(run, tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    (root / "lyon").mkdir(parents=True)
    monkeypatch.setattr(bsl, "BENCHMARK_ROOT", root)
    argv = _argv(run, tmp_path / "out")
    i = argv.index("--benchmark-ids")
    del argv[i:i + 2]
    with pytest.raises(SystemExit, match="--benchmark-ids is required"):
        bsl.main(argv)
    monkeypatch.setattr(bsl, "BENCHMARK_ROOT", tmp_path / "none")
    assert bsl.main(argv) == 0                          # no bundle for the city: optional


@pytest.mark.parametrize("city, dirs, want", [
    ("lyon", ["Lyon"], ["Lyon"]),                                    # case
    ("laurens", ["laurens_mapillary", "laurens_gsv"], ["laurens_gsv", "laurens_mapillary"]),
    ("laurens_mapillary", ["laurens"], ["laurens"]),                 # either way round
    ("lyon", ["lyonnais", "bayonne"], []),                           # not at a _ boundary
])
def test_benchmark_bundle_matching(tmp_path, monkeypatch, city, dirs, want):
    for d in dirs:
        (tmp_path / d).mkdir()
    monkeypatch.setattr(bsl, "BENCHMARK_ROOT", tmp_path)
    assert bsl.benchmark_bundles_for(city) == want


def test_ambiguous_bundles_name_every_candidate(run, tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    for d in ("lyon_ladybug", "lyon_gopro"):
        (root / d).mkdir(parents=True)
    monkeypatch.setattr(bsl, "BENCHMARK_ROOT", root)
    argv = _argv(run, tmp_path / "out")
    i = argv.index("--benchmark-ids")
    del argv[i:i + 2]
    with pytest.raises(SystemExit, match="lyon_gopro, lyon_ladybug.*several match"):
        bsl.main(argv)


def test_benchmark_and_buffer_drop_whole_sites(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert not {"B1", "R1", "R2", "X1", "T1", "T2", "Z1"} & set(_emitted(out))
    c = _manifest(out)["counts"]
    assert c["dropped_sites_benchmark"] == 1 and c["dropped_sites_excluded"] == 1
    assert c["dropped_mined_sites"] == 1
    # without the buffer file, S5's panos and the miner's Z1 train
    argv = _argv(run, tmp_path / "o2")
    i = argv.index("--exclude-ids")
    del argv[i:i + 2]
    bsl.main(argv)
    assert {"T1", "T2", "X1", "Z1"} <= set(_emitted(tmp_path / "o2"))


def test_benchmark_flag_on_a_mined_row_excludes_the_pano(run, tmp_path):
    rows = [dict(r) for r in MINED]
    rows[0]["benchmark_pano"] = True                    # P5
    _write_jsonl(run / "labels.jsonl", rows)
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert "P5" not in _emitted(out)


def test_select_dedupes_a_point_two_sources_agree_on():
    lat, lng = ll(0, 0)
    panos = {p: {"pano_id": p, "lat": lat, "lng": lng, "peaks": ((0.5, 0.6, 0.7),),
                 "n_peaks": 1} for p in ("A", "B", "C")}
    site = {"site_id": 7, "lat": lat, "lng": lng, "n_operational_panos": 3,
            "members": [{"pano_id": p, "x_normalized": 0.5, "y_normalized": 0.6,
                         "confidence": 0.7} for p in ("A", "B", "C")]}
    mined = [{"pano_id": "A", "site_id": 9, "emit": True, "x": 0.5, "y": 0.6},
             {"pano_id": "A", "site_id": 9, "emit": True, "x": 0.1, "y": 0.6}]
    chosen, _ = bsl.select(panos, [site], mined, "x")
    assert chosen["A"] == ("sparse", [(0.1, 0.6, "mined", "x:mined:9"),
                                      (0.5, 0.6, "strong_site", "x:7")], [])


def test_a_mined_row_for_an_unknown_pano_is_an_error():
    with pytest.raises(SystemExit, match="not in results"):
        bsl.select({}, [], [{"pano_id": "Z", "site_id": 1, "emit": True, "x": 0.1,
                             "y": 0.5}], "x")


def test_city_must_be_a_slug(run, tmp_path):
    argv = _argv(run, tmp_path / "out")
    argv[argv.index("--city") + 1] = "Lyon FR"
    with pytest.raises(SystemExit, match="slug"):
        bsl.main(argv)


def test_grid_cells_are_sized_at_the_poleward_edge():
    """Review L5: a 3x3 read must find a neighbour 24.9 m east at the run's highest
    latitude even when the run spans several degrees."""
    g = bsl.Grid(25.0, lat_ref=60.0)
    lat = 60.0
    east = 24.9 / (111_195.0 * math.cos(math.radians(lat)))
    for k in range(200):                                 # every phase within a cell
        lng = 4.0 + k * 0.00001
        g.cells.clear()
        g.add(lat, lng + east, "n")
        assert list(g.within(lat, lng, 25.0)) == ["n"]


# ------------------------------------------------------------ output and manifest

def test_record_fields_and_image(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    rec = _load(out, "P1")
    lat, lng = ll(0, 5)
    assert rec["pano_id"] == "P1"
    assert rec["pano_coord"] == [round(lat, 6), round(lng, 6)]
    assert rec["pano_azimuth"] == -90.0                      # heading 270 wrapped
    assert rec["pano_meta"]["native_sha256"] == bsl.sha256_file(run / "panos" / "P1.jpg")
    assert rec["pano_meta"]["camera_model"] == "GoPro Max"
    with Image.open(out / "P1.jpg") as im:
        assert im.size == (4096, 2048) and im.format == "JPEG"
    raw = (out / "P1.json").read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n")


def test_manifest_counts_and_hash(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    m = _manifest(out)
    c = m["counts"]
    assert c["emitted"] == 4 and c["by_mode"] == {"sparse": 4, "verified_empty": 0}
    assert c["missing_image"] == 1 and m["missing_image_ids"] == ["P7"]
    assert c["points_by_source"] == {"strong_site": 3, "mined": 1}
    assert c["mined_emit_rows"] == 1
    assert m["params"]["run_detection_storage_floor"] == 0.1
    assert m["inputs"]["results"]["sha256"] == bsl.sha256_file(run / "results.jsonl")
    assert m["inputs"]["exclude_ids"]["name"] == "exclude.txt"
    h = hashlib.sha256()
    for pid in sorted(m["pano_ids"]):
        h.update(f"{pid}.json\n".encode())
        h.update((out / f"{pid}.json").read_bytes())
    assert m["labels_sha256"] == h.hexdigest()


def test_check_passes_then_catches_a_tampered_label(run, tmp_path, capsys):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert bsl.main(_argv(run, out, "--check")) == 0
    rec = _load(out, "P2")
    rec["curb_ramp_points_normalized"] = [[0.5, 0.5]]
    (out / "P2.json").write_text(json.dumps(rec), encoding="utf-8")
    assert bsl.main(_argv(run, out, "--check")) == 1
    assert "P2.json differs" in capsys.readouterr().out


def test_check_catches_a_changed_input(run, tmp_path):
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    _write_jsonl(run / "labels.jsonl", MINED + [
        {"pano_id": "P6", "site_id": 8, "status": "emit", "emit": True, "x": 0.2, "y": 0.6,
         "peak_conf": 0.2, "band": "0-8 m", "benchmark_pano": False}])
    assert bsl.main(_argv(run, out, "--check")) == 1


def test_rebuild_is_byte_identical_and_resumes_under_prune(run, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    bsl.main(_argv(run, a))
    bsl.main(_argv(run, b, "--workers", "1"))
    for name in sorted(os.listdir(a)):
        if name.endswith(".json"):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
    mtime = os.path.getmtime(a / "P1.jpg")
    bsl.main(_argv(run, a, "--prune"))                   # resume: nothing rewritten
    assert os.path.getmtime(a / "P1.jpg") == mtime


def test_a_non_empty_out_needs_prune_and_prune_removes_stale_panos(run, tmp_path):
    """Review M3: a second build with a different --limit sample used to leave the first
    sample's panos on disk, where train.py would read them."""
    out = tmp_path / "out"
    bsl.main(_argv(run, out, "--limit", "2", "--seed", "0"))
    first = _emitted(out)
    with pytest.raises(SystemExit, match="--prune"):
        bsl.main(_argv(run, out, "--limit", "2", "--seed", "3"))
    bsl.main(_argv(run, out, "--limit", "2", "--seed", "3", "--prune"))
    second = _manifest(out)["pano_ids"]
    assert first != second and _emitted(out) == second
    assert sorted(p.stem for p in out.glob("*.jpg")) == second
    assert bsl.main(_argv(run, out, "--limit", "2", "--seed", "3", "--check")) == 0


@pytest.mark.parametrize("intruder, content", [
    ("G1.json", json.dumps({"curb_ramp_points_normalized": []})),     # a dense pano
    ("G2.jpg", "native pano bytes"),                                  # a lone jpg (R1)
    ("notes.txt", "x")])
def test_prune_refuses_what_it_did_not_write(run, tmp_path, intruder, content):
    out = tmp_path / "out"
    out.mkdir()
    (out / intruder).write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to prune"):
        bsl.main(_argv(run, out, "--prune"))
    assert (out / intruder).read_text(encoding="utf-8") == content


def test_prune_takes_a_lone_jpg_the_previous_manifest_listed(run, tmp_path):
    """A build killed between a pano's image and its JSON leaves a lone jpg that IS ours:
    the previous manifest lists it, so the prune may remove it."""
    out = tmp_path / "out"
    bsl.main(_argv(run, out, "--limit", "2", "--seed", "0"))
    gone = _manifest(out)["pano_ids"][0]
    (out / f"{gone}.json").unlink()
    bsl.main(_argv(run, out, "--limit", "2", "--seed", "3", "--prune"))
    assert sorted(p.stem for p in out.glob("*.jpg")) == _manifest(out)["pano_ids"]


@pytest.mark.parametrize("where", ["panos", "panos/sub", "index_dir"])
def test_out_inside_the_native_panos_is_refused(run, tmp_path, where):
    """Re-review R1: --out pointed at the --panos directory overwrote 4 native panos and a
    prune deleted the other 20. Refused before anything is written, with or without --prune."""
    before = {p.name: p.read_bytes() for p in (run / "panos").iterdir()}
    argv = _argv(run, run / "panos" / "sub" if where == "panos/sub" else run / "panos",
                 "--prune")
    if where == "index_dir":
        idx = run / "index.csv"
        idx.write_text("panorama_id,filename,bytes,sha256\n", encoding="utf-8")
        argv[argv.index("--panos") + 1] = str(idx)
        argv[argv.index("--out") + 1] = str(run)
    with pytest.raises(SystemExit, match="native pano directory"):
        bsl.main(argv)
    assert {p.name: p.read_bytes() for p in (run / "panos").iterdir()} == before


def test_limit_is_a_seeded_subset(run, tmp_path):
    outs = {}
    for name, seed in (("s0", 0), ("s0b", 0), ("s1", 1)):
        out = tmp_path / name
        bsl.main(_argv(run, out, "--limit", "3", "--seed", str(seed)))
        outs[name] = _manifest(out)["pano_ids"]
    assert outs["s0"] == outs["s0b"] and len(outs["s0"]) == 3
    assert outs["s0"] == sorted(random.Random(0).sample(DEFAULT_EMIT, 3))


def test_index_csv_input_and_its_sha_is_enforced(run, tmp_path):
    idx = run / "index.csv"
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panorama_id", "filename", "bytes", "sha256"])
        for p in sorted((run / "panos").glob("*.jpg")):
            sha = bsl.sha256_file(p) if p.stem != "P3" else "0" * 64
            w.writerow([p.stem, p.name, p.stat().st_size, sha])
    argv = _argv(run, tmp_path / "out")
    argv[argv.index("--panos") + 1] = str(idx)
    with pytest.raises(SystemExit, match="sha256"):
        bsl.main(argv)


def test_output_trains_through_the_masked_dataset(run, tmp_path):
    """End to end with train.py's own dataset class, lifted: a sparse pano's weight is its
    positive's disc with the ignore point's side carved out; verified_empty is all ones on
    an empty target."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torchvision.transforms.functional as F
    from torchvision import transforms
    sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
    from check_lr_schedule_135 import load_from_train_py
    from rampnet import augment as A
    mod = load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points", "generate_weight_from_points",
        "supervision_mode", "SUPERVISION_MODES", "_point_list", "_nearest_dist_sq",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)
    bsl.main(_argv(run, tmp_path / "ds" / "train", "--emit-verified-empty"))
    ds = mod.EquiHeatmapDataset(
        root_dir=str(tmp_path / "ds"), split="train", target_heatmap_shape=(512, 1024),
        transform_input=transforms.Compose([transforms.Resize((64, 128)),
                                            transforms.ToTensor()]),
        points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
        apply_horizontal_flip=False, loss_mask=True)
    got = {os.path.basename(p)[:-4]: ds[i] for i, p in enumerate(ds.image_paths)}
    assert sorted(got) == sorted(DEFAULT_EMIT + ["E4"])
    _, hm, w = got["E4"]
    assert hm.sum() == 0 and torch.equal(w, torch.ones_like(w))
    _, hm, w = got["P1"]
    assert 2700 < w.sum() < 2900 and w[0, 307, 256] == 1 and w[0, 0, 0] == 0
    assert hm.max() == 1
    # P1's ignore point at x=0.7 (717 px) is 461 px away: no overlap, so a full disc
    want = mod.generate_weight_from_points([(0.25, 0.6)], heatmap_shape=(512, 1024))
    assert torch.equal(w, want)
