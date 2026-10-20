"""scripts/build_sparse_labels.py (#263) on a synthetic labeler run.

The run is laid out in metres around one strong site, so every rule is exercised by
exactly one pano and a wrong rule changes exactly one assertion:

    S1  strong site (3 operational panos) at the origin
        P1, P2, P3   its members                          -> sparse, strong_site
        B1           its member, but a benchmark pano      -> excluded
        P7           its member, but no native image       -> missing_image
        P5           mined from S1 (emit true)             -> sparse, mined
        E2           zero peaks, 14 m from S1              -> skipped (strong site nearby)
    200 m east:
        S2  weak site (2 operational panos); P4 its member -> skipped
        P6  peaks; a mined row with emit false             -> skipped
        E1  zero peaks; P4 and P6 within 25 m, no strong site -> verified_empty
    400 m east:
        E3  zero peaks, one peak-carrying neighbour (P8)   -> skipped

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


LAYOUT = {   # pano: (east m, north m, number of stored peaks)
    "P1": (0, 5, 2), "P2": (0, -5, 1), "P3": (5, 0, 1), "B1": (0, 10, 1), "P7": (3, 3, 1),
    "P5": (-8, 0, 1), "E2": (10, 10, 0),
    "P4": (200, 5, 1), "P6": (205, 0, 2), "E1": (200, -5, 0),
    "E3": (400, 0, 0), "P8": (405, 0, 1),
}


def _results(path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for pid, (e, n, k) in LAYOUT.items():
            lat, lng = ll(e, n)
            f.write(json.dumps({
                "detections": [{"x_normalized": 0.1 * (i + 1), "y_normalized": 0.6,
                                "confidence": 0.7} for i in range(k)],
                "pano": {"panorama_id": pid, "width": 400, "height": 200, "lat": lat,
                         "lng": lng, "camera_heading": 270.0, "capture_date": "2024-05",
                         "source": "mapillary", "camera_make": "GoPro",
                         "camera_model": "GoPro Max"}}) + "\n")


def _member(pid, x, y, conf=0.7):
    return {"pano_id": pid, "det_index": 0, "x_normalized": x, "y_normalized": y,
            "confidence": conf, "range_m": 5.0, "bearing_deg": 0.0}


def _sites(path):
    s1_lat, s1_lng = ll(0, 0)
    s2_lat, s2_lng = ll(200, 0)
    rows = [
        {"site_id": 1, "lat": s1_lat, "lng": s1_lng, "n_members": 5, "n_panos": 5,
         "n_operational": 5, "n_operational_panos": 3, "best_confidence": 0.9,
         "members": [_member("P1", 0.25, 0.6), _member("P2", 0.75, 0.62),
                     _member("P3", 0.5, 0.58), _member("B1", 0.3, 0.6),
                     _member("P7", 0.4, 0.6)]},
        {"site_id": 2, "lat": s2_lat, "lng": s2_lng, "n_members": 2, "n_panos": 2,
         "n_operational": 2, "n_operational_panos": 2, "best_confidence": 0.8,
         "members": [_member("P4", 0.1, 0.6)]},
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mined(path):
    rows = [
        {"pano_id": "P5", "site_id": 1, "status": "emit", "emit": True, "x": 0.9,
         "y": 0.61, "peak_conf": 0.3, "band": "8-12 m", "benchmark_pano": False},
        {"pano_id": "P6", "site_id": 2, "status": "operational_present", "emit": False,
         "x": None, "y": None, "peak_conf": None, "band": "0-8 m", "benchmark_pano": False},
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "run"
    (d / "panos").mkdir(parents=True)
    _results(d / "results.jsonl")
    _sites(d / "sites.jsonl")
    _mined(d / "labels.jsonl")
    rng = np.random.default_rng(0)
    for pid in LAYOUT:
        if pid == "P7":
            continue                                   # no native image
        Image.fromarray(rng.integers(0, 256, (200, 400, 3), dtype=np.uint8)).save(
            d / "panos" / f"{pid}.jpg", quality=90)
    (d / "bench.txt").write_text("# benchmark\nB1\n", encoding="utf-8")
    return d


def _argv(run, out, *extra):
    return ["--results", str(run / "results.jsonl"), "--sites", str(run / "sites.jsonl"),
            "--mined", str(run / "labels.jsonl"), "--panos", str(run / "panos"),
            "--benchmark-ids", str(run / "bench.txt"), "--city", "lyon",
            "--out", str(out), "--workers", "2", *extra]


def _load(out, pid):
    return json.loads((out / f"{pid}.json").read_text(encoding="utf-8"))


def test_every_rule_lands_on_its_one_pano(run, tmp_path):
    out = tmp_path / "out"
    assert bsl.main(_argv(run, out)) == 0
    emitted = sorted(p.stem for p in out.glob("*.json") if p.name != bsl.MANIFEST)
    assert emitted == ["E1", "P1", "P2", "P3", "P5"]
    for pid in ("P1", "P2", "P3"):
        rec = _load(out, pid)
        assert rec["supervision"]["mode"] == "sparse"
        assert rec["supervision"]["sources"] == ["strong_site"]
        assert rec["supervision"]["site_ids"] == ["lyon:1"]
    assert _load(out, "P1")["curb_ramp_points_normalized"] == [[0.25, 0.6]]
    p5 = _load(out, "P5")
    assert p5["curb_ramp_points_normalized"] == [[0.9, 0.61]]
    assert p5["supervision"]["sources"] == ["mined"]
    e1 = _load(out, "E1")
    assert e1["curb_ramp_points_normalized"] == []
    assert e1["supervision"] == {"mode": "verified_empty", "sources": [], "site_ids": [],
                                 "run_sha": bsl.sha256_file(run / "results.jsonl"),
                                 "city": "lyon"}


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
    m = json.loads((out / bsl.MANIFEST).read_text(encoding="utf-8"))
    c = m["counts"]
    assert c["emitted"] == 5 and c["by_mode"] == {"sparse": 4, "verified_empty": 1}
    assert c["excluded_benchmark"] == 1 and c["missing_image"] == 1
    assert m["missing_image_ids"] == ["P7"]
    assert c["points_by_source"] == {"strong_site": 3, "mined": 1}
    assert c["strong_sites"] == 1 and c["mined_emit_rows"] == 1
    assert m["params"]["verified_empty_rule"] == "proposed"
    assert m["inputs"]["results"]["sha256"] == bsl.sha256_file(run / "results.jsonl")
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
    with open(run / "labels.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"pano_id": "P6", "site_id": 1, "status": "emit", "emit": True,
                            "x": 0.2, "y": 0.6, "peak_conf": 0.2, "band": "0-8 m",
                            "benchmark_pano": False}) + "\n")
    assert bsl.main(_argv(run, out, "--check")) == 1


def test_rebuild_is_byte_identical_and_resumes(run, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    bsl.main(_argv(run, a))
    bsl.main(_argv(run, b, "--workers", "1"))
    for name in sorted(os.listdir(a)):
        if name.endswith(".json"):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
    mtime = os.path.getmtime(a / "P1.jpg")
    bsl.main(_argv(run, a))                              # resume: nothing rewritten
    assert os.path.getmtime(a / "P1.jpg") == mtime


def test_limit_is_a_seeded_subset(run, tmp_path):
    outs = {}
    for name, seed in (("s0", 0), ("s0b", 0), ("s1", 1)):
        out = tmp_path / name
        bsl.main(_argv(run, out, "--limit", "3", "--seed", str(seed)))
        outs[name] = json.loads((out / bsl.MANIFEST).read_text(encoding="utf-8"))["pano_ids"]
    assert outs["s0"] == outs["s0b"] and len(outs["s0"]) == 3
    assert set(outs["s0"]) <= {"E1", "P1", "P2", "P3", "P5"}
    assert outs["s0"] == sorted(random.Random(0).sample(["E1", "P1", "P2", "P3", "P5"], 3))


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


def test_benchmark_flag_on_a_mined_row_excludes_the_pano(run, tmp_path):
    rows = [json.loads(ln) for ln in open(run / "labels.jsonl", encoding="utf-8")]
    rows[0]["benchmark_pano"] = True                    # P5
    with open(run / "labels.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    bsl.main(_argv(run, out))
    assert not (out / "P5.json").exists()


def test_select_dedupes_a_point_two_sources_agree_on():
    lat, lng = ll(0, 0)
    panos = {"A": {"pano_id": "A", "lat": lat, "lng": lng, "n_peaks": 1}}
    site = {"site_id": 7, "lat": lat, "lng": lng, "n_operational_panos": 3,
            "members": [_member("A", 0.5, 0.6)]}
    mined = [{"pano_id": "A", "site_id": 9, "emit": True, "x": 0.5, "y": 0.6},
             {"pano_id": "A", "site_id": 9, "emit": True, "x": 0.1, "y": 0.6}]
    chosen, _, _ = bsl.select(panos, [site], mined, "x")
    assert chosen["A"] == ("sparse", [(0.1, 0.6, "mined", "x:9"),
                                      (0.5, 0.6, "strong_site", "x:7")])


def test_a_mined_row_for_an_unknown_pano_is_an_error():
    with pytest.raises(SystemExit, match="not in results"):
        bsl.select({}, [], [{"pano_id": "Z", "site_id": 1, "emit": True, "x": 0.1,
                             "y": 0.5}], "x")


def test_refuses_to_overwrite_a_dense_pano(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "P1.json").write_text(json.dumps({"curb_ramp_points_normalized": []}),
                                 encoding="utf-8")
    with pytest.raises(SystemExit, match="not a sparse-label JSON"):
        bsl.main(_argv(run, out))


def test_city_must_be_a_slug(run, tmp_path):
    argv = _argv(run, tmp_path / "out")
    argv[argv.index("--city") + 1] = "Lyon FR"
    with pytest.raises(SystemExit, match="slug"):
        bsl.main(argv)


def test_output_trains_through_the_masked_dataset(run, tmp_path):
    """End to end with train.py's own dataset class, lifted: the sparse pano's weight is
    its disc, the verified_empty pano's weight is all ones on an empty target."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torchvision.transforms.functional as F
    from torchvision import transforms
    sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
    from check_lr_schedule_135 import load_from_train_py
    from rampnet import augment as A
    mod = load_from_train_py(
        "EquiHeatmapDataset", "generate_heatmap_from_points", "generate_weight_from_points",
        "supervision_mode", "SUPERVISION_MODES",
        os=os, json=json, math=math, np=np, torch=torch, Image=Image, F=F,
        random=random, rank=0, A=A, Dataset=torch.utils.data.Dataset)
    bsl.main(_argv(run, tmp_path / "ds" / "train"))
    ds = mod.EquiHeatmapDataset(
        root_dir=str(tmp_path / "ds"), split="train", target_heatmap_shape=(512, 1024),
        transform_input=transforms.Compose([transforms.Resize((64, 128)),
                                            transforms.ToTensor()]),
        points_to_heatmap_transform_fn=mod.generate_heatmap_from_points,
        apply_horizontal_flip=False, loss_mask=True)
    got = {os.path.basename(p)[:-4]: ds[i] for i, p in enumerate(ds.image_paths)}
    assert sorted(got) == ["E1", "P1", "P2", "P3", "P5"]
    _, hm, w = got["E1"]
    assert hm.sum() == 0 and torch.equal(w, torch.ones_like(w))
    _, hm, w = got["P1"]
    assert 2700 < w.sum() < 2900 and w[0, 307, 256] == 1 and w[0, 0, 0] == 0
    assert hm.max() == 1
