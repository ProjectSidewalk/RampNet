"""mined_placement_158: the #158 pair list carries no answer, and the committed files agree."""
import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import mined_placement_158 as MP  # noqa: E402


def _src(city, site, pano, src="s1"):
    return {"city": city, "site_id": str(site), "pano_id": pano, "src_pano": src,
            "src_det_index": "0", "src_x": "0.5", "src_y": "0.6", "src_conf": "0.9",
            "baseline_m": "7.0", "src_range_m": "9.0", "oth_range_m": "11.0",
            "proj_x": "0.4", "proj_y": "0.58"}


def test_build_pairs_has_no_answer_columns_and_keys_round_trip():
    pairs, keys = MP.build_pairs([_src("bend", 3, "t2"), _src("richmond", 7, "t1")],
                                 lambda c, p: "2024-05")
    assert [p["city"] for p in pairs] == ["richmond", "bend"]      # harness city order
    assert all(not k.startswith("ref_") for p in pairs for k in p)
    assert set(pairs[0]) == set(MP.COLUMNS)
    assert pairs[0]["ramp_uid"] == "richmond:7:t1"                  # one corner per pair
    assert [(k["pair_id"], k["city"], k["site_id"], k["pano_id"]) for k in keys] == \
        [("m000", "richmond", 7, "t1"), ("m001", "bend", 3, "t2")]


def test_read_pairs_refuses_answer_columns(tmp_path):
    p = tmp_path / "pairs.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MP.COLUMNS + ["ref_x"])
        w.writerow(["0"] * (len(MP.COLUMNS) + 1))
    with pytest.raises(SystemExit, match="answer columns"):
        MP.read_pairs(str(p))


@pytest.mark.skipif(not os.path.exists(MP.PAIRS_CSV), reason="committed outputs absent")
def test_committed_outputs_are_consistent():
    pairs = MP.read_pairs()
    with open(MP.KEYS_CSV, encoding="utf-8", newline="") as f:
        keys = list(csv.DictReader(f))
    assert [p["pair_id"] for p in pairs] == [k["pair_id"] for k in keys]
    m = MP.load_manifest()                                  # pins pairs.csv by sha256
    assert {c["ramp_uid"] for c in m["corners"]} == {p["ramp_uid"] for p in pairs}
    want = {(k["city"], int(k["site_id"]), k["pano_id"]) for k in keys}
    for name in os.listdir(MP.PRED_DIR):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(MP.PRED_DIR, name), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert {(r["city"], r["site_id"], r["pano_id"]) for r in rows} == want, name
        meta = json.load(open(os.path.join(MP.PRED_DIR, name[:-6] + ".meta.json")))
        assert meta["pairs_sha256"] == m["pairs_sha256"]
