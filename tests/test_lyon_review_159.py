"""The Lyon label audit bundle (#159, #263): staging, the never-train lists, the plan, the
artifact reads, the overlay measurement, the per-group scorer and the gallery options.

CPU only, committed inputs only. Lyon is not a benchmark split and has no verdicts yet;
the scorer is exercised on synthetic verdicts. The byte check against the panos runs only
where ``benchmark/lyon/panos/`` exists (a developer checkout).
"""
import collections
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import lyon_review_159 as L  # noqa: E402

BUNDLE = os.path.join(REPO, "benchmark", "lyon")
PLAN = os.path.join(BUNDLE, L.PLAN_NAME)


# --------------------------------------------------------------------------- #
# the staged bundle, the never-train lists and the plan
# --------------------------------------------------------------------------- #
def test_the_staged_records_verify():
    problems, notes, facts = L.verify_bundle(BUNDLE, require_imagery=False)
    assert problems == [], problems
    assert facts["panos"] == 200
    assert facts["strata"] == {"empty": 25, "random": 170, "top": 5}
    assert facts["detections_055"] == 338 and facts["run_peaks_030"] == 555


def test_verify_fails_until_the_imagery_is_staged():
    # Review of #264, finding 8: "verify: OK" must mean the imagery is pinned.
    if os.path.isdir(os.path.join(BUNDLE, "panos")):
        pytest.skip("imagery staged in this checkout")
    problems, _, _ = L.verify_bundle(BUNDLE)
    assert any("imagery not staged" in p for p in problems)


def test_the_records_show_only_the_055_tier():
    with open(os.path.join(BUNDLE, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    assert "record_confidence" not in sample          # the default, 0.55
    assert all(d["confidence"] >= 0.55 for r in L.load_records(BUNDLE) for d in r["detections"])


def test_every_record_is_in_one_of_the_four_audit_groups_with_its_floor():
    recs = L.load_records(BUNDLE)
    assert {r["benchmark_stratum"] for r in recs} == set(L.GROUPS)
    by = collections.Counter((r["benchmark_stratum"], r["benchmark_group"]) for r in recs)
    for g in L.GROUPS:
        assert by[(g, "random")] >= 30 and by[(g, "empty")] >= 3


def test_ids_txt_is_the_sorted_record_ids_and_the_buffer_contains_them():
    ids = [r["pano"]["panorama_id"] for r in L.load_records(BUNDLE)]
    with open(os.path.join(BUNDLE, "ids.txt"), "rb") as f:
        raw = f.read()
    assert b"\r" not in raw and raw.decode("utf-8").splitlines() == sorted(ids)
    excl = L._read_lines(os.path.join(BUNDLE, "exclude_ids.txt"))
    assert excl == sorted(excl) and set(ids) <= set(excl) and len(excl) > len(ids)


def test_the_plan_reproduces_byte_for_byte_and_carries_the_bar():
    with open(PLAN, "rb") as f:
        assert f.read() == L._dumps(L.build_plan(BUNDLE)).encode("utf-8")
    plan = L.build_plan(BUNDLE)
    bar = plan["pass_bar"]
    assert bar["threshold"] == 0.80 and ">= 0.80" in bar["rule"]
    assert len(bar["intervals"]) == 2 and "not a separate outcome" in bar["decisive"]
    assert "both upper bounds" in bar["decisive_fail"] and "no verdict" in bar["all_row"]
    assert "lower-confidence" in bar["duplicates"]
    assert "held out of every training pool" in plan["never_train"]
    assert plan["sampler"]["seed"] == 159 and plan["sampler"]["groups"]["top"] == 5
    assert plan["sampler_floors"] == {"group_min": 30, "empty_group_min": 3}


def test_posted_mismatches_catches_records_and_code_drift(tmp_path, monkeypatch):
    # Re-review N2: score checks the posted records hash too, and that the code still
    # rebuilds the committed plan.
    import shutil
    b = tmp_path / "lyon"
    shutil.copytree(BUNDLE, b, ignore=shutil.ignore_patterns("panos", "gallery"))
    assert L.posted_mismatches(str(b)) == []
    monkeypatch.setitem(L.PASS_BAR, "threshold", 0.75)
    assert any("does not rebuild" in m for m in L.posted_mismatches(str(b)))
    monkeypatch.undo()
    lines = (b / "records.jsonl").read_bytes().splitlines(keepends=True)
    (b / "records.jsonl").write_bytes(b"".join(lines[1:]))
    assert any(m.startswith("records.jsonl hashes") for m in L.posted_mismatches(str(b)))


def test_the_posted_pin_matches_the_committed_plan_and_records():
    # Review of #264, finding 3: the plan in force when the review opens is the posted one.
    with open(os.path.join(BUNDLE, "bundle_provenance.json"), encoding="utf-8") as f:
        posted = json.load(f)["posted"]
    assert posted["audit_plan_sha256"] == L._sha256(PLAN)
    assert posted["records_jsonl_sha256"] == L._sha256(os.path.join(BUNDLE, "records.jsonl"))
    assert "issuecomment" in posted["comment"]


def test_check_compares_bytes_so_crlf_drift_fails(tmp_path):
    # Review of #264, finding 7.
    path = tmp_path / "x.json"
    obj = {"a": 1}
    assert L._check_or_write(obj, str(path), True) == 0
    assert L._check_or_write(obj, str(path), False) == 0
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert L._check_or_write(obj, str(path), False) == 1


def test_no_email_reaches_any_committed_bundle_file():
    for name in L.TEXT_FILES:
        path = os.path.join(BUNDLE, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                assert not L.EMAIL_RE.search(f.read()), name


def test_mask_email_handles_non_ascii_and_spares_the_model_id():
    assert L.mask_email('"producer": "josé.n@ign.fr"') == '"producer": "j***@***"'
    assert L.mask_email("rampnet-model@606a11956743") == "rampnet-model@606a11956743"


def test_stage_masks_every_file_and_writes_the_never_train_lists(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    rec = lambda pid, prod, lat: {"pano": {"panorama_id": pid, "copyright": prod,
                                           "lat": lat, "lng": 4.8},
                                  "detections": [], "benchmark_group": "empty",
                                  "benchmark_stratum": "other_cameras"}
    (src / "records.jsonl").write_text(json.dumps(rec("A", "a.b@example.org", 45.0)) + "\n"
                                       + json.dumps(rec("B", "grand lyon", 45.1)) + "\n",
                                       encoding="utf-8")
    (src / "sample.json").write_text('{"strata": {"Max|a.b@example.org": 1}}',
                                     encoding="utf-8")
    (src / "group_map.json").write_text("[]", encoding="utf-8")
    run = [rec("A", "x", 45.0), rec("B", "x", 45.1),
           rec("near", "x", 45.0 + 20 / 111320), rec("far", "x", 45.0 + 60 / 111320)]
    run[0]["detections"] = [{"x_normalized": 0.5, "y_normalized": 0.6, "confidence": 0.2}]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in run), encoding="utf-8")
    out = L.stage(str(src), str(results), str(dst), peaks_path=str(tmp_path / "peaks.json"))
    assert out["ids"] == 2 and out["exclude_ids"] == 3
    for name in ("records.jsonl", "sample.json"):
        text = (dst / name).read_text(encoding="utf-8")
        assert "example.org" not in text and "a***@***" in text
    assert (dst / "ids.txt").read_text(encoding="utf-8") == "A\nB\n"
    assert (dst / "exclude_ids.txt").read_text(encoding="utf-8") == "A\nB\nnear\n"
    assert not (dst / "run_peaks.json").exists()       # kept out of the bundle (N4)
    peaks = json.loads((tmp_path / "peaks.json").read_text(encoding="utf-8"))["panos"]
    assert peaks["A"][0]["confidence"] == 0.2 and peaks["B"] == []


def test_exclusion_matches_brute_force_haversine():
    import random
    rng = random.Random(4)
    audited = {f"a{i}": (45.75 + rng.uniform(0, 0.01), 4.84 + rng.uniform(0, 0.01))
               for i in range(20)}
    run = [(f"r{i}", 45.75 + rng.uniform(0, 0.01), 4.84 + rng.uniform(0, 0.01))
           for i in range(3000)]
    got = L.exclusion_ids(run, audited, 30)
    want = set(audited) | {pid for pid, lat, lng in run
                           if any(L.haversine_m(lat, lng, a, b) <= 30
                                  for a, b in audited.values())}
    assert got == want and len(want) > len(audited)


def test_the_group_map_names_the_four_groups_of_the_decision():
    with open(os.path.join(BUNDLE, "group_map.json"), encoding="utf-8") as f:
        names = [name for _, name in json.load(f)]
    assert set(names) == set(L.GROUPS)


# --------------------------------------------------------------------------- #
# the artifact reads
# --------------------------------------------------------------------------- #
def _rec(pid, seq, dets, group="grand_lyon_ladybug", stratum="random"):
    return {"pano": {"panorama_id": pid, "sequence_id": seq},
            "detections": [{"x_normalized": x, "y_normalized": y, "confidence": c}
                           for x, y, c in dets],
            "benchmark_stratum": group, "benchmark_group": stratum}


def test_fixed_cells_count_panos_per_sequence_not_peaks():
    run = [_rec(f"p{i}", "S", [(0.503, 0.756, 0.4), (0.503, 0.756, 0.35)]) for i in range(5)]
    run += [_rec("q", "T", [(0.503, 0.756, 0.9)])]
    cells = L.sequence_cells(run, {"S", "T"})
    assert cells["S"][L.cell_of({"x_normalized": 0.503, "y_normalized": 0.756})] == 5
    art = L.build_artifacts([run[0], run[-1]], cells, "sha")
    row = art["per_group"]["grand_lyon_ladybug"]
    assert row["fixed_cell_030"] == 2 and row["fixed_cell_055"] == 0   # 'q' is alone in T
    assert row["peaks030"] == 3 and row["peaks055"] == 1


def test_above_horizon_is_y_below_half():
    art = L.build_artifacts([_rec("a", "S", [(0.2, 0.48, 0.6), (0.2, 0.6, 0.6)],
                                  group="ecartip_gopro_max")], {}, "sha")
    row = art["per_group"]["ecartip_gopro_max"]
    assert row["above_horizon_055"] == 1 and row["peaks055"] == 2
    assert art["items"][0]["above_horizon"] is True


def test_committed_artifacts_cover_the_bundle_peaks():
    with open(L.ARTIFACTS, encoding="utf-8") as f:
        art = json.load(f)
    assert art["results_jsonl_sha256"].startswith("60e688b2")
    assert sum(r["peaks030"] for r in art["per_group"].values()) == 555
    assert "DO NOT open" in art["what"]


# --------------------------------------------------------------------------- #
# the overlay measurement
# --------------------------------------------------------------------------- #
def test_band_top_walks_up_and_none_without_an_overlay():
    frac = [0.0] * 400 + [1.0] * 112
    assert L.band_top_from_rows(frac) == 400 / 512
    assert L.band_top_from_rows([0.0] * 512) is None


def test_measure_band_finds_a_black_nadir(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (2048, 1024), (120, 130, 110))
    im.paste((0, 0, 0), (0, 830, 2048, 1024))
    path = tmp_path / "p.jpg"
    im.save(path, quality=95)
    kind, y = L.measure_band(str(path))
    assert kind == "black" and abs(y - 830 / 1024) < 2 / 512


def test_measure_band_none_on_plain_imagery(tmp_path):
    from PIL import Image
    path = tmp_path / "p.jpg"
    Image.new("RGB", (2048, 1024), (120, 130, 110)).save(path)
    assert L.measure_band(str(path)) == (None, None)


# --------------------------------------------------------------------------- #
# the scorer (synthetic verdicts; Lyon has none)
# --------------------------------------------------------------------------- #
class _Args:
    bundle = BUNDLE
    write = False
    lenient_duplicates = False
    partial = False
    allow_unposted_plan = False


def test_score_refuses_without_verdicts(capsys):
    if os.path.exists(os.path.join(BUNDLE, "verdicts.json")):
        pytest.skip("the review has happened")
    assert L.cmd_score(_Args) == 1
    assert "nothing to score" in capsys.readouterr().out


def _toy():
    recs = [_rec(f"g{i}", "S", [(0.1, 0.6, 0.9)]) for i in range(10)]
    recs += [_rec(f"e{i}", "T", [(0.1, 0.6, 0.9)], group="ecartip_gopro_max")
             for i in range(10)]
    recs += [_rec("top", "S", [(0.1, 0.6, 0.9)], stratum="top")]
    peaks = {r["pano"]["panorama_id"]: list(r["detections"]) for r in recs}
    # a 0.40 peak on each Ladybug pano, where the reviewer marks a missed ramp on half
    for i in range(10):
        peaks[f"g{i}"].append({"x_normalized": 0.3, "y_normalized": 0.6, "confidence": 0.4})
    v = {f"g{i}": {"dets": [True],
                   "missed": [{"x": 0.3, "y": 0.6}] if i < 5 else [], "no_missed": i >= 5}
         for i in range(10)}
    v.update({f"e{i}": {"dets": [i < 7], "missed": [{"x": 0.5, "y": 0.6}] if i == 0 else [],
                        "no_missed": i != 0} for i in range(10)})
    v["top"] = {"dets": [False], "missed": [], "no_missed": True}     # excluded
    return recs, v, peaks


def test_score_groups_reads_055_from_verdicts_and_030_from_the_run_peaks():
    recs, v, peaks = _toy()
    out, warnings = L.score_groups(recs, v, peaks)
    assert warnings == []
    gl = out["grand_lyon_ladybug"]
    assert gl["precision_055"]["p"] == 1.0 and gl["precision_055"]["n"] == 10
    assert gl["recall_055"]["tp"] == 10 and gl["recall_055"]["n"] == 15
    # 0.30: the 0.40 peak on a marked missed ramp is a TP, elsewhere an FP.
    assert gl["precision_030"]["tp"] == 15 and gl["precision_030"]["n"] == 20
    assert gl["recall_030"]["tp"] == 15 and gl["recall_030"]["n"] == 15
    # 10 of 10 is eligible but not decisive: Wilson's lower bound is 0.72.
    assert gl["eligible"] is True and gl["decisive"] is False
    assert gl["bar"].startswith("eligible, not decisive")
    ec = out["ecartip_gopro_max"]
    assert ec["precision_055"]["p"] == 0.7 and ec["recall_055"]["n"] == 8
    assert ec["eligible"] is False and ec["bar"].startswith("not eligible")
    assert out["all"]["precision_055"]["n"] == 20                       # top excluded
    assert "no verdict" in out["all"]["bar"]


def test_bar_reading_eligibility_is_the_point_estimate():
    # Re-review N1: eligibility is p >= 0.80; decisiveness is information beside it.
    r = L.bar_reading(0.9, [(0.82, 0.96), (0.81, 0.95)], 50)
    assert r["eligible"] and r["decisive"] and "top_up" not in r
    r = L.bar_reading(0.85, [(0.82, 0.92), (0.75, 0.93)], 50)     # Wilson straddles
    assert r["eligible"] and not r["decisive"] and r["reading"].startswith("eligible, not")
    # 50 * (0.10 / 0.05)^2 - 50 = 150 more random panos, from the wider interval.
    assert r["top_up"]["extra_random_panos"] == 150
    assert L.bar_reading(0.6, [(0.5, 0.75), (0.48, 0.74)], 50)["reading"].startswith(
        "decisive fail")
    r = L.bar_reading(0.75, [(0.6, 0.78), (0.62, 0.85)], 50)      # one upper bound >= 0.80
    assert not r["eligible"] and not r["decisive"]
    assert L.bar_reading(None, [], 0)["reading"] == "no detections judged"


def test_top_up_is_capped_by_the_pool_and_has_no_finite_size_at_the_bar():
    # Re-review N3: near the bar the formula explodes; cap it and say so.
    r = L.bar_reading(0.804, [(0.70, 0.90), (0.71, 0.89)], 59, pool_left=17026)
    assert r["top_up"].startswith("no finite top-up")
    r = L.bar_reading(0.83, [(0.70, 0.93), (0.71, 0.92)], 30, pool_left=100)
    assert r["top_up"].startswith("not reachable")
    r = L.bar_reading(0.86, [(0.78, 0.93), (0.77, 0.92)], 30, pool_left=1000, rate=1.7)
    assert r["top_up"] == {"extra_random_panos": 38, "extra_detections_at_current_rate": 65}


def test_bootstrap_is_wider_than_wilson_when_detections_cluster():
    units = [(5, 5)] * 8 + [(0, 5)] * 2          # 40 / 50 = 0.8, all on 10 panos
    lo, hi = L.bootstrap_ci(units)
    wlo, whi = L.wilson(40, 50)
    assert lo < wlo and hi > whi
    assert L.bootstrap_ci(units) == (lo, hi)     # seeded


def test_score_refuses_misaligned_or_partly_judged_panos():
    # Review of #264, finding 6: collect() used to skip these silently.
    recs, v, _ = _toy()
    v["g0"] = {"dets": [True, False], "missed": [], "no_missed": True}
    v["g1"] = {"dets": [None], "missed": [], "no_missed": True}
    del v["g2"]
    problems = L._scan_entries(recs, v)
    assert len(problems) == 3
    assert L._scan_entries(recs, _toy()[1]) == []


# --------------------------------------------------------------------------- #
# the gallery names the overlay and can hide the producer
# --------------------------------------------------------------------------- #
def test_gallery_labels_a_black_nadir(tmp_path):
    import gt_gallery as G
    (tmp_path / "nadir_band.json").write_text(json.dumps({"panos": {
        "A": {"band_top_y": 0.81, "kind": "black", "label": "black nadir (no camera)"},
        "B": {"band_top_y": None}}}), encoding="utf-8")
    assert G.load_nadir_band(tmp_path) == {"A": 0.81}
    labels = G.load_nadir_band_labels(tmp_path)
    assert labels == {"A": "black nadir (no camera)"}
    e = G.entry_meta({"pano": {"panorama_id": "A", "source": "panoramax"}, "detections": []},
                     "random", 0.81, labels["A"])
    assert e["band"] == 0.81 and e["band_label"] == "black nadir (no camera)"
    html = G.build_html([e], {}, "k", "n", "src")
    assert "e.band_label" in html and "inside the nadir logo band" in html


def test_gallery_hide_credit_drops_the_producer_and_the_link():
    import gt_gallery as G
    rec = {"pano": {"panorama_id": "A", "source": "panoramax", "copyright": "ecartip",
                    "license": "etalab-2.0", "panoramax_instance": "panoramax.ign.fr"},
           "detections": [], "benchmark_stratum": "ecartip_gopro_max"}
    shown = G.entry_meta(rec, "random")
    hidden = G.entry_meta(rec, "random", hide_credit=True)
    assert shown["credit"] == "ecartip / etalab-2.0" and shown["url"]
    assert "credit" not in hidden and hidden["url"] == ""
    assert "ecartip" not in json.dumps(hidden)


def test_logo_band_top_finds_the_first_sustained_blue_row():
    frac = [0.0] * 430 + [0.05] * 2 + [1.0] * 80
    assert L.logo_band_top(frac) == 432 / 512
    assert L.logo_band_top([0.0] * 400 + [0.5] * 3 + [0.0] * 109) is None   # a blip


def test_committed_band_file_names_every_overlay():
    with open(os.path.join(BUNDLE, L.NADIR_BAND_NAME), encoding="utf-8") as f:
        band = json.load(f)
    kinds = collections.Counter(e.get("kind") for e in band["panos"].values())
    assert kinds == {"black": 76, "logo": 10, None: 114}
    assert set(L.LOGO_BANDS) == {p for p, e in band["panos"].items() if e.get("kind") == "logo"}
    gl = [e for e in band["panos"].values() if e["group"] == "grand_lyon_ladybug"]
    assert len(gl) == 75 and all(e.get("kind") == "black" for e in gl)
    assert all(0.81 <= e["band_top_y"] <= 0.83 for e in gl)


# --------------------------------------------------------------------------- #
# after the review: the committed score and the recall-by-elevation split
# --------------------------------------------------------------------------- #
def test_committed_score_reproduces_from_the_committed_verdicts():
    if not os.path.exists(os.path.join(BUNDLE, "verdicts.json")):
        pytest.skip("no review yet")
    assert L.cmd_score(_Args) == 0


def test_below_horizon_is_level_reading_without_tilt():
    assert L.below_horizon_deg(0.3, 0.5) == pytest.approx(0.0, abs=1e-9)
    assert L.below_horizon_deg(0.9, 0.6) == pytest.approx(18.0)
    assert L.below_horizon_deg(0.1, 0.45) == pytest.approx(-9.0)


def test_below_horizon_undoes_a_tilt_exactly():
    from rampnet import stage1_geometry as sg
    # a point 7 deg below the gravity horizon, seen through a tilted rig, reads 7 again
    for b in (-150.0, -40.0, 0.0, 75.0, 170.0):
        x, y = sg.pixel_from_bearing_elevation(*sg.gravity_to_rig(b, -7.0, 12.0, -4.0), 1.0, 1.0)
        assert L.below_horizon_deg(float(x), float(y), 12.0, -4.0) == pytest.approx(7.0, abs=1e-6)


def test_elevation_band_edges():
    assert L.elevation_band(-0.1) == "above"
    assert L.elevation_band(0.0) == "0-2"
    assert L.elevation_band(5.999) == "4-6" and L.elevation_band(6.0) == "6-10"
    assert L.elevation_band(20.0) == ">20"
    assert L.band_labels()[0] == "above" and L.band_labels()[-1] == ">20"
    assert L.band_distance_m("6-10") == [14.2, 23.8]
    assert L.band_distance_m("0-2")[1] is None


def test_elevation_pool_is_the_score_recall_pool():
    recs, v, peaks = _toy()
    v["e1"]["missed"] = [{"x": 0.2, "y": 0.52, "unsure": True}]
    v["e1"]["no_missed"] = False
    out, _ = L.score_groups(recs, v, peaks)
    table = L.elevation_table(L.elevation_items(recs, v))
    for g in ("grand_lyon_ladybug", "ecartip_gopro_max", "all"):
        tot = table[g]["total"]
        assert (tot["tp"], tot["tp"] + tot["missed"]) == (out[g]["recall_055"]["tp"],
                                                          out[g]["recall_055"]["n"])
    assert table["all"]["total"]["missed_unsure"] == 1
    assert table["all"]["bands"]["2-4"]["missed_unsure"] == 1     # y 0.52 is 3.6 deg down
