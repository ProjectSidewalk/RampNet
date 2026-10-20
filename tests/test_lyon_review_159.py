"""The Lyon label audit bundle (#159, #263): staging, the plan, the artifact reads, the
overlay measurement, the per-group scorer and the gallery's overlay labels.

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


# --------------------------------------------------------------------------- #
# the staged bundle and the plan
# --------------------------------------------------------------------------- #
def test_the_staged_bundle_verifies():
    problems, notes, facts = L.verify_bundle(BUNDLE)
    assert problems == [], problems
    assert facts["panos"] == 175
    assert facts["strata"] == {"empty": 25, "random": 140, "top": 10}
    assert facts["detections_055"] == 330 and facts["detections_030"] == 524


def test_every_record_is_in_one_of_the_four_audit_groups():
    recs = L.load_records(BUNDLE)
    assert {r["benchmark_stratum"] for r in recs} == set(L.GROUPS)
    by = collections.Counter((r["benchmark_stratum"], r["benchmark_group"]) for r in recs)
    for g in L.GROUPS:                       # the floor of 20 random panos per group
        assert by[(g, "random")] >= 20 and by[(g, "empty")] >= 3


def test_the_plan_reproduces_and_carries_the_bar():
    with open(L.PLAN, encoding="utf-8") as f:
        assert f.read() == L._dumps(L.build_plan(BUNDLE))
    plan = L.build_plan(BUNDLE)
    assert plan["pass_bar"]["threshold"] == 0.80
    assert "held out of every training pool" in plan["never_train"]
    assert plan["sampler"]["seed"] == 159 and plan["sampler"]["record_confidence"] == 0.3


def test_no_email_reaches_the_committed_records():
    with open(os.path.join(BUNDLE, "records.jsonl"), encoding="utf-8") as f:
        assert not L.EMAIL_RE.search(f.read())


def test_stage_masks_emails_and_copies_the_sampler_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "records.jsonl").write_text(
        json.dumps({"pano": {"copyright": "a.b@example.org"}}) + "\n"
        + json.dumps({"pano": {"copyright": "grand lyon"}}) + "\n", encoding="utf-8")
    (src / "sample.json").write_text("{}", encoding="utf-8")
    (src / "group_map.json").write_text("[]", encoding="utf-8")
    assert L.stage(str(src), str(dst)) == 1
    text = (dst / "records.jsonl").read_text(encoding="utf-8")
    assert "a***@***" in text and "example.org" not in text and "grand lyon" in text
    assert (dst / "sample.json").exists() and (dst / "group_map.json").exists()


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
    assert row["det030"] == 3 and row["det055"] == 1


def test_above_horizon_is_y_below_half():
    art = L.build_artifacts([_rec("a", "S", [(0.2, 0.48, 0.6), (0.2, 0.6, 0.6)],
                                  group="ecartip_gopro_max")], {}, "sha")
    row = art["per_group"]["ecartip_gopro_max"]
    assert row["above_horizon_055"] == 1 and row["det055"] == 2
    assert art["items"][0]["above_horizon"] is True


def test_committed_artifacts_cover_the_bundle():
    with open(L.ARTIFACTS, encoding="utf-8") as f:
        art = json.load(f)
    assert art["results_jsonl_sha256"].startswith("60e688b2")
    assert sum(r["det030"] for r in art["per_group"].values()) == 524
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
def test_score_refuses_without_verdicts(tmp_path, capsys):
    class A:
        bundle = BUNDLE
        write = False
        lenient_duplicates = False
    if os.path.exists(os.path.join(BUNDLE, "verdicts.json")):
        pytest.skip("the review has happened")
    assert L.cmd_score(A) == 1
    assert "nothing to score" in capsys.readouterr().out


def test_score_groups_reads_the_bar_per_group_at_055_and_030():
    recs = [_rec(f"g{i}", "S", [(0.1, 0.6, 0.9), (0.3, 0.6, 0.4)]) for i in range(10)]
    recs += [_rec(f"e{i}", "T", [(0.1, 0.6, 0.9)], group="ecartip_gopro_max")
             for i in range(10)]
    recs += [_rec("top", "S", [(0.1, 0.6, 0.9)], stratum="top")]
    v = {f"g{i}": {"dets": [True, i < 5], "missed": [], "no_missed": True}
         for i in range(10)}
    v.update({f"e{i}": {"dets": [i < 7], "missed": [{"x": 0.5, "y": 0.6}] if i == 0 else [],
                        "no_missed": i != 0} for i in range(10)})
    v["top"] = {"dets": [False], "missed": [], "no_missed": True}     # excluded
    out = L.score_groups(recs, v)
    gl = out["grand_lyon_ladybug"]
    assert gl["precision_055"]["p"] == 1.0 and gl["precision_055"]["n"] == 10
    assert gl["precision_030"]["tp"] == 15 and gl["precision_030"]["n"] == 20
    # A true 0.40 detection is a ramp the model missed at 0.55.
    assert gl["recall_055"]["tp"] == 10 and gl["recall_055"]["n"] == 15
    assert gl["bar"].startswith("PASS")
    ec = out["ecartip_gopro_max"]
    assert ec["precision_055"]["p"] == 0.7 and ec["bar"].startswith("FAIL")
    assert ec["recall_055"]["n"] == 8          # 7 true + 1 missed mark
    assert out["all"]["precision_055"]["n"] == 20   # top excluded


# --------------------------------------------------------------------------- #
# the gallery names the overlay
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
