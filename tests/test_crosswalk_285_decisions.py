"""The rules that decided #285's crosswalk_s6 round (docs/crosswalk_followups_285.md, A4), on synthetic inputs.

Each test pins one rule the committed numbers depend on, so that a change to it fails here and not only in
`check_285.py`'s byte compare: guard 6's rig mask, the 25 m population and one-to-one matching of the primary
(`recall_check._pano_stats`), what counts as a confirmed-empty pano, `primary_met` and the seed spread
(`cmd_compare`), the score threshold (`cmd_score`), the false-alarm sheet's own count check (`cmd_fa_sheet`),
guards 4 and 5 (`score_274.py retention`), the pano-store path `cards` reads, `add-panos --source`, and
`pos_sheet.tally`'s verdict and point flag. CPU only; no committed data is read except where a test says so.
"""
import argparse
import csv
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274"))

import export_stage2  # noqa: E402
import pos_sheet as ps  # noqa: E402
import recall_check as rc  # noqa: E402
import score_274  # noqa: E402

CITY = "juneau-ak"


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _recall_out(tmp_path, monkeypatch, verdicts, preds_by_label):
    """A recall/ layout in tmp_path: one city's intersections.csv (every verdict pano in the draw), one
    preds/<label>/<city>.jsonl per label (every pano listed, with no detections unless given), and the verdicts."""
    out = tmp_path / "recall"
    _write_csv(str(out / CITY / "intersections.csv"),
               [{"pano_id": p, "in_draw": "1", "camera_model": "cam", "flagged": "0", "capture_date": "2024-05",
                 "sequence_id": "s"} for p in sorted(verdicts)])
    for label, preds in preds_by_label.items():
        d = out / "preds" / label
        d.mkdir(parents=True)
        with open(d / f"{CITY}.jsonl", "w", encoding="utf-8", newline="") as f:
            for p in sorted(verdicts):
                f.write(json.dumps({"pano_id": p, "preds": preds.get(p, [])}) + "\n")
    monkeypatch.setattr(rc, "OUT", str(out))
    vpath = tmp_path / "verdicts.json"
    vpath.write_text(json.dumps({"panos": verdicts}), encoding="utf-8")
    return out, str(vpath)


def test_pano_stats_rig_mask_25m_population_one_to_one_and_empties(tmp_path, monkeypatch):
    verdicts = {
        # two sure marks within 25 m, close together, and one beyond 25 m (y below Y_25M)
        "A": {"missed": [{"x": 0.30, "y": 0.60}, {"x": 0.31, "y": 0.61}, {"x": 0.70, "y": 0.50}]},
        "E": {"missed": [], "no_missed": True},                          # confirmed empty
        "U": {"missed": [{"x": 0.5, "y": 0.6, "unsure": True}]},          # only an unsure mark: not empty
        "S": {"missed": []},                                              # seen, not confirmed: skipped
    }
    preds = {"A": [[0.305, 0.605, 0.9], [0.70, 0.50, 0.9]],            # one detection between the two near marks
             "E": [[0.1, 0.6, 0.9], [0.2, 0.7, 0.9], [0.3, 0.9, 0.9],   # two above the rig mask, one on the vehicle
                   [0.4, 0.6, 0.4]],                                    # one under the threshold
             "U": [[0.5, 0.6, 0.9]]}
    _, vpath = _recall_out(tmp_path, monkeypatch, verdicts, {"m": preds})
    assert 0.6 < rc.RIG_MASK_Y < 0.9 and 0.50 < rc.Y_25M < 0.60
    marks, empty = rc._pano_stats(json.load(open(vpath))["panos"], [CITY], "m", 0.5, CITY)
    assert marks == {"A": (2, 1)}           # 2 marks within 25 m, one detection serves one mark
    assert empty == {"E": (CITY, 3, 2)}     # 3 detections at the threshold, 2 above the rig mask; U is not empty


def _compare(tmp_path, monkeypatch, base_hits, new_hits, n=10):
    """cmd_compare on n panos with one near mark each; ``*_hits`` is, per seed, how many of them it finds."""
    verdicts = {f"P{i}": {"missed": [{"x": 0.5, "y": 0.6}]} for i in range(n)}
    verdicts["E"] = {"missed": [], "no_missed": True}
    labels = {}
    for side, hits in (("b", base_hits), ("n", new_hits)):
        for k, h in enumerate(hits, 1):
            labels[f"{side}{k}"] = {f"P{i}": [[0.5, 0.6, 0.9]] for i in range(h)}
    out, vpath = _recall_out(tmp_path, monkeypatch, verdicts, labels)
    a = argparse.Namespace(verdicts=vpath, cities=[CITY], primary_city=CITY, out_label="t",
                           base=[f"b{k}@0.5" for k in range(1, len(base_hits) + 1)],
                           new=[f"n{k}@0.5" for k in range(1, len(new_hits) + 1)])
    rc.cmd_compare(a)
    return json.load(open(out / "compare" / "t" / "compare.json", encoding="utf-8"))


def test_primary_needs_both_the_interval_and_seed_1(tmp_path, monkeypatch, capsys):
    # every bootstrap draw gives new minus base = 0.5, so the interval is above 0, but seed 1 gains nothing
    res = _compare(tmp_path, monkeypatch, base_hits=[10, 0], new_hits=[10, 10])
    nb = res["new_minus_base"]
    assert nb["ci95"][0] > 0 and nb["first_seed_recall_diff"] == 0
    assert nb["primary_met"] is False


def test_primary_met_when_both_hold(tmp_path, monkeypatch, capsys):
    res = _compare(tmp_path, monkeypatch, base_hits=[5, 5], new_hits=[10, 10])
    assert res["new_minus_base"]["primary_met"] is True


def test_seed_spread_is_the_largest_absolute_difference_from_seed_1(tmp_path, monkeypatch, capsys):
    res = _compare(tmp_path, monkeypatch, base_hits=[0, 10, 4], new_hits=[10])
    assert res["base"]["seed_spread"]["hit"] == 10


def test_score_counts_a_detection_at_exactly_the_threshold(tmp_path, monkeypatch, capsys):
    verdicts = {"P": {"missed": [{"x": 0.5, "y": 0.6}]}}
    out, vpath = _recall_out(tmp_path, monkeypatch, verdicts, {"m": {"P": [[0.5, 0.6, 0.23]]}})
    rc.cmd_score(argparse.Namespace(verdicts=vpath, preds=[str(out / "preds" / "m" / f"{CITY}.jsonl")],
                                    cities=[CITY], label="m", threshold=0.23))
    res = json.load(open(out / "score" / "m" / "recall.json", encoding="utf-8"))
    assert res["groups"]["ALL"]["within_25m"]["r0.05"]["hit"] == 1


def test_spots_take_the_strongest_detection_first_whatever_the_input_order():
    spots = rc._spots([("a", 0.10, 0.6, 0.3), ("b", 0.12, 0.6, 0.9)])
    assert len(spots) == 1 and spots[0]["x"] == 0.12


def _fa_sheet(tmp_path, monkeypatch, guard):
    Image = pytest.importorskip("PIL.Image")
    verdicts = {"E": {"missed": [], "no_missed": True}}
    # one detection exactly at the threshold above the rig mask (counted), one on the vehicle (not counted)
    out, vpath = _recall_out(tmp_path, monkeypatch, verdicts, {"m": {"E": [[0.5, 0.6, 0.5], [0.2, 0.9, 0.9]]}})
    native = tmp_path / "native"
    native.mkdir()
    Image.new("RGB", (400, 200), (90, 90, 90)).save(native / "E.jpg")
    cmp_path = tmp_path / "compare.json"
    cmp_path.write_text(json.dumps({"models": {"m@0.5": {"empty_false_alarms_rig_masked": guard}}}), encoding="utf-8")
    rc.cmd_fa_sheet(argparse.Namespace(verdicts=vpath, cities=[CITY], compare=[str(cmp_path)],
                                       native=[f"{CITY}={native}"], out=str(tmp_path / "page"), out_label="t"))
    return json.load(open(out / "fa_sheet" / "t" / "summary.json", encoding="utf-8"))


def test_fa_sheet_counts_what_guard_6_counts(tmp_path, monkeypatch):
    summary = _fa_sheet(tmp_path, monkeypatch, guard=1)
    assert summary["detections"] == {"m": 1} and summary["spots"] == 1


def test_fa_sheet_refuses_a_count_that_differs_from_compare_json(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="compare.json 2"):
        _fa_sheet(tmp_path, monkeypatch, guard=2)


def _retention(tmp_path, models):
    ratings = tmp_path / "ratings.csv"
    _write_csv(str(ratings), [
        {"card_id": "a", "pano_id": "pa", "x": "0.5", "y": "0.6", "verdict": "crosswalk"},
        {"card_id": "b", "pano_id": "pb", "x": "0.2", "y": "0.6", "verdict": "not_crosswalk"},
        {"card_id": "c", "pano_id": "pc", "x": "0.3", "y": "0.6", "verdict": "crosswalk"}])
    out = tmp_path / "ret"
    score_274.cmd_retention(argparse.Namespace(ratings=str(ratings), models=models, out=str(out)))
    return json.load(open(out / "retention.json", encoding="utf-8"))


def test_retention_counts_confirmed_cards_kept_at_threshold_within_the_radius(tmp_path, capsys):
    preds = tmp_path / "m.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in [
        {"pano_id": "pa", "preds": [[0.5, 0.6, 0.40]]},             # at the threshold: kept
        {"pano_id": "pb", "preds": [[0.2, 0.6, 0.9]]},              # a card that is not a crosswalk
        {"pano_id": "pc", "preds": [[0.37, 0.6, 0.9]]}]) + "\n",    # 0.07 away, outside the 0.05 radius
        encoding="utf-8")
    res = _retention(tmp_path, [f"m={preds}@0.40"])
    assert res["confirmed"] == 2
    assert res["models"]["m"] == {"threshold": 0.4, "kept": 1, "retention": 0.5}


def test_retention_refuses_a_missing_predictions_file(tmp_path):
    with pytest.raises(SystemExit, match="no predictions file"):
        _retention(tmp_path, [f"m={tmp_path / 'absent.jsonl'}@0.40"])


def test_cards_reads_a_flat_dir_or_a_sharded_pano_store(tmp_path):
    flat, store = tmp_path / "flat", tmp_path / "store"
    (store / "ab").mkdir(parents=True)
    flat.mkdir()
    (flat / "abc.jpg").write_bytes(b"x")
    (store / "ab" / "abc.jpg").write_bytes(b"x")
    assert score_274.card_image_path("abc", str(flat)) == os.path.join(str(flat), "abc.jpg")
    assert score_274.card_image_path("abc", str(store)) == os.path.join(str(store), "ab", "abc.jpg")
    assert score_274.card_image_path("abc", None, "root", "train") == os.path.join("root", "train", "abc.jpg")


def test_add_panos_writes_the_given_source(tmp_path):
    src = tmp_path / "src"
    for s in ("train", "val", "test"):
        (src / s).mkdir(parents=True)
        (src / s / f"{s}0.jpg").write_bytes(b"x")
        (src / s / f"{s}0.json").write_text(json.dumps({"pano_id": f"{s}0"}), encoding="utf-8")
    img = tmp_path / "img"
    img.mkdir()
    (img / "np.jpg").write_bytes(b"x")
    _write_csv(str(tmp_path / "pos.csv"), [{"label_uid": "osm:1@np", "city": "c", "pano_id": "np", "x": "0.5",
                                            "y": "0.6"}])
    _write_csv(str(tmp_path / "at.csv"), [{"pano_id": "np", "lat": "10.0", "lng": "10.0", "capture_date": "2024"}])
    _write_csv(str(tmp_path / "p1.csv"), [{"node_lat": "-40.0", "node_lng": "-40.0", "gt_id": "g"}])
    export_stage2.cmd_add_panos(argparse.Namespace(
        src_root=str(src), root=str(tmp_path / "root"), positives=[f"{tmp_path / 'pos.csv'}={img}"], negatives=[],
        positions=str(tmp_path / "at.csv"), phase1_sheet=str(tmp_path / "p1.csv"), census_rawlabels=None,
        source="osm_vistas_auto", exclude=[], ignore=[], out=str(tmp_path / "out")))
    js = json.load(open(tmp_path / "root" / "train" / "np.json", encoding="utf-8"))
    assert js["supervision"]["sources"] == ["osm_vistas_auto"]


# ---- pos_sheet.tally (A4 follow-up 2) --------------------------------------------------------------------------

def test_the_three_rule_constants_are_the_preregistered_ones():
    assert (ps.SUPPORT_MIN, ps.NOT_SUPPORTED_MARGIN, ps.POINT_FLAG_MARGIN) == (0.10, 0.03, 0.15)


def _point_rows(off):
    """20 cells per stratum, ``off[stratum]`` of them with the peak off the paint, all painted."""
    rows, tags, cell = [], {}, 1
    for s in ("auto", "human", "sdot_b2"):
        for i in range(20):
            rows.append({"cell": cell, "stratum": s, "city": "c1" if s == "auto" else s})
            tags[str(cell)] = {"paint": "clear", "point": "off" if i < off[s] else "on_paint"}
            cell += 1
    return rows, tags


@pytest.mark.parametrize("off, flag", [
    ({"auto": 12, "human": 3, "sdot_b2": 3}, True),     # 0.60 against 0.15 and 0.15
    ({"auto": 10, "human": 8, "sdot_b2": 0}, False),    # 0.50: above SDOT by far, above human by 0.10 only
    ({"auto": 10, "human": 0, "sdot_b2": 8}, False),    # the same, the other way round
    ({"auto": 10, "human": 8, "sdot_b2": 8}, False),    # 0.10 above both: under the 0.15 margin
])
def test_point_flag_needs_the_margin_over_both_human_and_sdot(off, flag):
    rows, tags = _point_rows(off)
    assert ps.tally(rows, tags)["point_flag"] is flag


def test_not_supported_when_the_interval_stays_under_the_bar():
    # 12 of 300 auto cells without paint (4%): above human (0%) by more than the 0.03 margin, so only the
    # "upper Wilson bound under SUPPORT_MIN" branch makes this not_supported rather than inconclusive
    rows = [{"cell": i, "stratum": "auto", "city": "c1" if i <= 150 else "c2"} for i in range(1, 301)]
    rows += [{"cell": i, "stratum": "human", "city": "h"} for i in range(301, 321)]
    tags = {str(r["cell"]): {"paint": "clear", "point": "on_paint"} for r in rows}
    for c in list(range(1, 7)) + list(range(151, 157)):
        tags[str(c)] = {"paint": "none", "point": "off"}
    res = ps.tally(rows, tags)
    assert res["strata"]["auto"]["no_paint_wilson"][1] < ps.SUPPORT_MIN
    assert res["auto_no_paint_training_weighted"] > ps.NOT_SUPPORTED_MARGIN
    assert res["verdict"] == "not_supported"


def test_committed_summary_weights_cities_by_training_positives():
    """The committed share is weighted by each city's positives in root_v6 (positives_full.csv), not equally."""
    path = os.path.join(ps.OUT, "summary.json")
    if not os.path.exists(path):
        pytest.skip("pos_sheet outputs not committed")
    saved = json.load(open(path, encoding="utf-8"))
    tags = json.load(open(os.path.join(ps.OUT, saved["tags_file"]), encoding="utf-8"))["tags"]
    rows = ps.read_csv(os.path.join(ps.OUT, "sample.csv"))
    for r in rows:
        r["cell"] = int(r["cell"])
    res = ps.tally(rows, tags)
    assert res["weights"] == saved["weights"] == {"brussels-be": 1347, "kaysville-ut": 380, "sacramento-ca": 1313}
    assert res["auto_no_paint_training_weighted"] == saved["auto_no_paint_training_weighted"]


# ---- boundaries and guards (exact binary fractions, so the boundary is the boundary) -----------------------------

def test_a_mark_exactly_at_the_radius_is_matched():
    assert rc.match_one_to_one([(0.5, 0.5)], [(0.5625, 0.5)], 0.0625) == {0}


def test_a_detection_exactly_at_the_spot_radius_joins_the_spot():
    assert len(rc._spots([("a", 0.5, 0.5, 0.9), ("b", 0.5625, 0.5, 0.5)], radius=0.0625)) == 1


def test_supported_at_exactly_the_bar():
    # 10% without paint in each of two equal cities: the weighted share is exactly SUPPORT_MIN
    rows = [{"cell": i, "stratum": "auto", "city": "c1" if i <= 50 else "c2"} for i in range(1, 101)]
    rows += [{"cell": i, "stratum": "human", "city": "h"} for i in range(101, 121)]
    tags = {str(r["cell"]): {"paint": "clear", "point": "on_paint"} for r in rows}
    for c in list(range(1, 6)) + list(range(51, 56)):
        tags[str(c)] = {"paint": "none", "point": "off"}
    res = ps.tally(rows, tags)
    assert res["auto_no_paint_training_weighted"] == ps.SUPPORT_MIN
    assert res["verdict"] == "supported"


def test_the_human_stratum_includes_jon_confirmed_crowd_labels():
    assert ps.HUMAN_SOURCES == ("crowd", "ps_crowd_jon_confirmed")
    v4 = [{"uid": f"h{i}", "source": "ps_crowd_jon_confirmed", "city": "s", "pano_id": f"q{i}", "x": 0.5, "y": 0.55}
          for i in range(4)]
    rows = ps.draw([], v4, per_city=0, n_human=2, n_sdot=0)
    assert [r["stratum"] for r in rows] == ["human", "human"]


def test_score_refuses_a_tags_file_not_marked_as_a_pre_pass(tmp_path, monkeypatch):
    _write_csv(str(tmp_path / "sample.csv"), [{"cell": "1", "stratum": "auto", "city": "c1"}])
    monkeypatch.setattr(ps, "OUT", str(tmp_path))
    tags = tmp_path / "tags.json"
    tags.write_text(json.dumps({"tags": {"1": {"paint": "clear", "point": "on_paint"}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not_ground_truth"):
        ps.cmd_score(argparse.Namespace(tags=str(tags), out=str(tmp_path / "page")))


# ---- the rubric each human rating file was made under (review of #301, S4) ---------------------------------------

def test_ratings_definitions_sidecar_285():
    """analysis_out/crosswalk_followups_285/ratings_definitions.csv: every #285 human rating file is listed once and
    exists, and carries the exact instruction text its page showed: recall_check.BANNER for the missed-crosswalk
    page, score_274.DEFINITIONS['paint_brick'] for the A3 gate pages (whose ratings postdate that definition)."""
    import glob
    import re
    base = os.path.join(REPO, "analysis_out", "crosswalk_followups_285")
    norm = lambda v: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", v)).strip()  # noqa: E731
    texts = {"recall285_marks": norm(rc.BANNER), "paint_brick": norm(score_274.DEFINITIONS["paint_brick"])}
    with open(os.path.join(base, "ratings_definitions.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    listed = [r["file"] for r in rows]
    assert len(listed) == len(set(listed))
    on_disk = {os.path.relpath(p, base).replace(os.sep, "/")
               for p in glob.glob(os.path.join(base, "**", "ratings", "*"), recursive=True)}
    assert on_disk == set(listed), sorted(on_disk ^ set(listed))
    for r in rows:
        assert r["definition_text"] == texts[r["definition"]], r["file"]
        if r["definition"] == "paint_brick":
            with open(os.path.join(base, r["file"]), encoding="utf-8", newline="") as f:
                times = [t["rated_at"] for t in csv.DictReader(f) if t.get("rated_at")]
            assert times and min(times) >= "2026-11-08T15:52:58Z", r["file"]   # score_274's PAINT_BRICK_FROM
