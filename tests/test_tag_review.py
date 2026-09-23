"""Tests for the curb-ramp tag review pass (#86 item 3): list builder, export format,
production reconstruction, sheet route and agreement. CPU only, no network: the list builder
runs on a synthetic API cache written into tmp_path, and the committed list is checked only
against its own committed meta file and the rubric doc."""
import csv
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from rampnet import tag_review as tr  # noqa: E402
import tag_review_list as trl  # noqa: E402
import tag_review_pull as trp  # noqa: E402
import tag_review_agreement as tra  # noqa: E402

RUBRIC_DOC = REPO / "docs" / "tag_rubric_draft.md"
LIST = REPO / "benchmark" / "tag_review" / "review_list.csv"
META = REPO / "benchmark" / "tag_review" / "review_list.meta.json"


# ----------------------------------------------------------------------------- kappa

def test_cohen_kappa_known_values():
    assert tr.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # 2x2 table a=20 b=5 c=10 d=15: po 0.7, pe 0.5 -> 0.4
    x = [1] * 25 + [0] * 25
    y = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert tr.cohen_kappa(x, y) == pytest.approx(0.4)
    # both raters constant and identical: undefined, not 1
    assert tr.cohen_kappa([0, 0, 0], [0, 0, 0]) is None


def test_weighted_kappa_known_values():
    assert tr.weighted_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    x, y = [1, 1, 2, 3, 3], [1, 2, 2, 2, 3]
    # hand computation, quadratic weights 0 / .25 / 1 on k = 3
    obs = np.zeros((3, 3))
    for a, b in zip(x, y):
        obs[a - 1, b - 1] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[0, .25, 1], [.25, 0, .25], [1, .25, 0]])
    assert tr.weighted_kappa(x, y) == pytest.approx(1 - (w * obs).sum() / (w * exp).sum())
    # a one-step disagreement costs less under quadratic than linear weights
    assert tr.weighted_kappa(x, y, weights="quadratic") > tr.weighted_kappa(x, y, weights="linear")


def test_kappa_ci_halfwidth_shrinks_with_positives():
    few = tr.kappa_ci_halfwidth(500, 10 / 500, 0.6, sims=800)
    many = tr.kappa_ci_halfwidth(500, 80 / 500, 0.6, sims=800)
    assert many < few


# ----------------------------------------------------------------------------- rubric + json

def test_committed_rubric_loads_and_is_marked_draft():
    r = tr.load_rubric(RUBRIC_DOC)
    assert r["version"].endswith("-draft")
    assert "PROPOSED DRAFT" in r["text"]
    assert r["sha256"] == tr.sha256_bytes(r["text"].encode("utf-8"))
    # The retired tag is never an applicable tag; the rubric says so.
    assert "tactile warning" in tr.RETIRED_TAGS


def test_rubric_hash_ignores_line_endings_and_text_outside_markers(tmp_path):
    body = f"intro\n{tr.RUBRIC_BEGIN}\n**Rubric version:** `v9`\nrule\n{tr.RUBRIC_END}\nfooter\n"
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(body.encode())
    b.write_bytes(body.replace("footer", "other footer").replace("\n", "\r\n").encode())
    assert tr.load_rubric(a) == tr.load_rubric(b)
    c = tmp_path / "c.md"
    c.write_bytes(body.replace("rule", "rule changed").encode())
    assert tr.load_rubric(c)["sha256"] != tr.load_rubric(a)["sha256"]


def test_write_json_is_lf_and_rounded(tmp_path):
    p = tmp_path / "x.json"
    tr.write_json(p, {"b": 1 / 3, "a": [np.float64(0.1234567891), np.int64(3)]})
    raw = p.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    assert json.loads(raw) == {"a": [0.123457, 3], "b": 0.333333}


# ----------------------------------------------------------------------------- synthetic list rows

APPL = ";".join(tr.CORE_TAGS)


def _row(i, city="alpha", tags="", sev="1", state="tagged", band="near"):
    return {"item_id": f"tr{i:04d}", "city": city, "label_id": str(100 + i),
            "label_uid": f"{city}:{100 + i}", "pano_id": f"P{i}", "tag_state": state,
            "distance_band": band, "tags_at_list": tags, "severity_at_list": sev,
            "applicable_tags": APPL, "editor_url": f"https://sidewalk-{city}.example.edu/gallery?labelId={100 + i}",
            "gsv_url": "", "placed_by_rater": ""}


def _export(items, rater, rubric=None, list_sha="L" * 64):
    rubric = rubric or {"version": "v-test", "text": "rubric text\n",
                        "sha256": tr.sha256_bytes(b"rubric text\n")}
    return tr.make_export(rater=rater, items=items, rubric=rubric, list_path_rel="x.csv",
                          list_sha256=list_sha, method="test", exported_at="2026-09-22T00:00:00+00:00")


# ----------------------------------------------------------------------------- production route

def test_items_from_prod_windows_last_edit_and_unchanged_agree():
    rows = [_row(1, tags="narrow"), _row(2, tags="steep", sev="2"), _row(3), _row(4, tags="narrow")]
    edits = [
        # item 1: two edits in the window; the last wins
        {"city": "alpha", "label_id": "101", "label_edit_id": "1", "new_tags": '["narrow","steep"]',
         "new_severity": "2", "edit_time": "2026-09-23T10:00:00-07:00", "source": "GalleryExpanded"},
        {"city": "alpha", "label_id": "101", "label_edit_id": "2", "new_tags": '["steep"]',
         "new_severity": "3", "edit_time": "2026-09-23T11:00:00-07:00", "source": "GalleryExpanded"},
        # item 4: an edit BEFORE the pass started must be ignored
        {"city": "alpha", "label_id": "104", "label_edit_id": "3", "new_tags": "[]",
         "new_severity": "1", "edit_time": "2026-09-01T00:00:00-07:00", "source": "ExpertValidate"},
    ]
    vals = [
        {"city": "alpha", "label_id": "101", "label_validation_id": "7", "validation_result": "Agree",
         "end_timestamp": "2026-09-23T11:00:01-07:00", "source": "GalleryExpanded"},
        # item 2: Agree with no edit = correct as is -> list-time tags and severity
        {"city": "alpha", "label_id": "102", "label_validation_id": "8", "validation_result": "Agree",
         "end_timestamp": "2026-09-23T12:00:00Z", "source": "GalleryExpanded"},
        # item 3: Unsure = whole-item cannot judge
        {"city": "alpha", "label_id": "103", "label_validation_id": "9", "validation_result": "Unsure",
         "end_timestamp": "2026-09-23T12:05:00Z", "source": "GalleryExpanded"},
        # another city with the same label_id must not leak in
        {"city": "beta", "label_id": "104", "label_validation_id": "10", "validation_result": "Agree",
         "end_timestamp": "2026-09-23T12:05:00Z", "source": "GalleryExpanded"},
    ]
    side = {"tr0002": {"cannot_judge_tags": "not level with street", "note": "car at kerb"}}
    items = {it["item_id"]: it for it in
             tr.items_from_prod(rows, edits, vals, since="2026-09-23T00:00:00Z", sidecar=side)}
    one = items["tr0001"]
    assert one["reviewed"] and one["tags_affirmed"] == ["steep"] and one["severity"] == 3
    assert one["tags_added"] == ["steep"] and one["tags_removed"] == ["narrow"]
    assert one["evidence"]["edit_ids"] == [1, 2] and one["verdict"] == "agree"
    two = items["tr0002"]
    assert two["tags_affirmed"] == ["steep"] and two["severity"] == 2 and two["tags_removed"] == []
    assert two["cannot_judge_tags"] == ["not level with street"] and two["note"] == "car at kerb"
    assert items["tr0003"]["cannot_judge"] is True
    assert items["tr0004"]["reviewed"] is False and items["tr0004"]["tags_affirmed"] == []


def test_non_applicable_tag_is_rejected():
    row = _row(1)
    row["applicable_tags"] = "narrow"
    with pytest.raises(ValueError):
        tr.make_item(row, reviewed=True, verdict="agree", tags_affirmed=["steep"], severity=1)


# ----------------------------------------------------------------------------- agreement

def _pair():
    rows = [_row(i, tags="narrow" if i % 2 else "") for i in range(1, 9)]
    a, b = [], []
    for i, r in enumerate(rows, 1):
        ta = ["narrow"] if i <= 4 else []
        tb = ["narrow"] if i in (1, 2, 3, 5) else []
        a.append(tr.make_item(r, reviewed=True, verdict="agree", tags_affirmed=ta, severity=1 + (i % 3)))
        b.append(tr.make_item(r, reviewed=True, verdict="agree", tags_affirmed=tb, severity=1 + (i % 3)))
    return rows, a, b


def test_agreement_per_tag_counts_and_exclusions():
    rows, a, b = _pair()
    # item 7: rater b votes Disagree (not a ramp) -> out of every tag table
    b[6] = tr.make_item(rows[6], reviewed=True, verdict="disagree")
    # item 8: rater a cannot judge narrow -> out of the narrow table only
    a[7] = tr.make_item(rows[7], reviewed=True, verdict="agree", tags_affirmed=[], severity=3,
                        cannot_judge_tags=["narrow"])
    rep = tr.agreement(_export(a, "ra"), _export(b, "rb"), group_by="tag_state")
    narrow = next(r for r in rep["per_tag"] if r["tag"] == "narrow")
    assert narrow["n"] == 6
    assert (narrow["pos_a"], narrow["pos_b"], narrow["both"]) == (4, 4, 3)
    x = [1, 1, 1, 1, 0, 0]
    y = [1, 1, 1, 0, 1, 0]
    assert narrow["kappa"] == pytest.approx(tr.cohen_kappa(x, y))
    assert narrow["pos_specific_agree"] == pytest.approx(6 / 8)
    assert rep["items"]["both_judgeable"] == 7
    assert rep["severity"]["n"] == 7 and rep["severity"]["weighted_kappa_quadratic"] == pytest.approx(1.0)
    assert rep["verdicts"] == {"agree|agree": 7, "agree|disagree": 1}
    assert "tagged" in rep["by_tag_state"]


def test_agreement_refuses_mismatched_rubric_or_list():
    _, a, b = _pair()
    other = {"version": "v2", "text": "different\n", "sha256": tr.sha256_bytes(b"different\n")}
    with pytest.raises(ValueError, match="rubric mismatch"):
        tr.agreement(_export(a, "ra"), _export(b, "rb", rubric=other))
    rep = tr.agreement(_export(a, "ra"), _export(b, "rb", rubric=other), allow_rubric_mismatch=True)
    assert rep["rubric"]["same_text"] is False
    with pytest.raises(ValueError, match="different review lists"):
        tr.agreement(_export(a, "ra"), _export(b, "rb", list_sha="M" * 64))


def test_validate_export_catches_tampered_rubric():
    _, a, _ = _pair()
    e = _export(a, "ra")
    e["rubric"]["text"] += "edited after the fact\n"
    with pytest.raises(ValueError, match="sha256"):
        tr.validate_export(e)


# ----------------------------------------------------------------------------- sheet route end to end

def test_sheet_route_and_agreement_cli(tmp_path):
    rows = [_row(i, tags="narrow" if i < 3 else "") for i in range(1, 5)]
    lst = tmp_path / "list.csv"
    with open(lst, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    sheet = tmp_path / "sheet.csv"
    trp.main(["sheet-template", "--list", str(lst), "--out", str(sheet)])
    got = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert "editor_url" not in got[0], "the blind sheet must not link the production editor"
    assert got[0]["tags"] == "narrow"
    for i, g in enumerate(got):
        g["verdict"] = "agree" if i < 3 else ""
        g["severity"] = "2"
    with open(sheet, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(got[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(got)
    outs = []
    for rater in ("ra", "rb"):
        out = tmp_path / f"{rater}.json"
        trp.main(["sheet", "--rater", rater, "--list", str(lst), "--sheet", str(sheet),
                  "--rubric", str(RUBRIC_DOC), "--out", str(out)])
        e = tr.validate_export(tr.read_json(out))
        assert e["counts"] == {"items": 4, "reviewed": 3, "not_reviewed": 1}
        assert e["rubric"]["version"] == tr.load_rubric(RUBRIC_DOC)["version"]
        outs.append(out)
    rep = tra.main([str(outs[0]), str(outs[1]), "--md", str(tmp_path / "a.md")])
    narrow = next(r for r in rep["per_tag"] if r["tag"] == "narrow")
    assert narrow["n"] == 3 and narrow["pct_agree"] == 1.0
    assert (tmp_path / "a.md").read_text(encoding="utf-8").startswith("# Tag review agreement")


# ----------------------------------------------------------------------------- list builder

def test_geometry_matches_the_labellers_view():
    # seattle-wa label 9 (API row): pano_x 13740, pano_y 4754 of 16384 x 8192,
    # camera heading 180.369, camera pitch 1.461; the labeller's POV was heading 299.3,
    # pitch -17.5 with the label right of and above centre.
    dep = trl.depression_deg(4754, 8192, 1.4609375)
    assert float(dep) == pytest.approx(13.0, abs=0.1)
    heading, pitch = trl.label_view(13740, 16384, 180.36891174316406, dep)
    assert 299.3 < float(heading) < 306 and -17.5 < float(pitch) < 0
    d = trl.flat_ground_distance_m(np.array([30.0, 10.0, 5.0, -1.0]))
    assert list(trl.distance_band(d)) == ["near", "mid", "far", "far"]


def _write_cache(root):
    cities = {"alpha": list(tr.CORE_TAGS), "beta": list(tr.CORE_TAGS),
              "gamma": [t for t in tr.CORE_TAGS if t != "points into traffic"],
              "validation-study": list(tr.CORE_TAGS)}
    manifest = {"hosts": {c: {"url": f"https://sidewalk-{c}.example.edu"} for c in cities}, "files": {}}
    header = trl.RAW_COLS + ["label_type"]
    jon = "549187e0-82c9-4014-a48d-31f18083d575"
    for c, vocab in cities.items():
        (root / f"{c}__labelTags.json").write_text(json.dumps({"label_tags": [
            {"id": k, "label_type": "CurbRamp", "tag": t, "description": t, "mutually_exclusive_with": []}
            for k, t in enumerate(vocab + ["tactile warning"])]}), encoding="utf-8")
        rows, vals = [], []
        for i in range(1, 161):
            user = trl.SIDEWALK_AI_USER if i % 17 == 0 else (jon if i % 11 == 0 else f"u{i % 5}")
            tags = [] if i % 3 else [vocab[i % len(vocab)]]
            placed = "2023-01-01T00:00:00Z" if i % 13 == 0 else "2025-06-01T00:00:00Z"
            rows.append({"label_id": i, "user_id": user, "pano_id": f"{c}-p{i}", "pano_source": "gsv",
                         "severity": 1 + i % 3, "tags": json.dumps(tags), "time_created": placed,
                         "correct": "false" if i % 19 == 0 else "true", "pano_y": 4096 + 40 * (i % 60),
                         "pano_height": 8192, "camera_pitch": 0.5, "latitude": 47 + i * 0.001,
                         "longitude": -122 + (i % 7) * 0.001, "image_capture_date": "2023-05",
                         "pano_x": 1000 * (i % 16), "pano_width": 16384, "camera_heading": 90.0,
                         "label_type": "CurbRamp"})
            if i % 7 == 0 and not tags:
                vals.append({"label_id": i, "source": "ExpertValidate"})
        with open(root / f"{c}__rawLabels__CurbRamp.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        with open(root / f"{c}__validations__CurbRamp.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["label_id", "source"], lineterminator="\n")
            w.writeheader()
            w.writerows(vals)
    (root / "fetch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_list_builder_is_deterministic_and_applies_every_filter(tmp_path):
    cache = tmp_path / "raw"
    cache.mkdir()
    _write_cache(cache)
    outs = []
    for k in range(2):
        out = tmp_path / f"l{k}" / "review_list.csv"
        trl.main(["build", "--cache", str(cache), "--out", str(out), "--n", "40", "--seed", "3",
                  "--min-city-pool", "10", "--min-sep-m", "0"])
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    assert b"\r\n" not in outs[0].read_bytes()
    rows = trl.read_list(outs[0])
    assert len(rows) == 40
    assert [r["item_id"] for r in rows] == [f"tr{k:04d}" for k in range(1, 41)]
    assert {r["city"] for r in rows} <= {"alpha", "beta"}          # gamma lacks a core tag
    assert all(r["label_uid"] == f"{r['city']}:{r['label_id']}" for r in rows)
    assert all(r["placed_at"] >= "2023-10-12" for r in rows)
    ids = {int(r["label_id"]) for r in rows}
    assert not any(i % 17 == 0 or i % 19 == 0 or i % 13 == 0 for i in ids)  # AI, incorrect, pre-crop
    assert all("tactile warning" not in tr.parse_tag_list(r["applicable_tags"]) for r in rows)
    assert {r["tag_state"] for r in rows} == {"affirmed_empty", "tagged_trusted", "tagged", "untagged"}
    assert all(r["editor_url"].endswith(f"labelId={r['label_id']}") for r in rows)
    meta = tr.read_json(outs[0].with_suffix("").as_posix() + ".meta.json")
    assert meta["list"]["sha256"] == tr.sha256_file(outs[0])
    assert "validation-study" in meta["cities_dropped"] and "gamma" in meta["cities_dropped"]
    # a different seed draws a different list
    other = tmp_path / "l9" / "review_list.csv"
    trl.main(["build", "--cache", str(cache), "--out", str(other), "--n", "40", "--seed", "4",
              "--min-city-pool", "10", "--min-sep-m", "0"])
    assert other.read_bytes() != outs[0].read_bytes()


# ----------------------------------------------------------------------------- the committed list

def test_committed_list_matches_its_meta_and_the_rubric_doc():
    meta = tr.read_json(META)
    digest = tr.sha256_file(LIST)
    assert meta["list"]["sha256"] == digest
    assert digest in RUBRIC_DOC.read_text(encoding="utf-8"), "update the list sha256 in the rubric doc"
    rows = trl.read_list(LIST)
    assert len(rows) == meta["list"]["rows"]
    assert len({r["label_uid"] for r in rows}) == len(rows)
    assert len({(r["city"], r["pano_id"]) for r in rows}) == len(rows)
    assert not any(r["city"] == "validation-study" for r in rows)
    for r in rows:
        assert r["label_uid"] == f"{r['city']}:{r['label_id']}"
        appl = tr.parse_tag_list(r["applicable_tags"])
        assert set(tr.CORE_TAGS) <= set(appl) and "tactile warning" not in appl
        assert set(tr.parse_tag_list(r["tags_at_list"])) <= set(appl)
        assert r["editor_url"].endswith(f"/gallery?labelType=CurbRamp&labelId={r['label_id']}")
        assert r["labelmap_url"].endswith(f"/labelMap?labelId={r['label_id']}")
        assert f"pano={r['pano_id']}&" in r["gsv_url"]


# ----------------------------------------------------------------------------- network (opt-in)

@pytest.mark.skipif(not __import__("os").environ.get("RAMPNET_NETWORK_TESTS"),
                    reason="network: set RAMPNET_NETWORK_TESTS=1 to query production")
def test_prod_pull_reaches_the_api():
    rows = trl.read_list(LIST)[:1]
    edits, vals, pulls = trp.fetch_rater_rows(rows, trp.RATER_IDS["jonfroehlich"])
    assert len(pulls) == 2 and all(p["sha256"] for p in pulls.values())
    assert all(r["user_id"] == trp.RATER_IDS["jonfroehlich"] for r in edits + vals)
