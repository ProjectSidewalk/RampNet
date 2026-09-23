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
    assert "working draft" in r["text"]
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


def _write_list(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_sheet_template_prefills_the_severity_anchor(tmp_path):
    rows = [_row(1, tags="narrow", sev="2"), _row(2, sev="")]
    lst, sheet = tmp_path / "list.csv", tmp_path / "sheet.csv"
    _write_list(lst, rows)
    trp.main(["sheet-template", "--list", str(lst), "--out", str(sheet)])
    got = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert tuple(got[0]) == trp.SHEET_COLUMNS
    assert (got[0]["severity_at_list"], got[0]["severity"], got[0]["tags_at_list"]) == ("2", "2", "narrow")
    assert got[1]["severity"] == ""


def test_blank_sheet_severity_is_kept_and_counted(tmp_path):
    rows = [_row(i, tags="narrow") for i in range(1, 4)]
    sheet = [{"item_id": r["item_id"], "verdict": "agree", "tags": "narrow", "severity": "2"} for r in rows]
    sheet[0]["severity"] = ""                           # blanked: must not drop the item
    a = tr.items_from_sheet(rows, sheet)
    assert all(it["reviewed"] for it in a)
    assert a[0]["severity"] is None and a[0]["severity_missing"] is True
    assert a[1]["severity_missing"] is False
    b = [tr.make_item(r, reviewed=True, verdict="agree", tags_affirmed=["narrow"], severity=2) for r in rows]
    rep = tr.agreement(_export(a, "ra"), _export(b, "rb"))
    assert rep["items"]["both_judgeable"] == 3
    narrow = next(r for r in rep["per_tag"] if r["tag"] == "narrow")
    assert narrow["n"] == 3                              # still in every tag table
    assert rep["severity"]["n"] == 2 and rep["severity"]["missing_a"] == 1 and rep["severity"]["missing_b"] == 0


def test_excel_saved_sheet_with_bom_and_crlf(tmp_path):
    rows = [_row(i, tags="narrow") for i in range(1, 3)]
    lst = tmp_path / "list.csv"
    _write_list(lst, rows)
    body = ("item_id,verdict,tags,severity\n" + "\n".join(f"{r['item_id']},agree,narrow,1" for r in rows) + "\n")
    lf, excel = tmp_path / "lf.csv", tmp_path / "excel.csv"
    lf.write_bytes(body.encode("utf-8"))
    excel.write_bytes(b"\xef\xbb\xbf" + body.replace("\n", "\r\n").encode("utf-8"))
    outs = {}
    for name, sheet in (("lf", lf), ("excel", excel)):
        out = tmp_path / f"{name}.json"
        trp.main(["sheet", "--rater", "rb", "--list", str(lst), "--sheet", str(sheet),
                  "--rubric", str(RUBRIC_DOC), "--out", str(out)])
        outs[name] = tr.read_json(out)
    assert outs["excel"]["counts"]["reviewed"] == 2
    # the recorded hash is of the LF, BOM-free bytes, i.e. what git stores
    assert outs["excel"]["sheet_sha256"] == outs["lf"]["sheet_sha256"] == tr.sha256_file(lf)


def test_user_filter_fails_closed():
    recs = [{"user_id": "me", "label_id": "1"}, {"user_id": "", "label_id": "1"},
            {"label_id": "2"}, {"user_id": "other", "label_id": "1"}, {"user_id": "other", "label_id": "9"}]
    mine, others = trp.split_by_user(recs, "me", "alpha", listed={("alpha", 1), ("alpha", 2)})
    assert [r["label_id"] for r in mine] == ["1"] and all(r["city"] == "alpha" for r in mine)
    # blank / missing user ids are never the rater's; label 9 is not listed
    assert [(r.get("user_id"), r["label_id"]) for r in others] == [("", "1"), (None, "2"), ("other", "1")]


def test_prod_pull_handles_retired_tags_type_changes_and_other_edits():
    rows = [_row(1, tags="narrow"), _row(2), _row(3)]
    rows[0]["tags_at_list"] = "narrow;tactile warning"        # a retired tag survives on the label
    edits = [{"city": "alpha", "label_id": "102", "label_edit_id": "5", "new_tags": "[]", "new_severity": "",
              "new_label_type": "NoCurbRamp", "edit_time": "2026-09-23T10:00:00Z"}]
    vals = [{"city": "alpha", "label_id": "101", "label_validation_id": "1", "validation_result": "Agree",
             "end_timestamp": "2026-09-23T11:00:00Z"},
            {"city": "alpha", "label_id": "103", "label_validation_id": "2", "validation_result": "Agree",
             "end_timestamp": "2026-09-23T11:00:00Z"}]
    others = [{"city": "alpha", "label_id": "103", "label_edit_id": "77", "edit_time": "2026-09-22T23:00:00Z"},
              {"city": "alpha", "label_id": "103", "label_edit_id": "78", "edit_time": "2026-09-20T00:00:00Z"}]
    items = {it["item_id"]: it for it in tr.items_from_prod(
        rows, edits, vals, since="2026-09-23T00:00:00Z", other_edits=others,
        list_fetched_at={"alpha": "2026-09-22T20:00:00+00:00"})}
    one = items["tr0001"]
    assert one["tags_affirmed"] == ["narrow"] and one["tags_not_applicable"] == ["tactile warning"]
    assert one["tags_removed"] == []
    assert items["tr0002"]["problems"] == ["label type changed to NoCurbRamp"]
    assert items["tr0003"]["edited_by_others"] == [77]        # after the fetch, before the vote
    assert not tr._judgeable(items["tr0002"]) and tr._judgeable(items["tr0003"])


def test_agreement_reports_methods_and_prior_contact():
    rows = [_row(i, tags="narrow" if i < 5 else "") for i in range(1, 9)]
    for r in rows:
        r["prior_contact_ra"] = "validated" if r["item_id"] in ("tr0001", "tr0002") else ""
        r["prior_contact_rb"] = ""
    a = [tr.make_item(r, reviewed=True, verdict="agree", tags_affirmed=["narrow"] if i < 4 else [], severity=1)
         for i, r in enumerate(rows, 1)]
    b = [tr.make_item(r, reviewed=True, verdict="agree", tags_affirmed=["narrow"] if i < 5 else [], severity=1)
         for i, r in enumerate(rows, 1)]
    ea, eb = _export(a, "ra"), _export(b, "rb")
    eb["method"] = "review_sheet"
    rep = tr.agreement(ea, eb)
    assert rep["method"] == {"a": "test", "b": "review_sheet", "same": False}
    assert any("different routes" in w for w in rep["warnings"])
    assert rep["prior_contact"] == {"items_with_contact": 2, "items_without": 6}
    assert next(r for r in rep["per_tag_without_prior_contact"] if r["tag"] == "narrow")["n"] == 6
    assert "WARNING" in tra.render(rep)
    rep_same = tr.agreement(ea, _export(b, "rb"))
    assert rep_same["warnings"] == [] and rep_same["method"]["same"] is True


# ----------------------------------------------------------------------------- list builder

def test_geometry_matches_the_labellers_view():
    # seattle-wa:274404 (API row, crop era): placed with the label at the canvas centre
    # (canvas 358, 241 of about 720 x 480), so the labeller's recorded POV *is* the label's
    # world-frame direction: heading 3.4375, pitch -30.125. The camera is tilted 10.708 deg,
    # so this row separates the two conventions by 10.7 deg: pano_y is world-frame
    # (SidewalkWebpage povToPanoCoord), and subtracting camera_pitch again, as the first
    # draft did, would put the view at about -19.6.
    pano_y, pano_h, cam_pitch, pov_pitch, pov_heading = 5474, 8192, 10.707985, -30.125, 3.4375
    dep = trl.depression_deg(pano_y, pano_h)
    heading, pitch = trl.label_view(12684, 16384, 264.373962, dep)
    assert float(pitch) == pytest.approx(pov_pitch, abs=0.5)
    assert float(heading) == pytest.approx(pov_heading, abs=0.5)
    double_corrected = -(float(dep) - cam_pitch)
    assert abs(double_corrected - pov_pitch) > 5, "fixture must tell the two conventions apart"
    d = trl.flat_ground_distance_m(np.array([30.0, 10.0, 5.0, -1.0]))
    assert list(trl.distance_band(d)) == ["near", "mid", "far", "far"]
    url = trl.gsv_url("P", float(heading), float(pitch))
    assert "&pitch=-30.3&" in url


def _cands(n, *, state="untagged", tags=None, pano=None, latlon=None):
    """A hand-built candidate frame for ``draw``: one city, one band."""
    import pandas as pd
    rows = []
    for i in range(n):
        lat, lon = latlon(i) if latlon else (47.0 + i * 0.001, -122.0)
        rows.append({"city": "a", "label_id": i + 1, "pano_id": pano(i) if pano else f"p{i}",
                     "latitude": lat, "longitude": lon, "state": state, "band": "near",
                     "tag_list": tags(i) if tags else []})
    return pd.DataFrame(rows)


def _only(state):
    return {s: (1.0 if s == state else 0.0) for s in trl.STATES}


def test_draw_enforces_the_min_separation():
    # pairs of labels 5 m apart (different panos), pairs 100 m from each other
    c = _cands(40, latlon=lambda i: (47.0 + (i // 2) * 0.0009 + (i % 2) * 0.000045, -122.0))
    idx, _, short = trl.draw(c, 40, _only("untagged"), seed=1, min_sep_m=10.0)
    assert len(idx) == 20 and short == [("untagged", "*", 20)]
    pts = [(c.latitude[i], c.longitude[i]) for i in idx]
    assert min(trl.haversine_m(*p, *q) for k, p in enumerate(pts) for q in pts[k + 1:]) >= 10.0
    # removing the rule lets both labels of a pair in
    idx0, _, _ = trl.draw(c, 40, _only("untagged"), seed=1, min_sep_m=0.0)
    assert len(idx0) == 40


def test_draw_takes_one_label_per_pano():
    # pairs of labels share a pano but sit 100 m apart, so only the pano rule can block them
    c = _cands(40, pano=lambda i: f"p{i // 2}")
    idx, _, _ = trl.draw(c, 40, _only("untagged"), seed=1, min_sep_m=0.0)
    assert len(idx) == 20 and len({c.pano_id[i] for i in idx}) == 20


def test_rare_tag_weighting_lifts_the_rare_tag():
    c = _cands(200, state="tagged", tags=lambda i: ["steep"] if i < 10 else ["narrow"])
    def steep(power):
        idx, _, _ = trl.draw(c, 20, _only("tagged"), seed=5, min_sep_m=0.0, rare_power=power)
        return sum("steep" in c.tag_list[i] for i in idx)
    assert steep(1.5) >= 8        # (1/10)^1.5 vs (1/190)^1.5: the 10 rare labels dominate
    assert steep(0.0) <= 5        # unweighted: about 1 expected
    assert trl.rare_tag_weights(c)[0] == pytest.approx(1 / 10)


def test_vstudy_flags_by_distance_and_pano():
    import pandas as pd
    vs = pd.DataFrame({"label_id": [4429, 6114, 9], "pano_id": ["X", "Y", "Z"],
                       "latitude": [47.0, 47.00002, 47.01], "longitude": [-122.0, -122.0, -122.0]})
    same, near = trl.vstudy_flags(47.000004, -122.0, "Y", vs)   # 0.4 m and 1.8 m away; nearest first
    assert same is True and near == ["validation-study:4429", "validation-study:6114"]
    same, near = trl.vstudy_flags(47.02, -122.0, "Q", vs)
    assert same is False and near == []


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
                # an ExpertValidate Agree affirms; an Unsure alone does not (S2)
                vals.append({"label_id": i, "source": "ExpertValidate", "user_id": "u9",
                             "validation_result": "Agree" if i % 14 == 0 else "Unsure"})
            if i % 23 == 0:
                vals.append({"label_id": i, "source": "Validate", "user_id": jon, "validation_result": "Agree"})
        with open(root / f"{c}__rawLabels__CurbRamp.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        with open(root / f"{c}__validations__CurbRamp.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["label_id", "source", "user_id", "validation_result"],
                               lineterminator="\n")
            w.writeheader()
            w.writerows(vals)
        # an ExpertValidate edit affirms label 5 (untagged, never voted on) in every city
        with open(root / f"{c}__labelEdits.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["label_id", "source", "user_id"], lineterminator="\n")
            w.writeheader()
            w.writerow({"label_id": 5, "source": "ExpertValidate", "user_id": "u9"})
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


def test_candidates_affirm_only_on_agree_or_edit_and_flag_prior_contact(tmp_path):
    cache = tmp_path / "raw"
    cache.mkdir()
    _write_cache(cache)
    c, info = trl.build_candidates(
        str(cache), exclude_cities={"validation-study"}, require_tags=list(tr.CORE_TAGS),
        crop_date=trl.CROP_DATE, sources={"gsv"}, trusted=dict(trl.OWNERS), min_city_pool=10,
        raters=dict(trl.OWNERS))
    a = c[c.city == "alpha"].set_index("label_id")
    assert a.loc[14, "state"] == "affirmed_empty"          # ExpertValidate Agree
    assert a.loc[5, "state"] == "affirmed_empty"           # ExpertValidate edit
    assert a.loc[7, "state"] == "untagged"                 # ExpertValidate Unsure only
    assert a.loc[23, "prior_contact_jonfroehlich"] == "validated"
    assert a.loc[22, "prior_contact_jonfroehlich"] == "placed"
    assert a.loc[1, "prior_contact_jonfroehlich"] == "" and a.loc[1, "prior_contact_mikey"] == ""


# ----------------------------------------------------------------------------- the committed list

def test_committed_list_matches_its_meta_and_the_rubric_doc():
    meta = tr.read_json(META)
    digest = tr.sha256_file(LIST)
    assert meta["list"]["sha256"] == digest
    assert digest in RUBRIC_DOC.read_text(encoding="utf-8"), "update the list sha256 in the rubric doc"
    rows = trl.read_list(LIST)
    assert tuple(rows[0]) == trl.LIST_COLUMNS
    assert len(rows) == meta["list"]["rows"]
    assert "no camera_pitch term" in meta["params"]["depression"]
    for r in rows:   # the committed band agrees with the world-frame geometry
        dep = float(r["depression_deg"])
        assert r["distance_band"] == str(trl.distance_band(trl.flat_ground_distance_m(dep)))
        assert float(r["label_pitch_deg"]) == pytest.approx(-dep, abs=0.06)
        assert set(r["prior_contact_jonfroehlich"].split(";")) <= {"", *trl.CONTACT_KINDS}
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
    edits, vals, others, pulls = trp.fetch_rater_rows(rows, trp.RATER_IDS["jonfroehlich"])
    assert len(pulls) == 2 and all(p["sha256"] for p in pulls.values())
    assert all(r["user_id"] != trp.RATER_IDS["jonfroehlich"] for r in others)
    assert all(r["user_id"] == trp.RATER_IDS["jonfroehlich"] for r in edits + vals)
