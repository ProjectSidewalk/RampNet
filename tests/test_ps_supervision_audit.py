"""The Project Sidewalk supervision audit (#86) on a tiny synthetic cache.

The real cache is ~650 MB of API pulls and is gitignored, so this builds a two-deployment
cache in tmp_path with known counts and checks that `report` reduces it to the numbers a
human would get by hand: the SidewalkAI account is dropped, the tag-era cutoff and the
crop date are applied by label date, tier 2 is `correct == true`, and the tag table sees a
tag the deployment no longer lists.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "analysis"))

import ps_supervision_audit as audit  # noqa: E402

JON = "549187e0-82c9-4014-a48d-31f18083d575"
AI = audit.SIDEWALK_AI_USER
CROWD = "00000000-0000-0000-0000-000000000001"

LABEL_COLS = ["label_id", "user_id", "pano_id", "pano_source", "label_type", "severity", "tags", "description",
              "time_created", "high_quality_user", "correct", "agree_count", "disagree_count", "unsure_count",
              "validations", "image_capture_date"]


def _label(label_id, user, when, tags=(), severity=1, correct="", validations=(), pano="p1"):
    return dict(label_id=label_id, user_id=user, pano_id=pano, pano_source="gsv", label_type="CurbRamp",
                severity=severity, tags=json.dumps(list(tags)), description="", time_created=when,
                high_quality_user="true", correct=correct, agree_count=0, disagree_count=0, unsure_count=0,
                validations=json.dumps(list(validations)), image_capture_date="2020-01")


def _agree(user):
    return dict(validator_type="Human", validation="Agree", user_id=user)


def _write_cache(cache):
    os.makedirs(cache)
    a = [
        _label(1, CROWD, "2017-06-01T00:00:00Z"),                                   # pre-tag era
        _label(2, CROWD, "2019-06-01T00:00:00Z", ["narrow"], 2, "true", [_agree(JON), _agree(CROWD)]),
        _label(3, JON, "2024-01-01T00:00:00Z", ["missing tactile warning", "steep"], 3, "true"),
        _label(4, AI, "2025-01-01T00:00:00Z", ["narrow"], 1, "true"),               # SidewalkAI: excluded
        _label(5, CROWD, "2025-01-01T00:00:00Z", ["tactile warning"], 1, "false"),  # tag the city no longer lists
    ]
    b = [
        _label(1, CROWD, "2022-01-01T00:00:00Z", [], 1, "true"),
        _label(2, JON, "2023-11-01T00:00:00Z", ["narrow"], 2, ""),
    ]
    for city, rows in (("alpha", a), ("beta", b)):
        pd.DataFrame(rows, columns=LABEL_COLS).to_csv(os.path.join(cache, f"{city}__rawLabels__CurbRamp.csv"), index=False)
        pd.DataFrame([], columns=LABEL_COLS).to_csv(os.path.join(cache, f"{city}__rawLabels__NoCurbRamp.csv"), index=False)
        tags = ["narrow", "steep", "missing tactile warning"] + (["tactile warning"] if city == "beta" else [])
        with open(os.path.join(cache, f"{city}__labelTags.json"), "w", encoding="utf-8") as fh:
            json.dump({"label_tags": [dict(id=i, label_type="CurbRamp", tag=t) for i, t in enumerate(tags)]}, fh)
    vals = [dict(label_validation_id=1, label_id=2, validation_result="Agree", user_id=JON, validator_type="Human",
                 source="ExpertValidate", end_timestamp="2024-01-01T00:00:00Z"),
            dict(label_validation_id=2, label_id=2, validation_result="Agree", user_id=CROWD, validator_type="Human",
                 source="Validate", end_timestamp="2024-01-01T00:00:00Z")]
    pd.DataFrame(vals).to_csv(os.path.join(cache, "alpha__validations__CurbRamp.csv"), index=False)
    edits = [dict(label_edit_id=1, label_id=3, label_type="CurbRamp", user_id=JON, old_label_type="CurbRamp",
                  new_label_type="CurbRamp", old_severity=1, new_severity=3, old_tags="[]",
                  new_tags=json.dumps(["missing tactile warning", "steep"]), source="ExpertValidate",
                  edit_time="2024-02-01T00:00:00Z", label_validation_id=1)]
    pd.DataFrame(edits).to_csv(os.path.join(cache, "alpha__labelEdits.csv"), index=False)
    with open(os.path.join(cache, "fetch_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"hosts": {"alpha": {"visibility": "public"}, "beta": {"visibility": "private"}},
                   "files": {"alpha__labelEdits": {"status": "fetched", "fetched_at": "2026-09-22T00:00:00+00:00"}}}, fh)


def test_report_reduces_the_cache_to_the_expected_counts(tmp_path):
    cache = str(tmp_path / "raw")
    _write_cache(cache)
    tables = str(tmp_path / "tables")
    doc = str(tmp_path / "audit.md")
    audit.main(["report", "--cache", cache, "--tables", tables, "--out", doc, "--hf-index", str(tmp_path / "none.csv")])

    tiers = pd.read_csv(os.path.join(tables, "tiers.csv")).set_index("tier")
    t1 = tiers.loc["tier 1: every human label"]
    assert t1.labels == 6                      # 7 labels minus the SidewalkAI one
    assert t1.users == 2 and t1.cities == 2
    assert t1.tagged == 4 and t1.correct == 3
    assert t1.tag_era == 5                     # label 1 (2017) is pre-tag
    assert t1.with_server_crop == 3            # labels dated on/after 2023-10-12: alpha 3, 5 and beta 2
    assert tiers.loc["tier 2: crowd-validated correct (correct == true)"].labels == 3
    assert tiers.loc["tier 3: placed by an Owner"].labels == 2
    assert tiers.loc["tier 3v: Agree-validated by an Owner"].labels == 1

    by_year = pd.read_csv(os.path.join(tables, "by_year.csv")).set_index("year")
    assert by_year.loc[2017].labels == 1 and by_year.loc[2017].private == 0
    assert by_year.loc[2022].private == 1

    tags = pd.read_csv(os.path.join(tables, "tags.csv")).set_index("tag")
    assert tags.loc["narrow"].labels == 2 and tags.loc["narrow"].deployments_listing == 2
    assert tags.loc["tactile warning"].deployments_listing == 1
    assert tags.loc["tactile warning"].labels_where_unlisted == 1 and tags.loc["tactile warning"].unlisted_in == "alpha"

    review = pd.read_csv(os.path.join(tables, "review_corpus.csv")).set_index("city")
    assert review.loc["alpha"].expert_validations == 1 and review.loc["alpha"].human_validations == 2

    edits = pd.read_csv(os.path.join(tables, "edits_by_source.csv"))
    assert edits.iloc[0].from_empty == 1 and edits.iloc[0].tags_added == 2 and edits.iloc[0].severity_changed == 1

    text = open(doc, encoding="utf-8").read()
    assert "## 2. Supervision tiers" in text and "jonfroehlich" in text
    assert "2,017" not in text                 # years are never thousands-formatted
    assert "\r" not in open(doc, "rb").read().decode()


def test_trusted_users_file_widens_tier_3(tmp_path):
    roles = tmp_path / "roles.psv"
    roles.write_text(f"Administrator|someone|{CROWD}\n", encoding="utf-8")
    trusted = audit.load_trusted(str(roles))
    assert trusted[JON] == ("Owner", "jonfroehlich")
    assert trusted[CROWD] == ("Administrator", "someone")
    assert audit.load_trusted(None).keys() == set(audit.OWNERS)


def _write_mixed_format_cache(cache):
    """The synthetic cache, with timestamps in the mixed shapes the real API returns.

    alpha's FIRST row carries fractional seconds and a later tag-era label does not, so a
    parse that infers one format from the first value (pandas without `format="ISO8601"`)
    turns the later one into NaT. The edit rows use the real `labelEdits` offset shape.
    Regression fixture for #183: the uniform timestamps above pass with or without the fix.
    """
    _write_cache(cache)
    path = os.path.join(cache, "alpha__rawLabels__CurbRamp.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[df.label_id == "1", "time_created"] = "2017-06-01T00:00:00.123Z"
    extra = _label(6, CROWD, "2021-03-05T14:58:59Z", ["narrow"], 2, "true")
    df = pd.concat([df, pd.DataFrame([extra], columns=LABEL_COLS).astype(str)], ignore_index=True)
    df.to_csv(path, index=False)
    edits_path = os.path.join(cache, "alpha__labelEdits.csv")
    edits = pd.read_csv(edits_path, dtype=str, keep_default_na=False)
    first = edits.iloc[0].copy()
    edits.loc[0, "edit_time"] = "2026-04-02T17:39:20.200850-07:00"
    first["label_edit_id"], first["edit_time"] = "2", "2024-02-01T00:00:00-08:00"
    pd.concat([edits, first.to_frame().T], ignore_index=True).to_csv(edits_path, index=False)


def test_mixed_timestamp_formats_keep_every_row(tmp_path):
    cache = str(tmp_path / "raw")
    _write_mixed_format_cache(cache)

    labels, _ = audit.load_labels(cache, "CurbRamp", {AI})
    assert labels.time_created.isna().sum() == 0
    assert labels.loc[labels.label_id.astype(str) == "6", "time_created"].iloc[0] == pd.Timestamp("2021-03-05T14:58:59Z")

    edits = audit.load_edits(cache)
    assert edits.edit_time.isna().sum() == 0
    assert list(edits.edit_time) == [pd.Timestamp("2026-04-03T00:39:20.200850Z"), pd.Timestamp("2024-02-01T08:00:00Z")]

    tables = str(tmp_path / "tables")
    audit.main(["report", "--cache", cache, "--tables", tables, "--out", str(tmp_path / "audit.md"),
                "--hf-index", str(tmp_path / "none.csv")])
    t1 = pd.read_csv(os.path.join(tables, "tiers.csv")).set_index("tier").loc["tier 1: every human label"]
    assert t1.labels == 7                      # the base 6 plus alpha label 6 (2021, no fraction)
    assert t1.tag_era == 6                     # only label 1 (2017) is pre-tag
    by_year = pd.read_csv(os.path.join(tables, "by_year.csv")).set_index("year")
    assert by_year.labels.sum() == t1.labels   # a NaT row would be in `labels` but in no year
    assert by_year.loc[2021].labels == 1


def test_unparseable_timestamp_fails_loudly(tmp_path):
    cache = str(tmp_path / "raw")
    _write_cache(cache)
    path = os.path.join(cache, "beta__rawLabels__CurbRamp.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[0, "time_created"] = "22/01/2022 10:00"
    df.to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"1 of 7 rawLabels__CurbRamp time_created .*'22/01/2022 10:00'"):
        audit.load_labels(cache, "CurbRamp", {AI})


def test_unparseable_edit_time_fails_loudly(tmp_path):
    cache = str(tmp_path / "raw")
    _write_cache(cache)
    path = os.path.join(cache, "alpha__labelEdits.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[0, "edit_time"] = ""
    df.to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"1 of 1 labelEdits edit_time"):
        audit.load_edits(cache)
