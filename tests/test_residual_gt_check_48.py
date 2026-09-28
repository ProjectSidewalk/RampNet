"""Tests for the #48 residual GT check (scripts/analysis/residual_gt_check_48.py).

CPU only, no network, committed files only: the rate and agreement arithmetic on toy
verdicts, the ring flag on a synthetic image, and the committed plan, verdict file and
crop manifest checked against each other.
"""
import hashlib
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import residual_gt_check_48 as gc  # noqa: E402


def _file(verdicts, items=("a", "b", "c", "d"), digest="x", rater="r1"):
    return {"rater": rater, "items": list(items), "manifest_digest": digest,
            "item_class": {"a": "association_placement", "b": "self_detected_site_displaced",
                           "c": "sub_threshold_only", "d": "never_fired"},
            "verdicts": {u: {"answer": a, "note": ""} for u, a in verdicts.items()}}


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_angular_gap_wraps_at_the_seam():
    assert gc.angular_gap_deg(0.999, 0.5, 0.001, 0.5) == pytest.approx(0.72)
    assert gc.angular_gap_deg(0.5, 0.5, 0.5, 0.6) == pytest.approx(18.0)


def test_rate_excludes_cant_tell_and_unanswered():
    r = gc.gt_error_rate({"a": {"answer": "no"}, "b": {"answer": "yes"},
                          "c": {"answer": "cant_tell"}}, ["a", "b", "c", "d"])
    assert (r["yes"], r["no"], r["cant_tell"], r["unanswered"]) == (1, 1, 1, 1)
    assert r["n_decided"] == 2 and r["gt_error_rate"] == 0.5
    lo, hi = r["wilson_95"]
    assert 0 < lo < 0.5 < hi < 1


def test_rate_with_nothing_decided_is_none():
    r = gc.gt_error_rate({}, ["a"])
    assert r["gt_error_rate"] is None and r["wilson_95"] == [0.0, 1.0]


def test_rates_split_out_the_merging_cases():
    out = gc.rates(_file({"a": "no", "b": "no", "c": "yes", "d": "yes"}))
    assert out["overall"]["gt_error_rate"] == 0.5
    assert out["merging_cases"]["n_items"] == 2
    assert out["merging_cases"]["gt_error_rate"] == 1.0
    assert out["by_class"]["sub_threshold_only"]["no"] == 0


def test_agreement_and_kappa():
    d1 = _file({"a": "yes", "b": "no", "c": "yes", "d": "cant_tell"})
    d2 = _file({"a": "yes", "b": "yes", "c": "yes", "d": "cant_tell"}, rater="r2")
    ag = gc.agreement(d1, d2)
    assert ag["n_both"] == 4 and ag["percent_agreement"] == 0.75
    assert ag["n_both_yes_no"] == 3
    assert ag["disagreements"] == [{"uid": "b", "r1": "no", "r2": "yes"}]
    # identical files agree perfectly
    assert gc.agreement(d1, d1)["kappa"] == pytest.approx(1.0)
    # kappa is undefined when both raters used one category throughout
    assert gc.cohen_kappa([("yes", "yes")] * 3, gc.ANSWERS) is None


def test_agreement_refuses_files_from_different_galleries():
    with pytest.raises(ValueError):
        gc.agreement(_file({}), _file({}, digest="y"))


def test_load_verdicts_rejects_an_answer_outside_the_rubric(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps(_file({"a": "gt-error"})), encoding="utf-8")
    with pytest.raises(ValueError):
        gc.load_verdicts(str(p))


def test_cut_one_ring_flag():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (3600, 1800), (40, 40, 40))
    plain = gc.mv.cut_one(img, 0.5, 0.5, ring=False)
    ringed = gc.mv.cut_one(img, 0.5, 0.5)
    assert plain.getcolors() == [(plain.size[0] * plain.size[1], (40, 40, 40))]
    assert ringed.getpixel((plain.size[0] // 2 + 14, plain.size[1] // 2)) != (40, 40, 40)


# --------------------------------------------------------------------------- #
# committed artifacts
# --------------------------------------------------------------------------- #
def _load(path):
    assert os.path.exists(path), f"{path} is missing (a committed #48 artifact)"
    raw = open(path, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    return json.loads(raw)


def test_plan_rings_only_the_source_view_at_the_reviewers_click():
    plan = _load(gc.PLAN_PATH)["items"]
    res = _load(gc.RESIDUAL_PATH)["ramps"]
    clicks = gc.load_source_clicks(res)
    src = [it for it in plan if it["is_source"]]
    assert all(it["ring"] for it in src)
    assert not any(it["ring"] for it in plan if not it["is_source"])
    assert sorted(it["ramp_uid"] for it in src) == sorted(r["uid"] for r in res)
    for it in src:  # the plan stores 4 decimals
        cx, cy = clicks[it["ramp_uid"]]
        assert abs(it["x"] - cx) <= 6e-5 and abs(it["y"] - cy) <= 6e-5, it["ramp_uid"]


def test_verdict_file_carries_the_rubric_and_the_fixed_item_list():
    v = _load(gc.VERDICTS_PATH)
    res = _load(gc.RESIDUAL_PATH)["ramps"]
    assert v["question"] == gc.QUESTION
    assert [r["key"] for r in v["rubric"]] == list(gc.ANSWERS)
    assert v["items"] == [r["uid"] for r in res]
    assert v["item_class"] == {r["uid"]: r["class"] for r in res}
    assert sum(c in gc.MERGING_CLASSES for c in v["item_class"].values()) == 58
    gc.rates(v)  # an empty or filled file must score


def test_crop_manifest_matches_the_crops_and_the_verdict_file():
    man = _load(os.path.join(gc.GALLERY_DIR, "manifest.json"))
    v = _load(gc.VERDICTS_PATH)
    assert man["manifest_digest"] == v["manifest_digest"]
    assert gc.manifest_digest(v["items"], man["crops_sha256"]) == man["manifest_digest"]
    plan = _load(gc.PLAN_PATH)["items"]
    assert sorted(man["crops_sha256"]) == sorted(gc.mv.crop_name(it) for it in plan)
    for name, sha in man["crops_sha256"].items():
        with open(os.path.join(gc.GALLERY_DIR, "crops", name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == sha, name
    page = open(os.path.join(gc.GALLERY_DIR, "gallery.html"), encoding="utf-8").read()
    assert man["manifest_digest"] in page and gc.EXPORT_NAME in page
    assert "mv48_residual_" not in page  # not the old gallery's storage key
