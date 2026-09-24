"""CPU-only checks for scripts/analysis/context_fov_86.py (plan item 4): the arm naming
matches the crop cutter's, the common-label logic, and the 640 px box arithmetic."""
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import context_fov_86 as cf  # noqa: E402
import crop_cutter  # noqa: E402


def test_crop_name_matches_the_cutter():
    """The label tables point at files the cutter actually writes."""
    for fov, tag in ((25, "fov25"), (50, "fov50"), (90, "fov90"), ("viewport", "viewport")):
        assert crop_cutter.fov_tag(fov, "gnomonic", None) == tag
        assert cf.crop_name("seattle-wa", 178655, tag) == crop_cutter.crop_name("seattle-wa", 178655, tag)
    assert cf.crop_name("chicago-il", "28878", "fov25") == "chicago-il__28878__fov25.jpg"


def _lab():
    return pd.DataFrame({
        "split": ["train", "train", "test"],
        "filename": ["gsv-a-1-CurbRamp.png", "gsv-a-2-CurbRamp.png", "gsv-b-3-CurbRamp.png"],
        "city": ["a", "a", "b"], "label_id": [1, 2, 3],
        "label_uid": ["a:1", "a:2", "b:3"], "pano_id": ["p", "p", "q"],
        "lat": [0.0] * 3, "lng": [0.0] * 3, "normalized_x": [0.5] * 3, "normalized_y": [0.5] * 3,
        "narrow": [1, 0, 0], "steep": [0, 1, 0],
    })


def test_build_labels_keeps_only_labels_every_arm_has(tmp_path):
    arms = ["viewport", "fov25"]
    manifest = {}
    for arm in arms:
        for city, lid in (("a", 1), ("a", 2), ("b", 3)):
            manifest[cf.crop_name(city, lid, arm)] = {"name": cf.crop_name(city, lid, arm), "status": "ok"}
    manifest[cf.crop_name("a", 2, "fov25")]["status"] = "missing_pano"   # one arm short of one label
    tables, split, dropped, summary = cf.build_labels(_lab(), manifest, arms)
    assert list(split.label_uid) == ["a:1", "b:3"]
    assert list(tables["fov25"].filename) == ["a__1__fov25.jpg", "b__3__fov25.jpg"]
    assert list(tables["viewport"].filename) == ["a__1__viewport.jpg", "b__3__viewport.jpg"]
    assert list(tables["fov25"].columns) == list(_lab().columns)          # the benchmark's table, same columns
    assert list(tables["fov25"].narrow) == [1, 0]
    assert summary["n_common"] == 2 and summary["per_arm_ok"] == {"viewport": 3, "fov25": 2}
    assert dropped.to_dict("records") == [{"label_uid": "a:2", "filename": "gsv-a-2-CurbRamp.png", "split": "train",
                                           "status_viewport": "ok", "status_fov25": "missing_pano"}]
    # a label absent from the manifest altogether is dropped too, and says so
    del manifest[cf.crop_name("b", 3, "viewport")]
    _, split2, dropped2, _ = cf.build_labels(_lab(), manifest, arms)
    assert list(split2.label_uid) == ["a:1"]
    assert dropped2.set_index("label_uid").loc["b:3", "status_viewport"] == "absent"


def test_build_labels_requires_the_file_when_images_is_given(tmp_path):
    arms = ["fov25"]
    manifest = {cf.crop_name(c, l, "fov25"): {"name": cf.crop_name(c, l, "fov25"), "status": "ok"}
                for c, l in (("a", 1), ("a", 2), ("b", 3))}
    (tmp_path / "a__1__fov25.jpg").write_bytes(b"x")
    _, split, dropped, _ = cf.build_labels(_lab(), manifest, arms, images=str(tmp_path))
    assert list(split.label_uid) == ["a:1"]
    assert set(dropped.status_fov25) == {"file_missing"}


def test_read_manifests_latest_row_wins_and_skips_truncated(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"name": "x.jpg", "status": "missing_pano"}) + "\n"
                 + json.dumps({"name": "x.jpg", "status": "ok"}) + "\n"
                 + '{"name": "y.jpg", "sta', encoding="utf-8")
    m = cf.read_manifests([str(p)])
    assert m == {"x.jpg": {"name": "x.jpg", "status": "ok"}}


def test_crop_box_is_the_taggers_arithmetic():
    """crop.py: x = int(nx * w); left = max(0, x - 320) ... right = min(w, x + 320)."""
    assert cf.crop_box(0.5, 0.5, 1440, 960) == (400, 160, 1040, 800)
    assert cf.crop_box(0.05, 0.9, 1440, 960) == (0, 544, 392, 960)       # clamped at the left and bottom
    # a real row (pittsburgh-pa:9291): int(0.2111111 * 1440) = int(303.99998) = 303, so left clamps
    # to 0 and right is 623; y = 306 clamps top to 0. crop.py truncates, it does not round.
    assert cf.crop_box(0.2111111, 0.31875, 1440, 960) == (0, 0, 623, 626)


def test_report_names_the_control_row_as_told(tmp_path, capsys):
    """The control row is whatever score file --control-scores points at, so the table has
    to say what that file is: the default label names the 100-epoch benchmark control, and
    an interim snapshot must be labelled as such rather than pass for it."""
    arm_scores = os.path.join(REPO, "analysis_out", "context_fov_86", "train_viewport_final_scores.json")
    out = tmp_path / "out"
    out.mkdir()
    import shutil
    shutil.copy(arm_scores, out / "train_viewport_final_scores.json")
    cf.main(["report", "--out-dir", str(out), "--control-scores", arm_scores,
             "--control-label", "control (#178 epoch-49 snapshot, INTERIM)", "--arms", "viewport"])
    summary = json.load(open(out / "summary.json"))
    assert [r["arm"] for r in summary["rows"]] == ["control (#178 epoch-49 snapshot, INTERIM)", "viewport"]
    assert "INTERIM" in (out / "summary.md").read_text()
    cf.main(["report", "--out-dir", str(out), "--control-scores", arm_scores, "--arms", "viewport"])
    assert json.load(open(out / "summary.json"))["rows"][0]["arm"] == "control (#178, HF crops)"


def test_contrast_of_an_arm_with_itself_is_exactly_zero(tmp_path):
    """The paired contrast scores both arms on the same pano-clustered resample, so an arm
    against itself is 0 on every draw, not merely 0 in expectation: the interval must be
    [0, 0], which is what distinguishes a paired draw from two independent ones."""
    out = os.path.join(REPO, "analysis_out", "context_fov_86")
    pred = os.path.join(out, "train_viewport_final_test_predictions.csv")
    labels = os.path.join(out, "labels_viewport.csv")
    c = cf.paired_contrast(pred, labels, pred, labels, os.path.join(out, "split_common.csv"), n_boot=5)
    assert c["n"] == 2182 and c["tags_fixed"] == list(cf.tb.FIXED_TAGS)
    for k in ("mAP", "micro_f1", "macro_f1"):
        assert c["a_minus_b"][k]["point"] == 0.0 and c["a_minus_b"][k]["ci95"] == [0.0, 0.0]
    assert all(v["ci95"] == [0.0, 0.0] for v in c["per_tag_ap_a_minus_b"].values())


def test_leak_free_contrast_keeps_only_panos_absent_from_train():
    """--subset leak_free is the score files' leak_free subset: the 957 common test rows whose
    panorama has no train label (docs/context_fov_86.md section 4), still paired."""
    out = os.path.join(REPO, "analysis_out", "context_fov_86")
    pred = os.path.join(out, "train_viewport_final_test_predictions.csv")
    labels = os.path.join(out, "labels_viewport.csv")
    c = cf.paired_contrast(pred, labels, pred, labels, os.path.join(out, "split_common.csv"), n_boot=3,
                           subset="leak_free")
    assert c["subset"] == "leak_free" and c["n"] == 957
    assert c["a_minus_b"]["mAP"]["ci95"] == [0.0, 0.0]
    with open(os.path.join(out, "train_viewport_final_scores.json")) as fh:
        lf = json.load(fh)["subsets"]["leak_free"]
    assert lf["n"] == 957 and abs(c["a_mAP"] - lf["fixed_tags"]["mAP"]) < 1e-6  # the score file rounds to 6 places


def test_contrast_refuses_prediction_files_over_different_rows(tmp_path):
    out = os.path.join(REPO, "analysis_out", "context_fov_86")
    pred = os.path.join(out, "train_viewport_final_test_predictions.csv")
    short = tmp_path / "short.csv"
    pd.read_csv(pred).iloc[:-1].to_csv(short, index=False)
    import pytest
    with pytest.raises(SystemExit, match="same test rows"):
        labels = os.path.join(out, "labels_viewport.csv")
        cf.paired_contrast(pred, labels, str(short), labels, os.path.join(out, "split_common.csv"), n_boot=2)


def test_crops_as_trained_listing_matches_the_labels_and_the_manifests():
    """crops_as_trained.sha256 is the only hash of the viewport crops as trained (crop640
    rewrote them in place after the cutter's manifest was written). Pin what it must be: one
    line per crop every arm trains or is scored on, the label-centred crops byte-identical
    to the cut (the manifest's sha256), and every viewport crop different from its 1440x960 cut."""
    import gzip
    out = os.path.join(REPO, "analysis_out", "context_fov_86")
    listing = {}
    with open(os.path.join(out, "crops_as_trained.sha256"), "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw
    for line in raw.decode("utf-8").splitlines():
        h, name = line.split("  ", 1)
        assert len(h) == 64 and name not in listing
        listing[name] = h
    names = set()
    for arm in cf.ARMS:
        names |= set(pd.read_csv(os.path.join(out, f"labels_{arm}.csv")).filename)
    assert set(listing) == names and len(names) == 4 * 10848
    cut = {}
    for m in ("fov", "viewport"):
        with gzip.open(os.path.join(out, f"manifest_{m}.jsonl.gz"), "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                cut[r["name"]] = r
    for name, h in listing.items():
        assert cut[name]["status"] == "ok"
        if name.endswith("__viewport.jpg"):
            assert h != cut[name]["sha256"], name
        else:
            assert h == cut[name]["sha256"], name
