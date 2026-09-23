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
    # a real row (pittsburgh-pa:9291): x = int(304.0) = 304 clamps left to 0; y = 306 clamps top
    assert cf.crop_box(0.2111111, 0.31875, 1440, 960) == (0, 0, 624, 626)
