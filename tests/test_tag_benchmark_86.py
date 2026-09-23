"""Tests for the #86 tag benchmark (scripts/analysis/tag_benchmark_86.py).

CPU only, synthetic data, no network, no checkpoint. What must hold:

- the metric code gives exactly what the tagger's own ``notebooks/evaluate.py`` gives
  (the reference below is that file's arithmetic, copied line for line), otherwise
  "reproduction" would compare two different metrics;
- the leak filter puts a test label in ``leaked`` iff its panorama appears in train, and a
  label whose panorama is unknown in neither subset;
- the re-split never puts one panorama on both sides and is a pure function of its seed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import tag_benchmark_86 as tb  # noqa: E402

pytest.importorskip("sklearn")


# --------------------------------------------------------------------------- #
# reference: notebooks/evaluate.py at sidewalk-tagger-ai 3b7405c, lines 402-420 and 504-618
# --------------------------------------------------------------------------- #
def _tagger_reference(y_true_np, y_pred_np, min_instances=10, min_threshold=0.3):
    from sklearn.metrics import average_precision_score, f1_score
    col_mask = np.sum(y_true_np, axis=0) >= min_instances
    yt, yp = y_true_np[:, col_mask], y_pred_np[:, col_mask]
    yb = (yp >= min_threshold).astype(int)
    micro = f1_score(yt, yb, average="micro")
    macro = f1_score(yt, yb, average="macro")
    weighted = f1_score(yt, yb, average="weighted")
    aps = []
    for i in range(y_true_np.shape[1]):
        if int(np.sum(y_true_np[:, i])) < min_instances:
            continue
        aps.append(average_precision_score(y_true_np[:, i], y_pred_np[:, i], average="weighted"))
    return sum(aps) / len(aps), micro, macro, weighted


def _fixture(n=400, k=5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([0.4, 0.2, 0.08, 0.03, 0.01])[:k]
    y = (rng.random((n, k)) < base).astype(float)
    s = np.clip(0.5 * y + rng.random((n, k)) * 0.7, 0, 1)
    return y, s


def test_metrics_match_the_tagger_evaluate_py():
    y, s = _fixture()
    tags = [f"t{i}" for i in range(y.shape[1])]
    ref = _tagger_reference(y, s)
    got = tb.tagger_metrics(y, s, tags)
    assert got["mAP"] == pytest.approx(ref[0], abs=1e-12)
    assert got["micro_f1"] == pytest.approx(ref[1], abs=1e-12)
    assert got["macro_f1"] == pytest.approx(ref[2], abs=1e-12)
    assert got["weighted_f1"] == pytest.approx(ref[3], abs=1e-12)
    # the <10-positive tag is dropped from every aggregate, exactly as the tagger does
    rare = [t for t, i in zip(tags, range(5)) if y[:, i].sum() < 10]
    assert rare and not set(rare) & set(got["tags_averaged"])


def test_threshold_is_inclusive_like_the_tagger():
    # evaluate.py binarises with >= min_threshold; a score of exactly 0.3 is a positive.
    y = np.array([[1], [0]] * 10, float)
    s = np.array([[0.3], [0.29]] * 10, float)
    got = tb.tagger_metrics(y, s, ["t"])
    assert got["micro_f1"] == 1.0


def test_fixed_tag_set_overrides_the_per_subset_rule():
    y, s = _fixture()
    tags = [f"t{i}" for i in range(5)]
    full = tb.tagger_metrics(y, s, tags)
    sub = tb.tagger_metrics(y[:100], s[:100], tags, selected=full["tags_averaged"])
    own = tb.tagger_metrics(y[:100], s[:100], tags)
    # on 100 rows a tag can fall under 10 positives; the fixed set keeps it so a subset's
    # mAP averages the same tags as the full set's
    assert set(sub["tags_averaged"]) == set(full["tags_averaged"])
    assert set(own["tags_averaged"]) <= set(full["tags_averaged"])


# --------------------------------------------------------------------------- #
# labels, filenames
# --------------------------------------------------------------------------- #
def test_parse_filename_handles_the_walla_walla_underscore():
    assert tb.parse_filename("gsv-walla_walla-124-CurbRamp.png") == ("walla-walla", 124)
    assert tb.parse_filename("gsv-pittsburgh-9291-CurbRamp.png") == ("pittsburgh", 9291)


def test_tag_columns_follow_the_tagger_offset_rule():
    df = pd.DataFrame(columns=["filename", "normalized_x", "normalized_y", "a", "b"])
    assert tb.tag_columns(df) == ["a", "b"]
    df = pd.DataFrame(columns=["filename", "x", "validated_by", "a"])
    assert tb.tag_columns(df) == ["a"]


def _splits():
    cols = ["filename", "normalized_x", "normalized_y", "narrow", "steep"]
    train = pd.DataFrame([["gsv-seattle-1-CurbRamp.png", .5, .5, 1, 0],
                          ["gsv-seattle-2-CurbRamp.png", .5, .5, 0, 1],
                          ["gsv-walla_walla-3-CurbRamp.png", .5, .5, 0, 0]], columns=cols)
    test = pd.DataFrame([["gsv-seattle-4-CurbRamp.png", .5, .5, 1, 1],      # shares pano A
                         ["gsv-seattle-5-CurbRamp.png", .5, .5, 0, 0],      # own pano C
                         ["gsv-seattle-6-CurbRamp.png", .5, .5, 0, 0],      # deleted label
                         ["gsv-walla_walla-7-CurbRamp.png", .5, .5, 1, 0]],  # own pano D
                        columns=cols)
    raw = {"seattle-wa": pd.DataFrame({"label_id": [1, 2, 4, 5], "pano_id": ["A", "B", "A", "C"],
                                       "latitude": [47.6, 47.6001, 47.6, 47.7],
                                       "longitude": [-122.3, -122.3, -122.3, -122.3]}),
           "walla-walla": pd.DataFrame({"label_id": [3, 7], "pano_id": ["E", "D"],
                                        "latitude": [46.0, 46.0], "longitude": [-118.3, -118.30005]})}
    return {"train": train, "test": test}, raw


def test_label_table_keeps_a_deleted_label_with_no_pano():
    splits, raw = _splits()
    lab, tags = tb.build_label_table(splits, raw)
    assert tags == ["narrow", "steep"]
    assert len(lab) == 7 and lab.label_uid.is_unique
    assert lab.set_index("label_uid").pano_id.isna().to_dict()["seattle-wa:6"]
    assert set(lab.city) == {"seattle-wa", "walla-walla"}


def test_leak_filter():
    splits, raw = _splits()
    lab, _ = tb.build_label_table(splits, raw)
    te = tb.leak_table(lab).set_index("label_uid")
    assert te.loc["seattle-wa:4", "pano_in_train"]                      # same pano as train label 1
    assert not te.loc["seattle-wa:5", "pano_in_train"]
    assert te.loc["seattle-wa:6", "pano_unknown"] and not te.loc["seattle-wa:6", "pano_in_train"]
    assert not te.loc["walla-walla:7", "pano_in_train"]
    # a different pano, but ~4 m from a train label: pano-disjoint, not spatially disjoint
    assert te.loc["walla-walla:7", "nearest_train_m"] == pytest.approx(3.9, abs=0.3)
    assert te.loc["seattle-wa:5", "nearest_train_m"] > 10_000


def test_score_subsets_partition():
    splits, raw = _splits()
    lab, tags = tb.build_label_table(splits, raw)
    te = lab[lab.split == "test"]
    pred = pd.DataFrame({"filename": te.filename, "score:narrow": [.9, .1, .1, .8],
                         "score:steep": [.7, .2, .1, .1]})
    out, per_label = tb.score_subsets(pred, lab, tags, n_boot=0)
    s = out["subsets"]
    assert s["full"]["n"] == 4
    assert s["leaked"]["n"] == 1 and s["leak_free"]["n"] == 2   # the no-pano label is in neither
    assert len(per_label) == 4


# --------------------------------------------------------------------------- #
# re-split
# --------------------------------------------------------------------------- #
def _synthetic_labels(n_panos=300, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_panos):
        city = ["seattle-wa", "chicago-il", "spgg"][p % 3]
        for k in range(int(rng.integers(1, 5))):
            lid = len(rows) + 1
            rows.append(dict(split="test" if rng.random() < 0.2 else "train", filename=f"f{lid}.png",
                             city=city, label_id=lid, label_uid=f"{city}:{lid}",
                             pano_id=None if lid % 97 == 0 else f"P{p}"))
    return pd.DataFrame(rows)


def test_resplit_is_pano_disjoint_seeded_and_city_matched():
    lab = _synthetic_labels()
    a = tb.pano_grouped_split(lab, seed=86)
    b = tb.pano_grouped_split(lab, seed=86)
    c = tb.pano_grouped_split(lab, seed=87)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    tr = set(a[a.split == "train"].pano_id.dropna())
    te = set(a[a.split == "test"].pano_id.dropna())
    assert not tr & te
    want = lab[lab.split == "test"].city.value_counts()
    got = a[a.split == "test"].city.value_counts()
    for city in want.index:   # overshoots by at most one pano's labels (< 5 here)
        assert want[city] <= got[city] < want[city] + 5


# --------------------------------------------------------------------------- #
# running ONE function from a tagger file
# --------------------------------------------------------------------------- #
CROP_PY = '''
import csv
import os
from PIL import Image

def crop_image(input_dir, csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            x = float(row['normalized_x'])
            y = float(row['normalized_y'])
            image_path = os.path.join(input_dir, filename)
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    width, height = img.size
                    x = int(x * width)
                    y = int(y * height)
                    left = max(0, x - 320)
                    top = max(0, y - 320)
                    right = min(width, x + 320)
                    bottom = min(height, y + 320)
                    cropped = img.crop((left, top, right, bottom))
                    cropped.save(image_path)

crop_image("does/not/exist", "does/not/exist.csv")   # module-level side effect, must not run
'''


def test_load_tagger_function_runs_only_that_function(tmp_path):
    from PIL import Image
    (tmp_path / "crop.py").write_text(CROP_PY)
    fn = tb.load_tagger_function(str(tmp_path), "crop.py", "crop_image")  # would raise if the call ran
    d = tmp_path / "imgs"
    d.mkdir()
    Image.new("RGB", (1000, 800)).save(d / "a.png")
    Image.new("RGB", (1000, 800)).save(d / "b.png")
    (d / "test.csv").write_text("filename,normalized_x,normalized_y\na.png,0.5,0.5\nb.png,0.1,0.95\n")
    fn(str(d), str(d / "test.csv"))
    assert Image.open(d / "a.png").size == (640, 640)
    assert Image.open(d / "b.png").size == (420, 360)   # clamped at the left and bottom edges


def test_write_csv_is_lf_and_rounded(tmp_path):
    p = tmp_path / "x.csv"
    tb.write_csv(pd.DataFrame({"a": [1 / 3], "b": ["x"]}), str(p), float_digits=4)
    assert p.read_bytes() == b"a,b\n0.3333,x\n"


def test_parse_tagger_title():
    svg = "<text>mAP: 0.34 | Micro F1: 0.67 | Macro F1: 0.31 | Weighted F1: 0.6 | Manual avg.: 0.31 | Threshold: 0.3</text>"
    assert tb.parse_tagger_title(svg) == {"mAP": 0.34, "micro_f1": 0.67, "macro_f1": 0.31,
                                          "weighted_f1": 0.6, "manual_avg_f1": 0.31, "threshold": 0.3}
    assert tb.parse_tagger_title("<svg/>") is None


def test_usage_row_is_unpaid_and_shaped_like_the_ledger():
    r = tb.usage_row("infer-released", 12.3456, "ok", "x", n=2183, ts="2026-09-22T00:00:00+00:00")
    assert r["paid"] is False and r["elapsed_s"] == 12.346 and r["hardware"]["gpus"] == ["NVIDIA A40"]
    assert r["est_cost_usd"] == 0.0 and r["provider"] != "claude"


def test_cell_split_is_pano_disjoint_and_keeps_near_panos_together():
    lab = pd.DataFrame([
        # panos P1 and P2 are 5 m apart (same corner); P3 is 2 km away
        dict(split="train", filename="a", city="seattle-wa", label_id=1, label_uid="s:1", pano_id="P1", lat=47.60000, lng=-122.3),
        dict(split="test", filename="b", city="seattle-wa", label_id=2, label_uid="s:2", pano_id="P2", lat=47.60004, lng=-122.3),
        dict(split="test", filename="c", city="seattle-wa", label_id=3, label_uid="s:3", pano_id="P3", lat=47.62, lng=-122.3),
        dict(split="train", filename="d", city="seattle-wa", label_id=4, label_uid="s:4", pano_id="P3", lat=47.62001, lng=-122.3),
    ])
    g = tb.cell_groups(lab, cell_m=100.0)
    assert g[0] == g[1]           # the same corner seen from two panos: one group
    assert g[2] == g[3] != g[0]   # one pano, one group
    sp = tb.pano_grouped_split(lab, seed=0, group="cell")
    assert sp.groupby("pano_id").split.nunique().max() == 1
