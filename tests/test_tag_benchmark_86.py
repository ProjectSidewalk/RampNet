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
    pred = pd.DataFrame({"filename": te.filename, "logit:narrow": [2.0, -2.0, -2.0, 1.5],
                         "logit:steep": [1.0, -1.5, -2.0, -2.0]})
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
    # This holds for THIS fixture only: both points sit well inside one ~100 m grid cell. Any
    # fixed grid splits some near pairs that straddle a cell edge, which is why the committed
    # block-grouped split still has 70 test labels within 10 m of a train label (doc §5).
    assert g[0] == g[1]           # the same corner seen from two panos: one group
    assert g[2] == g[3] != g[0]   # one pano, one group
    sp = tb.pano_grouped_split(lab, seed=0, group="cell")
    assert sp.groupby("pano_id").split.nunique().max() == 1


def test_logits_keep_saturated_scores_distinct():
    # at 6 dp, sigmoid(-15) and sigmoid(-17) are both 0.000000; their logits are not
    pred = pd.DataFrame({"filename": ["a", "b"], "logit:t": [-15.0, -17.0]})
    p = tb.score_probs(pred, ["t"])
    assert p[0, 0] > p[1, 0] > 0
    assert round(p[0, 0], 6) == round(p[1, 0], 6) == 0.0


# --------------------------------------------------------------------------- #
# review fixes (PR #178): prepare guard, dedup assert, paired diff, test-only, ledger rows
# --------------------------------------------------------------------------- #
def test_label_table_refuses_a_duplicated_live_row():
    splits, raw = _splits()
    raw["seattle-wa"] = pd.concat([raw["seattle-wa"], raw["seattle-wa"].iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate"):
        tb.build_label_table(splits, raw)


def test_prepare_guard_flags_crops_that_are_not_the_hf_size(tmp_path):
    from PIL import Image
    Image.new("RGB", (1440, 960)).save(tmp_path / "a.png")
    Image.new("RGB", (640, 640)).save(tmp_path / "b.png")     # already cropped by an interrupted run
    assert tb.uncropped_violations(str(tmp_path), ["a.png", "b.png"]) == ["b.png"]


def test_paired_diff_is_zero_when_the_subset_is_everything():
    rng = np.random.default_rng(0)
    y = (rng.random((60, 2)) < 0.4).astype(float)
    s = rng.random((60, 2))
    g = np.repeat(np.arange(20), 3)
    d = tb.paired_subset_diff_samples(y, s, g, np.ones(60, bool), ["a", "b"], ["a", "b"], n_boot=20)
    assert np.allclose(d["mAP"], 0) and np.allclose(d["micro_f1"], 0)


def test_filter_test_lines_is_byte_exact(tmp_path):
    p = tmp_path / "p.csv"
    p.write_bytes(b"filename,logit:a\nx.png,-1.23450\ny.png,0.5\nz.png,3\n")
    data, n = tb.filter_test_lines(str(p), {"y.png", "z.png"})
    assert n == 2 and data == b"filename,logit:a\ny.png,0.5\nz.png,3\n"


def test_concurrency_and_gpu_share():
    t = lambda h, m: tb._utc(f"2026-09-23T{h:02d}:{m:02d}:00Z")  # noqa: E731
    trains = {"train-control": (t(1, 58), None), "train-pano": (t(1, 58), None),
              "train-cell": (t(1, 58), t(2, 0))}
    conc = tb.concurrent_runs(trains, t(2, 53), t(3, 1))
    assert conc == ["train-control", "train-pano"]          # cell had ended; still-running arms count
    assert tb.share_fields(conc) == {"concurrent_with": conc, "gpu_share": 0.333333}
    assert tb.share_fields([])["gpu_share"] == 1.0


def test_a_final_row_replaces_its_in_progress_row_in_ledger_totals(tmp_path):
    from rampnet import ledger
    rid = "tagger-86:train-control:2026-09-23T01:58:54+00:00"
    interim = tb.usage_row("train-control", 6000, "in_progress", "x", extra={"run_id": rid})
    final = tb.usage_row("train-control", 61000, "ok", "x", extra={"run_id": rid})
    api_leg = {"label": "gemini", "elapsed_s": 10.0, "est_cost_usd": 1.0}
    log = tmp_path / "usage_log.jsonl"
    ledger.append_rows(log, [api_leg, interim, dict(api_leg)])
    assert ledger.ledger_totals(log)[0] == 3
    ledger.append_rows(log, [final])
    n, usd, hours, _ = ledger.ledger_totals(log)
    assert n == 3 and usd == 2.0                       # API legs are never superseded
    assert hours == pytest.approx((10 + 10 + 61000) / 3600)   # 6000 s of interim time is gone
    assert ledger.latest_rows(ledger.read_rows(log))[1]["status"] == "ok"   # takes its place


# --------------------------------------------------------------------------- #
# the committed artifacts: headline numbers, meta files, ledger rows
# --------------------------------------------------------------------------- #
OUT = os.path.join(REPO, "analysis_out", "tag_benchmark_86")


def _score_committed(pred, split_csv=None):
    lab, tags = tb.load_labels(os.path.join(OUT, "hf_curbramp_labels.csv"),
                               os.path.join(OUT, split_csv) if split_csv else None)
    out, _ = tb.score_subsets(pd.read_csv(os.path.join(OUT, pred)), lab, tags, n_boot=0,
                              fixed_tags=tb.FIXED_TAGS)
    return out


def test_headline_numbers_rederive_from_the_committed_predictions():
    out = _score_committed("released_test_predictions.csv")
    s = out["subsets"]
    assert out["tags_fixed"] == tb.FIXED_TAGS
    assert (s["full"]["n"], s["full"]["n_panos"]) == (2183, 1869)
    assert round(s["full"]["fixed_tags"]["mAP"], 4) == 0.3408
    assert round(s["full"]["fixed_tags"]["micro_f1"], 3) == 0.665
    assert round(s["full"]["fixed_tags"]["macro_f1"], 3) == 0.315
    assert (s["leak_free"]["n"], s["leak_free"]["n_panos"]) == (957, 889)
    assert round(s["leak_free"]["fixed_tags"]["mAP"], 4) == 0.3410
    assert (s["leaked"]["n"], s["leaked"]["n_panos"]) == (1226, 980)
    assert round(s["leaked"]["fixed_tags"]["mAP"], 3) == 0.363
    assert (s["leak_free_no_train_within_10m"]["n"], s["leak_free_no_train_within_10m"]["n_panos"]) == (759, 717)
    # and the committed JSON holds the same point estimates
    import json
    committed = json.load(open(os.path.join(OUT, "released_scores.json"), encoding="utf-8"))
    for k, v in s.items():
        for m in ("mAP", "micro_f1", "macro_f1"):
            assert committed["subsets"][k]["fixed_tags"][m] == pytest.approx(v["fixed_tags"][m], abs=1e-6)


@pytest.mark.parametrize("arm,n,map4", [("control", 2183, 0.3372), ("pano", 2197, 0.3598), ("cell", 2219, 0.3727)])
def test_epoch4_arm_numbers_rederive(arm, n, map4):
    out = _score_committed(f"train_{arm}_ep4_test_predictions.csv", tb.ARM_SPLITS[arm])
    assert out["subsets"]["full"]["n"] == n
    assert round(out["subsets"]["full"]["fixed_tags"]["mAP"], 4) == map4


@pytest.mark.parametrize("arm", ["control", "pano", "cell"])
def test_each_meta_describes_the_file_beside_it(arm):
    import json
    p = os.path.join(OUT, f"train_{arm}_ep4_test_predictions.csv")
    meta = json.load(open(p + ".meta.json", encoding="utf-8"))
    with open(p, "rb") as fh:
        body = fh.read()
    assert meta["file"] == os.path.basename(p)
    assert meta["n_rows"] == body.count(b"\n") - 1
    assert meta["predictions_sha256"] == tb.sha256_file(p)
    assert meta["source_meta"]["n_scored"] == 10857          # the 10,857-crop inference it came from
    assert meta["source_meta"]["checkpoint"] == "best_after_ep4.pth"


def test_committed_tagger86_ledger_rows_are_supersedable_and_carry_gpu_share():
    from rampnet import ledger
    rows = [r for r in ledger.read_rows(os.path.join(REPO, "analysis_out", "usage_log.jsonl"))
            if r.get("provider") == "tagger-86"]
    assert rows
    for r in rows:
        assert r["run_id"].startswith(f"tagger-86:{r['label']}:"), r
        assert 0 < r["gpu_share"] <= 1 and isinstance(r["concurrent_with"], list), r
    assert len({r["run_id"] for r in rows}) == len(rows)
    for r in rows:
        if r["status"] == "in_progress":
            # the final row collect() writes uses exactly this key, so it replaces this one
            assert r["run_id"] == f"tagger-86:{r['label']}:{r['ts']}"


def _fake_run(tmp_path, with_torch):
    """A finished control arm in a fake $WORK: logs, meta, best.pth, 10,857-style predictions."""
    import json
    work, out = tmp_path / "work", tmp_path / "out"
    (work / "train_control").mkdir(parents=True)
    out.mkdir()
    tags = tb.FIXED_TAGS + ["parallel-lines", "tactile-warning"]
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        rows.append(dict(split="test" if i % 4 == 0 else "train", filename=f"gsv-seattle-{i}-CurbRamp.png",
                         city="seattle-wa", label_id=i, label_uid=f"seattle-wa:{i}", pano_id=f"P{i // 2}",
                         lat=47.6 + i * 1e-3, lng=-122.3, normalized_x=.5, normalized_y=.5,
                         **{t: int(rng.random() < .5) for t in tags}))
    lab = pd.DataFrame(rows)
    tb.write_csv(lab, str(out / "hf_curbramp_labels.csv"))
    best = work / "train_control" / "best.pth"
    if with_torch:
        import torch
        torch.save({"epoch": 57, "model_state_dict": {}}, str(best))
    else:
        best.write_bytes(b"not a checkpoint")
    (work / "train_control.log").write_text("2026-09-23T01:58:54Z\n{\"epoch\": 0}\nEXIT 0\n2026-09-23T19:00:00Z\n")
    tb.write_json({"best_epoch": 57, "epochs": 100, "lr": 1e-6, "batch": 4, "n_train": 30, "elapsed_s": 61000.0,
                   "gpu": "NVIDIA A40", "ts": "2026-09-23T18:55:00+00:00"}, str(work / "train_control" / "train_meta.json"))
    (work / "train_control" / "train_log.csv").write_text("epoch,loss\n0,0.2\n")
    for name, ck_sha, ts in (("train_control_predictions.csv", tb.sha256_file(str(best)), "2026-09-23T19:00:00+00:00"),
                             ("snap_ep4_control_predictions.csv", "ab" * 32, "2026-09-23T03:01:48+00:00")):
        pred = pd.DataFrame({"filename": lab.filename, **{f"logit:{t}": rng.normal(size=len(lab)) for t in tags}})
        tb.write_csv(pred, str(work / name), float_digits=5)
        tb.write_json({"checkpoint_sha256": ck_sha, "n_scored": len(lab), "elapsed_s": 300.0, "ts": ts,
                       "gpu": "NVIDIA A40"}, str(work / name) + ".meta.json")
    return work, out


def test_collect_finishes_an_arm_and_is_idempotent(tmp_path):
    from rampnet import ledger
    try:
        import torch  # noqa: F401
        with_torch = True
    except ImportError:
        with_torch = False
    work, out = _fake_run(tmp_path, with_torch)
    log = tmp_path / "usage_log.jsonl"
    interim = tb.usage_row("train-control", 6000, "in_progress", "x", ts="2026-09-23T01:58:54+00:00",
                           extra={"run_id": "tagger-86:train-control:2026-09-23T01:58:54+00:00"})
    ledger.append_rows(log, [interim])
    for _ in range(2):   # a second run must replace, not add
        tb.collect_arm("control", str(work), str(log), epochs=(4, 9), final=True, n_boot=0, out_dir=str(out))
    rows = ledger.latest_rows(ledger.read_rows(log))
    assert [r["label"] for r in rows] == ["train-control", "infer-train-control-ep4", "infer-train-control-final"]
    train = rows[0]
    assert train["status"] == "ok" and train["elapsed_s"] == 61000.0   # the final row, from train_meta.json
    assert train["concurrent_with"] == [] and train["gpu_share"] == 1.0
    import json
    final = json.load(open(out / "train_control_final_scores.json", encoding="utf-8"))
    assert final["best_epoch"] == 57 and final["best_epoch_from_checkpoint"] == (57 if with_torch else None)
    assert (out / "train_control_log.csv").exists() and (out / "train_control_meta.json").exists()
    assert (out / "train_control_ep4_scores.json").exists() and not (out / "train_control_ep9_scores.json").exists()
    assert json.load(open(out / "train_control_final_test_predictions.csv.meta.json"))["n_rows"] == 10


def test_collect_refuses_an_arm_that_did_not_exit_cleanly(tmp_path):
    work, out = _fake_run(tmp_path, with_torch=False)
    (work / "train_control.log").write_text("2026-09-23T01:58:54Z\nEXIT 1\n")
    with pytest.raises(SystemExit, match="EXIT 0"):
        tb.collect_arm("control", str(work), str(tmp_path / "u.jsonl"), epochs=(), final=True, n_boot=0,
                       out_dir=str(out))
