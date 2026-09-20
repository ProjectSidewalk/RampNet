"""The YOLO warmup-LR collapse caveat (#72), pinned to the committed curves.

CPU only, committed fixtures only: the eight ``results.csv`` files under
``scripts/model_comparison/yolo_baseline/runs/`` and the three seed-replicate curves
under ``docs/data/seed_variance_51_135/``. What is load-bearing:

* every fact the caveat prose states (epoch-1 peak range, dip minimum range and
  epoch, the lr/pg0 peak at epoch 3, the recall-collapse signature, recovery epochs,
  the reported checkpoints sitting after the recovery) holds on the data;
* the seed replicates -- different seeds, different hardware -- reproduce the dip;
* the schedule and grid cell the prose cites are in each run's ``args.yaml``;
* the table in ``yolo_baseline/README.md`` is byte-for-byte what the script prints,
  so the README cannot drift from the CSVs without this failing;
* the headline figures quoted in the prose of the four documents that carry the caveat
  are present verbatim, so a prose edit that changes a number fails here too.
"""
import csv
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

import yolo_warmup_dip_72 as dip  # noqa: E402

README = os.path.join(ROOT, "scripts", "model_comparison", "yolo_baseline", "README.md")
BEGIN, END = "<!-- yolo_warmup_dip_72:begin -->", "<!-- yolo_warmup_dip_72:end -->"

# The documents that quote the caveat's figures, and the exact strings each must carry
# (the docs' own characters: en-dashes in ranges, arrows in the lr ramp).
CAVEAT_DOCS = {
    "scripts/model_comparison/yolo_baseline/README.md": [
        "0.010 → 0.020 → 0.029", "0.65–0.78", "0.00–0.25", "15–22", "5–7",
        "1.133 at epoch 1 to 8.227 at epoch 4", "1.7–7.3×", "+0.07", "2/4/6/12",
        "0.94–1.00", "0.10–0.12 at epoch 3",
    ],
    "docs/model_comparison.md": [
        "0.010 → 0.020 → 0.029", "0.65–0.78", "0.00–0.25", "recall ≤ 0.03",
        "epoch 5–7", "epoch 15–22", "2/4/6/12", "1.7–7.3×", "+0.07",
        "minimum 0.10–0.12 at epoch 3", "ends at epoch 55", "at epoch 21",
    ],
    "docs/yolo_geometry_51.md": [
        "0.678 val mAP@50 at epoch 1 to 0.078 at epoch 3", "epochs 3–7",
        "ends at epoch 21", "end at epoch 9",
    ],
    "docs/seed_variance_51_135.md": [
        "0.686 / 0.691 / 0.692 at epoch 1", "0.116 / 0.110 / 0.103", "by epoch 5",
        "0.078 at epoch 3",
    ],
}


def _stats():
    return dip.summarize_all()


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_every_committed_curve_exists_and_is_contiguous():
    stats = _stats()
    assert list(stats) == [name for name, _, _ in dip.RUNS]
    assert stats["y11l_pano"]["epochs"] == 60
    assert stats["y11x_pano_h200"]["epochs"] == 60
    assert all(stats[n]["epochs"] == 60 for n in dip.SEED_REPLICATES)


def test_pinned_facts_hold():
    failures = dip.check(_stats())
    assert failures == [], "\n".join(failures)


def test_the_numbers_the_prose_quotes():
    """The specific figures written into docs/model_comparison.md and the README."""
    s = _stats()
    grid = [s[n] for n in dip.GRID_CONFIGS]
    # epoch-1 peak 0.65-0.78 across the six configs
    assert round(min(g["map50_ep1"] for g in grid), 2) == 0.65
    assert round(max(g["map50_ep1"] for g in grid), 2) == 0.78
    # minimum 0.00-0.25, landing at epochs 3-6
    assert min(g["dip_min"] for g in grid) == 0.0
    assert round(max(g["dip_min"] for g in grid), 2) == 0.25
    assert {g["dip_epoch"] for g in grid} == {3, 4, 6}
    # lr/pg0 0.010 -> 0.020 -> 0.029 over epochs 1-3, warmup_epochs=3
    lr = [float(r[dip.LR]) for r in dip.read_results_csv(
        os.path.join(dip.GRID_DIR, "y11l_pano", "results.csv"))][:3]
    assert [round(v, 3) for v in lr] == [0.010, 0.020, 0.029]
    # val/cls_loss 1.1-1.2 at epoch 1 -> 8.2 at the bottom on y11l_pano
    assert round(s["y11l_pano"]["val_cls_peak"], 1) == 8.2
    # pano arms regain the epoch-1 level at ep15 / ep18 / ep22, tiles at ep5 / ep5 / ep7
    assert [s[n]["recovered_epoch"] for n in ("y11l_pano", "y11x_pano", "y26_pano")] == [15, 18, 22]
    assert [s[n]["recovered_epoch"] for n in ("y11l_tiles", "y11x_tiles", "y26_tiles")] == [5, 5, 7]
    # the reported checkpoints: ep59 / ep60 / ep54-in-CSV (scored ep56) / ep21-in-CSV (scored ep44)
    assert s["y11l_pano"]["best_map5095_epoch"] == 59
    assert s["y11x_pano_h200"]["best_map5095_epoch"] == 60
    assert s["y26_pano"]["best_map5095_epoch"] == 54
    assert s["y11x_tiles"]["best_map5095_epoch"] == 21
    # the committed curves that end before the scored epoch (y26_pano ep56, y11x_tiles ep44)
    assert s["y26_pano"]["epochs"] == 55
    assert s["y11x_tiles"]["epochs"] == 21
    assert s["y11l_tiles"]["epochs"] == 9 and s["y26_tiles"]["epochs"] == 9
    # training cls_loss flat (<= +0.07 over its epoch-2 value, ep8 < ep1), val cls_loss 1.7-7.3x
    assert round(max(s[n]["train_cls_rise_over_ep2"] for n in s), 2) == 0.07
    assert all(s[n]["train_cls_ep8"] < s[n]["train_cls_ep1"] for n in s)
    ratios = [s[n]["val_cls_peak"] / s[n]["val_cls_ep1"] for n in s]
    assert round(min(ratios), 1) == 1.7 and round(max(ratios), 1) == 7.3
    assert round(s["y11l_pano"]["val_cls_ep1"], 3) == 1.133 and round(s["y11l_pano"]["val_cls_peak"], 3) == 8.227
    # recall collapse at epoch 3 while precision holds (y11l_pano 0.673 -> 0.038 at P 0.824)
    assert round(s["y11l_pano"]["recall"][0], 3) == 0.673 and round(s["y11l_pano"]["recall"][2], 3) == 0.038
    assert round(s["y11l_pano"]["precision"][2], 2) == 0.82
    # epoch 1 is the pre-collapse high, not the run's peak
    assert all(s[n]["map50_max_epoch"] > s[n]["recovered_epoch"] for n in s)
    assert round(s["y11l_pano"]["map50_max"], 3) == 0.828 and s["y11l_pano"]["map50_max_epoch"] == 60


def test_args_yaml_pins_the_schedule_and_grid():
    """The `(every runs/*/args.yaml)` claims, read from the files rather than asserted."""
    s = _stats()
    for name, (batch, imgsz, seed) in dip.GRID_ARGS.items():
        for line in dip.SCHEDULE_ARGS + (f"batch: {batch}", f"imgsz: {imgsz}", f"seed: {seed}"):
            assert line in s[name]["args"], f"{name}: {line}"
    assert sorted({b for b, _, _ in dip.GRID_ARGS.values()}) == [2, 4, 6, 12]
    assert sorted({i for _, i, _ in dip.GRID_ARGS.values()}) == [1024, 1280]
    assert sorted({sd for _, _, sd in dip.GRID_ARGS.values()}) == [0, 1, 2, 3]
    # and the check fails when a pinned line is absent
    broken = {n: dict(v) for n, v in s.items()}
    broken["y11l_pano"]["args"] = s["y11l_pano"]["args"] - {"warmup_epochs: 3.0"}
    assert any("warmup_epochs: 3.0" in f for f in dip.check(broken))


def test_prose_figures_are_present_verbatim():
    """The headline figures in the four caveat-carrying docs, as the exact strings they use."""
    for rel, needles in CAVEAT_DOCS.items():
        text = _read(rel)
        for needle in needles:
            assert needle in text, f"{rel}: expected the literal {needle!r}"


def _write_csv(path, rows):
    cols = ["epoch", dip.MAP50, dip.MAP5095, dip.PRECISION, dip.RECALL, dip.LR, dip.VAL_CLS, dip.TRAIN_CLS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(r)


def test_short_run_is_rejected_clearly(tmp_path):
    p = tmp_path / "results.csv"
    _write_csv(p, [[1, 0.7, 0.4, 0.9, 0.6, 0.01, 1.1, 1.3], [2, 0.6, 0.3, 0.9, 0.5, 0.02, 1.5, 1.2]])
    with pytest.raises(ValueError, match="at least 3"):
        dip.summarize(str(p))


def test_non_contiguous_no_box_epochs_are_refused():
    assert dip._fmt_no_box("x", []) == "0"
    assert dip._fmt_no_box("x", [3, 4, 5]) == "3 (ep3-5)"
    with pytest.raises(ValueError, match="not contiguous"):
        dip._fmt_no_box("x", [3, 5])


def test_check_and_markdown_together_runs_both(capsys):
    assert dip.main(["--check", "--markdown"]) == 0
    out, err = capsys.readouterr()
    assert out == dip.markdown_table(_stats())  # stdout is the table alone
    assert err.strip() == "ok"


def test_seed_replicates_reproduce_the_dip():
    s = _stats()
    for n in dip.SEED_REPLICATES:
        assert s[n]["dip_epoch"] == 3
        assert 0.10 <= s[n]["dip_min"] <= 0.12
        assert s[n]["recovered_epoch"] == 5
        assert s[n]["best_map5095_epoch"] >= 56


def test_readme_table_matches_the_script():
    with open(README, encoding="utf-8") as f:
        text = f.read()
    m = re.search(re.escape(BEGIN) + r"\n(.*?)" + re.escape(END), text, re.S)
    assert m, f"README is missing the {BEGIN} / {END} markers"
    assert m.group(1) == dip.markdown_table(_stats()), (
        "yolo_baseline/README.md dip table is stale; regenerate with "
        "`python scripts/analysis/yolo_warmup_dip_72.py --markdown` and diff the output "
        "against the block (do not paste a redirected stdout: on Windows it is CRLF, the README is LF)")


def test_check_mode_exits_zero():
    assert dip.main(["--check"]) == 0
