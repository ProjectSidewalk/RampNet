"""The YOLO warmup-LR collapse caveat (#72), pinned to the committed curves.

CPU only, committed fixtures only: the eight ``results.csv`` files under
``scripts/model_comparison/yolo_baseline/runs/`` and the three seed-replicate curves
under ``docs/data/seed_variance_51_135/``. What is load-bearing:

* every fact the caveat prose states (epoch-1 peak range, dip minimum range and
  epoch, the lr/pg0 peak at epoch 3, the recall-collapse signature, recovery epochs,
  the reported checkpoints sitting after the recovery) holds on the data;
* the seed replicates -- different seeds, different hardware -- reproduce the dip;
* the table in ``yolo_baseline/README.md`` is byte-for-byte what the script prints,
  so the README cannot drift from the CSVs without this failing.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

import yolo_warmup_dip_72 as dip  # noqa: E402

README = os.path.join(ROOT, "scripts", "model_comparison", "yolo_baseline", "README.md")
BEGIN, END = "<!-- yolo_warmup_dip_72:begin -->", "<!-- yolo_warmup_dip_72:end -->"


def _stats():
    return dip.summarize_all()


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
        "`python scripts/analysis/yolo_warmup_dip_72.py --markdown`")


def test_check_mode_exits_zero():
    assert dip.main(["--check"]) == 0
