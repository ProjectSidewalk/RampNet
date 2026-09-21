#!/usr/bin/env python3
"""The YOLO baseline's warmup-LR collapse, read from the committed curves (#72).

Every supervised-YOLO run in this repo trained under the Ultralytics default schedule
(``optimizer=auto`` -> MuSGD, ``lr0=0.01``, ``warmup_epochs=3.0``), and every one of
them lost most of its validation mAP@50 during the warmup ramp and got it back as the
learning rate decayed. The caveat that travels with the reported baseline numbers
(``docs/model_comparison.md``, ``scripts/model_comparison/yolo_baseline/README.md``) is
written from the per-run statistics this script computes, so that the prose cannot
drift from the data: ``tests/test_yolo_warmup_dip_72.py`` pins the statistics and checks
that the table in the README is the one this script prints.

Inputs, all committed, CPU only, no network:

* ``scripts/model_comparison/yolo_baseline/runs/<arm>/results.csv`` -- the eight #51
  grid run directories (six configs; ``y11x_pano_h200`` and ``y26_tiles_l40s`` are
  continuations of ``y11x_pano`` and ``y26_tiles`` and share their early epochs).
* ``docs/data/seed_variance_51_135/y11x_tiles_s{1,2,3}/results.csv`` -- the three
  Campaign A seed replicates (Tillicum H200, seeds 1-3), the only runs of this recipe
  that are not seed 0.
* the ``args.yaml`` beside each of those eleven CSVs -- read as plain text lines, so
  ``--check`` can pin the schedule (``optimizer: auto``, ``lr0: 0.01``,
  ``warmup_epochs: 3.0``) and each run's batch / imgsz / seed without a YAML parser.

Per run it reports: val mAP@50 at epoch 1; the minimum over epochs 2-12 and the epoch
it lands on; the first epoch after that minimum at which mAP@50 is back to at least
its epoch-1 value ("recovered by"); how many epochs before that sat below the epoch-1
value; how many epochs the model emitted no boxes at all (precision = recall = 0);
the epoch at which ``lr/pg0`` peaks and its value; and the epoch of the run's best
``metrics/mAP50-95(B)``, which is what ``best.pt`` selects on in this Ultralytics build.
A run needs at least three epochs (the pre-collapse high, the first drop, the
``lr/pg0`` peak); shorter CSVs are rejected with a ``ValueError``.

    python scripts/analysis/yolo_warmup_dip_72.py             # per-run table, plain text
    python scripts/analysis/yolo_warmup_dip_72.py --markdown  # the table in the README (diff, do not paste)
    python scripts/analysis/yolo_warmup_dip_72.py --check     # exit 1 if any pinned fact fails
    python scripts/analysis/yolo_warmup_dip_72.py --check --markdown  # check first; table only if it passes

``--markdown`` writes LF; a redirected stdout on Windows may write CRLF, so compare the
output against the README block with a diff rather than pasting it in.

The definitions are deliberately simple. "Minimum over epochs 2-12" is a fixed window
rather than a change-point search because the dip is over by epoch 12 on every curve;
"recovered" means back to the epoch-1 value, not converged. Both are readable off
``figures/fig1_learning_curves.png`` directly.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID_DIR = os.path.join(REPO, "scripts", "model_comparison", "yolo_baseline", "runs")
SEED_DIR = os.path.join(REPO, "docs", "data", "seed_variance_51_135")

# Display order: the six grid configs, then the two continuations, then the seed
# replicates. The label column says what each row is a continuation or replicate of.
RUNS = [
    # (name, csv path relative to REPO, note)
    ("y11l_pano", "scripts/model_comparison/yolo_baseline/runs/y11l_pano/results.csv", "YOLO11l pano, batch 4, imgsz 1280, seed 0"),
    ("y11x_pano", "scripts/model_comparison/yolo_baseline/runs/y11x_pano/results.csv", "YOLO11x pano, batch 2, imgsz 1280, seed 0"),
    ("y26_pano", "scripts/model_comparison/yolo_baseline/runs/y26_pano/results.csv", "YOLO26l pano, batch 4, imgsz 1280, seed 0"),
    ("y11l_tiles", "scripts/model_comparison/yolo_baseline/runs/y11l_tiles/results.csv", "YOLO11l tiles, batch 6, imgsz 1024, seed 0"),
    ("y11x_tiles", "scripts/model_comparison/yolo_baseline/runs/y11x_tiles/results.csv", "YOLO11x tiles, batch 12, imgsz 1024, seed 0"),
    ("y26_tiles", "scripts/model_comparison/yolo_baseline/runs/y26_tiles/results.csv", "YOLO26l tiles, batch 6, imgsz 1024, seed 0"),
    ("y11x_pano_h200", "scripts/model_comparison/yolo_baseline/runs/y11x_pano_h200/results.csv", "y11x_pano resumed on Tillicum (shares epochs 1-21)"),
    ("y26_tiles_l40s", "scripts/model_comparison/yolo_baseline/runs/y26_tiles_l40s/results.csv", "y26_tiles resumed on gpu-l40s (shares epochs 1-3)"),
    ("y11x_tiles_s1", "docs/data/seed_variance_51_135/y11x_tiles_s1/results.csv", "y11x_tiles replicate, seed 1, H200"),
    ("y11x_tiles_s2", "docs/data/seed_variance_51_135/y11x_tiles_s2/results.csv", "y11x_tiles replicate, seed 2, H200"),
    ("y11x_tiles_s3", "docs/data/seed_variance_51_135/y11x_tiles_s3/results.csv", "y11x_tiles replicate, seed 3, H200"),
]
GRID_CONFIGS = ("y11l_pano", "y11x_pano", "y26_pano", "y11l_tiles", "y11x_tiles", "y26_tiles")
CONTINUATIONS = ("y11x_pano_h200", "y26_tiles_l40s")
SEED_REPLICATES = ("y11x_tiles_s1", "y11x_tiles_s2", "y11x_tiles_s3")

# What the caveat says every run trained under, and the grid it says was covered. Each
# run's ``args.yaml`` sits beside its ``results.csv``; ``check()`` matches these as plain
# text lines (Ultralytics writes ``key: value``, one per line) so no YAML parser is needed.
SCHEDULE_ARGS = ("optimizer: auto", "lr0: 0.01", "warmup_epochs: 3.0")
GRID_ARGS = {  # name -> (batch, imgsz, seed)
    "y11l_pano": (4, 1280, 0),
    "y11x_pano": (2, 1280, 0),
    "y26_pano": (4, 1280, 0),
    "y11l_tiles": (6, 1024, 0),
    "y11x_tiles": (12, 1024, 0),
    "y26_tiles": (6, 1024, 0),
    "y11x_pano_h200": (2, 1280, 0),
    "y26_tiles_l40s": (6, 1024, 0),
    "y11x_tiles_s1": (12, 1024, 1),
    "y11x_tiles_s2": (12, 1024, 2),
    "y11x_tiles_s3": (12, 1024, 3),
}

MAP50 = "metrics/mAP50(B)"
MAP5095 = "metrics/mAP50-95(B)"
PRECISION = "metrics/precision(B)"
RECALL = "metrics/recall(B)"
LR = "lr/pg0"
VAL_CLS = "val/cls_loss"
TRAIN_CLS = "train/cls_loss"

DIP_WINDOW_LAST_EPOCH = 12  # the dip is over by here on every committed curve
MIN_EPOCHS = 3  # ep1 (the pre-collapse high), ep2 (already lower), ep3 (the lr/pg0 peak)


def read_results_csv(path):
    """Rows of an Ultralytics results.csv as dicts, header whitespace stripped."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows]


def summarize(path):
    """Per-run dip statistics. Epochs are 1-based, as in the CSV's ``epoch`` column."""
    rows = read_results_csv(path)
    epochs = [int(r["epoch"]) for r in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: epochs are not 1..N contiguous: {epochs[:5]}...")
    if len(rows) < MIN_EPOCHS:
        raise ValueError(
            f"{path}: {len(rows)} epoch(s); the dip statistics need at least {MIN_EPOCHS} "
            f"(epoch 1 is the pre-collapse high, epoch 3 the lr/pg0 peak)")
    map50 = [float(r[MAP50]) for r in rows]
    map5095 = [float(r[MAP5095]) for r in rows]
    prec = [float(r[PRECISION]) for r in rows]
    rec = [float(r[RECALL]) for r in rows]
    lr = [float(r[LR]) for r in rows]
    val_cls = [float(r[VAL_CLS]) for r in rows]
    train_cls = [float(r[TRAIN_CLS]) for r in rows]

    n = len(rows)
    window = range(1, min(n, DIP_WINDOW_LAST_EPOCH))  # indices of epochs 2..12
    i_min = min(window, key=lambda i: map50[i])
    i_rec = next((i for i in range(i_min + 1, n) if map50[i] >= map50[0]), None)
    below = sum(1 for i in range(1, i_rec if i_rec is not None else n) if map50[i] < map50[0])
    no_box_epochs = [i + 1 for i in range(n) if prec[i] == 0.0 and rec[i] == 0.0]
    i_lr_peak = max(range(n), key=lambda i: lr[i])
    i_best = max(range(n), key=lambda i: (map5095[i], -i))  # ties to the earlier epoch
    i_vcls = max(range(n), key=lambda i: val_cls[i])
    i_map50_max = max(range(n), key=lambda i: (map50[i], -i))
    # Training-side reading of the collapse: the largest single-epoch rise in train
    # cls_loss over epochs 2-6, and the most it sits above its epoch-2 value over epochs
    # 3-6 (the pano arms tick up 0.02-0.07 at epochs 3-4; the y11 tiles arms are monotone).
    early = range(1, min(n, 6))  # indices of epochs 2..6
    train_cls_max_uptick = max(train_cls[i] - train_cls[i - 1] for i in early)
    train_cls_rise_over_ep2 = max(train_cls[i] - train_cls[1] for i in range(2, min(n, 6)))
    return {
        "epochs": n,
        "map50_ep1": map50[0],
        "map50_ep2": map50[1],
        "map50_ep3": map50[2],
        "map50_max": map50[i_map50_max],
        "map50_max_epoch": i_map50_max + 1,
        "dip_min": map50[i_min],
        "dip_epoch": i_min + 1,
        "dip_depth": map50[0] - map50[i_min],
        "recovered_epoch": None if i_rec is None else i_rec + 1,
        "recovered_map50": None if i_rec is None else map50[i_rec],
        "epochs_below_ep1": below,
        "no_box_epochs": no_box_epochs,
        "lr_peak_epoch": i_lr_peak + 1,
        "lr_peak": lr[i_lr_peak],
        "lr_ep1": lr[0],
        "best_map5095_epoch": i_best + 1,
        "best_map5095": map5095[i_best],
        "val_cls_peak": val_cls[i_vcls],
        "val_cls_peak_epoch": i_vcls + 1,
        "val_cls_ep1": val_cls[0],
        "train_cls_ep1": train_cls[0],
        "train_cls_ep8": train_cls[7] if n > 7 else None,
        "train_cls_max_uptick": train_cls_max_uptick,
        "train_cls_rise_over_ep2": train_cls_rise_over_ep2,
        "precision": prec,
        "recall": rec,
        "map50": map50,
    }


def read_args_lines(csv_path):
    """The ``key: value`` lines of the ``args.yaml`` beside a run's ``results.csv``.

    Plain text, whitespace-stripped, one entry per line -- Ultralytics writes the file
    flat, so a line match is enough and no YAML dependency is needed.
    """
    path = os.path.join(os.path.dirname(csv_path), "args.yaml")
    with open(path, encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def summarize_all(repo=REPO):
    out = {}
    for name, rel, note in RUNS:
        csv_path = os.path.join(repo, rel)
        out[name] = dict(summarize(csv_path), note=note, args=read_args_lines(csv_path))
    return out


def _fmt_rec(s):
    if s["recovered_epoch"] is None:
        return "not within run"
    return f"ep{s['recovered_epoch']} ({s['recovered_map50']:.3f})"


def _fmt_no_box(name, epochs):
    """``N (epA-epB)`` -- a span, so it refuses a non-contiguous set rather than misreport it."""
    if not epochs:
        return "0"
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise ValueError(f"{name}: no-box epochs {epochs} are not contiguous; the span format would misreport them")
    return f"{len(epochs)} (ep{epochs[0]}-{epochs[-1]})"


def markdown_table(stats):
    """The table committed in yolo_baseline/README.md. Regenerate, do not hand-edit."""
    lines = [
        "| run | epochs | mAP50 ep1 | ep2 | ep3 | dip minimum (epoch) | back to ep1 level | epochs below ep1 | no-box epochs | best mAP50-95 epoch |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, s in stats.items():
        lines.append(
            f"| `{name}` | {s['epochs']} | {s['map50_ep1']:.3f} | {s['map50_ep2']:.3f} | {s['map50_ep3']:.3f} "
            f"| {s['dip_min']:.3f} (ep{s['dip_epoch']}) | {_fmt_rec(s)} | {s['epochs_below_ep1']} "
            f"| {_fmt_no_box(name, s['no_box_epochs'])} | {s['best_map5095_epoch']} |"
        )
    return "\n".join(lines) + "\n"


def check(stats):
    """The facts the caveat prose states, as assertions. Returns a list of failures."""
    failures = []

    def expect(cond, msg):
        if not cond:
            failures.append(msg)

    for name, s in stats.items():
        # The schedule and the grid cell, from the run's own args.yaml (plain-text lines).
        batch, imgsz, seed = GRID_ARGS[name]
        for line in SCHEDULE_ARGS + (f"batch: {batch}", f"imgsz: {imgsz}", f"seed: {seed}"):
            expect(line in s["args"], f"{name}: args.yaml has no line `{line}`")

        expect(s["lr_peak_epoch"] == 3 and abs(s["lr_peak"] - 0.0290) < 0.0005,
               f"{name}: lr/pg0 should peak at epoch 3 at 0.029, got ep{s['lr_peak_epoch']} {s['lr_peak']:.4f}")
        expect(abs(s["lr_ep1"] - 0.0100) < 0.0005, f"{name}: lr/pg0 at epoch 1 should be 0.010")
        expect(s["map50_ep2"] < s["map50_ep1"], f"{name}: mAP50 should already be falling at epoch 2")
        expect(3 <= s["dip_epoch"] <= 6, f"{name}: dip minimum should land at epoch 3-6, got ep{s['dip_epoch']}")
        expect(s["dip_depth"] >= 0.35, f"{name}: dip depth {s['dip_depth']:.3f} < 0.35")
        # 0.255 is what the prose claims ("0.00-0.25"). y26_tiles_l40s is the one
        # exception: it resumed from y26_tiles' epoch-3 checkpoint on other hardware, so
        # its epoch 4 is a different epoch 4 (0.268 vs the parent's 0.251) and gets its own
        # bound below.
        dip_bound = 0.27 if name in CONTINUATIONS else 0.255
        expect(s["dip_min"] <= dip_bound, f"{name}: dip minimum {s['dip_min']:.3f} > {dip_bound}")
        expect(s["recovered_epoch"] is not None, f"{name}: never regained its epoch-1 mAP50")
        expect(s["recovered_epoch"] is not None and s["best_map5095_epoch"] > s["recovered_epoch"],
               f"{name}: best mAP50-95 epoch {s['best_map5095_epoch']} is not after recovery")
        # Epoch 1 is the pre-collapse high, not the run's peak: every run's global
        # mAP50 maximum comes after the recovery.
        expect(s["recovered_epoch"] is not None and s["map50_max_epoch"] > s["recovered_epoch"],
               f"{name}: global mAP50 max at ep{s['map50_max_epoch']} is not after recovery")
        expect(s["val_cls_peak_epoch"] in range(3, 7),
               f"{name}: val/cls_loss should peak inside the dip (ep3-6), got ep{s['val_cls_peak_epoch']}")
        # Validation cls_loss rises 1.7-7.3x to its peak while training cls_loss stays
        # flat: at most +0.07 above its epoch-2 value over epochs 3-6, no single-epoch rise
        # above 0.1, and below its epoch-1 value by epoch 8.
        ratio = s["val_cls_peak"] / s["val_cls_ep1"]
        expect(1.7 <= ratio <= 7.5, f"{name}: val/cls_loss peak is {ratio:.2f}x epoch 1, expected 1.7-7.3x")
        expect(s["train_cls_max_uptick"] <= 0.10,
               f"{name}: train/cls_loss rose {s['train_cls_max_uptick']:+.3f} in one epoch over epochs 2-6")
        expect(s["train_cls_rise_over_ep2"] <= 0.075,
               f"{name}: train/cls_loss sits {s['train_cls_rise_over_ep2']:+.3f} above its epoch-2 value, expected <= 0.07")
        expect(s["train_cls_ep8"] is not None and s["train_cls_ep8"] < s["train_cls_ep1"],
               f"{name}: train/cls_loss at epoch 8 should be below epoch 1")

    # The six grid configs: epoch-1 high 0.65-0.78, minimum 0.00-0.25.
    grid = [stats[n] for n in GRID_CONFIGS]
    expect(0.645 <= min(s["map50_ep1"] for s in grid) and max(s["map50_ep1"] for s in grid) <= 0.785,
           "grid epoch-1 mAP50 should span 0.65-0.78")
    expect(min(s["dip_min"] for s in grid) == 0.0 and max(s["dip_min"] for s in grid) <= 0.255,
           "grid dip minima should span 0.00-0.25")
    # Pano arms take far longer to recover than tiles arms.
    for n in ("y11l_pano", "y11x_pano", "y26_pano"):
        expect(15 <= stats[n]["recovered_epoch"] <= 22, f"{n}: pano recovery expected at ep15-22")
    for n in ("y11l_tiles", "y11x_tiles", "y26_tiles"):
        expect(5 <= stats[n]["recovered_epoch"] <= 8, f"{n}: tiles recovery expected at ep5-8")

    # The failure signature: recall collapse, not a false-positive flood.
    expect(stats["y11x_pano"]["no_box_epochs"] == [3, 4, 5, 6, 7],
           "y11x_pano should emit no boxes at epochs 3-7")
    expect(stats["y11l_pano"]["no_box_epochs"] == [4, 5], "y11l_pano should emit no boxes at epochs 4-5")
    p, r = stats["y11l_pano"]["precision"], stats["y11l_pano"]["recall"]
    expect(all(p[i] >= 0.94 for i in range(5, 9)) and all(r[i] <= 0.03 for i in range(5, 9)),
           "y11l_pano epochs 6-9 should hold precision >= 0.94 at recall <= 0.03")
    for n in ("y11l_pano", "y26_pano"):
        p3, r1, r3 = stats[n]["precision"][2], stats[n]["recall"][0], stats[n]["recall"][2]
        expect(p3 >= 0.82, f"{n}: precision at epoch 3 should still be >= 0.82, got {p3:.3f}")
        expect(r3 <= r1 - 0.10, f"{n}: recall should fall at epoch 3 (ep1 {r1:.3f} -> ep3 {r3:.3f})")
    expect(stats["y11l_pano"]["recall"][2] <= 0.04,
           "y11l_pano at epoch 3 should be at recall ~0.038")

    # The seed replicates reproduce the dip: minimum at epoch 3, back by epoch 5.
    for n in SEED_REPLICATES:
        expect(stats[n]["dip_epoch"] == 3 and 0.10 <= stats[n]["dip_min"] <= 0.12,
               f"{n}: expected minimum 0.10-0.12 at epoch 3")
        expect(stats[n]["recovered_epoch"] == 5, f"{n}: expected recovery at epoch 5")
        expect(stats[n]["no_box_epochs"] == [], f"{n}: no zero-box epochs expected")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="With both --check and --markdown, the check runs first and the table is "
               "printed only if it passes (exit 1 otherwise, no table).")
    ap.add_argument("--markdown", action="store_true", help="print the README table (LF; diff it against the README block)")
    ap.add_argument("--check", action="store_true", help="assert the pinned facts; exit 1 on failure")
    args = ap.parse_args(argv)

    stats = summarize_all()
    if args.check:
        failures = check(stats)
        out = sys.stderr if args.markdown else sys.stdout  # keep stdout clean for the table
        for f in failures:
            print("FAIL:", f, file=out)
        print("ok" if not failures else f"{len(failures)} failure(s)", file=out)
        if failures:
            return 1
        if not args.markdown:
            return 0
    if args.markdown:
        sys.stdout.write(markdown_table(stats))
        return 0
    for name, s in stats.items():
        nb = s["no_box_epochs"]
        print(f"{name:16s} n={s['epochs']:2d}  ep1 {s['map50_ep1']:.3f}  ep2 {s['map50_ep2']:.3f}  "
              f"ep3 {s['map50_ep3']:.3f}  min {s['dip_min']:.3f}@ep{s['dip_epoch']}  "
              f"back {_fmt_rec(s):18s} below-ep1 {s['epochs_below_ep1']:2d}  "
              f"no-box {len(nb)}  lr peak ep{s['lr_peak_epoch']} {s['lr_peak']:.4f}  "
              f"val/cls peak {s['val_cls_peak']:.2f}@ep{s['val_cls_peak_epoch']}  "
              f"best mAP50-95 ep{s['best_map5095_epoch']}  -- {s['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
