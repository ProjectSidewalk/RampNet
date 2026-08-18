#!/usr/bin/env python3
"""Assemble the `manual_gold` half of the Stage 2 epoch curve (#84).

`stage_two/evaluate.py --threshold 0.0` writes, per checkpoint, a precision/recall value
at every unique confidence. This reads one such file per epoch and produces the two
numbers the #84 pre-registration asks for:

- **F1@0.30** — the #54 protocol operating point, held fixed so the column is comparable
  to every other number in `docs/model_comparison.md`.
- **max-F1** — the calibration-free reading. Holding a threshold fixed across a training
  curve confounds capability with calibration, so the epoch-peak question is answered
  here. If the two columns peak at different epochs, that is itself a finding.

It also writes a **downsampled** copy of each PR-vs-confidence curve on a fixed
confidence grid. The full curves are ~4 MB each (80k rows) and cannot reasonably live in
the repo; the downsampled ones are a few KB, re-derive F1 at any threshold to 3 decimal
places, and mean the committed table is checkable without cluster access.

Usage:

    # on the machine that ran the eval
    python scripts/analysis/stage2_manual_gold_curve.py \\
        --results-root /homes/gws/jonf/RampNet/run_a_84/evaluation_results \\
        --out-dir docs/data/run_a_84_manual_gold

    # afterwards, anywhere, from the committed downsampled curves
    python scripts/analysis/stage2_manual_gold_curve.py \\
        --results-root docs/data/run_a_84_manual_gold --downsampled
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The #54 protocol operating point. Held fixed across the curve on purpose.
PROTOCOL_THRESHOLD = 0.30

# Confidence grid for the committed downsampled curves. 0.005 steps over [0, 1] keeps
# every file a few KB while resolving F1 to ~3 decimals, which is well below the 0.01
# tie bar the pre-registration set.
GRID_STEP = 0.005

PR_FILENAME = "pr_rc_vs_c_data_manual_r0.022_pt0.0.csv"
METRICS_FILENAME = "metrics_manual_r0.022_pt0.0.json"


def f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def read_curve(path: Path) -> list[tuple[float, float, float]]:
    """Return [(confidence, precision, recall)] sorted by confidence."""
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    float(row["confidence_threshold"]),
                    float(row["precision"]),
                    float(row["recall"]),
                )
            )
    rows.sort()
    return rows


def downsample(curve: list[tuple[float, float, float]], step: float = GRID_STEP):
    """Snap the curve to a fixed confidence grid, taking the last row at or below each
    grid point -- i.e. the operating point you would actually get by thresholding there.
    """
    out = []
    idx = 0
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        target = i * step
        while idx + 1 < len(curve) and curve[idx + 1][0] <= target:
            idx += 1
        if curve[idx][0] <= target:
            out.append((target, curve[idx][1], curve[idx][2]))
    return out


def summarize(curve: list[tuple[float, float, float]]) -> dict:
    at_protocol = max((r for r in curve if r[0] <= PROTOCOL_THRESHOLD), default=curve[0],
                      key=lambda r: r[0])
    best = max(curve, key=lambda r: f1(r[1], r[2]))
    return {
        "f1_at_protocol": f1(at_protocol[1], at_protocol[2]),
        "precision_at_protocol": at_protocol[1],
        "recall_at_protocol": at_protocol[2],
        "max_f1": f1(best[1], best[2]),
        "max_f1_conf": best[0],
        "precision_at_max_f1": best[1],
        "recall_at_max_f1": best[2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="directory holding epoch_N/ subdirectories")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write downsampled curves + summary here")
    parser.add_argument("--downsampled", action="store_true",
                        help="read already-downsampled curves (epoch_N.csv) instead of eval output")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    rows = []
    for epoch in range(1, args.epochs + 1):
        if args.downsampled:
            path = args.results_root / f"epoch_{epoch}.csv"
        else:
            path = args.results_root / f"epoch_{epoch}" / PR_FILENAME
        if not path.exists():
            print(f"epoch {epoch}: MISSING {path} -- not scored yet")
            continue

        curve = read_curve(path)
        summary = summarize(curve)
        summary["epoch"] = epoch
        summary["n_points"] = len(curve)

        metrics_path = args.results_root / f"epoch_{epoch}" / METRICS_FILENAME
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as handle:
                blob = json.load(handle)
            summary["ap"] = blob.get("ap")
            # evaluate.py stamps the checkpoint's fingerprint into its own metrics file,
            # so the row can be tied back to the exact weights that produced it rather
            # than to a directory name someone could have moved.
            summary["checkpoint_fingerprint"] = blob.get("checkpoint_fingerprint", "")
            summary["tta"] = blob.get("tta")
            if blob.get("tta"):
                print(f"epoch {epoch}: WARNING scored WITH TTA -- the #54 protocol headline "
                      "is single-pass, so this row is not comparable to the rest")
        rows.append(summary)

        if args.out_dir and not args.downsampled:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            small = downsample(curve)
            with (args.out_dir / f"epoch_{epoch}.csv").open("w", newline="\n", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["confidence_threshold", "precision", "recall"])
                for conf, precision, recall in small:
                    writer.writerow([f"{conf:.3f}", f"{precision:.6f}", f"{recall:.6f}"])

    if not rows:
        raise SystemExit("no epochs scored yet")

    print()
    print("| epoch | F1@0.30 | P | R | max-F1 | at conf | AP |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    best_protocol = max(rows, key=lambda r: r["f1_at_protocol"])
    best_max = max(rows, key=lambda r: r["max_f1"])
    for row in rows:
        a = "**" if row is best_protocol else ""
        b = "**" if row is best_max else ""
        ap = f"{row['ap']:.4f}" if row.get("ap") is not None else ""
        print(f"| {row['epoch']} | {a}{row['f1_at_protocol']:.4f}{a} | {row['precision_at_protocol']:.4f} | "
              f"{row['recall_at_protocol']:.4f} | {b}{row['max_f1']:.4f}{b} | {row['max_f1_conf']:.3f} | {ap} |")

    print()
    print(f"F1@0.30 peaks at epoch {best_protocol['epoch']}  ({best_protocol['f1_at_protocol']:.4f})")
    print(f"max-F1  peaks at epoch {best_max['epoch']}  ({best_max['max_f1']:.4f})")
    if best_protocol["epoch"] != best_max["epoch"]:
        print("NOTE: the two columns peak at DIFFERENT epochs -- calibration is moving, "
              "which the pre-registration flags as a finding in its own right.")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.out_dir / "summary.csv"
        fields = ["epoch", "f1_at_protocol", "precision_at_protocol", "recall_at_protocol",
                  "max_f1", "max_f1_conf", "precision_at_max_f1", "recall_at_max_f1", "ap",
                  "checkpoint_fingerprint"]
        with summary_path.open("w", newline="\n", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{row[k]:.6f}" if isinstance(row.get(k), float) else row.get(k, ""))
                                 for k in fields})
        print(f"\nWrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
