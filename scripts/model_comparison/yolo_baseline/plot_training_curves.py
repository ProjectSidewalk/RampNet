#!/usr/bin/env python3
"""Training-diagnostic figures for the supervised-YOLO baseline (issue #51).

Reads the committed per-epoch records in ``runs/<config>/results.csv`` (pulled from
the Hyak run directories, which live on ``/gscratch/scrubbed`` and auto-purge) and
writes the standard set of figures used to diagnose a detection training run:

  fig1_learning_curves   val mAP50 / mAP50-95 vs epoch, with the LR schedule overlaid
  fig2_precision_recall  val precision and recall vs epoch, separately
  fig3_loss_divergence   train vs val loss per component, one panel per config
  fig4_per_config        per-config mAP50 with LR and preemption/resume markers

Everything here is CPU-only and reads nothing but the committed CSVs, so the
figures are reproducible from a clean checkout:

    python scripts/model_comparison/yolo_baseline/plot_training_curves.py

Why the LR overlay is on the headline figure: every config's validation collapse
begins at the epoch where the warmup ramp peaks, and recovery tracks the decay.
See README.md for the reading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

# Display order and styling. Geometry picks the linestyle, model family the colour,
# so "is this a pano or a tiles run?" and "is this v11 or v26?" are both readable
# without consulting the legend.
CONFIGS = [
    # name,               label,                        colour,    linestyle
    ("y26_pano",        "YOLO26  pano  (b4, 1280)",       "#1b6ca8", "-"),
    ("y11l_pano",       "YOLO11l pano  (b4, 1280)",       "#c1121f", "-"),
    ("y11x_pano",       "YOLO11x pano  (b2, 1280)",       "#e07a5f", "-"),
    ("y11x_pano_h200",  "YOLO11x pano  (b2, 1280) H200",  "#f2a541", "-"),
    ("y26_tiles",       "YOLO26  tiles (b6, 1024)",       "#457b9d", "--"),
    ("y11l_tiles",      "YOLO11l tiles (b6, 1024)",       "#8b1e3f", "--"),
    ("y11x_tiles",      "YOLO11x tiles (b12, 1024)",      "#6a4c93", "--"),
    ("y26_tiles_l40s",  "YOLO26  tiles (b6, 1024) L40S",  "#7fb069", "--"),
]

# Two of the eight are continuations of another arm rather than independent configs, and
# a reader comparing curves needs to know which: y11x_pano_h200 is y11x_pano resumed on
# Tillicum at ep21, and y26_tiles_l40s is y26_tiles resumed on a dedicated L40S at ep4.
# Both are plotted because each is the furthest-advanced member of its lineage -- the
# h200 arm is the only completed 60-epoch pano run -- but they share early epochs with
# their parents and are not replicates. See README, "The y26_tiles_l40s fork".

MAP50 = "metrics/mAP50(B)"
MAP5095 = "metrics/mAP50-95(B)"
PRECISION = "metrics/precision(B)"
RECALL = "metrics/recall(B)"
LR = "lr/pg0"


def load_runs(runs_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every config that has a results.csv, preserving CONFIGS order.

    A config with no results.csv is skipped with a note rather than an error --
    y11x_tiles was dropped before it finished epoch 1 and has args.yaml only, and
    that absence is itself part of the record.
    """
    runs: dict[str, pd.DataFrame] = {}
    for name, *_ in CONFIGS:
        path = runs_dir / name / "results.csv"
        if not path.exists():
            print(f"  skip {name}: no results.csv (dropped before epoch 1?)")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        runs[name] = df
    return runs


def resume_epochs(df: pd.DataFrame) -> list[int]:
    """Epochs at which the training process restarted.

    Ultralytics writes ``time`` as seconds elapsed in the *current* process, so it
    resets on every requeue. A decrease therefore marks a ckpt preemption + resume.
    """
    t = df["time"].to_numpy()
    return [int(df["epoch"].iloc[i]) for i in range(1, len(t)) if t[i] < t[i - 1]]


def warmup_peak_epoch(df: pd.DataFrame) -> int | None:
    """Epoch where the LR schedule peaks (end of the warmup ramp)."""
    if LR not in df.columns or df[LR].isna().all():
        return None
    return int(df["epoch"].iloc[int(df[LR].to_numpy().argmax())])


def _style(ax, xlabel="epoch", ylabel=None, title=None):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def fig_learning_curves(runs, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for metric, ax, label in ((MAP50, axes[0], "val mAP@50"),
                              (MAP5095, axes[1], "val mAP@50-95")):
        for name, lbl, colour, ls in CONFIGS:
            if name not in runs:
                continue
            df = runs[name]
            ax.plot(df["epoch"], df[metric], color=colour, linestyle=ls,
                    marker="o", markersize=3.5, linewidth=1.8, label=lbl, zorder=3)
        _style(ax, ylabel=label, title=label + " vs epoch")
        ax.set_ylim(bottom=-0.02)

    # LR overlay on the headline panel only -- the point is the alignment, and
    # repeating it on both panels just adds ink.
    ref = next(iter(runs.values()))
    twin = axes[0].twinx()
    twin.plot(ref["epoch"], ref[LR], color="#666666", linestyle=":", linewidth=1.6,
              label="learning rate", zorder=1)
    twin.set_ylabel("learning rate (lr/pg0)", color="#666666")
    twin.tick_params(axis="y", colors="#666666")
    twin.spines[["top"]].set_visible(False)

    peak = warmup_peak_epoch(ref)
    if peak is not None:
        for ax in axes:
            ax.axvline(peak, color="#666666", linestyle="-", linewidth=1.0, alpha=0.5, zorder=1)
        axes[0].annotate(
            f"warmup peak\n(epoch {peak}, lr={ref[LR].max():.3f})",
            xy=(peak, axes[0].get_ylim()[1] * 0.92), xytext=(peak + 0.5, axes[0].get_ylim()[1] * 0.80),
            fontsize=8.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.9),
        )

    axes[0].legend(fontsize=8, loc="center right", framealpha=0.92)
    fig.suptitle(
        "Supervised YOLO baseline (#51): every config peaks before the warmup LR peak, "
        "collapses at it, and recovers as the LR decays",
        fontsize=11.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_precision_recall(runs, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for metric, ax, label in ((PRECISION, axes[0], "val precision"),
                              (RECALL, axes[1], "val recall")):
        for name, lbl, colour, ls in CONFIGS:
            if name not in runs:
                continue
            df = runs[name]
            ax.plot(df["epoch"], df[metric], color=colour, linestyle=ls,
                    marker="o", markersize=3.5, linewidth=1.8, label=lbl)
        _style(ax, ylabel=label, title=label + " vs epoch")
        ax.set_ylim(-0.03, 1.05)

    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.92)
    # Precision reads 0 where a model emits *no* boxes at all (y11x_pano from epoch 3),
    # which is the 0/0 convention, not a flood of false positives. y11l_pano is the
    # legible case: precision holds at 0.94-1.00 while recall sits near 0.01.
    fig.suptitle(
        "The failure signature is a RECALL collapse: the model stops firing rather than "
        "starting to guess\n(precision holds high, or reads 0 where nothing is emitted at all)",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_loss_divergence(runs, out: Path) -> None:
    """Train vs val loss per config -- the generalization-gap view.

    YOLO26 logs an l1_loss where YOLO11 logs dfl_loss, so the third component is
    resolved per config rather than assumed.
    """
    names = [n for n, *_ in CONFIGS if n in runs]
    fig, axes = plt.subplots(2, len(names), figsize=(3.3 * len(names), 6.6), squeeze=False)

    for col, name in enumerate(names):
        df = runs[name]
        lbl = next(c[1] for c in CONFIGS if c[0] == name)
        for row, comp in enumerate(("box_loss", "cls_loss")):
            ax = axes[row][col]
            ax.plot(df["epoch"], df[f"train/{comp}"], color="#2a9d8f", linewidth=1.8,
                    marker="o", markersize=3, label="train")
            ax.plot(df["epoch"], df[f"val/{comp}"], color="#c1121f", linewidth=1.8,
                    linestyle="--", marker="s", markersize=3, label="val")
            peak = warmup_peak_epoch(df)
            if peak is not None:
                ax.axvline(peak, color="#888888", linewidth=1.0, alpha=0.5)
            _style(ax, ylabel=comp if col == 0 else None,
                   title=lbl.split("(")[0].strip() if row == 0 else None)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Train vs validation loss: training loss keeps falling while validation "
        "cls_loss explodes -- an optimization instability, not a crash or a data fault",
        fontsize=11.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_per_config(runs, out: Path) -> None:
    """One panel per config: mAP50 + LR + ckpt preemption markers."""
    names = [n for n, *_ in CONFIGS if n in runs]
    ncol = min(3, len(names))
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)

    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        df = runs[name]
        lbl, colour, ls = next((c[1], c[2], c[3]) for c in CONFIGS if c[0] == name)

        ax.plot(df["epoch"], df[MAP50], color=colour, linestyle=ls, marker="o",
                markersize=4, linewidth=2, label="val mAP@50", zorder=3)

        twin = ax.twinx()
        twin.plot(df["epoch"], df[LR], color="#666666", linestyle=":", linewidth=1.4, zorder=1)
        twin.set_ylabel("lr", color="#666666", fontsize=9)
        twin.tick_params(axis="y", colors="#666666", labelsize=8)
        twin.spines[["top"]].set_visible(False)

        for j, ep in enumerate(resume_epochs(df)):
            ax.axvline(ep, color="#7209b7", linewidth=1.2, alpha=0.55, linestyle="-.",
                       zorder=2, label="ckpt resume" if j == 0 else None)
        peak = warmup_peak_epoch(df)
        if peak is not None:
            ax.axvline(peak, color="#888888", linewidth=1.2, alpha=0.6, zorder=2)

        best = df.loc[df[MAP50].idxmax()]
        ax.annotate(f"best ep{int(best['epoch'])}: {best[MAP50]:.3f}",
                    xy=(best["epoch"], best[MAP50]), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color="#333333")

        _style(ax, ylabel="val mAP@50", title=lbl)
        ax.set_ylim(-0.03, 0.9)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(
        "Per-config detail. Grey line = warmup LR peak, purple = ckpt preemption/resume. "
        "Collapse tracks the LR peak, not the resumes.",
        fontsize=11.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def print_summary(runs) -> None:
    print("\nPer-config summary (internal YOLO val-split proxy, NOT benchmark eval)\n")
    hdr = f"{'config':<15} {'eps':>4} {'best ep':>8} {'best mAP50':>11} {'last mAP50':>11} {'last P':>7} {'last R':>7} {'resumes':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, *_ in CONFIGS:
        if name not in runs:
            print(f"{name:<15} {'--':>4}   (dropped before epoch 1 -- args.yaml only)")
            continue
        df = runs[name]
        best = df.loc[df[MAP50].idxmax()]
        last = df.iloc[-1]
        print(f"{name:<15} {len(df):>4} {int(best['epoch']):>8} {best[MAP50]:>11.3f} "
              f"{last[MAP50]:>11.3f} {last[PRECISION]:>7.3f} {last[RECALL]:>7.3f} "
              f"{len(resume_epochs(df)):>8}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs",
                    help="directory of <config>/results.csv (default: ./runs)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures",
                    help="where to write the PNGs (default: ./figures)")
    args = ap.parse_args()

    print(f"reading {args.runs_dir}")
    runs = load_runs(args.runs_dir)
    if not runs:
        print("no results.csv found -- nothing to plot", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing {args.out_dir}")
    fig_learning_curves(runs, args.out_dir / "fig1_learning_curves.png")
    fig_precision_recall(runs, args.out_dir / "fig2_precision_recall.png")
    fig_loss_divergence(runs, args.out_dir / "fig3_loss_divergence.png")
    fig_per_config(runs, args.out_dir / "fig4_per_config.png")
    print_summary(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
