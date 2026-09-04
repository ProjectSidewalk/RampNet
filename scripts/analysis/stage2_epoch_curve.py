#!/usr/bin/env python3
"""Extract the Stage 2 per-epoch validation-loss curve from TensorBoard event files (#84).

`stage_two/train.py` writes one `Loss/val_epoch` scalar per epoch via
`torch.utils.tensorboard.SummaryWriter`. This script reads those scalars straight out of
the committed event files -- for Run A and for the 2025-06 paper run alike -- and emits
the comparison as CSV plus a markdown table.

Both runs are re-derivable from this repo alone:

- **Run A** wrote `stage_two/run_a_84_events/` (6 files, 4.0 MB).
- **The paper run**'s surviving events were rescued in #104 and are committed at
  `docs/data/rampnet1_stage2_run/` (18 files). Its epoch-N *weights* were deleted in a
  2025-07-11 cleanup -- that is the real gap, and it is why Run A exists -- but the
  scalars survived, so the paper column here is read at full float32 precision rather
  than transcribed.

The TFRecord + protobuf reading is `stage2_train_cost.read_scalars`, in this same
directory: one reader, one set of tests (`tests/test_stage2_train_cost.py`), no second
copy to drift. That reader is standard-library only, so this runs from a clean clone with
no tensorboard install; its output was cross-checked against
`tensorboard.backend.event_processing.EventAccumulator` on klone (2026-08-17), and both
give the same values to full float32 precision.

Every event file is checked against the `SHA256SUMS` committed beside it, so a
regenerated or corrupted copy fails loudly instead of quietly producing a different
curve. `--verify` prints a fresh manifest (that is how SHA256SUMS is regenerated).

Usage:

    python scripts/analysis/stage2_epoch_curve.py \
        --events-dir stage_two/run_a_84_events \
        --out-csv docs/data/stage2_epoch_curve_84.csv

    python scripts/analysis/stage2_epoch_curve.py --verify   # sha256 manifest

The run this reads by default is Run A of #84: 8 epochs of the released Stage 2 recipe,
world size 16, constant lr 1e-5, seed 42. See docs/stage2_epoch_curve_84.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_train_cost import read_scalars as read_event_scalars  # noqa: E402

# train.py logs the val scalar under this tag, once per epoch, keyed by global step.
VAL_TAG = "Loss/val_epoch"

# 150,063 train panoramas with drop_last=True at world size 16. Used only to turn a global
# step back into an epoch number, so the curve can be read without the checkpoint
# filenames. Asserted against the data rather than trusted -- see epoch_curve().
STEPS_PER_EPOCH = 9378

RUN_A_EVENTS = REPO / "stage_two" / "run_a_84_events"
PAPER_EVENTS = REPO / "docs" / "data" / "rampnet1_stage2_run"

MANIFEST_NAME = "SHA256SUMS"


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def read_manifest(events_dir: Path) -> dict[str, str]:
    """Parse a `sha256  filename` manifest. Returns {} when there is none."""
    path = events_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    manifest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip()
        if not name:
            raise SystemExit(f"malformed line in {path}: {line!r}")
        manifest[name] = digest
    return manifest


def read_scalars_by_file(events_dir: Path, tag: str = VAL_TAG, verify: bool = True):
    """Return [(path, sha256, {global_step: value})] for one tag, in chronological order.

    Files sort chronologically because SummaryWriter names them
    `events.out.tfevents.<unix_seconds>.<host>.<pid>.<n>`.

    Keeping the per-file split (rather than flattening straight to {step: value}) is what
    lets a caller see an epoch that was computed twice by two job incarnations after a
    requeue -- that duplicate is a free noise-floor measurement, not a nuisance.
    """
    files = sorted(events_dir.glob("events.out.tfevents.*"))
    if not files:
        raise SystemExit(f"no event files in {events_dir}")

    manifest = read_manifest(events_dir) if verify else {}
    out, mismatched = [], []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest.get(path.name)
        if expected is not None and expected != digest:
            mismatched.append(f"  {path.name}\n    expected {expected}\n    found    {digest}")
        points = {step: value
                  for _wall, step, this_tag, value in read_event_scalars(path)
                  if this_tag == tag}
        out.append((path, digest, points))

    missing = sorted(set(manifest) - {p.name for p in files})
    if mismatched or missing:
        detail = "\n".join(mismatched)
        if missing:
            detail += "\n  listed in the manifest but absent: " + ", ".join(missing)
        raise SystemExit(
            f"{events_dir / MANIFEST_NAME} does not match the files on disk:\n{detail}\n"
            "These are checksummed replication artifacts. Re-fetch them rather than "
            "regenerating the manifest, unless you meant to replace the run."
        )
    return out


def read_scalars(events_dir: Path, tag: str = VAL_TAG, verify: bool = True) -> dict[int, float]:
    """Return {global_step: value} for one scalar tag across every event file in a dir.

    A resumed run re-emits steps it had already written, so later files deliberately win
    on collision -- that is the value the run finished with.
    """
    scalars: dict[int, float] = {}
    for _path, _digest, points in read_scalars_by_file(events_dir, tag, verify):
        scalars.update(points)
    return scalars


def epoch_of(step: int, steps_per_epoch: int = STEPS_PER_EPOCH) -> int:
    """Turn a global step into an epoch number, refusing to guess.

    train.py validates at an epoch boundary, so every `Loss/val_epoch` step is an exact
    multiple of steps/epoch. If it is not, the steps/epoch constant does not describe this
    run -- a different train-set size or world size -- and rounding would hand back
    plausible-looking but wrong epoch numbers. Fail instead.
    """
    if steps_per_epoch < 1:
        raise SystemExit(f"--steps-per-epoch must be >= 1, got {steps_per_epoch}")
    if step % steps_per_epoch:
        raise SystemExit(
            f"validation step {step:,} is not a multiple of {steps_per_epoch:,} steps/epoch.\n"
            "  That constant is 150,063 train panoramas / world size 16 (drop_last=True). "
            "A run with a different dataset size or world size needs --steps-per-epoch set "
            "to match, otherwise every epoch label below would be wrong."
        )
    return step // steps_per_epoch


def epoch_curve(scalars: dict[int, float], steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[int, float]:
    """{epoch: value}, one entry per epoch."""
    return {epoch_of(step, steps_per_epoch): value for step, value in sorted(scalars.items())}


def read_curve(events_dir: Path, tag: str = VAL_TAG,
               steps_per_epoch: int = STEPS_PER_EPOCH, verify: bool = True) -> dict[int, float]:
    return epoch_curve(read_scalars(events_dir, tag, verify), steps_per_epoch)


def read_paper_curve(events_dir: Path = PAPER_EVENTS, tag: str = VAL_TAG,
                     steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[int, float]:
    """The 2025-06 paper run's curve, from the events rescued in #104.

    Returns {} rather than raising when the directory is absent, so the script still
    produces Run A's own curve in a checkout that does not carry the rescued events.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir() or not any(events_dir.glob("events.out.tfevents.*")):
        return {}
    return read_curve(events_dir, tag, steps_per_epoch)


def group_by_epoch(by_file, steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[int, list[float]]:
    """{epoch: [one value per job incarnation that computed it]}, chronological."""
    seen: dict[int, list[float]] = {}
    for _path, _digest, points in by_file:
        for step, value in sorted(points.items()):
            seen.setdefault(epoch_of(step, steps_per_epoch), []).append(value)
    return dict(sorted(seen.items()))


def repeat_measurements(events_dir: Path = RUN_A_EVENTS, tag: str = VAL_TAG,
                        steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[int, list[float]]:
    """{epoch: [value, ...]} for every epoch that more than one job incarnation computed.

    A requeue that lands mid-epoch makes the resumed job re-run that epoch's tail from
    `latest_checkpoint.pth`, so the epoch is validated twice on two nodes from two resume
    points. The spread between the two bounds resume-path nondeterminism and evaluation
    together -- a measurement floor, obtained for free.
    """
    grouped = group_by_epoch(read_scalars_by_file(Path(events_dir), tag), steps_per_epoch)
    return {epoch: values for epoch, values in grouped.items() if len(values) > 1}


def spread_pct(values) -> float:
    """Max-to-min spread of repeated measurements, in percent."""
    return (max(values) / min(values) - 1.0) * 100.0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=RUN_A_EVENTS,
        help="directory of TensorBoard event files (default: %(default)s)",
    )
    parser.add_argument(
        "--paper-events-dir",
        type=Path,
        default=PAPER_EVENTS,
        help="the 2025-06 paper run's rescued events, for the comparison column "
             "(default: %(default)s)",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="write the curve here as CSV")
    parser.add_argument(
        "--curve-label", default="run_a",
        help="name for the --events-dir run, used as a CSV column prefix and in the "
             "printed table (default: %(default)s). Set it when reading a run that is "
             "not Run A, or the artifact claims to be a run it is not.",
    )
    parser.add_argument(
        "--reference-label", default="paper",
        help="name for the --paper-events-dir comparison run (default: %(default)s)",
    )
    parser.add_argument("--tag", default=VAL_TAG, help="scalar tag to extract (default: %(default)s)")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=STEPS_PER_EPOCH,
        help="global steps in one epoch; validation steps must be exact multiples of it "
             "(default: %(default)s = 150,063 train panos / world size 16)",
    )
    parser.add_argument("--verify", action="store_true",
                        help="print a sha256 manifest of the event files and exit")
    args = parser.parse_args()
    # Column names travel with the data: a CSV headed run_a_val_loss that holds some other
    # run is worse than no CSV, because nothing downstream can tell.
    curve_col = f"{args.curve_label}_val_loss"
    ref_col = f"{args.reference_label}_val_loss"

    if args.verify:
        for path in sorted(args.events_dir.glob("events.out.tfevents.*")):
            print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        return 0

    print(f"Reading '{args.tag}' from {args.events_dir}")
    by_file = read_scalars_by_file(args.events_dir, args.tag)
    for path, digest, points in by_file:
        print(f"  {path.name}  sha256={digest[:16]}...  {len(points)} '{args.tag}' point(s)")

    scalars: dict[int, float] = {}
    for _path, _digest, points in by_file:
        scalars.update(points)
    if not scalars:
        raise SystemExit(f"no '{args.tag}' scalars found -- wrong tag or wrong directory?")

    curve = epoch_curve(scalars, args.steps_per_epoch)
    paper = read_paper_curve(args.paper_events_dir, args.tag, args.steps_per_epoch)
    if not paper:
        print(f"  note: no paper-run events under {args.paper_events_dir}; "
              "the comparison column will be empty", file=sys.stderr)

    rows = []
    for epoch in sorted(curve):
        value = curve[epoch]
        reference = paper.get(epoch)
        rows.append({
            "epoch": epoch,
            "step": epoch * args.steps_per_epoch,
            curve_col: value,
            ref_col: reference,
            "delta_pct": (value / reference - 1.0) * 100.0 if reference else None,
        })

    best = min(rows, key=lambda r: r[curve_col])

    print()
    print(f"| epoch | auto-label val loss | {args.reference_label} run | delta | "
          f"vs. {args.curve_label} min |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        mark = "**" if row is best else ""
        paper_txt = f"{row[ref_col]:.8f}" if row[ref_col] else ""
        delta_txt = f"{row['delta_pct']:+.3f}%" if row["delta_pct"] is not None else ""
        gap = (row[curve_col] / best[curve_col] - 1.0) * 100.0
        gap_txt = "min" if row is best else f"+{gap:.1f}%"
        print(
            f"| {row['epoch']} | {mark}{row[curve_col]:.8f}{mark} | {paper_txt} | "
            f"{delta_txt} | {gap_txt} |"
        )

    deltas = [abs(row["delta_pct"]) for row in rows if row["delta_pct"] is not None]
    if deltas:
        print()
        print(f"Largest |delta| vs. the {args.reference_label} run: {max(deltas):.3f}%   "
              f"mean {sum(deltas) / len(deltas):.3f}%  (n={len(deltas)})")

    print()
    print(f"{args.curve_label} minimum:     epoch {best['epoch']}  ({best[curve_col]:.8f})")
    if paper:
        paper_best = min(paper, key=paper.get)
        print(f"{args.reference_label} run minimum: epoch {paper_best}  ({paper[paper_best]:.8f})")
        print("MATCH" if best["epoch"] == paper_best
              else "MISMATCH -- the selection epoch did not replicate")

    repeats = {epoch: values
               for epoch, values in group_by_epoch(by_file, args.steps_per_epoch).items()
               if len(values) > 1}
    if repeats:
        print()
        for epoch, values in repeats.items():
            joined = ", ".join(f"{v:.8f}" for v in values)
            print(f"Epoch {epoch} was computed {len(values)}x (a requeue landed mid-epoch): "
                  f"{joined} -- spread {spread_pct(values):.4f}%")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Line endings are pinned to LF on every platform, so the committed CSV is
        # byte-stable and a regenerated copy can be proven identical rather than assumed
        # to be. Both halves are needed: csv.writer defaults to lineterminator="\r\n" and
        # emits it regardless of how the file was opened, and newline="" would then let
        # the platform translate on top of that.
        with args.out_csv.open("w", newline="\n", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "step", curve_col, ref_col, "delta_pct"],
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "epoch": row["epoch"],
                        "step": row["step"],
                        curve_col: f"{row[curve_col]:.8f}",
                        ref_col: f"{row[ref_col]:.8f}" if row[ref_col] else "",
                        "delta_pct": f"{row['delta_pct']:.3f}" if row["delta_pct"] is not None else "",
                    }
                )
        print(f"\nWrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
