#!/usr/bin/env python3
"""Extract the Stage 2 per-epoch validation-loss curve from TensorBoard event files (#84).

`stage_two/train.py` writes one `Loss/val_epoch` scalar per epoch via
`torch.utils.tensorboard.SummaryWriter`. This script reads those scalars straight out of
the committed event files and emits the curve as CSV plus a markdown table.

Why it parses the files itself instead of importing tensorboard: the repo's test suite is
deliberately CPU-only and dependency-light, and this needs to run from a clean clone
without installing a ~100 MB package to read 8 floating-point numbers. The TFRecord
framing and the two protobuf messages involved are small enough to read directly, and the
output was cross-checked against `tensorboard.backend.event_processing.EventAccumulator`
on klone (2026-08-17) -- both readers give the same 8 values to full float32 precision.

Usage:

    python scripts/analysis/stage2_epoch_curve.py \
        --events-dir stage_two/run_a_84_events \
        --out-csv docs/data/stage2_epoch_curve_84.csv

The run this reads is Run A of #84: 8 epochs of the released Stage 2 recipe, world size 16,
constant lr 1e-5, seed 42. See docs/stage2_epoch_curve_84.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
from pathlib import Path

# The paper run's per-epoch auto-label validation loss, epochs 1-8.
#
# Provenance: recovered in #104 from the surviving TensorBoard events of the June-2025
# paper run (the epoch-N weights themselves were deleted in a 2025-07-11 cleanup; only the
# scalars survived). Recorded here to 3 significant figures because that is the precision
# at which they were transcribed into #104 -- the raw events are not in this repo, which is
# a stated gap, not an oversight. See docs/stage2_epoch_curve_84.md.
PAPER_RUN_VAL_LOSS = {
    1: 0.000520,
    2: 0.000478,
    3: 0.000463,
    4: 0.000466,
    5: 0.000458,
    6: 0.000468,
    7: 0.000470,
    8: 0.000473,
}

# train.py logs the val scalar under this tag, once per epoch, keyed by global step.
VAL_TAG = "Loss/val_epoch"

# 150,063 train panoramas with drop_last=True at world size 16. Used only to turn a global
# step back into an epoch number, so the curve can be read without the checkpoint filenames.
STEPS_PER_EPOCH = 9378


# --------------------------------------------------------------------------------------
# Minimal protobuf wire-format reader. Only what Event/Summary need, nothing else.
# --------------------------------------------------------------------------------------


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Read a base-128 varint. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _iter_fields(buf: bytes):
    """Yield (field_number, wire_type, payload) for each field in a protobuf message.

    payload is an int for varint/fixed fields and a bytes slice for length-delimited ones.
    """
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field_no, wire_type = key >> 3, key & 0x07
        if wire_type == 0:  # varint
            value, pos = _read_varint(buf, pos)
        elif wire_type == 1:  # 64-bit
            value = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire_type == 5:  # 32-bit
            value = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        yield field_no, wire_type, value


def _iter_tfrecords(path: Path):
    """Yield the payload of each TFRecord in an event file.

    Framing is: uint64 length, uint32 masked-crc32c of the length bytes, the payload, then
    uint32 masked-crc32c of the payload. The CRCs are skipped -- crc32c is not in the
    stdlib, and file integrity is covered by the sha256 digests this script prints instead.
    A job killed mid-write can leave a torn final record; that is tolerated and reported
    rather than raised, because a preempted run is the normal case on ckpt partitions.
    """
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        if pos + 12 > len(data):
            print(f"  note: {path.name} ends with a torn record header, ignored", file=sys.stderr)
            return
        (length,) = struct.unpack_from("<Q", data, pos)
        payload_start = pos + 12
        payload_end = payload_start + length
        if payload_end + 4 > len(data):
            print(f"  note: {path.name} ends with a torn record payload, ignored", file=sys.stderr)
            return
        yield data[payload_start:payload_end]
        pos = payload_end + 4


def read_scalars(events_dir: Path, tag: str) -> dict[int, float]:
    """Return {global_step: value} for one scalar tag across every event file in a dir.

    Files are read in sorted order, which is chronological because SummaryWriter names them
    `events.out.tfevents.<unix_seconds>.<host>.<pid>.<n>`. A resumed run re-emits steps it
    had already written, so later files deliberately win on collision -- that is the value
    the run finished with.
    """
    files = sorted(events_dir.glob("events.out.tfevents.*"))
    if not files:
        raise SystemExit(f"no event files in {events_dir}")

    scalars: dict[int, float] = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        found = 0
        for payload in _iter_tfrecords(path):
            step = 0
            summary = None
            for field_no, _wire, value in _iter_fields(payload):
                if field_no == 2:  # Event.step
                    step = value
                elif field_no == 5:  # Event.summary
                    summary = value
            if summary is None:
                continue
            for field_no, _wire, value in _iter_fields(summary):
                if field_no != 1:  # Summary.value
                    continue
                this_tag = None
                simple_value = None
                for sub_no, _sub_wire, sub_value in _iter_fields(value):
                    if sub_no == 1:  # Summary.Value.tag
                        this_tag = sub_value.decode("utf-8")
                    elif sub_no == 2:  # Summary.Value.simple_value (float32)
                        simple_value = struct.unpack("<f", struct.pack("<I", sub_value))[0]
                if this_tag == tag and simple_value is not None:
                    scalars[step] = simple_value
                    found += 1
        print(f"  {path.name}  sha256={digest[:16]}...  {found} '{tag}' point(s)")
    return scalars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("stage_two/run_a_84_events"),
        help="directory of TensorBoard event files (default: %(default)s)",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="write the curve here as CSV")
    parser.add_argument("--tag", default=VAL_TAG, help="scalar tag to extract (default: %(default)s)")
    args = parser.parse_args()

    print(f"Reading '{args.tag}' from {args.events_dir}")
    scalars = read_scalars(args.events_dir, args.tag)
    if not scalars:
        raise SystemExit(f"no '{args.tag}' scalars found -- wrong tag or wrong directory?")

    rows = []
    for step, value in sorted(scalars.items()):
        epoch = round(step / STEPS_PER_EPOCH)
        paper = PAPER_RUN_VAL_LOSS.get(epoch)
        delta_pct = (value / paper - 1.0) * 100.0 if paper else None
        rows.append({"epoch": epoch, "step": step, "run_a_val_loss": value, "paper_val_loss": paper, "delta_pct": delta_pct})

    best = min(rows, key=lambda r: r["run_a_val_loss"])
    paper_best = min(PAPER_RUN_VAL_LOSS, key=PAPER_RUN_VAL_LOSS.get)

    print()
    print("| epoch | auto-label val loss | paper run | delta | vs. Run A min |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        mark = "**" if row is best else ""
        paper_txt = f"{row['paper_val_loss']:.6f}".lstrip("0") if row["paper_val_loss"] else ""
        delta_txt = f"{row['delta_pct']:+.2f}%" if row["delta_pct"] is not None else ""
        gap = (row["run_a_val_loss"] / best["run_a_val_loss"] - 1.0) * 100.0
        gap_txt = "min" if row is best else f"+{gap:.1f}%"
        print(
            f"| {row['epoch']} | {mark}{row['run_a_val_loss']:.8f}{mark} | {paper_txt} | "
            f"{delta_txt} | {gap_txt} |"
        )

    print()
    print(f"Run A minimum:    epoch {best['epoch']}  ({best['run_a_val_loss']:.8f})")
    print(f"Paper run minimum: epoch {paper_best}  ({PAPER_RUN_VAL_LOSS[paper_best]:.6f})")
    print("MATCH" if best["epoch"] == paper_best else "MISMATCH -- the selection epoch did not replicate")

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
                fieldnames=["epoch", "step", "run_a_val_loss", "paper_val_loss", "delta_pct"],
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "epoch": row["epoch"],
                        "step": row["step"],
                        "run_a_val_loss": f"{row['run_a_val_loss']:.8f}",
                        "paper_val_loss": f"{row['paper_val_loss']:.6f}" if row["paper_val_loss"] else "",
                        "delta_pct": f"{row['delta_pct']:.2f}" if row["delta_pct"] is not None else "",
                    }
                )
        print(f"\nWrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
