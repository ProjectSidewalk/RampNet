"""What did RampNet 1.0's Stage 2 training actually cost? 3.5 h/epoch on 16 GPUs — not 36.

`docs/curb_ramp_data_sourcing.md` §7 estimated "≳36 h on 16 L40s for one epoch
(≳580 GPU-h)" by extrapolating the README's ">24 hours". That extrapolation is
~10x too high: the ">24 hours" describes the *whole* multi-epoch, preemption-
riddled run, not one epoch.

This script measures it instead, from the paper run's own TensorBoard event
files (committed at docs/data/rampnet1_stage2_run/, rescued 2026-08-03 from
/gscratch/makelab/jsomeara/RampNet/stage_two/runs/experiment_1). It parses the
TFRecord + protobuf framing directly, so it needs no TensorFlow, no
TensorBoard, and no network:

  - 1.341 s/step median (rank 0), 16 GPUs x batch 1 = 16 panos/step;
  - one epoch = 9,378 steps exactly (150,063 train panos / 16), confirmed by
    validation scalars landing on exact multiples of 9,378;
  - so one epoch = 3.49 h wall-clock, ~56 GPU-h;
  - the run itself did ~12 epochs (max step 112,434 = 11.99 x 9,378) over
    44.7 h of active compute and 74.6 h of calendar time across 15 preemptions
    (x1.67 overhead on the `ckpt-all` scavenger partition);
  - auto-label val loss bottoms at epoch 5 (0.000458) vs epoch 1 (0.000520),
    then rises through epoch 11 — half of #84's epoch curve, already run.

The released model is nonetheless epoch 1: best_model.pth is byte-identical
(`cmp`) to checkpoints/epoch_1_step_9378.pth, copied back by hand on
2025-06-21. So the documented "1 epoch" recipe describes the released artifact
correctly; it just omits that 11 further epochs were run and discarded.

Because the step time is I/O-bound (~3% MFU: 8.4 MP JPEG decode + resize on
~3 cores per rank), it scales with *panoramas*, not with label count -- which
is what makes the --records projection below meaningful.

    python scripts/analysis/stage2_train_cost.py
    python scripts/analysis/stage2_train_cost.py --records 500000
    python scripts/analysis/stage2_train_cost.py --verify   # sha256 manifest
"""
import argparse
import glob
import hashlib
import os
import statistics
import struct

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EVENTS = os.path.join(REPO, "docs", "data", "rampnet1_stage2_run")

# RampNet 1.0 ground truth, from README.md's dataset table and docs/curb_ramp_data_sourcing.md §7.
RAMPNET1_RECORDS = 278_544      # government ramp records (NYC + Portland + Bend)
RAMPNET1_PANOS = 214_376        # panoramas generated from them
RAMPNET1_LABELS = 849_895       # point labels on those panoramas
RAMPNET1_TRAIN_PANOS = 150_063  # train split
WORLD_SIZE = 16                 # 4 nodes x 4 GPUs, run_train.slurm
MAX_PLAUSIBLE_STEP_GAP_S = 600  # drop gaps that span a preemption, not a step


def _varint(buf, j):
    val = shift = 0
    while j < len(buf):
        byte = buf[j]
        j += 1
        val |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    return val, j


def _fields(buf):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    j = 0
    while j < len(buf):
        key, j = _varint(buf, j)
        fno, wire = key >> 3, key & 7
        if wire == 0:
            val, j = _varint(buf, j)
            yield fno, wire, val
        elif wire == 1:
            yield fno, wire, buf[j:j + 8]
            j += 8
        elif wire == 2:
            ln, j = _varint(buf, j)
            yield fno, wire, buf[j:j + ln]
            j += ln
        elif wire == 5:
            yield fno, wire, buf[j:j + 4]
            j += 4
        else:
            return


def _records(path):
    """Yield each Event payload from a TFRecord-framed tfevents file."""
    with open(path, "rb") as handle:
        buf = handle.read()
    i, n = 0, len(buf)
    while i + 12 <= n:
        (length,) = struct.unpack_from("<Q", buf, i)
        i += 12                       # uint64 length + masked crc32
        if i + length + 4 > n:
            break                     # truncated tail: the job was killed mid-write
        yield buf[i:i + length]
        i += length + 4               # payload + masked crc32


def read_scalars(path):
    """Return [(wall_time, step, tag, value)] for every scalar summary in a file."""
    out = []
    for payload in _records(path):
        wall = summary = None
        step = 0
        for fno, wire, val in _fields(payload):
            if fno == 1 and wire == 1:
                (wall,) = struct.unpack("<d", val)
            elif fno == 2 and wire == 0:
                step = val
            elif fno == 5 and wire == 2:
                summary = val
        if summary is None or wall is None:
            continue                  # file_version header, or a graph/blob event
        for fno, wire, value_msg in _fields(summary):
            if fno != 1 or wire != 2:
                continue              # Summary.value is field 1, repeated
            tag = scalar = None
            for f2, w2, v2 in _fields(value_msg):
                if f2 == 1 and w2 == 2:
                    tag = v2.decode("utf8", "replace")
                elif f2 == 2 and w2 == 5:
                    (scalar,) = struct.unpack("<f", v2)
            if tag is not None and scalar is not None:
                out.append((wall, step, tag, scalar))
    return out


def hms(hours):
    return f"{hours:.2f} h" if hours < 48 else f"{hours:.1f} h ({hours / 24:.1f} d)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events-dir", default=DEFAULT_EVENTS,
                    help="directory of tfevents files (default: the committed paper run)")
    ap.add_argument("--world-size", type=int, default=WORLD_SIZE,
                    help="GPUs the run used; batch is 1 pano per GPU (default: 16)")
    ap.add_argument("--train-panos", type=int, default=RAMPNET1_TRAIN_PANOS,
                    help="panoramas in the train split of the measured run")
    ap.add_argument("--records", type=int, default=None,
                    help="project cost for a corpus of this many government ramp records")
    ap.add_argument("--epochs", type=int, nargs="*", default=[1, 5, 12],
                    help="epoch counts to project (default: 1 5 12)")
    ap.add_argument("--verify", action="store_true",
                    help="print a sha256 manifest of the event files and exit")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.events_dir, "events.out.tfevents.*")))
    if not paths:
        raise SystemExit(f"no tfevents files under {args.events_dir}")

    if args.verify:
        for path in paths:
            with open(path, "rb") as handle:
                print(f"{hashlib.sha256(handle.read()).hexdigest()}  {os.path.basename(path)}")
        return

    segments, step_gaps, val_points = [], [], []
    for path in paths:
        scalars = read_scalars(path)
        train = sorted((w, s) for w, s, t, _ in scalars if t == "Loss/train_step")
        val_points += [(s, v) for _, s, t, v in scalars if t == "Loss/val_epoch"]
        if len(train) < 2:
            continue                  # a restart that exited before its first step
        walls = [w for w, _ in train]
        steps = [s for _, s in train]
        step_gaps += [b - a for a, b in zip(walls, walls[1:])
                      if 0 < b - a < MAX_PLAUSIBLE_STEP_GAP_S]
        segments.append((walls[0], walls[-1], steps[0], steps[-1], len(train)))

    segments.sort()
    val_points.sort()
    s_per_step = statistics.median(step_gaps)
    active_h = sum(end - start for start, end, _, _, _ in segments) / 3600
    calendar_h = (max(s[1] for s in segments) - min(s[0] for s in segments)) / 3600
    max_step = max(s[3] for s in segments)
    steps_per_epoch = args.train_panos // args.world_size

    print(f"== measured run ({len(paths)} event files, {len(segments)} training segments) ==")
    print(f"  median s/step (rank 0) : {s_per_step:.3f}  "
          f"[p25 {statistics.quantiles(step_gaps, n=4)[0]:.3f} / "
          f"p75 {statistics.quantiles(step_gaps, n=4)[2]:.3f}, n={len(step_gaps)}]")
    print(f"  panos/s (x{args.world_size:<2d} GPUs)     : {args.world_size / s_per_step:.2f}")
    print(f"  max global_step        : {max_step:,} = {max_step / steps_per_epoch:.2f} epochs")
    print(f"  active compute         : {hms(active_h)}")
    print(f"  calendar span          : {hms(calendar_h)}  "
          f"(preemption overhead x{calendar_h / active_h:.2f})")

    if val_points:
        print(f"\n== epoch boundaries ({len(val_points)} validations) ==")
        best = min(val_points, key=lambda p: p[1])
        for step, loss in val_points:
            mark = "  <- best" if (step, loss) == best else ""
            print(f"  epoch {step / steps_per_epoch:5.2f}  step {step:>7,}  "
                  f"val {loss:.6f}{mark}")

    print(f"\n== per-epoch cost, {args.train_panos:,} train panos / {args.world_size} GPUs ==")
    epoch_h = steps_per_epoch * s_per_step / 3600
    print(f"  {steps_per_epoch:,} steps x {s_per_step:.3f} s = {hms(epoch_h)}  "
          f"({epoch_h * args.world_size:.0f} GPU-h)")

    if args.records:
        ppr = RAMPNET1_PANOS / RAMPNET1_RECORDS
        train_share = args.train_panos / RAMPNET1_PANOS
        panos = args.records * ppr
        train_panos = panos * train_share
        steps = train_panos / args.world_size
        per_epoch_h = steps * s_per_step / 3600
        print(f"\n== projection: {args.records:,} records "
              f"({args.records / RAMPNET1_RECORDS:.2f}x RampNet 1.0) ==")
        print(f"  panoramas    : {panos:>12,.0f}   (x{ppr:.3f} per record, 1.0's ratio)")
        print(f"  train split  : {train_panos:>12,.0f}   ({train_share:.1%}, 1.0's share)")
        print(f"  steps/epoch  : {steps:>12,.0f}")
        print(f"  labels       : {args.records * RAMPNET1_LABELS / RAMPNET1_RECORDS:>12,.0f}   "
              f"(1.0 has {RAMPNET1_LABELS:,} -- 'records' and 'labels' differ ~3x)")
        print(f"  {'epochs':>8} {'compute':>12} {'GPU-h':>8} {'calendar on ckpt':>18}")
        for n in args.epochs:
            total = per_epoch_h * n
            print(f"  {n:>8} {hms(total):>12} {total * args.world_size:>8.0f} "
                  f"{hms(total * calendar_h / active_h):>18}")


if __name__ == "__main__":
    main()
