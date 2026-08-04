#!/usr/bin/env python3
"""Retarget an Ultralytics checkpoint so it can resume on a DIFFERENT cluster.

Why this exists
---------------
``YOLO(last).train(resume=True)`` reuses **every** training argument saved inside the
checkpoint -- that is the whole point of resume, and it is what keeps the LR schedule,
epoch counter and augmentation config continuous across a preemption. It is also what
makes a checkpoint non-portable: the saved ``data`` / ``save_dir`` / ``project`` paths
are absolute paths on the cluster that wrote them.

Move such a checkpoint to another cluster and resume it unmodified and one of two
things happens:

1. The saved ``data`` path does not exist there -> the run dies at dataset load; or
2. Worse, the saved ``save_dir`` DOES exist (e.g. you copied the tree) -> ultralytics
   writes results into the ORIGINAL run directory, corrupting the source run. This is
   the trap recorded in the #51 weight-snapshot MANIFEST, and it is silent.

This script rewrites exactly those path arguments and nothing else, so a checkpoint
trained on klone can continue on Tillicum as one continuous training trajectory.

What it deliberately does NOT touch
-----------------------------------
Every hyperparameter: ``epochs``, ``patience``, ``lr0``/``lrf``, ``batch``, ``imgsz``,
``workers``, ``close_mosaic``, ``seed``, ``optimizer``. Those define the pre-registered
schedule (issue #71), and changing any of them mid-run would make the resumed arm a
different config rather than a continuation. In particular ``epochs=60`` is the
DENOMINATOR of the LR decay, not a label -- rewriting it does not shorten the
experiment, it changes what every remaining epoch does.

``save_period`` is likewise left alone. A checkpoint saved with ``save_period=-1``
keeps only ``last.pt``/``best.pt``, and resume honours the saved value, so per-epoch
weights cannot be recovered for an arm that did not start with it.

Usage
-----
    python retarget_yolo_checkpoint.py CKPT.pt \
        --data    /gpfs/scrubbed/$USER/yolo/pano/data.yaml \
        --project /gpfs/projects/makelab/$USER/yolo_runs \
        --name    y11x_pano_h200

Prints a before/after table and refuses to write unless ``--apply`` is given, so the
default invocation is a dry run. Writes in place; pass ``--out`` to write elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# The six keys that carry a cluster-absolute path. `resume` and `model` both point at
# the checkpoint itself; ultralytics rewrites `model` internally on load, but we set
# both so the file is self-consistent if inspected.
PATH_KEYS = ("data", "project", "name", "save_dir", "model", "resume")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="checkpoint to retarget (usually last.pt)")
    ap.add_argument("--data", required=True, help="data.yaml path on the TARGET cluster")
    ap.add_argument("--project", required=True, help="runs root on the TARGET cluster")
    ap.add_argument("--name", required=True, help="run name under --project")
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of in place")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without this the script only reports")
    args = ap.parse_args()

    import torch  # imported late so --help works without a torch install

    if not args.ckpt.is_file():
        print(f"ERROR: no such checkpoint: {args.ckpt}", file=sys.stderr)
        return 2

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        # Catching this here turns a confusing mid-epoch failure into an obvious one.
        print(f"ERROR: --data does not exist on this host: {data_yaml}", file=sys.stderr)
        return 2

    save_dir = str(Path(args.project) / args.name)
    weights = str(Path(save_dir) / "weights" / "last.pt")
    new = {
        "data": str(data_yaml),
        "project": str(args.project),
        "name": args.name,
        "save_dir": save_dir,
        "model": weights,
        "resume": weights,
    }

    print(f"checkpoint : {args.ckpt}")
    print(f"sha256 (in): {sha256(args.ckpt)}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args")
    if not isinstance(ta, dict):
        print(f"ERROR: train_args is {type(ta)}, expected dict", file=sys.stderr)
        return 2

    # Epoch is 0-indexed inside the checkpoint; resume starts at epoch+2 in 1-indexed
    # `results.csv` terms. Printed so a stale checkpoint is obvious before submitting.
    done = int(ckpt.get("epoch", -1)) + 1
    print(f"epochs done: {done} of {ta.get('epochs')}   "
          f"best_fitness: {float(ckpt.get('best_fitness') or -1):.5f}")
    print()
    print(f"{'key':<10} {'from':<58} -> to")
    for k in PATH_KEYS:
        print(f"{k:<10} {str(ta.get(k)):<58} -> {new[k]}")
    print()

    unchanged = [k for k in PATH_KEYS if ta.get(k) == new[k]]
    if len(unchanged) == len(PATH_KEYS):
        print("Already retargeted; nothing to do.")
        return 0

    if not args.apply:
        print("DRY RUN -- rerun with --apply to write.")
        return 0

    ta.update(new)
    out = args.out or args.ckpt
    if out == args.ckpt:
        backup = args.ckpt.with_suffix(args.ckpt.suffix + ".preretarget")
        if not backup.exists():
            shutil.copy2(args.ckpt, backup)
            print(f"backup     : {backup}")
    torch.save(ckpt, out)
    print(f"wrote      : {out}")
    print(f"sha256(out): {sha256(out)}")

    # Reload and assert, so a silent torch.save/pickle problem cannot pass as success.
    check = torch.load(out, map_location="cpu", weights_only=False)["train_args"]
    bad = [k for k in PATH_KEYS if check.get(k) != new[k]]
    if bad:
        print(f"ERROR: keys did not persist: {bad}", file=sys.stderr)
        return 1
    print("verified   : all six path keys persisted on reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
