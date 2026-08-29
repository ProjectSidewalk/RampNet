#!/usr/bin/env python3
"""Rebuild YOLO label .txt files from an Ultralytics label cache (#51).

WHY THIS EXISTS
---------------
On 2026-08-19 the `y11x_tiles` arm failed at dataset init with

    ValueError: train: No labels found in .../yolo/tiles/labels/train.cache

Every label directory under `/gscratch/scrubbed/jfroehli/yolo/` was empty -- all five
dataset variants, train and val -- while every image directory was fully intact
(557,413 train tiles, 161,002 val). `/gscratch/scrubbed` purges by access time, and
the `.cache` files are exactly what stopped anything from reading the individual
label `.txt` files after 2026-07-25. Their atime froze while the images kept being
read every epoch, so the purge took the labels and left the images.

**The cache that made training fast is what got the labels deleted.** That is the
generalisable part, and it is the same shape as the partially-purged conda package
cache that broke the env build in #51 -- a cache is not a backup, and a *populated*
cache actively hides the thing it caches from anything that decides what is cold.

WHY THE CACHE IS A SUFFICIENT SOURCE
------------------------------------
Ultralytics' cache is not a digest: it stores the fully parsed labels. Each record
carries `cls` (n,1) and `bboxes` (n,4) as normalized xywh -- which is the on-disk
`.txt` format itself. So the labels are recoverable exactly, with no re-derivation
from the source panoramas and no GPU.

That matters for provenance as much as for cost: rebuilding from the cache reproduces
the labels the published `y11x_tiles` / `y26_pano` arms actually trained on, whereas
re-running `prepare_yolo_dataset.py` would produce labels from today's code and today's
geometry constants. Those should agree, but "should" is not "do", and the arms in
flight were trained on these.

The write format below is lifted from `prepare_yolo_dataset.py::_write_pair` and the
line construction beside it, so a rebuilt file is byte-identical to the original for
any value that survives the float32 round trip through the cache:

    line = f"0 {u:.6f} {v:.6f} {w:.6f} {h:.6f}"
    body = "\n".join(lines) + ("\n" if lines else "")

Background tiles therefore get a **zero-byte file, not a missing one**. That is
deliberate: Ultralytics counts a missing label as `nm` (missing) and an empty one as
`nf` (found, no objects), and the failure above is raised when `nf == 0`.

USAGE
-----
    python rebuild_yolo_labels_from_cache.py \
        --cache /gscratch/makelab/jonf/rampnet_yolo_baseline_51/label_cache_rescue/train.cache \
        --labels-dir /gscratch/scrubbed/jfroehli/yolo/tiles/labels/train \
        --verify

Prefer the durable copy of the cache under `/gscratch/makelab` (purchased, never
purged) over the one on `scrubbed`, which is one purge window from being the same
problem again.

`--verify` re-reads every file it wrote and compares the parsed boxes back against the
cache, which is the only check that actually proves the round trip rather than assuming
it. It roughly doubles the runtime; run it at least once per rebuilt split.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_cache(path: Path):
    """Return the list of label records from an Ultralytics *.cache file."""
    import numpy as np
    obj = np.load(str(path), allow_pickle=True).item()
    labels = obj.get("labels")
    if labels is None:
        raise SystemExit(f"{path}: no 'labels' key -- not an Ultralytics label cache")
    return labels


def format_record(rec) -> str:
    """Render one cache record as the .txt body prepare_yolo_dataset.py would have written."""
    cls, boxes = rec["cls"], rec["bboxes"]
    lines = [
        f"{int(cls[i][0])} {boxes[i][0]:.6f} {boxes[i][1]:.6f} {boxes[i][2]:.6f} {boxes[i][3]:.6f}"
        for i in range(len(boxes))
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True, help="Ultralytics *.cache to read")
    ap.add_argument("--labels-dir", type=Path, required=True, help="directory to write *.txt into")
    ap.add_argument("--verify", action="store_true",
                    help="re-read every written file and compare boxes back to the cache")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    ap.add_argument("--progress-every", type=int, default=50000, help="progress line cadence")
    args = ap.parse_args()

    labels = load_cache(args.cache)
    print(f"cache      : {args.cache}")
    print(f"records    : {len(labels)}")
    total_boxes = sum(len(r["bboxes"]) for r in labels)
    print(f"boxes      : {total_boxes}")
    print(f"labels-dir : {args.labels_dir}")
    if args.dry_run:
        print("DRY RUN -- nothing written")
        return 0

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    written = empty = 0
    for i, rec in enumerate(labels, 1):
        stem = Path(rec["im_file"]).stem
        body = format_record(rec)
        (args.labels_dir / f"{stem}.txt").write_text(body)
        written += 1
        if not body:
            empty += 1
        if args.progress_every and i % args.progress_every == 0:
            print(f"  ... {i}/{len(labels)}", flush=True)

    print(f"written    : {written}  ({empty} background/empty, {written - empty} with boxes)")

    if not args.verify:
        print("OK (unverified -- pass --verify to prove the round trip)")
        return 0

    print("verifying ...", flush=True)
    bad = 0
    seen_boxes = 0
    for i, rec in enumerate(labels, 1):
        stem = Path(rec["im_file"]).stem
        text = (args.labels_dir / f"{stem}.txt").read_text()
        got = [ln.split() for ln in text.splitlines() if ln.strip()]
        exp = rec["bboxes"]
        if len(got) != len(exp):
            print(f"  MISMATCH count {stem}: file {len(got)} vs cache {len(exp)}")
            bad += 1
            continue
        for j, parts in enumerate(got):
            vals = [float(p) for p in parts[1:5]]
            for k in range(4):
                # .6f is the on-disk precision, so agreement is bounded by rounding,
                # not by anything about the data.
                if abs(vals[k] - float(exp[j][k])) > 1e-6:
                    print(f"  MISMATCH value {stem} box {j} coord {k}: "
                          f"{vals[k]} vs {float(exp[j][k])}")
                    bad += 1
                    break
        seen_boxes += len(got)
        if args.progress_every and i % args.progress_every == 0:
            print(f"  ... verified {i}/{len(labels)}", flush=True)

    print(f"verified   : {len(labels)} files, {seen_boxes} boxes, {bad} mismatched")
    if bad or seen_boxes != total_boxes:
        print("FAIL")
        return 1
    print("PASS -- every file round-trips to the cache it came from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
