#!/usr/bin/env python3
"""Acceptance test for the YOLO label rebuild: does Ultralytics load the dataset again? (#51)

The rebuild in `rebuild_yolo_labels_from_cache.py` verifies its own output against the
cache it read, which proves the round trip but *not* the thing that actually broke. The
failure was in Ultralytics' dataset init:

    ValueError: train: No labels found in .../yolo/tiles/labels/train.cache

so the only test that closes the loop is making Ultralytics build the dataset and report
a non-zero `nf` (labels found). This does that and nothing else -- it builds the dataset,
checks what the scan produced, and exits. No model, no GPU, no training step.

Three things it checks that a bare "did it crash" run would not:

- **Missing label files against empty ones.** This is the one that needs care, because
  Ultralytics makes the two look identical downstream: `verify_image_label` counts a
  missing file as `nm` and an empty file as `ne`, but BOTH append a record with zero
  boxes to `dataset.labels`. So a rebuild that skipped the 59,923 label-less background
  tiles would produce the same image count, the same box count and the same number of
  zero-box records as a correct one -- and the dataset would still be wrong, because the
  original failure (`No labels found ... nf == 0`) is raised off `nf`, which only counts
  files that exist. `dataset.labels` therefore cannot answer this; the check stats the
  label paths directly and fails if any is absent.
- **Total boxes against the expected count.** Loading is not the same as loading
  everything. The cache the labels came from holds 968,227 train boxes; if the rebuilt
  dataset scans to a different number, the labels are wrong in a way that trains fine and
  scores wrong.
- **Background count against the expected count.** 59,923 of the 557,413 train tiles are
  background and must be present as zero-byte files.

Note this builds the dataset with `augment=False`, where Ultralytics only *warns* on
`nf == 0` instead of raising -- so the original ValueError is not reproduced verbatim.
The `--expect-boxes` and missing-file checks are what catch that state here.

Usage:

    python check_yolo_dataset_loads.py --data-root /gscratch/scrubbed/jfroehli/yolo/tiles \\
        --split train --expect-images 557413 --expect-boxes 968227 \\
        --expect-background 59923

Exit status is 0 only when every check passes, so it can be run unattended.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True,
                    help="dataset root holding images/ and labels/")
    ap.add_argument("--split", default="train", help="split to load (default: %(default)s)")
    ap.add_argument("--expect-images", type=int, default=None,
                    help="fail unless this many image/label pairs are found")
    ap.add_argument("--expect-boxes", type=int, default=None,
                    help="fail unless the scanned labels hold this many boxes")
    ap.add_argument("--expect-background", type=int, default=None,
                    help="fail unless this many scanned tiles have zero boxes")
    ap.add_argument("--imgsz", type=int, default=640, help="only affects the scan, not results")
    args = ap.parse_args()

    from ultralytics.data.dataset import YOLODataset

    img_dir = args.data_root / "images" / args.split
    if not img_dir.is_dir():
        print(f"FATAL: no image directory at {img_dir}")
        return 1

    print(f"data-root : {args.data_root}")
    print(f"split     : {args.split}")
    print("building the dataset (this is the call that raised the original ValueError) ...",
          flush=True)

    ds = YOLODataset(img_path=str(img_dir), imgsz=args.imgsz, augment=False,
                     data={"names": {0: "curb_ramp"}, "channels": 3})

    n_images = len(ds.labels)
    n_boxes = sum(len(rec["bboxes"]) for rec in ds.labels)
    n_empty = sum(1 for rec in ds.labels if len(rec["bboxes"]) == 0)

    # A record with zero boxes is either an empty label file (correct) or a MISSING one
    # (the rebuild skipped it), and ds.labels cannot tell them apart. Stat the paths.
    from ultralytics.data.utils import img2label_paths

    label_files = img2label_paths(ds.im_files)
    missing = [f for f in label_files if not Path(f).is_file()]

    print(f"images    : {n_images}")
    print(f"boxes     : {n_boxes}")
    print(f"background: {n_empty} (zero-box tiles, which must be PRESENT as empty files)")
    print(f"missing   : {len(missing)} label files absent from disk")

    ok = True
    if n_images == 0:
        print("FAIL: zero image/label pairs -- this is the original failure, unfixed")
        ok = False
    if missing:
        print(f"FAIL: {len(missing)} label files are MISSING, not empty -- Ultralytics "
              f"counts these as nm, not nf, and nf == 0 is what raised the original "
              f"ValueError. First few: {[str(m) for m in missing[:3]]}")
        ok = False
    if args.expect_images is not None and n_images != args.expect_images:
        print(f"FAIL: expected {args.expect_images} images, scanned {n_images}")
        ok = False
    if args.expect_boxes is not None and n_boxes != args.expect_boxes:
        print(f"FAIL: expected {args.expect_boxes} boxes, scanned {n_boxes}")
        ok = False
    if args.expect_background is not None and n_empty != args.expect_background:
        print(f"FAIL: expected {args.expect_background} background tiles, scanned {n_empty}")
        ok = False

    print("PASS -- Ultralytics builds the dataset and the counts match" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
