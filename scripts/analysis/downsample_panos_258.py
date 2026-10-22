"""Downsample one split's native panoramas to 4096x2048 the way manual_gold's were made (#258).

The #258 step 2 read-out leaves one confound open: manual_gold's panos are 4096x2048, so
the validator's crops there are upsampled 2x to the 8192x4096 serving resize, while the
verdict splits are read at native resolution. This produces the control: a verdict split
(richmond) brought down to manual_gold's resolution by manual_gold's own transform, so
the same scorer can be run on both copies and the difference read directly.

**The transform.** manual_gold's images are the ``projectsidewalk/rampnet-dataset`` test
split, written by ``stage_one/dataset_generation/download_dataset.py``: the GSV fetch in
``rampnet.gsv.fetch_panorama`` ends with ``cv2.resize(..., (4096, 2048),
interpolation=cv2.INTER_LINEAR)`` and the pano is saved with a plain ``cv2.imwrite``
(OpenCV's default JPEG quality, 95). This script applies exactly that resize and that
write to the native JPEG. What it cannot reproduce is Google's server-side zoom-3 tile
render that manual_gold's fetch started from; that difference is stated in the doc.

The HF ``rampnet-benchmark`` ``4096x2048`` config is a different transform (PIL
BILINEAR, JPEG q82, the gt_gallery review copy), so it is not used here.

    python scripts/analysis/downsample_panos_258.py --src <native>/richmond/panos \
        --dst <stage>/richmond/panos

Writes ``<dst>/../downsample_manifest.json`` with a sha256 per source and output file.
"""
import argparse
import hashlib
import json
import os
import sys

WIDTH, HEIGHT = 4096, 2048


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.dst, exist_ok=True)
    files = {}
    for name in sorted(os.listdir(args.src)):
        if not name.endswith(".jpg"):
            continue
        src = os.path.join(args.src, name)
        img = cv2.imread(src)
        if img is None:
            raise FileNotFoundError(src)
        out = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        dst = os.path.join(args.dst, name)
        if not cv2.imwrite(dst, out):
            raise OSError(dst)
        files[name] = {"native_wh": [img.shape[1], img.shape[0]],
                       "src_sha256": sha256_file(src), "out_sha256": sha256_file(dst)}
    manifest = {"script": "scripts/analysis/downsample_panos_258.py",
                "transform": "cv2.resize INTER_LINEAR to 4096x2048, cv2.imwrite default "
                             "(JPEG quality 95), as rampnet.gsv.fetch_panorama + "
                             "download_dataset.py wrote manual_gold's panos",
                "opencv": cv2.__version__, "n": len(files), "files": files}
    path = os.path.join(os.path.dirname(os.path.abspath(args.dst)), "downsample_manifest.json")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(files)} panos -> {args.dst}; manifest {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
