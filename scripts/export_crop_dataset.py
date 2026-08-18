"""Package the round-1 Project Sidewalk crop set as a HuggingFace dataset, in Parquet.

This is the training data behind **round 1** of the Stage 1 crop model. The round-2 manual crops
are already published at `projectsidewalk/rampnet-crop-model-dataset-round2` (named
`rampnet-crop-model-dataset` until the 2026-08-05 rename; the old id redirects); this is its larger,
Project-Sidewalk-sourced counterpart -- 27,704 crops, 13.4 GB -- and it is **not reproducible**:
`stage_one/crop_model/ps_model/data/download_data.py` reads live from Project Sidewalk servers
whose databases keep growing, so a re-run builds a different set.

Parquet rather than loose JPEGs because that is what HF asks for on large datasets, it is what
makes the viewer work, and 27,704 loose files would consume a quarter of the <100k-files-per-repo
guidance for no benefit. (The 10k limit is per *folder*, so loose files inside train/val/test would
have been legal -- this is a recommendation, not a wall.)

**Labels are in the filenames**, and this export makes them a real column. A crop named
`007mz25c_-_118_596_-_478_611.jpg` carries keypoints (118, 596) and (478, 611).

The leading token is **not a panorama id** -- `download_data.py:277` builds it with
`random.choices(alphabet, k=8)`, so it is an opaque per-crop token and the source panorama is not
recoverable from this artifact. The column is named `crop_uid` accordingly. (Measured: 27,704
distinct tokens for 27,704 crops, so no collision happened to occur.)

Coordinates are stored **verbatim**, in the pixel space of the stored crop (683x2048), and are not
normalised here on purpose. The loader scales BOTH axes by 0.5, but `transforms.Resize((1024, 352))`
scales the image by 0.5 on y and 352/683 = 0.5154 on x -- so y is consistent and x is under-scaled
by 3.1%, drifting a label up to ~10.5 px left at the right edge. That is in the original code;
normalising here would silently bake in one reading of it. See
`stage_one/crop_model/ps_model/model/train.py`.

Splits are the committed train/val/test directories -- a 70/15/15 split made by `splititup.sh`,
which shuffles with `shuf` and **no seed**, so the partition is not reproducible either. That is
another reason to publish it rather than describe how to rebuild it.

Usage
-----
    python scripts/export_crop_dataset.py build  --src <dataset_1/> --out dist/round1
    python scripts/export_crop_dataset.py verify --out dist/round1
    python scripts/export_crop_dataset.py card   --out dist/round1   # card only, no rebuild
    python scripts/export_crop_dataset.py push   --out dist/round1 \
        --repo-id projectsidewalk/rampnet-crop-model-dataset-round1
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_export_common import (  # noqa: E402
    clear_build_dir, git_commit, hf_features_metadata, hf_value, sha256_bytes)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "hf_package" / "README.crop_dataset_card.template.md"

SPLITS = ["train", "val", "test"]
SHARD_TARGET_BYTES = 1_500_000_000     # keep individual parquet files resumable
ROWS_PER_BATCH = 32

# "<pano>_-_<x>_<y>[_-_<x>_<y>...]" -- train.py splits on the same separator.
KEYPOINT_RE = re.compile(r"^(-?\d+)_(-?\d+)$")

SCHEMA = pa.schema([
    pa.field("crop_id", pa.string()),
    pa.field("crop_uid", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("keypoints", pa.list_(pa.struct([
        pa.field("x", pa.int32()), pa.field("y", pa.int32())]))),
    pa.field("n_keypoints", pa.int32()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("sha256", pa.string()),
])

FEATURES = {
    "crop_id": hf_value("string"), "crop_uid": hf_value("string"), "image": {"_type": "Image"},
    "keypoints": [{"x": hf_value("int32"), "y": hf_value("int32")}],
    "n_keypoints": hf_value("int32"), "width": hf_value("int32"), "height": hf_value("int32"),
    "sha256": hf_value("string"),
}


def schema_with_metadata():
    return SCHEMA.with_metadata(hf_features_metadata(FEATURES))


def parse_name(stem):
    """`007mz25c_-_118_596_-_478_611` -> ("007mz25c", [(118, 596), (478, 611)]).

    The first element is the opaque 8-char token, NOT a panorama id -- see the module docstring.
    """
    parts = stem.split("_-_")
    keypoints = []
    for part in parts[1:]:
        match = KEYPOINT_RE.match(part)
        if match:
            keypoints.append({"x": int(match.group(1)), "y": int(match.group(2))})
    return parts[0], keypoints


def write_split(src_dir, out_dir, split):
    """Stream one split into shards of roughly SHARD_TARGET_BYTES."""
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".jpg")
    if not files:
        return []
    schema = schema_with_metadata()
    shards, batch, writer, shard_path = [], [], None, None
    written_in_shard = 0
    unlabelled = 0

    def open_shard(index):
        path = out_dir / "{}-{:05d}.parquet".format(split, index)
        return path, pq.ParquetWriter(str(path), schema, compression="zstd")

    def flush():
        nonlocal batch
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch = []

    for path in files:
        data = path.read_bytes()
        with Image.open(path) as im:
            width, height = im.size
        crop_uid, keypoints = parse_name(path.stem)
        if not keypoints:
            unlabelled += 1
        if writer is None:
            shard_path, writer = open_shard(len(shards))
            written_in_shard = 0
        batch.append({
            "crop_id": path.stem, "crop_uid": crop_uid,
            "image": {"bytes": data, "path": path.name},
            "keypoints": keypoints, "n_keypoints": len(keypoints),
            "width": width, "height": height,
            "sha256": sha256_bytes(data),
        })
        written_in_shard += len(data)
        if len(batch) >= ROWS_PER_BATCH:
            flush()
        if written_in_shard >= SHARD_TARGET_BYTES:
            flush()
            writer.close()
            shards.append(shard_path)
            writer = None
    if writer is not None:
        flush()
        writer.close()
        shards.append(shard_path)

    if unlabelled:
        print("  note: {:,} crops in {} carry no keypoints in their filename".format(unlabelled, split))
    return shards


def keypoint_summary(out):
    """Describe the label distribution *from the built shards*, so the card cannot drift from it.

    Round 1 happens to contain no zero-keypoint crops, but that is a fact about this data rather
    than a guarantee of the schema -- so it is measured here, not asserted in the template.
    """
    histogram = {}
    for path in sorted(out.rglob("*.parquet")):
        for batch in pq.ParquetFile(str(path)).iter_batches(
                batch_size=4096, columns=["n_keypoints"]):
            for n in batch.column("n_keypoints").to_pylist():
                histogram[n] = histogram.get(n, 0) + 1
    if not histogram:
        return "", 0
    total = sum(n * c for n, c in histogram.items())
    lines = ["| keypoints in a crop | crops |", "| ---: | ---: |"]
    lines += ["| {} | {:,} |".format(n, histogram[n]) for n in sorted(histogram)]
    if 0 in histogram:
        note = ("{:,} crops carry no keypoint at all -- those are the negatives."
                .format(histogram[0]))
    else:
        note = ("**There are no negative crops in this round**: every crop carries at least one "
                "keypoint, so a model trained on this set alone never sees an empty example.")
    return "{}\n\n{}".format(note, "\n".join(lines)), total


def write_card(out, repo_id):
    """Render the dataset card from what is actually on disk. Split out of `build` so a card fix
    does not require rebuilding 13 GB of Parquet."""
    configs = ["- config_name: default", "  data_files:"]
    for split in SPLITS:
        if (out / "data" / split).is_dir():
            configs.append("  - split: {}".format(split))
            configs.append("    path: data/{}/{}-*.parquet".format(split, split))

    total = sum(f.stat().st_size for f in out.rglob("*.parquet"))
    n_rows = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in out.rglob("*.parquet"))
    summary, n_kps = keypoint_summary(out)
    (out / "README.md").write_text(TEMPLATE.read_text(encoding="utf-8").format(
        configs_yaml="\n".join(configs), git_commit=git_commit(),
        export_date=datetime.date.today().isoformat(), repo_id=repo_id,
        n_crops="{:,}".format(n_rows), total_gb="{:.2f}".format(total / 1e9),
        n_keypoints="{:,}".format(n_kps), keypoint_summary=summary,
    ), encoding="utf-8")
    return n_rows, total


def build(args):
    out = Path(args.out)
    index = {}
    print("{:<8} {:>8} {:>16} {:>16} {:>7}".format("split", "crops", "bytes in", "parquet", "shards"))
    print("-" * 60)
    for split in SPLITS:
        src_dir = Path(args.src) / split
        if not src_dir.is_dir():
            print("{:<8} (missing, skipped)".format(split))
            continue
        src_bytes = sum(p.stat().st_size for p in src_dir.iterdir() if p.suffix.lower() == ".jpg")
        # Shards are named by position, so a rebuild yielding fewer of them than last time leaves
        # orphans that write_card counts into the totals and upload_folder then publishes --
        # duplicating crops in the released split, with every orphan row still matching its own
        # recorded sha256, so `verify` cannot see it. Clear the split before rewriting it.
        target = clear_build_dir(out, "data/{}".format(split))
        target.mkdir(parents=True, exist_ok=True)
        shards = write_split(src_dir, target, split)
        n = sum(pq.ParquetFile(str(s)).metadata.num_rows for s in shards)
        size = sum(s.stat().st_size for s in shards)
        index[split] = len(shards)
        print("{:<8} {:>8,} {:>16,} {:>16,} {:>7}".format(split, n, src_bytes, size, len(shards)))

    if not index:
        sys.exit("error: no train/val/test directories under {}".format(args.src))

    n_rows, total = write_card(out, args.repo_id)
    print("-" * 60)
    print("{:,} crops, {:,} bytes ({:.2f} GB) -> {}".format(n_rows, total, total / 1e9, out))


def card(args):
    n_rows, total = write_card(Path(args.out), args.repo_id)
    print("card rewritten for {:,} crops ({:.2f} GB)".format(n_rows, total / 1e9))


def verify(args):
    out = Path(args.out)
    files = sorted(out.rglob("*.parquet"))
    if not files:
        sys.exit("error: nothing built under {}".format(out))
    bad = rows = kps = 0
    for path in files:
        n = 0
        for batch in pq.ParquetFile(str(path)).iter_batches(
                batch_size=ROWS_PER_BATCH, columns=["crop_id", "image", "sha256", "keypoints"]):
            for row in batch.to_pylist():
                rows += 1
                n += 1
                kps += len(row["keypoints"])
                if sha256_bytes(row["image"]["bytes"]) != row["sha256"]:
                    bad += 1
                    print("  MISMATCH {} in {}".format(row["crop_id"], path.name))
        print("  {:<34} {:>7,} rows OK".format(str(path.relative_to(out)), n))
    print("-" * 60)
    if bad:
        sys.exit("FAIL: {} of {:,} crops do not match their recorded sha256".format(bad, rows))
    print("PASS: {:,} crops round-trip byte-identical, {:,} keypoints".format(rows, kps))


def push(args):
    from huggingface_hub import HfApi                # imported late: not needed to build or verify
    out = Path(args.out)
    if not (out / "README.md").exists():
        sys.exit("error: build first")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out),
                      commit_message="Add the round-1 Project Sidewalk crop set as Parquet")
    print("Done: https://huggingface.co/datasets/{}".format(args.repo_id))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["build", "verify", "card", "push"])
    parser.add_argument("--src", type=Path, help="dataset_1/ directory holding train/val/test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-id", default="projectsidewalk/rampnet-crop-model-dataset-round1")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if args.mode == "build" and not args.src:
        sys.exit("error: build needs --src")
    {"build": build, "verify": verify, "card": card, "push": push}[args.mode](args)


if __name__ == "__main__":
    main()
