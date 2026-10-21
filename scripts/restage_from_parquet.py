#!/usr/bin/env python
"""Re-stage the Stage 2 dataset from the cached Hugging Face parquet shards, without
`datasets`' Arrow conversion cache.

Why this exists (issue #263): `download_dataset.py` calls `load_dataset(...)`, which first
converts every parquet shard into an Arrow cache (another ~460 GB) before `.map()` writes the
jpg/json pairs. On klone that conversion died with `OSError: [Errno 122] Disk quota exceeded`
(job 42083469, 2026-10-20) after the 431 GB download had completed. This script reads the
cached shards directly with pyarrow and writes the pairs, so the only space it needs is the
output itself.

Differences from `download_dataset.py`, all deliberate:
  * the jpg is the parquet's own bytes (the file that was uploaded to the Hub), not a
    PIL re-encode at quality 95, so the pixels are the published ones;
  * the HF split named `validation` on the Hub is `val` in the snapshot used here; both map
    to `<out>/val`, as `download_dataset.py` does;
  * rows whose image bytes are not a JPEG are quarantined, not written. The Hub dataset
    carries three placeholder rows per split (`pano_id` ``file_1``, ``file_2``, ``file_3``,
    a 19-byte blob beginning ``dumm``). `download_dataset.py` skips them only because its
    PIL re-encode raises; the published recipe's counts (150,063 / 42,875 / 21,438) exclude
    them. They land in `<out>/_quarantine/<split>/` with a manifest line each.
The json carries every non-image column, as `download_dataset.py` does.

Resumable: a pano is complete when its .json exists (the json is written after the jpg,
both via atomic rename), so a requeued job skips finished panos.

History: the 2026-10-20 run on klone (job 42090420) used the version of this script without
the quarantine step; its nine placeholder files were moved to `_quarantine/` by hand and
`--check` was re-run. The committed version reproduces that end state in one pass.

Usage:
  python restage_from_parquet.py --snapshot <hf snapshot dir> --out <dataset dir> --workers 12
  python restage_from_parquet.py --snapshot ... --out ... --check   # counts only
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

SPLIT_DIRS = {"train": "train", "val": "val", "validation": "val", "test": "test"}
EXPECTED = {"train": 150063, "val": 42875, "test": 21438}
JPEG_MAGIC = b"\xff\xd8\xff"


def is_jpeg(data) -> bool:
    """True when ``data`` is bytes that start with the JPEG SOI marker."""
    return isinstance(data, (bytes, bytearray)) and len(data) > 3 and bytes(data[:3]) == JPEG_MAGIC


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def process_shard(args):
    """Write every valid row of one shard; quarantine the rest. Returns counts."""
    import pyarrow.parquet as pq  # imported here so --check and the tests need no pyarrow

    shard, out_dir, quarantine_dir = args
    out_dir = Path(out_dir)
    quarantine_dir = Path(quarantine_dir)
    pf = pq.ParquetFile(shard)
    written = skipped = quarantined = 0
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            pano_id = row["pano_id"]
            img = row.pop("image")
            data = img["bytes"] if isinstance(img, dict) else None
            if not is_jpeg(data):
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                write_atomic(quarantine_dir / f"{pano_id}.jpg", data or b"")
                write_atomic(quarantine_dir / f"{pano_id}.json", json.dumps(row).encode("utf-8"))
                quarantined += 1
                continue
            jpg = out_dir / f"{pano_id}.jpg"
            js = out_dir / f"{pano_id}.json"
            if js.exists() and jpg.exists():
                skipped += 1
                continue
            write_atomic(jpg, data)
            write_atomic(js, json.dumps(row).encode("utf-8"))
            written += 1
    return os.path.basename(shard), written, skipped, quarantined


def count_pairs(d: Path):
    """(complete pairs, jpg without json, json without jpg) in one split dir."""
    jpgs = {p.stem for p in d.glob("*.jpg")}
    jsons = {p.stem for p in d.glob("*.json")}
    return len(jpgs & jsons), len(jpgs - jsons), len(jsons - jpgs)


def write_quarantine_manifest(root: Path) -> int:
    """List every quarantined file (sorted, LF) in ``_quarantine/MANIFEST.txt``; returns the count."""
    q = root / "_quarantine"
    if not q.is_dir():
        return 0
    files = sorted(p.relative_to(q).as_posix() for p in q.rglob("*") if p.is_file() and p.name != "MANIFEST.txt")
    with open(q / "MANIFEST.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(files) + ("\n" if files else ""))
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check", action="store_true", help="count pairs only")
    a = ap.parse_args()

    rc = 0
    if a.check:
        # Counts only: enumerate the output splits, so the check runs after the shard cache
        # has been removed (the 431 GB cache is deleted once the copy verifies).
        work = [(None, s) for s in ("train", "val", "test") if (a.out / s).is_dir()]
        if not work:
            print(f"no split directories under {a.out}", flush=True)
            return 1
    else:
        work = [(a.snapshot / hf, out) for hf, out in SPLIT_DIRS.items() if (a.snapshot / hf).is_dir()]
        if not work:
            print(f"no split directories under {a.snapshot}", flush=True)
            return 1
    for src, out_split in work:
        out_dir = a.out / out_split
        quarantine_dir = a.out / "_quarantine" / out_split
        out_dir.mkdir(parents=True, exist_ok=True)
        if not a.check:
            shards = sorted(str(p) for p in src.glob("*.parquet"))
            print(f"[{out_split}] {len(shards)} shards -> {out_dir}", flush=True)
            jobs = [(sh, str(out_dir), str(quarantine_dir)) for sh in shards]
            with Pool(a.workers) as pool:
                for name, w, s, q in pool.imap_unordered(process_shard, jobs):
                    print(f"  {name}: wrote {w}, skipped {s}, quarantined {q}", flush=True)
        pairs, jpg_only, json_only = count_pairs(out_dir)
        exp = EXPECTED.get(out_split)
        status = "OK" if exp == pairs and not jpg_only and not json_only else "MISMATCH"
        if status != "OK":
            rc = 1
        print(f"[{out_split}] pairs {pairs} (expected {exp}), jpg-only {jpg_only}, json-only {json_only}: {status}", flush=True)
    n_q = write_quarantine_manifest(a.out)
    print(f"[quarantine] {n_q} files listed in {a.out / '_quarantine' / 'MANIFEST.txt'}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
