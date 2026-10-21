#!/usr/bin/env python
"""Re-stage the Stage 2 dataset from the cached Hugging Face parquet shards, without
`datasets`' Arrow conversion cache.

Why this exists (issue #263): `download_dataset.py` calls `load_dataset(...)`, which first
converts every parquet shard into an Arrow cache (about as large again as the shards) before
`.map()` writes the jpg/json pairs. On klone that conversion died with `OSError: [Errno 122]
Disk quota exceeded` (job 42083469, 2026-10-20) after the shard download had completed. This
script reads the cached shards directly with pyarrow and writes the pairs, so the only space it
needs is the output itself.

Differences from `download_dataset.py`, all deliberate:
  * the jpg is the parquet's own bytes (the file that was uploaded to the Hub), not a
    PIL re-encode at quality 95, so the pixels are the published ones;
  * the json is compact (`json.dumps` default) rather than `indent=4`; the keys and values
    are the same non-image columns;
  * the HF split named `validation` on the Hub is `val` in the snapshot used here; both map
    to `<out>/val`, as `download_dataset.py` does;
  * rows whose image bytes are not a JPEG are quarantined, not written. The Hub dataset
    carries three placeholder rows per split (`pano_id` ``file_1``, ``file_2``, ``file_3``,
    a 19-byte blob beginning ``dumm``). `download_dataset.py` never writes them because the
    image decode raises inside its per-row `try`, so the published recipe's counts
    (150,063 / 42,875 / 21,438) exclude them. They land in `<out>/_quarantine/<split>/` with
    a manifest line each, and any copy of them left in `<out>/<split>/` by an earlier run of
    this script is removed.

Resumable: a pano is complete when both its .jpg and .json exist (the json is written after
the jpg, both via atomic rename), so a requeued job skips finished panos. Leftover ``.tmp``
files from a preempted write are deleted at the start of each split in write mode and are a
failure in check mode.

Gate: the script exits 0 only when every split in ``EXPECTED`` exists and holds exactly the
expected number of complete pairs with no jpg-only or json-only strays and no ``.tmp`` files.
Write mode also records a per-split sha256 over the sorted pano ids in ``<out>/COUNTS.json``;
check mode recomputes it and fails on a difference, so a copy that merely has the right count
but a different id set is caught. Check mode is read-only and needs no shard cache.

History: the 2026-10-20 run on klone (job 42090420) used the version of this script without
the quarantine step, the ``.tmp`` sweep or ``COUNTS.json``; the nine placeholder files were
moved to ``_quarantine/`` by hand, and ``--check --record`` was run once on 2026-10-21 to write
the manifest and ``COUNTS.json`` from the staged copy (the shard cache had been removed by
then). Every later ``--check`` verifies against that record.

Usage:
  python restage_from_parquet.py --snapshot <hf snapshot dir> --out <dataset dir> --workers 12
  python restage_from_parquet.py --out <dataset dir> --check
  python restage_from_parquet.py --out <dataset dir> --check --record   # first verification only
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

SPLIT_DIRS = {"train": "train", "val": "val", "validation": "val", "test": "test"}
EXPECTED = {"train": 150063, "val": 42875, "test": 21438}
JPEG_MAGIC = b"\xff\xd8\xff"
COUNTS_FILE = "COUNTS.json"


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
    import pyarrow.parquet as pq  # imported here so check mode needs no pyarrow

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
            jpg = out_dir / f"{pano_id}.jpg"
            js = out_dir / f"{pano_id}.json"
            if not is_jpeg(data):
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                write_atomic(quarantine_dir / f"{pano_id}.jpg", data or b"")
                write_atomic(quarantine_dir / f"{pano_id}.json", json.dumps(row).encode("utf-8"))
                for stale in (jpg, js):  # left by a run of the pre-quarantine version
                    if stale.exists():
                        stale.unlink()
                quarantined += 1
                continue
            if js.exists() and jpg.exists():
                skipped += 1
                continue
            write_atomic(jpg, data)
            write_atomic(js, json.dumps(row).encode("utf-8"))
            written += 1
    return os.path.basename(shard), written, skipped, quarantined


def sweep_tmp(d: Path) -> int:
    """Delete ``*.tmp`` left by an interrupted write; returns how many."""
    n = 0
    for p in d.glob("*.tmp"):
        p.unlink()
        n += 1
    return n


def split_state(d: Path):
    """(complete pairs, jpg-only, json-only, tmp files, sha256 of sorted pano ids) for one split."""
    jpgs = {p.stem for p in d.glob("*.jpg")}
    jsons = {p.stem for p in d.glob("*.json")}
    pairs = jpgs & jsons
    digest = hashlib.sha256("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()
    return len(pairs), len(jpgs - jsons), len(jsons - jpgs), len(list(d.glob("*.tmp"))), digest


def count_pairs(d: Path):
    """(complete pairs, jpg without json, json without jpg) in one split dir."""
    pairs, jpg_only, json_only, _, _ = split_state(d)
    return pairs, jpg_only, json_only


def quarantine_listing(root: Path):
    """Sorted relative paths of every quarantined file (the manifest's content)."""
    q = root / "_quarantine"
    if not q.is_dir():
        return []
    return sorted(p.relative_to(q).as_posix() for p in q.rglob("*")
                  if p.is_file() and p.name != "MANIFEST.txt")


def write_quarantine_manifest(root: Path) -> int:
    """List every quarantined file (sorted, LF) in ``_quarantine/MANIFEST.txt``; returns the count."""
    files = quarantine_listing(root)
    if not (root / "_quarantine").is_dir():
        return 0
    with open(root / "_quarantine" / "MANIFEST.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(files) + ("\n" if files else ""))
    return len(files)


def check_quarantine_manifest(root: Path) -> bool:
    """True when the committed manifest equals the directory listing (or neither exists)."""
    q = root / "_quarantine"
    if not q.is_dir():
        return True
    m = q / "MANIFEST.txt"
    if not m.is_file():
        return False
    return m.read_text(encoding="utf-8").splitlines() == quarantine_listing(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, help="HF snapshot dir with train/ val/ test/ shards "
                                                 "(not needed with --check)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check", action="store_true", help="read-only: verify counts, id hashes, "
                                                        "tmp files and the quarantine manifest")
    ap.add_argument("--record", action="store_true",
                    help="with --check: write COUNTS.json and the quarantine manifest when they "
                         "are missing (the one-time step for a copy staged before this version)")
    a = ap.parse_args()
    if a.record and not a.check:
        ap.error("--record goes with --check")
    if not a.check and a.snapshot is None:
        ap.error("--snapshot is required unless --check")

    rc = 0
    previous = {}
    counts_path = a.out / COUNTS_FILE
    if a.check and counts_path.is_file():
        previous = json.loads(counts_path.read_text(encoding="utf-8"))

    if not a.check:
        hf_dirs = {out: a.snapshot / hf for hf, out in SPLIT_DIRS.items() if (a.snapshot / hf).is_dir()}
        missing = [s for s in EXPECTED if s not in hf_dirs]
        if missing:
            print(f"snapshot {a.snapshot} lacks split(s) {missing}", flush=True)
            return 1
        for out_split in EXPECTED:
            out_dir = a.out / out_split
            quarantine_dir = a.out / "_quarantine" / out_split
            out_dir.mkdir(parents=True, exist_ok=True)
            swept = sweep_tmp(out_dir)
            shards = sorted(str(p) for p in hf_dirs[out_split].glob("*.parquet"))
            print(f"[{out_split}] {len(shards)} shards -> {out_dir} (swept {swept} .tmp)", flush=True)
            jobs = [(sh, str(out_dir), str(quarantine_dir)) for sh in shards]
            if a.workers <= 1:
                # In-process, so the tests (and a debug run) need no picklable module path.
                results = map(process_shard, jobs)
                for name, w, s, q in results:
                    print(f"  {name}: wrote {w}, skipped {s}, quarantined {q}", flush=True)
            else:
                with Pool(a.workers) as pool:
                    for name, w, s, q in pool.imap_unordered(process_shard, jobs):
                        print(f"  {name}: wrote {w}, skipped {s}, quarantined {q}", flush=True)

    counts = {}
    for out_split, exp in EXPECTED.items():
        out_dir = a.out / out_split
        if not out_dir.is_dir():
            print(f"[{out_split}] MISSING directory {out_dir}: MISMATCH", flush=True)
            rc = 1
            continue
        pairs, jpg_only, json_only, tmps, digest = split_state(out_dir)
        ok = exp == pairs and not jpg_only and not json_only and not tmps
        if a.check and previous.get(out_split, {}).get("ids_sha256") not in (None, digest):
            ok = False
            print(f"[{out_split}] id hash {digest[:12]} differs from {COUNTS_FILE}", flush=True)
        if not ok:
            rc = 1
        counts[out_split] = {"pairs": pairs, "expected": exp, "ids_sha256": digest}
        print(f"[{out_split}] pairs {pairs} (expected {exp}), jpg-only {jpg_only}, "
              f"json-only {json_only}, tmp {tmps}, ids sha256 {digest[:12]}: {'OK' if ok else 'MISMATCH'}",
              flush=True)

    if a.check and a.record:
        if not (a.out / "_quarantine" / "MANIFEST.txt").is_file():
            write_quarantine_manifest(a.out)
        if rc == 0 and not counts_path.is_file():
            with open(counts_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(counts, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"[counts] recorded {counts_path}", flush=True)
    if a.check:
        if not check_quarantine_manifest(a.out):
            print("[quarantine] MANIFEST.txt missing or differs from the directory: MISMATCH", flush=True)
            rc = 1
        else:
            print(f"[quarantine] {len(quarantine_listing(a.out))} files, manifest matches", flush=True)
        if not counts_path.is_file():
            print(f"[counts] {COUNTS_FILE} missing (write mode records it): MISMATCH", flush=True)
            rc = 1
    else:
        n_q = write_quarantine_manifest(a.out)
        print(f"[quarantine] {n_q} files listed in {a.out / '_quarantine' / 'MANIFEST.txt'}", flush=True)
        if rc == 0:
            with open(counts_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(counts, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"[counts] wrote {counts_path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
