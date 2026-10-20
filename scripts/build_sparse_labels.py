#!/usr/bin/env python3
"""Turn a sidewalk-auto-labeler run into sparse-labeled Stage 2 training panos (#263).

The schema this writes, and what ``stage_two/train.py --loss-mask`` does with it, is in
``docs/sparse_supervision.md``. In short, each emitted pano gets ``<id>.jpg`` (4096x2048)
and ``<id>.json`` in the Stage 2 dataset format plus a ``supervision`` block:

    sparse          verified positives only; training sees 3-sigma discs around them and
                    nothing else. Positives come from two sources:
                      strong_site  the pano's own detections that are members of a fused site
                                   with >= 3 operational panos (sites.jsonl, labeler
                                   fuse_sites.py; operational = confidence >= 0.55)
                      mined        consensus-mined misses (labels.jsonl, labeler
                                   scripts/mine_city.py, rows with emit == true)
    verified_empty  PROPOSED rule: zero stored peaks (results.jsonl keeps every peak >= 0.1),
                    no strong site within 25 m, and >= 2 other panos within 25 m that do carry
                    peaks. Emitted with an empty point list; training sees the whole pano.

Every other pano in the run is skipped. Benchmark panos (``--benchmark-ids``, plus any mined
row flagged ``benchmark_pano``) are never emitted.

Positions are normalized (0..1 of width and height), so resizing the native pano to 4096x2048
changes no label. Nothing here touches the network.

**Reproducibility.** The JSON is a pure function of the inputs (their sha256s are in the
manifest, and each pano's JSON carries the sha256 of its native image), written with sorted
keys, rounded floats and LF line ends. ``labels_manifest.json`` records counts per mode and
source and ``labels_sha256``, a hash over the sorted JSON contents. ``--check`` re-derives
every JSON from the inputs without writing anything and compares it against both the
manifest and the files on disk. The re-encoded JPEGs are *not* hashed: libjpeg versions
differ across machines, and the native image's hash is the durable fact.

Usage:

    python scripts/build_sparse_labels.py \\
        --results runs/lyon/results.jsonl --sites runs/lyon/sites.jsonl \\
        --mined runs/lyon/mined/labels.jsonl --panos /data/lyon/index.csv \\
        --benchmark-ids benchmark/lyon/ids.txt --city lyon \\
        --out /gscratch/scrubbed/$USER/sparse/lyon --limit 20000 --seed 0

    python scripts/build_sparse_labels.py ...same args... --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rampnet.gsv import heading_to_azimuth  # noqa: E402

#: A strong site has at least this many operational panos (the miners' rule, #158).
MIN_OP_PANOS = 3
#: verified_empty (proposed): radius for "no strong site nearby" and for the neighbour count.
EMPTY_RADIUS_M = 25.0
#: verified_empty (proposed): other panos within EMPTY_RADIUS_M that must carry peaks.
EMPTY_MIN_PEAK_NEIGHBOURS = 2
#: The Stage 2 dataset's image size, width x height.
OUT_SIZE = (4096, 2048)
JPEG_QUALITY = 95
#: Decimal places for every float written (positions, coordinates, headings).
ROUND = 6
EARTH_RADIUS_M = 6371008.8
MANIFEST = "labels_manifest.json"
SOURCES = ("strong_site", "mined")
CITY_RE = re.compile(r"^[a-z0-9_]+$")


# ------------------------------------------------------------------------- inputs

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(path):
    """{pano_id: pano record} from a labeler results.jsonl. A pano id that appears twice
    with different content is an error; an exact repeat collapses."""
    panos = {}
    for rec in read_jsonl(path):
        pano = rec["pano"]
        pid = str(pano["panorama_id"])
        row = {
            "pano_id": pid,
            "lat": float(pano["lat"]), "lng": float(pano["lng"]),
            "camera_heading": pano.get("camera_heading"),
            "width": pano.get("width"), "height": pano.get("height"),
            "capture_date": pano.get("capture_date"), "source": pano.get("source"),
            "camera_make": pano.get("camera_make"), "camera_model": pano.get("camera_model"),
            "n_peaks": len(rec.get("detections") or []),
        }
        if pid in panos and panos[pid] != row:
            raise SystemExit(f"{path}: pano {pid} appears twice with different content")
        panos[pid] = row
    return panos


def load_benchmark_ids(path):
    if path is None:
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def resolve_images(panos_arg, ids):
    """{pano_id: (path, sha256 or None)} for the ids whose native image exists.

    ``panos_arg`` is a directory of ``<id>.jpg``, or a labeler archive ``index.csv``
    (panorama_id,filename,bytes,sha256 -- labeler scripts/export_benchmark.py), whose files
    sit in a ``panos/`` directory beside it or beside it directly. The index's sha256 is
    carried and checked when the image is read.
    """
    p = Path(panos_arg)
    out = {}
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            index = {row["panorama_id"]: row for row in csv.DictReader(f)}
        for pid in ids:
            row = index.get(pid)
            if row is None:
                continue
            for cand in (p.parent / "panos" / row["filename"], p.parent / row["filename"]):
                if cand.is_file():
                    out[pid] = (cand, row.get("sha256") or None)
                    break
    else:
        if not p.is_dir():
            raise SystemExit(f"--panos {p}: neither a directory nor an index.csv")
        for pid in ids:
            cand = p / f"{pid}.jpg"
            if cand.is_file():
                out[pid] = (cand, None)
    return out


# ------------------------------------------------------------------------ geometry

def haversine_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


class Grid:
    """Bucket (lat, lng) items into ~cell_m squares so a radius query reads 3x3 cells.
    Candidates are filtered with haversine, so the bucketing only has to be conservative."""

    def __init__(self, cell_m, lat0):
        # 110 km per degree is below the sphere's 111.2 km, so cells come out slightly
        # LARGER than cell_m and a 3x3 read can never miss a point inside the radius.
        self.cell_lat = cell_m / 110_000.0
        self.cell_lng = cell_m / (110_000.0 * max(math.cos(math.radians(lat0)), 0.01))
        self.cells = defaultdict(list)

    def key(self, lat, lng):
        return (math.floor(lat / self.cell_lat), math.floor(lng / self.cell_lng))

    def add(self, lat, lng, item):
        self.cells[self.key(lat, lng)].append((lat, lng, item))

    def within(self, lat, lng, radius_m):
        ki, kj = self.key(lat, lng)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for la, ln, item in self.cells.get((ki + di, kj + dj), ()):
                    if haversine_m(lat, lng, la, ln) <= radius_m:
                        yield item


# ----------------------------------------------------------------------- selection

def select(panos, sites, mined, city, min_op_panos=MIN_OP_PANOS,
           empty_radius_m=EMPTY_RADIUS_M, empty_min_neighbours=EMPTY_MIN_PEAK_NEIGHBOURS):
    """({pano_id: (mode, [(x, y, source, site_uid)])}, stats). Pure: no files, no images.

    site_uid is ``"<city>:<site_id>"`` -- a site id is a per-run serial, meaningless without
    the run it came from.
    """
    strong = [s for s in sites if s["n_operational_panos"] >= min_op_panos]
    points = defaultdict(dict)          # pid -> {(x, y): (source, site_uid)}, first wins
    mined_bench = set()

    def add(pid, x, y, source, site_id, what):
        if pid not in panos:
            raise SystemExit(f"{what} names pano {pid}, which is not in results.jsonl")
        key = (round(float(x), ROUND), round(float(y), ROUND))
        points[pid].setdefault(key, (source, f"{city}:{site_id}"))

    for s in sorted(strong, key=lambda s: s["site_id"]):
        for m in s["members"]:
            add(str(m["pano_id"]), m["x_normalized"], m["y_normalized"], "strong_site",
                s["site_id"], f"strong site {s['site_id']}")
    n_mined_rows = 0
    for r in sorted(mined, key=lambda r: (str(r["site_id"]), str(r["pano_id"]))):
        if r.get("benchmark_pano"):
            mined_bench.add(str(r["pano_id"]))
        if r.get("emit") is True and r.get("x") is not None and r.get("y") is not None:
            n_mined_rows += 1
            add(str(r["pano_id"]), r["x"], r["y"], "mined", r["site_id"],
                f"mined row (site {r['site_id']})")

    out = {}
    for pid, pts in points.items():
        out[pid] = ("sparse", sorted((x, y, src, uid) for (x, y), (src, uid) in pts.items()))

    # verified_empty (proposed). Strong sites and peak-carrying panos, bucketed once.
    lat0 = sum(p["lat"] for p in panos.values()) / max(1, len(panos))
    site_grid, peak_grid = Grid(empty_radius_m, lat0), Grid(empty_radius_m, lat0)
    for s in strong:
        site_grid.add(float(s["lat"]), float(s["lng"]), s["site_id"])
    for p in panos.values():
        if p["n_peaks"]:
            peak_grid.add(p["lat"], p["lng"], p["pano_id"])
    n_empty_candidates = 0
    for pid, p in panos.items():
        if p["n_peaks"] or pid in out:
            continue
        n_empty_candidates += 1
        if next(site_grid.within(p["lat"], p["lng"], empty_radius_m), None) is not None:
            continue
        neighbours = sum(1 for q in peak_grid.within(p["lat"], p["lng"], empty_radius_m)
                         if q != pid)
        if neighbours >= empty_min_neighbours:
            out[pid] = ("verified_empty", [])

    stats = {"panos_in_run": len(panos), "sites": len(sites), "strong_sites": len(strong),
             "mined_emit_rows": n_mined_rows, "zero_peak_panos": n_empty_candidates}
    return out, mined_bench, stats


# ------------------------------------------------------------------------- records

def r6(v):
    return None if v is None else round(float(v), ROUND)


def build_record(pano, mode, pts, run_sha, city, native_sha):
    heading = pano["camera_heading"]
    return {
        "pano_id": pano["pano_id"],
        "curb_ramp_points_normalized": [[x, y] for x, y, _, _ in pts],
        "pano_coord": [r6(pano["lat"]), r6(pano["lng"])],
        "pano_azimuth": None if heading is None else r6(heading_to_azimuth(float(heading))),
        "pano_meta": {
            "source": pano["source"], "camera_make": pano["camera_make"],
            "camera_model": pano["camera_model"], "capture_date": pano["capture_date"],
            "native_width": pano["width"], "native_height": pano["height"],
            "native_sha256": native_sha,
        },
        "supervision": {
            "mode": mode,
            "sources": [src for _, _, src, _ in pts],
            "site_ids": [uid for _, _, _, uid in pts],
            "run_sha": run_sha,
            "city": city,
        },
    }


def record_bytes(rec):
    return (json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def labels_sha256(blobs):
    """sha256 over ``<id>.json\\n`` + the file's bytes, in sorted pano-id order."""
    h = hashlib.sha256()
    for pid in sorted(blobs):
        h.update(f"{pid}.json\n".encode("utf-8"))
        h.update(blobs[pid])
    return h.hexdigest()


# ---------------------------------------------------------------------------- build

def plan(args):
    """Everything up to (not including) image work: (records {pid: bytes}, images
    {pid: (path, index sha)}, manifest dict). Deterministic in the inputs and --seed."""
    if not CITY_RE.match(args.city):
        raise SystemExit(f"--city {args.city!r}: lowercase slug [a-z0-9_] expected")
    panos = load_results(args.results)
    sites = read_jsonl(args.sites)
    mined = read_jsonl(args.mined) if args.mined else []
    chosen, mined_bench, stats = select(panos, sites, mined, args.city)
    bench = load_benchmark_ids(args.benchmark_ids) | mined_bench
    excluded = sorted(pid for pid in chosen if pid in bench)
    for pid in excluded:
        del chosen[pid]
    images = resolve_images(args.panos, sorted(chosen))
    missing = sorted(pid for pid in chosen if pid not in images)
    for pid in missing:
        del chosen[pid]
    eligible = sorted(chosen)
    if args.limit is not None and args.limit < len(eligible):
        eligible = sorted(random.Random(args.seed).sample(eligible, args.limit))
    run_sha = sha256_file(args.results)

    native = {}

    def native_sha(pid):
        path, idx_sha = images[pid]
        return idx_sha if idx_sha else sha256_file(path)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pid, sha in zip(eligible, ex.map(native_sha, eligible)):
            native[pid] = sha
    records = {pid: record_bytes(build_record(panos[pid], chosen[pid][0], chosen[pid][1],
                                              run_sha, args.city, native[pid]))
               for pid in eligible}

    by_mode = Counter(chosen[pid][0] for pid in eligible)
    pts_by_src = Counter(src for pid in eligible for _, _, src, _ in chosen[pid][1])
    panos_by_src = Counter(src for pid in eligible
                           for src in {s for _, _, s, _ in chosen[pid][1]})
    inputs = {"results": args.results, "sites": args.sites, "mined": args.mined,
              "benchmark_ids": args.benchmark_ids}
    manifest = {
        "tool": "scripts/build_sparse_labels.py",
        "issue": "https://github.com/ProjectSidewalk/RampNet/issues/263",
        "city": args.city,
        "params": {"min_op_panos": MIN_OP_PANOS, "empty_radius_m": EMPTY_RADIUS_M,
                   "empty_min_peak_neighbours": EMPTY_MIN_PEAK_NEIGHBOURS,
                   "verified_empty_rule": "proposed",
                   "limit": args.limit, "seed": args.seed,
                   "out_size": list(OUT_SIZE), "jpeg_quality": JPEG_QUALITY},
        "inputs": {k: None if v is None else {"name": Path(v).name, "sha256": sha256_file(v)}
                   for k, v in inputs.items()},
        "counts": {**stats,
                   "selected_before_exclusions": len(chosen) + len(excluded) + len(missing),
                   "excluded_benchmark": len(excluded), "missing_image": len(missing),
                   "emitted": len(eligible),
                   "by_mode": {m: by_mode.get(m, 0) for m in ("sparse", "verified_empty")},
                   "points_by_source": {s: pts_by_src.get(s, 0) for s in SOURCES},
                   "panos_by_source": {s: panos_by_src.get(s, 0) for s in SOURCES}},
        "missing_image_ids": missing,
        "pano_ids": eligible,
        "labels_sha256": labels_sha256(records),
    }
    return records, {pid: images[pid] for pid in eligible}, manifest


def write_image(src, dst, expect_sha):
    """Native jpg -> 4096x2048 bilinear, JPEG q95, written atomically."""
    from PIL import Image
    data = Path(src).read_bytes()
    if expect_sha and hashlib.sha256(data).hexdigest() != expect_sha:
        raise SystemExit(f"{src}: sha256 does not match its index.csv row")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(OUT_SIZE, Image.Resampling.BILINEAR)
    tmp = dst.with_name(dst.name + ".tmp")
    img.save(tmp, format="JPEG", quality=JPEG_QUALITY)
    os.replace(tmp, dst)


def write_bytes(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def build(args):
    records, images, manifest = plan(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for pid, blob in records.items():
        jpath, ipath = out / f"{pid}.json", out / f"{pid}.jpg"
        if jpath.exists():
            old = jpath.read_bytes()
            if b'"supervision"' not in old:
                raise SystemExit(f"{jpath} exists and is not a sparse-label JSON (a dense "
                                 f"pano?); refusing to overwrite it")
            if old == blob and ipath.exists():
                continue                                   # resume: already done
        todo.append(pid)

    def one(pid):
        path, idx_sha = images[pid]
        # image first, JSON second: a JSON on disk always has its finished image beside it
        write_image(path, out / f"{pid}.jpg", idx_sha)
        write_bytes(out / f"{pid}.json", records[pid])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)} panos written", flush=True)
    write_bytes(out / MANIFEST,
                (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    c = manifest["counts"]
    print(f"{args.city}: {c['emitted']} panos emitted ({c['by_mode']}), "
          f"{len(todo)} written this run, {c['excluded_benchmark']} benchmark excluded, "
          f"{c['missing_image']} without a native image")
    print(f"labels_sha256 {manifest['labels_sha256']}")
    return 0


def check(args):
    """Re-derive every JSON from the inputs and compare with the manifest and the disk."""
    out = Path(args.out)
    mpath = out / MANIFEST
    if not mpath.exists():
        raise SystemExit(f"--check: no {mpath}")
    have = json.loads(mpath.read_text(encoding="utf-8"))
    records, _, want = plan(args)
    problems = []
    if want["labels_sha256"] != have.get("labels_sha256"):
        problems.append(f"labels_sha256 from the inputs {want['labels_sha256']} != manifest "
                        f"{have.get('labels_sha256')}")
    for key in ("inputs", "counts", "params", "pano_ids"):
        if want[key] != have.get(key):
            problems.append(f"manifest '{key}' differs from a rebuild")
    on_disk = {}
    for pid in want["pano_ids"]:
        jpath = out / f"{pid}.json"
        if not jpath.exists() or not (out / f"{pid}.jpg").exists():
            problems.append(f"{pid}: .json or .jpg missing")
            continue
        on_disk[pid] = jpath.read_bytes()
        if on_disk[pid] != records[pid]:
            problems.append(f"{pid}.json differs from a rebuild")
    if len(on_disk) == len(records) and labels_sha256(on_disk) != want["labels_sha256"]:
        problems.append("hash of the JSONs on disk differs")
    extra = sorted(p.stem for p in out.glob("*.json")
                   if p.name != MANIFEST and p.stem not in records
                   and b'"supervision"' in p.read_bytes())
    if extra:
        problems.append(f"{len(extra)} sparse-label JSONs on disk are not in this build, "
                        f"e.g. {extra[:3]}")
    for p in problems[:20]:
        print("FAIL", p)
    if problems:
        print(f"--check: {len(problems)} problem(s)")
        return 1
    print(f"--check OK: {len(records)} panos, labels_sha256 {want['labels_sha256']}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="labeler results.jsonl of the run")
    ap.add_argument("--sites", required=True, help="labeler fuse_sites.py sites.jsonl")
    ap.add_argument("--mined", default=None, help="labeler mine_city.py labels.jsonl")
    ap.add_argument("--panos", required=True,
                    help="directory of native <id>.jpg, or a labeler archive index.csv")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--city", required=True, help="city slug, e.g. lyon")
    ap.add_argument("--benchmark-ids", default=None,
                    help="file of pano ids (one per line) never to emit")
    ap.add_argument("--limit", type=int, default=None,
                    help="emit a seeded uniform sample of this many eligible panos")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit (default 0)")
    ap.add_argument("--workers", type=int, default=4,
                    help="threads for hashing and resizing (default 4); output is "
                         "independent of it")
    ap.add_argument("--check", action="store_true",
                    help="re-derive the labels and compare with --out; writes nothing")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    return check(args) if args.check else build(args)


if __name__ == "__main__":
    sys.exit(main())
