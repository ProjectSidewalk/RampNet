#!/usr/bin/env python3
"""Turn a sidewalk-auto-labeler run into sparse-labeled Stage 2 training panos (#263).

The schema this writes, and what ``stage_two/train.py --loss-mask`` does with it, is in
``docs/sparse_supervision.md``. In short, each emitted pano gets ``<id>.jpg`` (4096x2048)
and ``<id>.json`` in the Stage 2 dataset format plus a ``supervision`` block:

    sparse          verified positives; training sees 3-sigma discs around them, minus the
                    discs of the pano's other stored peaks (``ignore_points``), and nothing
                    else. Positives come from two sources:
                      strong_site  members of a fused site (sites.jsonl, labeler fuse_sites.py)
                                   with confidence >= --member-min-confidence (0.55) and
                                   in_refit true; a site is strong when those members span
                                   >= 3 panos (recounted here, not read from the file)
                      mined        consensus-mined misses (labels.jsonl, labeler
                                   scripts/mine_city.py, rows with emit == true)
    verified_empty  PROPOSED, UNREVIEWED, and off unless --emit-verified-empty: zero stored
                    peaks, no site of any tier within 25 m, and >= 2 other panos within 25 m
                    that carry peaks. Emitted with an empty point list; training sees the
                    whole pano.

Every other pano in the run is skipped. Benchmark panos (``--benchmark-ids``, required when
``benchmark/<city>/`` exists in this repo, plus any mined row flagged ``benchmark_pano``) and
the panos of ``--exclude-ids`` (a spatial buffer around them) are never emitted, and every
site with a member among them is dropped whole.

The run must store peaks down to 0.1 (``detection_storage_floor`` in its manifest.json):
"zero stored peaks" and the ignore points both mean nothing on a run that kept only >= 0.55.

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
        --benchmark-ids benchmark/lyon/ids.txt --exclude-ids benchmark/lyon/exclude_ids.txt \\
        --city lyon --out /gscratch/scrubbed/$USER/sparse/lyon --limit 20000 --seed 0

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

#: A strong site has qualifying members on at least this many panos (the miners' rule, #158).
MIN_OP_PANOS = 3
#: A site member is a positive only at or above this confidence (the operational threshold).
MEMBER_MIN_CONFIDENCE = 0.55
#: The run must have stored every peak down to at least this confidence (labeler #28).
MAX_STORAGE_FLOOR = 0.1
#: What labeler main.py assumes for a manifest written before the floor existed.
PRE_FLOOR_DEFAULT = 0.55
#: verified_empty (proposed): radius for "no site nearby" and for the neighbour count.
EMPTY_RADIUS_M = 25.0
#: verified_empty (proposed): other panos within EMPTY_RADIUS_M that must carry peaks.
EMPTY_MIN_PEAK_NEIGHBOURS = 2
#: A stored peak this close to a positive (normalized x, y) is that positive, not an ignore
#: point: 2 heatmap pixels at 512x1024. Covers a mined position read from a separate floor
#: pass of the same model; an exact match is the common case.
SAME_PEAK_TOL = (2 / 1024, 2 / 512)
#: The Stage 2 dataset's image size, width x height.
OUT_SIZE = (4096, 2048)
JPEG_QUALITY = 95
#: Decimal places for every float written (positions, coordinates, headings).
ROUND = 6
EARTH_RADIUS_M = 6371008.8
MANIFEST = "labels_manifest.json"
SOURCES = ("strong_site", "mined")
CITY_RE = re.compile(r"^[a-z0-9_]+$")
#: Where committed benchmark bundles live; a city with a bundle must pass --benchmark-ids.
BENCHMARK_ROOT = REPO_ROOT / "benchmark"


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
        peaks = tuple((float(d["x_normalized"]), float(d["y_normalized"]),
                       float(d["confidence"])) for d in (rec.get("detections") or []))
        row = {
            "pano_id": pid,
            "lat": float(pano["lat"]), "lng": float(pano["lng"]),
            "camera_heading": pano.get("camera_heading"),
            "width": pano.get("width"), "height": pano.get("height"),
            "capture_date": pano.get("capture_date"), "source": pano.get("source"),
            "camera_make": pano.get("camera_make"), "camera_model": pano.get("camera_model"),
            "peaks": peaks, "n_peaks": len(peaks),
        }
        if pid in panos and panos[pid] != row:
            raise SystemExit(f"{path}: pano {pid} appears twice with different content")
        panos[pid] = row
    return panos


def load_run_floor(manifest_path):
    """The run's detection_storage_floor; refuse a run that stored less than every peak
    >= MAX_STORAGE_FLOOR (review H1). A manifest without the key predates the floor and
    stored >= 0.55 only, which is what labeler main.py assumes too."""
    if not Path(manifest_path).is_file():
        raise SystemExit(f"no run manifest at {manifest_path}; pass --manifest (the labeler's "
                         f"manifest.json beside results.jsonl)")
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    floor = float(m.get("detection_storage_floor", PRE_FLOOR_DEFAULT))
    if floor > MAX_STORAGE_FLOOR:
        raise SystemExit(f"{manifest_path}: detection_storage_floor {floor} > "
                         f"{MAX_STORAGE_FLOOR}. On such a run 'zero stored peaks' and the "
                         f"ignore points only cover peaks >= {floor}; re-run the labeler "
                         f"with the floor before building training labels from it.")
    return floor


def load_sites_meta(meta_path, member_min_conf):
    """fuse_sites.py's params (written beside sites.jsonl as <stem>_meta.json). Refused when
    the fuse dropped detections below a confidence the member filter would keep (review H2)."""
    if not Path(meta_path).is_file():
        raise SystemExit(f"no sites meta at {meta_path}; pass --sites-meta (fuse_sites.py "
                         f"writes <sites stem>_meta.json beside sites.jsonl)")
    params = json.loads(Path(meta_path).read_text(encoding="utf-8")).get("params") or {}
    mc = params.get("min_confidence")
    if mc is None:
        raise SystemExit(f"{meta_path}: no params.min_confidence")
    if float(mc) > member_min_conf:
        raise SystemExit(f"{meta_path}: the fuse ran at min_confidence {mc}, above the member "
                         f"threshold {member_min_conf}; sites would be missing members the "
                         f"recount needs")
    return params


def load_ids(path):
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
    """Bucket (lat, lng) items into cells at least cell_m on a side, so a radius query reads
    3x3 cells. Candidates are filtered with haversine, so the bucketing only has to be
    conservative: the longitude cell is sized at ``lat_ref``, the run's largest |latitude|,
    where a degree of longitude is shortest (review L5)."""

    def __init__(self, cell_m, lat_ref):
        # 110 km per degree is below the sphere's 111.2 km, so cells come out slightly
        # LARGER than cell_m and a 3x3 read can never miss a point inside the radius.
        self.cell_lat = cell_m / 110_000.0
        self.cell_lng = cell_m / (110_000.0 * max(math.cos(math.radians(abs(lat_ref))), 0.01))
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

def qualifying_members(site, member_min_conf):
    """Members that may be positives: confidence >= the threshold and, when the field is
    present, in_refit true (fuse_sites.py's sub-threshold support members are in_refit
    false and never move a site)."""
    return [m for m in site["members"]
            if float(m["confidence"]) >= member_min_conf and m.get("in_refit", True) is True]


def ignore_points(peaks, positives):
    """The pano's stored peaks that are not positives (review M4), sorted."""
    tx, ty = SAME_PEAK_TOL
    out = set()
    for x, y, _ in peaks:
        if not any(abs(x - px) <= tx and abs(y - py) <= ty for px, py, _, _ in positives):
            out.add((round(x, ROUND), round(y, ROUND)))
    return sorted(out)


def select(panos, sites, mined, city, member_min_conf=MEMBER_MIN_CONFIDENCE,
           emit_empty=False, benchmark=frozenset(), exclude=frozenset(),
           min_op_panos=MIN_OP_PANOS, empty_radius_m=EMPTY_RADIUS_M,
           empty_min_neighbours=EMPTY_MIN_PEAK_NEIGHBOURS):
    """({pano_id: (mode, [(x, y, source, site_uid)], ignore_points)}, stats). Pure: no
    files, no images.

    site_uid is ``"<city>:<site_id>"`` for a sites.jsonl site and ``"<city>:mined:<id>"``
    for a mined row (mine_city.py fuses on its own, so its site ids are a different serial).
    """
    mined_bench = {str(r["pano_id"]) for r in mined if r.get("benchmark_pano")}
    bench = set(benchmark) | mined_bench
    exclude = set(exclude) - bench
    banned = bench | exclude
    stats = Counter()

    points = defaultdict(dict)          # pid -> {(x, y): (source, site_uid)}, first wins

    def add(pid, x, y, source, uid, what):
        if pid not in panos:
            raise SystemExit(f"{what} names pano {pid}, which is not in results.jsonl")
        key = (round(float(x), ROUND), round(float(y), ROUND))
        points[pid].setdefault(key, (source, uid))

    n_strong = 0
    for s in sorted(sites, key=lambda s: s["site_id"]):
        member_panos = {str(m["pano_id"]) for m in s["members"]}
        if member_panos & bench:
            stats["dropped_sites_benchmark"] += 1
            continue
        if member_panos & exclude:
            stats["dropped_sites_excluded"] += 1
            continue
        keep = qualifying_members(s, member_min_conf)
        if len({str(m["pano_id"]) for m in keep}) < min_op_panos:
            continue
        n_strong += 1
        for m in keep:
            add(str(m["pano_id"]), m["x_normalized"], m["y_normalized"], "strong_site",
                f"{city}:{s['site_id']}", f"site {s['site_id']}")

    by_site = defaultdict(list)
    for r in mined:
        by_site[str(r["site_id"])].append(r)
    n_mined_rows = 0
    for sid in sorted(by_site):
        rows = sorted(by_site[sid], key=lambda r: str(r["pano_id"]))
        if any(str(r["pano_id"]) in banned for r in rows):
            stats["dropped_mined_sites"] += 1
            continue
        for r in rows:
            if r.get("emit") is True and r.get("x") is not None and r.get("y") is not None:
                n_mined_rows += 1
                add(str(r["pano_id"]), r["x"], r["y"], "mined", f"{city}:mined:{sid}",
                    f"mined row (site {sid})")

    out = {}
    for pid, pts in points.items():
        pos = sorted((x, y, src, uid) for (x, y), (src, uid) in pts.items())
        out[pid] = ("sparse", pos, ignore_points(panos[pid]["peaks"], pos))

    n_empty_candidates = 0
    if emit_empty:
        # verified_empty (proposed, unreviewed). Vetoed by ANY site of any tier -- its own
        # position and its members' ground points -- within the radius (review H1).
        lat_ref = max([abs(p["lat"]) for p in panos.values()]
                      + [abs(float(s["lat"])) for s in sites], default=0.0)
        site_grid, peak_grid = Grid(empty_radius_m, lat_ref), Grid(empty_radius_m, lat_ref)
        for s in sites:
            site_grid.add(float(s["lat"]), float(s["lng"]), s["site_id"])
            for m in s["members"]:
                if m.get("lat") is not None and m.get("lng") is not None:
                    site_grid.add(float(m["lat"]), float(m["lng"]), s["site_id"])
        for p in panos.values():
            if p["n_peaks"]:
                peak_grid.add(p["lat"], p["lng"], p["pano_id"])
        for pid, p in panos.items():
            if p["n_peaks"] or pid in out:
                continue
            n_empty_candidates += 1
            if next(site_grid.within(p["lat"], p["lng"], empty_radius_m), None) is not None:
                continue
            neighbours = sum(1 for q in peak_grid.within(p["lat"], p["lng"], empty_radius_m)
                             if q != pid)
            if neighbours >= empty_min_neighbours:
                out[pid] = ("verified_empty", [], [])

    excluded_bench = sorted(pid for pid in out if pid in bench)
    excluded_buffer = sorted(pid for pid in out if pid in exclude)
    for pid in excluded_bench + excluded_buffer:
        del out[pid]
    stats.update({"panos_in_run": len(panos), "sites": len(sites), "strong_sites": n_strong,
                  "mined_emit_rows": n_mined_rows, "zero_peak_panos_considered":
                  n_empty_candidates, "excluded_benchmark_panos": len(excluded_bench),
                  "excluded_buffer_panos": len(excluded_buffer)})
    for k in ("dropped_sites_benchmark", "dropped_sites_excluded", "dropped_mined_sites"):
        stats.setdefault(k, 0)
    return out, dict(stats)


# ------------------------------------------------------------------------- records

def r6(v):
    return None if v is None else round(float(v), ROUND)


def build_record(pano, mode, pts, ignore, run_sha, city, native_sha):
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
            "ignore_points": [[x, y] for x, y in ignore],
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
    if args.benchmark_ids is None and (BENCHMARK_ROOT / args.city).is_dir():
        raise SystemExit(f"{BENCHMARK_ROOT / args.city} exists, so --benchmark-ids is "
                         f"required: its panos must never train")
    manifest_path = args.manifest or str(Path(args.results).with_name("manifest.json"))
    floor = load_run_floor(manifest_path)
    meta_path = args.sites_meta or str(Path(args.sites).with_name(
        Path(args.sites).stem + "_meta.json"))
    fuse_params = load_sites_meta(meta_path, args.member_min_confidence)
    panos = load_results(args.results)
    sites = read_jsonl(args.sites)
    mined = read_jsonl(args.mined) if args.mined else []
    chosen, stats = select(panos, sites, mined, args.city, args.member_min_confidence,
                           args.emit_verified_empty, load_ids(args.benchmark_ids),
                           load_ids(args.exclude_ids))
    images = resolve_images(args.panos, sorted(chosen))
    missing = sorted(pid for pid in chosen if pid not in images)
    for pid in missing:
        del chosen[pid]
    eligible = sorted(chosen)
    if args.limit is not None and args.limit < len(eligible):
        eligible = sorted(random.Random(args.seed).sample(eligible, args.limit))
    run_sha = sha256_file(args.results)

    def native_sha(pid):
        path, idx_sha = images[pid]
        return idx_sha if idx_sha else sha256_file(path)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        native = dict(zip(eligible, ex.map(native_sha, eligible)))
    records = {pid: record_bytes(build_record(panos[pid], *chosen[pid], run_sha, args.city,
                                              native[pid]))
               for pid in eligible}

    by_mode = Counter(chosen[pid][0] for pid in eligible)
    pts_by_src = Counter(src for pid in eligible for _, _, src, _ in chosen[pid][1])
    panos_by_src = Counter(src for pid in eligible
                           for src in {s for _, _, s, _ in chosen[pid][1]})
    inputs = {"results": args.results, "sites": args.sites, "sites_meta": meta_path,
              "run_manifest": manifest_path, "mined": args.mined,
              "benchmark_ids": args.benchmark_ids, "exclude_ids": args.exclude_ids}
    manifest = {
        "tool": "scripts/build_sparse_labels.py",
        "issue": "https://github.com/ProjectSidewalk/RampNet/issues/263",
        "city": args.city,
        "params": {"min_op_panos": MIN_OP_PANOS,
                   "member_min_confidence": args.member_min_confidence,
                   "member_requires_in_refit": True,
                   "run_detection_storage_floor": floor,
                   "same_peak_tol": list(SAME_PEAK_TOL),
                   "emit_verified_empty": args.emit_verified_empty,
                   "verified_empty_rule": "proposed, unreviewed",
                   "empty_radius_m": EMPTY_RADIUS_M,
                   "empty_min_peak_neighbours": EMPTY_MIN_PEAK_NEIGHBOURS,
                   "limit": args.limit, "seed": args.seed,
                   "out_size": list(OUT_SIZE), "jpeg_quality": JPEG_QUALITY},
        "fuse_params": fuse_params,
        "inputs": {k: None if v is None else {"name": Path(v).name, "sha256": sha256_file(v)}
                   for k, v in inputs.items()},
        "counts": {**stats,
                   "missing_image": len(missing),
                   "emitted": len(eligible),
                   "by_mode": {m: by_mode.get(m, 0) for m in ("sparse", "verified_empty")},
                   "points_by_source": {s: pts_by_src.get(s, 0) for s in SOURCES},
                   "panos_by_source": {s: panos_by_src.get(s, 0) for s in SOURCES},
                   "ignore_points": sum(len(chosen[pid][2]) for pid in eligible)},
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


def prune_plan(out, records):
    """Files in ``out`` that are not part of this build. Refuses (SystemExit) when ``out``
    holds anything a prune must not delete: a dense pano JSON, or a file that is not a
    pano's jpg/json, a leftover .tmp, or the manifest (review M3)."""
    keep = {MANIFEST} | {f"{pid}.{ext}" for pid in records for ext in ("jpg", "json")}
    doomed = []
    for p in sorted(out.iterdir()):
        if p.name in keep:
            continue
        if p.is_dir() or p.suffix not in (".jpg", ".json", ".tmp"):
            raise SystemExit(f"{p} is not a file this tool writes; refusing to prune {out}")
        if p.suffix == ".json" and b'"supervision"' not in p.read_bytes():
            raise SystemExit(f"{p} is a dense pano JSON; refusing to prune {out}")
        doomed.append(p)
    return doomed


def build(args):
    records, images, manifest = plan(args)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.prune:
            raise SystemExit(f"{out} is not empty. Pass --prune to rebuild into it: files "
                             f"this build does not list are deleted, and panos whose JSON "
                             f"is already identical are kept (that is how a build resumes).")
        doomed = prune_plan(out, records)
        for p in doomed:
            p.unlink()
        if doomed:
            print(f"pruned {len(doomed)} files not in this build")
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for pid, blob in records.items():
        jpath, ipath = out / f"{pid}.json", out / f"{pid}.jpg"
        if jpath.exists() and jpath.read_bytes() == blob and ipath.exists():
            continue                                       # resume: already done
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
          f"{len(todo)} written this run, {c['excluded_benchmark_panos']} benchmark and "
          f"{c['excluded_buffer_panos']} buffer panos excluded, {c['missing_image']} "
          f"without a native image")
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
    for key in ("inputs", "counts", "params", "fuse_params", "pano_ids"):
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
    extra = sorted(p.name for p in out.iterdir()
                   if p.name != MANIFEST and p.stem not in records)
    if extra:
        problems.append(f"{len(extra)} files on disk are not in this build, e.g. {extra[:3]}")
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
    ap.add_argument("--manifest", default=None,
                    help="the run's manifest.json (default: beside --results); its "
                         f"detection_storage_floor must be <= {MAX_STORAGE_FLOOR}")
    ap.add_argument("--sites", required=True, help="labeler fuse_sites.py sites.jsonl")
    ap.add_argument("--sites-meta", default=None,
                    help="fuse_sites.py's meta file (default: <sites stem>_meta.json beside "
                         "--sites); its params are recorded in the manifest")
    ap.add_argument("--mined", default=None, help="labeler mine_city.py labels.jsonl")
    ap.add_argument("--panos", required=True,
                    help="directory of native <id>.jpg, or a labeler archive index.csv")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--city", required=True, help="city slug, e.g. lyon")
    ap.add_argument("--benchmark-ids", default=None,
                    help="pano ids (one per line) never to emit; required when "
                         "benchmark/<city>/ exists in this repo")
    ap.add_argument("--exclude-ids", default=None,
                    help="more pano ids never to emit, e.g. the spatial buffer around the "
                         "benchmark (benchmark/<city>/exclude_ids.txt); sites with a member "
                         "among them are dropped whole")
    ap.add_argument("--member-min-confidence", type=float, default=MEMBER_MIN_CONFIDENCE,
                    help=f"site members below this are not positives (default "
                         f"{MEMBER_MIN_CONFIDENCE})")
    ap.add_argument("--emit-verified-empty", action="store_true",
                    help="also emit the PROPOSED, UNREVIEWED verified_empty panos (default off)")
    ap.add_argument("--limit", type=int, default=None,
                    help="emit a seeded uniform sample of this many eligible panos")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit (default 0)")
    ap.add_argument("--workers", type=int, default=4,
                    help="threads for hashing and resizing (default 4); output is "
                         "independent of it")
    ap.add_argument("--prune", action="store_true",
                    help="allow a non-empty --out: delete what this build does not list, "
                         "keep what is already identical (resume)")
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
