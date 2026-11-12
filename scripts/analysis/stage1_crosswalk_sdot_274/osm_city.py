#!/usr/bin/env python3
"""A city's OSM marked crossings and the open-imagery panos beside them (#285 pilot inputs).

Four steps, none of which needs a GPU or the rampnet package. They run in the
sidewalk-auto-labeler environment, because coverage and pano metadata come from its Mapillary source:

``crossings``  One Overpass pull of every ``highway=crossing`` node inside the city's
               administrative boundary (by Wikidata id), plus the ways through them. Writes
               ``crossings.csv`` with the marked / unmarked / unknown class (``osm_arm.osm_class``'s
               rule, restated here) and a grade-separation flag (any way with a tunnel, a bridge
               or a non-zero layer).
``panos``      A seeded sample of marked, at-grade crossings; the z14 Mapillary tiles around them;
               panos within ``--near-m`` of a sampled crossing, thinned to the labeler's 5 m grid
               (``--tracker-census`` reads the panos from a streetscape-tracker census, with no tile requests);
               up to ``--per-crossing`` panos per crossing, seeded. Writes ``candidates.csv``.
``meta``       Graph metadata for each candidate pano: one labeler-format record per pano (SfM position,
               compass, camera) to ``<keep>/osm285/<city>/records.jsonl``, and ``pano_list.csv``, the
               download list for sidewalk-panorama-tools. No image bytes. Resumable.
``store``      The original images, downloaded by sidewalk-panorama-tools' ``DownloadRunner.py -c`` into
               ``<keep>/pano_store/<city>/<id[:2]>/<id>.jpg`` with its ledgers. That is the Project Sidewalk
               pano store layout, so a later deployment of the city can reuse the files. Resumable.

Usage::

    PY=<sidewalk-auto-labeler env>/python.exe;  L=<sidewalk-auto-labeler checkout>
    $PY osm_city.py crossings --city sacramento-ca --wikidata Q18013
    $PY osm_city.py panos --city sacramento-ca --labeler $L --take 300
    $PY osm_city.py meta --city sacramento-ca --labeler $L --keep D:/Git/sal-scratch-keep
    $PY osm_city.py store --city sacramento-ca --labeler $L --keep D:/Git/sal-scratch-keep \
        --pano-tools <sidewalk-panorama-tools checkout> --python <its python>
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "osm_arm", "cities")
SEED = 285
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "osm-crossing-research/0.1"
# Kept in step with pilot.OSM_MARKED / OSM_UNMARKED (not imported: pilot needs pyproj and the SDOT helpers).
OSM_MARKED = {"marked", "zebra", "lines", "ladder", "dashes", "dots", "surface", "pedestrian_signals"}
OSM_UNMARKED = {"unmarked", "no", "informal"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def haversine_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(h))


def osm_class(tags):
    """marked / unmarked / unknown, and which tag decided it."""
    m, c = tags.get("crossing:markings"), tags.get("crossing")
    if m is not None:
        return ("unmarked", "markings=no") if m == "no" else ("marked", "crossing:markings")
    if c in OSM_MARKED:
        return "marked", "crossing=" + c
    if c == "traffic_signals" or tags.get("crossing:signals") == "yes":
        return "marked", "signals_only"
    return ("unmarked", "crossing=" + c) if c in OSM_UNMARKED else ("unknown", "")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- crossings

def overpass(query, attempts=4):
    import requests
    for i in range(attempts):
        r = requests.post(OVERPASS, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=400)
        if r.status_code == 200:
            return r.json()
        log(f"  overpass HTTP {r.status_code}; waiting")
        time.sleep(20 * (i + 1))
    raise SystemExit("Overpass did not answer; check https://overpass-api.de/api/status before retrying")


def cmd_crossings(a):
    out = os.path.join(OUT, a.city)
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, "osm_crossings.json")
    if not os.path.exists(raw) or a.refresh:
        q = (f'[out:json][timeout:300];area["wikidata"="{a.wikidata}"]["boundary"="administrative"]->.a;'
             'node(area.a)["highway"="crossing"]->.c;.c out body;way(bn.c)["highway"];out body;')
        data = overpass(q)
        with open(raw, "w", encoding="utf-8") as f:
            json.dump(data, f)
    data = json.load(open(raw, encoding="utf-8"))
    separated, on_ways = set(), defaultdict(set)
    for e in data["elements"]:
        if e["type"] != "way":
            continue
        t = e.get("tags") or {}
        sep = t.get("tunnel", "no") != "no" or t.get("bridge", "no") != "no" or t.get("layer", "0") not in ("0", "")
        for n in e["nodes"]:
            on_ways[n].add(t.get("highway", ""))
            if sep:
                separated.add(n)
    rows = []
    for e in data["elements"]:
        t = e.get("tags") or {}
        if e["type"] != "node" or t.get("highway") != "crossing":
            continue
        cls, basis = osm_class(t)
        rows.append({"node_id": e["id"], "lat": e["lat"], "lng": e["lon"], "osm_class": cls, "basis": basis,
                     "crossing": t.get("crossing", ""), "markings": t.get("crossing:markings", ""),
                     "grade_separated": int(e["id"] in separated),
                     "way_highways": "|".join(sorted(on_ways.get(e["id"], ())))})
    write_csv(os.path.join(out, "crossings.csv"), rows)
    n = len(rows)
    by = defaultdict(int)
    for r in rows:
        by[(r["osm_class"], r["basis"].split("=")[0])] += 1
    lats, lngs = [r["lat"] for r in rows], [r["lng"] for r in rows]
    summary = {"city": a.city, "wikidata": a.wikidata, "osm_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
               "crossing_nodes": n, "marked": sum(r["osm_class"] == "marked" for r in rows),
               "marked_at_grade": sum(r["osm_class"] == "marked" and not r["grade_separated"] for r in rows),
               "grade_separated": sum(r["grade_separated"] for r in rows),
               "by_class_and_basis": {f"{k[0]}/{k[1] or 'none'}": v for k, v in sorted(by.items())},
               "bbox": [min(lats), min(lngs), max(lats), max(lngs)]}
    with open(os.path.join(out, "crossings_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- panos

def labeler_mapillary(labeler):
    """The labeler's Mapillary source, with its token loaded from the labeler's .env."""
    sys.path.insert(0, labeler)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(labeler, ".env"))
    from sources import mapillary
    mapillary.prepare()
    return mapillary


def tile_of(lat, lng, zoom=14):
    n = 2.0 ** zoom
    return int((lng + 180.0) / 360.0 * n), int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def census_panos(path, sample, near_m):
    """{pano_id: (lat, lng, captured_at_ms, quality)} for the 360 panos of a streetscape-tracker Mapillary
    census (its per-city run CSV: every `is_pano` image of the z14 tiles) within near_m of a sampled crossing.
    No request is made: the tracker already paid for the tiles."""
    import calendar
    import gzip
    bands = defaultdict(list)
    for r in sample:
        bands[int(r["lat"] * 1000)].append((r["lat"], r["lng"]))
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "OK" or row["is_pano"] != "True" or not row["pano_id"] or row["pano_id"] in out:
                continue
            lat, lng = float(row["pano_lat"]), float(row["pano_lon"])
            b = int(lat * 1000)
            if any(haversine_m(lat, lng, clat, clng) <= near_m
                   for k in (b - 1, b, b + 1) for clat, clng in bands.get(k, ())):
                d = row["capture_date"]
                ms = calendar.timegm(time.strptime(d[:10], "%Y-%m-%d")) * 1000 if len(d) >= 10 else 0
                out[row["pano_id"]] = (lat, lng, ms, float(row["quality_score"] or 0))
    return out


def cmd_panos(a):
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely import affinity
    mly = labeler_mapillary(a.labeler)
    out = os.path.join(OUT, a.city)
    cross = [r for r in read_csv(os.path.join(out, "crossings.csv"))
             if r["osm_class"] == "marked" and r["grade_separated"] == "0"]
    for r in cross:
        r["lat"], r["lng"] = float(r["lat"]), float(r["lng"])
    cross.sort(key=lambda r: int(r["node_id"]))
    sample = random.Random(SEED).sample(cross, min(a.take, len(cross)))
    # 20 m discs (ellipses in degrees) around the sampled crossings: the source keeps panos inside this shape.
    discs, tiles = [], set()
    for r in sample:
        dlat = a.near_m / 111320.0
        dlng = a.near_m / (111320.0 * math.cos(math.radians(r["lat"])))
        discs.append(affinity.scale(Point(r["lng"], r["lat"]).buffer(1.0, 16), dlng, dlat))
        pad = 4  # tiles within 4 discs' radius of the crossing, so a pano returned by a neighbouring tile is not lost
        for sy in (-pad, 0, pad):
            for sx in (-pad, 0, pad):
                tiles.add(tile_of(r["lat"] + sy * dlat, r["lng"] + sx * dlng))
    shape = unary_union(discs)
    panos, failed = {}, 0
    if a.tracker_census:
        tiles = set()
        panos = census_panos(a.tracker_census, sample, a.near_m)
        log(f"{len(sample)} sampled crossings; panos from the tracker census {os.path.basename(a.tracker_census)}")
    else:
        log(f"{len(sample)} sampled crossings, {len(tiles)} z14 tiles")
        with ThreadPoolExecutor(8) as ex:
            for got in ex.map(lambda t: mly.fetch_panos_for_tile(t[0], t[1], shape), sorted(tiles)):
                if got is None:
                    failed += 1
                else:
                    panos.update(got)
    if failed:
        raise SystemExit(f"{failed} tiles failed; rerun")
    thinned = mly.thin_panos(panos)
    log(f"{len(panos)} panos within {a.near_m:g} m of a sampled crossing; {len(thinned)} after 5 m thinning")
    rows, rng = [], random.Random(SEED + 1)
    with_pano = 0
    for r in sample:
        near = sorted((pid, v) for pid, v in thinned.items()
                      if abs(v[0] - r["lat"]) < 0.0005 and haversine_m(r["lat"], r["lng"], v[0], v[1]) <= a.near_m)
        with_pano += bool(near)
        for pid, v in rng.sample(near, min(a.per_crossing, len(near))):
            rows.append({"node_id": r["node_id"], "basis": r["basis"], "pano_id": pid, "tile_lat": v[0], "tile_lng": v[1],
                         "captured_at_ms": v[2], "captured_year": time.gmtime(v[2] / 1000).tm_year if v[2] else "",
                         "tile_range_m": round(haversine_m(r["lat"], r["lng"], v[0], v[1]), 1)})
    write_csv(os.path.join(out, "candidates.csv"), rows)
    summary = {"seed": SEED, "sampled_crossings": len(sample), "near_m": a.near_m, "per_crossing": a.per_crossing,
               "pano_source": os.path.basename(a.tracker_census) if a.tracker_census else "mapillary z14 tiles",
               "tiles": len(tiles), "panos_in_range": len(panos), "panos_after_thinning": len(thinned),
               "crossings_with_a_pano": with_pano, "candidate_pairs": len(rows),
               "candidate_panos": len({r["pano_id"] for r in rows})}
    with open(os.path.join(out, "candidates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- meta, store

def meta_one(mly, pid, lat, lng):
    """('ok', record) / ('skip', reason) / ('fail', reason) from one Graph metadata call."""
    meta, gone = mly.fetch_image_metadata(pid)
    if gone:
        return "skip", "gone"
    if meta is None:
        return "fail", "metadata"
    w, h = meta.get("width"), meta.get("height")
    if (meta.get("camera_type") not in ("spherical", "equirectangular") or not meta.get("captured_at")
            or not meta.get("thumb_original_url") or not w or not h
            or abs(w - 2 * h) > mly.ASPECT_TOLERANCE * w or mly._compass_angle(meta) is None):
        return "skip", "not a usable 360 pano"
    return "ok", mly.build_pano_record(pid, lat, lng, meta)


def city_keep(a):
    return os.path.join(a.keep, "osm285", a.city)


def read_records(a):
    path = os.path.join(city_keep(a), "records.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["pano"] for line in f if line.strip()]


def cmd_meta(a):
    """SfM position, compass and camera for each candidate pano (the placer's inputs); no image bytes."""
    mly = labeler_mapillary(a.labeler)
    os.makedirs(city_keep(a), exist_ok=True)
    done = {p["panorama_id"] for p in read_records(a)}
    todo = {}
    for r in read_csv(os.path.join(OUT, a.city, "candidates.csv")):
        if r["pano_id"] not in done:
            todo[r["pano_id"]] = (float(r["tile_lat"]), float(r["tile_lng"]))
    log(f"{len(done)} already have a record, {len(todo)} to ask")
    counts = defaultdict(int)
    with open(os.path.join(city_keep(a), "records.jsonl"), "a", encoding="utf-8") as rec, \
            open(os.path.join(city_keep(a), "fetch_skips.txt"), "a", encoding="utf-8") as sk, \
            ThreadPoolExecutor(a.workers) as ex:
        jobs = [(pid, ex.submit(meta_one, mly, pid, lat, lng)) for pid, (lat, lng) in sorted(todo.items())]
        for i, (pid, job) in enumerate(jobs):
            status, val = job.result()
            counts[status] += 1
            if status == "ok":
                rec.write(json.dumps({"pano": val}) + "\n")
                rec.flush()
            else:
                sk.write(f"{pid}\t{status}\t{val}\n")
                sk.flush()
            if i % 200 == 0:
                log(f"  {i}/{len(jobs)} {dict(counts)}")
    print(json.dumps({"already": len(done), **counts, "pano_list": write_pano_list(a, read_records(a))}))


def write_pano_list(a, recs):
    """pano_list.csv: the list sidewalk-panorama-tools' `DownloadRunner.py -c` reads, wherever it is run.
    has_labels is true on every row because DownloadRunner downloads only labelled panos without --all-panos."""
    lst = os.path.join(OUT, a.city, "pano_list.csv")
    write_csv(lst, [{"pano_id": p["panorama_id"], "source": "mapillary", "width": p["width"], "height": p["height"],
                     "lat": p["lat"], "lng": p["lng"], "camera_heading": p["camera_heading"],
                     "camera_pitch": p["camera_pitch"] if p.get("camera_pitch") is not None else "",
                     "has_labels": "true"} for p in sorted(recs, key=lambda p: p["panorama_id"])])
    return lst


def store_dir(a):
    """The city's folder in a Project Sidewalk-layout pano store: <id[:2]>/<id>.jpg plus pano-tools' ledgers."""
    return os.path.join(a.keep, "pano_store", a.city)


def cmd_store(a):
    """Image bytes through sidewalk-panorama-tools' DownloadRunner, into a store a deployment could reuse.

    pano-tools reads the pano list from a CSV (-c) instead of a server, downloads Mapillary originals one at a
    time with its own retry and back-off, skips files already on disk and ledgers every permanent outcome.
    """
    import subprocess
    from dotenv import dotenv_values
    recs = read_records(a)
    lst = write_pano_list(a, recs)
    env = dict(os.environ)
    token = (env.get("MAPILLARY_ACCESS_TOKEN")
             or dotenv_values(os.path.join(a.labeler, ".env")).get("MAPILLARY_ACCESS_TOKEN"))
    if not token:
        sys.exit("MAPILLARY_ACCESS_TOKEN is in neither the environment nor the labeler's .env")
    env["MAPILLARY_ACCESS_TOKEN"] = token
    os.makedirs(store_dir(a), exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.pano_tools, capture_output=True, text=True).stdout.strip()
    # The first positional is the server to ask for a pano list; with -c it is never contacted.
    cmd = [a.python, "DownloadRunner.py", "unused.invalid", os.path.abspath(store_dir(a)), "-c", os.path.abspath(lst),
           "--skip-depth"]                      # Mapillary serves no depth
    log(f"{len(recs)} panos listed; pano-tools {rev[:12]} -> {store_dir(a)}")
    t0 = time.time()
    code = subprocess.run(cmd, cwd=a.pano_tools, env=env).returncode
    have = sum(os.path.exists(os.path.join(store_dir(a), p["panorama_id"][:2], p["panorama_id"] + ".jpg")) for p in recs)
    summary = {"listed": len(recs), "in_store": have, "pano_tools_commit": rev, "exit_code": code,
               "seconds": round(time.time() - t0)}
    with open(os.path.join(OUT, a.city, "store_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crossings")
    c.add_argument("--city", required=True)
    c.add_argument("--wikidata", required=True, help="the city boundary relation's Wikidata id, e.g. Q18013")
    c.add_argument("--refresh", action="store_true")
    p = sub.add_parser("panos")
    p.add_argument("--city", required=True)
    p.add_argument("--labeler", required=True, help="sidewalk-auto-labeler checkout (sources/, .env)")
    p.add_argument("--take", type=int, default=300, help="marked at-grade crossings to sample")
    p.add_argument("--near-m", type=float, default=20.0)
    p.add_argument("--per-crossing", type=int, default=3)
    p.add_argument("--tracker-census", default=None,
                   help="a streetscape-tracker Mapillary run CSV (<city>_..._mapillary_<date>.csv.gz); read the "
                        "pano list from it instead of requesting tiles")
    f = sub.add_parser("meta")
    f.add_argument("--city", required=True)
    f.add_argument("--labeler", required=True)
    f.add_argument("--keep", required=True)
    f.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("store")
    f.add_argument("--city", required=True)
    f.add_argument("--labeler", required=True, help="its .env supplies the Mapillary token when the environment has none")
    f.add_argument("--keep", required=True)
    f.add_argument("--pano-tools", required=True, help="sidewalk-panorama-tools checkout")
    f.add_argument("--python", required=True, help="a python with pano-tools' requirements")
    a = ap.parse_args()
    {"crossings": cmd_crossings, "panos": cmd_panos, "meta": cmd_meta, "store": cmd_store}[a.cmd](a)


if __name__ == "__main__":
    main()
