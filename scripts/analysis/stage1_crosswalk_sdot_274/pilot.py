"""Stage 1 for crosswalks from Seattle's SDOT inventory: a 200-crosswalk pilot (RampNet#274).

RampNet's Stage 1 turned government curb-ramp inventories into auto-labelled GSV panos. This is the crosswalk
analogue on one city: SDOT's "Marked Crosswalks" layer (GT-grade inside its coverage under the amended rule,
`scripts/analysis/crosswalk_inventory_274/sdot_inventory.py` on branch crosswalk-inventory-274) gives the
locations, GSV gives the pixels, and two placers put each crosswalk into the pano:

  B1  the SDOT point itself, projected: bearing exactly as `download_dataset.py` computes it
      (`pyproj.Geod.inv` minus `heading_to_azimuth(get_pano_heading)`), dip = atan(CAMERA_HEIGHT_M / range).
  B2  B1 snapped to the Mapillary Vistas crosswalk component (classes 8 + 23, the Phase 0 probe's checkpoint and
      code, `scripts/analysis/crosswalk_phase0_274/vistas_crosswalk_probe.py`) whose nearest pixel lies within
      SNAP_R of B1 in RampNet's matcher metric; the label point is that component's centroid.

Subcommands (all write under --out; only `meta` and `fetch` touch the network, GET only):
  sample   seeded stratified draw of the pilot crosswalks -> sample.csv
  meta     GSV pano search around each crosswalk, inclusion + install-date + clean-pano rules -> dataset.jsonl
  fetch    pixels for the clean panos -> pixels/<pano>.jpg + .json (untracked), resumable
  place    B1 + B2 per (pano, crosswalk) on the GPU -> placements.csv, views/ thumbnails (untracked)
  report   report.md with the pre-registered reading (RULE); `figures` -> docs/figures/stage1_crosswalk_sdot_274/
  gallery  gallery.html: a local contact sheet of GALLERY_N seeded placements (never an Artifact)

  python pilot.py sample --inventory <crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274 --out OUT
  python pilot.py meta   --inventory ... --census <Phase 0 census OUT> --out OUT
  python pilot.py fetch  --out OUT
  python pilot.py place  --out OUT
  python pilot.py report --out OUT && python pilot.py figures --out OUT && python pilot.py gallery --out OUT

`--limit N` on meta/fetch/place smoke-tests the first N items. Phase 1 GT nodes can be excluded with
`--phase1-sheet` on `sample` (the 2026-11-03 sheet has no Seattle rows, so it excludes nothing today; the check
stays so a later sheet cannot leak into the training pool). CurbRamp Stage 1 scripts are not touched.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STAGE1 = os.path.join(REPO, "stage_one", "dataset_generation")
for p in (REPO, STAGE1, os.path.join(REPO, "scripts", "model_comparison"),
          os.path.join(REPO, "scripts", "analysis", "crosswalk_phase0_274")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pyproj import Geod  # noqa: E402
import sdot_source  # noqa: E402

GEOD = Geod(ellps="WGS84")
SEED = 274
# Strata (drawn in this order, without replacement; mid-block first so it is not starved).
STRATA = (("midblock", 20, lambda r: r["midblock"] == "Y"),
          ("BRICK/CUSTOM/OTHER", 30, lambda r: r["marking"] in ("BRICK", "CUSTOM", "OTHER")),
          ("PARSL", 60, lambda r: r["marking"] == "PARSL"),
          ("LADER", 90, lambda r: r["marking"] == "LADER"))
SEARCH_OFFSET_M = 15.0      # search at the point and 15 m N/E/S/W of it (one search returns one location's panos)
DISCOVERY_M = 20.0          # run value; RampNet's ramp value (10 m) is reported beside it
DISCOVERY_RAMPNET_M = 10.0
INCLUSION_M = 35.0          # RampNet's INCLUSION_DISTANCE_THRESHOLD
CLEAN_MATCH_M = 10.0        # an OSM marked node with no current SDOT point within this = paint SDOT doesn't record
HOLE_M = 100.0              # sdot_inventory's coverage mask
CAMERA_HEIGHT_M = 2.5       # GSV pre-2025 rig (labeler #79); the 2025-26 rig is ~2.0 m, reported by capture year
TREAT_UNDATED_AS_PREDATING = True
SNAP_R = 0.05               # B2 snap radius, pano-width units (Phase 0 probe's MATCH_R)
TOL_R = 0.022               # RampNet's matcher radius, reported beside the exact-pixel reading
FETCH_PER_CROSSWALK = 6     # pixel arms run on <= 6 clean panos per sampled crosswalk, spread over capture dates
VIEW_PX = 1024
GALLERY_N = 60
CURRENT = sdot_source.CURRENT
OSM_MARKED = {"marked", "zebra", "lines", "ladder", "dashes", "dots", "surface", "pedestrian_signals"}
OSM_UNMARKED = {"unmarked", "no", "informal"}

# THE READING, fixed 2026-11-03 before `report` first ran (plan approved by Jon the same day):
RULE = {
    "proceed": "B1 lands on a Vistas crosswalk pixel (exact pixel) for >= 0.70 of included points within 20 m",
    "proceed_min": 0.70, "proceed_range_m": 20.0,
    "b2_default": "where a component exists within SNAP_R, the B1->B2 shift is <= 0.05 pano width in >= 0.80",
    "b2_min": 0.80, "b2_shift_max": 0.05,
    "otherwise": "report and stop for a look (demote, don't halt)",
    "fixed": "2026-11-03",
}
# SECONDARY reading, added 2026-11-03 while the pilot ran and BEFORE `report` first ran (prompted by Jon:
# street-level, aerial and inventory dates do not line up). SDOT is a 2026 snapshot with no history, and ~94% of
# its points are undated, so a pano captured before a crosswalk was painted gets a label on bare asphalt. The
# primary verdict above stays over all capture years. This reading repeats the proceed test on captures from
# RECENT_YEAR on, and the per-year curve (section 2b) is what a recency window for the city-wide run is chosen from.
RECENT_YEAR = 2021

SAMPLE_CSV, SEARCH_CACHE, DATASET, PLACEMENTS = "sample.csv", "search_cache.jsonl", "dataset.jsonl", "placements.csv"


# ---------------------------------------------------------------- helpers

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def offset(lat, lng, north_m, east_m):
    return lat + north_m / 111320.0, lng + east_m / (111320.0 * math.cos(math.radians(lat)))


class Grid:
    """Within-radius lookups over (lat, lng, item) by ~50 m buckets."""

    def __init__(self, items, cell_m=50.0, lat0=47.6):
        self.dlat = cell_m / 111320.0
        self.dlng = cell_m / (111320.0 * math.cos(math.radians(lat0)))
        self.cell_m, self.b = cell_m, {}
        for it in items:
            self.b.setdefault((int(it[0] // self.dlat), int(it[1] // self.dlng)), []).append(it)

    def within(self, lat, lng, r):
        ci, cj = int(lat // self.dlat), int(lng // self.dlng)
        k = int(r // self.cell_m) + 1
        out = []
        for i in range(ci - k, ci + k + 1):
            for j in range(cj - k, cj + k + 1):
                for it in self.b.get((i, j), ()):
                    d = haversine_m(lat, lng, it[0], it[1])
                    if d <= r:
                        out.append((d, it))
        return sorted(out, key=lambda t: t[0])


def point_in_ring(lat, lng, ring):
    inside, n = False, len(ring)
    for k in range(n):
        x1, y1 = ring[k][0], ring[k][1]
        x2, y2 = ring[(k + 1) % n][0], ring[(k + 1) % n][1]
        if (y1 > lat) != (y2 > lat) and lng < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def regions_contains(gj):
    polys = []
    for f in gj["features"]:
        g = f["geometry"]
        for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            xs, ys = [c[0] for c in p[0]], [c[1] for c in p[0]]
            polys.append(((min(ys), max(ys), min(xs), max(xs)), p))

    def contains(lat, lng):
        return any(y0 <= lat <= y1 and x0 <= lng <= x1 and point_in_ring(lat, lng, p[0])
                   and not any(point_in_ring(lat, lng, h) for h in p[1:])
                   for (y0, y1, x0, x1), p in polys)
    return contains


def load_inventory(inv_dir):
    """SDOT features + region test from the inventory branch's pull, hash-checked against its inputs.json."""
    rec = json.load(open(os.path.join(inv_dir, "inputs.json"), encoding="utf-8"))
    for k in ("sdot", "regions"):
        got = sha256(os.path.join(inv_dir, rec[k]["file"]))
        assert got == rec[k]["sha256"], f"{k} changed since fetch ({got[:12]} != {rec[k]['sha256'][:12]})"
    sd = json.load(open(os.path.join(inv_dir, rec["sdot"]["file"]), encoding="utf-8"))
    regions = regions_contains(json.load(open(os.path.join(inv_dir, rec["regions"]["file"]), encoding="utf-8")))
    return rec, sd["features"], regions


def all_sdot_points(features):
    """(lat, lng, {objectid, status}) for every SDOT point of any status."""
    out = []
    for f in features:
        g, p = f.get("geometry"), f["properties"]
        if g and g.get("coordinates"):
            out.append((g["coordinates"][1], g["coordinates"][0],
                        {"objectid": p["OBJECTID"], "status": (p.get("CURRENT_STATUS") or "").strip()}))
    return out


def osm_marked_nodes(census):
    """(lat, lng, node_id) of Seattle OSM crossing nodes that a map would call painted (marked or signalised)."""
    import glob
    out = {}
    for fp in sorted(glob.glob(os.path.join(census, "osm", "seattle-wa", "Crosswalk", "*.json"))):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            t = e.get("tags") or {}
            if e.get("type") != "node" or t.get("highway") != "crossing":
                continue
            m, c = t.get("crossing:markings"), t.get("crossing")
            cls = ("unmarked" if m == "no" else "marked") if m is not None else \
                ("marked" if c in OSM_MARKED else "unmarked" if c in OSM_UNMARKED else "unknown")
            sig = c == "traffic_signals" or t.get("crossing:signals") == "yes"
            if cls == "marked" or (cls == "unknown" and sig):
                out[e["id"]] = (e["lat"], e["lon"], e["id"])
    return list(out.values())


def project(pano_lat, pano_lng, heading_deg, lat, lng, camera_h=None):
    """B1: SDOT point -> (x, y) pano-normalized, plus range (m) and the bearing relative to the pano centre.

    The bearing is `download_dataset.py`'s exactly: Geod.inv forward azimuth minus heading_to_azimuth(heading).
    Pano centre column = the pano heading (x = 0.5), x grows clockwise; y = 0.5 at the horizon, dip below it.
    """
    from rampnet.gsv import heading_to_azimuth
    h = CAMERA_HEIGHT_M if camera_h is None else camera_h
    az, _, d = GEOD.inv(pano_lng, pano_lat, lng, lat)
    rel = az - heading_to_azimuth(heading_deg)
    x = (0.5 + rel / 360.0) % 1.0
    y = 0.5 + math.degrees(math.atan2(h, d)) / 180.0
    return x, y, d, rel


def pano_dist(ax, ay, bx, by):
    """RampNet matcher distance in pano-width units (x wraps at the seam; y counts half)."""
    dx = abs(ax - bx)
    dx = min(dx, 1.0 - dx) * 1024.0
    return math.hypot(dx, (ay - by) * 512.0) / 1024.0


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def frac(k, n):
    if not n:
        return "–"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.2f}, {hi:.2f}]"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


# ---------------------------------------------------------------- sample

def cmd_sample(a):
    rec, feats, regions = load_inventory(a.inventory)
    rows = sdot_source.sdot_locations(feats, keep=regions)
    excl = set()
    if a.phase1_sheet:
        sheet = list(csv.DictReader(open(a.phase1_sheet, encoding="utf-8", newline="")))
        nodes = [(float(r["node_lat"]), float(r["node_lng"]), r["gt_id"]) for r in sheet]   # positions only
        g = Grid(nodes)
        excl = {r["source_id"] for r in rows if g.within(r["latitude"], r["longitude"], INCLUSION_M)}
    pool = [r for r in rows if r["source_id"] not in excl]
    rng = random.Random(SEED)
    taken, out = set(), []
    for name, n, sel in STRATA:
        cand = [r for r in pool if sel(r) and r["source_id"] not in taken]
        pick = rng.sample(cand, min(n, len(cand)))
        for r in pick:
            taken.add(r["source_id"])
            out.append({**r, "stratum": name})
        log(f"{name}: {len(pick)} of {len(cand)}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, SAMPLE_CSV), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    meta = {"inventory_inputs": {k: {kk: rec[k][kk] for kk in ("url", "sha256", "fetched_utc")}
                                 for k in ("sdot", "regions")},
            "seed": SEED, "strata": [(s, n) for s, n, _ in STRATA], "pool": len(rows),
            "phase1_excluded": len(excl), "phase1_sheet": bool(a.phase1_sheet), "sampled": len(out)}
    json.dump(meta, open(os.path.join(a.out, "sample_meta.json"), "w", encoding="utf-8"), indent=1)
    log(f"{len(out)} sampled from {len(rows)} current SDOT crosswalks ({len(excl)} excluded as Phase 1) -> {SAMPLE_CSV}")


# ---------------------------------------------------------------- meta

def tile_xy(lat, lng, z=17):
    n = 2 ** z
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def cmd_meta(a):
    # Google decommissioned GeoPhotoService.SingleImageSearch (it answers "decommissioned and turned down",
    # checked 2026-11-03), which stage_one's search_panos.search_panoramas calls, so discovery here uses what the
    # labeler uses: streetlevel's z17 coverage tiles (current panos), then find_panorama_by_id for each one's
    # historical captures and for every historical capture's own position, heading and date.
    from streetlevel import streetview as sv
    _, feats, regions = load_inventory(a.inventory)
    sample = list(csv.DictReader(open(os.path.join(a.out, SAMPLE_CSV), encoding="utf-8", newline="")))
    if a.limit:
        sample = sample[:a.limit]
    allp = all_sdot_points(feats)
    cur = [p for p in allp if p[2]["status"] in CURRENT]
    g_cur, g_all = Grid(cur), Grid(allp)
    g_planned = Grid([p for p in allp if p[2]["status"] not in CURRENT])
    dates = {}
    for f in feats:
        dates[f["properties"]["OBJECTID"]] = sdot_source.install_month(f["properties"].get("INSTALL_DATE"))
    g_osm = Grid(osm_marked_nodes(a.census))

    # 1. coverage tiles around each crosswalk (the point and SEARCH_OFFSET_M N/E/S/W, so a crosswalk near a tile
    #    edge still sees the neighbour tile), cached per tile; then metadata per pano, cached per pano id.
    #    Both caches are the network record and make `meta` resumable.
    cache_p = os.path.join(a.out, SEARCH_CACHE)
    tiles, meta = {}, {}
    for r in read_jsonl(cache_p):
        if r["kind"] == "tile":
            tiles[tuple(r["xy"])] = r["panos"]
        else:
            meta[r["pano_id"]] = r
    lock = threading.Lock()

    def append(rec):
        with lock:
            with open(cache_p, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")

    pts = {}
    for r in sample:
        lat, lng = float(r["latitude"]), float(r["longitude"])
        pts[r["source_id"]] = (lat, lng)
        for dn, de in ((0, 0), (SEARCH_OFFSET_M, 0), (-SEARCH_OFFSET_M, 0), (0, SEARCH_OFFSET_M), (0, -SEARCH_OFFSET_M)):
            tiles.setdefault(tile_xy(*offset(lat, lng, dn, de)), None)

    def get_tile(xy):
        try:
            ps = [{"pano_id": p.id, "lat": p.lat, "lon": p.lon} for p in sv.get_coverage_tile(*xy)]
        except Exception as e:  # noqa: BLE001 -- left uncached so a re-run retries it
            log(f"tile {xy} failed: {e}")
            return
        tiles[xy] = ps
        append({"kind": "tile", "xy": list(xy), "panos": ps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    todo = [xy for xy, v in tiles.items() if v is None]
    log(f"{len(todo)} coverage tiles to fetch")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(get_tile, todo))

    def get_meta(pid):
        try:
            p = sv.find_panorama_by_id(pid)
        except Exception as e:  # noqa: BLE001
            log(f"metadata {pid} failed: {e}")
            return
        rec = {"kind": "pano", "pano_id": pid, "found": p is not None}
        if p is not None:
            rec.update({"lat": p.lat, "lon": p.lon, "heading_deg": math.degrees(p.heading) % 360.0,
                        "date": f"{p.date.year}-{p.date.month:02d}" if p.date else None,
                        "source": getattr(p, "source", None),
                        "historical": [h.id for h in (p.historical or [])]})
        meta[pid] = rec
        append(rec)

    def near(lat, lng, r):
        return [s for s, (la, ln) in pts.items() if haversine_m(la, ln, lat, lng) <= r]

    current = {p["pano_id"]: p for ps in tiles.values() if ps for p in ps if near(p["lat"], p["lon"], DISCOVERY_M)}
    todo = [pid for pid in current if pid not in meta]
    log(f"{len(current)} current panos within {DISCOVERY_M:.0f} m; metadata for {len(todo)}")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(get_meta, todo))
    hist = {h for pid in current for h in meta.get(pid, {}).get("historical", [])} - set(meta)
    log(f"metadata for {len(hist)} historical captures")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(get_meta, sorted(hist)))

    # 2. discovery: every pano (current or historical) whose OWN position is within DISCOVERY_M of a crosswalk.
    #    Indoor / user-uploaded sources are skipped, as the labeler skips them.
    panos, disc = {}, []
    ids = set(current) | {h for pid in current for h in meta.get(pid, {}).get("historical", [])}
    for pid in sorted(ids):
        m = meta.get(pid)
        if not m or not m.get("found"):
            continue
        if m.get("source") and str(m["source"]) not in ("launch", "scout", "None"):
            continue
        for s in near(m["lat"], m["lon"], 50.0):
            d = haversine_m(pts[s][0], pts[s][1], m["lat"], m["lon"])
            disc.append({"source_id": s, "pano_id": pid, "d_m": round(d, 2), "date": m["date"]})
            if d <= DISCOVERY_M:
                panos[pid] = m
    with open(os.path.join(a.out, "discovery.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "pano_id", "d_m", "date"], lineterminator="\n")
        w.writeheader()
        w.writerows(disc)

    # 3. inclusion + install-date + clean rules per discovered pano
    out = []
    for pid, p in sorted(panos.items()):
        inc = g_cur.within(p["lat"], p["lon"], INCLUSION_M)
        reasons = []
        pano_ym = tuple(int(v) for v in (p.get("date") or "0-0").split("-")[:2])
        for d, it in inc:
            dt = dates.get(it[2]["objectid"], "")
            if not dt:
                if not TREAT_UNDATED_AS_PREDATING:
                    reasons.append("undated")
                continue
            if tuple(int(v) for v in dt.split("-")) >= pano_ym:
                reasons.append(f"installed_after_capture:sdot:{it[2]['objectid']}")
        for d, it in g_osm.within(p["lat"], p["lon"], INCLUSION_M):
            if not g_cur.within(it[0], it[1], CLEAN_MATCH_M):
                reasons.append(f"osm_paint_not_in_sdot:osm:{it[2]}")
        for d, it in g_planned.within(p["lat"], p["lon"], INCLUSION_M):
            reasons.append(f"sdot_{it[2]['status'] or 'empty'}_nearby:sdot:{it[2]['objectid']}")
        if not g_all.within(p["lat"], p["lon"], HOLE_M) or not regions(p["lat"], p["lon"]):
            reasons.append("outside_coverage")
        out.append({"pano_id": pid, "pano_coords": [p["lat"], p["lon"]], "pano_date": p.get("date"),
                    "streetlevel_heading_deg": p.get("heading_deg"),
                    "label_type": sdot_source.LABEL_TYPE,
                    "crosswalk_coords": [[it[0], it[1]] for _, it in inc],
                    "sdot_ids": [f"sdot:{it[2]['objectid']}" for _, it in inc],
                    "clean": not reasons, "dirty_reasons": reasons})
    with open(os.path.join(a.out, DATASET), "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    log(f"{len(panos)} panos within {DISCOVERY_M:.0f} m of {len(sample)} crosswalks; "
        f"{sum(r['clean'] for r in out)} clean -> {DATASET}")


# ---------------------------------------------------------------- fetch

def cmd_fetch(a):
    import cv2
    from rampnet.gsv import fetch_panorama
    from search_panos import get_pano_heading
    pix = os.path.join(a.out, "pixels")
    os.makedirs(pix, exist_ok=True)
    # Historical captures make the yield ~40 panos per crosswalk, so the pilot fetches at most FETCH_PER_CROSSWALK
    # clean panos per sampled crosswalk, spread evenly over capture dates (deterministic). Yield is still reported
    # in full from the metadata; only the pixel arms run on this subset. Selection is frozen in fetch_selection.txt.
    sel_p = os.path.join(a.out, "fetch_selection.txt")
    ds = {r["pano_id"]: r for r in read_jsonl(os.path.join(a.out, DATASET)) if r["clean"]}
    if os.path.exists(sel_p):
        chosen = [l.strip() for l in open(sel_p, encoding="utf-8") if l.strip()]
    else:
        per = {}
        for r in csv.DictReader(open(os.path.join(a.out, "discovery.csv"), encoding="utf-8", newline="")):
            if float(r["d_m"]) <= DISCOVERY_M and r["pano_id"] in ds:
                per.setdefault(r["source_id"], set()).add(r["pano_id"])
        chosen = set()
        for s in sorted(per):
            c = sorted(per[s], key=lambda p: (ds[p]["pano_date"] or "", p))
            k = min(FETCH_PER_CROSSWALK, len(c))
            chosen.update(c[round(i * (len(c) - 1) / max(k - 1, 1))] for i in range(k))
        chosen = sorted(chosen)
        with open(sel_p, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(chosen) + "\n")
    todo = [ds[p] for p in chosen if p in ds and not os.path.exists(os.path.join(pix, p + ".json"))]
    if a.limit:
        todo = todo[:a.limit]
    log(f"{len(todo)} panos to fetch")
    ok = [0]
    lock = threading.Lock()

    def one(r):
        pid = r["pano_id"]
        try:
            equi = fetch_panorama(pid)
            if equi is None:
                log(f"no pixels for {pid}")
                return
            heading = get_pano_heading(pid)
            cv2.imwrite(os.path.join(pix, pid + ".jpg"), equi, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(pix, pid + ".json"), "w", encoding="utf-8") as f:   # written last = done marker
                json.dump({"pano_id": pid, "heading": heading, "shape": list(equi.shape),
                           "streetlevel_heading_deg": r.get("streetlevel_heading_deg"),
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
            with lock:
                ok[0] += 1
        except Exception as e:  # noqa: BLE001 -- retryable; a re-run picks it up
            log(f"fetch failed {pid}: {e}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i and i % 50 == 0:
                log(f"  {i}/{len(todo)}")
    log(f"fetched {ok[0]} of {len(todo)}")


# ---------------------------------------------------------------- place

def cmd_place(a):
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from skimage.measure import label as cc_label
    from equirect_tiling import View, equirect_point_to_perspective, equirect_to_perspective, \
        perspective_point_to_equirect
    from detectors import load_pano_image
    import vistas_crosswalk_probe as probe

    pix, views = os.path.join(a.out, "pixels"), os.path.join(a.out, "views")
    os.makedirs(views, exist_ok=True)
    chosen = {l.strip() for l in open(os.path.join(a.out, "fetch_selection.txt"), encoding="utf-8") if l.strip()}
    ds = [r for r in read_jsonl(os.path.join(a.out, DATASET))
          if r["clean"] and r["pano_id"] in chosen and os.path.exists(os.path.join(pix, r["pano_id"] + ".json"))]
    if a.limit:
        ds = ds[:a.limit]
    out_p = os.path.join(a.out, PLACEMENTS)
    done = set()
    if os.path.exists(out_p):
        done = {(r["pano_id"], r["sdot_id"]) for r in csv.DictReader(open(out_p, encoding="utf-8", newline=""))}
    fields = ["pano_id", "sdot_id", "pano_date", "range_m", "rel_bearing_deg", "b1_x", "b1_y", "on_paint",
              "near_px_dist", "comp_found", "b2_x", "b2_y", "b1_b2_shift", "comp_area_px", "share_plain",
              "view_pitch_deg", "segmenter_backend"]
    new = not os.path.exists(out_p)
    fh = open(out_p, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    if new:
        w.writeheader()
    seg = probe.Segmenter()
    log(f"{len(ds)} clean fetched panos; segmenter {seg.backend}, revision {seg.revision}")
    for k, r in enumerate(ds):
        todo = [(c, s) for c, s in zip(r["crosswalk_coords"], r["sdot_ids"]) if (r["pano_id"], s) not in done]
        if not todo:
            continue
        heading = json.load(open(os.path.join(pix, r["pano_id"] + ".json"), encoding="utf-8"))["heading"]
        pano = load_pano_image(os.path.join(pix, r["pano_id"] + ".jpg"), probe.SOURCE_MAX_EDGE)
        plat, plng = r["pano_coords"]
        for (lat, lng), sid in todo:
            x, y, d, rel = project(plat, plng, heading, lat, lng)
            dip = (y - 0.5) * 180.0
            view = View((x - 0.5) * 360.0, -dip, 90.0, 90.0, VIEW_PX, VIEW_PX)
            img = equirect_to_perspective(pano, view)
            sg, _ = seg.seg_prob(img)
            sel = np.isin(sg, list(probe.CLASS_IDS))
            uv = equirect_point_to_perspective(x, y, view)
            pu = min(VIEW_PX - 1, int(uv[0] * VIEW_PX))
            pv = min(VIEW_PX - 1, int(uv[1] * VIEW_PX))
            row = {"pano_id": r["pano_id"], "sdot_id": sid, "pano_date": r["pano_date"], "range_m": round(d, 2),
                   "rel_bearing_deg": round(rel, 3), "b1_x": round(x, 6), "b1_y": round(y, 6),
                   "on_paint": int(bool(sel[pv, pu])), "near_px_dist": "", "comp_found": 0, "b2_x": "",
                   "b2_y": "", "b1_b2_shift": "", "comp_area_px": "", "share_plain": "",
                   "view_pitch_deg": round(-dip, 3), "segmenter_backend": seg.backend}
            cx = cy = None
            if sel.any():
                lab = cc_label(sel, connectivity=2)
                _, (ii, jj) = ndimage.distance_transform_edt(~sel, return_indices=True)
                ni, nj = int(ii[pv, pu]), int(jj[pv, pu])
                nx, ny = perspective_point_to_equirect((nj + 0.5) / VIEW_PX, (ni + 0.5) / VIEW_PX, view)
                nd = pano_dist(x, y, nx, ny)
                row["near_px_dist"] = round(nd, 5)
                if nd <= SNAP_R:
                    comp = lab == lab[ni, nj]
                    ys, xs = np.nonzero(comp)
                    cy, cx = ys.mean(), xs.mean()
                    bx, by = perspective_point_to_equirect((cx + 0.5) / VIEW_PX, (cy + 0.5) / VIEW_PX, view)
                    row.update({"comp_found": 1, "b2_x": round(bx, 6), "b2_y": round(by, 6),
                                "b1_b2_shift": round(pano_dist(x, y, bx, by), 5), "comp_area_px": int(comp.sum()),
                                "share_plain": round(float((sg[comp] == 8).mean()), 3)})
            w.writerow(row)
            fh.flush()
            # thumbnail: mask tint, B1 red ring, B2 cyan cross
            th = Image.fromarray(np.asarray(img).copy())
            arr = np.asarray(th).astype(np.float32)
            arr[sel] = arr[sel] * 0.5 + np.array([255, 200, 0]) * 0.5
            th = Image.fromarray(arr.astype(np.uint8)).resize((512, 512))
            dr = ImageDraw.Draw(th)
            dr.ellipse([pu / 2 - 9, pv / 2 - 9, pu / 2 + 9, pv / 2 + 9], outline=(255, 0, 0), width=3)
            if cx is not None:
                dr.line([cx / 2 - 10, cy / 2, cx / 2 + 10, cy / 2], fill=(0, 255, 255), width=3)
                dr.line([cx / 2, cy / 2 - 10, cx / 2, cy / 2 + 10], fill=(0, 255, 255), width=3)
            th.save(os.path.join(views, f"{r['pano_id']}__{sid.replace(':', '_')}.jpg"), quality=85)
        if k % 25 == 0:
            log(f"  pano {k + 1}/{len(ds)}")
    fh.close()
    log(f"placements -> {PLACEMENTS}")


# ---------------------------------------------------------------- report

def load_placements(out):
    rows = list(csv.DictReader(open(os.path.join(out, PLACEMENTS), encoding="utf-8", newline="")))
    for r in rows:
        r["range_m"] = float(r["range_m"])
        r["on_paint"] = int(r["on_paint"])
        r["comp_found"] = int(r["comp_found"])
        r["near"] = float(r["near_px_dist"]) if r["near_px_dist"] else None
        r["shift"] = float(r["b1_b2_shift"]) if r["b1_b2_shift"] else None
        r["year"] = int(r["pano_date"].split("-")[0]) if r["pano_date"] else None
    return rows


def cmd_report(a):
    sample = list(csv.DictReader(open(os.path.join(a.out, SAMPLE_CSV), encoding="utf-8", newline="")))
    smeta = json.load(open(os.path.join(a.out, "sample_meta.json"), encoding="utf-8"))
    ds = read_jsonl(os.path.join(a.out, DATASET))
    disc = list(csv.DictReader(open(os.path.join(a.out, "discovery.csv"), encoding="utf-8", newline="")))
    pl = load_placements(a.out)
    strat = {r["source_id"]: r for r in sample}
    clean = [r for r in ds if r["clean"]]
    L = []
    w = L.append
    w("# Stage 1 crosswalk pilot from SDOT, Seattle (RampNet#274)\n")
    w(f"Generated by `scripts/analysis/stage1_crosswalk_sdot_274/pilot.py report`. SDOT pull "
      f"{smeta['inventory_inputs']['sdot']['fetched_utc']} (sha256 `{smeta['inventory_inputs']['sdot']['sha256'][:16]}…`); "
      f"{smeta['sampled']} crosswalks drawn (seed {SEED}) from {smeta['pool']} current SDOT crosswalks in the "
      f"Project Sidewalk Seattle regions; {smeta['phase1_excluded']} excluded as Phase 1 GT nodes.\n")
    w("**Pre-registered reading** (fixed " + RULE["fixed"] + "): proceed to a city-wide run if " + RULE["proceed"]
      + "; B2 becomes the default placer if " + RULE["b2_default"] + "; otherwise " + RULE["otherwise"] + ".\n")

    w("## 1. Yield: panos per crosswalk\n")
    by = {}
    for r in disc:
        by.setdefault(r["source_id"], []).append(float(r["d_m"]))
    for R in (DISCOVERY_RAMPNET_M, DISCOVERY_M):
        ns = [sum(d <= R for d in by.get(s, [])) for s in strat]
        w(f"- within {R:.0f} m: median {pct(ns, .5):.0f} panos per crosswalk (p25 {pct(ns, .25):.0f}, p75 "
          f"{pct(ns, .75):.0f}); {sum(n == 0 for n in ns)} crosswalks with none")
    w(f"- {len(ds)} unique panos within {DISCOVERY_M:.0f} m; {len(clean)} pass the clean rule "
      f"({len(clean) / max(len(ds), 1):.3f}); {sum(len(r['sdot_ids']) for r in clean)} (pano, crosswalk) pairs "
      f"within {INCLUSION_M:.0f} m on clean panos; {len(pl)} placed.\n")
    reasons = {}
    for r in ds:
        for k in {x.split(":")[0] for x in r["dirty_reasons"]}:
            reasons[k] = reasons.get(k, 0) + 1
    w("Why panos were dropped (a pano can have several reasons):\n")
    w("| reason | panos |\n|---|---|")
    for k, n in sorted(reasons.items(), key=lambda t: -t[1]):
        w(f"| `{k}` | {n} |")
    years = {}
    for r in clean:
        y = (r["pano_date"] or "?").split("-")[0]
        years[y] = years.get(y, 0) + 1
    w("\nClean panos by capture year: " + ", ".join(f"{y} {n}" for y, n in sorted(years.items())) + "\n")

    w("## 2. Does B1 land on paint? (Vistas classes 8 + 23 at the projected pixel)\n")
    w("| range | placements | exact pixel on paint | paint within 0.022 width | component within 0.05 (B2 exists) |")
    w("|---|---|---|---|---|")
    for lo, hi in ((0, 10), (10, 20), (20, 35), (0, 20), (0, 35)):
        s = [r for r in pl if lo <= r["range_m"] < hi or (hi == 35 and r["range_m"] == 35)]
        w(f"| {lo}–{hi} m | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} | "
          f"{frac(sum(1 for r in s if r['near'] is not None and r['near'] <= TOL_R), len(s))} | "
          f"{frac(sum(r['comp_found'] for r in s), len(s))} |")
    near20 = [r for r in pl if r["range_m"] <= RULE["proceed_range_m"]]
    k = sum(r["on_paint"] for r in near20)
    w("")
    w("By the sampled crosswalk's stratum and by capture year (within 20 m):\n")
    w("| group | placements | exact pixel on paint |\n|---|---|---|")
    sid_stratum = {s: strat[s]["stratum"] for s in strat}
    for name, _, _ in STRATA:
        s = [r for r in near20 if sid_stratum.get(r["sdot_id"]) == name]
        w(f"| {name} (sampled crosswalk) | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} |")
    s = [r for r in near20 if r["sdot_id"] not in sid_stratum]
    w(f"| other SDOT crosswalks in the same panos | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} |")
    for yb, sel in (("capture ≤ 2015", lambda y: y is not None and y <= 2015),
                    ("2016–2020", lambda y: y is not None and 2016 <= y <= 2020),
                    ("2021–2024", lambda y: y is not None and 2021 <= y <= 2024),
                    ("2025+", lambda y: y is not None and y >= 2025)):
        s = [r for r in near20 if sel(r["year"])]
        w(f"| {yb} | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} |")
    w("")

    w("## 3. B2: what Vistas snapping does\n")
    comp = [r for r in pl if r["comp_found"]]
    sh = [r["shift"] for r in comp]
    k2 = sum(1 for x in sh if x <= RULE["b2_shift_max"])
    w(f"{len(comp)} of {len(pl)} placements have a Vistas component within {SNAP_R} of B1. B1→B2 shift "
      f"(pano-width units): median {pct(sh, .5):.4f}, p75 {pct(sh, .75):.4f}, p90 {pct(sh, .9):.4f}; "
      f"≤ {RULE['b2_shift_max']}: {frac(k2, len(sh))}.\n")
    w("## 4. Verdict\n")
    p1 = k / len(near20) if near20 else float("nan")
    p2 = k2 / len(sh) if sh else float("nan")
    w(f"- **Proceed to city-wide: {'YES' if p1 >= RULE['proceed_min'] else 'NO'}** — exact-pixel hit within "
      f"{RULE['proceed_range_m']:.0f} m is {frac(k, len(near20))} against {RULE['proceed_min']}.")
    w(f"- **B2 as default placer: {'YES' if p2 >= RULE['b2_min'] else 'NO'}** — shift ≤ {RULE['b2_shift_max']} "
      f"in {frac(k2, len(sh))} against {RULE['b2_min']}.\n")
    rec = [r for r in near20 if r["year"] is not None and r["year"] >= RECENT_YEAR]
    kr = sum(r["on_paint"] for r in rec)
    w(f"- *Secondary (pre-declared, not the verdict): captures from {RECENT_YEAR} on* — exact-pixel hit within "
      f"{RULE['proceed_range_m']:.0f} m is {frac(kr, len(rec))}; "
      f"{'would pass' if rec and kr / len(rec) >= RULE['proceed_min'] else 'would not pass'} the same bar.\n")
    w("### 4b. Temporal alignment, by capture year (within 20 m)\n")
    w("SDOT is a 2026 snapshot with ~94% undated points, so an old capture can predate the paint (a label on bare "
      "asphalt) or show paint since removed. A falling curve toward old years is that misalignment.\n")
    w("| capture year | placements | exact pixel on paint | Vistas component within 0.05 |\n|---|---|---|---|")
    for y in sorted({r["year"] for r in near20 if r["year"] is not None}):
        s = [r for r in near20 if r["year"] == y]
        w(f"| {y} | {len(s)} | {frac(sum(r['on_paint'] for r in s), len(s))} | "
          f"{frac(sum(r['comp_found'] for r in s), len(s))} |")
    w("")
    w("Caveat: \"on paint\" is Vistas' opinion, the Phase 0 extent filter (0.82 recall on crowd labels), not a "
      "human reading; a Vistas miss counts against B1 here. `gallery.html` shows the marks for an eyeball.\n")
    w("## Replication\n")
    w("```bash\nP=scripts/analysis/stage1_crosswalk_sdot_274/pilot.py; OUT=analysis_out/stage1_crosswalk_sdot_274\n"
      "python $P sample --inventory <crosswalk-inventory-274>/analysis_out/crosswalk_inventory_274 --out $OUT\n"
      "python $P meta --inventory <same> --census <Phase 0 census OUT> --out $OUT   # network: GSV search\n"
      "python $P fetch --out $OUT                                                   # network: GSV pixels\n"
      "python $P place --out $OUT                                                   # GPU: Vistas\n"
      "python $P report --out $OUT && python $P figures --out $OUT && python $P gallery --out $OUT\n```\n")
    w("`search_cache.jsonl` is the network record (re-running `meta` reuses it; delete it to re-search). GSV "
      "coverage changes, so a fresh search moves the yield numbers.\n")
    w("| number | lives in |\n|---|---|")
    w("| sample, strata | `sample.csv`, `sample_meta.json` |")
    w("| yield | `discovery.csv` (one row per crosswalk × pano seen) |")
    w("| clean rule, dropped reasons | `dataset.jsonl` (`clean`, `dirty_reasons`) |")
    w("| on-paint, B2, shifts | `placements.csv` (one row per pano × crosswalk) |")
    w("| figures | `docs/figures/stage1_crosswalk_sdot_274/` from `pilot.py figures` |")
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log(f"wrote {a.out}/report.md")


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(REPO, "docs", "figures", "stage1_crosswalk_sdot_274")
    os.makedirs(figdir, exist_ok=True)
    sample = list(csv.DictReader(open(os.path.join(a.out, SAMPLE_CSV), encoding="utf-8", newline="")))
    disc = list(csv.DictReader(open(os.path.join(a.out, "discovery.csv"), encoding="utf-8", newline="")))
    by = {}
    for r in disc:
        if float(r["d_m"]) <= DISCOVERY_M:
            by[r["source_id"]] = by.get(r["source_id"], 0) + 1
    ns = [by.get(r["source_id"], 0) for r in sample]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(ns, bins=range(0, max(ns) + 2), color="#3b6ea8", edgecolor="white")
    ax.set_xlabel(f"GSV panos within {DISCOVERY_M:.0f} m of the SDOT point")
    ax.set_ylabel("crosswalks")
    ax.set_title("Yield: panos per sampled crosswalk")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "yield_hist.png"), dpi=150)
    pl = load_placements(a.out)
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 35.01)]
    xs, on, cf = [], [], []
    for lo, hi in bins:
        s = [r for r in pl if lo <= r["range_m"] < hi]
        xs.append(f"{lo}–{int(hi)}")
        on.append(sum(r["on_paint"] for r in s) / len(s) if s else float("nan"))
        cf.append(sum(r["comp_found"] for r in s) / len(s) if s else float("nan"))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, on, marker="o", color="#3b6ea8", label="B1 exact pixel on Vistas paint")
    ax.plot(xs, cf, marker="s", color="#c8742c", label="Vistas component within 0.05 (B2 exists)")
    ax.axhline(RULE["proceed_min"], color="#888", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("range from pano to SDOT point (m)")
    ax.set_ylabel("share of placements")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "range_hit_rate.png"), dpi=150)
    log(f"figures -> {figdir}")


def cmd_gallery(a):
    import html
    pl = load_placements(a.out)
    pl = [r for r in pl if os.path.exists(os.path.join(a.out, "views",
                                                       f"{r['pano_id']}__{r['sdot_id'].replace(':', '_')}.jpg"))]
    pick = random.Random(SEED).sample(pl, min(GALLERY_N, len(pl)))
    pick.sort(key=lambda r: r["range_m"])
    cards = []
    for r in pick:
        src = f"views/{r['pano_id']}__{r['sdot_id'].replace(':', '_')}.jpg"
        cap = (f"{html.escape(r['sdot_id'])} · {r['range_m']:.1f} m · {html.escape(r['pano_date'] or '?')} · "
               f"{'on paint' if r['on_paint'] else 'off paint'}"
               + (f" · shift {r['shift']:.3f}" if r["shift"] is not None else " · no component"))
        cards.append(f'<figure><img src="{src}" alt="View toward {html.escape(r["sdot_id"])} from pano '
                     f'{html.escape(r["pano_id"])}" loading="lazy"><figcaption>{cap}</figcaption></figure>')
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SDOT crosswalk pilot</title>
  <style>
    :root {{ --bg: #fafafa; --fg: #222; --muted: #666; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #161616; --fg: #e8e8e8; --muted: #aaa; }} }}
    body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(256px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>SDOT crosswalk pilot: B1 and B2 marks</h1>
  <p>{len(pick)} seeded placements, nearest first. Red ring: B1, the SDOT point projected. Cyan cross: B2, the
  centroid of the Vistas crosswalk component it snaps to. Yellow tint: Vistas crosswalk pixels. View is 90°,
  centred on B1.</p>
  <div class="grid">
    {chr(10).join(cards)}
  </div>
</body>
</html>
"""
    with open(os.path.join(a.out, "gallery.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    log(f"gallery -> {os.path.join(a.out, 'gallery.html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "meta", "fetch", "place", "report", "figures", "gallery"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        if name in ("sample", "meta"):
            s.add_argument("--inventory", required=True, help="crosswalk-inventory-274 analysis_out dir")
        if name == "sample":
            s.add_argument("--phase1-sheet")
        if name == "meta":
            s.add_argument("--census", required=True, help="Phase 0 census OUT dir (osm/ tiles)")
        if name in ("meta", "fetch", "place"):
            s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"sample": cmd_sample, "meta": cmd_meta, "fetch": cmd_fetch, "place": cmd_place, "report": cmd_report,
     "figures": cmd_figures, "gallery": cmd_gallery}[a.cmd](a)


if __name__ == "__main__":
    main()
