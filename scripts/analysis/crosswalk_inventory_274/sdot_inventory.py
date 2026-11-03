"""Seattle's own marked-crosswalk inventory (SDOT) as a crosswalk oracle (RampNet#274, Phase 1 side work).

Question: can SDOT's "Marked Crosswalks" layer stand in for human aerial ratings in Seattle? If it agrees with
the 120 human verdicts of the Phase 0 aerial spot-check, it is free ground truth for every OSM crossing node and
every crowd Crosswalk label in the city, which a rating page can never be.

  python sdot_inventory.py fetch --out analysis_out/crosswalk_inventory_274
  python sdot_inventory.py score --out analysis_out/crosswalk_inventory_274 \
      --phase0 analysis_out/crosswalk_phase0_274/aerial_sample_seattle-wa.rated.csv \
      --census D:/Git/sal-scratch-keep/phase0-census-2026-10-28/census/out

`fetch` (network: SDOT's ArcGIS FeatureServer + the Seattle Project Sidewalk server's regions, GET only) writes
`inputs/sdot_marked_crosswalks.geojson`, `inputs/ps_regions_seattle.geojson` (both untracked) and the tracked
`inputs.json` (url, fetch time, sha256, feature count per pull). A pull whose feature count differs from the
server's own count is refused. `score` reads only those files plus the Phase 0 census cache (rawLabels + OSM
tiles, 2026-10-28) and the committed Phase 0 sheet; it writes `report.md` and three CSVs.

THE RULE, fixed before any score was computed (2026-11-03):
  * An inventory point "is at" an OSM node when it lies within MATCH_M = 10 m of it (5, 15 and 25 m reported
    as sensitivity, never used for the verdict). Points are SDOT's own coordinates; their convention (centre of
    the crosswalk or a reference offset) is unknown and is read from the distance distribution, not assumed.
  * Inventory = the layer's own recommended filter, CURRENT_STATUS in (INSVC, PLNRECON) or empty. All statuses
    are reported beside it.
  * Calibration set = the 120 Phase 0 rows, human final verdict (`human_painted`); `unclear` is left out.
    Sensitivity = share of human-yes rows with an inventory point at the node. Specificity = share of human-no
    rows without one.
  * Verdict GT-GRADE when sensitivity >= 0.90 with its Wilson 95% lower bound >= 0.85 AND specificity >= 0.75;
    else THIRD OPINION. The no-row count is small (12 in Phase 0), so specificity is reported with its interval
    and flagged as underpowered whatever it reads.
  Caveat built into the sample: every Phase 0 row is a node with a crowd label near it, so the calibration is
  conditional on the crowd having labelled there.

The city-wide tables (OSM nodes by marking class, SDOT points without an OSM node, crowd labels by human
verdict) are descriptive and carry no rule. Every table is restricted to the Seattle server's region polygons.
Labels are keyed `seattle-wa:<label_id>` (label ids are per city).
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "crosswalk_phase0_274"))
from crossing_census import is_marked, verdict  # noqa: E402  (Phase 0's classifier and human-margin rule)

CITY = "seattle-wa"
SDOT_URL = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Marked_Crosswalks_CDL/FeatureServer/0"
REGIONS_URL = "https://sidewalk-sea.cs.washington.edu/v3/api/regions?filetype=geojson"
CURRENT = {"INSVC", "PLNRECON", ""}
MATCH_M = 10.0
SENS_M = (5.0, 10.0, 15.0, 25.0)
HOLE_M = 100.0   # post hoc only: farther than this from ANY SDOT point = outside the inventory's coverage
RULE = {"match_m": MATCH_M, "sens_min": 0.90, "sens_wilson_lo_min": 0.85, "spec_min": 0.75,
        "inventory_filter": "CURRENT_STATUS in (INSVC, PLNRECON) or empty", "fixed": "2026-11-03"}
# AMENDMENT, 2026-11-03, made after the first score (THIRD OPINION, Wilson lower 0.848 vs 0.85) and announced as
# such, not folded into RULE: a calibration row with no SDOT point of ANY status within HOLE_M is outside the
# inventory's coverage and leaves the denominator, the same way `unclear` does. The thresholds are unchanged.
# Why: 5 of the 9 misses sat in two places SDOT does not own (a private shopping-centre lot, a state route), so
# they measure coverage, not paint. RULE's verdict is still computed and printed beside the amended one.
AMENDMENT = {"coverage_mask_m": HOLE_M, "coverage_points": "any status", "dated": "2026-11-03",
             "post_hoc": True}
UA = {"User-Agent": "ProjectSidewalk-RampNet-crosswalk-274 (research; jonf@cs.uw.edu)"}


# ---------------------------------------------------------------- helpers

def get_json(url, timeout=120, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 -- network: retry, then raise
            if k == retries - 1:
                raise
            print(f"  retry {k + 1} after {e}")
            time.sleep(5 * (k + 1))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Nearest:
    """Exact nearest-item lookup over (lat, lng, item) by ~50 m buckets; searches until no closer cell can exist."""

    def __init__(self, items, cell_m=50.0, lat0=47.6):
        self.dlat = cell_m / 111320.0
        self.dlng = cell_m / (111320.0 * math.cos(math.radians(lat0)))
        self.cell_m = cell_m
        self.b = {}
        for it in items:
            self.b.setdefault((int(it[0] // self.dlat), int(it[1] // self.dlng)), []).append(it)

    def query(self, lat, lng, max_m=500.0):
        """(distance_m, item) of the nearest item within max_m, else (None, None)."""
        ci, cj = int(lat // self.dlat), int(lng // self.dlng)
        best = (None, None)
        for ring in range(int(max_m // self.cell_m) + 2):
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    if max(abs(i - ci), abs(j - cj)) != ring:
                        continue
                    for it in self.b.get((i, j), ()):
                        d = haversine_m(lat, lng, it[0], it[1])
                        if best[0] is None or d < best[0]:
                            best = (d, it)
            # every unvisited cell is at least ring * cell_m away
            if best[0] is not None and best[0] <= ring * self.cell_m:
                break
        return best if best[0] is not None and best[0] <= max_m else (None, None)


def point_in_ring(lat, lng, ring):
    inside = False
    n = len(ring)
    for k in range(n):
        x1, y1 = ring[k][0], ring[k][1]
        x2, y2 = ring[(k + 1) % n][0], ring[(k + 1) % n][1]
        if (y1 > lat) != (y2 > lat) and lng < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class Regions:
    def __init__(self, gj):
        self.polys = []
        for f in gj["features"]:
            g = f["geometry"]
            ps = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for p in ps:
                xs = [c[0] for c in p[0]]
                ys = [c[1] for c in p[0]]
                self.polys.append(((min(ys), max(ys), min(xs), max(xs)), p))

    def contains(self, lat, lng):
        for (y0, y1, x0, x1), p in self.polys:
            if y0 <= lat <= y1 and x0 <= lng <= x1 and point_in_ring(lat, lng, p[0]) \
                    and not any(point_in_ring(lat, lng, h) for h in p[1:]):
                return True
        return False


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


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


def fmt_m(x):
    return "" if x is None else f"{x:.2f}"


# ---------------------------------------------------------------- fetch

def fetch(a):
    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    rec = {}
    n_server = get_json(f"{SDOT_URL}/query?where=1%3D1&returnCountOnly=true&f=json")["count"]
    feats, off = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                                    "orderByFields": "OBJECTID", "resultOffset": off, "resultRecordCount": 2000})
        page = get_json(f"{SDOT_URL}/query?{q}")
        fs = page.get("features", [])
        feats += fs
        print(f"  SDOT page at {off}: {len(fs)}")
        if len(fs) < 2000:
            break
        off += len(fs)
    ids = [f["properties"]["OBJECTID"] for f in feats]
    assert len(set(ids)) == len(ids), "duplicate OBJECTID across pages"
    if len(feats) != n_server:
        sys.exit(f"REFUSED: pulled {len(feats)} features, server count is {n_server}")
    p = os.path.join(a.out, "inputs", "sdot_marked_crosswalks.geojson")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    meta = get_json(f"{SDOT_URL}?f=json")
    rec["sdot"] = {"url": SDOT_URL, "features": len(feats), "server_count": n_server, "sha256": sha256(p),
                   "layer_last_edit_utc": datetime.datetime.fromtimestamp(
                       meta["editingInfo"]["lastEditDate"] / 1000, datetime.timezone.utc).isoformat()
                   if meta.get("editingInfo") else None,
                   "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "file": "inputs/sdot_marked_crosswalks.geojson"}
    gj = get_json(REGIONS_URL)
    p = os.path.join(a.out, "inputs", "ps_regions_seattle.geojson")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(gj, f)
    rec["regions"] = {"url": REGIONS_URL, "features": len(gj["features"]), "sha256": sha256(p),
                      "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                      "file": "inputs/ps_regions_seattle.geojson"}
    with open(os.path.join(a.out, "inputs.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"SDOT {len(feats)} features, regions {len(gj['features'])} -> {a.out}/inputs.json")


# ---------------------------------------------------------------- score

def load_inputs(a):
    rec = json.load(open(os.path.join(a.out, "inputs.json"), encoding="utf-8"))
    for k, v in rec.items():
        got = sha256(os.path.join(a.out, v["file"]))
        assert got == v["sha256"], f"{k}: {v['file']} changed since fetch (sha256 {got[:12]} != {v['sha256'][:12]})"
    sd = json.load(open(os.path.join(a.out, rec["sdot"]["file"]), encoding="utf-8"))
    regions = Regions(json.load(open(os.path.join(a.out, rec["regions"]["file"]), encoding="utf-8")))
    return rec, sd, regions


def sdot_points(sd):
    pts = []
    for f in sd["features"]:
        g, p = f.get("geometry"), f["properties"]
        if not g or not g.get("coordinates"):
            continue
        lng, lat = g["coordinates"][:2]
        pts.append((lat, lng, {"objectid": p["OBJECTID"], "unitid": p.get("UNITID") or "",
                               "status": (p.get("CURRENT_STATUS") or "").strip(),
                               "marking": (p.get("MARKING_TYPE") or "").strip(),
                               "midblock": (p.get("MIDBLOCK_CROSSWALK") or "").strip(),
                               "ownership": (p.get("OWNERSHIP") or "").strip(),
                               "condition": (p.get("CONDITION") or "").strip(),
                               "install": p.get("INSTALL_DATE")}))
    return pts


def osm_nodes(census):
    nodes = {}
    for fp in sorted(glob.glob(os.path.join(census, "osm", CITY, "Crosswalk", "*.json"))):
        for e in json.load(open(fp, encoding="utf-8")).get("elements", []):
            if e.get("type") == "node" and (e.get("tags") or {}).get("highway") == "crossing":
                nodes[e["id"]] = (e["lat"], e["lon"], is_marked(e.get("tags") or {}))
    return nodes


def crowd_labels(census):
    out = []
    with open(os.path.join(census, "rawlabels", f"{CITY}_Crosswalk.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("latitude"):
                continue
            out.append((float(r["latitude"]), float(r["longitude"]),
                        {"label_uid": f"{CITY}:{r['label_id']}", "verdict": verdict(r.get("validations"))}))
    uids = [x[2]["label_uid"] for x in out]
    assert len(set(uids)) == len(uids), "duplicate label_uid"
    return out


def score(a):
    rec, sd, regions = load_inputs(a)
    allp = sdot_points(sd)
    cur = [p for p in allp if p[2]["status"] in CURRENT]
    cur_in = [p for p in cur if regions.contains(p[0], p[1])]
    near_cur, near_all = Nearest(cur), Nearest(allp)
    lines = []
    w = lines.append
    count = lambda xs, key: sorted(((v, sum(1 for x in xs if x[2][key] == v)) for v in {x[2][key] for x in xs}),
                                   key=lambda t: -t[1])

    w("# Seattle SDOT marked-crosswalk inventory as a crosswalk oracle (RampNet#274)\n")
    w(f"Generated by `scripts/analysis/crosswalk_inventory_274/sdot_inventory.py score`. SDOT layer pulled "
      f"{rec['sdot']['fetched_utc']} ({rec['sdot']['features']} features, layer last edited "
      f"{rec['sdot']['layer_last_edit_utc']}, sha256 `{rec['sdot']['sha256'][:16]}…`); Project Sidewalk "
      f"Seattle regions {rec['regions']['features']} polygons. OSM tiles and crowd labels: the Phase 0 census "
      f"cache (2026-10-28).\n")
    w(f"Rule (fixed before scoring, {RULE['fixed']}): match radius {MATCH_M:.0f} m; inventory = "
      f"{RULE['inventory_filter']}; GT-GRADE iff sensitivity ≥ {RULE['sens_min']} with Wilson lower bound ≥ "
      f"{RULE['sens_wilson_lo_min']} and specificity ≥ {RULE['spec_min']}.\n")

    w("## 1. The inventory\n")
    w(f"{len(sd['features'])} features, {len(sd['features']) - len(allp)} of them without a geometry (dropped); "
      f"{len(allp)} points, {len(cur)} pass the recommended status filter, {len(cur_in)} of them inside "
      f"Project Sidewalk's Seattle regions.\n")
    w("| field | values (count), all statuses |\n|---|---|")
    for key in ("status", "marking", "midblock", "ownership", "condition"):
        w(f"| {key} | " + ", ".join(f"`{v or '(empty)'}` {n}" for v, n in count(allp, key)[:12]) + " |")
    w("")

    # ---- 2. calibration against the Phase 0 human verdicts
    p0 = list(csv.DictReader(open(a.phase0, encoding="utf-8", newline="")))
    rows = []
    for r in p0:
        nl, ng = float(r["node_lat"]), float(r["node_lng"])
        d, it = near_cur.query(nl, ng)
        da, ita = near_all.query(nl, ng)
        dl, _ = near_cur.query(float(r["label_lat"]), float(r["label_lng"]))
        rows.append({"label_uid": r["label_uid"], "node_id": r["node_id"], "osm_class": r["osm_marking_class"],
                     "human_painted": r["human_painted"], "human_label": r["human_label"],
                     "human_bike": r.get("human_bike", ""),
                     "d_node_current_m": fmt_m(d), "nearest_current_status": it[2]["status"] if it else "",
                     "nearest_current_marking": it[2]["marking"] if it else "",
                     "nearest_current_objectid": it[2]["objectid"] if it else "",
                     "d_node_any_status_m": fmt_m(da), "nearest_any_status": ita[2]["status"] if ita else "",
                     "d_label_current_m": fmt_m(dl)})
    with open(os.path.join(a.out, "phase0_calibration.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        cw.writeheader()
        cw.writerows(rows)

    def dist(r, k="d_node_current_m"):
        return float(r[k]) if r[k] else None

    yes = [r for r in rows if r["human_painted"] == "yes"]
    no = [r for r in rows if r["human_painted"] == "no"]
    w("## 2. Calibration: SDOT vs the 120 human Phase 0 verdicts\n")
    w(f"Human final verdicts: {len(yes)} painted, {len(no)} not, "
      f"{sum(r['human_painted'] == 'unclear' for r in rows)} unclear (left out).\n")
    dy = [dist(r) for r in yes if dist(r) is not None]
    w(f"Distance from the OSM node to the nearest current SDOT point, human-yes rows: median "
      f"{pct(dy, .5):.2f} m, p75 {pct(dy, .75):.2f}, p90 {pct(dy, .9):.2f} m ({len(dy)} of {len(yes)} have one "
      f"within 500 m). This is the point convention read from data.\n")
    w("| radius | sensitivity (human yes, SDOT within r) | specificity (human no, none within r) |\n|---|---|---|")
    res = {}
    for R in SENS_M:
        ks = sum(1 for r in yes if dist(r) is not None and dist(r) <= R)
        kn = sum(1 for r in no if dist(r) is None or dist(r) > R)
        res[R] = (ks, len(yes), kn, len(no))
        w(f"| {R:.0f} m{' (rule)' if R == MATCH_M else ''} | {frac(ks, len(yes))} | {frac(kn, len(no))} |")
    w("")
    ks, ny, kn, nn = res[MATCH_M]
    sens, spec = ks / ny, (kn / nn if nn else float("nan"))
    lo = wilson(ks, ny)[0]
    grade = sens >= RULE["sens_min"] and lo >= RULE["sens_wilson_lo_min"] and spec >= RULE["spec_min"]
    w(f"**Verdict: {'GT-GRADE' if grade else 'THIRD OPINION'}** (sensitivity {sens:.3f}, Wilson lower "
      f"{lo:.3f}; specificity {spec:.3f} on {nn} rows — underpowered at this n).\n")
    # POST HOC (added after the first score, 2026-11-03; never read by the verdict above): a human-yes row whose
    # nearest SDOT point of ANY status is > HOLE_M away is outside the inventory's coverage (private lots,
    # state routes), not a paint disagreement. Reported so the coverage reading and the paint reading stay apart.
    covered = lambda r: dist(r, "d_node_any_status_m") is not None and dist(r, "d_node_any_status_m") <= HOLE_M
    hole = [r for r in yes if not covered(r)]
    cov = [r for r in yes if covered(r)]
    cov_no = [r for r in no if covered(r)]
    w(f"*Post hoc:* {len(hole)} human-yes rows have no SDOT point of any status within {HOLE_M:.0f} m — "
      f"outside the inventory's coverage.\n")
    aks = sum(1 for r in cov if dist(r) is not None and dist(r) <= MATCH_M)
    akn = sum(1 for r in cov_no if dist(r) is None or dist(r) > MATCH_M)
    alo = wilson(aks, len(cov))[0]
    aspec = akn / len(cov_no) if cov_no else float("nan")
    agrade = aks / len(cov) >= RULE["sens_min"] and alo >= RULE["sens_wilson_lo_min"] and aspec >= RULE["spec_min"]
    w(f"**Amended verdict ({AMENDMENT['dated']}, post hoc, announced; coverage mask {HOLE_M:.0f} m, thresholds "
      f"unchanged): {'GT-GRADE' if agrade else 'THIRD OPINION'}** inside SDOT coverage — sensitivity "
      f"{frac(aks, len(cov))}, specificity {frac(akn, len(cov_no))}. Outside coverage SDOT is not evidence "
      f"either way.\n")
    w("By OSM marking class at the rule radius:\n")
    w("| OSM class | human yes: SDOT within r | human no: SDOT within r |\n|---|---|---|")
    for c in ("marked", "unknown", "unmarked"):   # Phase 0's sheet calls the untagged class `unknown`
        cy = [r for r in yes if r["osm_class"] == c]
        cn = [r for r in no if r["osm_class"] == c]
        w(f"| {c} | {frac(sum(1 for r in cy if dist(r) is not None and dist(r) <= MATCH_M), len(cy))} | "
          f"{frac(sum(1 for r in cn if dist(r) is not None and dist(r) <= MATCH_M), len(cn))} |")
    w("")
    dis = [r for r in rows if r["human_painted"] in ("yes", "no")
           and (r["human_painted"] == "yes") != (dist(r) is not None and dist(r) <= MATCH_M)]
    w(f"Disagreements at the rule radius ({len(dis)}), for a human look (nearest any-status point shown, so a "
      f"retired crosswalk is visible):\n")
    w("| label_uid | OSM node | class | human | nearest current (m) | status, marking | nearest any status (m) | status |")
    w("|---|---|---|---|---|---|---|---|")
    for r in dis:
        w(f"| {r['label_uid']} | [{r['node_id']}](https://www.openstreetmap.org/node/{r['node_id']}) | "
          f"{r['osm_class']} | {r['human_painted']} | {r['d_node_current_m'] or '>500'} | "
          f"{r['nearest_current_status']}, {r['nearest_current_marking']} | {r['d_node_any_status_m'] or '>500'} | "
          f"{r['nearest_any_status']} |")
    w("")

    # ---- 3. city-wide: OSM crossing nodes
    nodes = {k: v for k, v in osm_nodes(a.census).items() if regions.contains(v[0], v[1])}
    near_osm = Nearest([(v[0], v[1], {"id": k, "cls": v[2]}) for k, v in nodes.items()])
    near_osm_m = Nearest([(v[0], v[1], {"id": k, "cls": v[2]}) for k, v in nodes.items() if v[2] == "marked"])
    orows = []
    for k, (lat, lng, cls) in sorted(nodes.items()):
        d, it = near_cur.query(lat, lng)
        orows.append({"node_id": k, "osm_class": cls, "d_current_m": fmt_m(d),
                      "nearest_marking": it[2]["marking"] if it else ""})
    with open(os.path.join(a.out, "osm_nodes.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(orows[0]), lineterminator="\n")
        cw.writeheader()
        cw.writerows(orows)
    w("## 3. City-wide: OSM crossing nodes inside the Seattle regions\n")
    w("Share of OSM `highway=crossing` nodes with a current SDOT point within r. OSM class as in Phase 0 "
      "(`unknown` = no marking tag, Phase 0's \"untagged\"). Nodes on SDOT-only streets and on crossings of "
      "footways over driveways count too, so this is a description, not an error rate.\n")
    w("| OSM class | nodes | " + " | ".join(f"within {R:.0f} m" for R in SENS_M) + " |")
    w("|---|---|" + "---|" * len(SENS_M))
    for c in ("marked", "unknown", "unmarked"):
        cs = [r for r in orows if r["osm_class"] == c]
        w(f"| {c} | {len(cs)} | " + " | ".join(
            f"{sum(1 for r in cs if r['d_current_m'] and float(r['d_current_m']) <= R) / max(len(cs), 1):.3f}"
            for R in SENS_M) + " |")
    w("")
    cs = [r for r in orows if r["osm_class"] == "marked"]
    far = sum(1 for r in cs if not r["d_current_m"] or float(r["d_current_m"]) > HOLE_M)
    w(f"Of the {len(cs)} OSM-marked nodes, {far} ({far / max(len(cs), 1):.3f}) have no current SDOT point within "
      f"{HOLE_M:.0f} m: coverage holes (private property, campuses, state routes) rather than mismatched points.\n")

    # ---- 4. SDOT points with no OSM node
    w("## 4. City-wide: current SDOT crosswalks, seen from OSM\n")
    srows = []
    for lat, lng, p in cur_in:
        d, _ = near_osm.query(lat, lng)
        dm, _ = near_osm_m.query(lat, lng)
        srows.append((p, d, dm))
    w("| r | an OSM crossing node within r | a MARKED OSM node within r |\n|---|---|---|")
    for R in SENS_M:
        w(f"| {R:.0f} m | {frac(sum(1 for _, d, _ in srows if d is not None and d <= R), len(srows))} | "
          f"{frac(sum(1 for _, _, dm in srows if dm is not None and dm <= R), len(srows))} |")
    w("")
    w("By SDOT marking type, at the rule radius (any OSM node):\n")
    w("| marking | points | OSM node within r |\n|---|---|---|")
    for v, n in count([(0, 0, p) for p, _, _ in srows], "marking"):
        k = sum(1 for p, d, _ in srows if p["marking"] == v and d is not None and d <= MATCH_M)
        w(f"| `{v or '(empty)'}` | {n} | {frac(k, n)} |")
    w("")

    # ---- 5. crowd labels
    labs = [x for x in crowd_labels(a.census) if regions.contains(x[0], x[1])]
    lrows = []
    for lat, lng, p in labs:
        d, it = near_cur.query(lat, lng)
        do, _ = near_osm.query(lat, lng)
        lrows.append({"label_uid": p["label_uid"], "verdict": p["verdict"], "d_current_m": fmt_m(d),
                      "nearest_marking": it[2]["marking"] if it else "", "d_osm_node_m": fmt_m(do)})
    with open(os.path.join(a.out, "crowd_labels.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(lrows[0]), lineterminator="\n")
        cw.writeheader()
        cw.writerows(lrows)
    w("## 5. City-wide: crowd Crosswalk labels vs SDOT\n")
    w("Human-only margin verdict (Phase 0's rule: agree − disagree ≥ 2 is correct, ≤ −2 incorrect). Share with "
      "a current SDOT point within r; median distance to the nearest one.\n")
    w("| verdict | labels | median to SDOT (m) | " + " | ".join(f"within {R:.0f} m" for R in SENS_M) + " |")
    w("|---|---|---|" + "---|" * len(SENS_M))
    for v in ("correct", "undecided", "incorrect"):
        vs = [r for r in lrows if r["verdict"] == v]
        ds = [float(r["d_current_m"]) for r in vs if r["d_current_m"]]
        w(f"| {v} | {len(vs)} | {pct(ds, .5):.2f} | " + " | ".join(
            f"{sum(1 for x in ds if x <= R) / max(len(vs), 1):.3f}" for R in SENS_M) + " |")
    w("")
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}/report.md and 3 CSVs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--phase0", required=True, help="aerial_sample_seattle-wa.rated.csv (human final verdicts)")
    s.add_argument("--census", required=True, help="Phase 0 census OUT dir (rawlabels/ + osm/)")
    a = ap.parse_args()
    {"fetch": fetch, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
