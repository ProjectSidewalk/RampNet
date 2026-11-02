"""Phase 0 item 2 (RampNet#274): crowd Crosswalk label vs nearest OSM crossing node, decomposed
along and across the crossing axis, per dense city.

Per city: labels (server lat/lng) matched to the nearest OSM crossing node within MATCH_M; a seeded
sample of matched labels; Overpass `way(bn)` with geometry for the sampled nodes (cached); for each
matched label the crossing axis is the footway=crossing way through the node when one exists (axis
source `footway`), else perpendicular to the road way (`road`); the offset vector label - node in
metres is projected on the crossing axis (`along`, signed, sign arbitrary) and on the road axis
(`across`).

Usage: python -I offset_read.py --out ./out [--only seattle-wa] [--sample 1500]
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossing_census import Grid, haversine_m, verdict, is_marked, get, cached, log, OVERPASS  # noqa: E402

MATCH_M = 10.0
IDS_PER_QUERY = 400
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
         "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link",
         "tertiary_link", "road", "busway"}


def dense_cities(census):
    return sorted(k.split("/")[0] for k, v in census.items()
                  if k.endswith("/Crosswalk") and "error" not in v
                  and (v["dist_any"].get("median_m") or 999) <= 5 and v["labels"] >= 100)


def load_nodes(out, city):
    d = os.path.join(out, "osm", city, "Crosswalk")
    nodes = {}
    for f in os.listdir(d):
        if f.endswith(".json"):
            for el in json.load(open(os.path.join(d, f))).get("elements", []):
                if el.get("type") == "node":
                    nodes[el["id"]] = (el["lat"], el["lon"], el.get("tags", {}))
    return nodes


class IdGrid(Grid):
    """Grid over (lat, lng, id) points; nearest_id returns (dist, id)."""

    def nearest_id(self, lat, lng):
        ci, cj = int(lat // self.cell), int(lng // self.cell)
        best = (None, None)
        for ring in range(3):
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    if max(abs(i - ci), abs(j - cj)) != ring:
                        continue
                    for p in self.b.get((i, j), ()):
                        d = haversine_m(lat, lng, p[0], p[1])
                        if best[0] is None or d < best[0]:
                            best = (d, p[2])
            if best[0] is not None and best[0] < ring * self.cell * 111000 * 0.9:
                return best
        return best


def fetch_ways(out, city, node_ids):
    """Ways through the given nodes, with geometry and tags; one cached response per id chunk."""
    d = os.path.join(out, "ways", city)
    os.makedirs(d, exist_ok=True)
    ways = {}
    ids = sorted(node_ids)
    n_chunks = math.ceil(len(ids) / IDS_PER_QUERY)
    for k in range(0, len(ids), IDS_PER_QUERY):
        chunk = ids[k:k + IDS_PER_QUERY]
        path = os.path.join(d, f"ways_{k // IDS_PER_QUERY:03d}_{len(ids)}.json")
        ql = "[out:json][timeout:120];node(id:" + ",".join(map(str, chunk)) + ");way(bn);out geom;"

        def fetch():
            last = None
            for base in OVERPASS:
                for _ in range(2):
                    try:
                        data = get(base + "?" + urllib.parse.urlencode({"data": ql}), timeout=200, retries=1)
                        json.loads(data)
                        time.sleep(2.0)
                        return data
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        log("    overpass miss " + base.split("/")[2] + ": " + str(exc)[:80])
                        time.sleep(15)
            raise RuntimeError("ways fetch failed: " + str(last))

        for el in json.loads(cached(path, fetch)).get("elements", []):
            if el.get("type") == "way":
                ways[el["id"]] = el
        log(f"  {city}: ways chunk {k // IDS_PER_QUERY + 1}/{n_chunks}")
    return ways


def bearing_deg(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def way_bearing_at(way, node_id):
    """Bearing of a way at one of its nodes, from the neighbouring vertices (geometry parallels nodes)."""
    nodes, geom = way.get("nodes", []), way.get("geometry", [])
    if node_id not in nodes or len(geom) != len(nodes) or len(nodes) < 2:
        return None
    i = nodes.index(node_id)
    a = geom[max(0, i - 1)]
    b = geom[min(len(geom) - 1, i + 1)]
    if a is b:
        return None
    return bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"])


def is_crossing_footway(tags):
    return tags.get("footway") == "crossing" or tags.get("cycleway") == "crossing" or tags.get("path") == "crossing"


def is_road(tags):
    return tags.get("highway", "") in ROADS


def stats(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    return {"n": n, "median": round(s[n // 2], 2), "p90": round(s[min(n - 1, int(0.9 * n))], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--sample", type=int, default=1500)
    a = ap.parse_args()
    census = json.load(open(os.path.join(a.out, "census.json")))
    cities = dense_cities(census)
    if a.only:
        cities = [c for c in cities if c in set(a.only.split(","))]
    log("dense cities:", cities)
    res_path = os.path.join(a.out, "offset_read.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    fields = ["label_uid", "node_id", "dist_m", "axis", "along_m", "across_m", "node_marking",
              "verdict", "n_roads", "n_footways", "oneway_road"]
    for city in cities:
        if city in results:
            continue
        with open(os.path.join(a.out, "rawlabels", f"{city}_Crosswalk.csv"), encoding="utf-8", errors="replace") as f:
            labels = [r for r in csv.DictReader(f) if r.get("label_type") == "Crosswalk"]
        nodes = load_nodes(a.out, city)
        g = IdGrid([(v[0], v[1], k) for k, v in nodes.items()])
        matched = []
        for r in labels:
            try:
                lat, lng = float(r["latitude"]), float(r["longitude"])
            except (KeyError, ValueError):
                continue
            d, nid = g.nearest_id(lat, lng)
            if d is not None and d <= MATCH_M:
                matched.append((r, lat, lng, nid, d))
        rng = random.Random(274)
        sample = matched if len(matched) <= a.sample else rng.sample(matched, a.sample)
        log(f"{city}: {len(labels)} labels, {len(matched)} matched <= {MATCH_M} m, sample {len(sample)}")
        try:
            ways = fetch_ways(a.out, city, {s[3] for s in sample})
        except RuntimeError as exc:
            log(city, 'ERROR', str(exc)[:120])
            continue
        by_node = {}
        for w in ways.values():
            for n in w.get("nodes", []):
                by_node.setdefault(n, []).append(w)
        rows = []
        for r, lat, lng, nid, d in sample:
            nlat, nlng, ntags = nodes[nid]
            uid = f"{city}:{r.get('label_id')}"
            fw = [w for w in by_node.get(nid, []) if is_crossing_footway(w.get("tags", {}))]
            rd = [w for w in by_node.get(nid, []) if is_road(w.get("tags", {}))]
            axis_src, cross_b = None, None
            if fw:
                cross_b = way_bearing_at(fw[0], nid)
                axis_src = "footway"
            if cross_b is None and rd:
                rb = way_bearing_at(rd[0], nid)
                if rb is not None:
                    cross_b = (rb + 90) % 360
                    axis_src = "road"
            if cross_b is None:
                rows.append({"label_uid": uid, "node_id": nid, "dist_m": round(d, 2), "axis": "none"})
                continue
            dn = (lat - nlat) * 111320.0
            de = (lng - nlng) * 111320.0 * math.cos(math.radians(nlat))
            cb = math.radians(cross_b)
            along = dn * math.cos(cb) + de * math.sin(cb)
            across = -dn * math.sin(cb) + de * math.cos(cb)
            rows.append({"label_uid": uid, "node_id": nid, "dist_m": round(d, 2), "axis": axis_src,
                         "along_m": round(along, 2), "across_m": round(across, 2), "node_marking": is_marked(ntags),
                         "verdict": verdict(r.get("validations", "")), "n_roads": len(rd), "n_footways": len(fw),
                         "oneway_road": int(bool(rd) and rd[0].get("tags", {}).get("oneway") == "yes")})
        os.makedirs(os.path.join(a.out, "offsets"), exist_ok=True)
        with open(os.path.join(a.out, "offsets", f"{city}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        ok = [x for x in rows if x["axis"] != "none"]
        summary = {"labels": len(labels), "matched": len(matched), "sample": len(sample), "with_axis": len(ok),
                   "axis_footway": sum(1 for x in ok if x["axis"] == "footway"),
                   "abs_along": stats([abs(x["along_m"]) for x in ok]),
                   "abs_across": stats([abs(x["across_m"]) for x in ok]),
                   "share_along_gt3": round(sum(1 for x in ok if abs(x["along_m"]) > 3) / len(ok), 3) if ok else None,
                   "share_across_gt3": round(sum(1 for x in ok if abs(x["across_m"]) > 3) / len(ok), 3) if ok else None}
        results[city] = summary
        log(city, json.dumps(summary))
        with open(res_path + ".tmp", "w") as f:
            json.dump(results, f, indent=1)
        os.replace(res_path + ".tmp", res_path)
    log("done")


if __name__ == "__main__":
    main()
