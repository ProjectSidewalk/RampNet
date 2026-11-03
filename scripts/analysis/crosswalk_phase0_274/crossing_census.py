"""Phase 0 census for Planning#36 / RampNet#274: crowd Crosswalk and Signal labels vs OSM crossing and
signal nodes, per public Project Sidewalk city.

Stdlib only. Everything fetched is cached under --out so a re-run is offline.

Per city it reports:
  labels            crowd labels of the type (all users), and how many are human-validated correct /
                    incorrect under a simplified margin rule (human votes only: net >= 2 Agree -> correct,
                    net >= 2 Disagree -> incorrect; the AI account's votes are ignored)
  osm nodes         OSM nodes of the matching kind inside the 0.1-degree tiles that hold >= 1 label
                    (crossing: highway=crossing; signal: highway=traffic_signals or crossing=traffic_signals)
  marked share      of crossing nodes: crossing:markings != no, or crossing in {marked, zebra, ...}
  nearest distance  for each label (server lat/lng), haversine distance to the nearest OSM node of the kind;
                    median and the share within 5 / 10 / 15 / 25 m, for all nodes and for marked-only.

Usage:
  python -I crossing_census.py --out <dir> [--only seattle-wa,richmond-va] [--types Crosswalk,Signal]
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CITIES_URL = "https://sidewalk-sea.cs.washington.edu/v3/api/cities"
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]   # private.coffee unreachable 2026-10-28
TILE_DEG = 0.1
PAD_DEG = 0.002   # ~200 m
MARKED_VALUES = {"marked", "zebra", "lines", "ladder", "dashes", "dots", "surface", "pedestrian_signals"}
UNMARKED_VALUES = {"unmarked", "no", "informal"}

KIND = {
    "Crosswalk": {"ql": 'node["highway"="crossing"]', "label": "Crosswalk"},
    "Signal": {"ql": '(node["highway"="traffic_signals"];node["crossing"="traffic_signals"];)', "label": "Signal"},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, timeout=120, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sidewalk-auto-labeler census (jonf@cs.uw.edu)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def cached(path, fetch):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    data = fetch()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return data


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def verdict(validations_json):
    """Human-only margin rule. Returns 'correct' | 'incorrect' | 'undecided'."""
    try:
        votes = json.loads(validations_json) if validations_json else []
    except json.JSONDecodeError:
        return "undecided"
    agree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Agree")
    disagree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Disagree")
    if agree - disagree >= 2:
        return "correct"
    if disagree - agree >= 2:
        return "incorrect"
    return "undecided"


def is_marked(tags):
    m = tags.get("crossing:markings")
    if m is not None:
        return "unmarked" if m == "no" else "marked"
    c = tags.get("crossing")
    if c in MARKED_VALUES:
        return "marked"
    if c in UNMARKED_VALUES:
        return "unmarked"
    return "unknown"


class Grid:
    """Nearest-point lookup over a lat/lng point set by 0.002-degree buckets (~200 m)."""

    def __init__(self, pts, cell=0.002):
        self.cell = cell
        self.b = {}
        for p in pts:
            k = (int(p[0] // cell), int(p[1] // cell))
            self.b.setdefault(k, []).append(p)

    def nearest(self, lat, lng, max_rings=3):
        ci, cj = int(lat // self.cell), int(lng // self.cell)
        best = None
        for ring in range(max_rings + 1):
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    if max(abs(i - ci), abs(j - cj)) != ring:
                        continue
                    for p in self.b.get((i, j), ()):
                        d = haversine_m(lat, lng, p[0], p[1])
                        if best is None or d < best:
                            best = d
            if best is not None and best < ring * self.cell * 111000 * 0.9:
                return best
        return best


def fetch_labels(city, url, label_type, out):
    path = os.path.join(out, "rawlabels", f"{city}_{label_type}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q = urllib.parse.urlencode({"labelType": label_type, "filetype": "csv"})
    data = cached(path, lambda: get(f"{url}/v3/api/rawLabels?{q}", timeout=600))
    text = data.decode("utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    return [r for r in rows if r.get("label_type") == label_type]


def tiles_for(labels):
    t = set()
    for r in labels:
        try:
            lat, lng = float(r["latitude"]), float(r["longitude"])
        except (KeyError, ValueError):
            continue
        t.add((math.floor(lat / TILE_DEG), math.floor(lng / TILE_DEG)))
    return sorted(t)


def overpass_query(ql_for_bbox, bbox, depth=0, max_depth=2):
    """Run one bbox query against the mirrors; on failure split the bbox into quarters (down to
    max_depth) so that an overloaded server gets small, fast queries. Returns a list of elements."""
    s, w, n, e = bbox
    bb = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    ql = ql_for_bbox(bb)
    last = None
    for base in OVERPASS:
        for attempt in range(2):
            try:
                data = get(base + "?" + urllib.parse.urlencode({"data": ql}), timeout=200, retries=1)
                j = json.loads(data)
                if "remark" in j and "runtime error" in j.get("remark", "").lower():
                    raise RuntimeError(j["remark"])
                time.sleep(2.0)
                return j.get("elements", [])
            except Exception as exc:  # noqa: BLE001
                last = exc
                log(f"    overpass miss {base.split('/')[2]} depth={depth} {bb}: {str(exc)[:80]}")
                time.sleep(15)
    if depth >= max_depth:
        raise RuntimeError(f"all mirrors failed for {bb}: {last}")
    mlat, mlng = (s + n) / 2, (w + e) / 2
    out = []
    for q in [(s, w, mlat, mlng), (s, mlng, mlat, e), (mlat, w, n, mlng), (mlat, mlng, n, e)]:
        out.extend(overpass_query(ql_for_bbox, q, depth + 1, max_depth))
    return out


def fetch_osm(city, kind, tiles, out):
    """Union of nodes over tiles; each tile response cached as JSON."""
    d = os.path.join(out, "osm", city, kind)
    os.makedirs(d, exist_ok=True)
    nodes = {}
    for (i, j) in tiles:
        s, w = i * TILE_DEG - PAD_DEG, j * TILE_DEG - PAD_DEG
        n, e = (i + 1) * TILE_DEG + PAD_DEG, (j + 1) * TILE_DEG + PAD_DEG
        path = os.path.join(d, f"tile_{i}_{j}.json")
        if kind == "Signal":
            ql_for = lambda bb: f'[out:json][timeout:120];(node["highway"="traffic_signals"]({bb});node["crossing"="traffic_signals"]({bb}););out;'
        else:
            ql_for = lambda bb: f'[out:json][timeout:120];{KIND[kind]["ql"]}({bb});out;'

        def fetch():
            els = overpass_query(ql_for, (s, w, n, e))
            # de-duplicate across quadrant seams
            seen, uniq = set(), []
            for el in els:
                if el.get("id") not in seen:
                    seen.add(el.get("id"))
                    uniq.append(el)
            return json.dumps({"elements": uniq, "split_fetch": True}).encode()

        data = cached(path, fetch)
        for el in json.loads(data).get("elements", []):
            if el.get("type") == "node":
                nodes[el["id"]] = (el["lat"], el["lon"], el.get("tags", {}))
    return nodes


def summarize(labels, nodes, kind):
    pts_all = [(v[0], v[1]) for v in nodes.values()]
    pts_marked = [(v[0], v[1]) for v in nodes.values() if kind == "Crosswalk" and is_marked(v[2]) == "marked"]
    g_all = Grid(pts_all) if pts_all else None
    g_marked = Grid(pts_marked) if pts_marked else None
    d_all, d_marked, per_label = [], [], []
    verdicts = {"correct": 0, "incorrect": 0, "undecided": 0}
    for r in labels:
        v = verdict(r.get("validations", ""))
        verdicts[v] += 1
        try:
            lat, lng = float(r["latitude"]), float(r["longitude"])
        except (KeyError, ValueError):
            continue
        da = g_all.nearest(lat, lng) if g_all else None
        dm = g_marked.nearest(lat, lng) if g_marked else None
        if da is not None:
            d_all.append(da)
        if dm is not None:
            d_marked.append(dm)
        per_label.append({"label_id": r.get("label_id"), "verdict": v, "d_any_m": da, "d_marked_m": dm,
                          "pano_source": r.get("pano_source")})

    def dist_stats(ds):
        if not ds:
            return {"n": 0}
        s = sorted(ds)
        n = len(s)
        return {"n": n, "median_m": round(s[n // 2], 1),
                "le5": round(sum(d <= 5 for d in s) / n, 3), "le10": round(sum(d <= 10 for d in s) / n, 3),
                "le15": round(sum(d <= 15 for d in s) / n, 3), "le25": round(sum(d <= 25 for d in s) / n, 3)}

    marks = {"marked": 0, "unmarked": 0, "unknown": 0}
    if kind == "Crosswalk":
        for v in nodes.values():
            marks[is_marked(v[2])] += 1
    return {"labels": len(labels), **{f"labels_{k}": n for k, n in verdicts.items()},
            "osm_nodes": len(nodes), **{f"osm_{k}": n for k, n in marks.items()},
            "dist_any": dist_stats(d_all), "dist_marked": dist_stats(d_marked)}, per_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--types", default="Crosswalk,Signal")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cities = json.loads(cached(os.path.join(a.out, "cities.json"), lambda: get(CITIES_URL)))["cities"]
    cities = [c for c in cities if c.get("visibility") == "public"]
    only = set(a.only.split(",")) if a.only else None
    types = a.types.split(",")
    results_path = os.path.join(a.out, "census.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for c in cities:
        cid, url = c["city_id"], c["url"].rstrip("/")
        if only and cid not in only:
            continue
        for t in types:
            key = f"{cid}/{t}"
            if key in results and "error" not in results[key]:
                continue
            try:
                log(cid, t, "labels...")
                labels = fetch_labels(cid, url, t, a.out)
                tiles = tiles_for(labels)
                log(cid, t, f"{len(labels)} labels, {len(tiles)} tiles; osm...")
                nodes = fetch_osm(cid, t, tiles, a.out) if tiles else {}
                summary, per_label = summarize(labels, nodes, t)
                summary["tiles"] = len(tiles)
                results[key] = summary
                pl = os.path.join(a.out, "per_label", f"{cid}_{t}.csv")
                os.makedirs(os.path.dirname(pl), exist_ok=True)
                with open(pl, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["label_id", "verdict", "d_any_m", "d_marked_m", "pano_source"])
                    w.writeheader()
                    w.writerows(per_label)
                log(cid, t, json.dumps(summary))
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": str(e)}
                log(cid, t, "ERROR", e)
            with open(results_path + ".tmp", "w") as f:
                json.dump(results, f, indent=1)
            os.replace(results_path + ".tmp", results_path)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
