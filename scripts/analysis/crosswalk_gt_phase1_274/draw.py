"""Seeded, stratified candidate draw for the multi-city crosswalk GT round (RampNet#274 Phase 1 item 2).

Inputs are the Phase 0 census cache for each city (`crossing_census.py --out OUT`): the v3 rawLabels
Crosswalk pull `OUT/rawlabels/<city>_Crosswalk.csv` and the Overpass `highway=crossing` tiles
`OUT/osm/<city>/Crosswalk/*.json`. No network.

The unit is an OSM crossing node (one row per node), because the question Stage 1 asks of OSM is whether a
node is a painted crossing. Four strata:

  marked / untagged / unmarked   nodes whose nearest crowd Crosswalk label is within MATCH_M (10 m, the
                                 Phase 0 item 2 match), split by the node's tags (Phase 0 `is_marked`:
                                 untagged = neither `crossing:markings` nor a known `crossing` value).
                                 The row carries one of the node's labels: a human-confirmed one when the
                                 node has any (human-only margin rule, net >= 2 Agree), else an undecided one;
                                 `label_tier` says which. Labels the humans rejected (net >= 2 Disagree) never
                                 enter. Nodes with a confirmed label are drawn first; undecided ones fill a
                                 stratum only when the confirmed pool is short.
  osm_only                       nodes with NO crowd Crosswalk label (any verdict) within NOLABEL_MIN_M (15 m)
                                 but at least one within AUDITED_MAX_M (100 m): a crossing in a block the crowd
                                 audited and did not label. This is the stratum that measures more than tag
                                 precision on crowd-found crossings. Drawn uniformly over such nodes, any tags.

A label is attached to its single nearest node; a node is in a labeled stratum if any label attached to it
within 10 m. 10-15 m is a deliberate gap so a node is never both "labeled" and "unlabeled".

Allocation: n // 4 per stratum (the remainder to the first strata in STRATA order); a stratum whose pool is
short gives its shortfall to the others, round-robin in STRATA order, until n is met or every pool is empty.
Each stratum's candidate list is the quota plus RESERVE spares, in seeded order; `archive.py` takes the first
`quota` candidates whose imagery is not blank and records every skip, so the final sheet is a deterministic
function of the seed, the inputs (sha256 in the draw manifest) and the recorded skip list.

Seeds are strings, `f"{SEED}:{city}:{stratum}"`, so a city or stratum can be redrawn alone and Python's
`random.Random(str)` is stable across versions (it hashes the string with SHA-512).

Usage:
  python draw.py --census OUT --out analysis_out/crosswalk_gt_phase1_274 [--only amsterdam]
writes `<out>/candidates_<city>.csv` and `<out>/draw_manifest.json`.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "crosswalk_phase0_274"))
sys.path.insert(0, HERE)
from crossing_census import Grid, haversine_m, is_marked, verdict  # noqa: E402
from cities import CITIES, DRAW_CITIES  # noqa: E402

SEED = "crosswalk-gt-phase1-274"
MATCH_M = 10.0
NOLABEL_MIN_M = 15.0
AUDITED_MAX_M = 100.0
RESERVE = 10
STRATA = ("marked", "untagged", "unmarked", "osm_only")
CLASS_NAME = {"marked": "marked", "unmarked": "unmarked", "unknown": "untagged"}

FIELDS = ["city", "stratum", "cand_rank", "node_id", "node_lat", "node_lng", "osm_class", "crossing_tag",
          "crossing_markings_tag", "label_uid", "label_id", "label_tier", "label_lat", "label_lng", "dist_m",
          "n_labels_on_node", "nearest_label_m", "pano_id", "pano_source", "image_capture_date", "time_created",
          "pano_url"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PtGrid(Grid):
    """Grid over (lat, lng, key) points with a within-radius query."""

    def within(self, lat, lng, radius_m):
        ci, cj = int(lat // self.cell), int(lng // self.cell)
        rings = int(radius_m / (self.cell * 111000 * 0.5)) + 1
        out = []
        for i in range(ci - rings, ci + rings + 1):
            for j in range(cj - rings, cj + rings + 1):
                for p in self.b.get((i, j), ()):
                    d = haversine_m(lat, lng, p[0], p[1])
                    if d <= radius_m:
                        out.append((d, p[2]))
        return sorted(out)


def load_inputs(census, city):
    """(labels, nodes, input_files). labels: rows of the rawLabels pull with a position; nodes: {id: (lat, lng, tags)}."""
    lp = os.path.join(census, "rawlabels", f"{city}_Crosswalk.csv")
    with open(lp, encoding="utf-8", errors="replace", newline="") as f:
        labels = [r for r in csv.DictReader(f) if r.get("label_type") == "Crosswalk"]
    good = []
    for r in labels:
        try:
            r["_lat"], r["_lng"] = float(r["latitude"]), float(r["longitude"])
        except (KeyError, ValueError):
            continue
        r["_verdict"] = verdict(r.get("validations", ""))
        good.append(r)
    d = os.path.join(census, "osm", city, "Crosswalk")
    nodes, files = {}, {os.path.relpath(lp, census).replace(os.sep, "/"): sha256_file(lp)}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            p = os.path.join(d, fn)
            files[os.path.relpath(p, census).replace(os.sep, "/")] = sha256_file(p)
            for el in json.load(open(p, encoding="utf-8")).get("elements", []):
                if el.get("type") == "node":
                    nodes[str(el["id"])] = (el["lat"], el["lon"], el.get("tags", {}))
    return good, nodes, files


def pools(labels, nodes):
    """Per stratum, the list of candidate dicts in a canonical (sorted) order, before the seeded shuffle."""
    ngrid = PtGrid([(v[0], v[1], k) for k, v in nodes.items()])
    lgrid = PtGrid([(r["_lat"], r["_lng"], i) for i, r in enumerate(labels)])
    attached = {}
    for i, r in enumerate(labels):
        near = ngrid.within(r["_lat"], r["_lng"], MATCH_M)
        if near:
            attached.setdefault(near[0][1], []).append((near[0][0], i))
    out = {s: [] for s in STRATA}
    for nid in sorted(nodes, key=int):
        lat, lng, tags = nodes[nid]
        base = {"node_id": nid, "node_lat": lat, "node_lng": lng, "osm_class": CLASS_NAME[is_marked(tags)],
                "crossing_tag": tags.get("crossing", ""), "crossing_markings_tag": tags.get("crossing:markings", "")}
        mine = [(d, i) for d, i in attached.get(nid, []) if labels[i]["_verdict"] != "incorrect"]
        if attached.get(nid):
            if not mine:
                continue   # only rejected labels: neither a labeled crossing nor an unlabeled one
            conf = sorted((d, i) for d, i in mine if labels[i]["_verdict"] == "correct")
            tier = "confirmed" if conf else "undecided"
            choice = conf if conf else sorted(mine)
            out[base["osm_class"]].append({**base, "_tier": tier, "_choices": [i for _, i in choice],
                                           "n_labels_on_node": len(attached[nid])})
            continue
        near = lgrid.within(lat, lng, AUDITED_MAX_M)
        if near and near[0][0] > NOLABEL_MIN_M:
            out["osm_only"].append({**base, "_tier": "", "_choices": [], "n_labels_on_node": 0,
                                    "nearest_label_m": round(near[0][0], 2)})
    return out


def allocate(n, sizes):
    """Quota per stratum: n // 4 each (remainder to the first strata), shortfalls redistributed round-robin."""
    q = {s: n // len(STRATA) + (1 if k < n % len(STRATA) else 0) for k, s in enumerate(STRATA)}
    q = {s: min(q[s], sizes[s]) for s in STRATA}
    while sum(q.values()) < n and any(q[s] < sizes[s] for s in STRATA):
        for s in STRATA:
            if sum(q.values()) < n and q[s] < sizes[s]:
                q[s] += 1
    return q


def draw_city(census, city, n):
    labels, nodes, files = load_inputs(census, city)
    pl = pools(labels, nodes)
    sizes = {s: len(pl[s]) for s in STRATA}
    quota = allocate(n, sizes)
    rows = []
    for s in STRATA:
        rng = random.Random(f"{SEED}:{city}:{s}")
        pool = pl[s]
        if s == "osm_only":
            order = rng.sample(pool, len(pool))
        else:   # confirmed nodes first, each tier shuffled
            conf = [c for c in pool if c["_tier"] == "confirmed"]
            und = [c for c in pool if c["_tier"] == "undecided"]
            order = rng.sample(conf, len(conf)) + rng.sample(und, len(und))
        for rank, c in enumerate(order[:quota[s] + RESERVE]):
            row = {k: "" for k in FIELDS}
            row.update({k: v for k, v in c.items() if not k.startswith("_")})
            row.update({"city": city, "stratum": s, "cand_rank": rank, "label_tier": c["_tier"]})
            if c["_choices"]:
                lab = labels[rng.choice(c["_choices"])]
                row.update({"label_id": lab["label_id"], "label_uid": f"{city}:{lab['label_id']}",
                            "label_lat": lab["_lat"], "label_lng": lab["_lng"],
                            "dist_m": round(haversine_m(c["node_lat"], c["node_lng"], lab["_lat"], lab["_lng"]), 2),
                            "pano_id": lab.get("pano_id", ""), "pano_source": lab.get("pano_source", ""),
                            "image_capture_date": lab.get("image_capture_date", ""),
                            "time_created": (lab.get("time_created") or "")[:10], "pano_url": lab.get("pano_url", "")})
            rows.append(row)
    tiers = {s: {"confirmed": sum(c["_tier"] == "confirmed" for c in pl[s]),
                 "undecided": sum(c["_tier"] == "undecided" for c in pl[s])} for s in STRATA if s != "osm_only"}
    meta = {"n": n, "pool_sizes": sizes, "pool_tiers": tiers, "quota": quota, "reserve": RESERVE,
            "labels_in_pull": len(labels), "osm_nodes": len(nodes),
            "labels_by_verdict": {v: sum(r["_verdict"] == v for r in labels) for v in ("correct", "undecided", "incorrect")},
            "inputs_sha256": files}
    uids = [r["label_uid"] for r in rows if r["label_uid"]]
    assert len(uids) == len(set(uids)), f"{city}: a label drawn twice"
    assert len({r["node_id"] for r in rows}) == len(rows), f"{city}: a node drawn twice"
    return rows, meta


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, help="the Phase 0 census cache dir (crossing_census.py --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "draw_manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
    manifest.update({"seed": SEED, "match_m": MATCH_M, "nolabel_min_m": NOLABEL_MIN_M,
                     "audited_max_m": AUDITED_MAX_M, "strata": list(STRATA)})
    manifest.setdefault("cities", {})
    for city in (a.only.split(",") if a.only else DRAW_CITIES):
        rows, meta = draw_city(a.census, city, CITIES[city]["n"])
        path = os.path.join(a.out, f"candidates_{city}.csv")
        write_csv(path, rows)
        meta["candidates_sha256"] = sha256_file(path)
        manifest["cities"][city] = meta
        print(f"{city}: pools {meta['pool_sizes']} -> quota {meta['quota']}; {len(rows)} candidates -> {path}")
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
