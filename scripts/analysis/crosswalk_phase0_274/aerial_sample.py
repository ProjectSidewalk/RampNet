"""Seeded rating sample for the one-city aerial reference (RampNet#274 Phase 0 item 2).

Draws, from one city's offsets CSV, N human-confirmed Crosswalk labels per OSM node-marking class
(marked / unmarked / untagged) and writes a CSV with the label and node positions, the offset, and
map links, for a person to rate on an aerial basemap: is there a painted crossing at the node, and
is the label at its centre. Nothing here decides anything; it is the rating sheet.

Usage: python -I aerial_sample.py --out ./out --city seattle-wa --per-class 40
"""
import argparse
import csv
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--city", default="seattle-wa")
ap.add_argument("--per-class", type=int, default=40)
a = ap.parse_args()

offsets = [x for x in csv.DictReader(open(os.path.join(a.out, "offsets", f"{a.city}.csv")))
           if x["axis"] != "none" and x["verdict"] == "correct"]
labels = {}
with open(os.path.join(a.out, "rawlabels", f"{a.city}_Crosswalk.csv"), encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        labels[f"{a.city}:{r['label_id']}"] = r
nodes = {}
d = os.path.join(a.out, "osm", a.city, "Crosswalk")
for fn in os.listdir(d):
    if fn.endswith(".json"):
        for el in json.load(open(os.path.join(d, fn))).get("elements", []):
            if el.get("type") == "node":
                nodes[str(el["id"])] = el

rng = random.Random(274)
rows = []
for cls in ("marked", "unmarked", "unknown"):
    pool = [x for x in offsets if x["node_marking"] == cls]
    for x in (pool if len(pool) <= a.per_class else rng.sample(pool, a.per_class)):
        lab, node = labels[x["label_uid"]], nodes[x["node_id"]]
        tags = node.get("tags", {})
        rows.append({
            "label_uid": x["label_uid"], "node_id": x["node_id"], "osm_marking_class": cls,
            "crossing_tag": tags.get("crossing", ""), "crossing_markings_tag": tags.get("crossing:markings", ""),
            "label_lat": lab["latitude"], "label_lng": lab["longitude"],
            "node_lat": node["lat"], "node_lng": node["lon"],
            "dist_m": x["dist_m"], "along_m": x["along_m"], "across_m": x["across_m"], "axis": x["axis"],
            "pano_id": lab.get("gsv_panorama_id", ""),
            "osm_node_url": f"https://www.openstreetmap.org/node/{x['node_id']}",
            "label_map_url": f"https://www.openstreetmap.org/?mlat={lab['latitude']}&mlon={lab['longitude']}#map=20/{lab['latitude']}/{lab['longitude']}",
            "ps_label_url": "",
            "rating_painted_crossing_at_node": "", "rating_label_at_centre": "", "notes": "",
        })
path = os.path.join(a.out, f"aerial_sample_{a.city}.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(path, len(rows), "rows:", {c: sum(1 for r in rows if r["osm_marking_class"] == c) for c in ("marked", "unmarked", "unknown")})
