"""Per city: OSM crossing nodes inside the crowd labels' footprint (0.01-degree cells that hold >= 1
Crosswalk label, ~1 km), vs labels, and the share of labels within 25 m of any node. Tells OSM
sparsity apart from placement disagreement."""
import csv, glob, json, math, os
r = json.load(open("out/census.json"))
rows = []
for k, v in r.items():
    city, t = k.split("/")
    if t != "Crosswalk" or "error" in v:
        continue
    labs = [x for x in csv.DictReader(open(f"out/rawlabels/{city}_Crosswalk.csv", encoding="utf-8", errors="replace")) if x.get("label_type") == "Crosswalk"]
    cells = set()
    for x in labs:
        try: cells.add((math.floor(float(x["latitude"]) / 0.01), math.floor(float(x["longitude"]) / 0.01)))
        except (KeyError, ValueError): pass
    n_in = 0
    for f in glob.glob(f"out/osm/{city}/Crosswalk/tile_*.json"):
        for el in json.load(open(f)).get("elements", []):
            if (math.floor(el["lat"] / 0.01), math.floor(el["lon"] / 0.01)) in cells:
                n_in += 1
    rows.append((city, len(labs), len(cells), n_in, round(n_in / max(1, len(cells)), 1), v["dist_any"].get("median_m"), v["dist_any"].get("le25")))
print("| city | labels | 1 km cells with labels | OSM crossing nodes in those cells | nodes per cell | median m | <=25 m |")
print("|---|---|---|---|---|---|---|")
for row in sorted(rows, key=lambda x: -x[4]):
    print("| " + " | ".join(str(c) for c in row) + " |")
