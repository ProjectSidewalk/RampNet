"""Item 1 Readings 2-4 from OUT/census.json and OUT/per_label/<city>_<type>.csv (`confirmed_table.py OUT`).

Reading 2: in the dense group (>= 100 Crosswalk labels and a median <= 5 m to the nearest OSM crossing
node), human-confirmed labels (the census's margin rule) within 10 m of any node and of a marked node, per
city for the cities with >= 40 confirmed labels, and pooled over them and over the whole group.
Reading 3: the share of confirmed labels within 10 m of a node whose nearest node within 10 m is not marked,
pooled, without Chicago, and per city.
Reading 4: Signal labels, median distance and share within 10 m, over the dense-group cities with >= 100
Signal labels and over every city with >= 100.
"""
import csv
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MIN_CONFIRMED_IN_TABLE = 40
r = json.load(open(os.path.join(OUT, "census.json"), encoding="utf-8"))


def cw(city):
    return r.get(f"{city}/Crosswalk", {})


cities = sorted({k.split("/")[0] for k in r})
dense = [c for c in cities if "error" not in cw(c) and cw(c).get("labels", 0) >= 100
         and (cw(c).get("dist_any") or {}).get("median_m") is not None and cw(c)["dist_any"]["median_m"] <= 5]
print(f"dense group: {len(dense)} cities: {', '.join(dense)}\n")

rows = {}
for c in dense:
    with open(os.path.join(OUT, "per_label", f"{c}_Crosswalk.csv"), encoding="utf-8", newline="") as f:
        conf = [x for x in csv.DictReader(f) if x["verdict"] == "correct"]
    n = len(conf)
    a = sum(1 for x in conf if x["d_any_m"] and float(x["d_any_m"]) <= 10)
    m = sum(1 for x in conf if x["d_marked_m"] and float(x["d_marked_m"]) <= 10)
    rows[c] = (n, a, m)

print("Reading 2: human-confirmed Crosswalk labels")
print("| city | confirmed labels | <= 10 m any node | <= 10 m marked node |")
print("|---|---|---|---|")
table = [c for c in dense if rows[c][0] >= MIN_CONFIRMED_IN_TABLE]
for c in sorted(table, key=lambda c: -rows[c][0]):
    n, a, m = rows[c]
    print(f"| {c} | {n} | {a / n:.0%} | {m / n:.0%} |")
n, a, m = (sum(rows[c][i] for c in table) for i in range(3))
print(f"| **pooled, {len(table)} cities with >= {MIN_CONFIRMED_IN_TABLE} confirmed** | **{n}** | **{a / n:.0%}** | **{m / n:.0%}** |")
n17, a17, m17 = (sum(rows[c][i] for c in dense) for i in range(3))
print(f"| pooled, all {len(dense)} | {n17} | {a17 / n17:.0%} | {m17 / n17:.0%} |\n")

print("Reading 3: of confirmed labels within 10 m of a node, the share whose node is unmarked or untagged")
print(f"  pooled over the {len(dense)}: {(a17 - m17) / a17:.1%}  ({a17 - m17} of {a17})")
nc = [c for c in dense if c != "chicago-il"]
anc, mnc = sum(rows[c][1] for c in nc), sum(rows[c][2] for c in nc)
print(f"  without Chicago: {(anc - mnc) / anc:.1%}  (1 in {anc / (anc - mnc):.1f})")
for c in sorted(dense, key=lambda c: -rows[c][0]):
    n, a, m = rows[c]
    if a:
        print(f"  {c:18} {(a - m) / a:4.0%} of {a} (confirmed {n})")
print()

print("Reading 4: Signal labels vs OSM signal nodes, cities with >= 100 Signal labels")
sig = {k.split("/")[0]: v for k, v in r.items() if k.endswith("/Signal") and "error" not in v and v.get("labels", 0) >= 100}
for name, group in (("dense group", [c for c in sig if c in dense]), ("all", list(sig))):
    med = [sig[c]["dist_any"]["median_m"] for c in group]
    le = [sig[c]["dist_any"]["le10"] for c in group]
    lo, hi = min(group, key=lambda c: sig[c]["dist_any"]["le10"]), max(group, key=lambda c: sig[c]["dist_any"]["le10"])
    print(f"  {name}: {len(group)} cities, median {min(med):.1f}-{max(med):.1f} m, within 10 m {min(le):.0%}-{max(le):.0%} ({lo} to {hi})")
