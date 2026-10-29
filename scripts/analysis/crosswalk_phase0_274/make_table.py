"""Build the markdown census table from out/census.json."""
import json, sys
r = json.load(open("out/census.json"))
cities = sorted({k.split("/")[0] for k in r})
def f(x, d=0):
    return "" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def pct(x):
    return "" if x is None else f"{100*x:.0f}%"
rows = []
for c in cities:
    cw, sg = r.get(f"{c}/Crosswalk", {}), r.get(f"{c}/Signal", {})
    if "error" in cw:
        rows.append((c, "ERROR", "", "", "", "", "", "", "", "", "", "")); continue
    n = cw.get("osm_nodes", 0)
    marked = cw.get("osm_marked", 0); unm = cw.get("osm_unmarked", 0)
    tagged = marked + unm
    da, dm = cw.get("dist_any", {}), cw.get("dist_marked", {})
    sa = sg.get("dist_any", {}) if "error" not in sg else {}
    rows.append((c, cw.get("labels", 0), f"{cw.get('labels_correct',0)}/{cw.get('labels_incorrect',0)}",
                 n, pct(marked / n) if n else "", pct(tagged / n) if n else "",
                 f(da.get("median_m"), 1), pct(da.get("le10")), pct(dm.get("le10")),
                 sg.get("labels", 0) if sg else "", f(sa.get("median_m"), 1), pct(sa.get("le10"))))
hdr = ["city", "Crosswalk labels", "human ok/wrong", "OSM crossing nodes", "marked", "tagged either way",
       "median m to any node", "<=10 m any", "<=10 m marked", "Signal labels", "median m to signal node", "<=10 m"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for row in sorted(rows, key=lambda x: -(x[1] if isinstance(x[1], int) else -1)):
    print("| " + " | ".join(str(v) for v in row) + " |")
# pooled, label-weighted
tot = sum(v.get("labels", 0) for k, v in r.items() if k.endswith("/Crosswalk") and "error" not in v)
print(f"\nTotal Crosswalk labels: {tot}; Signal: {sum(v.get('labels',0) for k,v in r.items() if k.endswith('/Signal') and 'error' not in v)}")
