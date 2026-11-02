"""Markdown table from out/offset_read.json plus pooled numbers from out/offsets/*.csv."""
import csv
import glob
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "out"
r = json.load(open(os.path.join(out, "offset_read.json")))
hdr = ["city", "labels", "matched <=10 m", "sampled", "axis from footway", "median |along| m", "p90 |along|",
       "median |across| m", "p90 |across|", "|along| > 3 m", "|across| > 3 m"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for c, v in sorted(r.items(), key=lambda kv: -kv[1]["labels"]):
    al, ac = v["abs_along"], v["abs_across"]
    fw = f"{100 * v['axis_footway'] / v['with_axis']:.0f}%" if v["with_axis"] else ""
    print(f"| {c} | {v['labels']} | {v['matched']} | {v['with_axis']} | {fw} | {al.get('median', '')} | {al.get('p90', '')} | "
          f"{ac.get('median', '')} | {ac.get('p90', '')} | {100 * (v['share_along_gt3'] or 0):.0f}% | {100 * (v['share_across_gt3'] or 0):.0f}% |")

# pooled, by verdict and by axis source and by node marking
rows = []
for f in glob.glob(os.path.join(out, "offsets", "*.csv")):
    rows += [x for x in csv.DictReader(open(f)) if x["axis"] != "none"]


def q(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))] if s else None


def line(name, sub):
    al = [abs(float(x["along_m"])) for x in sub]
    ac = [abs(float(x["across_m"])) for x in sub]
    sg = [float(x["along_m"]) for x in sub]
    if not sub:
        return
    print(f"| {name} | {len(sub)} | {q(al, .5):.2f} | {q(al, .9):.2f} | {q(ac, .5):.2f} | {q(ac, .9):.2f} | "
          f"{100 * sum(v > 3 for v in al) / len(al):.0f}% | {q(sg, .5):+.2f} |")


print("\n| subset | n | median \\|along\\| | p90 | median \\|across\\| | p90 | \\|along\\| > 3 m | signed along median |")
print("|---|---|---|---|---|---|---|---|")
line("all sampled", rows)
line("human-confirmed", [x for x in rows if x["verdict"] == "correct"])
line("human-rejected", [x for x in rows if x["verdict"] == "incorrect"])
line("axis = footway way", [x for x in rows if x["axis"] == "footway"])
line("axis = road normal", [x for x in rows if x["axis"] == "road"])
line("node tagged marked", [x for x in rows if x["node_marking"] == "marked"])
line("node tagged unmarked", [x for x in rows if x["node_marking"] == "unmarked"])
line("node untagged", [x for x in rows if x["node_marking"] == "unknown"])
line("on a oneway road", [x for x in rows if x["oneway_road"] == "1"])
