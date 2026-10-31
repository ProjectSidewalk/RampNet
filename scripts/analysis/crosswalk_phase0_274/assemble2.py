"""Fill post2_tail.md placeholders from the offsets CSVs and assemble post2.md."""
import csv
import glob
import os
import subprocess
import sys

out = "out"
rows = []
for f in glob.glob(os.path.join(out, "offsets", "*.csv")):
    rows += [x for x in csv.DictReader(open(f)) if x["axis"] != "none"]


def q(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def st(sub):
    al = [abs(float(x["along_m"])) for x in sub]
    ac = [abs(float(x["across_m"])) for x in sub]
    sg = [float(x["along_m"]) for x in sub]
    return {"n": f"{len(sub):,}", "along": f"{q(al, .5):.2f}", "across": f"{q(ac, .5):.2f}",
            "signed": f"{q(sg, .5):+.2f}", "gt3": f"{100 * sum(v > 3 for v in al) / len(al):.0f}%",
            "p90al": f"{q(al, .9):.1f}", "p90ac": f"{q(ac, .9):.1f}"}


A = st(rows)
OK = st([x for x in rows if x["verdict"] == "correct"])
OW = st([x for x in rows if x["oneway_road"] == "1"])
RD = st([x for x in rows if x["axis"] == "road"])
FW = st([x for x in rows if x["axis"] == "footway"])
UN = st([x for x in rows if x["node_marking"] == "unmarked"])
MK = st([x for x in rows if x["node_marking"] == "marked"])
sheet = sum(1 for _ in csv.DictReader(open(os.path.join(out, "aerial_sample_seattle-wa.csv"), encoding="utf-8")))
sub = {"N_ALL": A["n"], "ALONG_ALL": A["along"], "ACROSS_ALL": A["across"], "SIGNED_ALL": A["signed"], "GT3_ALL": A["gt3"],
       "N_OK": OK["n"], "ALONG_OK": OK["along"], "ACROSS_OK": OK["across"],
       "N_ONEWAY": OW["n"], "ALONG_ONEWAY": OW["along"], "ACROSS_ONEWAY": OW["across"],
       "ALONG_ROAD": RD["along"], "ALONG_FW": FW["along"],
       "N_UNM": UN["n"], "ALONG_UNM": UN["along"], "ALONG_MARKED": MK["along"],
       "P90_ALONG": A["p90al"], "P90_ACROSS": A["p90ac"], "N_SHEET": str(sheet)}
tail = open("post2_tail.md", encoding="utf-8").read()
for k, v in sub.items():
    tail = tail.replace("{" + k + "}", v)
assert "{" not in tail.split("🤖")[0].replace("{N", ""), [l for l in tail.splitlines() if "{" in l]
table = subprocess.run([sys.executable, "-I", "offset_table.py", out], capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
per_city, pooled = table.split("\n\n", 1)
body = open("post2_head.md", encoding="utf-8").read() + per_city + "\n\n### Pooled cuts\n\n" + pooled + tail
open("post2.md", "w", encoding="utf-8").write(body)
print("post2.md", len(body), "chars;", sub)
