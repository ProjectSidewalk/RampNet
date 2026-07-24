"""Final recall analysis using DA3 metric depth (replaces the flat-ground proxy)."""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import json, math, os, statistics as st

RAMP_W = 1.2
PX_PER_RAD = 4096.0 / (2 * math.pi)
rows = json.load(open(os.path.join(OUT, "gt_depth_da3.json")))
rows = [r for r in rows if r.get("depth")]

PX_BUCKETS = [(0, 12), (12, 20), (20, 32), (32, 50), (50, 80), (80, 1e9)]
M_BUCKETS = [(0, 8), (8, 12), (12, 18), (18, 25), (25, 40), (40, 1e9)]


def px_of(r):
    return RAMP_W / r["depth"] * PX_PER_RAD


def spearman(pairs):
    n = len(pairs)
    if n < 3:
        return float("nan")
    a = sorted(range(n), key=lambda i: pairs[i][0])
    b = sorted(range(n), key=lambda i: pairs[i][1])
    ra = {v: i for i, v in enumerate(a)}; rb = {v: i for i, v in enumerate(b)}
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


print("=" * 74)
print("1) VALIDATION — DA3 metric depth vs flat-ground geometry")
print("=" * 74)
for city in ("bend", "richmond"):
    r = [x for x in rows if x["city"] == city]
    fin = [x for x in r if x["geom"] != float("inf") and x["geom"] < 150]
    ratios = [x["geom"] / x["depth"] for x in fin if x["depth"] > 0.5]
    rho = spearman([(x["geom"], x["depth"]) for x in fin])
    inf_n = sum(1 for x in r if x["geom"] == float("inf") or x["geom"] >= 150)
    note = "(level GSV mast -> geometry trustworthy)" if city == "bend" else "(Mapillary, variable rig)"
    print(f"\n{city} {note}   n={len(r)}")
    print(f"   median geom/depth ratio : {st.median(ratios):.3f}   (1.0 = agreement)")
    print(f"   Spearman rho            : {rho:.3f}")
    print(f"   geometry UNUSABLE (above-horizon / >150m): {inf_n}  -> now measured by depth")
    if inf_n:
        d = [x["depth"] for x in r if x["geom"] == float("inf") or x["geom"] >= 150]
        print(f"      depth for those: median {st.median(d):.1f} m, range {min(d):.1f}-{max(d):.1f}")

print("\n" + "=" * 74)
print("2) RECALL vs TRUE DISTANCE (DA3 metric depth)")
print("=" * 74)
print(f"{'distance':>14} {'n':>6} {'detected':>9} {'RECALL':>8}")
print("-" * 74)
for lo, hi in M_BUCKETS:
    b = [r for r in rows if lo <= r["depth"] < hi]
    if not b:
        continue
    hit = sum(1 for r in b if r["hit"])
    lab = f"{lo}-{int(hi)}m" if hi < 1e8 else f"{lo}m+"
    print(f"{lab:>14} {len(b):>6} {hit:>9} {hit/len(b):>8.3f}")
print(f"{'ALL':>14} {len(rows):>6} {sum(1 for r in rows if r['hit']):>9} "
      f"{sum(1 for r in rows if r['hit'])/len(rows):>8.3f}")

print("\n" + "=" * 74)
print("3) RECALL vs APPARENT SIZE (from depth) — and the resolution forecast")
print("=" * 74)
print(f"{'apparent size':>16} {'~dist':>8} {'n':>6} {'detected':>9} {'RECALL':>8}")
print("-" * 74)
recall_by = {}
for lo, hi in PX_BUCKETS:
    b = [r for r in rows if lo <= px_of(r) < hi]
    if not b:
        continue
    hit = sum(1 for r in b if r["hit"])
    recall_by[(lo, hi)] = hit / len(b)
    md = st.median([r["depth"] for r in b])
    lab = f"{lo}-{int(hi)}px" if hi < 1e8 else f"{lo}px+"
    print(f"{lab:>16} {md:>7.0f}m {len(b):>6} {hit:>9} {hit/len(b):>8.3f}")


def recall_at(px):
    for (lo, hi), v in recall_by.items():
        if lo <= px < hi:
            return v
    return max(recall_by.values())


base = sum(1 for r in rows if r["hit"]) / len(rows)
for factor in (1.5, 2.0, 3.0):
    pred = sum(recall_at(px_of(r) * factor) for r in rows) / len(rows)
    print(f"\n  forecast @ {factor}x linear resolution: recall {base:.3f} -> {pred:.3f} "
          f"({pred-base:+.3f})")
print("\n  (assumes native imagery actually holds the extra detail: Richmond median 11000px,"
      "\n   Bend 16384px vs the 4096px model input — so 2.7-4x is genuinely available.)")
