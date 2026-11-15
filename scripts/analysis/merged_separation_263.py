"""Stage 0a of the detector Phase 1 plan: the merged-pair separation histogram (#263).

A *merged* miss (``miss_taxonomy.py``, #46) is a missed ramp whose in-radius supra-threshold
peak was claimed by a neighbouring ramp: the model fired once for two adjacent ramps. This
script dumps the separation of every such pair (the missed ramp and the ramp that won the
peak), as a histogram, and evaluates the two Stage 0a triggers of the Phase 1 plan
(``detector-phase1-horizon-2x-2026-11-15.md`` section 4, written before this ran):

* **arm C** (stride-16 head) is triggered when **more than one third** of the pairs are
  under 8 px apart (the 8-px cell of the stride-32 head, #221);
* **arm A** (target sigma 10 -> 6) runs only if **at least 30 pairs outside paterson** are
  **12 px or more** apart (2 sigma for sigma 6).

Pixels are heatmap px on the 512x1024 grid (the matcher's anisotropic 1024x512 space).
``miss_taxonomy.merged_separation`` gives both Chebyshev (the extractor's
``peak_local_max`` window) and Euclidean distances. The plan's "under 8 px" figure (25 of
146) is Chebyshev; it does not name the metric for "12 px or more apart", so both
triggers are evaluated under both metrics and the doc reports them side by side.

**Pool.** The plan names the 9-split pool (``miss_decomposition.US_SPLITS``). juneau is a
held-out split of the same plan (read once, at Stage 0c), so it is NOT read here; the pool
is the other eight. Adding a split can only add pairs, so the arm A count on eight splits is
a lower bound on the nine-split count. The plan's own nine-split arm C figure (25 of 146,
17%) is quoted in the doc. The seven-split pool (S7) is reported as context and
cross-checked against the committed ``analysis_out/miss_taxonomy.json`` (124 pairs).

Reads the committed ``analysis_out/op_cache/<split>.json`` caches through
``miss_taxonomy.load_rows``: no GPU, no imagery, no network.

    python scripts/analysis/merged_separation_263.py           # -> merged_pairs.json + .md
    python scripts/analysis/merged_separation_263.py --check   # regenerate, compare bytes
"""
import argparse
import hashlib
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import miss_taxonomy as mt  # noqa: E402
from miss_decomposition import CACHE_DIR, DEFAULT_THRESHOLD  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "horizon_band_263")
OUT_JSON = os.path.join(OUT_DIR, "merged_pairs.json")
MIRROR_JSON = os.path.join(REPO, "analysis_out", "miss_taxonomy.json")

S7 = ("richmond", "annapolis", "clovis", "morgantown", "bend", "paterson", "gainesville")
#: The plan's 9-split pool minus juneau (held out until Stage 0c).
POOL = S7 + ("laurens_mapillary",)
HELD_OUT_NOT_READ = ("juneau",)
#: The plan's nine-split figures, quoted (computed by the planner and the reviewer on
#: 2026-11-15 with juneau included; not recomputed here).
PLAN_9SPLIT = {"n_pairs": 146, "under_8px_chebyshev": 25}

CELL_PX = 8.0          # the stride-32 head's cell on the 512x1024 heatmap (#221)
SIGMA_NEW = 6.0        # arm A's candidate sigma
SIGMA_OLD = 10.0
ARM_C_FRACTION = 1.0 / 3.0
ARM_A_MIN_PX = 2 * SIGMA_NEW       # 12 px
ARM_A_MIN_PAIRS = 30
ARM_A_EXCLUDED = "paterson"
BIN_PX = 1.0
METRICS = ("chebyshev", "euclidean")
ND = 4


# --------------------------------------------------------------------------- #
# pure core (unit-tested)
# --------------------------------------------------------------------------- #
def histogram(values, bin_px=BIN_PX):
    """[[lo, hi, count], ...] over [0, ceil(max)] in ``bin_px`` steps, half-open bins."""
    if not values:
        return []
    top = int(math.ceil(max(values) / bin_px)) + 1
    counts = [0] * top
    for v in values:
        counts[int(v // bin_px)] += 1
    return [[round(i * bin_px, 6), round((i + 1) * bin_px, 6), c] for i, c in enumerate(counts)]


def triggers(pairs):
    """Arm C and arm A triggers for ``pairs`` = [{"city", "chebyshev", "euclidean"}, ...].

    Returns {metric: {...}} with the counts each rule reads and its boolean verdict.
    ``under_8`` is strictly below 8 px; ``ge_12`` is 12 px or more (both as the plan
    words them)."""
    out = {}
    n = len(pairs)
    for m in METRICS:
        under = sum(1 for p in pairs if p[m] < CELL_PX)
        ge12_outside = sum(1 for p in pairs
                           if p[m] >= ARM_A_MIN_PX and p["city"] != ARM_A_EXCLUDED)
        out[m] = {
            "n_pairs": n,
            "under_8px": under,
            "under_8px_fraction": round(under / n, ND) if n else None,
            "arm_c_triggered": bool(n) and under / n > ARM_C_FRACTION,
            "ge_12px_outside_paterson": ge12_outside,
            "ge_12px_all": sum(1 for p in pairs if p[m] >= ARM_A_MIN_PX),
            "between_8_and_12px": sum(1 for p in pairs if CELL_PX <= p[m] < ARM_A_MIN_PX),
            "arm_a_runs": ge12_outside >= ARM_A_MIN_PAIRS,
        }
    return out


def per_city(pairs, cities):
    rows = {}
    for c in cities:
        ps = [p for p in pairs if p["city"] == c]
        rows[c] = {"n_pairs": len(ps)}
        for m in METRICS:
            rows[c][f"under_8px_{m}"] = sum(1 for p in ps if p[m] < CELL_PX)
            rows[c][f"ge_12px_{m}"] = sum(1 for p in ps if p[m] >= ARM_A_MIN_PX)
    return rows


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(cities, cache_dir=CACHE_DIR, threshold=DEFAULT_THRESHOLD):
    """Every merged miss with a measurable partner, in the order miss_taxonomy emits them."""
    pairs, n_merged = [], {}
    for c in cities:
        loaded = mt.load_rows(c, threshold, cache_dir, rng=None)
        if loaded is None:
            raise SystemExit(f"{c}: no op_cache at {cache_dir}")
        rows, _ = loaded
        n_merged[c] = sum(1 for r in rows if r["bucket"] == "merged")
        for r in rows:
            if r["bucket"] == "merged" and r["sep_cheb"] is not None:
                pairs.append({"city": c, "pano": r["pano"], "x": round(r["x"], 6),
                              "y": round(r["y"], 6), "dist_m": round(r["dist"], 3),
                              "chebyshev": round(r["sep_cheb"], ND),
                              "euclidean": round(r["sep_euc"], ND)})
    return pairs, n_merged


def build(cache_dir=CACHE_DIR):
    pairs, n_merged = load_pairs(POOL, cache_dir)
    s7_pairs = [p for p in pairs if p["city"] in S7]
    with open(MIRROR_JSON, encoding="utf-8") as f:
        committed = json.load(f)["merged_separation_px"]
    s7_above_md = sum(1 for p in s7_pairs if p["chebyshev"] > mt.PEAK_MIN_DISTANCE)
    cross = {"committed_miss_taxonomy_json": committed,
             "recomputed_S7": {"n": len(s7_pairs), "above_min_distance": s7_above_md},
             "agrees": committed["n"] == len(s7_pairs)
             and committed["above_min_distance"] == s7_above_md}
    assert cross["agrees"], ("S7 merged pairs no longer match analysis_out/miss_taxonomy.json",
                             cross)
    rep = {
        "protocol": {
            "plan": "detector-phase1-horizon-2x-2026-11-15.md, section 4, Stage 0a",
            "threshold": DEFAULT_THRESHOLD, "pool": list(POOL),
            "held_out_not_read": list(HELD_OUT_NOT_READ),
            "cell_px": CELL_PX, "sigma_old": SIGMA_OLD, "sigma_new": SIGMA_NEW,
            "arm_c_rule": "fraction of pairs under 8 px > 1/3",
            "arm_a_rule": "pairs outside paterson at >= 12 px >= 30",
            "units": "heatmap px on the 512x1024 grid",
            "plan_9split_quoted": PLAN_9SPLIT,
            "inputs": {c: sha256_file(os.path.join(cache_dir, f"{c}.json")) for c in POOL},
        },
        "pool_8": {"triggers": triggers(pairs),
                   "histogram": {m: histogram([p[m] for p in pairs]) for m in METRICS},
                   "per_city": per_city(pairs, POOL), "merged_misses": n_merged},
        "context_S7": {"triggers": triggers(s7_pairs)},
        "cross_check": cross,
        "pairs": pairs,
    }
    return rep


def markdown(rep):
    L = []
    t8, t7 = rep["pool_8"]["triggers"], rep["context_S7"]["triggers"]
    L.append("**Stage 0a triggers, 8-split pool (US_SPLITS minus juneau), merged pairs at 0.30**\n")
    L.append("| metric | pairs | under 8 px | share | arm C (> 1/3) | >= 12 px outside paterson "
             "| arm A (>= 30) | >= 12 px, all | 8 to 12 px |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in METRICS:
        t = t8[m]
        L.append(f"| {m} | {t['n_pairs']} | {t['under_8px']} | {t['under_8px_fraction']:.3f} | "
                 f"{'TRIGGERED' if t['arm_c_triggered'] else 'not triggered'} | "
                 f"{t['ge_12px_outside_paterson']} | {'RUNS' if t['arm_a_runs'] else 'does not run'}"
                 f" | {t['ge_12px_all']} | {t['between_8_and_12px']} |")
    L.append("")
    L.append("**Context: the seven-split pool S7 (cross-checked against "
             "`analysis_out/miss_taxonomy.json`)**\n")
    L.append("| metric | pairs | under 8 px | share | >= 12 px outside paterson |")
    L.append("|---|---|---|---|---|")
    for m in METRICS:
        t = t7[m]
        L.append(f"| {m} | {t['n_pairs']} | {t['under_8px']} | {t['under_8px_fraction']:.3f} | "
                 f"{t['ge_12px_outside_paterson']} |")
    L.append("")
    L.append("**Per split**\n")
    L.append("| split | merged misses | pairs | under 8 px (cheb / euc) | >= 12 px (cheb / euc) |")
    L.append("|---|---|---|---|---|")
    pc, nm = rep["pool_8"]["per_city"], rep["pool_8"]["merged_misses"]
    for c, r in pc.items():
        L.append(f"| {c} | {nm[c]} | {r['n_pairs']} | {r['under_8px_chebyshev']} / "
                 f"{r['under_8px_euclidean']} | {r['ge_12px_chebyshev']} / "
                 f"{r['ge_12px_euclidean']} |")
    L.append("")
    L.append("**Histogram, 1 px bins (pairs per bin)**\n")
    h = rep["pool_8"]["histogram"]
    nb = max(len(h["chebyshev"]), len(h["euclidean"]))
    L.append("| px | chebyshev | euclidean |")
    L.append("|---|---|---|")
    for i in range(nb):
        c = h["chebyshev"][i][2] if i < len(h["chebyshev"]) else 0
        e = h["euclidean"][i][2] if i < len(h["euclidean"]) else 0
        L.append(f"| {i}-{i + 1} | {c} | {e} |")
    L.append("")
    return "\n".join(L)


def outputs(cache_dir=CACHE_DIR):
    rep = build(cache_dir)
    return json.dumps(rep, indent=1) + "\n", markdown(rep) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=CACHE_DIR)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and compare bytes with --out and its .md")
    args = ap.parse_args(argv)
    js, md = outputs(args.cache)
    md_path = os.path.splitext(args.out)[0] + ".md"
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if args.check:
        bad = []
        for path, text in ((args.out, js), (md_path, md)):
            with open(path, "rb") as f:
                got = f.read()
            if got != text.encode("utf-8"):
                bad.append(f"{path}: committed sha256 {hashlib.sha256(got).hexdigest()[:12]} != "
                           f"regenerated {hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}")
        for b in bad:
            print(b)
        print("CHECK:", "FAIL" if bad else "OK (merged_pairs.json and .md reproduce byte for byte)")
        return 1 if bad else 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path, text in ((args.out, js), (md_path, md)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(md)
    print(f"-> {args.out}\n-> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
