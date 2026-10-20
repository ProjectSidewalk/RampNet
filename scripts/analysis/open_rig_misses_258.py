"""Where do RampNet's misses live on the open-imagery rigs? (#263 Track 1, #258, #46)

RampNet loses about a third of its recall on the open rigs Project Sidewalk deployments
are moving to (#151, #159). Before anything trains on open imagery, #263 asks which kind
of miss that is. ``miss_taxonomy.py`` (#46) already splits every miss into four buckets
from the committed low-floor caches:

* **merged** -- a peak above threshold sat inside the match radius, but a neighbouring
  ramp claimed it (one peak for two ramps);
* **sub_threshold** -- a peak sat inside the radius, scoring in [0.05, threshold);
* **localization** -- no peak in the radius, an unclaimed one within 2R;
* **silent** -- nothing, even at the 0.05 floor.

The reading rule, fixed before any number was looked at (#263): if **silent** dominates,
the gap is labels (the model never responds, so it has to be taught); if
**sub_threshold** dominates, it is calibration, and the cheap verifier rungs in #258 come
before any retrain. "Dominates" is operationalized here as the pano-bootstrap 95% interval
of (sub_threshold share - silent share) excluding zero; otherwise neither leads.

This script runs the taxonomy's own ``load_rows`` (same cache, matcher, geometry and
``fn_confirmed`` gate) on the open-rig splits at 0.30 and 0.55, and on the seven US splits
#46 pooled, as the reference. It adds what a per-split reading needs and the taxonomy
does not print: Wilson and pano-bootstrap intervals on every share, the near/far split
per split, the merged-pair separations per split, how far below threshold the
sub-threshold peaks sit, and the camera models actually in each split's scored panos.

    python scripts/analysis/open_rig_misses_258.py            # print + write the JSON
    python scripts/analysis/open_rig_misses_258.py --check    # re-derive, compare bytes

CPU only, no network, no imagery; reads ``analysis_out/op_cache/<split>.json`` and
``benchmark/<split>/records.jsonl``. A split with no op_cache is written as an explicit
gap with the command that fills it, never silently dropped.
"""
import argparse
import json
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import radius_sq_for  # noqa: E402
from rampnet.validation import wilson_interval  # noqa: E402

import miss_taxonomy as mt  # noqa: E402
from miss_decomposition import CACHE_DIR, FAR_BOUNDARY_M  # noqa: E402
from operating_point_curve import read_cache  # noqa: E402

DEFAULT_JSON = os.path.join(OUT, "open_rig_misses_258.json")
THRESHOLDS = (0.30, 0.55)

# The open-imagery splits, in the order the doc reads them.
OPEN_SPLITS = ("laurens_mapillary", "bayonne", "budapest_district5", "juneau",
               "laurens_gsv")
# The #46 pool: the seven US splits pooled when the 29.0 / 38.9 / 2.1 / 30.0 table was
# made (analysis_out/miss_taxonomy.json). Not "the GSV pool": three are GSV (bend,
# paterson, gainesville) and four are Mapillary (richmond, clovis, morgantown,
# annapolis), one of which (morgantown) is itself GoPro Max.
REFERENCE_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis",
                    "paterson", "gainesville")
# Rig groups, from pano.camera_model in each split's records.jsonl (checked in the
# output's camera_models field). juneau is 120 Trimble MX7 + 5 GoPro Max panos.
GROUPS = {
    "gopro_max": ("laurens_mapillary", "bayonne", "budapest_district5"),
    "reference_7": REFERENCE_SPLITS,
}

FILL_COMMAND = {
    "laurens_gsv": ("bash scripts/analysis/open_rig_misses_258_gpu.sh  "
                    "# GPU, ~86 panos; writes analysis_out/op_cache/laurens_gsv.json"),
}

BOOT_REPS = 2000
BOOT_SEED = 20261020
ROUND = 4


# --------------------------------------------------------------------------- #
# Pure helpers -- unit-tested in tests/test_open_rig_misses_258.py
# --------------------------------------------------------------------------- #
def _r(x):
    return None if x is None else round(float(x), ROUND)


def shares_of(rows):
    """``(counts, n_miss)`` of the four buckets over the misses in ``rows``."""
    misses = [r for r in rows if not r["hit"]]
    return {b: sum(1 for r in misses if r["bucket"] == b) for b in mt.BUCKETS}, len(misses)


def bootstrap(units_by_split, reps=BOOT_REPS, seed=BOOT_SEED):
    """Pano-cluster bootstrap of each bucket's share and of (sub - silent).

    ``units_by_split`` maps a split to a list of per-pano row lists. Panos are resampled
    with replacement **within** each split (stratified), because misses cluster by pano
    -- a merged pair is two misses on one pano -- and a Wilson interval treats them as
    independent. Returns ``{bucket: (lo, hi), "sub_minus_silent": (lo, hi)}``; a
    replicate with no misses is skipped.
    """
    rng = random.Random(seed)
    draws = {b: [] for b in mt.BUCKETS}
    diff = []
    for _ in range(reps):
        counts = {b: 0 for b in mt.BUCKETS}
        n = 0
        for split in sorted(units_by_split):
            units = units_by_split[split]
            for _k in range(len(units)):
                for r in units[rng.randrange(len(units))]:
                    if not r["hit"]:
                        counts[r["bucket"]] += 1
                        n += 1
        if not n:
            continue
        for b in mt.BUCKETS:
            draws[b].append(counts[b] / n)
        diff.append((counts["sub_threshold"] - counts["silent"]) / n)

    def ci(xs):
        if not xs:
            return (None, None)
        xs = sorted(xs)
        return (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))])
    out = {b: ci(draws[b]) for b in mt.BUCKETS}
    out["sub_minus_silent"] = ci(diff)
    return out


def reading(diff_ci):
    """The #263 rule applied to the bootstrap interval of (sub - silent)."""
    lo, hi = diff_ci
    if lo is None:
        return "no misses"
    if lo > 0:
        return "sub_threshold leads: calibration"
    if hi < 0:
        return "silent leads: labels"
    return "neither leads"


def separation_stats(rows):
    """Merged-pair separations, as ``miss_taxonomy`` reports them for the pool."""
    seps = [(r["sep_cheb"], r["sep_euc"]) for r in rows
            if r["bucket"] == "merged" and r["sep_cheb"] is not None]
    n = len(seps)
    if not n:
        return {"n": 0, "chebyshev_median": None, "euclidean_median": None,
                "above_min_distance": 0, "within_2sigma": 0, "below_8px": 0}
    cheb = sorted(s[0] for s in seps)
    euc = sorted(s[1] for s in seps)
    return {"n": n, "chebyshev_median": _r(cheb[n // 2]),
            "euclidean_median": _r(euc[n // 2]),
            "above_min_distance": sum(1 for c in cheb if c > mt.PEAK_MIN_DISTANCE),
            "within_2sigma": sum(1 for e in euc if e <= 2 * mt.TARGET_SIGMA),
            "below_8px": sum(1 for c in cheb if c < 8)}


def summarize_population(rows, units_by_split, sub_scores, reps=BOOT_REPS):
    """Everything the doc's table reads for one split or pooled group."""
    counts, n_miss = shares_of(rows)
    n_gt = len(rows)
    boot = bootstrap(units_by_split, reps=reps)
    buckets = {}
    for b in mt.BUCKETS:
        k = counts[b]
        lo, hi = wilson_interval(k, n_miss) if n_miss else (None, None)
        nulls = [r["null_supra"] if b == "merged" else r["null_sub"]
                 for r in rows if not r["hit"] and r["bucket"] == b
                 and r["null_sub"] is not None]
        buckets[b] = {
            "n": k,
            "share": _r(k / n_miss) if n_miss else None,
            "wilson95": [_r(lo), _r(hi)],
            "boot95": [_r(boot[b][0]), _r(boot[b][1])],
            "recall_pts": _r(k / n_gt) if n_gt else None,
            "null_rate": (_r(sum(nulls) / len(nulls))
                          if nulls and b != "silent" else None),
        }
    fields = {}
    for f in ("near", "far"):
        fr = [r for r in rows if r["field"] == f]
        fc, fm = shares_of(fr)
        fields[f] = {"n_gt": len(fr), "n_miss": fm,
                     "recall": _r((len(fr) - fm) / len(fr)) if fr else None,
                     "counts": fc,
                     "shares": {b: (_r(fc[b] / fm) if fm else None) for b in mt.BUCKETS}}
    diff = boot["sub_minus_silent"]
    scores = sorted(sub_scores)
    return {
        "n_panos": sum(len(u) for u in units_by_split.values()),
        "n_gt": n_gt, "n_hit": n_gt - n_miss, "n_miss": n_miss,
        "recall": _r((n_gt - n_miss) / n_gt) if n_gt else None,
        "buckets": buckets,
        "sub_minus_silent_boot95": [_r(diff[0]), _r(diff[1])],
        "reading": reading(diff),
        "field": fields,
        "merged_separation_px": separation_stats(rows),
        # How far below threshold the sub-threshold misses peak: the best in-radius
        # score per miss. Near the threshold is a cheap verifier's easy case; near the
        # 0.05 floor is barely a response.
        "sub_threshold_peak": {
            "n": len(scores),
            "median": _r(statistics.median(scores)) if scores else None,
            "q25": _r(scores[len(scores) // 4]) if scores else None,
            "q75": _r(scores[(3 * len(scores)) // 4]) if scores else None,
            "share_at_or_above_0_15": (_r(sum(1 for s in scores if s >= 0.15) / len(scores))
                                       if scores else None),
        },
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def camera_models(city, panos):
    """``{camera_model: n}`` over the scored panos, from ``records.jsonl``."""
    path = os.path.join(REPO, "benchmark", city, "records.jsonl")
    want, out = set(panos), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)["pano"]
            if p["panorama_id"] in want:
                key = f"{p.get('source')}:{p.get('camera_model') or 'none'}"
                out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def best_sub_scores(city, rows, threshold, cache_dir):
    """Best in-radius peak score for each sub_threshold miss in ``rows``."""
    panos, _ = read_cache(os.path.join(cache_dir, f"{city}.json"))
    preds = {pd["pano"]: pd["preds"] for pd in panos}
    rsq = radius_sq_for()
    out = []
    for r in rows:
        if r["bucket"] != "sub_threshold":
            continue
        gs = mt.scaled((r["x"], r["y"]))
        inr = [p[2] for p in preds[r["pano"]]
               if mt._d2(mt.scaled(p), gs) < rsq and p[2] < threshold]
        out.append(max(inr))
    return out


def load_split(city, threshold, cache_dir, trials):
    """``(rows, units, sub_scores, matcher_check)`` or ``None`` with no cache.

    The null RNG is seeded per split (not shared across splits as in
    ``miss_taxonomy.main``), so a split's null rates do not depend on which other
    splits ran before it.
    """
    rng = random.Random(f"{mt.NULL_SEED}:{city}") if trials else None
    loaded = mt.load_rows(city, threshold, cache_dir, rng, trials, FAR_BOUNDARY_M)
    if loaded is None:
        return None
    rows, mc = loaded
    by_pano = {}
    for r in rows:
        by_pano.setdefault(r["pano"], []).append(r)
    return rows, list(by_pano.values()), best_sub_scores(city, rows, threshold, cache_dir), mc


def build(cache_dir=CACHE_DIR, trials=mt.NULL_TRIALS, reps=BOOT_REPS):
    splits = list(dict.fromkeys(OPEN_SPLITS + REFERENCE_SPLITS))
    result = {
        "issue": "#263 Track 1 / #258",
        "script": "scripts/analysis/open_rig_misses_258.py",
        "thresholds": list(THRESHOLDS), "boundary_m": FAR_BOUNDARY_M,
        "null_trials": trials, "boot_reps": reps, "boot_seed": BOOT_SEED,
        "reading_rule": ("silent leads -> labels; sub_threshold leads -> calibration "
                         "(cheap #258 rungs first); leads = pano-bootstrap 95% CI of "
                         "(sub_threshold share - silent share) excludes 0"),
        "open_splits": list(OPEN_SPLITS), "reference_splits": list(REFERENCE_SPLITS),
        "groups": {g: list(c) for g, c in GROUPS.items()},
        "camera_models": {}, "gaps": {}, "by_threshold": {},
    }
    for th in THRESHOLDS:
        loaded = {}
        for city in splits:
            got = load_split(city, th, cache_dir, trials)
            if got is None:
                result["gaps"][city] = {
                    "reason": f"no op_cache at analysis_out/op_cache/{city}.json",
                    "fill": FILL_COMMAND.get(city, "operating_point_curve.py extract "
                                                   f"--cities {city}")}
                continue
            loaded[city] = got
            if th == THRESHOLDS[0]:
                result["camera_models"][city] = camera_models(
                    city, [u[0]["pano"] for u in got[1]])
        block = {"splits": {}, "groups": {}}
        for city in OPEN_SPLITS:
            if city in loaded:
                rows, units, subs, mc = loaded[city]
                s = summarize_population(rows, {city: units}, subs, reps)
                s["matcher_check"] = mc
                block["splits"][city] = s
        for g, members in GROUPS.items():
            have = [c for c in members if c in loaded]
            if not have:
                continue
            rows = [r for c in have for r in loaded[c][0]]
            subs = [x for c in have for x in loaded[c][2]]
            s = summarize_population(rows, {c: loaded[c][1] for c in have}, subs, reps)
            s["members"] = have
            block["groups"][g] = s
        # Recall points each bucket costs beyond what it costs the reference pool:
        # shares hide that a split can have the same mix and twice the misses.
        ref = block["groups"].get("reference_7")
        if ref:
            for s in list(block["splits"].values()) + [
                    v for k, v in block["groups"].items() if k != "reference_7"]:
                s["excess_recall_pts_vs_reference"] = {
                    b: _r(s["buckets"][b]["recall_pts"] - ref["buckets"][b]["recall_pts"])
                    for b in mt.BUCKETS}
                s["excess_recall_pts_vs_reference"]["total"] = _r(ref["recall"] - s["recall"])
        result["by_threshold"][f"{th:.2f}"] = block
    return result


def dumps(result):
    return json.dumps(result, indent=1, ensure_ascii=False) + "\n"


def print_table(result):
    for th, block in result["by_threshold"].items():
        print(f"\n### threshold {th}\n")
        print("| population | GT | misses | recall | merged | sub_threshold | "
              "localization | silent | sub - silent (boot 95%) | reading |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
        pops = list(block["splits"].items()) + [
            (f"**{g}** ({len(s['members'])} splits)", s) for g, s in block["groups"].items()]
        for name, s in pops:
            cells = []
            for b in mt.BUCKETS:
                v = s["buckets"][b]
                cells.append(f"{v['n']} = {100*v['share']:.1f}% "
                             f"[{100*v['wilson95'][0]:.0f}, {100*v['wilson95'][1]:.0f}]")
            d = s["sub_minus_silent_boot95"]
            print(f"| {name} | {s['n_gt']} | {s['n_miss']} | {s['recall']:.3f} | "
                  + " | ".join(cells)
                  + f" | [{100*d[0]:+.0f}, {100*d[1]:+.0f}] | {s['reading']} |")
    if result["gaps"]:
        print("\nGaps:")
        for c, g in result["gaps"].items():
            print(f"  {c}: {g['reason']}; fill with: {g['fill']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--json-out", default=DEFAULT_JSON)
    p.add_argument("--check", action="store_true",
                   help="Re-derive and compare byte-for-byte with --json-out; "
                        "exit 1 on any difference.")
    args = p.parse_args(argv)

    text = dumps(build(args.cache))
    if args.check:
        with open(args.json_out, encoding="utf-8", newline="") as f:
            have = f.read()
        if have != text:
            print(f"CHECK FAILED: {args.json_out} does not match a fresh derivation")
            return 1
        print(f"check ok: {args.json_out} re-derives byte-for-byte")
        return 0
    print_table(json.loads(text))
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
