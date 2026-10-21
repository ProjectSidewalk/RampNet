"""Where do RampNet's misses live on the open-imagery rigs? (#263 Track 1, #258, #46)

RampNet loses about a third of its recall on the open rigs Project Sidewalk deployments
are moving to (#151, #159). Before anything trains on open imagery, #263 asks which kind
of miss that is. ``miss_taxonomy.py`` (#46) splits every miss into four buckets from the
committed low-floor caches (``classify_miss``, used unchanged here):

* **merged** -- a peak at or above threshold sat inside the match radius R, but a
  neighbouring ramp claimed it (one peak for two ramps);
* **sub_threshold** -- a peak sat inside R, scoring in [0.05, threshold);
* **localization** -- no peak inside R, and the nearest at-or-above-threshold peak
  between R and 2R is unclaimed;
* **silent** -- everything else: no floor peak inside R, and no unclaimed
  at-or-above-threshold peak within 2R. A silent ramp can still have a *sub-threshold*
  peak between R and 2R, or a neighbour's peak there.

This script adds a stricter count beside ``silent``, **strict_silent**: no floor peak
(score >= 0.05) anywhere within 2R. It is computed here from the cache and does not
change ``classify_miss``.

The reading rule is #263's: if silent dominates, the gap is labels; if sub_threshold
dominates, it is calibration, and the cheap verifier rungs in #258 come before any
retrain. #263 did not say what "dominates" means; this script makes it the pano-bootstrap
95% interval of (sub_threshold share - silent share) excluding zero, and otherwise
"neither leads". See docs/open_rig_misses_258.md for when that test was chosen.

Populations: each open split on its own; the GoPro Max pool (laurens_mapillary +
bayonne -- budapest_district5 is GoPro Max too but is HELD_OUT for low-confidence
single-rater GT, so it is reported in its own row and never pooled); morgantown (GoPro Max
inside the reference); the #46 reference (seven US splits) and the same without paterson.
Every population is computed on all panos and on the **unbiased** subset (verdicts
``group != "top"``, i.e. random + empty strata), which benchmark/README.md names as the
between-city comparison.

    python scripts/analysis/open_rig_misses_258.py               # write JSON + doc tables
    python scripts/analysis/open_rig_misses_258.py --check       # re-derive, compare bytes

CPU only, no network, no imagery; reads ``analysis_out/op_cache/<split>.json``,
``benchmark/<split>/records.jsonl`` and ``benchmark/<split>/verdicts.json``. A split with
no op_cache is written as an explicit gap with the command that fills it, never dropped.
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import sys

import numpy as np

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
DEFAULT_DOC = os.path.join(REPO, "docs", "open_rig_misses_258.md")
THRESHOLDS = (0.30, 0.55)
SUBSETS = ("all", "unbiased")

# The open-imagery splits, in the order the doc reads them.
OPEN_SPLITS = ("laurens_mapillary", "bayonne", "budapest_district5", "juneau",
               "laurens_gsv")
# The #46 pool: the seven US splits pooled when the 29.0 / 38.9 / 2.1 / 30.0 table was
# made (analysis_out/miss_taxonomy.json). Not "the GSV pool": three are GSV (bend,
# paterson, gainesville) and four are Mapillary (richmond, clovis, morgantown,
# annapolis). Two of the four carry GoPro Max: morgantown (95 of 95 scored panos) and
# richmond (18 of 92, beside 64 iSTAR Pulsar).
REFERENCE_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis",
                    "paterson", "gainesville")
# Pooled groups, from pano.camera_model in each split's records.jsonl (checked in the
# output's camera_models field). budapest_district5 is GoPro Max (89 of 89) but is
# HELD_OUT in miss_decomposition for low-confidence single-rater GT, so it is never
# pooled. juneau is 92 Trimble MX7 + 2 GoPro Max scored panos.
GROUPS = {
    "gopro_max": ("laurens_mapillary", "bayonne"),
    "morgantown_gopro_max": ("morgantown",),
    # Paterson alone holds 80 of the reference's 124 merged misses at 0.30.
    "reference_no_paterson": tuple(c for c in REFERENCE_SPLITS if c != "paterson"),
    "reference_7": REFERENCE_SPLITS,
}
REFERENCES = ("reference_7", "reference_no_paterson")

FILL_COMMAND = {
    "laurens_gsv": ("bash scripts/analysis/open_rig_misses_258_gpu.sh  "
                    "# GPU, 86 panos; writes analysis_out/op_cache/laurens_gsv.json"),
}

BOOT_REPS = 2000
BOOT_SEED = 20261020
ROUND = 4
STRICT_FACTOR = 2.0       # strict_silent: no floor peak within this many radii

LABEL = {
    "laurens_mapillary": "`laurens_mapillary`",
    "bayonne": "`bayonne`",
    "budapest_district5": "`budapest_district5` (held out, not pooled)",
    "juneau": "`juneau`",
    "laurens_gsv": "`laurens_gsv` (held out; under-scored cache, see Gaps)",
    "gopro_max": "**GoPro Max pool** (laurens_mapillary + bayonne)",
    "morgantown_gopro_max": "`morgantown` (GoPro Max, in the reference)",
    "reference_no_paterson": "#46 reference without paterson (6 splits)",
    "reference_7": "**#46 reference** (7 US splits)",
}


# --------------------------------------------------------------------------- #
# Pure helpers -- unit-tested in tests/test_open_rig_misses_258.py
# --------------------------------------------------------------------------- #
def _r(x):
    return None if x is None else round(float(x), ROUND)


def _ci(xs):
    if not xs:
        return (None, None)
    xs = sorted(xs)
    return (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))])


def pano_vector(rows):
    """``(n_gt, merged, sub, loc, silent, strict)`` for one pano's rows."""
    v = [len(rows), 0, 0, 0, 0, 0]
    for r in rows:
        if r["hit"]:
            continue
        v[1 + mt.BUCKETS.index(r["bucket"])] += 1
        if r.get("strict_silent"):
            v[5] += 1
    return tuple(v)


def _draw(units_by_split, rng):
    """One stratified pano resample: summed ``pano_vector``s (integers, so exact).

    ``rng.choices`` draws ``floor(random() * n)``, the same stream on every Python
    version; numpy only does the integer sums.
    """
    tot = np.zeros(6, dtype=np.int64)
    for split in sorted(units_by_split):
        units = units_by_split[split]
        n = len(units)
        if not n:
            continue
        arr = units if isinstance(units, np.ndarray) else np.asarray(units, dtype=np.int64)
        tot += arr[rng.choices(range(n), k=n)].sum(axis=0)
    return [int(x) for x in tot]


def bootstrap(units_by_split, reps=BOOT_REPS, seed=BOOT_SEED):
    """Pano-cluster bootstrap of each bucket's share and of (sub - silent).

    ``units_by_split`` maps a split to a list of ``pano_vector`` tuples. Panos are
    resampled with replacement **within** each split (stratified), because misses
    cluster by pano -- a merged pair is two misses on one pano -- and a Wilson interval
    treats them as independent. Returns ``{bucket: (lo, hi), "strict_silent": (lo, hi),
    "sub_minus_silent": (lo, hi)}``; a replicate with no misses is skipped.
    """
    rng = random.Random(seed)
    draws = {b: [] for b in mt.BUCKETS}
    draws["strict_silent"] = []
    diff = []
    for _ in range(reps):
        tot = _draw(units_by_split, rng)
        n = sum(tot[1:5])
        if not n:
            continue
        for i, b in enumerate(mt.BUCKETS):
            draws[b].append(tot[1 + i] / n)
        draws["strict_silent"].append(tot[5] / n)
        diff.append((tot[2] - tot[4]) / n)
    out = {k: _ci(v) for k, v in draws.items()}
    out["sub_minus_silent"] = _ci(diff)
    return out


def bootstrap_excess(pop_units, ref_units, reps=BOOT_REPS, seed=BOOT_SEED):
    """Pano bootstrap of the recall-point excess of ``pop`` over ``ref``, per bucket.

    Excess for a bucket is (its misses / GT in pop) - (its misses / GT in ref). Both
    populations are resampled independently, each stratified by split. Also returns the
    total recall gap and (sub_threshold excess - silent excess), the interval that says
    whether the two levers are distinguishable in size.
    """
    rng = random.Random(seed)
    keys = mt.BUCKETS + ("strict_silent", "total", "sub_minus_silent_excess")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        p = _draw(pop_units, rng)
        q = _draw(ref_units, rng)
        if not p[0] or not q[0]:
            continue
        ex = [p[i] / p[0] - q[i] / q[0] for i in range(1, 6)]
        for i, b in enumerate(mt.BUCKETS):
            draws[b].append(ex[i])
        draws["strict_silent"].append(ex[4])
        draws["total"].append(sum(ex[:4]))
        draws["sub_minus_silent_excess"].append(ex[1] - ex[3])
    return {k: _ci(v) for k, v in draws.items()}


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


def _field_block(rows):
    misses = [r for r in rows if not r["hit"]]
    fc = {b: sum(1 for r in misses if r["bucket"] == b) for b in mt.BUCKETS}
    fm = len(misses)
    return {"n_gt": len(rows), "n_miss": fm,
            "recall": _r((len(rows) - fm) / len(rows)) if rows else None,
            "counts": fc,
            "strict_silent": sum(1 for r in misses if r.get("strict_silent")),
            "shares": {b: (_r(fc[b] / fm) if fm else None) for b in mt.BUCKETS}}


def summarize_population(rows, units_by_split, sub_scores, reps=BOOT_REPS):
    """Everything the doc's tables read for one split or pooled group.

    ``units_by_split`` maps a split to its ``pano_vector`` tuples.
    """
    misses = [r for r in rows if not r["hit"]]
    counts = {b: sum(1 for r in misses if r["bucket"] == b) for b in mt.BUCKETS}
    n_miss, n_gt = len(misses), len(rows)
    n_strict = sum(1 for r in misses if r.get("strict_silent"))
    boot = bootstrap(units_by_split, reps=reps)
    buckets = {}
    for b in mt.BUCKETS + ("strict_silent",):
        k = n_strict if b == "strict_silent" else counts[b]
        lo, hi = wilson_interval(k, n_miss) if n_miss else (None, None)
        nulls = [r["null_supra"] if b == "merged" else r["null_sub"]
                 for r in misses if r["bucket"] == b and r["null_sub"] is not None]
        buckets[b] = {
            "n": k,
            "share": _r(k / n_miss) if n_miss else None,
            "wilson95": [_r(lo), _r(hi)],
            "boot95": [_r(boot[b][0]), _r(boot[b][1])],
            "recall_pts": _r(k / n_gt) if n_gt else None,
            # math.fsum, not sum: Python 3.12 made sum() of floats compensated, so the
            # two disagree in the last bits and a 4-dp rounding flips between CI's
            # Python 3.10 and 3.12 legs. fsum is exact on every version.
            "null_rate": (_r(math.fsum(nulls) / len(nulls))
                          if nulls and b in ("merged", "sub_threshold", "localization")
                          else None),
        }
    diff = boot["sub_minus_silent"]
    scores = sorted(sub_scores)
    return {
        "n_panos": sum(len(u) for u in units_by_split.values()),
        "n_gt": n_gt, "n_hit": n_gt - n_miss, "n_miss": n_miss,
        "recall": _r((n_gt - n_miss) / n_gt) if n_gt else None,
        "buckets": buckets,
        "sub_minus_silent_boot95": [_r(diff[0]), _r(diff[1])],
        "reading": reading(diff),
        "field": {f: _field_block([r for r in rows if r["field"] == f])
                  for f in ("near", "far")},
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


def excess_block(pop, ref, pop_units, ref_units, reps=BOOT_REPS):
    """Point excess (pop recall points - ref recall points) with bootstrap intervals."""
    boot = bootstrap_excess(pop_units, ref_units, reps=reps)
    out = {}
    for b in mt.BUCKETS + ("strict_silent",):
        out[b] = {"pts": _r(pop["buckets"][b]["recall_pts"] - ref["buckets"][b]["recall_pts"]),
                  "boot95": [_r(boot[b][0]), _r(boot[b][1])]}
    out["total"] = {"pts": _r(ref["recall"] - pop["recall"]),
                    "boot95": [_r(boot["total"][0]), _r(boot["total"][1])]}
    d = boot["sub_minus_silent_excess"]
    out["sub_minus_silent_excess"] = {
        "pts": _r(out["sub_threshold"]["pts"] - out["silent"]["pts"]),
        "boot95": [_r(d[0]), _r(d[1])]}
    return out


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def camera_models(city, panos):
    """``{source:camera_model: n}`` over the scored panos, from ``records.jsonl``."""
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


def top_panos(city):
    """Panos in the hand-picked ``top`` stratum (``verdicts.json`` ``group``).

    The same field ``rampnet.validation.collect(exclude_top=True)`` drops for the
    unbiased estimate that ``score_validation.py`` prints second.
    """
    with open(os.path.join(REPO, "benchmark", city, "verdicts.json"), encoding="utf-8") as f:
        panos = json.load(f)["panos"]
    return {pid for pid, e in panos.items() if e.get("group") == "top"}


def annotate(city, rows, threshold, cache_dir):
    """Add ``strict_silent`` to silent rows; return each sub_threshold miss's best score."""
    panos, _ = read_cache(os.path.join(cache_dir, f"{city}.json"))
    preds = {pd["pano"]: pd["preds"] for pd in panos}
    rsq = radius_sq_for()
    outer = rsq * STRICT_FACTOR * STRICT_FACTOR
    subs = []
    for r in rows:
        if r["hit"]:
            continue
        gs = mt.scaled((r["x"], r["y"]))
        d2 = [(mt._d2(mt.scaled(p), gs), p[2]) for p in preds[r["pano"]]]
        if r["bucket"] == "silent":
            r["strict_silent"] = not any(d < outer for d, _ in d2)
        elif r["bucket"] == "sub_threshold":
            subs.append(max(s for d, s in d2 if d < rsq and s < threshold))
    return subs


def load_split(city, threshold, cache_dir, trials):
    """``(rows, matcher_check)`` or ``None`` with no cache.

    The null RNG is seeded per split (not shared across splits as in
    ``miss_taxonomy.main``), so a split's null rates do not depend on which other
    splits ran before it. Each row carries ``top`` (hand-picked stratum) and, for a
    sub_threshold miss, ``sub_score`` (best in-radius peak).
    """
    rng = random.Random(f"{mt.NULL_SEED}:{city}") if trials else None
    loaded = mt.load_rows(city, threshold, cache_dir, rng, trials, FAR_BOUNDARY_M)
    if loaded is None:
        return None
    rows, mc = loaded
    subs = iter(annotate(city, rows, threshold, cache_dir))
    top = top_panos(city)
    for r in rows:
        r["top"] = r["pano"] in top
        if not r["hit"] and r["bucket"] == "sub_threshold":
            r["sub_score"] = next(subs)
    return rows, mc


def _units(rows):
    by_pano = {}
    for r in rows:
        by_pano.setdefault(r["pano"], []).append(r)
    return np.asarray([pano_vector(v) for v in by_pano.values()],
                      dtype=np.int64).reshape(-1, 6)


def build(cache_dir=CACHE_DIR, trials=mt.NULL_TRIALS, reps=BOOT_REPS):
    splits = list(dict.fromkeys(OPEN_SPLITS + REFERENCE_SPLITS))
    result = {
        "issue": "#263 Track 1 / #258",
        "script": "scripts/analysis/open_rig_misses_258.py",
        "thresholds": list(THRESHOLDS), "subsets": list(SUBSETS),
        "boundary_m": FAR_BOUNDARY_M, "strict_silent_radii": STRICT_FACTOR,
        "null_trials": trials, "boot_reps": reps, "boot_seed": BOOT_SEED,
        "reading_rule": ("silent leads -> labels; sub_threshold leads -> calibration "
                         "(cheap #258 rungs first); leads = pano-bootstrap 95% CI of "
                         "(sub_threshold share - silent share) excludes 0"),
        "open_splits": list(OPEN_SPLITS), "reference_splits": list(REFERENCE_SPLITS),
        "groups": {g: list(c) for g, c in GROUPS.items()},
        "camera_models": {}, "gaps": {}, "by_subset": {s: {} for s in SUBSETS},
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
            loaded[city] = got[0]
            if th == THRESHOLDS[0]:
                result["camera_models"][city] = camera_models(
                    city, sorted({r["pano"] for r in got[0]}))
        for subset in SUBSETS:
            rows_of = {c: ([r for r in rs if not r["top"]] if subset == "unbiased" else rs)
                       for c, rs in loaded.items()}
            units_of = {c: _units(rs) for c, rs in rows_of.items()}

            def pop(members):
                rows = [r for c in members for r in rows_of[c]]
                subs = [r["sub_score"] for r in rows if "sub_score" in r]
                return summarize_population(rows, {c: units_of[c] for c in members},
                                            subs, reps)

            block = {"splits": {}, "groups": {}}
            for city in OPEN_SPLITS:
                if city in loaded:
                    block["splits"][city] = pop((city,))
            members_of = {}
            for g, members in GROUPS.items():
                have = tuple(c for c in members if c in loaded)
                if have:
                    block["groups"][g] = pop(have)
                    block["groups"][g]["members"] = list(have)
                    members_of[g] = have
            # Recall points each bucket costs beyond what it costs a reference pool,
            # with a pano bootstrap: shares hide that a split can have the same mix and
            # three times the misses.
            for ref_name in REFERENCES:
                if ref_name not in block["groups"]:
                    continue
                ref = block["groups"][ref_name]
                ref_units = {c: units_of[c] for c in members_of[ref_name]}
                targets = [((c,), block["splits"][c]) for c in block["splits"]] + [
                    (members_of[g], block["groups"][g]) for g in block["groups"]
                    if g not in REFERENCES]
                for members, s in targets:
                    s.setdefault("excess_vs", {})[ref_name] = excess_block(
                        s, ref, {c: units_of[c] for c in members}, ref_units, reps)
            result["by_subset"][subset][f"{th:.2f}"] = block
    return result


def dumps(result):
    return json.dumps(result, indent=1, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# Doc tables: generated from the JSON, so the doc cannot drift from it
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{100 * x:.1f}%"


def _signed(x, fmt):
    """``x`` with an explicit sign, the typographic minus (U+2212) the prose uses, and
    no sign at all on an exact zero -- so a bound of exactly 0 does not print as "+0.0"
    beside a "neither leads" that it caused (the rule needs a bound strictly above 0)."""
    if x == 0:
        return format(0.0, fmt)
    return ("+" if x > 0 else "−") + format(abs(x), fmt)


def _pts(lohi, digits=1):
    """A share-difference interval in percentage points."""
    lo, hi = lohi
    return f"{_signed(100 * lo, f'.{digits}f')} to {_signed(100 * hi, f'.{digits}f')}"


def _frac(lohi):
    """A recall-point interval, in the same fraction units as its point estimate."""
    lo, hi = lohi
    return f"{_signed(lo, '.3f')} to {_signed(hi, '.3f')}"


def _populations(block, include_refs=True):
    names = [c for c in OPEN_SPLITS if c in block["splits"]]
    out = [(c, block["splits"][c]) for c in names]
    for g in GROUPS:
        if g in block["groups"] and (include_refs or g not in REFERENCES):
            out.append((g, block["groups"][g]))
    return out


def table_shares(block, gaps):
    head = ("| population | panos | GT | misses | recall | merged | sub_threshold | "
            "localization | silent | of which strict | sub − silent, boot 95% (pts) | "
            "reading |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    lines = [head]
    for name, s in _populations(block):
        cells = []
        for b in mt.BUCKETS + ("strict_silent",):
            v = s["buckets"][b]
            lo, hi = v["wilson95"]
            cells.append(f"{v['n']} · {_pct(v['share'])} [{100*lo:.0f}–{100*hi:.0f}]")
        lines.append(f"| {LABEL[name]} | {s['n_panos']} | {s['n_gt']} | {s['n_miss']} | "
                     f"{s['recall']:.3f} | " + " | ".join(cells) +
                     f" | {_pts(s['sub_minus_silent_boot95'])} | {s['reading']} |")
    for c in OPEN_SPLITS:
        if c in gaps:
            lines.append(f"| `{c}` | — | — | — | — | not run: no op_cache (see Gaps) "
                         "| | | | | | |")
    return "\n".join(lines)


def table_excess(block, ref="reference_7"):
    lines = ["| population | merged | sub_threshold | localization | silent | "
             "strict silent | recall gap | sub excess − silent excess |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, s in _populations(block):
        cells = []
        ex = s.get("excess_vs", {}).get(ref)
        for b in mt.BUCKETS + ("strict_silent",):
            pts = s["buckets"][b]["recall_pts"]
            if ex:
                e = ex[b]
                cells.append(f"{pts:.3f} ({_signed(e['pts'], '.3f')}; {_frac(e['boot95'])})")
            else:
                cells.append(f"{pts:.3f}")
        if ex:
            t, d = ex["total"], ex["sub_minus_silent_excess"]
            tail = (f"{_signed(t['pts'], '.3f')} ({_frac(t['boot95'])}) | "
                    f"{_signed(d['pts'], '.3f')} ({_frac(d['boot95'])})")
        else:
            tail = "— | —"
        lines.append(f"| {LABEL[name]} | " + " | ".join(cells) + f" | {tail} |")
    return "\n".join(lines)


def table_nearfar(block):
    lines = ["| population | near GT | near misses | near recall | near merged / sub / loc / "
             "silent (strict) | far GT | far misses | far recall | far merged / sub / loc / "
             "silent (strict) |", "|---|--:|--:|--:|---|--:|--:|--:|---|"]
    for name, s in _populations(block):
        f = s["field"]

        def fc(x):
            c = x["counts"]
            return (f"{c['merged']} / {c['sub_threshold']} / {c['localization']} / "
                    f"{c['silent']} ({x['strict_silent']})")
        lines.append(f"| {LABEL[name]} | {f['near']['n_gt']} | {f['near']['n_miss']} | "
                     f"{f['near']['recall']:.3f} | {fc(f['near'])} | {f['far']['n_gt']} | "
                     f"{f['far']['n_miss']} | {f['far']['recall']:.3f} | {fc(f['far'])} |")
    return "\n".join(lines)


def table_detail(block):
    lines = ["| population | merged pairs | Chebyshev median px | > min_distance 10 | "
             "Euclid ≤ 2σ | < 8 px | sub-threshold peak median [q25, q75] | share ≥ 0.15 | "
             "null rate sub / merged |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, s in _populations(block):
        m, p, bk = s["merged_separation_px"], s["sub_threshold_peak"], s["buckets"]
        cm = f"{m['chebyshev_median']:.1f}" if m["chebyshev_median"] is not None else "—"
        nm = bk["merged"]["null_rate"]
        nm = f"{100*nm:.1f}%" if nm is not None else "—"
        lines.append(f"| {LABEL[name]} | {m['n']} | {cm} | {m['above_min_distance']} | "
                     f"{m['within_2sigma']} | {m['below_8px']} | {p['median']:.3f} "
                     f"[{p['q25']:.3f}, {p['q75']:.3f}] | "
                     f"{100*p['share_at_or_above_0_15']:.0f}% | "
                     f"{100*bk['sub_threshold']['null_rate']:.1f}% / {nm} |")
    return "\n".join(lines)


def table_two_rigs(result):
    """laurens_mapillary against laurens_gsv: one town, two rigs, all panos."""
    lines = ["| threshold | rig | GT | misses | recall | merged | sub_threshold | "
             "localization | silent (strict) |",
             "|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
    for th in result["thresholds"]:
        block = result["by_subset"]["all"][f"{th:.2f}"]["splits"]
        for c, rig in (("laurens_mapillary", "GoPro Max (Mapillary)"),
                       ("laurens_gsv", "GSV")):
            if c not in block:
                continue
            s, b = block[c], block[c]["buckets"]

            def cell(k):
                return f"{b[k]['n']} · {b[k]['recall_pts']:.3f}"
            lines.append(
                f"| {th:.2f} | {rig} | {s['n_gt']} | {s['n_miss']} | {s['recall']:.3f} | "
                f"{cell('merged')} | {cell('sub_threshold')} | {cell('localization')} | "
                f"{cell('silent')} ({b['strict_silent']['n']} · "
                f"{b['strict_silent']['recall_pts']:.3f}) |")
    return chr(10).join(lines)


def render_tables(result):
    """``{block name: markdown}`` for every generated table in the doc."""
    a, u = result["by_subset"]["all"], result["by_subset"]["unbiased"]
    gaps = result["gaps"]
    return {
        "shares-0.30-all": table_shares(a["0.30"], gaps),
        "excess-0.30-all": table_excess(a["0.30"]),
        "excess-0.30-all-no-paterson": table_excess(a["0.30"], "reference_no_paterson"),
        "shares-0.30-unbiased": table_shares(u["0.30"], gaps),
        "excess-0.30-unbiased": table_excess(u["0.30"]),
        "nearfar-0.30-all": table_nearfar(a["0.30"]),
        "detail-0.30-all": table_detail(a["0.30"]),
        "shares-0.55-all": table_shares(a["0.55"], gaps),
        "excess-0.55-all": table_excess(a["0.55"]),
        "shares-0.55-unbiased": table_shares(u["0.55"], gaps),
        "laurens-two-rigs": table_two_rigs(result),
    }


BLOCK_RE = re.compile(r"(<!-- BEGIN GENERATED: (?P<name>[\w.\-]+) -->\n)(?P<body>.*?)"
                      r"(<!-- END GENERATED: (?P=name) -->)", re.S)


def fill_doc(text, tables):
    """Replace every generated block in ``text``; error on an unknown or missing name."""
    seen = set()

    def sub(m):
        name = m.group("name")
        if name not in tables:
            raise SystemExit(f"doc has an unknown generated block {name!r}")
        seen.add(name)
        return m.group(1) + tables[name] + "\n" + m.group(4)
    out = BLOCK_RE.sub(sub, text)
    missing = sorted(set(tables) - seen)
    if missing:
        hint = (" -- the doc has CRLF line endings; it is pinned eol=lf in .gitattributes, "
                "so re-check it out" if chr(13) + chr(10) in text else "")
        raise SystemExit(f"doc is missing generated blocks {missing}{hint}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", default=CACHE_DIR)
    p.add_argument("--json-out", default=DEFAULT_JSON)
    p.add_argument("--doc", default=DEFAULT_DOC)
    p.add_argument("--check", action="store_true",
                   help="Re-derive the JSON and the doc's generated tables and compare "
                        "byte for byte; exit 1 on any difference.")
    args = p.parse_args(argv)

    text = dumps(build(args.cache))
    with open(args.doc, encoding="utf-8", newline="") as f:
        doc = f.read()
    new_doc = fill_doc(doc, render_tables(json.loads(text)))
    if args.check:
        with open(args.json_out, encoding="utf-8", newline="") as f:
            have = f.read()
        ok = True
        if have != text:
            print(f"CHECK FAILED: {args.json_out} does not match a fresh derivation")
            ok = False
        if new_doc != doc:
            print(f"CHECK FAILED: generated tables in {args.doc} are stale")
            ok = False
        if ok:
            print(f"check ok: {args.json_out} and the tables in {args.doc} re-derive "
                  "byte-for-byte")
        return 0 if ok else 1
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    with open(args.doc, "w", encoding="utf-8", newline="") as f:
        f.write(new_doc)
    print(f"Wrote {args.json_out} and the generated tables in {args.doc}")
    for c, g in result_gaps(text).items():
        print(f"  gap: {c}: {g['reason']}; fill with: {g['fill']}")
    return 0


def result_gaps(text):
    return json.loads(text)["gaps"]


if __name__ == "__main__":
    sys.exit(main())
