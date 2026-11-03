"""Per-ramp recall across views, how correlated the per-view misses are, and what
thinning the capture density does to per-ramp recall (#38).

Everything here is computed from one committed table,
``analysis_out/multiview_48/captures_R25.csv`` (#48 Phase 1, PR #200): one row per
(pool ramp, run pano whose camera is within 25 m of it), with the best stored detection
confidence that claims the ramp in world space (raycast within 5 m) and the camera
position. No GPU, no network, no labeler checkout, no imagery.

What it adds to ``docs/multiview_48.md`` §4-§5, which already measured the shape of the
per-ramp curve and that misses are correlated (153 ramps missed by every other view vs
76.5 predicted under independence):

1. **Uncertainty on the correlation.** A ramp-cluster bootstrap CI on the all-views-missed
   ratio (observed / independent), and a stratified permutation null: the miss flags of
   all captures in one (city, range bin) stratum are shuffled across ramps, which keeps
   every view's city and distance bin and breaks only the tie between views of one ramp.
2. **The issue's own arithmetic.** #38 multiplied the three nearest views' miss rates
   (~0.3%). Here that product is computed per ramp from range-matched marginals and set
   against the observed all-three-missed share.
3. **Who the all-missed ramps are**: city, imagery, distances, capture count, whether the
   GT-source view saw them, and their #48 residual class.
4. **Density.** Production thins Mapillary to one pano per 5 m grid cell (newest capture
   wins; labeler ``sources/mapillary.thin_panos``) and does not thin GSV. Re-applying that
   rule at coarser spacings shows how per-ramp recall falls as captures are removed.
   Denser-than-native sampling cannot be simulated from these data (see the doc).
5. **Nearest view vs any view** (#38's distance-aware item), with CIs.

The GT-source view is excluded from every "other views" number, as in #48 §4-§5: the GT
world point of a verdict-true ramp was raycast from that view's own detection. The
"union" rows add it back.

    python scripts/analysis/per_ramp_recall_38.py run         # about 1 min, CPU
    python scripts/analysis/per_ramp_recall_38.py check       # re-derive and compare bytes
    python scripts/analysis/per_ramp_recall_38.py doc-numbers # the numbers the doc quotes
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR = os.path.join(REPO, "analysis_out", "multiview_48")
CAPTURES = os.path.join(IN_DIR, "captures_R25.csv")
CAPTURES_HIT8 = os.path.join(IN_DIR, "captures_R25_hit8.csv")
RESIDUAL = os.path.join(IN_DIR, "residual_misses.json")
OUT_ROOT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
OUT = os.path.join(OUT_ROOT, "per_ramp_recall_38")

SOURCE = {"richmond": "mapillary", "paterson": "gsv", "gainesville": "gsv",
          "bend": "gsv", "sao_paulo": "gsv"}
SUB_CITIES = ("richmond", "paterson", "gainesville", "sao_paulo")   # store peaks < 0.55
RANGE_BINS = ((0.0, 6.0), (6.0, 12.0), (12.0, 18.0), (18.0, 25.0))  # as #48 §5
FINE_BINS = tuple((float(a), float(a + 3)) for a in range(0, 25, 3))
R_DEFAULT = 18.0
SEED = 38
N_BOOT = 2000
N_PERM = 5000
SPACINGS = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
N_OFFSETS = 40


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_captures(path):
    """{ramp_uid: {"city", "uid", "captures": [dict]}} from a captures CSV. A ramp is keyed
    by its ``city:index`` uid, so no id is used without its city."""
    ramps = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            r = ramps.setdefault(row["ramp_uid"], {"uid": row["ramp_uid"], "city": row["city"],
                                                   "captures": []})
            assert row["ramp_uid"].split(":")[0] == row["city"], row["ramp_uid"]
            r["captures"].append({
                "pano_id": row["pano_id"], "dist_m": float(row["dist_m"]),
                "is_source": row["is_source"] == "1",
                "world_conf": None if row["world_conf"] == "" else float(row["world_conf"]),
                "capture_date": row["capture_date"], "cam_e": float(row["cam_e"]),
                "cam_n": float(row["cam_n"])})
    return ramps


def is_hit(conf, floor):
    return conf is not None and conf >= floor - 1e-12


def bin_index(d, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= d < hi:
            return i
    return None


def others(ramp, radius=R_DEFAULT):
    """Non-source captures within ``radius``, nearest first (ties by pano id)."""
    return sorted((c for c in ramp["captures"] if not c["is_source"] and c["dist_m"] <= radius),
                  key=lambda c: (c["dist_m"], c["pano_id"]))


def source_capture(ramp):
    """The GT-source capture (the nearest if a merged ramp has two)."""
    src = sorted((c for c in ramp["captures"] if c["is_source"]), key=lambda c: c["dist_m"])
    return src[0] if src else None


# --------------------------------------------------------------------------- #
# the correlation design: ramps x strata count matrices
# --------------------------------------------------------------------------- #

class Design:
    """Per-ramp capture counts by stratum, so marginals, predictions and resamples are
    matrix products.

    ``C[r, s]`` captures of ramp r in stratum s, ``M[r, s]`` misses among them,
    ``allmiss[r]`` 1 if every capture of ramp r missed. Strata are (city, range bin) by
    default; ``source_stratum`` puts the GT-source view in its own (city, range bin)
    stratum, because its miss rate is the per-pano recall the reviewer measured, not an
    other-view rate."""

    def __init__(self, rows, strata):
        self.strata = strata
        idx = {s: i for i, s in enumerate(strata)}
        n, k = len(rows), len(strata)
        self.C = np.zeros((n, k))
        self.M = np.zeros((n, k))
        self.allmiss = np.zeros(n)
        self.city = []
        self.flat_stratum = []   # one entry per capture, for the permutation null
        self.flat_miss = []
        self.flat_ramp = []
        for r, (city, caps) in enumerate(rows):
            self.city.append(city)
            for s, miss in caps:
                self.C[r, idx[s]] += 1
                self.M[r, idx[s]] += miss
                self.flat_stratum.append(idx[s])
                self.flat_miss.append(miss)
                self.flat_ramp.append(r)
            self.allmiss[r] = all(m for _, m in caps)
        self.city = np.array(self.city)
        self.flat_stratum = np.array(self.flat_stratum, dtype=int)
        self.flat_miss = np.array(self.flat_miss, dtype=float)
        self.flat_ramp = np.array(self.flat_ramp, dtype=int)

    def stats(self, w=None):
        """(observed all-missed, predicted under independence) with ramp weights ``w``
        (multiplicities of a bootstrap resample; ones = the sample itself). Marginals are
        re-estimated from the weighted sample, as they would be from a fresh one."""
        w = np.ones(len(self.allmiss)) if w is None else w
        tot = w @ self.C
        mis = w @ self.M
        with np.errstate(divide="ignore", invalid="ignore"):
            pm = np.where(tot > 0, mis / np.where(tot > 0, tot, 1), 0.0)
            logpm = np.where(pm > 0, np.log(np.where(pm > 0, pm, 1)), -np.inf)
            # C[r,s] * log pm[s]; 0 * -inf must be 0
            terms = np.where(self.C > 0, self.C * logpm, 0.0)
        pred_r = np.exp(terms.sum(axis=1))
        return float(w @ self.allmiss), float(w @ pred_r), pred_r

    def bootstrap(self, n_boot, rng, clusters=None):
        """Cluster bootstrap, resampled within city so each city keeps its number of
        clusters. ``clusters`` gives one label per ramp (default: each ramp is its own
        cluster); ramps sharing a label are drawn together."""
        cities = sorted(set(self.city))
        n = len(self.allmiss)
        if clusters is None:            # one ramp per cluster: the draw sequence of the
            members = {c: np.flatnonzero(self.city == c) for c in cities}   # first release
            out = []
            for _ in range(n_boot):
                w = np.zeros(n)
                for c in cities:
                    m = members[c]
                    np.add.at(w, rng.choice(m, size=len(m), replace=True), 1)
                obs, pred, _ = self.stats(w)
                out.append((obs, pred))
            return np.array(out)
        labels = np.asarray(clusters, dtype=object)
        groups = {}
        for c in cities:
            idx = np.flatnonzero(self.city == c)
            by = {}
            for i in idx:
                by.setdefault(labels[i], []).append(i)
            groups[c] = [np.array(v) for _, v in sorted(by.items(), key=lambda t: str(t[0]))]
        out = []
        for _ in range(n_boot):
            w = np.zeros(n)
            for c in cities:
                g = groups[c]
                for j in rng.choice(len(g), size=len(g), replace=True):
                    np.add.at(w, g[j], 1)
            obs, pred, _ = self.stats(w)
            out.append((obs, pred))
        return np.array(out)

    def permutation_null(self, n_perm, rng):
        """All-missed counts after shuffling miss flags within each stratum across ramps.
        Keeps every capture's stratum and every stratum's miss count; breaks only which
        ramp a miss belongs to."""
        order = np.argsort(self.flat_stratum, kind="stable")
        strata_sorted = self.flat_stratum[order]
        bounds = np.flatnonzero(np.diff(strata_sorted)) + 1
        groups = np.split(order, bounds)
        n = len(self.allmiss)
        counts = np.bincount(self.flat_ramp, minlength=n)
        res = np.empty(n_perm)
        miss = self.flat_miss.copy()
        for i in range(n_perm):
            for g in groups:
                miss[g] = rng.permutation(self.flat_miss[g])
            hits = np.bincount(self.flat_ramp, weights=1.0 - miss, minlength=n)
            res[i] = np.sum((hits == 0) & (counts > 0))
        return res


def design_rows(ramps, floor, radius=R_DEFAULT, bins=RANGE_BINS, min_others=2,
                include_source=False, k_nearest=None, pop_radius=R_DEFAULT):
    """[(city, [(stratum, miss), ...]), ...] for the ramps with >= ``min_others`` other
    captures within ``pop_radius``, using their other captures within ``radius``
    (``k_nearest`` keeps only the nearest k of them). Keeping ``pop_radius`` at 18 m
    while ``radius`` goes to 25 m scores the same ramps with every stored view."""
    rows, kept = [], []
    for r in ramps:
        if len(others(r, pop_radius)) < min_others:
            continue
        os_ = others(r, radius)
        if k_nearest is not None:
            os_ = os_[:k_nearest]
        caps = [((r["city"], "other", bin_index(c["dist_m"], bins)),
                 not is_hit(c["world_conf"], floor)) for c in os_]
        if include_source:
            s = source_capture(r)
            if s is None:
                continue
            caps.append(((r["city"], "source", bin_index(s["dist_m"], bins)),
                          not is_hit(s["world_conf"], floor)))
        rows.append((r["city"], caps))
        kept.append(r)
    return rows, kept


def correlation_block(ramps, floor, rng, bins=RANGE_BINS, include_source=False,
                      k_nearest=None, min_others=2, n_boot=N_BOOT, n_perm=N_PERM,
                      radius=R_DEFAULT, cluster_by=None):
    """``cluster_by`` (ramp -> label) makes the bootstrap draw ramps sharing a label
    together, e.g. every ramp marked in one source pano."""
    rows, kept = design_rows(ramps, floor, radius=radius, bins=bins,
                             include_source=include_source, k_nearest=k_nearest,
                             min_others=min_others)
    strata = sorted({s for _, caps in rows for s, _ in caps},
                    key=lambda s: (s[0], s[1], -1 if s[2] is None else s[2]))
    d = Design(rows, strata)
    obs, pred, pred_r = d.stats()
    clusters = None if cluster_by is None else [cluster_by(r) for r in kept]
    boot = d.bootstrap(n_boot, rng, clusters) if n_boot else None
    perm = d.permutation_null(n_perm, rng) if n_perm else None
    out = {"ramps": len(rows), "observed_all_missed": obs, "predicted_independent": pred,
           "ratio": obs / pred if pred else None,
           "observed_share": obs / len(rows) if rows else None,
           "predicted_share": pred / len(rows) if rows else None}
    if boot is not None:
        ratios = boot[:, 0] / boot[:, 1]
        out["ratio_ci95"] = [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
        out["observed_share_ci95"] = [float(np.percentile(boot[:, 0], 2.5)) / len(rows),
                                      float(np.percentile(boot[:, 0], 97.5)) / len(rows)]
        out["n_boot"] = n_boot
    if perm is not None:
        out["permutation_null_mean"] = float(perm.mean())
        out["permutation_null_p95"] = float(np.percentile(perm, 95))
        out["permutation_p_value"] = float((1 + np.sum(perm >= obs)) / (1 + len(perm)))
        out["n_perm"] = n_perm
    return out, kept, pred_r


# --------------------------------------------------------------------------- #
# the all-missed ramps
# --------------------------------------------------------------------------- #

def all_missed_table(ramps, floor, residual_by_uid, radius=R_DEFAULT):
    """One row per ramp with >= 2 other captures within ``radius``, flagged when every one
    of them missed (``all_other_missed``), and when the source view and every other view
    in the table, out to 25 m, missed (``all_views_missed_25m``: the per-ramp floor)."""
    rows = []
    for r in ramps:
        os_ = others(r, radius)
        if len(os_) < 2:
            continue
        src = source_capture(r)
        months = sorted({(c["capture_date"] or "")[:7] for c in os_ if c["capture_date"]})
        confs = [c["world_conf"] for c in os_ if c["world_conf"] is not None]
        os25 = others(r, 25.0)
        hit_18_25 = any(is_hit(c["world_conf"], floor) for c in os25 if c["dist_m"] > radius)
        confs25 = [c["world_conf"] for c in os25 if c["world_conf"] is not None]
        rows.append({
            "ramp_uid": r["uid"], "city": r["city"], "imagery": SOURCE[r["city"]],
            "n_other": len(os_), "nearest_other_m": os_[0]["dist_m"],
            "median_other_m": float(np.median([c["dist_m"] for c in os_])),
            "n_months": len(months),
            "source_dist_m": None if src is None else src["dist_m"],
            "source_hit": None if src is None else is_hit(src["world_conf"], floor),
            "best_other_conf": max(confs) if confs else None,
            "all_other_missed": all(not is_hit(c["world_conf"], floor) for c in os_),
            "n_other_25m": len(os25),
            "other_hit_18_25m": hit_18_25,
            "best_other_conf_25m": max(confs25) if confs25 else None,
            "all_views_missed_25m": (src is not None and not is_hit(src["world_conf"], floor)
                                     and not any(is_hit(c["world_conf"], floor) for c in os25)),
            "residual_class_48": residual_by_uid.get(r["uid"]),
        })
    return rows


def describe(rows, key):
    v = np.array([x[key] for x in rows if x[key] is not None], dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "p75": float(np.percentile(v, 75)), "mean": float(v.mean())}


def characterise(table):
    miss = [x for x in table if x["all_other_missed"]]
    rest = [x for x in table if not x["all_other_missed"]]
    by_city = {}
    for c in sorted({x["city"] for x in table}):
        n = sum(1 for x in table if x["city"] == c)
        k = sum(1 for x in miss if x["city"] == c)
        by_city[c] = {"ramps": n, "all_other_missed": k, "share": k / n if n else None}
    src_seen = sum(1 for x in miss if x["source_hit"])
    classes = {}
    for x in miss:
        key = x["residual_class_48"] or "recalled_by_a_site (not a #48 residual)"
        classes[key] = classes.get(key, 0) + 1
    near = [x for x in miss if x["nearest_other_m"] < 6.0]
    return {
        "ramps": len(table), "all_other_missed": len(miss),
        "by_city": by_city,
        "nearest_other_m": {"all_other_missed": describe(miss, "nearest_other_m"),
                            "rest": describe(rest, "nearest_other_m")},
        "median_other_m": {"all_other_missed": describe(miss, "median_other_m"),
                           "rest": describe(rest, "median_other_m")},
        "n_other": {"all_other_missed": describe(miss, "n_other"),
                    "rest": describe(rest, "n_other")},
        "n_months": {"all_other_missed": describe(miss, "n_months"),
                     "rest": describe(rest, "n_months")},
        "nearest_other_under_6m": len(near),
        "source_view_hit": src_seen,
        "source_view_missed_too": len(miss) - src_seen,
        "best_other_conf_below_floor": describe(miss, "best_other_conf"),
        "with_any_stored_peak_below_floor": sum(1 for x in miss if x["best_other_conf"] is not None),
        "residual_class_48": dict(sorted(classes.items())),
        "floor": floor_block(table),
    }


def floor_block(table):
    """The ramps the deployment misses: source view and every other view out to 25 m.
    Also the 18 m version (source + other views within 18 m) and how many of those a view
    at 18-25 m finds."""
    fl = [x for x in table if x["all_views_missed_25m"]]
    rest = [x for x in table if not x["all_views_missed_25m"]]
    f18 = [x for x in table if x["all_other_missed"] and x["source_hit"] is False]
    classes = {}
    for x in fl:
        key = x["residual_class_48"] or "recalled_by_a_site (not a #48 residual)"
        classes[key] = classes.get(key, 0) + 1
    return {
        "ramps": len(table),
        "missed_by_every_view_25m": len(fl),
        "share": len(fl) / len(table) if table else None,
        "union_recall_25m": 1 - len(fl) / len(table) if table else None,
        "missed_by_every_view_18m": len(f18),
        "of_those_18m_found_at_18_25m": sum(1 for x in f18 if x["other_hit_18_25m"]),
        "by_city": {c: sum(1 for x in fl if x["city"] == c) for c in sorted({x["city"] for x in table})},
        "nearest_other_m": {"floor": describe(fl, "nearest_other_m"), "rest": describe(rest, "nearest_other_m")},
        "source_dist_m": {"floor": describe(fl, "source_dist_m"), "rest": describe(rest, "source_dist_m")},
        "n_other_25m": {"floor": describe(fl, "n_other_25m"), "rest": describe(rest, "n_other_25m")},
        "with_stored_peak_below_floor": sum(1 for x in fl if x["best_other_conf_25m"] is not None),
        "residual_class_48": dict(sorted(classes.items())),
    }


# --------------------------------------------------------------------------- #
# nearest view vs any view, and #38's three-nearest arithmetic
# --------------------------------------------------------------------------- #

def cluster_boot_share(flags, cities, n_boot, rng):
    flags = np.asarray(flags, dtype=float)
    cities = np.asarray(cities)
    groups = [np.flatnonzero(cities == c) for c in sorted(set(cities))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(flags[idx].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def nearest_vs_any(ramps, floor, rng, n_boot=N_BOOT):
    """Per-ramp recall from other views under four rules, on one fixed population (ramps
    with >= 3 other captures within 25 m, so every rule has something to use)."""
    pop = [r for r in ramps if len(others(r, 25.0)) >= 3]
    cities = [r["city"] for r in pop]
    rules = {
        "nearest_other_only": lambda r: others(r, 25.0)[:1],
        "any_other_within_6m": lambda r: others(r, 6.0),
        "any_other_within_12m": lambda r: others(r, 12.0),
        "any_other_within_18m": lambda r: others(r, 18.0),
        "any_other_within_25m": lambda r: others(r, 25.0),
    }
    out = {"ramps": len(pop)}
    for name, fn in rules.items():
        flags = [any(is_hit(c["world_conf"], floor) for c in fn(r)) for r in pop]
        out[name] = {"recall": float(np.mean(flags)),
                     "ramps_with_a_capture": sum(1 for r in pop if fn(r)),
                     "ci95": cluster_boot_share(flags, cities, n_boot, rng)}
    return out


def three_nearest(ramps, floor, rng):
    """#38's arithmetic: P(the three nearest views all miss), the per-ramp independence
    product of range-matched marginals vs the observed share."""
    blk, kept, pred_r = correlation_block(ramps, floor, rng, k_nearest=3, min_others=3,
                                          n_perm=N_PERM)
    dists = [others(r)[:3] for r in kept]
    blk["median_dist_of_3_nearest_m"] = [float(np.median([d[i]["dist_m"] for d in dists]))
                                         for i in range(3)]
    blk["recall_3_nearest_others"] = 1 - blk["observed_share"]
    return blk


# --------------------------------------------------------------------------- #
# density: re-apply production thinning at coarser spacings
# --------------------------------------------------------------------------- #

def pano_table(ramps, city):
    """{pano_id: (e, n, capture_month)} for one city; one camera position per pano."""
    out = {}
    for r in ramps:
        if r["city"] != city:
            continue
        for c in r["captures"]:
            out[c["pano_id"]] = (c["cam_e"], c["cam_n"], c["capture_date"] or "")
    return out


def grid_thin(panos, spacing, offset, rng):
    """The labeler's ``thin_panos`` rule on a grid of ``spacing`` metres shifted by
    ``offset``: one pano per cell, newest capture month wins, ties broken at random
    (the labeler breaks them on a quality score these data do not carry)."""
    best = {}
    for pid, (e, n, month) in panos.items():
        cell = (math.floor((e - offset[0]) / spacing), math.floor((n - offset[1]) / spacing))
        key = (month, rng.random())
        if cell not in best or key > best[cell][0]:
            best[cell] = (key, pid)
    return {pid for _, pid in best.values()}


def nn_spacing(panos):
    """Median distance from each pano to its nearest other pano (city-native density)."""
    xy = np.array([(e, n) for e, n, _ in panos.values()])
    if len(xy) < 2:
        return None
    d = []
    for i in range(len(xy)):
        dd = np.hypot(*(xy - xy[i]).T)
        dd[i] = np.inf
        d.append(dd.min())
    return float(np.median(d))


def thinning_curve(ramps, floor, rng, spacings=SPACINGS, n_offsets=N_OFFSETS,
                   n_boot=N_BOOT, radius=R_DEFAULT):
    """Per-ramp recall (other views within ``radius``; and the union with the source view)
    when every city's panos are re-thinned to one per ``spacing`` grid cell, averaged over
    ``n_offsets`` random grid offsets. ``random`` rows keep the same number of panos
    chosen uniformly, which has no grid edge effects (cross-check). The population is
    fixed: every ramp with >= 1 other capture within ``radius`` at native density."""
    pop = [r for r in ramps if len(others(r, radius)) >= 1]
    cities = sorted({r["city"] for r in pop})
    tables = {c: pano_table(ramps, c) for c in cities}
    groups = {"pooled": cities, "gsv": [c for c in cities if SOURCE[c] == "gsv"],
              "mapillary": [c for c in cities if SOURCE[c] == "mapillary"]}

    pre = [(r, others(r, radius), source_capture(r)) for r in pop]

    def per_ramp(kept_by_city):
        oth, uni, ncap = [], [], []
        for r, all_others, src in pre:
            kept = kept_by_city[r["city"]]
            os_ = [c for c in all_others if c["pano_id"] in kept]
            o = any(is_hit(c["world_conf"], floor) for c in os_)
            s = src is not None and src["pano_id"] in kept and is_hit(src["world_conf"], floor)
            oth.append(o)
            uni.append(o or s)
            ncap.append(len(os_))
        return np.array(oth, float), np.array(uni, float), np.array(ncap, float)

    native = {c: set(tables[c]) for c in cities}
    o0, u0, n0 = per_ramp(native)
    rows = [{"spacing_m": None, "scheme": "native", "panos": {c: len(tables[c]) for c in cities},
             "p_other": o0, "p_union": u0, "mean_other_captures": n0}]
    for s in spacings:
        for scheme in ("grid", "random"):
            acc_o = np.zeros(len(pop))
            acc_u = np.zeros(len(pop))
            acc_n = np.zeros(len(pop))
            kept_counts = {c: 0 for c in cities}
            for _ in range(n_offsets):
                kept = {}
                for c in cities:
                    g = grid_thin(tables[c], s, (rng.random() * s, rng.random() * s), rng)
                    if scheme == "random":
                        ids = sorted(tables[c])
                        g = set(rng.choice(ids, size=len(g), replace=False).tolist())
                    kept[c] = g
                    kept_counts[c] += len(g)
                o, u, n = per_ramp(kept)
                acc_o += o
                acc_u += u
                acc_n += n
            rows.append({"spacing_m": s, "scheme": scheme,
                         "panos": {c: kept_counts[c] / n_offsets for c in cities},
                         "p_other": acc_o / n_offsets, "p_union": acc_u / n_offsets,
                         "mean_other_captures": acc_n / n_offsets})
    city_arr = np.array([r["city"] for r in pop])
    out = {"population": len(pop), "n_offsets": n_offsets,
           "native_nn_spacing_m": {c: nn_spacing(tables[c]) for c in cities},
           "native_panos": {c: len(tables[c]) for c in cities}, "rows": []}
    for row in rows:
        entry = {"spacing_m": row["spacing_m"], "scheme": row["scheme"], "groups": {}}
        for gname, gc in groups.items():
            if not gc:
                continue
            mask = np.isin(city_arr, gc)
            panos = sum(row["panos"][c] for c in gc)
            entry["groups"][gname] = {
                "ramps": int(mask.sum()),
                "panos_kept": panos,
                "panos_kept_share": panos / sum(len(tables[c]) for c in gc),
                "recall_other": float(row["p_other"][mask].mean()),
                "recall_other_ci95": cluster_boot_share(row["p_other"][mask], city_arr[mask],
                                                        n_boot, rng),
                "recall_union": float(row["p_union"][mask].mean()),
                # paired with native on the same ramps: thinned minus native
                "delta_other_vs_native": float((row["p_other"] - o0)[mask].mean()),
                "delta_other_vs_native_ci95": cluster_boot_share(
                    (row["p_other"] - o0)[mask], city_arr[mask], n_boot, rng),
                "delta_union_vs_native": float((row["p_union"] - u0)[mask].mean()),
                "delta_union_vs_native_ci95": cluster_boot_share(
                    (row["p_union"] - u0)[mask], city_arr[mask], n_boot, rng),
                "mean_other_captures": float(row["mean_other_captures"][mask].mean()),
            }
        out["rows"].append(entry)
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def rnd(v, nd=4):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def dumps(payload):
    return json.dumps(rnd(payload), indent=1, sort_keys=True) + "\n"


def per_city_stats(ramps, floor):
    """Descriptives the doc quotes: per city and imagery, on the ramps with >= 2 other
    views within 18 m -- mean other views, other-view marginal miss rate, mean distinct
    capture months among the other views, and the median separation of the two nearest
    other cameras."""
    out = {}
    names = sorted(SOURCE) + ["gsv", "mapillary"]
    for name in names:
        pop = [r for r in ramps if (r["city"] == name or SOURCE[r["city"]] == name)
               and len(others(r)) >= 2]
        caps = [c for r in pop for c in others(r)]
        months = [len({(c["capture_date"] or "")[:7] for c in others(r) if c["capture_date"]})
                  for r in pop]
        seps = [math.hypot(o[0]["cam_e"] - o[1]["cam_e"], o[0]["cam_n"] - o[1]["cam_n"])
                for o in (others(r) for r in pop)]
        out[name] = {"ramps": len(pop),
                     "mean_other_views": float(np.mean([len(others(r)) for r in pop])),
                     "other_view_miss_rate": float(np.mean(
                         [not is_hit(c["world_conf"], floor) for c in caps])),
                     "mean_months": float(np.mean(months)),
                     "median_sep_two_nearest_m": float(np.median(seps))}
    return out


def block_rng(name):
    """One generator per output block, seeded by its name, so adding or removing a block
    never moves another block's bootstrap or permutation draws."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])


def compute():
    base = load_captures(CAPTURES)
    hit8 = load_captures(CAPTURES_HIT8)
    ramps = sorted(base.values(), key=lambda r: (r["city"], int(r["uid"].split(":")[1])))
    ramps8 = sorted(hit8.values(), key=lambda r: (r["city"], int(r["uid"].split(":")[1])))
    sub = [r for r in ramps if r["city"] in SUB_CITIES]
    groups = {"gsv": [r for r in ramps if SOURCE[r["city"]] == "gsv"],
              "mapillary": [r for r in ramps if SOURCE[r["city"]] == "mapillary"]}
    with open(RESIDUAL, encoding="utf-8") as f:
        residual = {x["uid"]: x["class"] for x in json.load(f)["ramps"]}

    def cb(name, pop, floor, **kw):
        blk, _, _ = correlation_block(pop, floor, block_rng(name), **kw)
        return blk

    corr = {"pooled_055": cb("pooled_055", ramps, 0.55)}
    # instrument check against docs/multiview_48.md §5 (153 vs 76.52)
    assert int(corr["pooled_055"]["observed_all_missed"]) == 153, corr["pooled_055"]
    assert abs(corr["pooled_055"]["predicted_independent"] - 76.5197) < 1e-3, corr["pooled_055"]
    corr["gsv_055"] = cb("gsv_055", groups["gsv"], 0.55)
    corr["mapillary_055"] = cb("mapillary_055", groups["mapillary"], 0.55)  # = richmond
    corr["pooled4_030"] = cb("pooled4_030", sub, 0.30)
    corr["pooled4_010"] = cb("pooled4_010", sub, 0.10)
    corr["pooled_055_fine_3m_bins"] = cb("pooled_055_fine_3m_bins", ramps, 0.55, bins=FINE_BINS)
    corr["pooled_055_hit8m"] = cb("pooled_055_hit8m", ramps8, 0.55)
    corr["pooled_055_union_with_source"] = cb("pooled_055_union_with_source", ramps, 0.55,
                                              include_source=True)
    # the same ramps with every stored view: source + other views out to 25 m
    corr["pooled_055_union_with_source_25m"] = cb("pooled_055_union_with_source_25m", ramps,
                                                  0.55, include_source=True, radius=25.0)
    for c in sorted(SOURCE):
        if SOURCE[c] == "mapillary":
            continue                     # richmond is the whole mapillary_055 row
        corr[f"{c}_055"] = cb(f"{c}_055", [r for r in ramps if r["city"] == c], 0.55,
                              n_perm=1000)

    # The all-missed ratio grows with the number of views per ramp, and richmond has
    # about twice as many as a GSV city, so compare imagery and cities at matched k.
    matched = {}
    for k in (2, 3, 4):
        pops = dict(groups)
        pops.update({c: [r for r in ramps if r["city"] == c] for c in sorted(SOURCE)
                     if SOURCE[c] == "gsv"})
        pops["pooled"] = ramps
        for name, pop in pops.items():
            key = f"{name}_k{k}"
            matched[key] = cb("matched_" + key, pop, 0.55, k_nearest=k, min_others=k,
                              n_boot=1000, n_perm=0)
    corr["matched_view_count_055"] = matched

    # Same k, one fixed population (ramps with >= 4 other views), so the rise with k is
    # not a change of population.
    fixed = {}
    for name, pop in (("pooled", ramps), ("gsv", groups["gsv"]),
                      ("mapillary", groups["mapillary"])):
        pop4 = [r for r in pop if len(others(r)) >= 4]
        for k in (2, 3, 4):
            key = f"{name}_k{k}"
            fixed[key] = cb("fixed4_" + key, pop4, 0.55, k_nearest=k, min_others=k,
                            n_boot=1000, n_perm=0)
    corr["matched_view_count_055_fixed_population_4plus"] = fixed

    # Robustness of the pooled ratio: (1) bootstrap over source panos, since ramps marked
    # in one pano share their captures; (2) the two GT populations separately -- ramps the
    # source view detected vs ramps the reviewer added as missed.
    corr["pooled_055_source_pano_clusters"] = cb(
        "pooled_055_source_pano_clusters", ramps, 0.55, n_perm=0,
        cluster_by=lambda r: (source_capture(r) or {}).get("pano_id"))
    corr["pooled_055_source_pano_clusters"]["clusters"] = len(
        {(r["city"], (source_capture(r) or {}).get("pano_id"))
         for r in ramps if len(others(r)) >= 2})
    by_src = {}
    for name, want in (("detected", True), ("reviewer_added", False)):
        pop = [r for r in ramps if source_capture(r) is not None
               and is_hit(source_capture(r)["world_conf"], 0.55) == want]
        by_src[name] = cb("pooled_055_by_source_hit_" + name, pop, 0.55, n_perm=0)
    corr["pooled_055_by_source_hit"] = by_src

    table = all_missed_table(ramps, 0.55, residual)
    return {
        "per_city_055": per_city_stats(ramps, 0.55),
        "inputs": {"captures_R25.csv": sha256(CAPTURES),
                   "captures_R25_hit8.csv": sha256(CAPTURES_HIT8),
                   "residual_misses.json": sha256(RESIDUAL)},
        "params": {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM, "radius_m": R_DEFAULT,
                   "range_bins": RANGE_BINS, "spacings_m": SPACINGS, "n_offsets": N_OFFSETS,
                   "hit_test": "world (raycast within 5 m; hit8m arm 8 m)",
                   "numpy": np.__version__},
        "correlation": corr,
        "three_nearest_055": three_nearest(ramps, 0.55, block_rng("three_nearest_055")),
        "three_nearest_055_hit8m": three_nearest(ramps8, 0.55,
                                                 block_rng("three_nearest_055_hit8m")),
        "three_nearest_030_pooled4": three_nearest(sub, 0.30,
                                                   block_rng("three_nearest_030_pooled4")),
        "all_missed_055": characterise(table),
        "nearest_vs_any_055": nearest_vs_any(ramps, 0.55, block_rng("nearest_vs_any_055")),
        "thinning_055": thinning_curve(ramps, 0.55, block_rng("thinning_055")),
        "thinning_030_pooled4": thinning_curve(sub, 0.30, block_rng("thinning_030_pooled4"),
                                               n_offsets=20),
    }, table


TABLE_COLUMNS = ["ramp_uid", "city", "imagery", "n_other", "nearest_other_m", "median_other_m",
                 "n_months", "source_dist_m", "source_hit", "best_other_conf",
                 "all_other_missed", "n_other_25m", "other_hit_18_25m", "best_other_conf_25m",
                 "all_views_missed_25m", "residual_class_48"]


def table_text(table):
    lines = [",".join(TABLE_COLUMNS)]
    for row in table:
        vals = []
        for k in TABLE_COLUMNS:
            v = rnd(row[k])
            vals.append("" if v is None else str(int(v)) if isinstance(v, bool) else str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# the doc's tables, generated from results.json (tests check the doc quotes them verbatim)
# --------------------------------------------------------------------------- #

def _f(v, nd=2):
    return f"{v:.{nd}f}"


def _ci(ci, nd=2):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _pct(v):
    return f"{100 * v:.1f}%"


def _n(v):
    return f"{int(round(v)):,}"


CORR_ROWS = (
    ("pooled, 5 cities", "pooled_055", "0.55"),
    ("GSV (4 cities)", "gsv_055", "0.55"),
    ("Mapillary (richmond)", "mapillary_055", "0.55"),
    ("paterson", "paterson_055", "0.55"),
    ("gainesville", "gainesville_055", "0.55"),
    ("sao_paulo", "sao_paulo_055", "0.55"),
    ("bend", "bend_055", "0.55"),
    ("pooled, 4 cities (no bend)", "pooled4_030", "0.30"),
    ("pooled, 4 cities (no bend)", "pooled4_010", "0.10"),
    ("pooled, 3 m range bins", "pooled_055_fine_3m_bins", "0.55"),
    ("pooled, 8 m world test", "pooled_055_hit8m", "0.55"),
    ("pooled, union with the source view, other views within 18 m",
     "pooled_055_union_with_source", "0.55"),
    ("pooled, union with the source view, other views within 25 m",
     "pooled_055_union_with_source_25m", "0.55"),
)


def doc_tables(res):
    """Markdown rows the doc quotes, keyed by section."""
    c = res["correlation"]
    out = {}
    rows = []
    for label, key, floor in CORR_ROWS:
        b = c[key]
        rows.append(f"| {label} | {floor} | {_n(b['ramps'])} | {_n(b['observed_all_missed'])} "
                    f"({_pct(b['observed_share'])}) | {b['predicted_independent']:.1f} "
                    f"({_pct(b['predicted_share'])}) | {_f(b['ratio'])} {_ci(b['ratio_ci95'])} | "
                    f"{b['permutation_null_mean']:.1f} / {int(b['permutation_null_p95'])} | "
                    f"{b['permutation_p_value']:.4f} |")
    out["correlation"] = rows
    m = c["matched_view_count_055"]
    rows = []
    for name in ("pooled", "gsv", "mapillary", "paterson", "gainesville", "sao_paulo", "bend"):
        cells = []
        for k in (2, 3, 4):
            b = m[f"{name}_k{k}"]
            cells.append(f"{_f(b['ratio'])} {_ci(b['ratio_ci95'])} ({_n(b['observed_all_missed'])} "
                         f"vs {b['predicted_independent']:.1f}, n {_n(b['ramps'])})")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    out["matched"] = rows
    rows = []
    for label, key in (("5 m world test, 0.55", "three_nearest_055"),
                       ("8 m world test, 0.55", "three_nearest_055_hit8m"),
                       ("5 m world test, 0.30 (4 cities)", "three_nearest_030_pooled4")):
        b = res[key]
        rows.append(f"| {label} | {_n(b['ramps'])} | {_pct(b['observed_share'])} "
                    f"[{100 * b['observed_share_ci95'][0]:.1f}, {100 * b['observed_share_ci95'][1]:.1f}] "
                    f"| {_pct(b['predicted_share'])} | {_f(b['ratio'])} {_ci(b['ratio_ci95'])} | "
                    f"{round(b['observed_share'] / 0.003)}x |")
    out["three_nearest"] = rows
    fl = res["all_missed_055"]["floor"]
    out["floor"] = [
        f"| missed by the source view and every other view within 25 m | "
        f"{fl['missed_by_every_view_25m']} of {_n(fl['ramps'])} ({_pct(fl['share'])}) |",
        f"| union recall on this population | {fl['union_recall_25m']:.3f} |",
        f"| missed by the source view and every other view within 18 m | "
        f"{fl['missed_by_every_view_18m']}, of which {fl['of_those_18m_found_at_18_25m']} "
        f"are found by a view at 18-25 m |",
        f"| nearest other camera, median [IQR], floor vs rest | "
        f"{fl['nearest_other_m']['floor']['median']:.1f} m "
        f"[{fl['nearest_other_m']['floor']['p25']:.1f}, {fl['nearest_other_m']['floor']['p75']:.1f}] "
        f"vs {fl['nearest_other_m']['rest']['median']:.1f} m |",
        f"| source camera, median, floor vs rest | {fl['source_dist_m']['floor']['median']:.1f} m "
        f"vs {fl['source_dist_m']['rest']['median']:.1f} m |",
        f"| other views within 25 m, median (mean), floor vs rest | "
        f"{fl['n_other_25m']['floor']['median']:.0f} ({fl['n_other_25m']['floor']['mean']:.1f}) vs "
        f"{fl['n_other_25m']['rest']['median']:.0f} ({fl['n_other_25m']['rest']['mean']:.1f}) |",
        f"| with a stored other-view peak in [0.10, 0.55) | {fl['with_stored_peak_below_floor']} |",
        "| by city | " + ", ".join(f"{k} {v}" for k, v in fl["by_city"].items()) + " |",
        "| #48 residual class | " + ", ".join(f"{k} {v}" for k, v in fl["residual_class_48"].items())
        + " |",
    ]
    rows = []
    t = {(r["spacing_m"], r["scheme"]): r["groups"] for r in res["thinning_055"]["rows"]}
    for sp in (None,) + SPACINGS:
        g = t[(sp, "native" if sp is None else "grid")]
        cells = [("native" if sp is None else f"{sp:g} m")]
        for name in ("gsv", "mapillary"):
            x = g[name]
            cells += [f"{x['panos_kept_share']:.3f}", f"{x['recall_other']:.3f}",
                      "" if sp is None else f"{x['delta_other_vs_native']:+.3f} "
                      f"{_ci(x['delta_other_vs_native_ci95'], 3)}"]
        rows.append("| " + " | ".join(cells) + " |")
    out["thinning"] = rows
    t30 = {(r["spacing_m"], r["scheme"]): r["groups"] for r in res["thinning_030_pooled4"]["rows"]}
    tu = []
    for name, tt, lab in (("mapillary", t, "richmond, 0.55, union"), ("gsv", t, "GSV, 0.55, union"),
                          ("mapillary", t30, "richmond, 0.30, other views")):
        key = "recall_union" if "union" in lab else "recall_other"
        tu.append(f"| {lab} | " + " | ".join(
            f"{tt[(sp, 'native' if sp is None else 'grid')][name][key]:.3f}"
            for sp in (None, 5.0, 10.0)) + " |")
    out["thinning_extra"] = tu
    n = res["nearest_vs_any_055"]
    rows = []
    for label, key in (("nearest other view only", "nearest_other_only"),
                       ("any other view within 6 m", "any_other_within_6m"),
                       ("any other view within 12 m", "any_other_within_12m"),
                       ("any other view within 18 m", "any_other_within_18m"),
                       ("any other view within 25 m", "any_other_within_25m")):
        x = n[key]
        rows.append(f"| {label} | {_n(x['ramps_with_a_capture'])} | {x['recall']:.3f} "
                    f"{_ci(x['ci95'], 3)} |")
    out["nearest_vs_any"] = rows

    fx = c["matched_view_count_055_fixed_population_4plus"]
    out["matched_fixed"] = [
        f"| {name} (fixed, n {_n(fx[name + '_k2']['ramps'])}) | "
        + " | ".join(f"{_f(fx[f'{name}_k{k}']['ratio'])} {_ci(fx[f'{name}_k{k}']['ratio_ci95'])}"
                     for k in (2, 3, 4)) + " |"
        for name in ("pooled", "gsv", "mapillary")]

    sp = c["pooled_055_source_pano_clusters"]
    rows = [f"| ramp clusters within city (headline) | {_n(c['pooled_055']['ramps'])} ramps | "
            f"{_f(c['pooled_055']['ratio'])} {_ci(c['pooled_055']['ratio_ci95'])} |",
            f"| source-pano clusters within city | {sp['clusters']} clusters | "
            f"{_f(sp['ratio'])} {_ci(sp['ratio_ci95'])} |"]
    for label, key in (("detected by the source view", "detected"),
                       ("added by the reviewer as missed", "reviewer_added")):
        b = c["pooled_055_by_source_hit"][key]
        rows.append(f"| {label} | {_n(b['ramps'])} ramps, {_n(b['observed_all_missed'])} vs "
                    f"{b['predicted_independent']:.1f} | {_f(b['ratio'])} {_ci(b['ratio_ci95'])} |")
    out["robustness"] = rows

    # Prose numbers in the takeaways and readings (each phrase appears in the doc verbatim).
    m = c["matched_view_count_055"]
    pc = res["per_city_055"]
    th = {(r["spacing_m"], r["scheme"]): r["groups"] for r in res["thinning_055"]["rows"]}
    t3, t8 = res["three_nearest_055"], res["three_nearest_055_hit8m"]
    gsv_cities = [k for k in sorted(SOURCE) if SOURCE[k] == "gsv"]
    gmin = min(pc[k]["mean_other_views"] for k in gsv_cities)
    gmax = max(pc[k]["mean_other_views"] for k in gsv_cities)
    others_mr = [pc[k]["other_view_miss_rate"] for k in ("paterson", "sao_paulo", "richmond")]
    rich5 = th[(5.0, "grid")]["mapillary"]
    rich10 = th[(10.0, "grid")]["mapillary"]
    out["prose"] = [
        f"[95% CI {c['pooled_055']['ratio_ci95'][0]:.2f}, {c['pooled_055']['ratio_ci95'][1]:.2f}]",
        f"it is {_f(m['pooled_k2']['ratio'])} {_ci(m['pooled_k2']['ratio_ci95'])}, on three "
        f"{_f(m['pooled_k3']['ratio'])}, on four {_f(m['pooled_k4']['ratio'])}",
        f"GSV {_f(m['gsv_k2']['ratio'])} {_ci(m['gsv_k2']['ratio_ci95'])}",
        f"Mapillary {_f(m['mapillary_k2']['ratio'])} {_ci(m['mapillary_k2']['ratio_ci95'])}",
        f"{_pct(t3['observed_share'])} [{100 * t3['observed_share_ci95'][0]:.1f}, "
        f"{100 * t3['observed_share_ci95'][1]:.1f}]",
        f"{_pct(t8['observed_share'])} [{100 * t8['observed_share_ci95'][0]:.1f}, "
        f"{100 * t8['observed_share_ci95'][1]:.1f}]",
        f"{fl['missed_by_every_view_25m']} of {_n(fl['ramps'])} ramps ({_pct(fl['share'])})",
        f"union recall {fl['union_recall_25m']:.3f}",
        f"{fl['nearest_other_m']['floor']['median']:.1f} m vs "
        f"{fl['nearest_other_m']['rest']['median']:.1f} m",
        f"costs {100 * -rich5['delta_other_vs_native']:.1f} points of other-view recall "
        f"[{100 * -rich5['delta_other_vs_native_ci95'][1]:.1f}, "
        f"{100 * -rich5['delta_other_vs_native_ci95'][0]:.1f}]",
        f"costs {100 * -rich10['delta_other_vs_native']:.1f} points",
        f"0.654 [{n['nearest_other_only']['ci95'][0]:.3f}, {n['nearest_other_only']['ci95'][1]:.3f}]"
        if round(n["nearest_other_only"]["recall"], 3) == 0.654 else "nearest-only changed",
        f"mean of {pc['mapillary']['mean_other_views']:.1f} other views within 18 m, the GSV "
        f"cities {gmin:.1f}-{gmax:.1f}",
        f"({pc['gainesville']['other_view_miss_rate']:.2f} and "
        f"{pc['bend']['other_view_miss_rate']:.2f}, vs {min(others_mr):.2f}-{max(others_mr):.2f} "
        f"elsewhere)",
        f"mean {pc['gsv']['mean_months']:.1f} distinct months",
        f"median {pc['mapillary']['median_sep_two_nearest_m']:.1f} m apart, against GSV's "
        f"{pc['gsv']['median_sep_two_nearest_m']:.1f} m",
    ]
    return out


def cmd_doc_numbers(args):
    with open(os.path.join(OUT, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    for sec, rows in doc_tables(res).items():
        print(f"## {sec}")
        for row in rows:
            print(row)


def cmd_run(args):
    payload, table = compute()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8", newline="") as f:
        f.write(dumps(payload))
    with open(os.path.join(OUT, "ramps_other_views.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(table_text(table))
    print(f"wrote {OUT}")


def cmd_check(args):
    payload, table = compute()
    # params.numpy records the version that wrote the file; it is provenance, not a
    # result, so compare against the committed value and only report a mismatch.
    with open(os.path.join(OUT, "results.json"), encoding="utf-8") as f:
        committed_np = json.load(f).get("params", {}).get("numpy")
    if committed_np is not None and committed_np != payload["params"]["numpy"]:
        print(f"note: numpy {payload['params']['numpy']} here, {committed_np} in the "
              "committed file; params.numpy is not compared")
        payload["params"]["numpy"] = committed_np
    ok = True
    for name, text in (("results.json", dumps(payload)),
                       ("ramps_other_views.csv", table_text(table))):
        with open(os.path.join(OUT, name), encoding="utf-8", newline="") as f:
            same = f.read() == text
        print(f"{name}: {'identical' if same else 'DIFFERS'}")
        ok &= same
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("run").set_defaults(fn=cmd_run)
    sp.add_parser("check").set_defaults(fn=cmd_check)
    sp.add_parser("doc-numbers").set_defaults(fn=cmd_doc_numbers)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
