"""Multi-view evidence for per-ramp recall and precision (#48, Phase 1).

sidewalk-auto-labeler#27 built the multi-view layer #48 proposed (raycast, association,
world-space scoring) and measured what it buys: deduplication and a tighter p90
placement, not recall, because production already submits every operational detection
from every pano, so the union of views is what a city receives today. This script asks
the questions that work left open, on the same five cities and with the labeler's own
code, so every number ties back to ``runs/<city>/fusion_eval/report.md``:

1. **Recall vs number of qualifying captures.** For every world GT ramp in eval_sites'
   recall pool, every run pano whose camera is within R of the ramp is a *qualifying
   capture*. Each is checked for a stored detection at the ramp -- in world space (its
   raycast lands within 5 m) and in pixel space (it lies within the benchmark's 0.022
   match radius of the ramp projected into that pano) -- at >= 0.55, >= 0.30 and the
   0.10 storage floor. Recall is then read as a function of how many captures a ramp
   has, and of how many of the nearest ones are used.
2. **Failure correlation** (#38's open checkbox): are misses in two views of one ramp
   independent? Observed joint-miss rates against the product of range-matched
   marginals, by camera separation and by range, plus the share of ramps that every
   qualifying view missed.
3. **Evidence accumulation vs k-of-n at 0.30**, on the cities that store sub-threshold
   detections: world P/R under the operational 0.55 fuse, a flat 0.30, k-of-n promotion
   of 0.30-tier sites, and a per-site log-likelihood score that also counts
   misses-in-range as negative evidence (calibrated in-sample, and leave-one-city-out).
4. **Residual-miss taxonomy**: the pool ramps no operational site recovered, classified
   mechanically, plus a crop gallery for a one-rater qualitative pass
   (``residual_taxonomy__jonf.json``; verdicts empty until a human fills them).

Nothing here needs a GPU or the network. It needs the labeler checkout (read-only; its
``geo``, ``fuse_sites`` and ``eval_sites`` are imported by path, never copied) and its
``runs/<city>/results.jsonl`` files, which are local artifacts of that repo.

    python scripts/analysis/multiview_evidence_48.py run --labeler-root ../sidewalk-auto-labeler
    python scripts/analysis/multiview_evidence_48.py figures
    python scripts/analysis/multiview_evidence_48.py crop-plan --labeler-root ../sidewalk-auto-labeler
    python scripts/analysis/multiview_evidence_48.py cut-crops PLAN.json --archive-root DIR --out DIR   # makelab2
    python scripts/analysis/multiview_evidence_48.py gallery --crops DIR

``run`` checks, before writing anything, that eval_sites reproduces each city's committed
report (world recall, precision, buckets) and refuses otherwise.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (  # noqa: E402
    PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y, build_ground_truth, radius_sq_for)
from rampnet.geometry import dist_sq  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

OUT_ROOT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
OUT = os.path.join(OUT_ROOT, "multiview_48")
FIG_DIR = os.path.join(REPO, "docs", "figures", "multiview_48")
BENCHMARK = os.path.join(REPO, "benchmark")

CITIES = ("richmond", "paterson", "gainesville", "bend", "sao_paulo")
SOURCE = {"richmond": "mapillary", "paterson": "gsv", "gainesville": "gsv",
          "bend": "gsv", "sao_paulo": "gsv"}
#: Runs whose results.jsonl stores every peak down to the 0.10 floor (labeler #27 stage 1).
STORED_SUBTHRESHOLD = ("paterson", "gainesville", "sao_paulo")
#: Richmond's results.jsonl stops at 0.55, but the labeler re-inferred every Richmond pano
#: at the 0.10 floor into results.f01.jsonl (scripts/reinfer.py, 2026-09-22). Those are
#: fresh forward passes, so their confidences are new numbers; the >= 0.55 set matches
#: results.jsonl's in (x, y) on 9,089 of 9,091 panos. load_city keeps results.jsonl's
#: operational detections exactly (the verdicts are keyed to them) and adds only the
#: re-inferred detections below 0.55 -- see augment_with_reinfer. Bend has no such file.
REINFER_FILE = {"richmond": "results.f01.jsonl"}

BENCHMARK_CONFIDENCE = 0.55
TIER_030 = 0.30
STORAGE_FLOOR = 0.10
HIT_FLOORS = (0.55, 0.30, 0.10)
MATCH_RADIUS_M = 5.0          # eval_sites' default world match radius
GT_MERGE_M = 2.5              # eval_sites' default cross-pano GT merge
CAMERA_HEIGHT_M = 2.6         # every committed fusion_eval report raycasts at 2.6 m
R_DEFAULT = 18.0
R_SWEEP = (12.0, 18.0, 25.0)
R_MAX = 25.0                  # geo.DEFAULT_MAX_RANGE_M: nothing beyond it is raycast
KMAX = 8
RANGE_BINS = ((0.0, 6.0), (6.0, 12.0), (12.0, 18.0), (18.0, 25.0))
SEP_BINS = ((0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 25.0), (25.0, 51.0))
CONF_BINS = (0.10, 0.20, 0.30, 0.45, 0.55, 0.75, 1.0001)
N_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6))

LABELER_FILES = ("geo.py", "depth.py", "detectors/__init__.py",
                 "scripts/fuse_sites.py", "scripts/eval_sites.py")

# --------------------------------------------------------------------------- #
# Pure helpers (no labeler import) -- these are what tests/ exercise
# --------------------------------------------------------------------------- #


def wilson(k, n, z=1.96):
    """Wilson score interval, the same form eval_sites and rampnet.validation use."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rnd(v, nd=4):
    """Round floats (recursively) so committed JSON does not depend on the float repr
    of one numpy / platform build."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def write_json(path, payload):
    """Committed-artifact writer: LF on every platform, sorted keys, rounded floats."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(rnd(payload), indent=1, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([("" if v is None else (round(v, 4) if isinstance(v, float) else v))
                        for v in r])
    return path


#: One row per (pool ramp, qualifying capture within 25 m). cam_e / cam_n are the camera
#: in the city's eval_sites LocalFrame (metres), so camera separations re-derive.
CAPTURE_COLUMNS = ["city", "ramp_uid", "pano_id", "dist_m", "is_source", "x_proj", "y_proj",
                   "world_conf", "pixel_conf", "capture_date", "cam_e", "cam_n"]


def ramps_from_capture_csv(path):
    """{city: [{"uid", "captures": [...]}, ...]} rebuilt from ``captures_R25.csv`` -- the
    input B.1 and B.2 are computed from, so the committed CSV re-derives them."""
    def f(v):
        return None if v == "" else float(v)
    by_city = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ramp = by_city[row["city"]].setdefault(row["ramp_uid"],
                                                   {"uid": row["ramp_uid"], "captures": []})
            ramp["captures"].append({
                "pano_id": row["pano_id"], "dist_m": float(row["dist_m"]),
                "is_source": row["is_source"] == "1", "world_conf": f(row["world_conf"]),
                "pixel_conf": f(row["pixel_conf"]), "cam_e": float(row["cam_e"]),
                "cam_n": float(row["cam_n"])})
    return {c: [r for _, r in sorted(rs.items(), key=lambda t: int(t[0].split(":")[1]))]
            for c, rs in by_city.items()}


def pixel_radius_sq(radius_norm=PANO_RADIUS_NORMALIZED):
    return radius_sq_for(radius_norm)


def pixel_best_conf(dets, x, y, radius_sq=None):
    """Highest confidence among ``dets`` [(x, y, conf), ...] within the benchmark's
    per-pano match radius of (x, y), wrapped at the seam; None if none is."""
    radius_sq = pixel_radius_sq() if radius_sq is None else radius_sq
    best = None
    for dx, dy, c in dets:
        if dist_sq(dx, dy, x, y, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True) < radius_sq:
            best = c if best is None or c > best else best
    return best


def world_best_conf(ground, e, n, radius_m=MATCH_RADIUS_M):
    """Highest confidence among raycast detections ``ground`` [(e, n, conf), ...] whose
    ground point lies within ``radius_m`` of (e, n); None if none does."""
    best = None
    r2 = radius_m * radius_m
    for ge, gn, c in ground:
        if (ge - e) ** 2 + (gn - n) ** 2 <= r2:
            best = c if best is None or c > best else best
    return best


def claim_by_confidence(dets, targets, dist_sq_fn, radius_sq):
    """One-to-one claims within one capture, the way score_pano matches: detections in
    descending confidence each claim the nearest unclaimed target strictly within the
    radius. ``dets`` [(x, y, conf)], ``targets`` [(tid, x, y)]. Returns {tid: conf of the
    claiming detection}.

    Because claims are made in confidence order, the claims made by detections at or
    above any floor are the same whether or not lower-confidence detections exist, so one
    pass answers every floor: target t is hit at floor f iff its claim conf >= f. That is
    what keeps one detection between two dual ramps from counting as a hit for both."""
    out = {}
    for x, y, c in sorted(dets, key=lambda d: -d[2]):
        best = None
        for tid, tx, ty in targets:
            if tid in out:
                continue
            d2 = dist_sq_fn(x, y, tx, ty)
            if d2 < radius_sq and (best is None or (d2, tid) < best):
                best = (d2, tid)
        if best is not None:
            out[best[1]] = c
    return out


def world_d2(x, y, tx, ty):
    return (x - tx) ** 2 + (y - ty) ** 2


def pixel_d2(x, y, tx, ty):
    return dist_sq(x, y, tx, ty, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)


def hit(conf, floor):
    return conf is not None and conf >= floor - 1e-12


def capture_hit(cap, floor, mode):
    """Did this capture see the ramp at ``floor``? mode: world | pixel | either."""
    w = hit(cap["world_conf"], floor)
    p = hit(cap["pixel_conf"], floor)
    return w if mode == "world" else p if mode == "pixel" else (w or p)


def bin_of(value, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= value < hi:
            return i
    return None


def n_bin_label(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def recall_by_capture_count(ramps, floor, mode, radius):
    """Per bin of n (number of NON-source qualifying captures within ``radius``): the
    share of ramps any of those captures saw, and the share seen by the source view or
    any other (the union production already submits).

    ``ramps`` is a list of {"captures": [cap, ...]} with cap["dist_m"], cap["is_source"].
    """
    rows = []
    for lo, hi in N_BINS:
        sel = []
        for r in ramps:
            others = [c for c in r["captures"] if not c["is_source"] and c["dist_m"] <= radius]
            if lo <= len(others) <= hi:
                sel.append((r, others))
        n = len(sel)
        k_other = sum(1 for r, o in sel if any(capture_hit(c, floor, mode) for c in o))
        k_union = sum(1 for r, o in sel
                      if any(capture_hit(c, floor, mode) for c in r["captures"]
                             if c["is_source"] or c["dist_m"] <= radius))
        rows.append({"n_bin": n_bin_label(lo, hi), "ramps": n,
                     "recall_other": k_other / n if n else None,
                     "recall_other_ci": wilson(k_other, n) if n else None,
                     "recall_union": k_union / n if n else None})
    return rows


def recall_k_nearest(ramps, floor, mode, radius, kmax=KMAX, fixed_population=True):
    """Recall when only the k nearest non-source captures within ``radius`` are used.

    ``fixed_population`` restricts to ramps with >= kmax such captures, so every point on
    the curve is the same set of ramps (no composition drift); otherwise ramps with
    fewer captures than k use all they have."""
    pop = []
    for r in ramps:
        others = sorted((c for c in r["captures"]
                         if not c["is_source"] and c["dist_m"] <= radius),
                        key=lambda c: (c["dist_m"], c["pano_id"]))
        if fixed_population and len(others) < kmax:
            continue
        if others:
            pop.append(others)
    rows = []
    for k in range(1, kmax + 1):
        n = len(pop)
        kk = sum(1 for o in pop if any(capture_hit(c, floor, mode) for c in o[:k]))
        rows.append({"k": k, "ramps": n, "recall": kk / n if n else None,
                     "ci": wilson(kk, n) if n else None})
    return rows


def failure_correlation(ramps, floor, mode, radius, range_bins=RANGE_BINS,
                        sep_bins=SEP_BINS):
    """Are misses in two qualifying (non-source) views of one ramp independent?

    The independence prediction for a pair is the product of the two views' marginal
    miss rates *at their ranges* (range-binned over every capture in the population), so
    'both views were far' is not counted as correlation. Returns the marginals, the
    per-separation-bin and per-range-bin observed vs predicted joint-miss rates and
    P(miss_j | miss_i), and the all-views-missed count against its prediction."""
    caps = [(r, [c for c in r["captures"] if not c["is_source"] and c["dist_m"] <= radius])
            for r in ramps]
    caps = [(r, cs) for r, cs in caps if len(cs) >= 2]
    tot = defaultdict(int)
    miss = defaultdict(int)
    for _, cs in caps:
        for c in cs:
            b = bin_of(c["dist_m"], range_bins)
            tot[b] += 1
            miss[b] += not capture_hit(c, floor, mode)
    p_miss = {b: (miss[b] / tot[b]) if tot[b] else None for b in tot}

    def pm(c):
        return p_miss[bin_of(c["dist_m"], range_bins)]

    sep_acc = defaultdict(lambda: [0, 0.0, 0, 0, 0.0])   # pairs, pred_both, obs_both, obs_mi, pred_marg_j|i
    rng_acc = defaultdict(lambda: [0, 0.0, 0])
    for _, cs in caps:
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                sep = math.hypot(a["cam_e"] - b["cam_e"], a["cam_n"] - b["cam_n"])
                sb = bin_of(sep, sep_bins)
                ma = not capture_hit(a, floor, mode)
                mb = not capture_hit(b, floor, mode)
                acc = sep_acc[sb]
                acc[0] += 1
                acc[1] += pm(a) * pm(b)
                acc[2] += ma and mb
                acc[3] += ma + mb                       # misses, for P(miss_j | miss_i)
                acc[4] += pm(a) + pm(b)
                rb = bin_of(max(a["dist_m"], b["dist_m"]), range_bins)
                racc = rng_acc[rb]
                racc[0] += 1
                racc[1] += pm(a) * pm(b)
                racc[2] += ma and mb

    def sep_row(sb, acc):
        pairs, pred, obs, n_miss, pred_miss = acc
        return {"bin": None if sb is None else list(sep_bins[sb]), "pairs": pairs,
                "obs_both_miss": obs / pairs if pairs else None,
                "pred_both_miss": pred / pairs if pairs else None,
                "ratio": (obs / pred) if pred else None,
                # P(miss_j | miss_i), symmetrized over the pair's two orders, against
                # the marginal the same views would have under independence
                "p_miss_given_miss": (2 * obs / n_miss) if n_miss else None,
                "p_miss_marginal": (pred_miss / (2 * pairs)) if pairs else None}

    all_obs = all_pred = 0.0
    by_n = defaultdict(lambda: [0, 0, 0.0])
    for _, cs in caps:
        allm = all(not capture_hit(c, floor, mode) for c in cs)
        pred = 1.0
        for c in cs:
            pred *= pm(c)
        all_obs += allm
        all_pred += pred
        nb = n_bin_label(*next(b for b in N_BINS if b[0] <= len(cs) <= b[1]))
        by_n[nb][0] += 1
        by_n[nb][1] += allm
        by_n[nb][2] += pred
    return {
        "ramps": len(caps),
        "marginal_miss_by_range": [{"bin": list(range_bins[b]), "captures": tot[b],
                                    "p_miss": p_miss[b]} for b in sorted(tot, key=lambda x: (x is None, x))],
        "by_separation": [sep_row(sb, acc) for sb, acc in sorted(sep_acc.items(), key=lambda t: (t[0] is None, t[0]))],
        "by_range": [{"bin": None if rb is None else list(range_bins[rb]), "pairs": a[0],
                      "obs_both_miss": a[2] / a[0] if a[0] else None,
                      "pred_both_miss": a[1] / a[0] if a[0] else None,
                      "ratio": (a[2] / a[1]) if a[1] else None}
                     for rb, a in sorted(rng_acc.items(), key=lambda t: (t[0] is None, t[0]))],
        "all_missed": {"observed": int(all_obs), "predicted_independent": all_pred,
                       "ratio": (all_obs / all_pred) if all_pred else None},
        "all_missed_by_n": [{"n_bin": k, "ramps": v[0], "observed": v[1],
                             "predicted_independent": v[2]}
                            for k, v in sorted(by_n.items(), key=lambda t: int(t[0].rstrip("+").split("-")[0]))],
    }


def classify_preds(preds, gt):
    """Per-prediction outcome in one judged pano, exactly as rampnet.detection_eval.
    score_pano decides it: ``preds`` [(key, x, y, conf-or-None)], ``gt`` a GroundTruth.

    Returns {key: ("tp", gt_index) | ("fp", None) | ("ignored", None)}."""
    radius_sq = radius_sq_for()
    confs = [p[3] for p in preds]
    if any(c is not None for c in confs):
        order = sorted(range(len(preds)),
                       key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                       reverse=True)
        preds = [preds[i] for i in order]
    assign = greedy_match([(p[1], p[2]) for p in preds], gt.gt_points, radius_sq,
                          PANO_SCALE_X, PANO_SCALE_Y, True)
    out = {}
    for p, (gi, _) in zip(preds, assign):
        if gi >= 0:
            out[p[0]] = ("tp", gi)
        elif any(dist_sq(p[1], p[2], ix, iy, PANO_SCALE_X, PANO_SCALE_Y, True) < radius_sq
                 for ix, iy in gt.ignore_points):
            out[p[0]] = ("ignored", None)
        else:
            out[p[0]] = ("fp", None)
    return out


def match_one_to_one(ramps_xy, sites_xy, radius_m):
    """Greedy ascending-distance one-to-one matching, identical in rule to
    eval_sites.match_one_to_one: ``ramps_xy`` [(e, n)], ``sites_xy`` [(site_id, e, n)].
    Returns {ramp_index: site_id}."""
    cell = max(radius_m, 1e-9)
    grid = defaultdict(list)
    for sid, se, sn in sites_xy:
        grid[(math.floor(se / cell), math.floor(sn / cell))].append((sid, se, sn))
    pairs = []
    for gi, (re_, rn) in enumerate(ramps_xy):
        kx, ky = math.floor(re_ / cell), math.floor(rn / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sid, se, sn in grid.get((kx + dx, ky + dy), ()):
                    d = math.hypot(re_ - se, rn - sn)
                    if d <= radius_m:
                        pairs.append((d, gi, sid))
    pairs.sort()
    matched, used = {}, set()
    for d, gi, sid in pairs:
        if gi in matched or sid in used:
            continue
        matched[gi] = sid
        used.add(sid)
    return matched


def score_world(pool, sites, accepted, pano_class, match_radius_m=MATCH_RADIUS_M):
    """World-space P/R with eval_sites' definitions, generalized to any detector.

    ``pool``: [{"e", "n", "gt_refs": [(pano_id, gt_index), ...]}] -- the recall pool.
    ``sites``: [{"id", "e", "n", "members": [(pano_id, det_key, conf), ...]}].
    ``accepted``: set of site ids the policy submits.
    ``pano_class``: {(pano_id, det_key): ("tp", gi) | ("fp", None) | ("ignored", None)}
    for every scored detection in a judged pano (at the policy's tier).

    Recall: a pool ramp is recalled when one of its judged-pano GT points is claimed
    by a detection whose site is accepted (eval_sites' ``self_detected``), or when an
    accepted site lies within ``match_radius_m`` under one-to-one matching. Precision:
    over accepted sites with a scored member in a judged pano -- TP if any such member
    is a TP, FP if all decided members are FPs, excluded if only ignored."""
    site_of = {}
    for s in sites:
        for pid, key, _ in s["members"]:
            site_of[(pid, key)] = s["id"]
    claimed = {}
    for (pid, key), (kind, gi) in pano_class.items():
        if kind == "tp" and site_of.get((pid, key)) in accepted:
            claimed[(pid, gi)] = True
    acc_sites = [(s["id"], s["e"], s["n"]) for s in sites if s["id"] in accepted]
    matched = match_one_to_one([(r["e"], r["n"]) for r in pool], acc_sites, match_radius_m)
    self_hits = [any((pid, gi) in claimed for pid, gi in r["gt_refs"]) for r in pool]
    recalled = [sh or (i in matched) for i, sh in enumerate(self_hits)]
    tp = fp = unsure = 0
    for s in sites:
        if s["id"] not in accepted:
            continue
        cls = [pano_class[(pid, key)][0] for pid, key, _ in s["members"]
               if (pid, key) in pano_class]
        if not cls:
            continue
        if "tp" in cls:
            tp += 1
        elif "fp" in cls:
            fp += 1
        else:
            unsure += 1
    n = len(pool)
    k = sum(recalled)
    p = tp / (tp + fp) if tp + fp else None
    r = k / n if n else None
    return {"recall": r, "recall_ci": wilson(k, n), "recalled": k, "pool": n,
            "self": sum(self_hits), "precision": p, "precision_ci": wilson(tp, tp + fp),
            "tp": tp, "fp": fp, "unsure_only": unsure,
            "f1": (2 * p * r / (p + r)) if p and r else None,
            "accepted_sites": len(accepted), "recalled_flags": recalled}


def site_tier_panos(site, tier):
    """Distinct panos contributing a member at or above ``tier``."""
    return len({pid for pid, _, c in site["members"] if c is not None and c >= tier - 1e-12})


def site_max_conf(site):
    cs = [c for _, _, c in site["members"] if c is not None]
    return max(cs) if cs else None


def policy_kofn(sites, k, tier=TIER_030, keep=BENCHMARK_CONFIDENCE):
    """Accept a site with any member >= ``keep``, or >= k distinct panos at >= ``tier``."""
    out = set()
    for s in sites:
        mc = site_max_conf(s)
        if mc is not None and mc >= keep - 1e-12:
            out.add(s["id"])
        elif site_tier_panos(s, tier) >= k:
            out.add(s["id"])
    return out


def conf_bin(c, bins=CONF_BINS):
    for i in range(len(bins) - 1):
        if bins[i] - 1e-12 <= c < bins[i + 1]:
            return i
    return len(bins) - 2


def fit_evidence_model(examples, range_bins=RANGE_BINS, conf_bins=CONF_BINS, alpha=1.0):
    """Naive-Bayes per-capture likelihoods for real vs false sites.

    ``examples``: [(label, [(dist_m, conf_or_None), ...])] -- label True (a real site)
    or False, and for each qualifying capture its range to the site and the site's
    member confidence from that pano (None = a miss in range). Per class:
    P(hit | range bin) with add-``alpha`` smoothing, and P(conf bin | hit) pooled over
    range. Returns a dict usable by ``evidence_score``."""
    nr, nc = len(range_bins), len(conf_bins) - 1
    cnt = {cls: {"n": [0] * nr, "hit": [0] * nr, "cb": [0] * nc} for cls in (True, False)}
    n_sites = {True: 0, False: 0}
    for label, caps in examples:
        n_sites[label] += 1
        for d, c in caps:
            rb = bin_of(d, range_bins)
            if rb is None:
                continue
            cnt[label]["n"][rb] += 1
            if c is not None:
                cnt[label]["hit"][rb] += 1
                cnt[label]["cb"][conf_bin(c, conf_bins)] += 1
    model = {"range_bins": [list(b) for b in range_bins], "conf_bins": list(conf_bins),
             "alpha": alpha, "n_sites": {"real": n_sites[True], "false": n_sites[False]}}
    for cls, name in ((True, "real"), (False, "false")):
        c = cnt[cls]
        p_hit = [(c["hit"][i] + alpha) / (c["n"][i] + 2 * alpha) for i in range(nr)]
        tot = sum(c["cb"])
        p_cb = [(c["cb"][i] + alpha) / (tot + nc * alpha) for i in range(nc)]
        model[name] = {"captures": c["n"], "hits": c["hit"], "conf_hist": c["cb"],
                       "p_hit": p_hit, "p_conf_given_hit": p_cb}
    return model


def capture_llr(model, dist_m, conf):
    """log P(outcome | real) / P(outcome | false) for one capture; 0 outside the bins."""
    rb = bin_of(dist_m, [tuple(b) for b in model["range_bins"]])
    if rb is None:
        return 0.0
    R, F = model["real"], model["false"]
    if conf is None:
        return math.log(1 - R["p_hit"][rb]) - math.log(1 - F["p_hit"][rb])
    cb = conf_bin(conf, model["conf_bins"])
    return (math.log(R["p_hit"][rb] * R["p_conf_given_hit"][cb])
            - math.log(F["p_hit"][rb] * F["p_conf_given_hit"][cb]))


def evidence_score(model, caps):
    """Sum of per-capture log-likelihood ratios: hits AND misses-in-range both count."""
    return sum(capture_llr(model, d, c) for d, c in caps)


def pr_counts(res):
    return {k: res[k] for k in ("recall", "recall_ci", "recalled", "pool", "precision",
                                "precision_ci", "tp", "fp", "unsure_only", "f1",
                                "accepted_sites", "self")}


def pool_results(rows):
    """Sum per-city score_world outputs into one pooled row (micro)."""
    k = sum(r["recalled"] for r in rows)
    n = sum(r["pool"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    p = tp / (tp + fp) if tp + fp else None
    rc = k / n if n else None
    return {"recall": rc, "recall_ci": wilson(k, n), "recalled": k, "pool": n,
            "precision": p, "precision_ci": wilson(tp, tp + fp), "tp": tp, "fp": fp,
            "unsure_only": sum(r["unsure_only"] for r in rows),
            "f1": (2 * p * rc / (p + rc)) if p and rc else None,
            "accepted_sites": sum(r["accepted_sites"] for r in rows),
            "self": sum(r["self"] for r in rows)}


def residual_class(ramp, stored_subthreshold, coverage_radius=R_DEFAULT):
    """Mechanical cause for a pool ramp that no operational site recovered.

    Precedence: a candidate at >= 0.55 in any capture (world or pixel test) means the
    detector fired and association/placement lost it; else a stored candidate in
    [0.10, 0.55) means sub-threshold-only; else no non-source capture within
    ``coverage_radius`` is a coverage gap; else nothing fired anywhere. In a run with no
    sub-threshold storage the last two cannot see below 0.55, so they say so."""
    caps = ramp["captures"]
    if any(capture_hit(c, BENCHMARK_CONFIDENCE, "either") for c in caps):
        return "association_placement"
    if stored_subthreshold and any(capture_hit(c, STORAGE_FLOOR, "either") for c in caps):
        return "sub_threshold_only"
    others = [c for c in caps if not c["is_source"] and c["dist_m"] <= coverage_radius]
    suffix = "" if stored_subthreshold else "_unknown_below_055"
    if not others:
        return "coverage_gap" + suffix
    return "never_fired" + suffix


# --------------------------------------------------------------------------- #
# Labeler-bound code
# --------------------------------------------------------------------------- #


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_labeler(root):
    """Import the labeler's geo / fuse_sites / eval_sites from ``root`` by path, and
    record exactly which bytes were imported (the file sha256s, plus the git commit when
    ``root`` is a checkout), because the numbers depend on them."""
    root = os.path.abspath(root)
    for p in (os.path.join(root, "scripts"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_sites as es  # noqa: E402
    import fuse_sites as fs  # noqa: E402
    import geo  # noqa: E402
    for mod in (geo, fs, es):
        if not os.path.abspath(mod.__file__).startswith(root):
            raise SystemExit(f"{mod.__name__} imported from {mod.__file__}, not {root}: "
                             "another module of that name is on sys.path")
    prov = {"files": {f: _sha256(os.path.join(root, f)) for f in LABELER_FILES
                      if os.path.exists(os.path.join(root, f))}}
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--",
                                *LABELER_FILES], capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            prov["git_commit"] = head.stdout.strip()
            prov["git_dirty_files"] = [ln[3:] for ln in dirty.stdout.splitlines()]
    except (OSError, subprocess.SubprocessError):
        pass
    return SimpleNamespace(geo=geo, fs=fs, es=es, prov=prov)


def fuse_params(L, min_confidence=BENCHMARK_CONFIDENCE, floor=STORAGE_FLOOR):
    """The FuseParams every committed fusion_eval report was produced under, pinned
    field by field so a labeler default changing (e.g. #79's camera-height `auto`, which
    is CLI-only) cannot move these numbers."""
    return L.fs.FuseParams(floor=floor, min_confidence=min_confidence, mask_rig=False,
                           camera_height_m=CAMERA_HEIGHT_M, apply_pose=L.fs.POSE_OFF)


def augment_with_reinfer(run_panos, reinfer_path, keep_below=BENCHMARK_CONFIDENCE):
    """Add a re-inference's sub-``keep_below`` detections to each pano (Richmond).

    Operational detections stay exactly as results.jsonl has them (the verdicts are
    keyed to their indices); the re-inferred ones below ``keep_below`` are appended
    with indices 1000+i. A pano whose re-inferred >= keep_below set does not match the
    original in (x, y) gets nothing added -- its frame of reference moved."""
    re = {}
    with open(reinfer_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                re[r["pano"]["panorama_id"]] = r["detections"]
    added = skipped = 0
    for p in run_panos:
        dets = re.get(p.pano_id)
        if dets is None:
            skipped += 1
            continue
        hi = sorted((round(d["x_normalized"], 9), round(d["y_normalized"], 9))
                    for d in dets if d["confidence"] >= keep_below)
        orig = sorted((round(x, 9), round(y, 9)) for _, x, y, c in p.detections
                      if c >= keep_below)
        if hi != orig:
            skipped += 1
            continue
        extra = [(1000 + i, d["x_normalized"], d["y_normalized"], d["confidence"])
                 for i, d in enumerate(dets) if d["confidence"] < keep_below]
        p.detections = list(p.detections) + extra
        added += len(extra)
    return {"reinfer_file": os.path.basename(reinfer_path), "detections_added": added,
            "panos_not_augmented": skipped}


#: sha256 of the results.jsonl each committed fusion_eval report was produced on: the
#: copies in the labeler's native-res archive on makelab2
#: (/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/results.jsonl, archived
#: 2026-08-01..05 with the imagery). The labeler checkout's own runs/ have since been
#: gap-filled for paterson (+260 records), gainesville (+2,231) and sao_paulo (+7,293), so
#: they no longer match the reports; richmond and bend are unchanged.
ARCHIVED_RESULTS_SHA256 = {
    "richmond": "109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c",
    "bend": "1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153",
    "paterson": "ba987cfde606ba6ed04b877a4fe95f32383729ecc68be56fa402adb56c86da00",
    "gainesville": "bb8a78729cfb97a98a4f99211c90763fb3760c9c216f2233b1c0cc8aa450abde",
    "sao_paulo": "72fa99b69ce2742c03becf427ca1674c2a0971531cc4f84ad793d5dc3a2fe926",
}


def results_dir(city, runs_root, results_root=None):
    """Where a city's results.jsonl is read from: ``results_root/<city>`` when given and
    present (the archived copies), else the labeler's ``runs/<city>``."""
    if results_root and os.path.exists(os.path.join(results_root, city, "results.jsonl")):
        return os.path.join(results_root, city)
    return os.path.join(runs_root, city)


def load_city(L, city, runs_root, benchmark_root=BENCHMARK, results_root=None,
              require_archived=True):
    """Verdicts, bundle ops and run panos for one city, exactly as eval_sites loads them
    (no re-inference added; see apply_reinfer). Refuses a results.jsonl that is not the
    archived one the committed report scored, unless ``require_archived`` is False."""
    from pathlib import Path
    run_dir = Path(results_dir(city, runs_root, results_root))
    sha = _sha256(run_dir / "results.jsonl")
    if require_archived and sha != ARCHIVED_RESULTS_SHA256.get(city):
        raise SystemExit(f"{city}: {run_dir / 'results.jsonl'} (sha256 {sha[:12]}...) is not "
                         "the archived run the committed report scored; pass --results-root "
                         "pointing at copies of the makelab2 archive's results.jsonl")
    verdicts, bundle_ops, run_panos = L.es.load_city_files(
        city, Path(benchmark_root), run_dir, read_heights=False)
    info = {"run_panos": len(run_panos), "stored_subthreshold": city in STORED_SUBTHRESHOLD,
            "results_sha256": sha}
    return verdicts, bundle_ops, run_panos, info


def apply_reinfer(city, runs_root, run_panos, info):
    """Richmond only: add results.f01.jsonl's sub-0.55 detections (augment_with_reinfer)."""
    if city not in REINFER_FILE:
        return False
    path = os.path.join(runs_root, city, REINFER_FILE[city])
    if not os.path.exists(path):
        return False
    info["reinfer"] = augment_with_reinfer(run_panos, path)
    info["reinfer"]["sha256"] = _sha256(path)
    info["stored_subthreshold"] = True
    return True


def read_bundle_records(city, benchmark_root=BENCHMARK):
    recs = {}
    with open(os.path.join(benchmark_root, city, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pano"]["panorama_id"]] = r
    return recs


def parse_report(path):
    """World recall / precision / buckets from a committed fusion_eval/report.md."""
    import re
    text = open(path, encoding="utf-8").read()
    out = {}
    m = re.search(r"world recall\s+([0-9.]+)", text)
    out["world_recall"] = float(m.group(1))
    m = re.search(r"world precision\s+([0-9.]+).*?TP (\d+), FP (\d+)", text)
    out["precision"], out["tp"], out["fp"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"buckets\s+self (\d+), other-view (\d+), subthreshold-only (\d+), unmatched (\d+)", text)
    out["buckets"] = {"self_detected": int(m.group(1)), "recovered_other_view": int(m.group(2)),
                      "subthreshold_only": int(m.group(3)), "unmatched": int(m.group(4))}
    m = re.search(r"note: (\d+) self-detected ramps had no operational site", text)
    out["self_detected_without_site"] = int(m.group(1)) if m else 0
    return out


def world_gt(L, city, verdicts, bundle_ops, run_panos, params, prefused=None):
    """eval_sites' world GT and recall pool for one city, plus what this script needs
    on top: each pool ramp's (pano_id, gt_index) references into the per-pano GroundTruth
    (so any detector can claim it), its bucket, and eval_sites' own result dict."""
    es, fs, geo = L.es, L.fs, L.geo
    sites, frame, stats = prefused or fs.fuse(run_panos, params)
    result = es.evaluate_city(verdicts, bundle_ops, run_panos, params,
                              match_radius_m=MATCH_RADIUS_M, gt_merge_m=GT_MERGE_M,
                              prefused=(sites, frame, stats))
    run_by_id = {p.pano_id: p for p in run_panos}
    points, op_verdicts, counts, _ = es.build_gt(verdicts, bundle_ops, run_by_id, params, frame)
    ramps = es.merge_gt_points(points, GT_MERGE_M)
    pool = [r for r in ramps if r.in_pool]
    op_sites = [s for s in sites if s.n_operational > 0]
    sub_sites = [s for s in sites if s.n_operational == 0]
    matched_op = es.match_one_to_one(pool, op_sites, MATCH_RADIUS_M)
    unmatched_idx = [i for i in range(len(pool)) if i not in matched_op]
    matched_sub = es.match_one_to_one([pool[i] for i in unmatched_idx], sub_sites, MATCH_RADIUS_M)
    sub_hits = {unmatched_idx[j] for j in matched_sub}

    # (pano_id, gt_index) for every placed GT point, by re-placing each per-pano GT point
    # exactly as build_gt does and finding the GTPoint it produced.
    records = read_bundle_records(city)
    by_pano = defaultdict(list)
    for pt in points:
        by_pano[pt.pano_id].append(pt)
    ref_of_point = {}
    judged_gt = {}
    for pid, entry in verdicts.items():
        if pid not in by_pano and pid not in run_by_id:
            continue
        rec = records[pid]
        gt = build_ground_truth(rec["detections"], entry["dets"], entry.get("missed", ()),
                                entry.get("no_missed"))
        judged_gt[pid] = gt
        rp = run_by_id.get(pid)
        if rp is None:
            continue
        pose = fs.pano_pose(rp, params.apply_pose)
        errors = geo.error_model_for(rp.source)
        for gi, (x, y) in enumerate(gt.gt_points):
            g = geo.detection_ground_point(pose, x, y, camera_height=params.camera_height_m,
                                           max_range_m=params.max_range_m, errors=errors,
                                           apply_pose=params.rotates)
            if g is None:
                continue
            e, n = frame.to_enu(g.lat, g.lng)
            for pt in by_pano[pid]:
                if abs(pt.e - e) < 1e-6 and abs(pt.n - n) < 1e-6 and id(pt) not in ref_of_point:
                    ref_of_point[id(pt)] = (pid, gi)
                    break
    missing_refs = sum(1 for pt in points if id(pt) not in ref_of_point)
    if missing_refs:
        raise SystemExit(f"{city}: {missing_refs} GT points could not be tied back to a "
                         "per-pano GT index; the placement here drifted from build_gt's")

    out = []
    for i, r in enumerate(pool):
        if r.self_detected:
            bucket = "self_detected"
        elif i in matched_op:
            bucket = "recovered_other_view"
        elif i in sub_hits:
            bucket = "subthreshold_only"
        else:
            bucket = "unmatched"
        lat, lng = frame.to_latlng(r.e, r.n)
        near_op = min((math.hypot(s.e - r.e, s.n - r.n) for s in op_sites), default=None)
        out.append({"uid": f"{city}:{i}", "index": i, "e": r.e, "n": r.n, "lat": lat, "lng": lng,
                    "source_panos": sorted(r.pano_ids),
                    "kinds": sorted({p.kind for p in r.points}),
                    "gt_refs": sorted(ref_of_point[id(p)] for p in r.points),
                    "bucket": bucket,
                    "self_detected_without_site": bool(r.self_detected and i not in matched_op),
                    "nearest_op_site_m": near_op})
    return SimpleNamespace(pool=out, sites=sites, frame=frame, stats=stats, result=result,
                           judged_gt=judged_gt, counts=counts, op_verdicts=op_verdicts)


def camera_index(L, frame, run_panos, cell=R_MAX):
    grid = L.geo.GridIndex(cell)
    cams = {}
    for p in run_panos:
        e, n = frame.to_enu(p.lat, p.lng)
        cams[p.pano_id] = (e, n)
        grid.add(e, n, p.pano_id)
    return grid, cams


def projected_detections(L, run_panos, params, frame):
    """{pano_id: [(e, n, conf), ...]} for every stored detection the production raycast
    places (floor, horizon and 25 m envelope applied exactly as fuse_sites.project)."""
    dets, _, drops = L.fs.project(run_panos, params)
    out = defaultdict(list)
    for d in dets:
        out[d.pano_id].append((d.e, d.n, d.conf))
    return out, drops


def capture_table(L, city, gtw, run_panos, params, radius=R_MAX):
    """Every (pool ramp, qualifying capture) pair within ``radius``: range, whether the
    capture is one of the ramp's GT-source panos, the ramp projected into it, and the best
    stored detection at the ramp by the world and the pixel test."""
    geo, fs = L.geo, L.fs
    frame = gtw.frame
    # Claim targets reach MATCH_RADIUS_M past the recording radius, so a detection near a
    # ramp just outside it is not free to claim a wrong one inside it.
    reach = radius + MATCH_RADIUS_M
    ramp_grid = geo.GridIndex(reach)
    for i, r in enumerate(gtw.pool):
        ramp_grid.add(r["e"], r["n"], i)
        r["captures"] = []
    _, cams = camera_index(L, frame, run_panos)
    ground, drops = projected_detections(L, run_panos, params, frame)
    stats = {"horizon_or_out_of_envelope_projections": 0}
    wr2 = MATCH_RADIUS_M ** 2
    pr2 = pixel_radius_sq()
    for p in run_panos:
        ce, cn = cams[p.pano_id]
        cand = sorted({i for i in ramp_grid.near(ce, cn)
                       if math.hypot(ce - gtw.pool[i]["e"], cn - gtw.pool[i]["n"]) <= reach})
        near = [i for i in cand
                if math.hypot(ce - gtw.pool[i]["e"], cn - gtw.pool[i]["n"]) <= radius]
        if not near:
            continue
        pose = fs.pano_pose(p, params.apply_pose)
        proj = {}
        for i in cand:
            r = gtw.pool[i]
            pr = geo.ground_point_to_pano(pose, r["lat"], r["lng"],
                                          camera_height=params.camera_height_m,
                                          max_range_m=reach)
            if pr is None and i in near:
                stats["horizon_or_out_of_envelope_projections"] += 1
            proj[i] = pr
        wclaim = claim_by_confidence(
            [(e, n, c) for e, n, c in ground.get(p.pano_id, ())],
            [(i, gtw.pool[i]["e"], gtw.pool[i]["n"]) for i in cand], world_d2, wr2)
        pclaim = claim_by_confidence(
            [(x, y, c) for _, x, y, c in p.detections],
            [(i, proj[i].x_norm, proj[i].y_norm) for i in cand if proj[i] is not None],
            pixel_d2, pr2)
        for i in near:
            r = gtw.pool[i]
            r["captures"].append({
                "pano_id": p.pano_id,
                "dist_m": math.hypot(ce - r["e"], cn - r["n"]), "cam_e": ce, "cam_n": cn,
                "is_source": p.pano_id in r["source_panos"],
                "x": None if proj[i] is None else proj[i].x_norm,
                "y": None if proj[i] is None else proj[i].y_norm,
                "world_conf": wclaim.get(i), "pixel_conf": pclaim.get(i),
                "capture_date": p.capture_date})
    for r in gtw.pool:
        r["captures"].sort(key=lambda c: (c["dist_m"], c["pano_id"]))
    stats["raycast_drops"] = drops
    return stats


# --------------------------------------------------------------------------- #
# B.3: sites, evidence and policies at the 0.30 tier
# --------------------------------------------------------------------------- #


def site_dicts(sites, key_of=lambda d: d.det_index):
    return [{"id": s.id, "e": s.e, "n": s.n,
             "members": [(d.pano_id, key_of(d), d.conf) for d, _ in s.members]}
            for s in sites]


def pano_class_for(judged_gt, dets_by_pano, tier):
    """Classify every judged-pano detection at >= ``tier`` against its GroundTruth.
    ``dets_by_pano``: {pano_id: [(key, x, y, conf), ...]}."""
    out = {}
    for pid, gt in judged_gt.items():
        preds = [d for d in dets_by_pano.get(pid, ())
                 if d[3] is None or d[3] >= tier - 1e-12]
        for key, v in classify_preds(preds, gt).items():
            out[(pid, key)] = v
    return out


def site_captures(site, grid, cams, radius=R_DEFAULT):
    """[(dist_m, conf-or-None, pano_id)] over every pano with its camera within
    ``radius`` of the site: the site's best member confidence from that pano, or None."""
    best = {}
    for pid, _, c in site["members"]:
        if c is not None and (pid not in best or c > best[pid]):
            best[pid] = c
    out = []
    for pid in set(grid.near(site["e"], site["n"])):
        ce, cn = cams[pid]
        d = math.hypot(ce - site["e"], cn - site["n"])
        if d <= radius:
            out.append((d, best.get(pid), pid))
    return out


def label_sites(sites, accepted_all, pano_class):
    """True for a TP site, False for an FP site, by the precision rule of score_world;
    sites with no decided judged member are absent."""
    lab = {}
    for s in sites:
        if s["id"] not in accepted_all:
            continue
        cls = [pano_class[(pid, key)][0] for pid, key, _ in s["members"]
               if (pid, key) in pano_class]
        if "tp" in cls:
            lab[s["id"]] = True
        elif "fp" in cls:
            lab[s["id"]] = False
    return lab


def evidence_examples(sites, labels, caps_of, judged):
    """(label, [(dist, conf)]) per labeled site, WITHOUT the judged panos' captures --
    those are what labeled the site, so counting them would calibrate on the answer."""
    ex = []
    for s in sites:
        if s["id"] not in labels:
            continue
        caps = [(d, c) for d, c, pid in caps_of[s["id"]] if pid not in judged]
        ex.append((labels[s["id"]], caps))
    return ex


def threshold_curve(sites, scores, pool, pano_class, keep_055=False, n_points=40):
    """P/R as the evidence-score threshold sweeps from permissive to strict."""
    cand = [s for s in sites if s["id"] in scores]
    vals = sorted({round(v, 6) for v in scores.values()})
    if not vals:
        return []
    qs = sorted({vals[min(len(vals) - 1, int(i * (len(vals) - 1) / (n_points - 1)))]
                 for i in range(n_points)})
    rows = []
    for t in [-math.inf] + qs:
        acc = {s["id"] for s in cand if scores[s["id"]] >= t
               or (keep_055 and (site_max_conf(s) or 0) >= BENCHMARK_CONFIDENCE - 1e-12)}
        res = score_world(pool, sites, acc, pano_class)
        rows.append({"threshold": None if t == -math.inf else t, **pr_counts(res)})
    return rows


def best_precision_at_recall(curve, recall):
    ok = [r for r in curve if r["recall"] is not None and r["recall"] >= recall - 1e-12
          and r["precision"] is not None]
    return max(ok, key=lambda r: (r["precision"], r["recall"])) if ok else None


# --------------------------------------------------------------------------- #
# `run`
# --------------------------------------------------------------------------- #


def check_reproduction(city, result, report_path):
    """eval_sites must reproduce the report's headline -- world recall, precision, TP/FP
    and the four buckets -- exactly, or nothing is written.

    The reports are local artifacts of the labeler (runs/ is not git-tracked), written
    2026-08-02; several results.jsonl files were rewritten since by metadata backfills.
    The 'self-detected ramps had no operational site' diagnostic is compared and
    recorded but does not gate: on gainesville it reads 3 today against the report's 2,
    with every headline number identical."""
    want = parse_report(report_path)
    want["report_sha256"] = _sha256(report_path)
    got = {"world_recall": round(result["world_recall"], 3),
           "precision": round(result["precision"]["value"], 3),
           "tp": result["precision"]["tp"], "fp": result["precision"]["fp"],
           "buckets": result["buckets"],
           "self_detected_without_site": result["self_detected_without_site"]}
    ok = (got["world_recall"] == want["world_recall"] and got["precision"] == want["precision"]
          and got["tp"] == want["tp"] and got["fp"] == want["fp"]
          and got["buckets"] == want["buckets"])
    got["diagnostic_matches"] = (got["self_detected_without_site"]
                                 == want["self_detected_without_site"])
    return ok, got, want


def b1_b2(ramps_by_city, sub_cities):
    """Recall-vs-captures (B.1) and failure-correlation (B.2) tables, per city, per
    imagery source, pooled; floors below 0.55 only over cities that store them."""
    groups = {c: [c] for c in ramps_by_city}
    groups["gsv"] = [c for c in ramps_by_city if SOURCE[c] == "gsv"]
    groups["mapillary"] = [c for c in ramps_by_city if SOURCE[c] == "mapillary"]
    groups["pooled"] = list(ramps_by_city)
    out = {}
    for g, cities in groups.items():
        for floor in HIT_FLOORS:
            use = cities if floor >= BENCHMARK_CONFIDENCE else [c for c in cities if c in sub_cities]
            if not use:
                continue
            ramps = [r for c in use for r in ramps_by_city[c]]
            for mode in ("world", "pixel", "either"):
                for radius in R_SWEEP:
                    key = f"{g}|{floor:.2f}|{mode}|R{int(radius)}"
                    src_caps = [c for r in ramps for c in r["captures"] if c["is_source"]]
                    out[key] = {
                        "cities": use, "ramps": len(ramps),
                        "source_view_hit_rate": (sum(capture_hit(c, floor, mode) for c in src_caps)
                                                 / len(src_caps)) if src_caps else None,
                        "by_capture_count": recall_by_capture_count(ramps, floor, mode, radius),
                        "k_nearest_fixed": recall_k_nearest(ramps, floor, mode, radius, KMAX, True),
                        "k_nearest_fixed4": recall_k_nearest(ramps, floor, mode, radius, 4, True),
                        "k_nearest_all": recall_k_nearest(ramps, floor, mode, radius, KMAX, False),
                        "failure_correlation": failure_correlation(ramps, floor, mode, radius),
                    }
    return out


def b3_city(L, city, verdicts, bundle_ops, run_panos, gtw, models=None):
    """Sites fused at the 0.30 tier plus everything B.3 needs from them."""
    params30 = fuse_params(L, min_confidence=TIER_030)
    sites30, frame, _ = L.fs.fuse(run_panos, params30)
    if abs(frame.lat0 - gtw.frame.lat0) > 1e-12 or abs(frame.lng0 - gtw.frame.lng0) > 1e-12:
        raise SystemExit(f"{city}: 0.30 fuse frame differs from the 0.55 one")
    sd = site_dicts(sites30)
    dets_by_pano = {p.pano_id: [(i, x, y, c) for i, x, y, c in p.detections]
                    for p in run_panos if p.pano_id in gtw.judged_gt}
    pc30 = pano_class_for(gtw.judged_gt, dets_by_pano, TIER_030)
    grid, cams = camera_index(L, frame, run_panos)
    tier_sites = [s for s in sd if (site_max_conf(s) or 0) >= TIER_030 - 1e-12]
    caps_of = {s["id"]: site_captures(s, grid, cams) for s in tier_sites}
    labels = label_sites(tier_sites, {s["id"] for s in tier_sites}, pc30)
    examples = evidence_examples(tier_sites, labels, caps_of, set(gtw.judged_gt))
    # The operational 0.55 fuse (eval_sites' own sites) scored by THIS script's rule, so
    # the definitional gap to eval_sites' verdict-based precision is on the record.
    sd55 = [s for s in site_dicts(gtw.sites) if (site_max_conf(s) or 0) >= BENCHMARK_CONFIDENCE - 1e-12]
    pc55 = pano_class_for(gtw.judged_gt, dets_by_pano, BENCHMARK_CONFIDENCE)
    return SimpleNamespace(sites=tier_sites, pano_class=pc30, caps_of=caps_of, labels=labels,
                           examples=examples, sites55=sd55, pano_class55=pc55)


def run_b3(city_b3, pools):
    """Policies, in-sample and leave-one-city-out evidence scores, per city and pooled."""
    out = {"per_city": {}, "models": {}}
    cities = list(city_b3)
    in_models = {c: fit_evidence_model(city_b3[c].examples) for c in cities}
    loco_models = {c: fit_evidence_model([e for o in cities if o != c for e in city_b3[o].examples])
                   for c in cities}
    out["models"] = {"in_sample": in_models, "loco": loco_models}
    pooled_policy = defaultdict(list)
    for c in cities:
        b = city_b3[c]
        pool = pools[c]
        res = {}
        res["operational_055"] = score_world(pool, b.sites55, {s["id"] for s in b.sites55},
                                             b.pano_class55)
        allowed = {s["id"] for s in b.sites}
        res["flat_030"] = score_world(pool, b.sites, allowed, b.pano_class)
        for k in (1, 2, 3):
            res[f"kofn_{k}"] = score_world(pool, b.sites, policy_kofn(b.sites, k), b.pano_class)
        res["tier055_in_030_fuse"] = score_world(
            pool, b.sites, {s["id"] for s in b.sites
                            if (site_max_conf(s) or 0) >= BENCHMARK_CONFIDENCE - 1e-12},
            b.pano_class)
        curves = {}
        for variant, model in (("in_sample", in_models[c]), ("loco", loco_models[c])):
            scores = {s["id"]: evidence_score(model, [(d, cc) for d, cc, _ in b.caps_of[s["id"]]])
                      for s in b.sites}
            curves[variant] = {
                "all_sites": threshold_curve(b.sites, scores, pool, b.pano_class, keep_055=False),
                "keep_055": threshold_curve(b.sites, scores, pool, b.pano_class, keep_055=True)}
        # the headline comparison: at each k-of-n point's recall, the best precision the
        # score reaches at or above that recall
        matched = []
        for k in (1, 2, 3):
            kr = res[f"kofn_{k}"]
            row = {"k": k, "kofn_recall": kr["recall"], "kofn_precision": kr["precision"],
                   "kofn_tp": kr["tp"], "kofn_fp": kr["fp"]}
            for variant in ("in_sample", "loco"):
                for mode in ("all_sites", "keep_055"):
                    b_ = best_precision_at_recall(curves[variant][mode], kr["recall"])
                    row[f"{variant}_{mode}"] = None if b_ is None else {
                        "precision": b_["precision"], "recall": b_["recall"],
                        "tp": b_["tp"], "fp": b_["fp"], "threshold": b_["threshold"]}
            matched.append(row)
        for name, r in res.items():
            pooled_policy[name].append(r)
        out["per_city"][c] = {
            "policies": {n: pr_counts(r) for n, r in res.items()},
            "labeled_sites": {"real": sum(1 for v in b.labels.values() if v),
                              "false": sum(1 for v in b.labels.values() if not v)},
            "score_curves": curves, "matched_recall": matched}
    out["pooled"] = {n: pool_results(rs) for n, rs in pooled_policy.items()}
    return out


def cmd_run(args):
    L = import_labeler(args.labeler_root)
    runs_root = args.runs_root or os.path.join(args.labeler_root, "runs")
    os.makedirs(OUT, exist_ok=True)
    meta = {"labeler": L.prov, "cities": list(args.cities), "radius_sweep": list(R_SWEEP),
            "match_radius_m": MATCH_RADIUS_M, "gt_merge_m": GT_MERGE_M,
            "camera_height_m": CAMERA_HEIGHT_M, "hit_floors": list(HIT_FLOORS),
            "pixel_radius_normalized": PANO_RADIUS_NORMALIZED, "city_info": {}}
    ramps_by_city, pools, b3_inputs, repro, residual = {}, {}, {}, {}, []
    capture_rows = []
    sub_cities = []
    for city in args.cities:
        print(f"== {city}", flush=True)
        verdicts, bundle_ops, run_panos, info = load_city(L, city, runs_root,
                                                          results_root=args.results_root)
        params = fuse_params(L)
        # Side note: the same eval on the labeler checkout's current (gap-filled) run.
        cur = os.path.join(runs_root, city, "results.jsonl")
        if os.path.exists(cur) and _sha256(cur) != info["results_sha256"]:
            v2, b2, p2, i2 = load_city(L, city, runs_root, require_archived=False)
            e2 = L.es.evaluate_city(v2, b2, p2, params, match_radius_m=MATCH_RADIUS_M,
                                    gt_merge_m=GT_MERGE_M)
            info["gap_filled_run"] = {
                "results_sha256": i2["results_sha256"], "run_panos": i2["run_panos"],
                "world_recall": e2["world_recall"], "world_recall_ci": e2["world_recall_ci"],
                "precision": e2["precision"]["value"], "tp": e2["precision"]["tp"],
                "fp": e2["precision"]["fp"], "buckets": e2["buckets"]}
        # The instrument check runs on the run exactly as the committed report saw it.
        exact = L.es.evaluate_city(verdicts, bundle_ops, run_panos, params,
                                   match_radius_m=MATCH_RADIUS_M, gt_merge_m=GT_MERGE_M)
        ok, got, want = check_reproduction(
            city, exact, os.path.join(runs_root, city, "fusion_eval", "report.md"))
        repro[city] = {"ok": ok, "got": got, "committed_report": want}
        if not ok:
            raise SystemExit(f"{city}: eval_sites does not reproduce the committed report:\n"
                             f"  got  {got}\n  want {want}")
        if not args.no_reinfer and apply_reinfer(city, runs_root, run_panos, info):
            print(f"   + {info['reinfer']}", flush=True)
        gtw = world_gt(L, city, verdicts, bundle_ops, run_panos, params)
        repro[city]["buckets_as_analysed"] = gtw.result["buckets"]
        stats = capture_table(L, city, gtw, run_panos, params)
        info.update({"gt_counts": gtw.counts, "pool_ramps": len(gtw.pool), "capture_stats": stats})
        meta["city_info"][city] = info
        ramps_by_city[city] = gtw.pool
        pools[city] = [{"e": r["e"], "n": r["n"], "gt_refs": r["gt_refs"]} for r in gtw.pool]
        if info["stored_subthreshold"]:
            sub_cities.append(city)
            b3_inputs[city] = b3_city(L, city, verdicts, bundle_ops, run_panos, gtw)
        for r in gtw.pool:
            for c in r["captures"]:
                capture_rows.append([city, r["uid"], c["pano_id"], c["dist_m"], int(c["is_source"]),
                                     c["x"], c["y"], c["world_conf"], c["pixel_conf"],
                                     c["capture_date"], c["cam_e"], c["cam_n"]])
            if r["bucket"] in ("unmatched", "subthreshold_only") or r["self_detected_without_site"]:
                cls = ("self_detected_site_displaced" if r["self_detected_without_site"]
                       else residual_class(r, info["stored_subthreshold"]))
                residual.append({
                    "uid": r["uid"], "city": city, "bucket": r["bucket"], "class": cls,
                    "lat": r["lat"], "lng": r["lng"], "kinds": r["kinds"],
                    "source_panos": r["source_panos"], "gt_refs": r["gt_refs"],
                    "nearest_op_site_m": r["nearest_op_site_m"],
                    "n_other_captures_18m": sum(1 for c in r["captures"]
                                                if not c["is_source"] and c["dist_m"] <= R_DEFAULT),
                    "best_world_conf": max((c["world_conf"] for c in r["captures"]
                                            if c["world_conf"] is not None), default=None),
                    "best_pixel_conf": max((c["pixel_conf"] for c in r["captures"]
                                            if c["pixel_conf"] is not None), default=None)})
    meta["sub_threshold_cities"] = sub_cities
    meta["reproduction"] = repro
    write_json(os.path.join(OUT, "meta.json"), meta)
    write_csv(os.path.join(OUT, "captures_R25.csv"), CAPTURE_COLUMNS, capture_rows)
    print("B.1/B.2 ...", flush=True)
    # B.1/B.2 are computed from the CSV as written, so the committed table alone
    # re-derives them (tests/test_multiview_48.py checks that it does).
    ramps_by_city = ramps_from_capture_csv(os.path.join(OUT, "captures_R25.csv"))
    write_json(os.path.join(OUT, "recall_vs_captures.json"), b1_b2(ramps_by_city, sub_cities))
    print("B.3 ...", flush=True)
    write_json(os.path.join(OUT, "evidence_vs_kofn.json"), run_b3(b3_inputs, pools))
    by_class = defaultdict(lambda: defaultdict(int))
    for r in residual:
        by_class[r["city"]][r["class"]] += 1
    write_json(os.path.join(OUT, "residual_misses.json"),
               {"n": len(residual), "by_city_class": by_class, "ramps": residual})
    print(f"wrote {OUT}")


# --------------------------------------------------------------------------- #
# `figures`
# --------------------------------------------------------------------------- #


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    rv = json.load(open(os.path.join(OUT, "recall_vs_captures.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(OUT, "evidence_vs_kofn.json"), encoding="utf-8"))
    ink, muted = "#1f2328", "#57606a"
    colors = {"gsv": "#0969da", "mapillary": "#bf3989", "pooled": "#1a7f37"}

    # 1. k-nearest curve (fixed population), world test, R18, three floors
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, floor in zip(axes, ("0.55", "0.30")):
        for g in ("gsv", "mapillary", "pooled"):
            key = f"{g}|{floor}|world|R18"
            if key not in rv:
                continue
            rows = rv[key]["k_nearest_fixed4"]
            ks = [r["k"] for r in rows]
            ys = [r["recall"] for r in rows]
            lo = [r["ci"][0] for r in rows]
            hi = [r["ci"][1] for r in rows]
            ax.plot(ks, ys, marker="o", color=colors[g],
                    label=f"{g} (n={rows[0]['ramps']})")
            ax.fill_between(ks, lo, hi, color=colors[g], alpha=0.12, linewidth=0)
            if g == "mapillary":
                r8 = rv[key]["k_nearest_fixed"]
                ax.plot([r["k"] for r in r8], [r["recall"] for r in r8], linestyle=":",
                        color=colors[g], label=f"mapillary, >= 8 views (n={r8[0]['ramps']})")
        ax.set_title(f"other views only, detection >= {floor}", color=ink, fontsize=10)
        ax.set_xlabel("k nearest other captures used (within 18 m)", color=muted)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("share of GT ramps seen by >= 1 of them", color=muted)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Recall from other captures vs how many are used (fixed population: ramps "
                 "with >= 4 other captures; dotted: >= 8)", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recall_k_nearest.png"), dpi=150)
    plt.close(fig)

    # 2. failure correlation: every-view-missed, observed vs independence, by capture count
    fc = rv["pooled|0.55|world|R18"]["failure_correlation"]
    rows = [r for r in fc["all_missed_by_n"] if r["ramps"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], [r["observed"] for r in rows], width=0.4,
           color=colors["pooled"], label="observed")
    ax.bar([x + 0.2 for x in xs], [r["predicted_independent"] for r in rows], width=0.4,
           color="#8c959f", label="if views failed independently")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r['n_bin']}\n({r['ramps']} ramps)" for r in rows], fontsize=7)
    ax.set_xlabel("other captures within 18 m", color=muted)
    ax.set_ylabel("ramps every other view missed", color=muted)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    seps = [r for r in fc["by_separation"] if r["pairs"]]
    labels = [f"{r['bin'][0]:g}-{r['bin'][1]:g}" for r in seps]
    ax.plot(labels, [r["p_miss_given_miss"] for r in seps], marker="o", color=colors["pooled"],
            label="P(view j misses | view i missed)")
    ax.plot(labels, [r["p_miss_marginal"] for r in seps], marker="o", color="#8c959f",
            label="P(view j misses), range-matched")
    ax.set_xlabel("distance between the two cameras (m)", color=muted)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Misses in different views of one ramp are correlated (5 cities, >= 0.55, "
                 "world test)", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "failure_correlation.png"), dpi=150)
    plt.close(fig)

    # 3. evidence score vs k-of-n, pooled over sub-threshold cities
    fig, axes = plt.subplots(1, len(ev["per_city"]), figsize=(3.3 * len(ev["per_city"]), 3.4),
                             sharey=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, (city, d) in zip(axes, sorted(ev["per_city"].items())):
        for variant, style in (("in_sample", "-"), ("loco", "--")):
            cur = d["score_curves"][variant]["all_sites"]
            ax.plot([r["recall"] for r in cur], [r["precision"] for r in cur], style,
                    color="#0969da", label=f"score ({variant})")
        for name, mk in (("operational_055", "o"), ("flat_030", "s"), ("kofn_2", "^"),
                         ("kofn_3", "v")):
            p = d["policies"][name]
            ax.plot(p["recall"], p["precision"], mk, color=ink, label=name)
        ax.set_title(city, fontsize=10, color=ink)
        ax.set_xlabel("world recall", color=muted)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("world precision", color=muted)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "evidence_vs_kofn.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {FIG_DIR}")


# --------------------------------------------------------------------------- #
# Residual-miss gallery: plan (local), cut (makelab2), render (local)
# --------------------------------------------------------------------------- #

CROP_FOV_H = 36.0     # degrees of azimuth per crop
CROP_FOV_V = 24.0
CROP_PX = (360, 240)
MAX_OTHER_VIEWS = 4

#: The one-rater rubric for the residual misses. It travels in the verdicts file.
RESIDUAL_SCHEME = [
    ("occluded", "A vehicle, person, pole or vegetation blocks the ramp in the views "
                 "that should have seen it"),
    ("flush-minimal-reveal", "The ramp is there but flush with the street or with almost "
                             "no visible reveal, so its shape barely registers (#151)"),
    ("far", "Every view is far enough that the ramp is a handful of pixels"),
    ("construction-changed", "The corner differs between views or dates: construction, "
                             "a rebuilt corner, a ramp that is not there in some captures"),
    ("gt-error", "No curb ramp at the marked spot in any view: the GT point is wrong or "
                 "misplaced"),
    ("other", "A reason not listed here; say which in a note if it recurs"),
    ("unclear", "Cannot tell from these crops (excluded from every rate)"),
]


def crop_plan(residual, captures_csv):
    """Crops to cut: every source view at its GT point, plus up to MAX_OTHER_VIEWS
    nearest other captures with the ramp projected into them."""
    caps = defaultdict(list)
    with open(captures_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            caps[row["ramp_uid"]].append(row)
    plan = []
    for r in residual:
        items = []
        rc = sorted(caps[r["uid"]], key=lambda c: float(c["dist_m"]))
        for c in rc:
            if c["is_source"] == "1" and c["x_proj"]:
                items.append(c)
        others = [c for c in rc if c["is_source"] == "0" and c["x_proj"]
                  and float(c["dist_m"]) <= R_DEFAULT][:MAX_OTHER_VIEWS]
        for c in items + others:
            plan.append({"ramp_uid": r["uid"], "city": r["city"], "pano_id": c["pano_id"],
                         "x": float(c["x_proj"]), "y": float(c["y_proj"]),
                         "dist_m": float(c["dist_m"]), "is_source": c["is_source"] == "1",
                         "world_conf": c["world_conf"] or None,
                         "pixel_conf": c["pixel_conf"] or None,
                         "capture_date": c["capture_date"]})
    return plan


def crop_name(item):
    return f"{item['ramp_uid'].replace(':', '_')}__{item['pano_id']}.jpg"


def cmd_crop_plan(args):
    res = json.load(open(os.path.join(OUT, "residual_misses.json"), encoding="utf-8"))
    plan = crop_plan(res["ramps"], os.path.join(OUT, "captures_R25.csv"))
    write_json(os.path.join(OUT, "residual_crop_plan.json"), {"n": len(plan), "items": plan})
    print(f"{len(plan)} crops over {len({p['ramp_uid'] for p in plan})} ramps -> "
          f"{os.path.join(OUT, 'residual_crop_plan.json')}")


def cut_one(img, x, y, fov_h=CROP_FOV_H, fov_v=CROP_FOV_V, size=CROP_PX):
    """An equirect window centred on (x, y), wrapped at the seam, with a ring at the
    target. Plain equirect (no reprojection): a 36 x 24 degree window near the horizon
    is close to rectilinear, and it keeps the cut exact and dependency-free."""
    from PIL import Image, ImageDraw
    W, H = img.size
    w = int(round(W * fov_h / 360.0))
    h = int(round(H * fov_v / 180.0))
    cx, cy = x * W, y * H
    left = int(round(cx - w / 2))
    top = max(0, min(H - h, int(round(cy - h / 2))))
    out = Image.new("RGB", (w, h))
    # paste in up to two pieces across the seam
    lx = left % W
    first = min(w, W - lx)
    out.paste(img.crop((lx, top, lx + first, top + h)), (0, 0))
    if first < w:
        out.paste(img.crop((0, top, w - first, top + h)), (first, 0))
    out = out.resize(size)
    d = ImageDraw.Draw(out)
    tx = (cx - left) / w * size[0]
    ty = (cy - top) / h * size[1]
    rr = 14
    d.ellipse((tx - rr, ty - rr, tx + rr, ty + rr), outline=(0, 255, 90), width=2)
    return out


def cmd_cut_crops(args):
    """Runs where the native-res archive lives (makelab2). Reads the plan, writes one JPEG
    per item into --out; never touches a pano it does not need."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    plan = json.load(open(args.plan, encoding="utf-8"))["items"]
    os.makedirs(args.out, exist_ok=True)
    by_pano = defaultdict(list)
    for it in plan:
        by_pano[(it["city"], it["pano_id"])].append(it)
    missing = []
    for (city, pid), items in sorted(by_pano.items()):
        path = os.path.join(args.archive_root, city, "panos", f"{pid}.jpg")
        if not os.path.exists(path):
            missing.append(path)
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            for it in items:
                cut_one(im, it["x"], it["y"]).save(os.path.join(args.out, crop_name(it)),
                                                   quality=82)
    print(f"cut {sum(len(v) for v in by_pano.values()) - len(missing)} crops; "
          f"{len(missing)} panos missing")
    for m in missing[:20]:
        print("  missing", m)


def manifest_digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]


GALLERY_DIR = os.path.join(BENCHMARK, "multiview_residual_48")
VERDICTS_PATH = os.path.join(OUT, "residual_taxonomy__jonf.json")


def cmd_gallery(args):
    """Render the one-rater gallery and, if absent, the empty per-rater verdicts file."""
    import html
    res = json.load(open(os.path.join(OUT, "residual_misses.json"), encoding="utf-8"))
    plan = json.load(open(os.path.join(OUT, "residual_crop_plan.json"), encoding="utf-8"))["items"]
    crops_dir = args.crops
    have = set(os.listdir(crops_dir)) if os.path.isdir(crops_dir) else set()
    by_ramp = defaultdict(list)
    for it in plan:
        by_ramp[it["ramp_uid"]].append(it)
    ids = [r["uid"] for r in res["ramps"]]
    digest = manifest_digest(ids)
    os.makedirs(os.path.join(GALLERY_DIR, "crops"), exist_ok=True)
    cards = []
    for r in res["ramps"]:
        panels = []
        for it in by_ramp[r["uid"]]:
            name = crop_name(it)
            if name not in have:
                continue
            src = os.path.join(crops_dir, name)
            dst = os.path.join(GALLERY_DIR, "crops", name)
            if os.path.abspath(src) != os.path.abspath(dst):
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())
            wc = it["world_conf"] or "-"
            pc = it["pixel_conf"] or "-"
            role = "GT source view" if it["is_source"] else f"other view, {it['dist_m']:.1f} m"
            panels.append(
                f'<figure><img src="crops/{html.escape(name)}" width="{CROP_PX[0]}" '
                f'height="{CROP_PX[1]}" alt="{html.escape(role)} of ramp {html.escape(r["uid"])}">'
                f'<figcaption>{html.escape(role)}<br>{html.escape(it["capture_date"] or "")} '
                f'&middot; world {html.escape(str(wc)[:4])} &middot; pixel {html.escape(str(pc)[:4])}'
                f'</figcaption></figure>')
        name = f"v_{r['uid'].replace(':', '_')}"
        radios = "".join(
            f'<label><input type="radio" name="{name}" value="{k}"> {html.escape(k)}</label>'
            for k, _ in RESIDUAL_SCHEME)
        cards.append(
            f'<section class="card" data-uid="{html.escape(r["uid"])}">'
            f'<h2>{html.escape(r["uid"])} <span class="meta">{html.escape(r["class"])} &middot; '
            f'bucket {html.escape(r["bucket"])} &middot; {r["n_other_captures_18m"]} other '
            f'captures within 18 m</span></h2><div class="strip">{"".join(panels)}</div>'
            f'<fieldset><legend>Why was this ramp missed?</legend>{radios}</fieldset></section>')
    scheme_html = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>"
                          for k, v in RESIDUAL_SCHEME)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Residual misses 48</title>
<style>
:root {{ --bg:#ffffff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --ring:#0969da; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --ring:#4493f8; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 16px 64px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:15px; margin:0 0 6px; }}
.meta {{ color:var(--muted); font-weight:normal; font-size:13px; }}
.card {{ border-top:1px solid var(--line); padding:12px 0; }}
.strip {{ display:flex; gap:8px; overflow-x:auto; }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:12px; }}
img {{ max-width:100%; height:auto; display:block; }}
fieldset {{ border:0; padding:6px 0 0; display:flex; flex-wrap:wrap; gap:4px 14px; }}
input:focus-visible {{ outline:2px solid var(--ring); }}
button {{ font:inherit; padding:6px 12px; }}
dt {{ font-weight:600; }} dd {{ margin:0 0 6px 16px; color:var(--muted); }}
</style></head><body>
<h1>Residual misses (#48): one-rater pass</h1>
<p>Each card is a GT curb ramp that no operational fused site recovered (or a self-detected
ramp whose site landed more than 5 m away). The ring marks the GT point in its source view
and the same ground point projected into the nearest other captures (projection error is
real: p90 about 4.4 m on the ground, so the ring can sit beside the ramp). Pick the one
reason that best explains why the model did not deliver this ramp. The captions give the
best stored detection at the ramp in that view (world 5 m test, pixel radius test).</p>
<dl>{scheme_html}</dl>
<p><button type="button" id="export">Export verdicts JSON</button>
<span id="count" aria-live="polite"></span></p>
{"".join(cards)}
<script>
const KEY = "mv48_residual_{digest}";
const saved = (() => {{ try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }} }})();
function update() {{
  const n = document.querySelectorAll('input[type=radio]:checked').length;
  document.getElementById('count').textContent = n + " of {len(ids)} tagged";
}}
document.querySelectorAll('.card').forEach(card => {{
  const uid = card.dataset.uid;
  card.querySelectorAll('input[type=radio]').forEach(inp => {{
    if (saved[uid] === inp.value) inp.checked = true;
    inp.addEventListener('change', () => {{
      saved[uid] = inp.value;
      try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}}
      update();
    }});
  }});
}});
update();
document.getElementById('export').addEventListener('click', () => {{
  const out = {{task: "RampNet #48 residual misses: why did multi-view fusion not deliver this ramp?",
    rater: "jonf", scheme: {json.dumps([[k, v] for k, v in RESIDUAL_SCHEME])},
    manifest_digest: "{digest}", n_items: {len(ids)},
    n_tagged: Object.keys(saved).length, verdicts: saved}};
  const blob = new Blob([JSON.stringify(out, null, 2) + "\\n"], {{type: "application/json"}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "residual_taxonomy__jonf.json";
  a.click();
}});
</script></body></html>
"""
    with open(os.path.join(GALLERY_DIR, "gallery.html"), "w", encoding="utf-8", newline="") as f:
        f.write(page)
    if not os.path.exists(VERDICTS_PATH):
        write_json(VERDICTS_PATH, {
            "task": "RampNet #48 residual misses: why did multi-view fusion not deliver "
                    "this ramp?",
            "rater": "jonf", "scheme": [[k, v] for k, v in RESIDUAL_SCHEME],
            "items": ids, "manifest_digest": digest, "n_items": len(ids), "n_tagged": 0,
            "gallery": "benchmark/multiview_residual_48/gallery.html",
            "verdicts": {}})
    print(f"wrote {GALLERY_DIR}/gallery.html ({len(cards)} cards, digest {digest})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labeler-root", required=True)
    r.add_argument("--runs-root", default=None,
                   help="default <labeler-root>/runs; separate so the code can come from a "
                        "pinned snapshot while the run files come from the checkout")
    r.add_argument("--results-root", default=None,
                   help="dir of <city>/results.jsonl copies of the makelab2 archive (the runs "
                        "the committed reports scored; see ARCHIVED_RESULTS_SHA256)")
    r.add_argument("--cities", nargs="+", default=list(CITIES))
    r.add_argument("--no-reinfer", action="store_true",
                   help="ignore richmond's results.f01.jsonl re-inference")
    sub.add_parser("figures")
    sub.add_parser("crop-plan")
    c = sub.add_parser("cut-crops")
    c.add_argument("plan")
    c.add_argument("--archive-root", required=True,
                   help="dir holding <city>/panos/<pano_id>.jpg (makelab2: "
                        "/projects/makeabilitylab/sidewalk-auto-labeler/runs)")
    c.add_argument("--out", required=True)
    g = sub.add_parser("gallery")
    g.add_argument("--crops", required=True)
    args = ap.parse_args(argv)
    {"run": cmd_run, "figures": cmd_figures, "crop-plan": cmd_crop_plan,
     "cut-crops": cmd_cut_crops, "gallery": cmd_gallery}[args.cmd](args)


if __name__ == "__main__":
    main()
