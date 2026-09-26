"""Two-scale inference for the frozen RampNet checkpoint (issue #197).

Near-field detections from the 1x pass (``r2048``, the model's own 2048x4096 input), far-field
detections from the 2x upsampled pass (``u4096``: 2048x4096 bicubic-upsampled to 4096x8192), fused
per panorama. No retraining, no GPU: every input is a committed #196 peak cache,
``analysis_out/input_res_sweep_25/cache/<arm>/<split>.json`` (peaks down to 0.05, extracted with
``exclude_border=False``), and the GT those caches carry.

The fusion rules and the verdict rule were committed in ``docs/two_scale_197.md`` before any
fusion number was computed. In short:

    R1  range split at 18 m (a priori): r2048 >= 0.30 below the 18 m image row, u4096 >= 0.30
        at or above it, deduped across the boundary.
    R2  union of r2048 >= 0.30 and u4096 >= 0.30, deduped (= the peak-level max of the two).
    R3  range split with the cutoff D and u4096 threshold chosen leave-one-split-out.

"Dedupe" is score-ordered suppression across the two passes at the scorer's own match radius
and metric (wrapped at the seam); peaks from one pass never suppress each other.

Everything is scored with ``benchmark_power_135.score_model`` (via
``input_res_sweep_25.scored_from_panos``) and bootstrapped with its paired, split-stratified
pano bootstrap, the machinery #196 used.

    python scripts/analysis/two_scale_197.py report          # -> results.json + results.md
    python scripts/analysis/two_scale_197.py report --check  # regenerate, compare bytes
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for, score_pano)
from rampnet.geometry import dist_sq  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402
import input_res_sweep_25 as irs  # noqa: E402
from recall_by_depth_112 import CAM_H, flat_range  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "two_scale_197")
CACHE_ROOT = irs.CACHE_ROOT
USAGE_LOG = irs.USAGE_LOG
SPLITS = irs.SPLITS
POOLS = irs.POOLS
NEAR_ARM, FAR_ARM = "r2048", "u4096"

T_OP = 0.30                       # the #54/#79 recommended operating point, both passes
D_PRIOR = 18.0                    # FAR_M: the far band of detection_recall_analysis.md / #46
LOSO_D = (12.0, 18.0, 25.0, 40.0)
LOSO_TU = (0.20, 0.30, 0.40)
MATCH_GRID = tuple(round(0.05 + 0.01 * i, 2) for i in range(91))   # 0.05 .. 0.95
N_REPS = 2000
SEED = 197
N_SHIFTS = 20
ND = 4
#: Post hoc sensitivity (added after the first results, NOT in the plan and NOT given a
#: verdict): dedupe at the extractor's own peak spacing, min_distance 10 heatmap px, instead
#: of the scorer's match radius (~22.5 px). Within one pass two peaks can sit 10 px apart,
#: so the match-radius dedupe is stricter across passes than peak_local_max is within one.
MIN_DIST_RSQ = 10.0 ** 2

RULES = ("R1", "R2", "R3")
RULE_NAMES = {
    "R1": "range split at 18 m (a priori)",
    "R2": "union + dedupe (max of the two passes)",
    "R3": "range split, D and t_u leave-one-split-out",
    "u4096": "u4096 alone at 0.30 (reference)",
    "naive_union": "union with no dedupe (reference)",
    "R2s": "R2 with dedupe at min_distance 10 px instead of the match radius (post hoc)",
}


# --------------------------------------------------------------------------- #
# pure fusion helpers (unit-tested)
# --------------------------------------------------------------------------- #
def y_cut_for(d_m, cam_h=CAM_H):
    """Normalized image row at flat-ground range ``d_m``: a peak with y <= this is at
    ``d_m`` or farther (or at/above the horizon, y <= 0.5).

    >>> round(y_cut_for(18.0), 5)
    0.54393
    """
    return 0.5 + math.atan(cam_h / d_m) / math.pi


def is_far(y, d_m, cam_h=CAM_H):
    """Far side of the split: flat range >= d_m, or no range (at/above the horizon).

    Decided by ``flat_range`` itself, so it agrees with the band tables exactly."""
    r = flat_range(y, cam_h)
    return r is None or r >= d_m


def dedupe(near, far, radius_sq):
    """Score-ordered cross-pass suppression.

    ``near`` and ``far`` are lists of (x, y, score). Walk both merged by descending score
    (ties: ``near`` first); drop a peak that lies strictly within the match radius
    (the scorer's metric, wrapped at the seam) of a peak already kept from the OTHER
    list. Returns (kept list sorted by descending score, n_near_dropped, n_far_dropped)."""
    tagged = [(s, 0, i, (x, y, s)) for i, (x, y, s) in enumerate(near)]
    tagged += [(s, 1, i, (x, y, s)) for i, (x, y, s) in enumerate(far)]
    tagged.sort(key=lambda t: (-t[0], t[1], t[2]))
    kept = {0: [], 1: []}
    dropped = [0, 0]
    for _, src, _, p in tagged:
        other = kept[1 - src]
        if any(dist_sq(p[0], p[1], q[0], q[1], PANO_SCALE_X, PANO_SCALE_Y, True) < radius_sq
               for q in other):
            dropped[src] += 1
            continue
        kept[src].append(p)
    out = sorted(kept[0] + kept[1], key=lambda p: -p[2])
    return out, dropped[0], dropped[1]


def fuse_range(r_preds, u_preds, d_m, radius_sq, t_r=T_OP, t_u=T_OP):
    """R1 / R3: r2048 on the near side of ``d_m``, u4096 on the far side, deduped."""
    near = [p for p in r_preds if p[2] >= t_r and not is_far(p[1], d_m)]
    far = [p for p in u_preds if p[2] >= t_u and is_far(p[1], d_m)]
    return dedupe(near, far, radius_sq)


def fuse_union(r_preds, u_preds, radius_sq, t_r=T_OP, t_u=T_OP):
    """R2: every r2048 and u4096 peak above threshold, deduped."""
    return dedupe([p for p in r_preds if p[2] >= t_r],
                  [p for p in u_preds if p[2] >= t_u], radius_sq)


def naive_union(r_preds, u_preds, t_r=T_OP, t_u=T_OP):
    return sorted([p for p in r_preds if p[2] >= t_r] + [p for p in u_preds if p[2] >= t_u],
                  key=lambda p: -p[2])


def thresholded(preds, t):
    return [p for p in preds if p[2] >= t]


def counts(panos, radius_sq):
    """Pooled scorer counts for panos whose ``preds`` are already the final list."""
    tp = fp = tp_rec = n_gt = 0
    for p in panos:
        s = score_pano(p["preds"], p["gt"], radius_sq=radius_sq)
        tp += s.tp
        fp += s.fp
        if s.fn_confirmed:
            tp_rec += s.tp
            n_gt += s.n_gt
    return {"tp": tp, "fp": fp, "tp_rec": tp_rec, "n_gt": n_gt}


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp_rec"] / c["n_gt"] if c["n_gt"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def add_counts(cs):
    out = {"tp": 0, "fp": 0, "tp_rec": 0, "n_gt": 0}
    for c in cs:
        for k in out:
            out[k] += c[k]
    return out


def verdict(d_r2048, d_matched):
    """The pre-stated rule (docs/two_scale_197.md), first match wins, unrounded bounds.

    ``d_r2048`` = paired contrast fused - r2048@0.30; ``d_matched`` = paired contrast
    fused - r2048 at the matched-recall threshold (None when no threshold reaches the
    fused recall, which counts as not beaten)."""
    dr, df = d_r2048["recall"], d_r2048["f1"]
    beats_matched = d_matched is not None and d_matched["f1"]["ci_lo"] > 0
    if dr["ci_lo"] > 0 and df["ci_lo"] > 0 and beats_matched:
        return "HELPS"
    if dr["ci_lo"] > 0 and beats_matched:
        return "RECALL LEVER"
    if dr["ci_lo"] > 0:
        return "NO BETTER THAN A THRESHOLD"
    if df["ci_hi"] < 0:
        return "HURTS"
    return "NULL"


def rnd(v, nd=ND):
    return irs.rnd(v, nd)


def write_json(path, obj):
    irs.write_json(path, obj)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_pairs(cache_root, cities):
    """{city: [(pano, gt, r2048 preds, u4096 preds)]} in sorted pano order."""
    out = {}
    for c in cities:
        r, _ = irs.read_arm_cache(irs.cache_path(cache_root, NEAR_ARM, c))
        u, _ = irs.read_arm_cache(irs.cache_path(cache_root, FAR_ARM, c))
        ub = {p["pano"]: p for p in u}
        assert set(ub) == {p["pano"] for p in r}, (c, "pano sets differ between arms")
        rows = []
        for p in sorted(r, key=lambda q: q["pano"]):
            q = ub[p["pano"]]
            assert irs._gt_to_json(p["gt"]) == irs._gt_to_json(q["gt"]), (c, p["pano"])
            rows.append((p["pano"], p["gt"], [tuple(t) for t in p["preds"]],
                         [tuple(t) for t in q["preds"]]))
        out[c] = rows
    return out


def variant_panos(rows, variant, rsq, setting=None, shift=0):
    """Final per-pano lists for one variant. Returns (panos, dropped_near, dropped_far).

    ``shift`` k>0 gives pano i the u4096 peaks of pano (i + k) mod n (the wrong-pano
    control); the r2048 peaks and GT stay with their own pano."""
    n = len(rows)
    panos, dn, df = [], 0, 0
    for i, (pid, gt, r, _) in enumerate(rows):
        u = rows[(i + shift) % n][3]
        if variant == "r2048":
            preds = thresholded(r, T_OP)
        elif variant == "u4096":
            preds = thresholded(u, T_OP)
        elif variant == "naive_union":
            preds = naive_union(r, u)
        elif variant in ("R2", "R2s"):
            preds, a, b = fuse_union(r, u, rsq if variant == "R2" else MIN_DIST_RSQ)
            dn, df = dn + a, df + b
        elif variant in ("R1", "R3"):
            d_m, t_u = setting if variant == "R3" else (D_PRIOR, T_OP)
            if d_m is None:                       # R3 chose "no fusion"
                preds = thresholded(r, T_OP)
            else:
                preds, a, b = fuse_range(r, u, d_m, rsq, t_u=t_u)
                dn, df = dn + a, df + b
        elif variant.startswith("r2048@"):
            preds = thresholded(r, float(variant.split("@")[1]))
        else:
            raise ValueError(variant)
        panos.append({"pano": pid, "gt": gt, "preds": preds})
    return panos, dn, df


def loso_settings(pairs, rsq):
    """{held-out city: (D, t_u) or (None, None)}, each chosen on the other splits' pooled F1.

    Ties are broken toward the earlier grid entry, and "no fusion" is listed first so
    fusion has to strictly beat r2048 to be chosen."""
    grid = [(None, None)] + [(d, t) for d in LOSO_D for t in LOSO_TU]
    table = {}
    for c, rows in pairs.items():
        for s in grid:
            panos, _, _ = variant_panos(rows, "R3", rsq, setting=s)
            table[(c, s)] = counts(panos, rsq)
    chosen, scores = {}, {}
    for held in pairs:
        best, best_f = None, -1.0
        per = []
        for s in grid:
            f = prf(add_counts(table[(c, s)] for c in pairs if c != held))[2]
            per.append({"D_m": s[0], "t_u": s[1], "F1_other_splits": rnd(f)})
            if f > best_f + 1e-12:
                best, best_f = s, f
        chosen[held] = best
        scores[held] = per
    return chosen, scores


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _contrast(sa, sb, sizes):
    rng = np.random.default_rng(SEED)
    r = bp.observed_and_se(sa, sizes, 0.0, rng, N_REPS, paired=sb)
    return {k: {"observed": v["observed"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
            for k, v in r.items() if k != "max_f1"}


def _round_contrast(c):
    return {k: {kk: rnd(vv) for kk, vv in v.items()} for k, v in c.items()}


def _metrics(c):
    p, r, f = prf(c)
    return {"P": rnd(p), "R": rnd(r), "F1": rnd(f), "tp": c["tp"], "fp": c["fp"],
            "tp_rec": c["tp_rec"], "n_gt": c["n_gt"]}


def _matched_threshold(r2048_counts_by_t, target_recall):
    """Highest threshold on MATCH_GRID whose r2048 recall reaches ``target_recall``."""
    best = None
    for t in MATCH_GRID:
        if prf(r2048_counts_by_t[t])[1] >= target_recall - 1e-12:
            best = t
    return best


def build(cache_root=CACHE_ROOT, cities=SPLITS, n_shifts=N_SHIFTS):
    rsq = radius_sq_for()
    pairs = load_pairs(cache_root, cities)
    chosen, loso_table = loso_settings(pairs, rsq)

    variants = ("r2048", "u4096", "naive_union") + RULES + ("R2s",)
    panos, drops, cnt, scored = {}, {}, {}, {}
    for c, rows in pairs.items():
        for v in variants:
            ps, dn, df = variant_panos(rows, v, rsq, setting=chosen[c])
            panos[(v, c)] = ps
            drops[(v, c)] = (dn, df)
            cnt[(v, c)] = counts(ps, rsq)
            scored[(v, c)] = irs.scored_from_panos(c, ps, rsq)
        for t in MATCH_GRID:
            ps, _, _ = variant_panos(rows, f"r2048@{t:.2f}", rsq)
            cnt[(f"r2048@{t:.2f}", c)] = counts(ps, rsq)

    def group(name, members):
        sizes = [len(pairs[c]) for c in members]
        ent = {"members": list(members), "n_panos": sum(sizes), "metrics": {}, "rules": {}}
        for v in variants:
            ent["metrics"][v] = _metrics(add_counts(cnt[(v, c)] for c in members))
        base = add_counts(cnt[("r2048", c)] for c in members)
        by_t = {t: add_counts(cnt[(f"r2048@{t:.2f}", c)] for c in members) for t in MATCH_GRID}
        s_base = bp.stack([scored[("r2048", c)] for c in members])
        for v in ("u4096", "naive_union") + RULES + ("R2s",):
            fc = add_counts(cnt[(v, c)] for c in members)
            s_v = bp.stack([scored[(v, c)] for c in members])
            d = _contrast(s_v, s_base, sizes)
            d_tp, d_fp = fc["tp_rec"] - base["tp_rec"], fc["fp"] - base["fp"]
            t_m = _matched_threshold(by_t, prf(fc)[1])
            matched, d_m = None, None
            if t_m is not None:
                mc = by_t[t_m]
                s_m = bp.stack([irs.scored_from_panos(
                    c, variant_panos(pairs[c], f"r2048@{t_m:.2f}", rsq)[0], rsq)
                    for c in members])
                d_m = _contrast(s_v, s_m, sizes)
                matched = {"threshold": t_m, **_metrics(mc),
                           "extra_fp_vs_r2048": mc["fp"] - base["fp"],
                           "fused_minus_matched": _round_contrast(d_m)}
            dn = sum(drops[(v, c)][0] for c in members)
            dfar = sum(drops[(v, c)][1] for c in members)
            ent["rules"][v] = {
                "vs_r2048": _round_contrast(d),
                "recovered_ramps": d_tp, "extra_fp": d_fp,
                "fp_per_recovered_ramp": rnd(d_fp / d_tp) if d_tp > 0 else None,
                "dropped_by_dedupe": {"r2048": dn, "u4096": dfar},
                "matched_recall_baseline": matched,
                "verdict": verdict(d, d_m) if v in RULES else None,
                # the bounds the verdict was taken on, to 8 dp (the tables round to 3-4)
                "verdict_bounds": {
                    "dR_ci_lo": rnd(d["recall"]["ci_lo"], 8),
                    "dF1_ci_lo": rnd(d["f1"]["ci_lo"], 8),
                    "dF1_ci_hi": rnd(d["f1"]["ci_hi"], 8),
                    "vs_matched_dF1_ci_lo": rnd(d_m["f1"]["ci_lo"], 8) if d_m else None}
                if v in RULES else None}
        return ent

    rep = {"protocol": {
        "arms": [NEAR_ARM, FAR_ARM], "t_op": T_OP, "d_prior_m": D_PRIOR,
        "y_cut_prior": rnd(y_cut_for(D_PRIOR), 6), "cam_h_m": CAM_H,
        "loso_grid": {"D_m": list(LOSO_D), "t_u": list(LOSO_TU)},
        "match_grid": [MATCH_GRID[0], MATCH_GRID[-1], 0.01],
        "n_reps": N_REPS, "seed": SEED, "n_shifts": n_shifts,
        "radius_normalized": 0.022, "rule_names": RULE_NAMES},
        "loso": {c: {"chosen": {"D_m": chosen[c][0], "t_u": chosen[c][1]},
                     "grid": loso_table[c]} for c in pairs},
        "per_split": {c: group(c, [c]) for c in pairs},
        "pooled": {name: group(name, [c for c in m if c in pairs])
                   for name, m in POOLS.items() if any(c in pairs for c in m)},
        "wrong_pano": {}, "bands": {}}

    # wrong-pano control: u4096 peaks shifted to other panos within each split
    for v in RULES + ("R2s",):
        ent = {}
        for name, members in [(c, [c]) for c in pairs] + [
                (n, [c for c in m if c in pairs]) for n, m in POOLS.items()]:
            if not members:
                continue
            base = add_counts(cnt[("r2048", c)] for c in members)
            real = add_counts(cnt[(v, c)] for c in members)
            shifted_dr, shifted_dfp = [], []
            ks = {}
            for c in members:
                n = len(pairs[c])
                ks[c] = sorted({max(1, int(round(j * n / (n_shifts + 1))))
                                for j in range(1, n_shifts + 1)} - {0, n})
            for j in range(n_shifts):
                cs = []
                for c in members:
                    k = ks[c][j % len(ks[c])]
                    ps, _, _ = variant_panos(pairs[c], v, rsq, setting=chosen[c], shift=k)
                    cs.append(counts(ps, rsq))
                sc = add_counts(cs)
                shifted_dr.append(prf(sc)[1] - prf(base)[1])
                shifted_dfp.append(sc["fp"] - base["fp"])
            real_dr = prf(real)[1] - prf(base)[1]
            ent[name] = {"real_dR": rnd(real_dr), "null_dR_mean": rnd(float(np.mean(shifted_dr))),
                         "null_dR_max": rnd(float(np.max(shifted_dr))),
                         "attributable_dR": rnd(real_dr - float(np.mean(shifted_dr))),
                         "real_dFP": real["fp"] - base["fp"],
                         "null_dFP_mean": rnd(float(np.mean(shifted_dfp)))}
        rep["wrong_pano"][v] = ent

    # R1 diagnostic (post hoc): where do the ramps r2048 finds and R1 loses sit?
    diag = {}
    for name, members in POOLS.items():
        members = [c for c in members if c in pairs]
        tally = {"lost": 0, "r2048_peak_far_side": 0, "u4096_peak_near_side_only": 0,
                 "u4096_no_peak": 0, "gt_far_side": 0}
        for c in members:
            s_r, s_1 = scored[("r2048", c)], scored[("R1", c)]
            pts, _ = irs.gt_points_in_scored_order(panos[("r2048", c)])
            gt_pano = [pid for pid, gt, _, _ in pairs[c] if gt.fn_confirmed
                       for _ in gt.gt_points]
            byp = {pid: (r, u) for pid, _, r, u in pairs[c]}
            for k, ((gx, gy), pid) in enumerate(zip(pts, gt_pano)):
                if not (s_r.hit_gt[k] > -np.inf and s_1.hit_gt[k] == -np.inf):
                    continue
                tally["lost"] += 1
                tally["gt_far_side"] += is_far(gy, D_PRIOR)
                r, u = byp[pid]
                near_of = [p for p in thresholded(r, T_OP) if dist_sq(
                    p[0], p[1], gx, gy, PANO_SCALE_X, PANO_SCALE_Y, True) < rsq]
                u_of = [p for p in thresholded(u, T_OP) if dist_sq(
                    p[0], p[1], gx, gy, PANO_SCALE_X, PANO_SCALE_Y, True) < rsq]
                tally["r2048_peak_far_side"] += any(is_far(p[1], D_PRIOR) for p in near_of)
                if not u_of:
                    tally["u4096_no_peak"] += 1
                elif not any(is_far(p[1], D_PRIOR) for p in u_of):
                    tally["u4096_peak_near_side_only"] += 1
        diag[name] = tally
    rep["r1_lost_ramps"] = diag

    # recall by flat-ground band, paired near/far CIs
    for name, members in POOLS.items():
        members = [c for c in members if c in pairs]
        if not members:
            continue
        pts, pano_of, sizes, off = [], [], [], 0
        for c in members:
            p, po = irs.gt_points_in_scored_order(panos[("r2048", c)])
            pts.extend(p)
            pano_of.append(po + off)
            off += len(pairs[c])
            sizes.append(len(pairs[c]))
        pano_of = np.concatenate(pano_of)
        mb = [irs.band_of_point(y)[0] for _, y in pts]
        rng_ = [flat_range(y, CAM_H) for _, y in pts]
        far = np.array([r is not None and r >= D_PRIOR for r in rng_])
        near = np.array([r is not None and r < D_PRIOR for r in rng_])
        hits = {}
        tab = {}
        for v in variants:
            s = bp.stack([scored[(v, c)] for c in members])
            assert len(s.hit_gt) == len(pts)
            hits[v] = s.hit_gt
            t = {}
            for lab in dict.fromkeys(mb):
                m = np.array([x == lab for x in mb])
                t[lab] = {"n": int(m.sum()), "recall": rnd(float((s.hit_gt[m] > -np.inf).mean()))}
            tab[v] = t
        w = bp.bootstrap_weights(np.random.default_rng(SEED), sizes, N_REPS)
        ones = np.ones((1, sum(sizes)))
        paired = {}
        for v in ("u4096",) + RULES + ("R2s",):
            paired[v] = {}
            for lab, m in (("near_lt_18m", near), ("far_ge_18m", far)):
                obs = irs.paired_subset_recall(hits[v], hits["r2048"], pano_of, m, ones, 0.0)[0]
                dr = irs.paired_subset_recall(hits[v], hits["r2048"], pano_of, m, w, 0.0)
                paired[v][lab] = {"n": int(m.sum()), **irs._ci(obs, dr)}
            # how much of u4096's own far gain the rule keeps
        rep["bands"][name] = {"members": members, "recall_by_m": tab, "paired_vs_r2048": paired}
    return rep


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _d(c):
    return f"{c['observed']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]"


def markdown(rep):
    L = []
    order = ("R1", "R2", "R3", "u4096", "naive_union", "R2s")

    def main_table(title, entries):
        L.append(f"**{title}: each rule vs r2048 at 0.30 (paired 95% CI)**\n")
        L.append("| split | rule | P | R | F1 | ΔP | ΔR | ΔF1 | verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name, ent in entries:
            m = ent["metrics"]["r2048"]
            L.append(f"| {name} | r2048 | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} | — | — "
                     "| — | |")
            for v in order:
                m, r = ent["metrics"][v], ent["rules"][v]
                d = r["vs_r2048"]
                L.append(f"| {name} | {v} | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} | "
                         f"{_d(d['precision'])} | {_d(d['recall'])} | {_d(d['f1'])} | "
                         f"{r['verdict'] or ''} |")
        L.append("")

    def cost_table(title, entries):
        L.append(f"**{title}: cost of the recovered ramps, and the matched-recall "
                 "threshold-only baseline**\n")
        L.append("| split | rule | recovered ramps | extra FP | FP / recovered ramp | dedupe "
                 "dropped (r2048 / u4096) | matched r2048 threshold | its extra FP | its F1 | "
                 "fused − matched ΔF1 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, ent in entries:
            for v in order:
                r = ent["rules"][v]
                mb = r["matched_recall_baseline"]
                fpr = "—" if r["fp_per_recovered_ramp"] is None else \
                    f"{r['fp_per_recovered_ramp']:.2f}"
                dd = r["dropped_by_dedupe"]
                if mb is None:
                    mcols = "unreachable | — | — | —"
                else:
                    mcols = (f"{mb['threshold']:.2f} | {mb['extra_fp_vs_r2048']:+d} | "
                             f"{mb['F1']:.3f} | {_d(mb['fused_minus_matched']['f1'])}")
                L.append(f"| {name} | {v} | {r['recovered_ramps']:+d} | {r['extra_fp']:+d} | "
                         f"{fpr} | {dd['r2048']} / {dd['u4096']} | {mcols} |")
        L.append("")

    pooled = list(rep["pooled"].items())
    per = list(rep["per_split"].items())
    main_table("Pooled", pooled)
    main_table("Per split", per)
    cost_table("Pooled", pooled)
    cost_table("Per split", per)

    L.append("**Leave-one-split-out choice for R3 (D m, t_u; None = no fusion)**\n")
    L.append("| held-out split | D | t_u |")
    L.append("|---|---|---|")
    for c, e in rep["loso"].items():
        L.append(f"| {c} | {e['chosen']['D_m']} | {e['chosen']['t_u']} |")
    L.append("")

    L.append(f"**Wrong-pano control ({rep['protocol']['n_shifts']} cyclic shifts of the u4096 "
             "peaks within each split)**\n")
    L.append("| rule | split / pool | real ΔR | null ΔR mean | null ΔR max | attributable ΔR "
             "| real ΔFP | null ΔFP mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v, ent in rep["wrong_pano"].items():
        for name, e in ent.items():
            L.append(f"| {v} | {name} | {e['real_dR']:+.4f} | {e['null_dR_mean']:+.4f} | "
                     f"{e['null_dR_max']:+.4f} | {e['attributable_dR']:+.4f} | "
                     f"{e['real_dFP']:+d} | {e['null_dFP_mean']:+.1f} |")
    L.append("")

    for name, ent in rep["bands"].items():
        tab = ent["recall_by_m"]
        labels = list(tab["r2048"].keys())
        from recall_by_depth_112 import M_BUCKETS, bucket_label
        orderl = [bucket_label(lo, hi, "m") for lo, hi in M_BUCKETS]
        labels = [x for x in orderl if x in labels] + [x for x in labels if x not in orderl]
        L.append(f"**{name}: recall at 0.30 by flat-ground range**\n")
        L.append("| band | n | " + " | ".join(tab) + " |")
        L.append("|---|---|" + "---|" * len(tab))
        for lb in labels:
            L.append(f"| {lb} | {tab['r2048'][lb]['n']} | "
                     + " | ".join(f"{tab[v][lb]['recall']:.3f}" for v in tab) + " |")
        L.append("")
        L.append(f"**{name}: paired recall change vs r2048 by band (95% CI)**\n")
        L.append("| rule | near < 18 m | far ≥ 18 m |")
        L.append("|---|---|---|")
        for v, e in ent["paired_vs_r2048"].items():
            L.append(f"| {v} (n {e['near_lt_18m']['n']} / {e['far_ge_18m']['n']}) | "
                     f"{_d(e['near_lt_18m'])} | {_d(e['far_ge_18m'])} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Q4: inference cost from the #196 ledger rows
# --------------------------------------------------------------------------- #
def inference_cost(usage_log=USAGE_LOG):
    """GPU-side seconds per pano for r2048 and u4096 from the full #196 runs
    (the 2026-09-26 rows with panos_scored > 6, i.e. not the smoke tests)."""
    tot = {"r2048": [0.0, 0], "u4096": [0.0, 0]}
    with open(usage_log, encoding="utf-8") as f:
        for line in f:
            if '"input-res-25:' not in line:
                continue
            r = json.loads(line)
            arm = r["label"].split(":", 1)[1]
            if arm in tot and r["panos_scored"] > 6:
                tot[arm][0] += r["elapsed_s"]
                tot[arm][1] += r["panos_scored"]
    s = {a: v[0] / v[1] for a, v in tot.items()}
    return {"r2048_s_per_pano": rnd(s["r2048"]), "u4096_s_per_pano": rnd(s["u4096"]),
            "two_scale_s_per_pano": rnd(s["r2048"] + s["u4096"]),
            "ratio_vs_r2048": rnd((s["r2048"] + s["u4096"]) / s["r2048"], 2),
            "panos": {a: v[1] for a, v in tot.items()},
            "source": "analysis_out/usage_log.jsonl rows input-res-25:{r2048,u4096}, "
                      "makelab2 A40 fp32, GPU-side seconds (copy + forward + peaks)"}


def _strip_nan(obj):
    """Round-trip through JSON (tuples -> lists). Every float is already rounded by rnd()
    where it is produced, so this changes no value."""
    return json.loads(json.dumps(obj))


def outputs(cache_root=CACHE_ROOT, cities=SPLITS, n_shifts=N_SHIFTS):
    rep = build(cache_root, cities, n_shifts)
    rep["inference_cost"] = inference_cost()
    rep = _strip_nan(rep)
    js = json.dumps(rep, indent=1) + "\n"
    md = markdown(rep) + "\n"
    return js, md


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="CPU: every table -> results.json + results.md")
    r.add_argument("--cache-root", default=CACHE_ROOT)
    r.add_argument("--cities", default=",".join(SPLITS))
    r.add_argument("--n-shifts", type=int, default=N_SHIFTS)
    r.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    r.add_argument("--check", action="store_true",
                   help="regenerate in memory and compare bytes with --out and its .md")
    args = ap.parse_args(argv)
    cities = tuple(c.strip() for c in args.cities.split(",") if c.strip())
    js, md = outputs(args.cache_root, cities, args.n_shifts)
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
            want = text.encode("utf-8")
            if got != want:
                bad.append(f"{path}: committed sha256 {hashlib.sha256(got).hexdigest()[:12]} "
                           f"!= regenerated {hashlib.sha256(want).hexdigest()[:12]}")
        for b in bad:
            print(b)
        print("CHECK:", "FAIL" if bad else "OK (results.json and results.md reproduce byte "
              "for byte)")
        return 1 if bad else 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path, text in ((args.out, js), (md_path, md)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(md)
    print(f"\n-> {args.out}\n-> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
