"""Tables for the pairwise-matching family of cross-view arms (#48).

Reads only committed files: the frozen ``pairs.csv`` and ``predictions/<arm>.jsonl`` under
``analysis_out/crossview_align_48/``. Writes ``analysis_out/crossview_align_48/matching/
report.json`` and prints the markdown tables used in ``docs/crossview_align_48/matching.md``.

What it adds to the harness's ``score``:

* **paired gain against ``proj_height_auto``** as well as against the 2.6 m projection, on
  all pairs and on the pairs the arm aligned (the harness reports only the latter against
  the projection). An image arm has to beat the free geometry prior to be worth running.
* **harm**: the share of aligned pairs the arm moved more than 2 deg *further* from the
  reference than ``proj_height_auto`` was.
* **why pairs fall back**: the ``why`` diagnostics the family's arms record, and fallback
  rates by imagery, range, capture month and camera baseline.

Usage::

    python scripts/analysis/crossview_matching_48.py            # default arm list
    python scripts/analysis/crossview_matching_48.py --arms roma,lg
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

import crossview_align_48 as H

OUT_JSON = os.path.join(H.OUT, "matching", "report.json")
AUTO = "proj_height_auto"
HARM_DEG = 2.0
DEFAULT_ARMS = ("lg", "lg_magsac", "lg_epi", "lg_hyb", "sp_lg", "sp_lg_epi", "sp_lg_hyb",
                "disk_lg", "disk_lg_epi", "siftlg", "siftlg_epi", "loftr", "loftr_epi",
                "loftr_hyb", "roma", "roma_magsac", "roma_local", "roma_epi", "roma_hyb",
                "roma_warp", "roma_warp_hyb")


def load_raw(name):
    path, _ = H.prediction_paths(name)
    with open(path, encoding="utf-8") as f:
        return {r["pair_id"]: r for r in (json.loads(line) for line in f if line.strip())}


def boot(groups, stat):
    ci = H.cluster_bootstrap(groups, stat)
    return None if ci is None else [round(ci[0], 3), round(ci[1], 3)]


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def block(pairs, idx, arm_e, proj_e, auto_e):
    """Summary of one arm on pair indices ``idx``. *_e: harness arm_errors rows."""
    by_ramp = defaultdict(list)
    for i in idx:
        by_ramp[pairs[i]["ramp_uid"]].append(i)
    groups = [v for _, v in sorted(by_ramp.items())]
    aligned = [i for i in idx if not arm_e[i][3]]
    ag = [v for v in ([i for i in g if not arm_e[i][3]] for g in groups) if v]

    def m_arm(ii):
        return med([arm_e[i][0] for i in ii])

    def g_proj(ii):
        return med([proj_e[i][0] - arm_e[i][0] for i in ii])

    def g_auto(ii):
        return med([auto_e[i][0] - arm_e[i][0] for i in ii])

    out = {"n": len(idx), "median_deg": m_arm(idx), "median_ci": boot(groups, m_arm),
           "within_2deg": float(np.mean([arm_e[i][0] <= 2.0 for i in idx])),
           "fallback_rate": float(np.mean([arm_e[i][3] for i in idx])),
           "gain_vs_projection": g_proj(idx), "gain_vs_projection_ci": boot(groups, g_proj),
           "gain_vs_auto": g_auto(idx), "gain_vs_auto_ci": boot(groups, g_auto),
           "projection_median_deg": med([proj_e[i][0] for i in idx]),
           "auto_median_deg": med([auto_e[i][0] for i in idx])}
    if aligned:
        out["aligned"] = {
            "n": len(aligned), "median_deg": m_arm(aligned), "median_ci": boot(ag, m_arm),
            "projection_median_deg": med([proj_e[i][0] for i in aligned]),
            "auto_median_deg": med([auto_e[i][0] for i in aligned]),
            "gain_vs_projection": g_proj(aligned), "gain_vs_projection_ci": boot(ag, g_proj),
            "gain_vs_auto": g_auto(aligned), "gain_vs_auto_ci": boot(ag, g_auto),
            "within_2deg": float(np.mean([arm_e[i][0] <= 2.0 for i in aligned])),
            "auto_within_2deg": float(np.mean([auto_e[i][0] <= 2.0 for i in aligned])),
            "harm_share": float(np.mean([arm_e[i][0] > auto_e[i][0] + HARM_DEG for i in aligned])),
            "help_share": float(np.mean([arm_e[i][0] < auto_e[i][0] - HARM_DEG for i in aligned]))}
    return out


def baseline_bin(b):
    return "baseline<10m" if b < 10 else ("baseline10-20m" if b < 20 else "baseline>=20m")


def fallback_strata(pairs, raw):
    """Fallback rate of one arm (raw predictions) by covariate."""
    keys = {"imagery": lambda p: p["imagery"], "city": lambda p: p["city"],
            "range": lambda p: H.range_bin(p["oth_range_m"]),
            "date": lambda p: "same_month" if p["same_date"] else "different_month",
            "baseline": lambda p: baseline_bin(p["baseline_m"])}
    out = {}
    for k, fn in keys.items():
        acc = defaultdict(list)
        for p in pairs:
            acc[fn(p)].append(raw[p["pair_id"]].get("x") is None)
        out[k] = {s: {"n": len(v), "fallback_rate": float(np.mean(v))} for s, v in sorted(acc.items())}
    return out


def lg_reasons(pairs, hyb_raw):
    """Why the pilot's ``lg`` fell back, read from ``<matcher>_hyb``'s first stage (the
    same RANSAC ground homography on the same matches). Three causes, in order:
    the views barely match at all; they match, but not on the ground; ground matches
    exist but no single plane explains 15 of them (or it maps the point off the view)."""
    out = Counter()
    for p in pairs:
        r = hyb_raw[p["pair_id"]]
        if r.get("via") == "homography":
            out["aligned"] += 1
            continue
        n, g = r.get("n_matches", 0), r.get("n_ground", 0)
        if n < H_MIN:
            out["views_barely_match (<15 matches anywhere)"] += 1
        elif g < H_MIN:
            out["matches_off_ground_only (<15 ground matches)"] += 1
        elif r.get("why_h") == "mapped_outside_view":
            out["plane_maps_point_off_view"] += 1
        else:
            out["ground_matches_but_no_plane (<15 inliers)"] += 1
    return dict(out)


H_MIN = 15


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)
    pairs = H.read_frozen_pairs()
    names = args.arms.split(",")
    proj_e = H.arm_errors(pairs, None)
    auto_e = H.arm_errors(pairs, H.read_predictions(AUTO))
    strata = {"all": list(range(len(pairs))),
              "gsv": [i for i, p in enumerate(pairs) if p["imagery"] == "gsv"],
              "mapillary": [i for i, p in enumerate(pairs) if p["imagery"] == "mapillary"]}
    res = {"config": {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT, "seed": H.SEED,
                      "harm_deg": HARM_DEG, "prior_arm": AUTO,
                      "ci": "2.5-97.5 percentile, ramps resampled"},
           "reference": {s: {"projection_median_deg": med([proj_e[i][0] for i in idx]),
                             "auto_median_deg": med([auto_e[i][0] for i in idx]),
                             "auto_within_2deg": float(np.mean([auto_e[i][0] <= 2 for i in idx])),
                             "projection_within_2deg": float(np.mean([proj_e[i][0] <= 2 for i in idx]))}
                         for s, idx in strata.items()},
           "arms": {}, "fallback_by": {}, "why": {}, "lg_family_first_stage": {}}
    for name in names:
        preds = H.read_predictions(name)
        e = H.arm_errors(pairs, preds)
        res["arms"][name] = {s: block(pairs, idx, e, proj_e, auto_e) for s, idx in strata.items()}
        raw = load_raw(name)
        if not name.endswith("hyb"):
            res["fallback_by"][name] = fallback_strata(pairs, raw)
        res["why"][name] = dict(Counter(r.get("why") or ("aligned" if r.get("x") is not None else "none")
                                        for r in raw.values()))
        if name.endswith("_hyb") and not name.startswith("roma_warp"):
            res["lg_family_first_stage"][name[:-4]] = lg_reasons(pairs, raw)
            res["arms"][name]["via"] = dict(Counter(str(r.get("via")) for r in raw.values()))
    H.write_json(args.out, res)
    print_tables(res)
    print(f"-> {args.out}")


def f(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def fci(v, ci):
    return f"{f(v)} [{f(ci[0])}, {f(ci[1])}]" if ci else f(v)


def print_tables(res):
    for s in ("all", "gsv", "mapillary"):
        ref = res["reference"][s]
        print(f"\n### {s}: projection {f(ref['projection_median_deg'])}, auto "
              f"{f(ref['auto_median_deg'])}\n")
        print("| arm | median ° [CI] | within 2° | fallback | gain vs projection [CI] | "
              "gain vs auto [CI] | aligned n | aligned: arm / proj / auto ° | aligned gain vs "
              "auto [CI] | harm / help |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, st in res["arms"].items():
            b = st[s]
            a = b.get("aligned", {})
            print(f"| {name} | {fci(b['median_deg'], b['median_ci'])} | {f(b['within_2deg'])} | "
                  f"{f(b['fallback_rate'])} | {fci(b['gain_vs_projection'], b['gain_vs_projection_ci'])} | "
                  f"{fci(b['gain_vs_auto'], b['gain_vs_auto_ci'])} | {a.get('n', 0)} | "
                  f"{f(a.get('median_deg'))} / {f(a.get('projection_median_deg'))} / "
                  f"{f(a.get('auto_median_deg'))} | "
                  f"{fci(a.get('gain_vs_auto'), a.get('gain_vs_auto_ci'))} | "
                  f"{f(a.get('harm_share'))} / {f(a.get('help_share'))} |")
    print("\n### why (first stage of each *_hyb = that matcher's RANSAC ground homography)\n")
    for k, v in res["lg_family_first_stage"].items():
        print(f"- {k}: {v}")
    print("\n### fallback by covariate\n")
    for k, v in res["fallback_by"].items():
        print(f"- {k}: " + "; ".join(f"{c}: " + ", ".join(f"{s} {f(x['fallback_rate'])} (n={x['n']})"
                                                          for s, x in d.items()) for c, d in v.items()))


if __name__ == "__main__":
    main()
