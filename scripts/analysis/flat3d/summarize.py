"""Summary tables for #214 from committed files only: reconstruction statistics per variant
(``corners/*.json``) and paired arm-vs-arm differences on the Richmond pairs
(``results_richmond.json`` per-pair errors). Writes ``analysis_out/flat_mapillary_3d/summary.json``.

    python scripts/analysis/flat3d/summarize.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import crossview_align_48 as H  # noqa: E402
import flat_mapillary_48 as F  # noqa: E402

PAIRED = [("flat_sfm", "noflat_sfm"), ("mlypano_sfm", "noflat_sfm"),
          ("mlypano_sfm", "flat_sfm"), ("flat_sfm", "sfm_colmap"),
          ("flat_sfm", "mast3r_pair"), ("flat_sfm", "roma_magsac"),
          ("flat_gsmed", "flat_gs"), ("flat_mvs", "flat_sfm"), ("flat_gsmed", "flat_sfm")]


def recon_stats():
    by = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(F.OUT, "corners", "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        by[r.get("variant", "flat")].append(r)
    out = {}
    for v, rs in sorted(by.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        reg = [r["model"]["n_registered"] for r in ok]
        inp = [sum(r["n_images_by_kind"].values()) for r in rs]
        kinds = defaultdict(lambda: [0, 0])
        for r in ok:
            for k, n in r["n_images_by_kind"].items():
                kinds[k][0] += n
                kinds[k][1] += r["model"]["registered_by_kind"].get(k, 0)
        n_pairs = sum(len(r["pairs"]) for r in rs)
        co = sum(1 for r in ok for pr in r["pairs"].values()
                 if "x" in pr.get("sparse", {}) or pr.get("sparse", {}).get("reason")
                 not in ("oth_not_registered",))
        gs = [r["gs"] for r in ok if "gs" in r]
        out[v] = {
            "corners": len(rs), "ok": len(ok),
            "src_and_all_oth_registered": sum(
                1 for r in ok if all(pr.get("sparse", {}).get("reason") != "oth_not_registered"
                                     for pr in r["pairs"].values())),
            "pairs": n_pairs, "pairs_co_registered": co,
            "images_in_median": float(np.median(inp)),
            "registered_median": float(np.median(reg)), "registered_range": [min(reg), max(reg)],
            "registered_by_kind": {k: {"input": a, "registered": b} for k, (a, b) in kinds.items()},
            "points3d_median": float(np.median([r["model"]["n_points3d"] for r in ok])),
            "mean_reproj_px_median": float(np.median([r["model"]["mean_reproj_px"] for r in ok])),
            "mean_reproj_px_range": [min(r["model"]["mean_reproj_px"] for r in ok),
                                     max(r["model"]["mean_reproj_px"] for r in ok)],
            "prior_residual_h_m_median_of_medians": float(np.median(
                [r["model"]["prior_residual_h_m_median"] for r in ok])),
            "elapsed_s_total": round(sum(r.get("elapsed_s", 0) for r in rs), 1),
            "gs_psnr_median": float(np.median([g["train_psnr_median"] for g in gs])) if gs else None,
            "gs_gaussians_median": float(np.median([g["n_gaussians"] for g in gs])) if gs else None,
            "gs_ply_mb_median": float(np.median([g["ply_bytes"] for g in gs])) / 1e6 if gs else None,
        }
    return out


def paired():
    res = json.load(open(F.RESULTS, encoding="utf-8"))
    pairs = [p for p in H.read_frozen_pairs() if p["city"] == F.CITY]
    pp = {r["pair_id"]: r for r in res["per_pair"]}
    out = {}
    for a, b in PAIRED:
        if a not in res["arms"] or b not in res["arms"]:
            continue
        by = defaultdict(list)
        for p in pairs:
            r = pp[p["pair_id"]]
            by[p["ramp_uid"]].append(r[b] - r[a])      # > 0: a is closer
        groups = [v for _, v in sorted(by.items())]
        flat = [x for g in groups for x in g]
        out[f"{a}_vs_{b}"] = {
            "median_gain_deg": float(np.median(flat)),
            "ci": H.cluster_bootstrap(groups, lambda xs: float(np.median(xs))),
            "a_closer": float(np.mean([x > 1e-9 for x in flat])),
            "n": len(flat), "sign": f"> 0 means {a} is closer to the reference than {b}"}
    return out


def main():
    s = {"reconstruction": recon_stats(), "paired": paired()}
    H.write_json(os.path.join(F.OUT, "summary.json"), s)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
