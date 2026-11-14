#!/usr/bin/env python
"""#276 rewrite arm: does Argon r1's open-pool gain replicate?

The held-out run (``prompt_276_rewrite_heldout.py``) found Argon r1 ahead of zs+confidence on
the open pool (laurens_mapillary + bayonne) by +0.046 macro F1, almost all of it on
laurens_mapillary, from one run per split. Gemini is not deterministic at temperature 0.0, so
Jon asked on 2026-11-14 for two more runs of both prompts on the open pool. Each replicate ran
into its own fresh cache directory (``.model_cache_argon_rep<k>``), so no call was served from
cache. This script asks the same two questions as Step 7 (``prompt_276_replicates.py``):

* is r1 ahead of zs in every run, and
* is the difference larger than what re-running the SAME prompt moves the score?

The operating points are NOT re-selected: each arm keeps the dev_select threshold the held-out
table chose (``rewrite_heldout.json``; zs 0.35, r1 0.32), so a replicate differs from the
original only in its detections. Scoring is Step 5's (``prompt_276_heldout``): per-pano counts
at the threshold, macro F1 over the two splits, and the paired pano bootstrap of the macro
difference (2,000 draws, seed 276). Never calls a model. Exploratory.

The detections live only in the local caches (like the held-out table), so this needs the main
checkout's cache plus the two replicate caches:

    python scripts/analysis/prompt_276_rewrite_replicates.py --cache-dir D:/Git/RampNet/.model_cache
    python scripts/analysis/prompt_276_rewrite_replicates.py --check --cache-dir ...

``--rep-cache-dir`` defaults to ``<cache-dir>_argon_rep1`` and ``<cache-dir>_argon_rep2``.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prompt_276_heldout as H  # noqa: E402

REPO_ROOT = H.REPO_ROOT
C, D = H.C, H.D
OUT_JSON = os.path.join(H.OUT_DIR, "rewrite_replicates.json")
OUT_MD = os.path.join(H.OUT_DIR, "rewrite_replicates.md")
HELDOUT_JSON = os.path.join(H.OUT_DIR, "rewrite_heldout.json")

MODEL, SHORT = "gemini-4-argon", "argon"
#: arm -> prompt body pin (None = the as-run prompt); both with --gemini-confidence.
ARMS = {"zs": None, "r1": "prompt276-argon-r1"}


def _r(v, nd=4):
    return round(float(v), nd)


def heldout_thresholds():
    """{arm: threshold} for Argon, as the held-out table selected them on dev_select."""
    with open(HELDOUT_JSON, encoding="utf-8") as f:
        rows = json.load(f)["thresholds"]
    thr = {t["arm"]: t["threshold"] for t in rows if t["short"] == SHORT and t["arm"] in ARMS}
    assert set(thr) == set(ARMS), thr
    return thr


def build(cache_dir, rep_cache_dirs):
    rsq = H.radius_sq_for()
    thr = heldout_thresholds()
    gts = {s: H.load_split_gts(s) for s in H.OPEN_POOL}
    dev_records, _, _ = C.load_bundle(os.path.join(REPO_ROOT, "benchmark", H.DEV_CITY))
    runs_in = [("original", cache_dir)] + [("rep%d" % (i + 1), d) for i, d in enumerate(rep_cache_dirs)]
    runs, macro = [], {a: [] for a in ARMS}
    for run, cdir in runs_in:
        cache = C.DetectionCache(cdir, enabled=True)
        args = H._compare_args(cdir)
        args.gemini_confidence = True
        counts, per_split = {}, {}
        for arm, body in ARMS.items():
            args.definition, args.prompt_body = None, body
            label, det = D.build_detector("gemini", MODEL, dev_records, args)
            sig = det.signature()
            counts[arm] = {}
            for s in H.OPEN_POOL:
                city, g = gts[s]
                pts, missing = H.cached_points(cache, label, sig, city, list(g))
                assert not missing, (run, arm, s, len(missing))
                counts[arm][s] = H.pano_counts(pts, g, rsq, thr[arm])
            per_split[arm] = {s: _r(H.f1_from_counts(*counts[arm][s].sum(0))) for s in H.OPEN_POOL}
            macro[arm].append(float(np.mean([H.f1_from_counts(*counts[arm][s].sum(0))
                                             for s in H.OPEN_POOL])))
        d, lo, hi, p = H.paired_bootstrap(counts["r1"], counts["zs"], np.random.default_rng(H.SEED))
        runs.append({
            "run": run, "f1": per_split,
            "macro_f1": {a: _r(macro[a][-1]) for a in ARMS},
            "macro_delta_f1": _r(d), "ci95": [_r(lo), _r(hi)], "p": _r(p),
            "r1_ahead_on": [s for s in H.OPEN_POOL if per_split["r1"][s] > per_split["zs"][s]],
        })
    spread = {a: {"macro_f1": [_r(v) for v in macro[a]],
                  "range": _r(max(macro[a]) - min(macro[a])),
                  "mean": _r(np.mean(macro[a]))} for a in ARMS}
    per_split_spread = {a: {s: _r(max(r["f1"][a][s] for r in runs) - min(r["f1"][a][s] for r in runs))
                            for s in H.OPEN_POOL} for a in ARMS}
    deltas = [r["macro_delta_f1"] for r in runs]
    return {
        "issue": "ProjectSidewalk/RampNet#276", "step": "rewrite-replicates",
        "model": MODEL, "pool": list(H.OPEN_POOL), "thresholds": thr,
        "prompt_bodies": ARMS, "draws": H.DRAWS, "seed": H.SEED,
        "runs": runs, "spread": spread, "per_split_range": per_split_spread,
        "summary": {
            "delta_min": _r(min(deltas)), "delta_max": _r(max(deltas)),
            "delta_mean": _r(np.mean(deltas)),
            "all_ci_exclude_zero": all(r["ci95"][0] > 0 for r in runs),
            "r1_ahead_every_split_every_run": all(len(r["r1_ahead_on"]) == len(H.OPEN_POOL)
                                                  for r in runs),
            "largest_same_arm_range": _r(max(s["range"] for s in spread.values())),
        },
    }


def render_md(res):
    L = ["# #276 rewrite arm: Argon r1 vs zero-shot on the open pool, three runs", "",
         "Generated by `scripts/analysis/prompt_276_rewrite_replicates.py` from the local caches. "
         "Do not edit by hand.", "",
         "Macro F1 over %s at the held-out table's dev_select thresholds (zs %.2f, r1 %.2f), "
         "both arms with confidence on. Paired pano bootstrap, %d draws, seed %d."
         % (" + ".join(res["pool"]), res["thresholds"]["zs"], res["thresholds"]["r1"],
            res["draws"], res["seed"]), "",
         "| Run | Zero-shot | r1 | Difference | 95% interval | p | r1 ahead on |",
         "|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        L.append("| %s | %.3f | %.3f | %+.3f | %+.3f to %+.3f | %s | %d of %d splits |" % (
            r["run"], r["macro_f1"]["zs"], r["macro_f1"]["r1"], r["macro_delta_f1"],
            r["ci95"][0], r["ci95"][1], H.fmt_p(r["p"]), len(r["r1_ahead_on"]), len(res["pool"])))
    L += ["", "Per-split F1:", "",
          "| Run | Arm | " + " | ".join(res["pool"]) + " |",
          "|---|---|" + "---|" * len(res["pool"])]
    for r in res["runs"]:
        for arm in ARMS:
            L.append("| %s | %s | %s |" % (r["run"], arm, " | ".join(
                "%.3f" % r["f1"][arm][s] for s in res["pool"])))
    L += ["", "Run-to-run spread of the same arm (three runs):", "",
          "| Arm | Macro F1 per run | Range | " + " | ".join("%s range" % s for s in res["pool"]) + " |",
          "|---|---|---|" + "---|" * len(res["pool"])]
    for arm in ARMS:
        s = res["spread"][arm]
        L.append("| %s | %s | %.3f | %s |" % (
            arm, ", ".join("%.3f" % v for v in s["macro_f1"]), s["range"],
            " | ".join("%.3f" % res["per_split_range"][arm][x] for x in res["pool"])))
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".model_cache"),
                    help="the cache holding the original held-out run")
    ap.add_argument("--rep-cache-dir", action="append",
                    help="a replicate's cache (repeatable); default <cache-dir>_argon_rep{1,2}")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed JSON instead of writing")
    args = ap.parse_args(argv)
    base = args.cache_dir.rstrip("/\\")
    reps = args.rep_cache_dir or [base + "_argon_rep1", base + "_argon_rep2"]
    res = build(args.cache_dir, reps)
    rel = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != res:
            print("MISMATCH: %s does not reproduce from the caches" % rel)
            return 1
        print("OK: %s reproduces from the caches (%d runs)" % (rel, len(res["runs"])))
        return 0
    os.makedirs(H.OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
