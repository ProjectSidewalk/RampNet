#!/usr/bin/env python
"""#276 Step 5: the frozen held-out run, regenerated from the caches.

Every held-out number in docs/vlm_prompt_276.md's Step 5 tables comes from here. For each
(model, arm) the script rebuilds the detector exactly as compare.py would, reads every pano of
every split from the cache, and scores. It never calls a model: a leg with uncached panos is
reported as pending and left out of every pool.

    python scripts/analysis/prompt_276_heldout.py            # write analysis_out/prompt_opt_276/step5_results.{json,md}
    python scripts/analysis/prompt_276_heldout.py --check    # regenerate and compare with the committed JSON

Arms (every one with elicited per-box confidence, concurrency 6, temperature 0.0):
    zs        the as-run prompt                                   both models
    alt       def276-flash38-alt, the Step 4 alternative (passed the dev_select gate)   Flash only
    para3     the best Step 3 paraphrase                          both models (amendment 2026-11-09)
    expert    Jon's blind definition                              both models (amendment 2026-11-09)

Operating point. Each (model, definition) gets ONE threshold, selected on sao_paulo
``dev_select`` with the same definition and confidence on: the threshold on the 0.01 grid of
observed scores that maximises dev_select F1 (lowest such threshold on a tie). The held-out
splits are then scored at that threshold, never at their own best. The unthresholded
(``raw``) F1 is reported beside it. ``manual_gold`` is scored on its committed 250-pano subset
for every arm; the zs arm's full 1,000 is reported as a separate row.

Pools. The pre-registered primary pool is the seven-city GSV US reference board (the board
before laurens_mapillary and juneau joined); the open pool is the Mapillary and Panoramax
splits. Pool statistics are the macro mean of per-split F1 at the selected thresholds. An
arm's comparison with the zs arm is a paired pano bootstrap (the same resampled panos score
both arms, per split, DRAWS draws, seed 276) of the macro ΔF1, with a two-sided bootstrap
p-value. The pre-registered primary test is Flash alt vs zs on the reference pool; the Holm
correction over the two models' primary tests reduces to that one test, since Argon has no
optimized arm. Everything else is exploratory and Holm-corrected within its family (model x
pool: the arms of one model against zs on one pool).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "model_comparison"))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import compare as C  # noqa: E402
import detectors as D  # noqa: E402
from fp_taxonomy import _compare_args  # noqa: E402
from rampnet.bundles import load_manual_ground_truths  # noqa: E402
from rampnet.detection_eval import aggregate, prediction_confidence, radius_sq_for, score_pano  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "analysis_out", "prompt_opt_276")
OUT_JSON = os.path.join(OUT_DIR, "step5_results.json")
OUT_MD = os.path.join(OUT_DIR, "step5_results.md")
SPLIT_FILE = "benchmark/prompt_opt_276/sao_paulo_split.json"
MG250_FILE = "benchmark/prompt_opt_276/manual_gold_250.json"
DEV_CITY = "sao_paulo"

MODELS = [("gemini-3.8-flash", "flash38"), ("gemini-4-argon", "argon")]
#: arm -> definition pin (None = the as-run prompt); per model
ARMS = {
    "flash38": [("zs", None), ("alt", "def276-flash38-alt"), ("para3", "para3"), ("expert", "expert")],
    "argon": [("zs", None), ("para3", "para3"), ("expert", "expert")],
}
PRIMARY = [("flash38", "alt")]          # the pre-registered primary test(s): arm vs zs, reference pool

REFERENCE_POOL = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson", "gainesville")
OPEN_POOL = ("laurens_mapillary", "bayonne")
OTHER_SPLITS = ("juneau", "laurens_gsv", "budapest_district5")
HELDOUT = REFERENCE_POOL + OPEN_POOL + OTHER_SPLITS
POOLS = {"reference": REFERENCE_POOL, "open": OPEN_POOL,
         "all12": HELDOUT, "all13": HELDOUT + ("manual_gold@250",)}
DRAWS = 2000
SEED = 276


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def load_split_gts(split):
    """{pid: GroundTruth} and the pano ids scored for one split name."""
    if split == "manual_gold@250":
        bundle = os.path.join(REPO_ROOT, "benchmark", "manual_gold")
        gts = load_manual_ground_truths(bundle)
        with open(os.path.join(REPO_ROOT, MG250_FILE), encoding="utf-8") as f:
            pids = json.load(f)["panos"]
        return "manual_gold", {p: gts[p] for p in pids}
    bundle = os.path.join(REPO_ROOT, "benchmark", split)
    if split == "manual_gold":
        return split, load_manual_ground_truths(bundle)
    records, verdicts, _ = C.load_bundle(bundle)
    return split, C.ground_truths_from_verdicts(records, verdicts)


def at_threshold(points, thr):
    return [p for p in points if (prediction_confidence(p) or 0.0) >= thr]


def pano_counts(points_by_pid, gts, rsq, thr):
    """Per-pano (tp, fp, tp_recall, n_gt_recall) at a threshold, in gts order: precision
    counts every pano, recall only the recall-confirmed ones, as ``aggregate`` does."""
    rows = []
    for pid, gt in gts.items():
        s = score_pano(at_threshold(points_by_pid[pid], thr), gt, rsq)
        rows.append((s.tp, s.fp, s.tp if s.fn_confirmed else 0, s.n_gt if s.fn_confirmed else 0))
    return np.array(rows, dtype=float)


def f1_from_counts(tp, fp, tp_recall, n_gt):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp_recall / n_gt if n_gt else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def report(points_by_pid, gts, rsq, thr):
    scores = [score_pano(at_threshold(points_by_pid[p], thr), gts[p], rsq) for p in gts]
    rep = aggregate(scores)
    return {"precision": _r(rep.precision), "recall": _r(rep.recall), "f1": _r(rep.f1),
            "precision_ci": [_r(v) for v in rep.precision_ci],
            "recall_ci": [_r(v) for v in rep.recall_ci],
            "ap": _r(rep.ap), "tp": int(rep.tp), "fp": int(rep.fp), "n_gt": int(rep.n_gt_recall),
            "n_pred": int(sum(len(at_threshold(points_by_pid[p], thr)) for p in gts))}


def select_threshold(points_by_pid, gts, rsq):
    """(threshold, dev F1 at it, sweep rows) on the 0.01 grid of observed scores plus 0."""
    confs = sorted({round(prediction_confidence(pt), 2) for pts in points_by_pid.values()
                    for pt in pts if prediction_confidence(pt) is not None})
    grid = [0.0] + [c for c in confs if c > 0.0]
    sweep, best = [], (0.0, -1.0)
    for thr in grid:
        c = pano_counts(points_by_pid, gts, rsq, thr).sum(axis=0)
        f1 = f1_from_counts(*c)
        sweep.append({"threshold": thr, "f1": _r(f1), "tp": int(c[0]), "fp": int(c[1])})
        if f1 > best[1] + 1e-12:
            best = (thr, f1)
    return best[0], _r(best[1]), sweep


def cached_points(cache, label, sig, city, pids):
    pts = {p: cache.get(C.cache_key(label, sig, city, p)) for p in pids}
    missing = [p for p, v in pts.items() if v is None]
    return pts, missing


def paired_bootstrap(counts_a, counts_b, rng):
    """Macro ΔF1 (a - b) over splits with a paired pano bootstrap.

    ``counts_*`` are {split: (n_panos, 3) arrays}; both arms resample the same panos
    of each split in each draw. Returns (delta, ci_lo, ci_hi, p_two_sided).
    """
    splits = sorted(counts_a)
    point = np.mean([f1_from_counts(*counts_a[s].sum(0)) - f1_from_counts(*counts_b[s].sum(0))
                     for s in splits])
    deltas = np.zeros(DRAWS)
    for s in splits:
        n = counts_a[s].shape[0]
        idx = rng.integers(0, n, size=(DRAWS, n))
        w = np.zeros((DRAWS, n))
        for d in range(DRAWS):
            w[d] = np.bincount(idx[d], minlength=n)
        ca, cb = w @ counts_a[s], w @ counts_b[s]
        fa = np.array([f1_from_counts(*row) for row in ca])
        fb = np.array([f1_from_counts(*row) for row in cb])
        deltas += (fa - fb) / len(splits)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return _r(point), _r(lo), _r(hi), _r(min(1.0, p))


def holm(tests):
    """Holm step-down over a list of dicts with ``p``; adds ``p_holm`` in place."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * tests[i]["p"])
        tests[i]["p_holm"] = _r(min(1.0, running))


def build(main_cache):
    rsq = radius_sq_for()
    cache = C.DetectionCache(main_cache, enabled=True)
    args = _compare_args(main_cache)
    args.gemini_confidence = True
    args.gemini_thinking_level = None
    with open(os.path.join(REPO_ROOT, SPLIT_FILE), encoding="utf-8") as f:
        dev_select = json.load(f)["dev_select"]
    dev_records, dev_verdicts, _ = C.load_bundle(os.path.join(REPO_ROOT, "benchmark", DEV_CITY))
    dev_gts_all = C.ground_truths_from_verdicts(dev_records, dev_verdicts)
    dev_gts = {p: dev_gts_all[p] for p in dev_select}
    split_gts = {s: load_split_gts(s) for s in HELDOUT + ("manual_gold@250", "manual_gold")}

    legs, thresholds, counts = [], [], {}   # counts[(short, arm)][split] -> per-pano array
    for model_id, short in MODELS:
        for arm, definition in ARMS[short]:
            args.definition = definition
            label, det = D.build_detector("gemini", model_id, dev_records, args)
            sig = det.signature()
            # 1. the operating point, from dev_select at this definition
            pts, missing = cached_points(cache, label, sig, DEV_CITY, dev_select)
            trow = {"model": model_id, "short": short, "arm": arm, "definition": definition,
                    "n_panos": len(dev_select), "n_cached": len(dev_select) - len(missing)}
            if missing:
                trow["status"] = "pending"
                thresholds.append(trow)
                continue
            thr, dev_f1, sweep = select_threshold(pts, dev_gts, rsq)
            trow.update({"status": "selected", "threshold": thr, "dev_select_f1": dev_f1,
                         "dev_select_f1_raw": sweep[0]["f1"], "sweep": sweep})
            thresholds.append(trow)
            # 2. every held-out split at that threshold
            counts[(short, arm)] = {}
            for split in HELDOUT + ("manual_gold@250",) + (("manual_gold",) if arm == "zs" else ()):
                city, gts = split_gts[split]
                pts, missing = cached_points(cache, label, sig, city, list(gts))
                row = {"model": model_id, "short": short, "arm": arm, "definition": definition,
                       "split": split, "threshold": thr, "n_panos": len(gts),
                       "n_cached": len(gts) - len(missing)}
                if missing:
                    row["status"] = "pending"
                    legs.append(row)
                    continue
                row["status"] = "scored"
                row.update(report(pts, gts, rsq, thr))
                raw = report(pts, gts, rsq, 0.0)
                row["raw"] = {k: raw[k] for k in ("precision", "recall", "f1", "tp", "fp")}
                confs = [prediction_confidence(pt) for ps in pts.values() for pt in ps]
                row["n_scored_boxes"] = len(confs)
                row["n_distinct_scores"] = len({round(c, 4) for c in confs if c is not None})
                legs.append(row)
                counts[(short, arm)][split] = pano_counts(pts, gts, rsq, thr)

    # 3. pools: macro F1 per arm, and each arm against zs with the paired bootstrap
    scored = {k: v for k, v in counts.items()}
    pools, tests = [], []
    for pool_name, members in POOLS.items():
        for model_id, short in MODELS:
            fam = []
            for arm, _definition in ARMS[short]:
                c = scored.get((short, arm), {})
                if any(m not in c for m in members):
                    pools.append({"pool": pool_name, "short": short, "arm": arm, "status": "pending",
                                  "missing": [m for m in members if m not in c]})
                    continue
                per = {m: _r(f1_from_counts(*c[m].sum(0))) for m in members}
                tot = np.sum([c[m].sum(0) for m in members], axis=0)
                pools.append({"pool": pool_name, "short": short, "arm": arm, "status": "scored",
                              "macro_f1": _r(np.mean(list(per.values()))),
                              "pooled_f1": _r(f1_from_counts(*tot)), "per_split_f1": per})
                if arm != "zs" and all(m in scored.get((short, "zs"), {}) for m in members):
                    rng = np.random.default_rng(SEED)
                    d, lo, hi, p = paired_bootstrap({m: c[m] for m in members},
                                                    {m: scored[(short, "zs")][m] for m in members}, rng)
                    t = {"pool": pool_name, "short": short, "arm": arm, "vs": "zs",
                         "macro_delta_f1": d, "ci95": [lo, hi], "p": p,
                         "family": "primary" if (pool_name == "reference" and (short, arm) in PRIMARY)
                         else "exploratory"}
                    fam.append(t)
            if fam:
                holm([t for t in fam if t["family"] == "exploratory"] or fam)
                tests.extend(fam)
    primary = [t for t in tests if t["family"] == "primary"]
    if primary:
        holm(primary)
    return {"issue": "ProjectSidewalk/RampNet#276", "step": 5, "match_radius": 0.022,
            "draws": DRAWS, "seed": SEED, "pools": {k: list(v) for k, v in POOLS.items()},
            "thresholds": thresholds, "legs": legs, "pool_stats": pools, "tests": tests}


def render_md(res):
    L = ["# #276 Step 5: frozen held-out run (regenerated)", "",
         "Match radius %s. One threshold per (model, definition), selected on sao_paulo dev_select; "
         "held-out splits scored at it. Pool numbers are macro means of per-split F1; Δ rows are "
         "paired pano bootstraps (%d draws, seed %d) of the macro ΔF1 against the zs arm. "
         "Regenerate with `python scripts/analysis/prompt_276_heldout.py`." % (
             res["match_radius"], res["draws"], res["seed"]), "",
         "## Selected thresholds (dev_select, confidence on)", "",
         "| model | arm | definition | threshold | dev_select F1 at it | raw F1 (no threshold) |",
         "|---|---|---|---|---|---|"]
    for t in res["thresholds"]:
        if t["status"] != "selected":
            L.append("| %s | %s | %s | pending (%d/%d cached) | | |" % (
                t["model"], t["arm"], t["definition"] or "seed", t["n_cached"], t["n_panos"]))
        else:
            L.append("| %s | %s | %s | %.2f | %.3f | %.3f |" % (
                t["model"], t["arm"], t["definition"] or "seed", t["threshold"],
                t["dev_select_f1"], t["dev_select_f1_raw"]))
    L += ["", "## Pools (macro F1 at the selected thresholds; pooled F1 in parentheses)", "",
          "| pool | model | arm | macro F1 | pooled F1 |", "|---|---|---|---|---|"]
    for p in res["pool_stats"]:
        if p["status"] != "scored":
            L.append("| %s | %s | %s | pending (%s) | |" % (p["pool"], p["short"], p["arm"],
                                                             ", ".join(p["missing"])))
        else:
            L.append("| %s | %s | %s | **%.3f** | %.3f |" % (
                p["pool"], p["short"], p["arm"], p["macro_f1"], p["pooled_f1"]))
    L += ["", "## Arm minus zs (paired pano bootstrap of the macro ΔF1)", "",
          "| pool | model | arm | ΔF1 | 95% CI | p | Holm p | family |", "|---|---|---|---|---|---|---|---|"]
    for t in res["tests"]:
        L.append("| %s | %s | %s | %+.3f | [%+.3f, %+.3f] | %.3f | %s | %s |" % (
            t["pool"], t["short"], t["arm"], t["macro_delta_f1"], t["ci95"][0], t["ci95"][1],
            t["p"], ("%.3f" % t["p_holm"]) if "p_holm" in t else "", t["family"]))
    for model_id, short in MODELS:
        L += ["", "## %s, per split" % model_id, "",
              "| arm | split | thr | P | R | F1 | AP | tp/fp/n_gt | raw F1 | distinct scores |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["legs"]:
            if r["short"] != short:
                continue
            if r["status"] != "scored":
                L.append("| %s | %s | | | | pending (%d/%d cached) | | | | |" % (
                    r["arm"], r["split"], r["n_cached"], r["n_panos"]))
                continue
            L.append("| %s | %s | %.2f | %.3f (%.3f-%.3f) | %.3f (%.3f-%.3f) | **%.3f** | %s | %d/%d/%d | %.3f | %d |" % (
                r["arm"], r["split"], r["threshold"],
                r["precision"], r["precision_ci"][0], r["precision_ci"][1],
                r["recall"], r["recall_ci"][0], r["recall_ci"][1], r["f1"],
                ("%.3f" % r["ap"]) if r["ap"] is not None else "-",
                r["tp"], r["fp"], r["n_gt"], r["raw"]["f1"], r["n_distinct_scores"]))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".model_cache"))
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed JSON instead of writing")
    args = ap.parse_args(argv)
    res = build(args.cache_dir)
    rel = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != res:
            print("MISMATCH: %s does not reproduce from the cache" % rel)
            return 1
        print("OK: %s reproduces from the cache (%d scored legs, %d tests)" % (
            rel, sum(r["status"] == "scored" for r in res["legs"]), len(res["tests"])))
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(res))
    for t in res["thresholds"]:
        print("%-8s %-7s threshold %s  dev_select F1 %s" % (
            t["short"], t["arm"], t.get("threshold", "pending"), t.get("dev_select_f1", "")))
    for p in res["pool_stats"]:
        if p["pool"] in ("reference", "open"):
            print("%-10s %-8s %-7s %s" % (p["pool"], p["short"], p["arm"],
                                         ("macro F1 %.3f" % p["macro_f1"]) if p["status"] == "scored"
                                         else "pending " + ",".join(p["missing"])))
    for t in res["tests"]:
        if t["pool"] in ("reference", "open"):
            print("%-10s %-8s %-7s vs zs  dF1 %+.3f [%+.3f, %+.3f] p %.3f %s" % (
                t["pool"], t["short"], t["arm"], t["macro_delta_f1"], t["ci95"][0], t["ci95"][1],
                t["p"], t["family"]))
    pend = [r for r in res["legs"] if r["status"] != "scored"]
    if pend:
        print("pending: " + ", ".join("%s/%s/%s %d/%d" % (r["short"], r["arm"], r["split"],
                                                          r["n_cached"], r["n_panos"]) for r in pend))
    print("wrote %s and .md" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
