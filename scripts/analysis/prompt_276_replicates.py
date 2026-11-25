"""#276 Step 7: does the Flash expert-definition gain on the open pool replicate?

Step 5 found that Jon's expert definition moves Gemini 3.8 Flash on the open pool
(laurens_mapillary + bayonne) by +0.068 macro F1 over zero-shot. That arm was an
exploratory amendment, and Gemini at temperature 0.0 is not deterministic, so Step 7
paid for both arms twice more (2026-11-11, own cache each) and asks two things:

* does expert beat zero-shot in every run, and
* is the difference larger than what re-running the SAME arm moves the score?

Everything is read from committed files, so a clean clone reproduces it with no cache:

* the original run: ``benchmark/model_detections/<leg>__<split>.json`` (Step 5 study legs)
* the replicates: ``benchmark/model_detections/replicates/<tag>/`` (``roster.REPLICATES``)
* the operating point: the dev_select thresholds in ``step5_results.json`` (0.00 for both
  Flash arms)

Scoring is Step 5's, imported from ``prompt_276_heldout``: per-pano counts at the
threshold, macro F1 over the two splits, and the paired pano bootstrap of the macro
difference (same panos resampled for both arms, 2,000 draws, seed 276).

    python scripts/analysis/prompt_276_replicates.py            # write step7_results.{json,md}
    python scripts/analysis/prompt_276_replicates.py --check    # regenerate and compare

A replicate differs from the original only in the detections; the per-arm range across
the three runs is the run-to-run noise. It is NOT a confidence interval, and three runs
are too few to call it a standard deviation with a straight face: read it as "re-running
moves the score by about this much".
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import export_model_cache as X          # noqa: E402
import prompt_276_heldout as H          # noqa: E402
from rampnet import roster              # noqa: E402

OUT_JSON = os.path.join(H.OUT_DIR, "step7_results.json")
OUT_MD = os.path.join(H.OUT_DIR, "step7_results.md")
STEP5_JSON = H.OUT_JSON

MODEL = "gemini-3.8-flash"
SHORT = "flash38"
#: arm -> the published study leg it is (``roster`` ``published_as``).
ARMS = {"zs": "gemini-3.8-flash-confidence-on",
        "expert": "gemini-3.8-flash-definition-expert-confidence-on"}
#: run -> the replicate-tag template, or None for the original Step 5 files.
RUNS = (("original", None), ("rep1", "flash-%s-conf-rep1-2026-11-11"),
        ("rep2", "flash-%s-conf-rep2-2026-11-11"))


def _r(v, nd=4):
    return round(float(v), nd)


def step5_thresholds():
    """{arm: threshold} for Flash, as selected on dev_select in Step 5."""
    with open(STEP5_JSON, encoding="utf-8") as f:
        rows = json.load(f)["thresholds"]
    thr = {t["arm"]: t["threshold"] for t in rows if t["short"] == SHORT and t["arm"] in ARMS}
    assert set(thr) == set(ARMS), thr
    return thr


def published_points(arm, run_tag, split):
    """{pano_id: [points]} of one (arm, run, split) from the committed files."""
    out_dir = X.PUBLISHED_DIR
    if run_tag:
        rep = roster.REPLICATES_BY_TAG[run_tag % arm]
        assert rep.of == ARMS[arm] and split in rep.splits, (rep, arm, split)
        out_dir = roster.replicate_dir(rep, X.PUBLISHED_DIR)
    det = X.load_detections(MODEL, split, published_dir=out_dir, publish_as=ARMS[arm])
    if det is None:
        raise FileNotFoundError("no published file for %s / %s / %s under %s"
                                % (arm, run_tag or "original", split, out_dir))
    return det


def build():
    rsq = H.radius_sq_for()
    thr = step5_thresholds()
    gts = {s: H.load_split_gts(s)[1] for s in H.OPEN_POOL}
    runs, macro = [], {a: [] for a in ARMS}
    for run, tag in RUNS:
        counts, per_split = {}, {}
        for arm in ARMS:
            counts[arm] = {}
            for s in H.OPEN_POOL:
                pts = published_points(arm, tag, s)
                missing = [p for p in gts[s] if p not in pts]
                assert not missing, (run, arm, s, len(missing))
                counts[arm][s] = H.pano_counts(pts, gts[s], rsq, thr[arm])
            per_split[arm] = {s: _r(H.f1_from_counts(*counts[arm][s].sum(0)))
                              for s in H.OPEN_POOL}
            macro[arm].append(float(np.mean([H.f1_from_counts(*counts[arm][s].sum(0))
                                             for s in H.OPEN_POOL])))
        d, lo, hi, p = H.paired_bootstrap(counts["expert"], counts["zs"],
                                          np.random.default_rng(H.SEED))
        runs.append({
            "run": run,
            "f1": per_split,
            "macro_f1": {a: _r(macro[a][-1]) for a in ARMS},
            "macro_delta_f1": _r(d), "ci95": [_r(lo), _r(hi)], "p": _r(p),
            "expert_ahead_on": [s for s in H.OPEN_POOL
                                if per_split["expert"][s] > per_split["zs"][s]],
        })
    spread = {a: {"macro_f1": [_r(v) for v in macro[a]],
                  "range": _r(max(macro[a]) - min(macro[a])),
                  "mean": _r(np.mean(macro[a]))} for a in ARMS}
    deltas = [r["macro_delta_f1"] for r in runs]
    return {
        "model": MODEL, "pool": list(H.OPEN_POOL), "thresholds": thr,
        "draws": H.DRAWS, "seed": H.SEED,
        "runs": runs, "spread": spread,
        "summary": {
            "delta_min": _r(min(deltas)), "delta_max": _r(max(deltas)),
            "all_ci_exclude_zero": all(r["ci95"][0] > 0 for r in runs),
            "expert_ahead_every_split_every_run": all(
                len(r["expert_ahead_on"]) == len(H.OPEN_POOL) for r in runs),
            "largest_same_arm_range": _r(max(s["range"] for s in spread.values())),
        },
    }


def render_md(res):
    lines = ["# #276 Step 7: Flash expert vs zero-shot on the open pool, three runs", "",
             "Generated by `scripts/analysis/prompt_276_replicates.py` from the committed "
             "detections. Do not edit by hand.", "",
             "Macro F1 over %s at the dev_select thresholds (zs %.2f, expert %.2f)."
             % (" + ".join(res["pool"]), res["thresholds"]["zs"], res["thresholds"]["expert"]),
             "",
             "| Run | Zero-shot | Expert | Difference | 95% interval | p | Expert ahead on |",
             "|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        lines.append("| %s | %.3f | %.3f | %+.3f | %+.3f to %+.3f | %s | %d of %d splits |" % (
            r["run"], r["macro_f1"]["zs"], r["macro_f1"]["expert"], r["macro_delta_f1"],
            r["ci95"][0], r["ci95"][1], H.fmt_p(r["p"]),
            len(r["expert_ahead_on"]), len(res["pool"])))
    lines += ["", "Per-split F1:", "",
              "| Run | Arm | " + " | ".join(res["pool"]) + " |",
              "|---|---|" + "---|" * len(res["pool"])]
    for r in res["runs"]:
        for arm in ("zs", "expert"):
            lines.append("| %s | %s | %s |" % (r["run"], arm, " | ".join(
                "%.3f" % r["f1"][arm][s] for s in res["pool"])))
    lines += ["", "Run-to-run spread of the same arm (three runs):", "",
              "| Arm | Macro F1 per run | Range |", "|---|---|---|"]
    for arm in ("zs", "expert"):
        s = res["spread"][arm]
        lines.append("| %s | %s | %.3f |" % (arm, ", ".join("%.3f" % v for v in s["macro_f1"]),
                                             s["range"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed JSON instead of writing")
    args = ap.parse_args(argv)
    res = build()
    rel = os.path.relpath(OUT_JSON, REPO_ROOT).replace("\\", "/")
    if args.check:
        with open(OUT_JSON, encoding="utf-8") as f:
            committed = json.load(f)
        if committed != res:
            print("MISMATCH: %s does not reproduce from the published detections" % rel)
            return 1
        print("OK: %s reproduces from the published detections (%d runs)" % (rel, len(res["runs"])))
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
