"""RampNet vs the YOLO baselines at MATCHED operating points (#51).

THE PROBLEM THIS FIXES
----------------------
``docs/model_scoreboard.md`` and ``docs/yolo_geometry_51.md`` compare F1 across rows
whose operating points were chosen by different procedures:

  * every YOLO leg is scored at **conf 0.25** -- the Ultralytics default. Nobody
    selected it; it is what ``predict()`` uses when you do not say otherwise.
  * RampNet is scored at **0.55** -- its shipped deployment threshold, which #54/#55
    already established is not F1-optimal (0.30 is the recommendation).

So the published gap mixes "which model is better" with "whose default happened to
suit this metric". The scoreboard warns to read the op column before comparing rows;
this script measures what that warning is worth.

WHAT PARITY MEANS HERE
----------------------
One threshold per model, chosen the same way for every model, on data the headline is
not reported over:

  1. **Select** each model's single uniform threshold on a DEV split, by F1.
  2. **Report** every model at that threshold, macro-meaned over ``POOLED_SPLITS``.

The dev split defaults to ``sao_paulo``: it is already outside the seven-city pool the
headline is computed over, so selection never touches a reported split, and unlike
``budapest_district5`` (the benchmark's only ranking inversion, low reviewer
confidence) or ``manual_gold`` (the only independently-labelled split) it is
unremarkable. ``--sensitivity`` re-runs the whole selection with each of the three
non-pooled splits as dev, so the choice can be seen not to carry the result.

The threshold is UNIFORM across splits. Picking a per-split best would be tune-on-test
and is not offered.

DELIBERATELY GENEROUS TO RAMPNET
--------------------------------
RampNet is swept on a full 0.05-step grid from its cache floor to 0.95, while the YOLO
legs are limited to the grid their committed reports carry (0.05-step to 0.30, then
0.40/0.50/0.60/0.70). Both models' optima are interior to their own dense regions, so
this changes nothing -- but where it could, it favours RampNet, which is the
conservative direction for the finding.

BOTH SIDES ARE FLOORED AT 0.05, AND THAT IS NOT AN ASYMMETRY
------------------------------------------------------------
``analysis_out/op_cache/*.json`` carries ``meta.score_floor = 0.05`` and
``YoloDetector.score_threshold = 0.05`` is the YOLO cache floor. The AP columns are
therefore already like-for-like. What the shared floor *does* mean is that a model
whose best grid point is 0.05 is reported at the edge of what was measured, so its
parity F1 is a LOWER BOUND -- flagged per model in the output.

SELF-CHECK, AND WHY IT IS NOT AN EQUALITY
-----------------------------------------
RampNet is swept from ``op_cache`` because that is the only source that goes below the
deployment threshold at all -- the committed bundle records stop at ~0.55 (measured
floor 0.5501 over the pooled splits). The two sources do not agree exactly at 0.55:

    op_cache filtered to >= 0.55   F1 0.824
    bundle records as shipped      F1 0.827   <- docs/model_scoreboard.md

That -0.0025 is not a bug, not a floor effect (the bundle floor is *above* 0.55), and
NOT a peak-extraction artifact. Lowering ``threshold_abs`` can only ADD candidates:
``peak_local_max`` suppresses on a maximum filter, so a pixel that is the maximum of its
``min_distance`` neighbourhood at 0.55 is still that maximum at 0.05, and the >= 0.55
subset of a 0.05-floor extraction is identical to a 0.55-floor extraction. (An earlier
version of this docstring gave that as the mechanism. It was wrong.)

The two sources differ because they are two different HEATMAPS, and
``docs/operating_point.md`` already measured which and why:

  * FIVE splits agree, to the 3 decimals the published per-split values carry:
    richmond, clovis, morgantown, annapolis, budapest_district5. These are the Mapillary
    splits, recorded there as reproducing bit-exactly. They are the real regression
    guard -- ``CONTROL_EXACT_SPLITS``, tolerance ``CONTROL_TOL_EXACT``.
  * The four GSV splits (bend, paterson, gainesville, sao_paulo) differ by -0.019 to
    +0.003. The production path assembled tiles into a 4096x2048 intermediate, so the
    shipped detections came from a DIFFERENT RESAMPLE of the same panorama than these
    native-resolution caches; detections sit up to 0.44 match radii apart.
  * ``manual_gold`` differs by -0.009 because its committed detections were exported
    WITH horizontal-flip TTA (``benchmark/manual_gold/detections_meta.json``) and
    op_cache is the no-TTA deployment path.

Two consequences travel with every number below, and they are stated in
``docs/operating_point_parity_51.md`` too:

  1. The pooled macro passes ``CONTROL_TOL`` partly by CANCELLATION (+0.003 on bend
     against -0.004 and -0.016 on paterson and gainesville). A macro-only control would
     have hidden a split-level divergence, so the control is asserted per split as well.
  2. The default dev split, ``sao_paulo``, has the LARGEST discrepancy of the ten
     (-0.019). ``--sensitivity`` is what answers that: the selection lands on 0.30-0.35
     for RampNet whichever candidate is used.

Everything here is op_cache-derived end to end, so the comparison is internally
consistent -- and because three of the seven pooled splits are GSV, RampNet's parity F1
is about 0.002 LOW relative to a bundle-derived path, i.e. the reported gap is very
slightly conservative rather than flattering. ``--check`` turns artifact drift into a
non-zero exit.

USAGE
    python scripts/analysis/operating_point_parity_51.py
    python scripts/analysis/operating_point_parity_51.py --dev-split manual_gold
    python scripts/analysis/operating_point_parity_51.py --sensitivity
    python scripts/analysis/operating_point_parity_51.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)

from analysis.yolo_geometry_51 import (  # noqa: E402
    ALL_SPLITS_AS_RUN, IN_DIR, LEGS, POOLED_SPLITS, PUBLISHED_RAMPNET_F1, rnd)
from rampnet.detection_eval import aggregate, radius_sq_for, score_pano  # noqa: E402

OP_CACHE = os.path.join(REPO, "analysis_out", "op_cache")
OUT_JSON = os.path.join(REPO, "docs", "data", "operating_point_parity_51.json")

RAMPNET = "RampNet"
DEPLOYED_THRESHOLD = 0.55

# The split population is PINNED to the 2026-08-30 YOLO run, via yolo_geometry_51 --
# NOT the live registry in ``analysis.low_floor_sweep``. A split added to the benchmark
# later has no YOLO report, so following the registry would drop every YOLO leg (each
# needs a full pool) while RampNet quietly re-pooled over a different population, and
# the two sides of the comparison would stop being the same comparison.

# ``PUBLISHED_RAMPNET_F1`` (0.827, bundle-derived at 0.55) is imported rather than
# restated so the two scripts cannot drift apart on the number they both check against.

# RampNet's per-split published F1 at 0.55, bundle-derived, from the by-split table in
# docs/model_scoreboard.md. The pooled 0.827 is the macro-mean of the first seven.
PUBLISHED_RAMPNET_PER_SPLIT_F1 = {
    "richmond": 0.855, "bend": 0.850, "clovis": 0.801, "morgantown": 0.835,
    "annapolis": 0.839, "paterson": 0.805, "gainesville": 0.803,
    "budapest_district5": 0.644, "sao_paulo": 0.777, "manual_gold": 0.908,
}

# Splits where the op_cache path and the published bundle are the SAME computation, so
# any disagreement beyond 3-dp rounding of the published value is a scoring regression.
# docs/operating_point.md records these five as reproducing bit-exactly.
CONTROL_EXACT_SPLITS = ("richmond", "clovis", "morgantown", "annapolis", "budapest_district5")
CONTROL_TOL_EXACT = 0.002         # 3-dp rounding of the published per-split value
# Splits with a DOCUMENTED reason for the two paths to differ (docs/operating_point.md):
# the four GSV splits were produced from a different resample of the panorama, and
# manual_gold's committed detections carry horizontal-flip TTA that op_cache does not.
CONTROL_TOL_DIVERGENT = 0.025
CONTROL_TOL = 0.005               # pooled macro; see SELF-CHECK in the module docstring

# Candidate dev splits: everything the headline is NOT macro-meaned over.
NON_POOLED = tuple(s for s in ALL_SPLITS_AS_RUN if s not in POOLED_SPLITS)
DEFAULT_DEV = "sao_paulo"

# A sweep row: "   0.05  0.789  0.823  0.806        255/68/55 <- best F1"
SWEEP_ROW_RE = re.compile(
    r"^\s*(?P<thr>\d\.\d+)\s+(?P<p>\d\.\d+)\s+(?P<r>\d\.\d+)\s+(?P<f1>\d\.\d+)\s+"
    r"(?P<tp>\d+)/(?P<fp>\d+)/(?P<fn>\d+)\s*(?:<- best F1)?\s*$"
)
SWEEP_HDR_RE = re.compile(r"^\[(?P<model>\S+)\] threshold sweep")


# --------------------------------------------------------------------------- #
# YOLO: the committed driver reports already carry a full sweep per leg
# --------------------------------------------------------------------------- #
def parse_sweeps(path):
    """``{model: {threshold: {p, r, f1, tp, fp, fn}}}`` from one driver report.

    ``yolo_geometry_51.parse_report`` keeps only each sweep's best row, because the
    pre-registered headline is conf 0.25 and it may not select on the sweep. Parity
    needs the whole curve, so this reads every row.
    """
    out, current = {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = SWEEP_HDR_RE.match(line)
            if m:
                current = m.group("model")
                continue
            if current is None:
                continue
            m = SWEEP_ROW_RE.match(line.rstrip("\n"))
            if m:
                d = m.groupdict()
                out.setdefault(current, {})[float(d["thr"])] = {
                    "p": float(d["p"]), "r": float(d["r"]), "f1": float(d["f1"]),
                    "tp": int(d["tp"]), "fp": int(d["fp"]), "fn": int(d["fn"]),
                }
            elif line.strip() and not line.startswith(" "):
                current = None   # a new section ends the sweep block
    return out


def collect_yolo():
    """``{model: {split: {threshold: metrics}}}`` over every committed report."""
    cells = {}
    for split in ALL_SPLITS_AS_RUN:
        for kind in ("tiles", "pano"):
            path = os.path.join(IN_DIR, f"{split}_{kind}.txt")
            if not os.path.exists(path):
                continue
            for model, sweep in parse_sweeps(path).items():
                cells.setdefault(model, {})[split] = sweep
    return cells


# --------------------------------------------------------------------------- #
# RampNet: re-scored from the committed low-floor cache, no GPU
# --------------------------------------------------------------------------- #
def rampnet_grid(floor=0.05, step=0.05, top=0.95):
    n = int(round((top - floor) / step))
    return [round(floor + i * step, 10) for i in range(n + 1)]


def rampnet_sweep(split, grid, radius_sq=None):
    """``{threshold: metrics}`` for RampNet on one split, or None if uncached.

    Re-scores ``analysis_out/op_cache/<split>.json`` at each threshold with the same
    ``score_pano``/``aggregate`` path ``scoreboard.py`` uses, so the numbers are the
    published ones by construction rather than by coincidence.
    """
    path = os.path.join(OP_CACHE, f"{split}.json")
    if not os.path.exists(path):
        return None
    if radius_sq is None:
        radius_sq = radius_sq_for()
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    panos = [(p["preds"], p["gt"]) for p in payload["panos"]]

    from rampnet.detection_eval import GroundTruth
    gts = [GroundTruth([tuple(q) for q in g["gt_points"]],
                       [tuple(q) for q in g["ignore_points"]],
                       bool(g["fn_confirmed"])) for _, g in panos]

    out = {}
    for thr in grid:
        rep = aggregate([
            score_pano([tuple(q) for q in preds if q[2] >= thr], gt, radius_sq=radius_sq)
            for (preds, _), gt in zip(panos, gts)])
        out[thr] = {"p": rep.precision, "r": rep.recall, "f1": rep.f1,
                    "tp": rep.tp, "fp": rep.fp, "fn": rep.fn}
    return out


def collect_rampnet(grid):
    per_split = {}
    for split in ALL_SPLITS_AS_RUN:
        sweep = rampnet_sweep(split, grid)
        if sweep is not None:
            per_split[split] = sweep
    return per_split


# --------------------------------------------------------------------------- #
# selection + reporting
# --------------------------------------------------------------------------- #
def select_threshold(per_split, dev_split):
    """The uniform threshold maximising F1 on ``dev_split``. Ties -> higher threshold,
    which is the precision-favouring side and the one a deployment would pick."""
    sweep = per_split.get(dev_split)
    if not sweep:
        return None
    best = max(sweep.items(), key=lambda kv: (kv[1]["f1"], kv[0]))
    return best[0]


def macro_at(per_split, thr, pool=POOLED_SPLITS):
    """Macro-mean P/R/F1 over ``pool`` at ``thr``, or None if any split is missing.

    Macro, not micro, to match ``yolo_geometry_51.pooled`` and the scoreboard: each
    city weighted equally so the largest split cannot dominate.
    """
    got = []
    for s in pool:
        sweep = per_split.get(s)
        if not sweep or thr not in sweep:
            return None
        got.append(sweep[thr])
    n = len(got)
    return {"threshold": thr,
            "p": sum(c["p"] for c in got) / n,
            "r": sum(c["r"] for c in got) / n,
            "f1": sum(c["f1"] for c in got) / n,
            "n_splits": n}


def control_per_split(rampnet_per_split):
    """Per-split op_cache-vs-bundle agreement for RampNet at the deployed threshold.

    The pooled macro control clears its tolerance partly by cancellation, so it cannot
    see a split-level divergence. This can. Each split carries its own tolerance and the
    reason it gets that tolerance: ``CONTROL_EXACT_SPLITS`` are the ones where the two
    paths are the same computation and must agree, while the rest have a measured,
    written-down reason to differ (``docs/operating_point.md``).
    """
    out = {}
    for split, published in PUBLISHED_RAMPNET_PER_SPLIT_F1.items():
        sweep = rampnet_per_split.get(split)
        if not sweep or DEPLOYED_THRESHOLD not in sweep:
            continue
        got = sweep[DEPLOYED_THRESHOLD]["f1"]
        exact = split in CONTROL_EXACT_SPLITS
        tol = CONTROL_TOL_EXACT if exact else CONTROL_TOL_DIVERGENT
        out[split] = {
            "f1_op_cache": got,
            "f1_published_bundle": published,
            "delta": got - published,
            "tolerance": tol,
            "paths_agree_by_construction": exact,
            "agrees": abs(got - published) <= tol,
        }
    return out


def build(dev_split=DEFAULT_DEV):
    grid = rampnet_grid()
    models = {RAMPNET: collect_rampnet(grid)}
    models.update(collect_yolo())

    # Self-check before anything is reported off this path.
    at_deployed = macro_at(models[RAMPNET], DEPLOYED_THRESHOLD)
    delta = None if at_deployed is None else at_deployed["f1"] - PUBLISHED_RAMPNET_F1
    per_split_control = control_per_split(models[RAMPNET])
    control = {
        "threshold": DEPLOYED_THRESHOLD,
        "f1_op_cache": at_deployed["f1"] if at_deployed else None,
        "f1_published_bundle": PUBLISHED_RAMPNET_F1,
        "delta": delta,
        "tolerance": CONTROL_TOL,
        "agrees": bool(delta is not None and abs(delta) <= CONTROL_TOL),
        "per_split": per_split_control,
        "per_split_agrees": all(c["agrees"] for c in per_split_control.values()),
    }

    rows = {}
    for model, per_split in models.items():
        thr = select_threshold(per_split, dev_split)
        if thr is None:
            continue
        pooled = macro_at(per_split, thr)
        if pooled is None:
            continue
        floor = min(per_split[dev_split])
        rows[model] = {
            "selected_threshold": thr,
            "at_floor": thr <= floor,      # reported value is a lower bound
            "pooled": pooled,
            "published_point": macro_at(
                per_split, DEPLOYED_THRESHOLD if model == RAMPNET else 0.25),
            "per_split": {s: per_split[s][thr] for s in ALL_SPLITS_AS_RUN
                          if s in per_split and thr in per_split[s]},
        }
    return {"dev_split": dev_split, "control": control, "models": rows,
            "pool": list(POOLED_SPLITS), "grid_rampnet": grid}


def sensitivity():
    """Selected threshold + pooled F1 per model, for each candidate dev split."""
    return {dev: {m: {"thr": r["selected_threshold"], "f1": r["pooled"]["f1"]}
                  for m, r in build(dev)["models"].items()}
            for dev in NON_POOLED}


def artifact(dev_split=DEFAULT_DEV):
    """The committed payload: the run, plus the sensitivity table that shows the dev
    split does not carry it.

    Sensitivity is always included rather than being a flag, so ``--check`` compares
    the same object every time -- an artifact whose content depends on which switches
    the last person typed cannot be checked at all.
    """
    result = build(dev_split)
    result["sensitivity"] = sensitivity()
    return result


# --------------------------------------------------------------------------- #
def _render(result):
    lines = []
    c = result["control"]
    lines.append(f"Dev split (selection only): {result['dev_split']}")
    lines.append(f"Reported over {len(result['pool'])} pooled US splits: "
                 f"{', '.join(result['pool'])}")
    status = "OK" if c["agrees"] else "MISMATCH"
    lines.append(f"Control -- RampNet @{c['threshold']:.2f}: op_cache {c['f1_op_cache']:.3f} "
                 f"vs published bundle {c['f1_published_bundle']:.3f} "
                 f"(delta {c['delta']:+.3f}, tol {c['tolerance']:.3f})  [{status}]")
    if c.get("per_split"):
        lines.append("  per split -- the macro above clears its tolerance partly by "
                     "cancellation, so it is also asserted here:")
        lines.append(f"    {'split':<20}{'op_cache':>10}{'bundle':>9}{'delta':>9}{'tol':>8}   why")
        for split, r in c["per_split"].items():
            why = ("same computation, must agree" if r["paths_agree_by_construction"]
                   else "documented divergence (docs/operating_point.md)")
            flag = "" if r["agrees"] else "   <- MISMATCH"
            lines.append(f"    {split:<20}{r['f1_op_cache']:>10.4f}"
                         f"{r['f1_published_bundle']:>9.3f}{r['delta']:>+9.4f}"
                         f"{r['tolerance']:>8.3f}   {why}{flag}")
    lines.append("")
    hdr = (f"{'model':<18}{'sel thr':>9}{'P':>8}{'R':>8}{'F1':>8}"
           f"{'dF1 vs RampNet':>16}{'published F1':>14}{'gain':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    base = result["models"].get(RAMPNET, {}).get("pooled", {}).get("f1")
    order = sorted(result["models"], key=lambda m: -result["models"][m]["pooled"]["f1"])
    for m in order:
        r = result["models"][m]
        p = r["pooled"]
        pub = r["published_point"]["f1"] if r["published_point"] else float("nan")
        d = "" if base is None else f"{p['f1'] - base:>+16.3f}"
        mark = " *" if r["at_floor"] else ""
        lines.append(f"{m:<18}{r['selected_threshold']:>9.2f}{p['p']:>8.3f}{p['r']:>8.3f}"
                     f"{p['f1']:>8.3f}{d}{pub:>14.3f}{p['f1'] - pub:>+8.3f}{mark}")
    if any(r["at_floor"] for r in result["models"].values()):
        lines.append("")
        lines.append("* selected threshold is the cache floor -- the true optimum may be "
                     "lower and unmeasured, so this F1 is a LOWER BOUND.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev-split", default=DEFAULT_DEV, choices=sorted(NON_POOLED),
                    help="split the uniform threshold is selected on (never reported over)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="re-run selection with every candidate dev split")
    ap.add_argument("--json", default=OUT_JSON, help="artifact path")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the artifact or the control has drifted")
    args = ap.parse_args()

    result = artifact(args.dev_split)
    print(_render(result))

    if args.sensitivity:
        print("\nSensitivity -- the dev split does not carry the result:\n")
        sens = result["sensitivity"]
        models = sorted({m for v in sens.values() for m in v})
        print(f"{'dev split':<22}" + "".join(f"{m:>22}" for m in models))
        for dev in NON_POOLED:
            row = "".join(f"{sens[dev][m]['thr']:>10.2f} -> {sens[dev][m]['f1']:<9.3f}"
                          if m in sens[dev] else f"{'-':>22}" for m in models)
            print(f"{dev:<22}{row}")


    if not result["control"]["agrees"]:
        print("\nFATAL: RampNet no longer reproduces its published 0.827 at 0.55.",
              file=sys.stderr)
        return 2

    if not result["control"]["per_split_agrees"]:
        bad = ", ".join(k for k, r in result["control"]["per_split"].items()
                        if not r["agrees"])
        print(f"\nFATAL: the per-split control failed on {bad} -- op_cache and"
              " the published bundle have diverged on a split where they should not.",
              file=sys.stderr)
        return 2

    payload = rnd(result)
    if args.check:
        if not os.path.exists(args.json):
            print(f"\n--check: {args.json} does not exist", file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            if json.load(fh) != payload:
                print(f"\n--check: {args.json} is stale", file=sys.stderr)
                return 1
        print(f"\n--check: {os.path.relpath(args.json, REPO)} is current")
        return 0

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(args.json, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
