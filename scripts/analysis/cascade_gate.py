"""Is a gated cascade live? Does RampNet already *see* the ramps a challenger recovers? (#126)

The complementarity gate (``complementarity.py``) says a Vistas-supervised segmenter at
input parity finds 54 of the 72 richmond ramps RampNet misses, ~44 of them after the
chance null. A naive union is nonetheless dead — those 54 arrive with 442 false
positives, ~8.2 FP per recovered ramp, and the union scores F1 0.555 against RampNet's
0.855.

What that leaves is a **gated cascade**: use the challenger's candidates as a spatial
prior and locally relax RampNet's threshold, keeping RampNet's precision everywhere
else. That has one precondition, and it is measurable without building anything —
**RampNet must already produce sub-threshold response at those ramps.** If the heatmap
is flat there, the miss is genuine absence and no prior can raise what is not there.

So: partition every GT ramp into the four complementarity cells, and read RampNet's
heatmap at each.

    cell                rampnet  challenger   role
    both                hit      hit          positive control
    rampnet_only        hit      miss         positive control
    challenger_only     MISS     hit          the recoverable set -- the question
    neither             MISS     miss         hard core

**The instrument is #46 Phase 1's, imported rather than reimplemented** —
``site_profile``, ``null_percentile``, ``nearest_peak``, ``class_of`` and its two
cutoffs (``ABSENT_MAX`` 0.01, ``PEAK_FLOOR`` 0.05) all come from
``silent_activation.py``. That is deliberate: it makes these numbers directly
comparable to that phase's 8% absent / 62% adjacent-tail / 30% faint decomposition,
and it means a fix to the probe fixes both analyses.

**Three outcomes, and the middle one is not the interesting one.** ``faint_local`` with
the in-window argmax *at* the site is signal a prior could raise. ``absent`` is nothing
to raise. ``tail`` — the in-window maximum at or above the peak floor — is an adjacent
mode reaching in, and for a *missed* ramp it means relaxing a threshold there would fire
on the neighbour, not on the ramp. ``tail`` therefore argues for sigma/NMS work (#46's
adjacent-pair merge), **not** for a threshold prior, even though it looks like "signal
is present" if you only read ``act``.

**Sub-threshold signal is necessary, not sufficient.** A positive here says the cascade
is not ruled out and is worth costing; it does not demonstrate one works. Any realisable
gain is bounded above by the ~44 attributable ramps, not the raw 54.

Inputs: the native-resolution panoramas at ``benchmark/<split>/panos/`` (git-ignored,
published as ``projectsidewalk/rampnet-benchmark``) — ``--panos-root`` points at
whichever checkout holds them, since a worktree will not. The challenger's detections
must already be in ``--cache-dir``; **pass the same ``--vistas-input-size`` the run
used, it is part of the cache key.** A GPU: ~124 panos, one forward each.

    python scripts/analysis/cascade_gate.py --panos-root /path/to/RampNet \\
        --model vistas:curb-cut --vistas-input-size 1024 1024 \\
        --json-out analysis_out/cascade_gate.json
"""
import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import build_ground_truth, radius_sq_for   # noqa: E402
from compare import load_bundle, DetectionCache, cache_key             # noqa: E402
from detectors import build_detector                                   # noqa: E402
from complementarity import matched_gt, model_spec, compare_args       # noqa: E402
from silent_activation import (                                        # noqa: E402
    NULL_SEED, NULL_TRIALS, class_of, nearest_peak, null_percentile, seam_of,
    site_profile)
from farfield_forensics import quartiles                               # noqa: E402

CELLS = ("both", "rampnet_only", "challenger_only", "neither")
#: Cells where RampNet did NOT find the ramp -- the only ones a null is meaningful for.
MISS_CELLS = ("challenger_only", "neither")


def cell_of(rampnet_hit, challenger_hit):
    if rampnet_hit:
        return "both" if challenger_hit else "rampnet_only"
    return "challenger_only" if challenger_hit else "neither"


def summarize(rows, cell):
    """Per-cell summary. ``None`` for null stats on the hit cells, which have none."""
    sel = [r for r in rows if r["cell"] == cell]
    if not sel:
        return {"cell": cell, "n": 0}
    acts = [r["act"] for r in sel]
    classes = {c: sum(1 for r in sel if r["class"] == c)
               for c in ("absent", "faint_local", "tail")}
    out = {
        "cell": cell,
        "n": len(sel),
        "act_median": round(quartiles(acts)[1], 4),
        "center_median": round(quartiles([r["center"] for r in sel])[1], 4),
        "argmax_off_px_median": round(quartiles([r["argmax_off_px"] for r in sel])[1], 1),
        "nearest_peak_px_median": round(
            quartiles([r["nearest_peak_px"] for r in sel
                       if r["nearest_peak_px"] is not None])[1], 1)
        if any(r["nearest_peak_px"] is not None for r in sel) else None,
        "classes": classes,
        "class_share": {c: round(v / len(sel), 3) for c, v in classes.items()},
        "seam": sum(1 for r in sel if r["seam"]),
    }
    nulls = [r["null_pct"] for r in sel if r["null_pct"] is not None]
    if nulls:
        out["null_pct_median"] = round(quartiles(nulls)[1], 3)
        out["above_null_p95"] = sum(1 for r in sel
                                    if r["null_pct"] is not None and r["act"] > r["null_p95"])
        out["null_med_median"] = round(
            quartiles([r["null_med"] for r in sel if r["null_med"] is not None])[1], 4)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="richmond")
    p.add_argument("--model", default="vistas:curb-cut",
                   help="Challenger model spec, as compare.py --models takes it.")
    p.add_argument("--panos-root", default=REPO,
                   help="Checkout holding benchmark/<split>/panos/ (a worktree will not).")
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--radius", type=float, default=0.022)
    p.add_argument("--tiling", choices=["perspective", "none"], default="perspective")
    p.add_argument("--vistas-input-size", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="Must match the run being analysed -- it is part of the cache key.")
    p.add_argument("--vistas-revision", default=None)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: first N panos.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if args.json_out and args.limit:
        p.error("--limit truncates the run; refusing to write it to --json-out")

    import torch
    import threshold_sweep as ts
    from miss_gallery import pano_path

    bundle = os.path.join(REPO, "benchmark", args.split)
    records, verdicts, _ = load_bundle(bundle)
    if verdicts is None:
        sys.exit(f"{bundle}: no verdicts.json -- this needs a reviewed split.")
    provider, model_id = model_spec(args.model)
    label, detector = build_detector(provider, model_id, records, compare_args(args))
    sig = detector.signature()
    cache = DetectionCache(args.cache_dir)
    radius_sq = radius_sq_for(args.radius)

    # ---- partition every GT ramp into a complementarity cell ------------------
    sites, missing = [], 0
    for pid, entry in verdicts.items():
        gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                entry["missed"], entry["no_missed"])
        if not gt.fn_confirmed:
            continue
        cp = cache.get(cache_key(label, sig, args.split, pid))
        if cp is None:
            missing += 1
            continue
        rp = [(d["x_normalized"], d["y_normalized"], d["confidence"])
              for d in records[pid]["detections"]]
        mr = matched_gt(rp, gt.gt_points, radius_sq)
        mc = matched_gt(cp, gt.gt_points, radius_sq)
        for i, (gx, gy) in enumerate(gt.gt_points):
            sites.append({"pano": pid, "x": gx, "y": gy,
                          "cell": cell_of(i in mr, i in mc)})
    if missing:
        print(f"WARNING: {missing} panos had no cached {label} detections and were "
              f"skipped. Pass the --vistas-input-size the run used.", flush=True)
    if not sites:
        sys.exit("No sites -- is the challenger cached for this split/input size?")

    by_pano = {}
    for s in sites:
        by_pano.setdefault(s["pano"], []).append(s)
    panos = sorted(by_pano)
    if args.limit:
        panos = panos[:args.limit]

    counts = {c: sum(1 for s in sites if s["cell"] == c) for c in CELLS}
    print(f"=== Cascade gate: rampnet heatmap at {label}'s recoveries "
          f"({args.split}, {len(sites)} GT ramps in {len(by_pano)} panos) ===")
    print("    cells: " + "  ".join(f"{c}={counts[c]}" for c in CELLS), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    print(f"    device={device} model=projectsidewalk/rampnet-model "
          f"(single-pass fp32, as op_cache)", flush=True)

    rng = random.Random(NULL_SEED)
    rows, skipped = [], 0
    for i, pid in enumerate(panos, 1):
        path = pano_path(args.split, pid, args.panos_root)
        if not os.path.exists(path):
            skipped += len(by_pano[pid])
            continue
        heat = ts.heatmap_for(model, device, path, use_fp16=False)
        preds = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                 for d in records[pid]["detections"]]
        for s in by_pano[pid]:
            act, off_px, center = site_profile(heat, s["x"], s["y"], radius_sq)
            npx, nscore = nearest_peak(preds, s["x"], s["y"])
            row = {**s, "act": round(act, 6), "center": round(center, 6),
                   "argmax_off_px": round(off_px, 1),
                   "nearest_peak_px": None if npx == float("inf") else round(npx, 1),
                   "nearest_peak_score": nscore,
                   "class": class_of(act), "seam": seam_of(s["x"], radius_sq),
                   "null_pct": None, "null_med": None, "null_p95": None}
            # The null is only meaningful where rampnet did NOT find the ramp; the
            # hit cells are high by construction and are here as a positive control.
            if s["cell"] in MISS_CELLS:
                a, pct, med, p95 = null_percentile(heat, s["x"], s["y"], rng,
                                                   radius_sq=radius_sq)
                row.update(null_pct=round(pct, 4), null_med=round(med, 6),
                           null_p95=round(p95, 6))
            rows.append(row)
        if i % 20 == 0 or i == len(panos):
            print(f"    {i}/{len(panos)} panos", flush=True)
    if skipped:
        print(f"WARNING: {skipped} sites skipped -- pano jpg not found under "
              f"--panos-root {args.panos_root}", flush=True)

    summaries = [summarize(rows, c) for c in CELLS]
    print()
    hdr = (f"{'cell':17s} {'n':>4s} {'act med':>8s} {'centre':>8s} {'argmax off':>11s} "
           f"{'absent':>7s} {'faint':>6s} {'tail':>5s} {'null pct':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        if not s["n"]:
            continue
        sh = s["class_share"]
        print(f"{s['cell']:17s} {s['n']:4d} {s['act_median']:8.4f} "
              f"{s['center_median']:8.4f} {s['argmax_off_px_median']:10.1f}p "
              f"{sh['absent']:7.0%} {sh['faint_local']:6.0%} {sh['tail']:5.0%} "
              + (f"{s['null_pct_median']:9.3f}" if "null_pct_median" in s else f"{'—':>9s}"))
    print()
    print("  faint_local = a real sub-threshold response AT the site -> a prior could raise it.")
    print("  tail        = in-window max at/above the 0.05 peak floor, i.e. an ADJACENT mode")
    print("                reaching in; relaxing a threshold there fires on the neighbour,")
    print("                not the ramp. Argues for sigma/NMS (#46), not for a prior.")
    print("  absent      = flat heatmap; nothing to raise.")

    if args.json_out:
        payload = {"split": args.split, "challenger": label,
                   "vistas_input_size": args.vistas_input_size,
                   "radius": args.radius, "null_trials": NULL_TRIALS,
                   "null_seed": NULL_SEED, "n_sites": len(rows),
                   "n_panos": len(panos), "skipped_sites": skipped,
                   "cells": summaries, "sites": rows}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        # newline="" so a Windows re-run does not emit CRLF and break byte-comparison.
        with open(args.json_out, "w", encoding="utf-8", newline="") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
