"""Is a gated cascade live? Does RampNet already *see* the ramps a challenger recovers? (#126)

The complementarity gate (``complementarity.py``) says a Vistas-supervised segmenter at
input parity finds 54 of the 72 richmond ramps RampNet misses, ~44 of them after the
chance null. A naive union of the two still loses — those 54 arrive with 442 false
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

**``class_of`` is imported for comparability, but the column the read turns on here is
``peak_in_radius``, not the class.** #46 Phase 1 applied those cutoffs to *silent*
misses — defined as no floor peak within the radius — so there ``tail`` (act >= 0.05)
could only mean an outside mode reaching in. This population is every RampNet miss, not
just the silent ones, so ``act >= 0.05`` here has two very different causes and the
class alone cannot separate them:

* **a floor peak inside the radius** — the model localized the ramp and the detection
  was lost *downstream*, either to the shipped threshold or to the greedy matcher giving
  that peak to an adjacent GT. Recoverable, and recoverable **without a second model**.
* **no floor peak inside the radius** — a threshold prior has nothing *of this ramp's*
  to promote, because promotion operates on peaks. ``act`` there is usually **not**
  unpeaked mass: on the committed richmond run the nearest floor peak sits a median
  35 px away (1-2 R) and the in-window maximum sits on the window edge (median
  ``argmax_off_px`` 22.4 of 22.5), i.e. a neighbouring mode's shoulder reaching in —
  #46 Phase 1's ``tail``, and that peak is mostly one the matcher already gave to an
  adjacent GT (#130). ``no_peak_profile`` reports this per cell so the row is read as
  what it is. Genuinely peakless mass — no floor peak within 2 R — is the minority
  (4 of 15 on richmond), and one of those 4 is a peak the op_cache dropped beside the
  seam (pre-``f4c71c8``), not a peak the model never made.

``peak_in_radius`` is therefore reported per cell and is what the read below turns on.

**Sub-threshold signal is necessary, not sufficient.** A positive here says the cascade
is not ruled out and is worth costing; it does not demonstrate one works. Any realisable
gain is bounded above by the ~44 attributable ramps, not the raw 54.

**The floor peaks come from ``analysis_out/op_cache/<split>.json``, NOT from the
bundle records.** The bundle's committed detections are the published operating point
— on richmond every one of them scores >= 0.5519 — so asking "is there a peak near this
missed ramp?" of *those* answers a different question and makes every miss look
peakless. The op_cache holds ``peak_local_max`` output down to the 0.05 floor, which is
what "did the model say anything here, below the threshold we ship?" actually needs.
The greedy match that DEFINED the miss still uses the bundle records, because that is
what produced the published 238/9/72 — the two are deliberately different inputs to
two different questions.

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

from rampnet.detection_eval import (                                   # noqa: E402
    radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y)
from rampnet.metrics import greedy_match                               # noqa: E402
from compare import load_bundle, DetectionCache, cache_key             # noqa: E402
from detectors import build_detector                                   # noqa: E402
from complementarity import (                                          # noqa: E402
    CELLS, cell_of, compare_args, floor_gap_warning, load_floor_peaks, model_spec,
    partition_cells)
from silent_activation import (                                        # noqa: E402
    NULL_SEED, NULL_TRIALS, class_of, nearest_peak, null_percentile, seam_of,
    site_profile)
from farfield_forensics import quartiles                               # noqa: E402

#: Cells where RampNet did NOT find the ramp -- the only ones a null is meaningful for.
MISS_CELLS = ("challenger_only", "neither")


def site_rng(pid, x, y):
    """One RNG per site, seeded from ``NULL_SEED`` and the site's identity.

    A single stream consumed in pano order made a site's ``null_pct`` depend on which
    sites preceded it: the miss set differs between the shipped-point and the 0.30
    artifacts (19 cell transitions), so the same site drew different azimuths in
    each — 43 of the 53 sites carrying a null in both files differed, by up to 0.075,
    with ``act`` and ``nearest_peak_px`` identical on all of them. Seeding per site
    makes a site's null a function of its heatmap and nothing else, so a per-site
    comparison across runs measures the heatmap rather than the neighbours.
    """
    return random.Random(f"{NULL_SEED}:{pid}:{x!r}:{y!r}")


def panos_without_floor(panos, floor_peaks):
    """Panos the op_cache does not list, in the order given.

    On the probe path a missing pano reads ``peak_in_radius: false`` and
    ``nearest_peak_px: null`` for every site on it while the header still says the
    floor came from the op_cache — the same silent shift ``partition_cells`` counts
    under ``--rampnet-op-threshold``, on the path that did not warn.
    """
    return [pid for pid in panos if pid not in floor_peaks]


def nearest_peak_index(preds, x, y):
    """Index into ``preds`` of the closest floor peak (``nearest_peak``'s geometry)."""
    best, best_i = float("inf"), None
    for i, p in enumerate(preds):
        dx = abs(p[0] - x) * PANO_SCALE_X
        dx = min(dx, PANO_SCALE_X - dx)
        d = (dx * dx + ((p[1] - y) * PANO_SCALE_Y) ** 2) ** 0.5
        if d < best:
            best, best_i = d, i
    return best_i


def claimed_by_adjacent(pano_sites, preds, threshold, radius_sq):
    """For each site, is its nearest floor peak one the matcher gave to a DIFFERENT GT?

    Re-runs the greedy match that defines RampNet's hits at ``threshold`` (highest
    score first, wrapped, as ``matched_gt``) over all GT on the pano, then asks of
    each site's nearest floor peak whether it was assigned to some other GT. That is
    the #130 mechanism — a neighbouring ramp's detection sitting 1-2 R from this one
    — and it is what most of the "no floor peak in radius" rows turn out to be.
    Returns one bool per site; ``False`` when there is no peak at all.
    """
    gt_points = [(s["x"], s["y"]) for s in pano_sites]
    kept = [i for i, p in enumerate(preds) if p[2] >= threshold]
    order = sorted(kept, key=lambda i: preds[i][2], reverse=True)
    assignments = greedy_match([(preds[i][0], preds[i][1]) for i in order], gt_points,
                               radius_sq, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    claimed = {order[k]: gi for k, (gi, _) in enumerate(assignments) if gi >= 0}
    out = []
    for si, s in enumerate(pano_sites):
        ni = nearest_peak_index(preds, s["x"], s["y"])
        out.append(ni is not None and ni in claimed and claimed[ni] != si)
    return out


def no_peak_profile(rows, cell, radius_px):
    """Where the nearest floor peak actually is, for the rows of ``cell`` that have
    none inside the radius.

    The cell summary's medians are over the whole cell; the table row in
    ``docs/model_comparison.md`` is about this subset, and what it says the subset
    *is* turns on these columns: a nearest peak 1-2 R away with the in-window
    maximum on the window edge is a neighbour's shoulder (``tail``), not mass the
    extractor overlooked. ``argmax_on_edge`` counts rows whose maximum is within
    0.5 px of the radius; ``claimed`` counts rows whose nearest peak the matcher gave
    to another GT (only present on rows that carry ``nearest_peak_claimed``, i.e.
    runs since this key was added).
    """
    sel = [r for r in rows if r["cell"] == cell and not r["peak_in_radius"]]
    if not sel:
        return {"cell": cell, "n": 0}
    near = [r["nearest_peak_px"] for r in sel if r["nearest_peak_px"] is not None]
    out = {
        "cell": cell,
        "n": len(sel),
        "act_median": round(quartiles([r["act"] for r in sel])[1], 4),
        "argmax_off_px_median": round(quartiles([r["argmax_off_px"] for r in sel])[1], 1),
        "argmax_on_edge": sum(1 for r in sel if r["argmax_off_px"] >= radius_px - 0.5),
        "nearest_peak_px_median": round(quartiles(near)[1], 1) if near else None,
        "peak_within_2r": sum(1 for d in near if d <= 2 * radius_px),
        "peak_beyond_2r": len(sel) - sum(1 for d in near if d <= 2 * radius_px),
        "classes": {c: sum(1 for r in sel if r["class"] == c)
                    for c in ("absent", "faint_local", "tail")},
        "seam": sum(1 for r in sel if r["seam"]),
    }
    if all("nearest_peak_claimed" in r for r in sel):
        out["claimed"] = sum(1 for r in sel if r["nearest_peak_claimed"])
    return out


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
    inr = [r for r in sel if r["peak_in_radius"]]
    out["peak_in_radius"] = len(inr)
    out["peak_in_radius_share"] = round(len(inr) / len(sel), 3)
    if inr:
        out["peak_in_radius_score_median"] = round(
            quartiles([r["nearest_peak_score"] for r in inr])[1], 4)
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
    p.add_argument("--rampnet-op-threshold", type=float, default=None,
                   help="Define RampNet's hits from op_cache floor peaks at this "
                        "threshold instead of the bundle's shipped detections "
                        "(>=0.5519 on richmond). This document recommends 0.30, and "
                        "the cells move: 19 of the shipped point's misses are ramps "
                        "RampNet already has, 16 of them from challenger_only. "
                        "Default: the bundle, as published.")
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

    # Floor peaks (>= 0.05). Used for the sub-threshold probe always, and to
    # DEFINE rampnet's hits when --rampnet-op-threshold is given.
    floor_peaks, floor_src, have_op_cache = {}, "op_cache", True
    try:
        floor_peaks = load_floor_peaks(args.split)
    except (OSError, ValueError, KeyError):
        have_op_cache = False
        floor_src = ("MISSING (fell back to bundle records -- distances are to the "
                     "shipped operating point, not the 0.05 floor)")
        if args.rampnet_op_threshold is not None:
            sys.exit("--rampnet-op-threshold needs analysis_out/op_cache/"
                     f"{args.split}.json, which could not be read.")

    # ---- partition every GT ramp into a complementarity cell ------------------
    # The same loop complementarity.py's table comes from, so the two cannot drift.
    part = partition_cells(
        records, verdicts,
        lambda pid: cache.get(cache_key(label, sig, args.split, pid)),
        radius_sq, args.rampnet_op_threshold,
        floor_peaks if args.rampnet_op_threshold is not None else None)
    sites, missing = part.sites, part.missing
    if missing:
        print(f"WARNING: {missing} panos had no cached {label} detections and were "
              f"skipped. Pass the --vistas-input-size the run used.", flush=True)
    warning = floor_gap_warning(part.no_floor, args.split)
    if warning:
        print(warning, flush=True)
    if not sites:
        sys.exit("No sites -- is the challenger cached for this split/input size?")

    by_pano = {}
    for s in sites:
        by_pano.setdefault(s["pano"], []).append(s)
    panos = sorted(by_pano)
    if args.limit:
        panos = panos[:args.limit]
    # The probe path reads the op_cache for every pano regardless of the threshold
    # flag, so a pano it does not list is a gap here too -- and until now this path
    # said nothing about it.
    no_floor_probe = panos_without_floor(panos, floor_peaks) if have_op_cache else []
    if no_floor_probe:
        print(f"WARNING: {len(no_floor_probe)} pano(s) are absent from analysis_out/"
              f"op_cache/{args.split}.json, so every site on them reads 'no floor peak' "
              f"(nearest_peak_px null) whatever the heatmap says. Regenerate the "
              f"op_cache for this split before reading the peak columns.", flush=True)

    counts = {c: sum(1 for s in sites if s["cell"] == c) for c in CELLS}
    rn = ("rampnet" if args.rampnet_op_threshold is None
          else f"rampnet@{args.rampnet_op_threshold:g}")
    print(f"=== Cascade gate: {rn} heatmap at {label}'s recoveries "
          f"({args.split}, {len(sites)} GT ramps in {len(by_pano)} panos) ===")
    print("    cells: " + "  ".join(f"{c}={counts[c]}" for c in CELLS), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    print(f"    device={device} model=projectsidewalk/rampnet-model "
          f"(single-pass fp32, as op_cache)", flush=True)
    print(f"    floor peaks (>=0.05) from: {floor_src}", flush=True)
    print(f"    rampnet hits defined by: "
          + ("bundle records (shipped point)" if args.rampnet_op_threshold is None
             else f"op_cache >= {args.rampnet_op_threshold:g}"), flush=True)

    # The threshold that defines RampNet's hits, for the "did the matcher give this
    # site's nearest peak to a neighbour" column. At the shipped point that is the
    # bundle's floor, 0.5519 on richmond, which is the lowest committed score.
    claim_threshold = (args.rampnet_op_threshold if args.rampnet_op_threshold is not None
                       else min((d["confidence"] for r in records.values()
                                 for d in r["detections"]), default=0.0))
    r_px = radius_sq ** 0.5
    rows, skipped = [], 0
    for i, pid in enumerate(panos, 1):
        path = pano_path(args.split, pid, args.panos_root)
        if not os.path.exists(path):
            skipped += len(by_pano[pid])
            continue
        heat = ts.heatmap_for(model, device, path, use_fp16=False)
        # One source per run, not per pano. `floor_peaks.get(pid) or <bundle>` would
        # substitute the shipped detections for any pano the op_cache lists with zero
        # floor peaks, while the header still says the floor came from op_cache -- and
        # a pano with no peak at the 0.05 floor genuinely has nothing to promote, which
        # is the answer, not a gap to fill.
        preds = (floor_peaks.get(pid, []) if have_op_cache else
                 [(d["x_normalized"], d["y_normalized"], d["confidence"])
                  for d in records[pid]["detections"]])
        claimed = claimed_by_adjacent(by_pano[pid], preds, claim_threshold, radius_sq)
        for s, is_claimed in zip(by_pano[pid], claimed):
            act, off_px, center = site_profile(heat, s["x"], s["y"], radius_sq)
            npx, nscore = nearest_peak(preds, s["x"], s["y"])
            row = {**s, "act": round(act, 6), "center": round(center, 6),
                   "argmax_off_px": round(off_px, 1),
                   "nearest_peak_px": None if npx == float("inf") else round(npx, 1),
                   "nearest_peak_score": nscore,
                   # A floor peak inside the match radius on a MISSED ramp means the
                   # model did localize it and the detection was lost downstream --
                   # either to the shipped threshold, or to the greedy matcher giving
                   # the peak to an adjacent GT. That is recoverable without a second
                   # model; a peak outside the radius is not.
                   "peak_in_radius": bool(npx < r_px),
                   # ...and whether that nearest peak is one the matcher already handed
                   # to a different GT on this pano (#130), which is what most of the
                   # "no peak in radius" rows are.
                   "nearest_peak_claimed": bool(is_claimed),
                   "class": class_of(act), "seam": seam_of(s["x"], radius_sq),
                   "null_pct": None, "null_med": None, "null_p95": None}
            # The null is only meaningful where rampnet did NOT find the ramp; the
            # hit cells are high by construction and are here as a positive control.
            # Seeded per site (site_rng), so the draw does not depend on which sites
            # came before it -- the committed artifacts predate this and were written
            # from one stream, so a regeneration moves individual null_pct values by
            # up to 0.075 without any change in the heatmap.
            if s["cell"] in MISS_CELLS:
                a, pct, med, p95 = null_percentile(heat, s["x"], s["y"],
                                                   site_rng(pid, s["x"], s["y"]),
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
           f"{'floor peak in R':>16s} {'its score':>10s} {'null pct':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        if not s["n"]:
            continue
        miss = s["cell"] in MISS_CELLS
        print(f"{s['cell']:17s} {s['n']:4d} {s['act_median']:8.4f} "
              f"{s['center_median']:8.4f} {s['argmax_off_px_median']:10.1f}p "
              f"{s['peak_in_radius']:9d} ({s['peak_in_radius_share']:3.0%}) "
              + (f"{s.get('peak_in_radius_score_median', float('nan')):10.3f}"
                 if s.get("peak_in_radius") else f"{'—':>10s}")
              + (f"{s['null_pct_median']:9.3f}" if miss else f"{'—':>9s}"))
    print()
    print("  The hit cells are the positive control: a matched detection is inside the")
    print("  radius by definition, so their act/centre ~0.85+ and 100% peak-in-R are what")
    print("  a working probe MUST show, not a finding.")
    print()
    print("  For the two MISS cells, 'floor peak in R' is the whole question:")
    print("   - peak inside R  -> the model DID localize the ramp and the detection was")
    print("     lost downstream, to the shipped threshold or to the greedy matcher handing")
    print("     the peak to an adjacent GT. Recoverable WITHOUT a second model.")
    print("   - peak outside R -> nothing of THIS ramp's to promote. 'act' there is usually")
    print("     a neighbouring peak's shoulder reaching into the window (nearest peak 1-2 R")
    print("     away, argmax on the window edge), not mass the extractor overlooked -- the")
    print("     profile below says which, per cell.")
    print()
    print(f"  no floor peak in R, by where the nearest peak is (R = {r_px:.1f} px):")
    for cell in MISS_CELLS:
        prof = no_peak_profile(rows, cell, r_px)
        if not prof["n"]:
            continue
        print(f"    {cell:16s} n={prof['n']:<3d} nearest peak median "
              f"{prof['nearest_peak_px_median']} px; within 2R {prof['peak_within_2r']}, "
              f"beyond {prof['peak_beyond_2r']}; argmax on edge {prof['argmax_on_edge']}; "
              f"nearest peak claimed by another GT {prof.get('claimed', '?')}; "
              f"classes {prof['classes']}; seam {prof['seam']}")

    if args.json_out:
        payload = {"split": args.split, "challenger": label,
                   "rampnet_op_threshold": args.rampnet_op_threshold,
                   "vistas_input_size": args.vistas_input_size,
                   "radius": args.radius, "null_trials": NULL_TRIALS,
                   "null_seed": NULL_SEED, "null_rng": "per-site",
                   "n_sites": len(rows),
                   "n_panos": len(panos), "skipped_sites": skipped,
                   "panos_without_floor_peaks": len(no_floor_probe),
                   "cells": summaries, "sites": rows}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        # newline="" so a Windows re-run does not emit CRLF and break byte-comparison.
        with open(args.json_out, "w", encoding="utf-8", newline="") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
