#!/usr/bin/env python3
"""Read the #51 geometry-pair eval and answer the equirect objection.

WHAT QUESTION THIS ANSWERS
    #51 reports that RampNet beats a supervised YOLO baseline by 0.252 F1 macro-meaned
    over the seven pooled US city splits.  The standing objection is that this is not
    architecture: the YOLO arms are fed 2048x4096 equirectangular panoramas, which is
    not the geometry a COCO-shaped detector expects.  The tiles arms are the control --
    same data, same schedule, fed through the perspective-view rig the VLMs get -- and
    until 2026-08-30 no tiles checkpoint had ever been scored, so the objection was
    live and unmeasured.

WHAT IT READS
    docs/data/yolo_geometry_51/<split>_{tiles,pano}.txt, the captured stdout of
    scripts/model_comparison/yolo_baseline/run_yolo_geometry_eval.sh (makelab2, A40).
    Parsing the driver's own output rather than re-deriving means this script cannot
    silently disagree with the run that produced the numbers; the run is the artifact.

THE CONTROL IS THE LOAD-BEARING PART
    y11x_pano_h200 is already published, scored 2026-08-14 against a repo predating the
    #132 seam fix.  It was re-run here under the same commit as the two new legs.  If
    the geometry comparison is sound, this leg must reproduce the committed scoreboard
    row for "YOLO11x (pano)" -- P 0.969 / R 0.416 / F1 0.575.  ``--check`` asserts
    exactly that, so a code change that moves YOLO scoring fails here loudly instead of
    being read as a geometry effect.

USAGE
    python scripts/analysis/yolo_geometry_51.py            # table + rewrite the JSON
    python scripts/analysis/yolo_geometry_51.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analysis.low_floor_sweep import HELD_OUT  # noqa: E402

# The split population this study was run over, PINNED -- deliberately NOT the live
# registry in ``analysis.low_floor_sweep``. The 2026-08-30 run scored these ten splits
# and no others, and the seven-split pool is the population every macro-mean in
# docs/yolo_geometry_51.md is taken over. A split added to the benchmark afterwards has
# no report here, so following the registry would silently turn every pooled number into
# ``None`` and take the committed artifact with it. Adding a split to this study means
# running the driver over it, not editing a tuple.
POOLED_SPLITS = ("richmond", "bend", "clovis", "morgantown", "annapolis", "paterson",
                 "gainesville")
ALL_SPLITS_AS_RUN = POOLED_SPLITS + ("budapest_district5", "sao_paulo", "manual_gold")

# Why each non-pooled split is scored but not pooled. The KEYS are pinned to the run;
# only the reason text follows the registry, so the two cannot disagree about a split
# this study never saw.
HELD_OUT_AS_RUN = {s: HELD_OUT[s] for s in ALL_SPLITS_AS_RUN if s in HELD_OUT}

IN_DIR = os.path.join(ROOT, "..", "docs", "data", "yolo_geometry_51")
OUT_JSON = os.path.join(ROOT, "..", "docs", "data", "yolo_geometry_51.json")

# The committed scoreboard row this run's control leg must reproduce. Restated here
# (not imported) on purpose: it is the PUBLISHED number as of 2026-08-14, and the point
# of the check is to catch the day the generator stops agreeing with it.
PUBLISHED_CONTROL = {"model": "YOLO11x (pano)", "p": 0.969, "r": 0.416, "f1": 0.575}
PUBLISHED_RAMPNET_F1 = 0.827

LEGS = {
    "y11x_tiles": {"geometry": "perspective tiles, imgsz 1024", "epoch": 44},
    "y11x_pano": {"geometry": "whole pano, imgsz 1280", "epoch": 38},
    "y11x_pano_h200": {"geometry": "whole pano, imgsz 1280", "epoch": 60},
}

ROW_RE = re.compile(
    r"^(?P<model>\S+)\s+"
    r"(?P<p>[\d.]+)\s+\([\d.]+-[\d.]+\)\s+"
    r"(?P<r>[\d.]+)\s+\([\d.]+-[\d.]+\)\s+"
    r"(?P<f1>[\d.]+)\s+"
    r"(?P<ap>[\d.]+|-)\s+"
    r"(?P<tp>\d+)/(?P<fp>\d+)/(?P<fn>\d+)/(?P<ign>\d+)\s*$"
)
BEST_RE = re.compile(r"^\s*(?P<thr>[\d.]+)\s+[\d.]+\s+[\d.]+\s+(?P<f1>[\d.]+)\s+\S+\s+<- best F1")


def parse_report(path):
    """Rows from the operating-point table, plus each model's tuned-on-test best-F1.

    The sweep's best row is parsed but never used for a headline: the pre-registered
    operating point is conf 0.25 (#71). It is carried so the write-up can say by how
    much a tuned threshold would have flattered each leg, which is the honest way to
    report a number nobody is allowed to select on.
    """
    rows, best, current = {}, {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = ROW_RE.match(line.rstrip("\n"))
            if m and m.group("model") not in ("model",):
                d = m.groupdict()
                rows[d["model"]] = {
                    "p": float(d["p"]), "r": float(d["r"]), "f1": float(d["f1"]),
                    "ap": None if d["ap"] == "-" else float(d["ap"]),
                    "tp": int(d["tp"]), "fp": int(d["fp"]),
                    "fn": int(d["fn"]), "ign": int(d["ign"]),
                }
                continue
            m = re.match(r"^\[(\S+)\] threshold sweep", line)
            if m:
                current = m.group(1)
                continue
            m = BEST_RE.match(line)
            if m and current:
                best[current] = {"thr": float(m.group("thr")), "f1": float(m.group("f1"))}
    return rows, best


def collect():
    cells, best = {}, {}
    for split in ALL_SPLITS_AS_RUN:
        for kind in ("tiles", "pano"):
            path = os.path.join(IN_DIR, f"{split}_{kind}.txt")
            if not os.path.exists(path):
                continue
            rows, bests = parse_report(path)
            for model, vals in rows.items():
                cells.setdefault(model, {})[split] = vals
            for model, vals in bests.items():
                best.setdefault(model, {})[split] = vals
    return cells, best


def pooled(per_split):
    """Macro-mean over POOLED_SPLITS, and count-pooled P/R alongside it.

    Macro is the published convention (each city weighted equally, so a big split
    cannot dominate); micro is carried because a macro-mean of ratios hides how many
    ramps are actually behind each city.
    """
    got = [per_split[s] for s in POOLED_SPLITS if s in per_split]
    if len(got) != len(POOLED_SPLITS):
        return None
    tp = sum(c["tp"] for c in got)
    fp = sum(c["fp"] for c in got)
    fn = sum(c["fn"] for c in got)
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_splits": len(got),
        "macro_p": sum(c["p"] for c in got) / len(got),
        "macro_r": sum(c["r"] for c in got) / len(got),
        "macro_f1": sum(c["f1"] for c in got) / len(got),
        "macro_ap": sum(c["ap"] for c in got) / len(got) if all(c["ap"] is not None for c in got) else None,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def rnd(o, n=4):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rnd(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v, n) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Verify the control leg and the committed JSON; write nothing.")
    args = ap.parse_args()

    cells, best = collect()
    if not cells:
        sys.exit(f"no eval reports found under {IN_DIR}")

    pools = {m: pooled(v) for m, v in cells.items()}

    # --- the control check, before any number is reported ------------------------
    ctrl = pools.get("y11x_pano_h200")
    problems = []
    if ctrl is None:
        problems.append("control leg y11x_pano_h200 did not cover all seven US splits")
    else:
        for key, pub in (("macro_p", "p"), ("macro_r", "r"), ("macro_f1", "f1")):
            got, want = round(ctrl[key], 3), PUBLISHED_CONTROL[pub]
            if abs(got - want) > 0.0005:
                problems.append(
                    f"control {key}={got:.3f} != published {PUBLISHED_CONTROL['model']} "
                    f"{pub}={want:.3f} -- YOLO scoring moved; the geometry read is NOT safe"
                )

    print("=" * 78)
    print("#51 geometry pair: does the equirect input explain the RampNet-YOLO gap?")
    print("=" * 78)
    print(f"\nControl: y11x_pano_h200 re-scored under this commit vs its published row")
    if ctrl:
        print(f"  published  P {PUBLISHED_CONTROL['p']:.3f}  R {PUBLISHED_CONTROL['r']:.3f}  F1 {PUBLISHED_CONTROL['f1']:.3f}")
        print(f"  re-scored  P {ctrl['macro_p']:.3f}  R {ctrl['macro_r']:.3f}  F1 {ctrl['macro_f1']:.3f}")
    print("  " + ("REPRODUCED - the #132 seam fix did not move YOLO pano scoring"
                  if not problems else "MISMATCH:\n    " + "\n    ".join(problems)))

    print("\nPer-split F1 at the pre-registered conf 0.25:\n")
    order = ["y11x_tiles", "y11x_pano", "y11x_pano_h200"]
    hdr = f"{'split':<20}" + "".join(f"{m:>17}" for m in order) + "   tiles-pano60"
    print(hdr)
    print("-" * len(hdr))
    for s in ALL_SPLITS_AS_RUN:
        line = f"{s:<20}"
        for m in order:
            c = cells.get(m, {}).get(s)
            line += f"{c['f1']:>17.3f}" if c else f"{'-':>17}"
        t = cells.get("y11x_tiles", {}).get(s)
        h = cells.get("y11x_pano_h200", {}).get(s)
        line += f"{t['f1'] - h['f1']:>15.3f}" if t and h else f"{'-':>15}"
        if s in HELD_OUT_AS_RUN:
            line += "  (held out)"
        print(line)

    print(f"\nMacro-mean over the seven pooled US splits ({', '.join(POOLED_SPLITS)}):\n")
    print(f"{'leg':<20}{'epoch':>7}{'P':>9}{'R':>9}{'F1':>9}{'AP':>9}   vs RampNet {PUBLISHED_RAMPNET_F1}")
    print("-" * 78)
    for m in order:
        p = pools.get(m)
        if not p:
            continue
        meta = LEGS.get(m, {})
        print(f"{m:<20}{meta.get('epoch', '?'):>7}{p['macro_p']:>9.3f}{p['macro_r']:>9.3f}"
              f"{p['macro_f1']:>9.3f}{(p['macro_ap'] or 0):>9.3f}"
              f"{p['macro_f1'] - PUBLISHED_RAMPNET_F1:>15.3f}")

    t = pools.get("y11x_tiles")        # tiles, ep44
    p38 = pools.get("y11x_pano")       # pano,  ep38
    h = pools.get("y11x_pano_h200")    # pano,  ep60 -- the published arm
    decomposition = None
    if t and p38 and h:
        # The published gap does NOT decompose into "geometry" alone. The tiles arm is
        # at ep44 and the published pano arm at ep60, so a raw tiles-minus-published
        # difference mixes geometry with training budget. Split it on the pano lineage,
        # where budget is the only thing that moves:
        #     budget   = pano ep38 - pano ep60   (same geometry, different budget)
        #     geometry = tiles ep44 - pano ep38  (different geometry, ~matched budget)
        budget = p38["macro_f1"] - h["macro_f1"]
        geometry = t["macro_f1"] - p38["macro_f1"]
        total = t["macro_f1"] - h["macro_f1"]
        gap = PUBLISHED_RAMPNET_F1 - h["macro_f1"]
        decomposition = {"budget": budget, "geometry": geometry, "total": total,
                         "published_gap": gap, "residual": PUBLISHED_RAMPNET_F1 - t["macro_f1"]}
        print(f"\nThe published {gap:.3f} F1 gap, decomposed (pooled US splits):")
        print(f"  over-training  pano ep60 -> ep38   {budget:+.3f}   same geometry, less budget")
        print(f"  geometry       pano ep38 -> tiles  {geometry:+.3f}   ~matched budget (38 vs 44)")
        print(f"  {'':<33}{'-' * 7}")
        print(f"  best YOLO cell we have           {total:+.3f}  ({total / gap * 100:.0f}% of the gap)")
        print(f"\n  Residual still to RampNet: {decomposition['residual']:.3f} F1.")
        print(f"  The geometry half is RECALL: R {p38['macro_r']:.3f} -> {t['macro_r']:.3f} "
              f"({t['macro_r'] - p38['macro_r']:+.3f}) at P {p38['macro_p']:.3f} -> "
              f"{t['macro_p']:.3f} ({t['macro_p'] - p38['macro_p']:+.3f}).")
        print("\n  CAVEAT: y11x_pano and y11x_pano_h200 are divergent continuations of one\n"
              "  lineage (h200 forked from y11x_pano/best.pt, MANIFEST-2026-08-03), trained\n"
              "  on different hardware. The budget term is therefore suggestive, not clean:\n"
              "  it is confounded with the fork. The geometry term is the better-controlled\n"
              "  of the two, and it is the smaller one.")

    payload = rnd({
        "what": "#51 geometry-pair eval: tiles vs pano at near-matched budget, "
                "plus the published pano arm re-scored as a control",
        "operating_point": 0.25,
        "pooled_splits": list(POOLED_SPLITS),
        "held_out": HELD_OUT_AS_RUN,
        "legs": LEGS,
        "published_control": PUBLISHED_CONTROL,
        "published_rampnet_pooled_f1": PUBLISHED_RAMPNET_F1,
        "control_reproduced": not problems,
        "per_split": cells,
        "best_f1_sweep_tune_on_test": best,
        "pooled": pools,
        "decomposition": decomposition,
    })

    if args.check:
        if not os.path.exists(OUT_JSON):
            sys.exit(f"--check: {OUT_JSON} does not exist")
        with open(OUT_JSON, encoding="utf-8") as fh:
            if json.load(fh) != payload:
                sys.exit("--check: committed JSON does not match a fresh read of the reports")
        print(f"\n--check: {os.path.relpath(OUT_JSON, ROOT)} matches.")
    else:
        # newline="" so the committed bytes are LF on every platform.
        with open(OUT_JSON, "w", encoding="utf-8", newline="") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
