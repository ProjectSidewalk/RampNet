#!/usr/bin/env python
"""How often does the published-style extractor split one ramp across the 360 deg seam? (#248)

``stage_two/evaluate.py`` finds peaks with ``peak_local_max(..., exclude_border=False)``,
whose non-maximum suppression does not wrap in x. A ramp sitting on the seam can therefore
come back as two peaks, one at each edge of the panorama. ``rampnet.subcell.detect_peaks``
gained an opt-in ``wrap_nms=True`` that suppresses across the seam (docs/seam_nms_248.md).
This script counts, from committed peaks, how many such "straddling pairs" the default
extractor emits.

Input: ``docs/data/run_a_84_detections/*.json`` -- the peaks of 8 Run A epoch checkpoints on
the 1,000 manual_gold panoramas (no TTA, floor 0.05, ``peak_min_distance`` 10,
``exclude_border=False``; the ``signature`` block of each file says so and is checked
here). No heatmaps are committed, so this works on peaks only.

A **straddling pair** is two peaks of one panorama whose Chebyshev distance on the 1024 x
512 heatmap grid is >= 10 px measured straight across the image but < 10 px with x wrapped:
the pair the default spacing keeps and a wrapped spacing would not. A pair counts at
threshold ``t`` when both peaks score >= ``t``.

The ``dropped`` column is an **estimate** of how many peaks ``wrap_nms=True`` removes: a
greedy pass over the stored peaks in descending score that rejects a peak within wrapped
Chebyshev 10 px of one already kept. The real ``wrap_nms`` also runs a wrapped maximum
filter over the heatmap, which can drop a peak whose stronger neighbour across the seam was
itself never a stored peak, and can pick another pixel of an edge plateau -- neither is
visible without the heatmap.

The pairs are then joined, **by panorama only**, to the seam adjudication of the ground
truth (``benchmark/manual_gold/seam_verdicts__jon.json``, docs/seam.md section 2): was the
panorama one where a seam ramp was judged ``one`` ramp marked twice, ``two`` real ramps, or
not adjudicated (``none``)? This says where the pairs fall, not that a given pair is the
adjudicated ramp; matching pairs to marks would need the pair geometry compared with the
marks, which is not done here.

Caveat that travels with the numbers: these are **Run A** checkpoints (#84), not the
published ``projectsidewalk/rampnet-model``; the published model's heatmaps would need a
GPU re-dump.

Usage::

    python scripts/analysis/seam_pairs_248.py                 # the table
    python scripts/analysis/seam_pairs_248.py --markdown      # the docs table
    python scripts/analysis/seam_pairs_248.py --check         # assert the pinned counts

Stdlib only. Writes nothing.
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(REPO, "docs", "data", "run_a_84_detections")
DEFAULT_VERDICTS = os.path.join(REPO, "benchmark", "manual_gold", "seam_verdicts__jon.json")
VERDICT_CLASSES = ("one", "two", "none")
W, H = 1024, 512
MIN_DISTANCE = 10
DEFAULT_THRESHOLDS = (0.05, 0.30, 0.55)

#: The numbers docs/seam_nms_248.md and the PR quote (all 8 epochs pooled).
PINNED = {
    "files": 8,
    "panos": 8000,
    "peaks": 36050,
    "pairs": {"0.05": 114, "0.3": 72, "0.55": 58},
    # distinct panoramas holding a pair, split by the GT seam verdict of the panorama
    "distinct_panos": {"0.05": {"one": 8, "two": 2, "none": 21},
                       "0.3": {"one": 8, "two": 2, "none": 5},
                       "0.55": {"one": 8, "two": 2, "none": 1}},
}

#: What every input file must say about how its peaks were extracted.
SIGNATURE = {"exclude_border": False, "peak_min_distance": MIN_DISTANCE,
             "heatmap_size": [H, W], "dataset": "manual", "tta": False}


def to_px(det):
    x, y, s = det
    return round(x * W), round(y * H), float(s)


def is_straddle(a, b, d=MIN_DISTANCE, width=W):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) >= d and max(min(dx, width - dx), dy) < d


def greedy_wrapped(peaks, d=MIN_DISTANCE, width=W):
    """Peaks a wrapped spacing pass keeps, highest first (the estimate; see module doc)."""
    kept = []
    for p in sorted(peaks, key=lambda q: -q[2]):
        if all(max(min(abs(p[0] - k[0]), width - abs(p[0] - k[0])), abs(p[1] - k[1])) >= d
               for k in kept):
            kept.append(p)
    return kept


def load(detections_dir):
    files = sorted(glob.glob(os.path.join(detections_dir, "*.json")))
    if not files:
        raise SystemExit(f"no *.json in {detections_dir}")
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        sig = d.get("signature", {})
        bad = {k: sig.get(k) for k, v in SIGNATURE.items() if sig.get(k) != v}
        if bad:
            raise SystemExit(f"{os.path.basename(f)}: signature differs from {SIGNATURE}: {bad}")
        out.append((d.get("model", os.path.basename(f)), sig.get("peak_floor"),
                    {pid: [to_px(x) for x in dets] for pid, dets in d["detections"].items()}))
    return out


def load_verdicts(path):
    with open(path, encoding="utf-8") as fh:
        return {v["pano"]: v["verdict"] for v in json.load(fh)["verdicts"]}


def measure(runs, thresholds, verdicts=None):
    verdicts = verdicts or {}
    keys = [f"{t:g}" for t in thresholds]
    pair_panos = {k: set() for k in keys}
    total = {"files": len(runs), "panos": 0, "peaks": 0,
             "peaks_at": dict.fromkeys(keys, 0), "pairs": dict.fromkeys(keys, 0),
             "panos_with_pair": dict.fromkeys(keys, 0), "dropped": dict.fromkeys(keys, 0)}
    per_model = []
    for model, floor, dets in runs:
        row = {"model": model, "floor": floor, "pairs": dict.fromkeys(keys, 0)}
        total["panos"] += len(dets)
        for peaks in dets.values():
            total["peaks"] += len(peaks)
            for t, k in zip(thresholds, keys):
                ps = [p for p in peaks if p[2] >= t]
                n = sum(is_straddle(ps[i], ps[j])
                        for i in range(len(ps)) for j in range(i + 1, len(ps)))
                total["peaks_at"][k] += len(ps)
                total["pairs"][k] += n
                row["pairs"][k] += n
                total["panos_with_pair"][k] += int(n > 0)
                if n:
                    total["dropped"][k] += len(ps) - len(greedy_wrapped(ps))
        for pid, peaks in dets.items():
            for t, k in zip(thresholds, keys):
                ps = [p for p in peaks if p[2] >= t]
                if any(is_straddle(ps[i], ps[j])
                       for i in range(len(ps)) for j in range(i + 1, len(ps))):
                    pair_panos[k].add(pid)
        per_model.append(row)
    total["distinct_panos"] = {
        k: {c: sum(verdicts.get(p, "none") == c for p in pair_panos[k]) for c in VERDICT_CLASSES}
        for k in keys}
    return total, per_model


def render(total, per_model, markdown=False):
    keys = list(total["pairs"])
    lines = []
    if markdown:
        lines += ["| threshold | peaks | straddling pairs | panos with a pair | "
                  "peaks `wrap_nms` would drop (estimate) |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for k in keys:
            lines.append(f"| {k} | {total['peaks_at'][k]:,} | {total['pairs'][k]} | "
                         f"{total['panos_with_pair'][k]} | {total['dropped'][k]} |")
        lines += ["", "| threshold | distinct panos with a pair | GT seam verdict `one` | "
                  "`two` | not adjudicated |", "| ---: | ---: | ---: | ---: | ---: |"]
        for k in keys:
            dp = total["distinct_panos"][k]
            lines.append(f"| {k} | {sum(dp.values())} | {dp['one']} | {dp['two']} | "
                         f"{dp['none']} |")
        lines += ["", "| checkpoint | " + " | ".join(f"pairs >= {k}" for k in keys) + " |",
                  "| :--- | " + " | ".join("---:" for _ in keys) + " |"]
        for r in per_model:
            lines.append(f"| {r['model']} | " + " | ".join(str(r['pairs'][k]) for k in keys)
                         + " |")
    else:
        lines.append(f"{total['files']} files, {total['panos']} pano-runs, "
                     f"{total['peaks']} stored peaks")
        lines.append(f"{'thr':>6} {'peaks':>7} {'pairs':>6} {'panos':>6} {'dropped~':>9}")
        for k in keys:
            lines.append(f"{k:>6} {total['peaks_at'][k]:>7} {total['pairs'][k]:>6} "
                         f"{total['panos_with_pair'][k]:>6} {total['dropped'][k]:>9}")
        lines.append("")
        lines.append("distinct panos with a pair, by GT seam verdict (one / two / none):")
        for k in keys:
            dp = total["distinct_panos"][k]
            lines.append(f"{k:>6} {dp['one']:>3} / {dp['two']} / {dp['none']}")
        lines.append("")
        for r in per_model:
            lines.append(f"  {r['model']:<16} " + "  ".join(f"{k}:{r['pairs'][k]}" for k in keys))
    return "\n".join(lines)


def check(total):
    bad = []
    for k in ("files", "panos", "peaks"):
        if total[k] != PINNED[k]:
            bad.append(f"{k}: {total[k]} != pinned {PINNED[k]}")
    for k, v in PINNED["pairs"].items():
        got = total["pairs"].get(k)
        if got is None:
            bad.append(f"pairs at {k}: threshold not measured (run with the default thresholds)")
        elif got != v:
            bad.append(f"pairs at {k}: {got} != pinned {v}")
    for k, v in PINNED["distinct_panos"].items():
        got = total["distinct_panos"].get(k)
        if got != v:
            bad.append(f"distinct panos at {k}: {got} != pinned {v}")
    return bad


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detections-dir", default=DEFAULT_DIR,
                    help="directory of run_a_84-style detection JSONs (default: %(default)s)")
    ap.add_argument("--seam-verdicts", default=DEFAULT_VERDICTS,
                    help="GT seam adjudication JSON (default: %(default)s)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                    help="score thresholds; a pair counts when both peaks are >= it")
    ap.add_argument("--markdown", action="store_true", help="print the docs tables")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the counts equal the pinned ones in PINNED")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    total, per_model = measure(load(args.detections_dir), args.thresholds,
                               load_verdicts(args.seam_verdicts))
    print(render(total, per_model, args.markdown))
    if args.check:
        bad = check(total)
        if bad:
            print("\nCHECK FAILED:\n  " + "\n  ".join(bad))
            return 1
        print("\ncheck OK: counts equal the pinned ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
