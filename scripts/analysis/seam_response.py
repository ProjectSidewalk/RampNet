"""Does the 360 seam cost the DETECTOR anything? A paired within-ramp test (#132).

The seam's effect on the model has been measured wrongly here once already. The first
attempt compared extracted detections and concluded the model was blind at the seam; that
was `peak_local_max`'s ``exclude_border`` discarding peaks near the array edge, not a
model property, and it is retracted (see ``docs/seam.md`` section 1).

This measures the thing that bug cannot touch: the **raw heatmap response** at the ramp,
with no peak extraction anywhere in the path.

It is paired and within-ramp, which is what makes it interpretable. For each ground-truth
ramp we take the maximum response inside the match radius twice:

    A. the panorama as stored     - the seam runs through or beside the ramp
    B. the panorama rolled 180 deg - the same ramp, same pixels, seam half a world away

Everything except the seam's position is held constant, so the paired difference isolates
it. Ramps far from the seam in the *same panoramas* are the control: the roll should do
nothing to them, and if it does, the roll itself is not neutral and no seam conclusion can
be drawn from it.

Read the RATE, not just the mean. The effect is heterogeneous — it depends on how much of
a ramp falls on each side of the split — so averaging over affected and unaffected ramps
dilutes it. The rate of ramps whose response moves materially, against the control's rate,
is the sharper statistic.

Usage:
    python scripts/analysis/seam_response.py --panos-root benchmark \
        --json-out analysis_out/seam_response.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampnet.detection_eval import PANO_SCALE_X, radius_sq_for  # noqa: E402
from rampnet.geometry import dist_to_seam  # noqa: E402
from seam_roll_diagnostic import (  # noqa: E402
    SEAM_BAND, collect_targets, find_pano_image, heatmap)
from threshold_sweep import load_model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

HEAT_W, HEAT_H = 1024, 512
R = radius_sq_for() ** 0.5
MATERIAL = 0.05          # a response change this size or larger counts as "moved"


def roll_image(img):
    w, h = img.size
    half = w // 2
    out = Image.new(img.mode, img.size)
    out.paste(img.crop((half, 0, w, h)), (0, 0))
    out.paste(img.crop((0, 0, half, h)), (w - half, 0))
    return out


def response(heat, x, y):
    """Max heatmap value within the match radius of a normalized point.

    Columns wrap; rows do not. No peak extraction, no threshold, no NMS — so nothing in
    this number can be an artifact of how detections are pulled out of the heatmap.
    """
    cx, cy = x * HEAT_W, y * HEAT_H
    best = 0.0
    for row in range(max(0, int(cy - R)), min(HEAT_H, int(cy + R) + 1)):
        dy2 = (row - cy) ** 2
        if dy2 >= R * R:
            continue
        span = (R * R - dy2) ** 0.5
        for col in range(int(cx - span), int(cx + span) + 1):
            v = float(heat[row][col % HEAT_W])
            if v > best:
                best = v
    return best


def summarize(rows):
    diffs = [b - a for _, _, a, b in rows]
    n = len(diffs)
    if n == 0:
        return {}
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n ** 0.5 if n > 1 else 0.0
    return {"n": n,
            "mean_stored": sum(a for _, _, a, _ in rows) / n,
            "mean_rolled": sum(b for _, _, _, b in rows) / n,
            "paired_diff": mean, "se": se,
            "t": (mean / se) if se > 0 else None,
            "gained": sum(1 for d in diffs if d > MATERIAL),
            "lost": sum(1 for d in diffs if d < -MATERIAL)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"))
    ap.add_argument("--json-out", default=os.path.join(REPO, "analysis_out",
                                                       "seam_response.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    targets = collect_targets("seam")
    keys = sorted(targets)
    if args.limit:
        keys = keys[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    print(f"panoramas: {len(keys)}   device={device}", flush=True)

    seam, control = [], []
    for i, (city, pid) in enumerate(keys, 1):
        path = find_pano_image(args.panos_root, city, pid)
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        stored = heatmap(model, device, img)
        rolled = np.roll(heatmap(model, device, roll_image(img)), HEAT_W // 2, axis=1)
        for g in targets[(city, pid)][0]:
            a = response(stored, g[0], g[1])
            b = response(rolled, g[0], g[1])
            d = dist_to_seam(g[0], PANO_SCALE_X)
            (seam if d < SEAM_BAND else control).append(
                (f"{city}:{pid}", round(d, 2), a, b))
        if i % 5 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)}", flush=True)

    s, c = summarize(seam), summarize(control)
    for name, r in (("SEAM-BAND (within ~4 deg of the seam)", s),
                    ("CONTROL (same panoramas, away from the seam)", c)):
        print(f"\n{name}  n={r['n']}")
        print(f"  stored {r['mean_stored']:.3f} -> rolled {r['mean_rolled']:.3f}"
              f"   paired diff {r['paired_diff']:+.4f} (s.e. {r['se']:.4f}"
              + (f", t={r['t']:+.2f}" if r["t"] is not None else "") + ")")
        print(f"  response gained >{MATERIAL} when the seam is rolled away: "
              f"{r['gained']}/{r['n']}   lost: {r['lost']}/{r['n']}")
    print("\nThe RATE is the statistic. A heterogeneous effect — some ramps hurt a lot, "
          "most not at all —\naverages away, but the control fixes how often a ramp's "
          "response should move at all.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", newline="") as f:
            json.dump({"material_threshold": MATERIAL, "seam_band_px": SEAM_BAND,
                       "summary": {"seam": s, "control": c},
                       "seam_rows": [{"pano": p, "seam_px": d,
                                      "stored": round(a, 4), "rolled": round(b, 4)}
                                     for p, d, a, b in sorted(seam, key=lambda r: r[1])],
                       "control_rows": [{"pano": p, "seam_px": d,
                                         "stored": round(a, 4), "rolled": round(b, 4)}
                                        for p, d, a, b in control]}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
