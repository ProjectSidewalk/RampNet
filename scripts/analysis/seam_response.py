"""Does the 360 seam cost the DETECTOR anything? A paired within-ramp test (#132).

The seam's effect on the model has been measured wrongly here once already. The first
attempt compared extracted detections and concluded the model was blind at the seam; that
was `peak_local_max`'s ``exclude_border`` discarding peaks near the array edge, not a
model property, and it is retracted (see ``docs/seam.md`` section 1).

This measures the thing that bug cannot touch: the **raw heatmap response** at the ramp,
with no peak extraction anywhere in the path. Responses are read with the same
unit-tested window every #46 analysis uses (``radius_max``: columns wrap, rows clamp,
values clipped to [0, 1] exactly as peak extraction clips), so a response here and a
detection score in the op_cache are on the same scale.

It is paired and within-ramp, which is what makes it interpretable. For each ground-truth
ramp we take the maximum response inside the match radius twice:

    A. the panorama as stored     - the seam runs through or beside the ramp
    B. the panorama rolled 180 deg - the same ramp, same pixels, seam half a world away

Everything except the seam's position is held constant, so the paired difference isolates
it. Ramps far from the seam in the *same panoramas* are the control: the roll should do
nothing to them, and if it does, the roll itself is not neutral and no seam conclusion can
be drawn from it.

That validity requirement is why there are three bins, not two. A ramp goes to
``excluded`` — usable by neither arm — when the roll is not neutral for it:

    * its response window (the match radius R) touches the seam as stored, or touches
      the column the seam lands on after the roll (the centre), so one of the two
      passes reads a window the seam cuts through;
    * its window overlaps a seam-band ramp's window (closer than 2 R), so its
      "control" response is partly the seam ramp's response measured again;
    * its window overlaps an already-kept control ramp's window — adjacent ramps
      sharing one response are one measurement, not two, and counting both would
      overstate the control sample the t statistic divides by.

Read the RATE, not just the mean. The effect is heterogeneous — it depends on how much of
a ramp falls on each side of the split — so averaging over affected and unaffected ramps
dilutes it. The rate of ramps whose response moves materially, against the control's rate,
is the sharper statistic.

Inputs, for someone who is not on this machine: the committed ``analysis_out/op_cache``
(ground truth), the published ``projectsidewalk/rampnet-model`` weights (downloaded from
Hugging Face; the revision used is recorded in the output), and the native-resolution
panoramas at ``benchmark/<city>/panos/<pano>.jpg`` — git-ignored, published as
``projectsidewalk/rampnet-benchmark``. A git worktree will not have the panoramas, so
``--panos-root`` must point at a checkout that does. A GPU-ish machine; the RTX 3070
does ~10 s/pano.

Usage:
    python scripts/analysis/seam_response.py --json-out analysis_out/seam_response.json
    python scripts/analysis/seam_response.py --limit 3     # smoke test, refuses --json-out
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miss_decomposition import ALL_SPLITS  # noqa: E402
from operating_point_curve import CACHE_DIR  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from rampnet.geometry import dist_sq, dist_to_seam  # noqa: E402
from seam_roll_diagnostic import (  # noqa: E402
    SEAM_BAND, collect_targets, find_pano_image, heatmap, roll_half)
from silent_activation import HEAT_W, radius_max  # noqa: E402
from threshold_sweep import load_model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

RSQ = radius_sq_for()
R = RSQ ** 0.5
CENTRE = PANO_SCALE_X / 2    # dist_to_seam of the column the seam lands on when rolled
MATERIAL = 0.05              # a response change of MORE than this counts as "moved"

MODEL_ID = "projectsidewalk/rampnet-model"


# --- pure core (no torch, no I/O) — unit-tested in tests/test_seam_response.py -------

def arm_of(d):
    """'seam', 'excluded', or 'control' for a ramp ``d`` matcher-px from the seam.

    ``excluded`` here covers only the window-touches-a-seam-position cases; the
    window-overlap exclusions need the other ramps in the panorama and live in
    ``bin_pano``.
    """
    if d < SEAM_BAND:
        return "seam"
    if d < R:
        return "excluded"            # window touches the seam as stored
    if CENTRE - d < R:
        return "excluded"            # window touches where the seam lands when rolled
    return "control"


def bin_pano(gt_points):
    """``[(g, arm, reason)]`` for one panorama's GT, applying every exclusion rule.

    Control ramps are visited in (x, y) order so that which member of an
    adjacent-ramp pair is kept does not depend on op_cache row order. Seam-band
    ramps are never dropped — they are the measured sample — but pairs of them
    closer than 2 R share a window, and the caller reports that count so the
    non-independence is visible rather than silent.
    """
    labeled = [(g, arm_of(dist_to_seam(g[0], PANO_SCALE_X))) for g in gt_points]
    seam_pts = [g for g, a in labeled if a == "seam"]

    def overlaps(g, others):
        return any(dist_sq(g[0], g[1], o[0], o[1], PANO_SCALE_X, PANO_SCALE_Y,
                           wrap_x=True) < (2 * R) ** 2 for o in others)

    out, kept_control = [], []
    for g, a in sorted(labeled, key=lambda t: (t[0][0], t[0][1])):
        reason = None
        if a == "excluded":
            reason = "window touches a seam position (stored or rolled)"
        elif a == "control":
            if overlaps(g, seam_pts):
                a, reason = "excluded", "window overlaps a seam-band ramp's window"
            elif overlaps(g, kept_control):
                a, reason = "excluded", "window overlaps a kept control ramp's window"
            else:
                kept_control.append(g)
        out.append((g, a, reason))
    return out


def seam_window_overlap_pairs(gt_points):
    """How many seam-band pairs in one panorama sit closer than 2 R (shared windows)."""
    pts = [g for g in gt_points if dist_to_seam(g[0], PANO_SCALE_X) < SEAM_BAND]
    return sum(1 for i in range(len(pts)) for j in range(i + 1, len(pts))
               if dist_sq(pts[i][0], pts[i][1], pts[j][0], pts[j][1],
                          PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True) < (2 * R) ** 2)


def summarize(rows):
    diffs = [r["rolled"] - r["stored"] for r in rows]
    n = len(diffs)
    if n == 0:
        return {}
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n ** 0.5 if n > 1 else 0.0
    return {"n": n,
            "mean_stored": sum(r["stored"] for r in rows) / n,
            "mean_rolled": sum(r["rolled"] for r in rows) / n,
            "paired_diff": mean, "se": se,
            "t": (mean / se) if se > 0 else None,
            "gained": sum(1 for d in diffs if d > MATERIAL),
            "lost": sum(1 for d in diffs if d < -MATERIAL)}


def round_floats(obj, digits=6):
    """Round every float in a JSON-shaped object, so the bytes on disk are portable."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


# --- provenance ----------------------------------------------------------------------

def model_revision():
    """The HF commit the loaded weights came from, read off the cached snapshot path."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(MODEL_ID, "model.safetensors")
    except Exception:
        try:
            path = hf_hub_download(MODEL_ID, "pytorch_model.bin")
        except Exception:
            return None
    m = re.search(r"snapshots[/\\]([0-9a-f]{7,40})", path)
    return m.group(1) if m else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- main ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"),
                    help="Directory holding <city>/panos/ (default: the repo's "
                         "benchmark/). The images are untracked, so a worktree needs "
                         "this pointed at a checkout that has them.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process this many panos (smoke test; refuses --json-out).")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if args.limit and args.json_out:
        ap.error("--limit with --json-out would write a partial result as if it were full")

    targets = collect_targets("seam")
    keys = sorted(targets)
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        sys.exit("no panoramas hold seam-band GT — is the committed analysis_out/"
                 "op_cache present? (its location honours $RAMPNET_ANALYSIS_OUT)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    print(f"panoramas: {len(keys)}   device={device}", flush=True)

    rows = {"seam": [], "control": [], "excluded": []}
    skipped = []
    overlap_pairs = 0
    for i, (city, pid) in enumerate(keys, 1):
        path = find_pano_image(args.panos_root, city, pid)
        if path is None:
            skipped.append(f"{city}:{pid}")
            continue
        img = Image.open(path).convert("RGB")
        stored = heatmap(model, device, img)
        rolled = np.roll(heatmap(model, device, roll_half(img)), HEAT_W // 2, axis=1)
        gt = targets[(city, pid)][0]
        overlap_pairs += seam_window_overlap_pairs(gt)
        for g, arm, reason in bin_pano(gt):
            # stored/rolled are rounded here, before summarize() sees them, so the
            # summary block in the JSON is exactly re-derivable from the rows in it.
            row = {"pano": f"{city}:{pid}", "x": round(g[0], 6), "y": round(g[1], 6),
                   "seam_px": round(dist_to_seam(g[0], PANO_SCALE_X), 2),
                   "stored": round(radius_max(stored, g[0], g[1]), 6),
                   "rolled": round(radius_max(rolled, g[0], g[1]), 6)}
            if reason:
                row["reason"] = reason
            rows[arm].append(row)
        if i % 5 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)}", flush=True)

    if skipped:
        print(f"\nWARNING skipped (no local imagery): {len(skipped)} of {len(keys)} "
              f"-> {skipped[:5]}", flush=True)
    if not rows["seam"] or not rows["control"]:
        sys.exit("an arm came back empty — --panos-root must point at a checkout "
                 "holding benchmark/<city>/panos (see the module docstring)")

    s, c = summarize(rows["seam"]), summarize(rows["control"])
    for name, r in (("SEAM-BAND (within ~4 deg of the seam)", s),
                    ("CONTROL (same panoramas, away from the seam)", c)):
        print(f"\n{name}  n={r['n']}")
        print(f"  stored {r['mean_stored']:.3f} -> rolled {r['mean_rolled']:.3f}"
              f"   paired diff {r['paired_diff']:+.4f} (s.e. {r['se']:.4f}"
              + (f", t={r['t']:+.2f}" if r["t"] is not None else "") + ")")
        print(f"  response gained >{MATERIAL} when the seam is rolled away: "
              f"{r['gained']}/{r['n']}   lost: {r['lost']}/{r['n']}")
    reasons = {}
    for r in rows["excluded"]:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"\nexcluded from both arms: {len(rows['excluded'])}")
    for reason, n in sorted(reasons.items()):
        print(f"  {n:3}  {reason}")
    if overlap_pairs:
        print(f"seam-band pairs sharing a window (kept, but not independent): "
              f"{overlap_pairs}")
    print("\nThe RATE is the statistic. A heterogeneous effect — some ramps hurt a lot, "
          "most not at all —\naverages away, but the control fixes how often a ramp's "
          "response should move at all.")

    if args.json_out:
        payload = {"material_threshold": MATERIAL,
                   "seam_band_px": SEAM_BAND, "radius_px": R,
                   "model_id": MODEL_ID, "model_revision": model_revision(),
                   "panos_root": os.path.abspath(args.panos_root),
                   "cache_dir": os.path.abspath(CACHE_DIR),
                   "op_cache_sha256": {
                       city: sha256_file(os.path.join(CACHE_DIR, f"{city}.json"))
                       for city in dict.fromkeys(ALL_SPLITS)
                       if os.path.exists(os.path.join(CACHE_DIR, f"{city}.json"))},
                   "panos": len(keys), "skipped": skipped,
                   "seam_window_overlap_pairs": overlap_pairs,
                   "summary": {"seam": s, "control": c},
                   "seam_rows": sorted(rows["seam"], key=lambda r: r["seam_px"]),
                   "control_rows": rows["control"],
                   "excluded_rows": sorted(rows["excluded"], key=lambda r: r["seam_px"])}
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8", newline="") as f:
            json.dump(round_floats(payload), f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
