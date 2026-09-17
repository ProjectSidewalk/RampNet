"""Is the seam recall deficit caused by border truncation? The roll test (#132 §7).

#132 §2 measures that RampNet finds well under half the ramps sitting within ~4 deg of
the 360 seam, after wrapping the matcher, deduping the seam-crossing GT duplicates, and
standardizing for range. That is a *measurement*; the *cause* is a hypothesis. The
leading candidate is that the convolutions pad the left and right borders with zeros, so
a ramp on the seam is processed with half its context replaced by nothing.

This tests it, and it is deliberately a **double dissociation** rather than a one-way
check, because a one-way check cannot separate the hypothesis from "rolling helps
everything".

Rolling a panorama by half its width does two things at once:

    ramps at the seam  (x ~ 0 or 1)  ->  move to the centre  (x ~ 0.5)
    ramps at the centre (x ~ 0.5)    ->  move to the seam    (x ~ 0 or 1)

So border truncation predicts a **crossover**, not a uniform lift:

    seam-band GT   : found rarely in the original pass, often in the rolled pass
    centre-band GT : found often in the original pass, rarely in the rolled pass

A uniform improvement in both bands would falsify the mechanism while still "recovering
the ramps" — which is exactly why the control band is not optional. The centre band is
free: it comes from the same panoramas in the same forward passes.

Scoring matches #132 throughout: wrapping matcher, seam-crossing duplicate GT merged,
peaks at min_distance=10, deployed threshold 0.30. The original pass is also compared
against the committed op_cache detections as a check that this inference path reproduces
the one that produced them.

Usage:
    python scripts/analysis/seam_roll_diagnostic.py --json-out analysis_out/seam_roll.json
    python scripts/analysis/seam_roll_diagnostic.py --limit 5      # smoke test, no json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.feature import peak_local_max

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from operating_point_curve import CACHE_DIR, read_cache  # noqa: E402
from miss_decomposition import ALL_SPLITS  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from threshold_sweep import PRE, load_model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

RSQ = radius_sq_for()
R = RSQ ** 0.5                      # 22.53 px in matcher units (1024-wide axis)
SEAM_BAND = R / 2                   # 11.27 px ~ 3.96 deg — the band #132 measures
PEAK_MIN_DISTANCE = 10
DEFAULT_THRESHOLD = 0.30


# --- geometry: the wrapped primitives #132 argues should exist once ------------------

def wrapped_dist_sq(ax, ay, bx, by, scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """Squared distance with the x axis wrapped at the 360 seam, in matcher units."""
    dx = abs(ax - bx) * scale_x
    dx = min(dx, scale_x - dx)
    return dx * dx + ((ay - by) * scale_y) ** 2


def dist_to_seam(x):
    """Distance from a normalized x to the nearest seam edge, in matcher units."""
    return min(x, 1.0 - x) * PANO_SCALE_X


def wrapped_match(pred_points, gt_points):
    """Greedy nearest-unclaimed match with a wrapping x axis. Returns claimed GT indices."""
    claimed = [False] * len(gt_points)
    hit = set()
    for px, py in pred_points:
        best_k, best_d = -1, RSQ
        for k, (gx, gy) in enumerate(gt_points):
            if claimed[k]:
                continue
            d = wrapped_dist_sq(px, py, gx, gy)
            if d < RSQ and d < best_d:
                best_d, best_k = d, k
        if best_k >= 0:
            claimed[best_k] = True
            hit.add(best_k)
    return hit


def dedup_seam_only(gt_points):
    """Drop the second member of a GT pair that duplicates ACROSS THE SEAM.

    Seam-crossing pairs only: genuinely adjacent ramps away from the seam are common in
    this data (#130) and merging them would strip hard cases out of the comparison.

    Which member survives is first-in-list-order, so it is deterministic against the
    committed op_cache but can differ against a regenerated cache whose rows come back
    in another order — the kept member's coordinates (and so its seam distance) would
    shift by the pair separation, up to ~25 px.
    """
    keep = []
    for g in gt_points:
        dup = False
        for k in keep:
            dx = abs(g[0] - k[0]) * PANO_SCALE_X
            if dx <= PANO_SCALE_X / 2:
                continue                                   # not seam-crossing
            if wrapped_dist_sq(g[0], g[1], k[0], k[1]) < RSQ:
                dup = True
                break
        if not dup:
            keep.append(g)
    return keep


# --- inference ----------------------------------------------------------------------

def find_pano_image(panos_root, city, pid):
    """The bundle's image for one pano, or None.

    ``panos_root`` is a directory holding ``<city>/panos/`` — normally the repo's own
    ``benchmark/``, but the images are untracked local artifacts, so a git worktree will
    not have them and the path has to be given explicitly.
    """
    for ext in ("jpg", "jpeg", "png", "webp"):
        hits = glob.glob(os.path.join(panos_root, city, "panos", f"{pid}.{ext}"))
        if hits:
            return hits[0]
    return None


def heatmap(model, device, img):
    t = PRE(img).unsqueeze(0).to(device)
    with torch.no_grad():
        h = model(t)
    return h.squeeze().float().cpu().numpy()


def peaks(h, threshold):
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=PEAK_MIN_DISTANCE,
                        threshold_abs=threshold)
    H, W = h.shape
    return [(float(c / W), float(r / H), float(h[r][c])) for r, c in pk]


def roll_half(img):
    """The panorama rolled by half its width — the seam moved to where the centre was.

    Applied to the image at native resolution, before preprocessing. This is the one
    definition of the roll; ``seam_response.py`` measures through the same function so
    the two seam experiments cannot drift apart.
    """
    w, h_px = img.size
    half = w // 2
    rolled = Image.new(img.mode, img.size)
    rolled.paste(img.crop((half, 0, w, h_px)), (0, 0))
    rolled.paste(img.crop((0, 0, half, h_px)), (w - half, 0))
    return rolled


def rolled_pass(model, device, img, threshold):
    """Detections from a pano rolled by half its width, mapped back to original coords.

    Peaks come back in rolled coordinates and are shifted by -0.5 (mod 1) to land in
    the original frame.
    """
    dets = peaks(heatmap(model, device, roll_half(img)), threshold)
    return [(((x - 0.5) % 1.0), y, s) for x, y, s in dets]


# --- main ---------------------------------------------------------------------------

def collect_targets(select="seam"):
    """{(city, pid): (gt_points_deduped, cached_preds)} for panos holding band GT.

    ``select`` picks which half of the dissociation to gather panos for. Selecting on
    ``seam`` measures whether the roll RECOVERS ramps; selecting on ``centre`` measures
    whether it BREAKS ramps that were fine — the falsifying direction. The centre band
    is only ~2% of the azimuth range, so panos have to be chosen for it deliberately;
    riding along on the seam selection yields almost none.
    """
    want = (lambda x: dist_to_seam(x) < SEAM_BAND) if select == "seam" else            (lambda x: abs(x - 0.5) * PANO_SCALE_X < SEAM_BAND)
    out = {}
    for city in dict.fromkeys(ALL_SPLITS):
        path = os.path.join(CACHE_DIR, f"{city}.json")
        if not os.path.exists(path):
            continue
        for rec in read_cache(path)[0]:
            gt = dedup_seam_only(list(rec["gt"].gt_points))
            if any(want(g[0]) for g in gt):
                out[(city, rec["pano"])] = (gt, list(rec["preds"]))
    return out


def band_of(x):
    """'seam' within SEAM_BAND of an edge, 'centre' within SEAM_BAND of x=0.5, else None.

    The centre band is exactly where the seam band lands after the roll, so the two are
    the mirror halves of the dissociation.
    """
    if dist_to_seam(x) < SEAM_BAND:
        return "seam"
    if abs(x - 0.5) * PANO_SCALE_X < SEAM_BAND:
        return "centre"
    return "elsewhere"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"),
                    help="Directory holding <city>/panos/ (default: the repo's "
                         "benchmark/). The images are untracked, so a worktree needs "
                         "this pointed at a checkout that has them.")
    ap.add_argument("--select", choices=("seam", "centre"), default="seam",
                    help="Which band to select panoramas for. 'seam' measures recovery; "
                         "'centre' measures the falsifying direction (ramps the roll "
                         "should BREAK by moving them onto the seam).")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process this many panos (smoke test; refuses --json-out).")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if args.limit and args.json_out:
        ap.error("--limit with --json-out would write a partial result as if it were full")

    targets = collect_targets(args.select)
    keys = sorted(targets)
    if args.limit:
        keys = keys[:args.limit]
    print(f"panos holding {args.select}-band GT: {len(targets)}"
          f"{f' (processing {len(keys)})' if args.limit else ''}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    print(f"device={device}", flush=True)

    rows = []
    cache_agree = [0, 0]
    skipped = []
    for i, (city, pid) in enumerate(keys, 1):
        gt, cached = targets[(city, pid)]
        path = find_pano_image(args.panos_root, city, pid)
        if path is None:
            skipped.append(f"{city}:{pid}")
            continue
        img = Image.open(path).convert("RGB")

        orig = peaks(heatmap(model, device, img), args.threshold)
        roll = rolled_pass(model, device, img, args.threshold)

        hit_o = wrapped_match([(p[0], p[1]) for p in orig], gt)
        hit_r = wrapped_match([(p[0], p[1]) for p in roll], gt)
        hit_c = wrapped_match([(p[0], p[1]) for p in cached
                               if p[2] >= args.threshold], gt)
        cache_agree[0] += len(hit_o & hit_c)
        cache_agree[1] += len(hit_c)

        for k, g in enumerate(gt):
            rows.append({"city": city, "pano": pid, "x": round(g[0], 6),
                         "y": round(g[1], 6), "band": band_of(g[0]),
                         "seam_px": round(dist_to_seam(g[0]), 2),
                         "found_original": k in hit_o, "found_rolled": k in hit_r,
                         "found_cached": k in hit_c})
        if i % 5 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)}", flush=True)

    print(f"\ncheck: this inference path reproduces {cache_agree[0]}/{cache_agree[1]} "
          f"of the committed cache's matched GT", flush=True)
    if skipped:
        print(f"skipped (no local imagery): {len(skipped)} -> {skipped[:5]}", flush=True)

    print(f"\n{'band':10} {'GT':>5} {'original':>10} {'rolled':>10} {'change':>9}")
    summary = {}
    for band in ("seam", "centre", "elsewhere"):
        sel = [r for r in rows if r["band"] == band]
        if not sel:
            continue
        n = len(sel)
        o = sum(r["found_original"] for r in sel)
        rr = sum(r["found_rolled"] for r in sel)
        summary[band] = {"n": n, "original": o, "rolled": rr,
                         "recall_original": o / n, "recall_rolled": rr / n}
        print(f"{band:10} {n:5} {o/n:10.4f} {rr/n:10.4f} {rr/n - o/n:+9.4f}")

    s, c = summary.get("seam"), summary.get("centre")
    if s and c:
        ds = s["recall_rolled"] - s["recall_original"]
        dc = c["recall_rolled"] - c["recall_original"]
        print(f"\ncrossover: seam {ds:+.4f} vs centre {dc:+.4f}")
        print("Border truncation predicts seam UP and centre DOWN. A lift in both is a "
              "uniform effect and does NOT support the mechanism.")

    if args.json_out:
        payload = {"select": args.select,
                   "threshold": args.threshold, "seam_band_px": SEAM_BAND,
                   "panos_root": os.path.abspath(args.panos_root),
                   "peak_min_distance": PEAK_MIN_DISTANCE,
                   "panos": len(keys), "skipped": skipped,
                   "cache_reproduction": {"agreed": cache_agree[0], "of": cache_agree[1]},
                   "summary": summary, "results": rows}
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", newline="") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
