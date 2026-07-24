"""Size-stratified recall + a visual montage of RampNet's hard misses.

Point labels carry no extent, so estimate each GT ramp's APPARENT SIZE from geometry:
curb ramps sit on the ground plane, so for a pano point at depression angle a below
the horizon, distance d = camera_height / tan(a), and a ramp of real width W subtends
W/d radians = W/d * (4096 / 2pi) px in RampNet's 4096-wide input space.

Then: recall as a function of apparent size. That forecasts the resolution lever —
if small-bucket recall is poor and large-bucket recall is high, doubling input
resolution should migrate ramps up a bucket.

Caveat: assumes camera height (GSV ~2.5 m mast; Mapillary consumer rigs vary) and
flat ground. That is exactly what depth estimation would fix.
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import math, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import (
    build_ground_truth, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
from compare import load_bundle, DetectionCache, cache_key
from detectors import build_detector

CAM_H, RAMP_W = 2.5, 1.2                 # metres
PX_PER_RAD = 4096.0 / (2 * math.pi)      # RampNet input space (4096 px = 360 deg)
R = math.sqrt(radius_sq_for())
PRO = "gemini-3.1-pro-preview"
BUCKETS = [(0, 12), (12, 20), (20, 32), (32, 50), (50, 80), (80, 1e9)]
class Args:
    gemini_model = PRO; qwen_model = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_coord_space = "auto"; tiling = "perspective"


def geom(y):
    """(distance_m, apparent_px) for a ground point at pano-normalized y."""
    depression = (y - 0.5) * math.pi          # radians below horizon
    if depression <= 1e-4:
        return float("inf"), 0.0
    d = min(CAM_H / math.tan(depression), 150.0)
    return d, RAMP_W / d * PX_PER_RAD


def dist(p, q):
    return math.hypot((p[0] - q[0]) * PANO_SCALE_X, (p[1] - q[1]) * PANO_SCALE_Y)


def matched_gt(preds, gt_points):
    confs = [_confidence(p) for p in preds]
    order = (sorted(range(len(preds)), key=lambda i: confs[i] if confs[i] is not None else -1e9,
                    reverse=True) if any(c is not None for c in confs) else range(len(preds)))
    claimed, hit = [False] * len(gt_points), set()
    for i in order:
        p = _xy(preds[i]); best_k, best = -1, R
        for k, g in enumerate(gt_points):
            if claimed[k]:
                continue
            dd = dist(p, g)
            if dd < best:
                best, best_k = dd, k
        if best_k >= 0:
            claimed[best_k] = True; hit.add(best_k)
    return hit


def collect(city):
    records, verdicts, panos_dir = load_bundle(os.path.join(REPO, "benchmark", city))
    label, gem = build_detector("gemini", PRO, records, Args())
    sig, cache = gem.signature(), DetectionCache(os.path.join(REPO, ".model_cache"))
    rows = []
    for pid, entry in verdicts.items():
        gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                entry["missed"], entry["no_missed"])
        if not gt.fn_confirmed:
            continue
        rp = [(d["x_normalized"], d["y_normalized"], d["confidence"])
              for d in records[pid]["detections"]]
        pro = cache.get(cache_key(label, sig, city, pid)) or []
        mr, mp = matched_gt(rp, gt.gt_points), matched_gt(pro, gt.gt_points)
        for i, g in enumerate(gt.gt_points):
            d_m, px = geom(g[1])
            rows.append({"city": city, "pid": pid, "x": g[0], "y": g[1], "dist": d_m,
                         "px": px, "hit": i in mr, "pro": i in mp,
                         "panos_dir": panos_dir})
    return rows


def stratified(rows, title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"{'apparent size (px)':>20} {'~dist':>8} {'n':>5} {'detected':>9} {'RECALL':>8}")
    print("-" * 72)
    for lo, hi in BUCKETS:
        b = [r for r in rows if lo <= r["px"] < hi]
        if not b:
            continue
        n, hit = len(b), sum(1 for r in b if r["hit"])
        md = sorted(r["dist"] for r in b)[n // 2]
        rng = f"{lo}-{int(hi)}" if hi < 1e8 else f"{lo}+"
        print(f"{rng:>20} {md:>7.0f}m {n:>5} {hit:>9} {hit/n:>8.3f}")
    n, hit = len(rows), sum(1 for r in rows if r["hit"])
    print(f"{'ALL':>20} {'':>8} {n:>5} {hit:>9} {hit/n:>8.3f}")


def montage(rows, city, out_png, n_max=30, fov_deg=12.0, cell=240):
    """Crop a fixed ANGULAR window around each hard miss so distant ramps genuinely
    look small, then tile. Fixed FOV keeps crops visually comparable."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    hard = sorted([r for r in rows if r["city"] == city and not r["hit"] and not r["pro"]],
                  key=lambda r: r["px"])[:n_max]
    by_pano = {}
    for r in hard:
        by_pano.setdefault(r["pid"], []).append(r)
    crops = []
    for pid, rs in by_pano.items():
        path = os.path.join(rs[0]["panos_dir"], f"{pid}.jpg")
        if not os.path.exists(path):
            continue
        im = Image.open(path).convert("RGB")
        W, H = im.size
        side = max(32, int(fov_deg / 360.0 * W))
        for r in rs:
            cx, cy = int(r["x"] * W), int(r["y"] * H)
            box = (cx - side // 2, max(0, cy - side // 2), cx + side // 2,
                   min(H, cy + side // 2))
            c = im.crop(box).resize((cell, cell), Image.LANCZOS)
            d = ImageDraw.Draw(c)
            d.rectangle([cell//2 - 3, cell//2 - 3, cell//2 + 3, cell//2 + 3], outline=(255, 60, 60), width=2)
            tag = f"{r['px']:.0f}px ~{r['dist']:.0f}m"
            d.rectangle([0, cell - 16, cell, cell], fill=(0, 0, 0))
            d.text((3, cell - 14), tag, fill=(255, 255, 0))
            crops.append(c)
        im.close()
    if not crops:
        print("no crops"); return None
    cols = 6
    rowsn = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * cell, rowsn * cell), (18, 18, 18))
    for i, c in enumerate(crops):
        sheet.paste(c, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(out_png)
    print(f"\nmontage: {len(crops)} hard misses -> {out_png}")
    return out_png


all_rows = []
for city in ("richmond", "bend"):
    r = collect(city)
    all_rows += r
    stratified(r, f"{city.upper()} — recall vs apparent ramp size (baseline thr 0.55)")
stratified(all_rows, "COMBINED — recall vs apparent ramp size")

print("\nMisses by size: " + ", ".join(
    f"{lo}-{int(hi) if hi<1e8 else '+'}: {sum(1 for r in all_rows if lo<=r['px']<hi and not r['hit'])}"
    for lo, hi in BUCKETS))
montage(all_rows, "richmond", os.path.join(OUT, "hard_misses_richmond.png"))
