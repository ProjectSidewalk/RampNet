"""Characterize RampNet's misses on the benchmark, to pick the recall lever.

For every GT ramp RampNet FAILED to detect (on recall-eligible panos), record:
  - how far the nearest RampNet detection was (in match-radius units) -> was this a
    LOCALIZATION near-miss (model fired nearby, peak just off / threshold) or a BLIND
    miss (model saw nothing there at all)?
  - vertical position in the equirect pano. Horizon = y 0.5; ground ramps sit below it.
    A DISTANT/small ramp projects near the horizon (y just over 0.5); a close one
    projects low (y -> 0.8+). So y is a proxy for apparent size/distance.
  - whether Gemini-3.1-Pro found it (complementary = inherently visible, RampNet-specific
    gap) or missed it too (HARD = plausibly resolution/occlusion-limited).
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import math
import os
import sys

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import (
    build_ground_truth, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
from compare import load_bundle, DetectionCache, cache_key
from detectors import build_detector

PRO = "gemini-3.1-pro-preview"
R = math.sqrt(radius_sq_for())          # match radius in scaled units (0.022*1024)

class Args:
    gemini_model = PRO; qwen_model = "Qwen/Qwen3-VL"; tiling = "perspective"


def d(p, q):
    return math.hypot((p[0] - q[0]) * PANO_SCALE_X, (p[1] - q[1]) * PANO_SCALE_Y)


def matched_gt(preds, gt_points):
    confs = [_confidence(p) for p in preds]
    order = (sorted(range(len(preds)), key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                    reverse=True) if any(c is not None for c in confs) else range(len(preds)))
    claimed, hit = [False] * len(gt_points), set()
    for i in order:
        p = _xy(preds[i]); best_k, best = -1, R
        for k, g in enumerate(gt_points):
            if claimed[k]:
                continue
            dist = d(p, g)
            if dist < best:
                best, best_k = dist, k
        if best_k >= 0:
            claimed[best_k] = True; hit.add(best_k)
    return hit


def pct(n, tot):
    return f"{n:4d} ({n/tot:5.1%})" if tot else f"{n:4d}   n/a"


def analyze(city):
    records, verdicts, _ = load_bundle(os.path.join(REPO, "benchmark", city))
    label, gem = build_detector("gemini", PRO, records, Args())
    sig, cache = gem.signature(), DetectionCache(os.path.join(REPO, ".model_cache"))

    misses, hit_ys, panos_with_miss, n_gt = [], [], {}, 0
    for pid, entry in verdicts.items():
        gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                entry["missed"], entry["no_missed"])
        if not gt.fn_confirmed:
            continue
        rp = [(x["x_normalized"], x["y_normalized"], x["confidence"])
              for x in records[pid]["detections"]]
        pro = cache.get(cache_key(label, sig, city, pid)) or []
        mr, mpro = matched_gt(rp, gt.gt_points), matched_gt(pro, gt.gt_points)
        n_gt += len(gt.gt_points)
        for i, g in enumerate(gt.gt_points):
            if i in mr:
                hit_ys.append(g[1]); continue
            near = min((d(g, _xy(p)) for p in rp), default=float("inf")) / R
            misses.append({"pid": pid, "y": g[1], "near": near, "pro": i in mpro})
            panos_with_miss[pid] = panos_with_miss.get(pid, 0) + 1
    return misses, hit_ys, n_gt, panos_with_miss


def report(city, misses, hit_ys, n_gt, panos_with_miss):
    n = len(misses)
    print(f"\n{'='*66}\n{city.upper()}  —  {n} RampNet misses of {n_gt} GT ramps "
          f"(recall {1-n/n_gt:.3f})\n{'='*66}")

    print("\n1) Did RampNet fire ANYWHERE near the missed ramp?")
    blind = [m for m in misses if m["near"] > 3]
    mid = [m for m in misses if 1.5 < m["near"] <= 3]
    near = [m for m in misses if m["near"] <= 1.5]
    print(f"   BLIND   (nearest det > 3x radius, saw nothing)  {pct(len(blind), n)}")
    print(f"   nearby  (1.5-3x radius)                         {pct(len(mid), n)}")
    print(f"   LOCALIZATION near-miss (<=1.5x radius)          {pct(len(near), n)}")

    print("\n2) Apparent size/distance proxy — vertical position (horizon=0.50; nearer 0.50 = farther/smaller)")
    def med(v):
        s = sorted(v); return s[len(s)//2] if s else float('nan')
    print(f"   median y of DETECTED ramps : {med(hit_ys):.3f}")
    print(f"   median y of MISSED   ramps : {med([m['y'] for m in misses]):.3f}")
    far_hits = sum(1 for y in hit_ys if y < 0.60)
    far_miss = sum(1 for m in misses if m["y"] < 0.60)
    print(f"   near-horizon (y<0.60, i.e. distant/small): "
          f"hits {far_hits}/{len(hit_ys)} ({far_hits/max(len(hit_ys),1):.1%})  vs  "
          f"misses {far_miss}/{n} ({far_miss/max(n,1):.1%})")

    print("\n3) Is the ramp inherently visible? (did Gemini-3.1-Pro find it?)")
    comp = [m for m in misses if m["pro"]]
    hard = [m for m in misses if not m["pro"]]
    print(f"   COMPLEMENTARY (Pro found it -> RampNet-specific gap) {pct(len(comp), n)}")
    print(f"   HARD          (Pro missed it too)                    {pct(len(hard), n)}")
    if hard:
        print(f"      hard misses: median y {med([m['y'] for m in hard]):.3f}, "
              f"blind-rate {sum(1 for m in hard if m['near']>3)/len(hard):.1%}")
    if comp:
        print(f"      complementary: median y {med([m['y'] for m in comp]):.3f}, "
              f"blind-rate {sum(1 for m in comp if m['near']>3)/len(comp):.1%}")

    print("\n4) Concentration")
    mx = max(panos_with_miss.values()) if panos_with_miss else 0
    print(f"   misses spread over {len(panos_with_miss)} panos; worst pano has {mx} misses")


all_m, all_h, all_gt = [], [], 0
for city in ("richmond", "bend"):
    m, h, g, pw = analyze(city)
    report(city, m, h, g, pw)
    all_m += m; all_h += h; all_gt += g

print(f"\n{'='*66}\nCOMBINED — {len(all_m)} misses of {all_gt} GT ramps\n{'='*66}")
n = len(all_m)
print(f"   BLIND (>3x radius)          {pct(sum(1 for m in all_m if m['near']>3), n)}")
print(f"   LOCALIZATION (<=1.5x)       {pct(sum(1 for m in all_m if m['near']<=1.5), n)}")
print(f"   HARD (Pro missed too)       {pct(sum(1 for m in all_m if not m['pro']), n)}")
print(f"   COMPLEMENTARY (Pro got it)  {pct(sum(1 for m in all_m if m['pro']), n)}")
print(f"   near-horizon y<0.60         {pct(sum(1 for m in all_m if m['y']<0.60), n)}")
