"""Is culling distant DETECTIONS worth it? -> precision as a function of distance.

recall-by-distance told us where ramps are MISSED. This asks the complementary
question about RampNet's own detections: are far ones more likely to be false
positives? If precision holds up at distance, culling only destroys recall.

Distance uses the flat-ground estimate (validated against DA3 depth to within
6.5-8.5%), so this needs no GPU.
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import math, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
from rampnet.detection_eval import (
    build_ground_truth, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
from compare import load_bundle

R = math.sqrt(radius_sq_for())
CAM_H = 2.5
BUCKETS = [(0, 8), (8, 12), (12, 18), (18, 25), (25, 40), (40, 1e9)]


def gdist(y):
    dep = (y - 0.5) * math.pi
    return CAM_H / math.tan(dep) if dep > 1e-4 else 999.0


def d2(p, q):
    return math.hypot((p[0] - q[0]) * PANO_SCALE_X, (p[1] - q[1]) * PANO_SCALE_Y)


rows = []
for city in ("richmond", "bend"):
    records, verdicts, _ = load_bundle(os.path.join(REPO, "benchmark", city))
    for pid, entry in verdicts.items():
        gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                entry["missed"], entry["no_missed"])
        preds = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                 for d in records[pid]["detections"]]
        order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
        claimed = [False] * len(gt.gt_points)
        for i in order:
            p = _xy(preds[i]); bk, bd = -1, R
            for k, g in enumerate(gt.gt_points):
                if claimed[k]:
                    continue
                dd = d2(p, g)
                if dd < bd:
                    bd, bk = dd, k
            if bk >= 0:
                claimed[bk] = True
                rows.append((gdist(p[1]), "TP"))
            else:
                ign = any(d2(p, q) < R for q in gt.ignore_points)
                rows.append((gdist(p[1]), "IGN" if ign else "FP"))

print("=" * 66)
print("PRECISION vs DISTANCE — RampNet's own detections (thr 0.55)")
print("=" * 66)
print(f"{'distance':>12} {'dets':>7} {'TP':>6} {'FP':>6} {'PRECISION':>11}")
print("-" * 66)
for lo, hi in BUCKETS:
    b = [r for r in rows if lo <= r[0] < hi and r[1] != "IGN"]
    if not b:
        continue
    tp = sum(1 for r in b if r[1] == "TP"); fp = len(b) - tp
    lab = f"{lo}-{int(hi)}m" if hi < 1e8 else f"{lo}m+"
    print(f"{lab:>12} {len(b):>7} {tp:>6} {fp:>6} {tp/len(b):>11.3f}")
tot = [r for r in rows if r[1] != "IGN"]
tp = sum(1 for r in tot if r[1] == "TP")
print(f"{'ALL':>12} {len(tot):>7} {tp:>6} {len(tot)-tp:>6} {tp/len(tot):>11.3f}")

print("\nIf we CULLED every detection beyond a cutoff:")
for cut in (12, 18, 25):
    keep = [r for r in tot if r[0] < cut]
    lost_tp = sum(1 for r in tot if r[0] >= cut and r[1] == "TP")
    rm_fp = sum(1 for r in tot if r[0] >= cut and r[1] == "FP")
    ktp = sum(1 for r in keep if r[1] == "TP")
    print(f"  cull >{cut}m: precision {tp/len(tot):.3f} -> {ktp/len(keep):.3f}   "
          f"but LOSES {lost_tp} true ramps to remove only {rm_fp} false ones")
