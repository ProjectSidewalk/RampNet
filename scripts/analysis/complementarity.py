"""RampNet vs Gemini-3.6-flash complementarity on richmond (read-only, from cache).

Answers issue #35's decision gate: do the two models miss *different* ramps?
For each GT ramp on recall-eligible panos, record whether RampNet found it, Gemini
found it, both, or neither -> oracle-union recall + the RampNet-miss n Gemini-hit set.
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import os, sys
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import (
    build_ground_truth, score_pano, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
from compare import load_bundle, DetectionCache, cache_key
from detectors import build_detector

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.6-flash"

class Args:
    gemini_model = MODEL; qwen_model = "Qwen/Qwen3-VL"; tiling = "perspective"

RSQ = radius_sq_for()

def matched_gt(preds, gt_points):
    """Greedy 1:1 match (mirrors score_pano); return the set of GT indices covered."""
    confs = [_confidence(p) for p in preds]
    order = (sorted(range(len(preds)), key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                    reverse=True) if any(c is not None for c in confs) else range(len(preds)))
    claimed, hit = [False] * len(gt_points), set()
    for i in order:
        pxn, pyn = _xy(preds[i]); px, py = pxn * PANO_SCALE_X, pyn * PANO_SCALE_Y
        best_k, best = -1, RSQ
        for k, (gx, gy) in enumerate(gt_points):
            if claimed[k]:
                continue
            d = (px - gx * PANO_SCALE_X) ** 2 + (py - gy * PANO_SCALE_Y) ** 2
            if d < best:
                best, best_k = d, k
        if best_k >= 0:
            claimed[best_k] = True; hit.add(best_k)
    return hit

records, verdicts, _ = load_bundle(os.path.join(REPO, "benchmark", "richmond"))
label, gem = build_detector("gemini", MODEL, records, Args())
sig, cache = gem.signature(), DetectionCache(os.path.join(REPO, ".model_cache"))

N = both = r_only = g_only = neither = 0
r_fp = g_fp = 0
panos = missing = 0
for pid, entry in verdicts.items():
    gt = build_ground_truth(records[pid]["detections"], entry["dets"], entry["missed"], entry["no_missed"])
    if not gt.fn_confirmed:
        continue
    gp = cache.get(cache_key(label, sig, "richmond", pid))
    if gp is None:
        missing += 1; continue
    rp = [(d["x_normalized"], d["y_normalized"], d["confidence"]) for d in records[pid]["detections"]]
    mr, mg = matched_gt(rp, gt.gt_points), matched_gt(gp, gt.gt_points)
    for i in range(len(gt.gt_points)):
        r, g = i in mr, i in mg
        both += r and g; r_only += r and not g; g_only += g and not r; neither += not r and not g
    N += len(gt.gt_points)
    r_fp += score_pano(rp, gt).fp
    g_fp += score_pano(gp, gt).fp
    panos += 1

r_tp, g_tp, union = both + r_only, both + g_only, both + r_only + g_only
rampnet_misses = g_only + neither
print(f"richmond complementarity — RampNet vs {MODEL}  ({panos} recall-eligible panos, {N} GT ramps"
      + (f"; {missing} panos missing from cache" if missing else "") + ")\n")
print(f"  RampNet recall     {r_tp/N:.3f}   ({r_tp}/{N})")
print(f"  Gemini recall      {g_tp/N:.3f}   ({g_tp}/{N})")
print(f"  ORACLE-UNION recall{union/N:.3f}   ({union}/{N})   <- ceiling if you could keep every right call")
print()
print(f"  found by BOTH        {both:4d}  ({both/N:.1%})")
print(f"  RampNet ONLY         {r_only:4d}  ({r_only/N:.1%})")
print(f"  Gemini  ONLY         {g_only:4d}  ({g_only/N:.1%})   <- complementary gain (RampNet-miss n Gemini-hit)")
print(f"  found by NEITHER     {neither:4d}  ({neither/N:.1%})   <- hard misses, no model helps")
print()
print(f"  Union recall lift over RampNet:  +{(union - r_tp)/N:.3f}  ({g_only} ramps)")
print(f"  Of RampNet's {rampnet_misses} misses, Gemini recovers {g_only} ({g_only/rampnet_misses:.0%}); {neither} nobody finds")
print(f"  FP cost on these panos:  RampNet {r_fp}  |  Gemini {g_fp}   (a naive union pays ~both)")
