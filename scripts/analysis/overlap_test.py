"""Do the THRESHOLD and RESOLUTION levers target the same ramps?

Threshold recovers ramps the model saw weakly; resolution targets small/distant ones.
If distant ramps are exactly the weak-response ones, the two gains OVERLAP rather
than add. Records per-GT-ramp recovery at several thresholds, then cross-tabs by
DA3 metric depth.
"""
import os as _os, sys as _sys
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = _os.environ.get("RAMPNET_ANALYSIS_OUT", _os.path.join(REPO, "analysis_out"))
_os.makedirs(OUT, exist_ok=True)
DA3_SRC = _os.environ.get("DA3_SRC")  # path to Depth-Anything-3/src (see README)
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))
import numpy as np, torch
from threshold_sweep import load_model, load_bundle, heatmap_for, peaks_to_dets
from rampnet.detection_eval import (build_ground_truth, radius_sq_for,
                                    PANO_SCALE_X, PANO_SCALE_Y, _xy, _confidence)
R = math.sqrt(radius_sq_for())
THRS = [0.55, 0.35, 0.25, 0.15]

def d2(p, q):
    return math.hypot((p[0]-q[0])*PANO_SCALE_X, (p[1]-q[1])*PANO_SCALE_Y)

def matched(preds, gts):
    confs=[_confidence(p) for p in preds]
    order=sorted(range(len(preds)), key=lambda i: confs[i] if confs[i] is not None else -1e9, reverse=True)
    claimed=[False]*len(gts); hit=set()
    for i in order:
        p=_xy(preds[i]); bk,bd=-1,R
        for k,g in enumerate(gts):
            if claimed[k]: continue
            dd=d2(p,g)
            if dd<bd: bd,bk=dd,k
        if bk>=0: claimed[bk]=True; hit.add(bk)
    return hit

dev = torch.device("cuda"); model = load_model().to(dev)
out=[]
for city in ("richmond","bend"):
    records, verdicts, panos_dir = load_bundle(city)
    for n,(pid,entry) in enumerate(verdicts.items(),1):
        gt = build_ground_truth(records[pid]["detections"], entry["dets"], entry["missed"], entry["no_missed"])
        if not gt.fn_confirmed or not gt.gt_points: continue
        h = heatmap_for(model, dev, os.path.join(panos_dir, f"{pid}.jpg"), False)
        hits = {t: matched(peaks_to_dets(h,t,10), gt.gt_points) for t in THRS}
        for i,g in enumerate(gt.gt_points):
            out.append({"city":city,"pid":pid,"x":round(g[0],6),"y":round(g[1],6),
                        **{f"t{t}": (i in hits[t]) for t in THRS}})
        del h
        if n%40==0: print(f"  {city} {n}/{len(verdicts)}", flush=True)
json.dump(out, open(os.path.join(OUT,"overlap.json"),"w"))
print("wrote", len(out))
