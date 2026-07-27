"""Would Euclidean peak-NMS at the match radius help? No — checked, it hurts.

Issue #62 proposed suppressing extracted peaks that sit closer together than
the match radius (min_distance=10 heatmap px vs R = 0.022*1024 = 22.5 px), on
the theory that the matcher can score at most one of them. That theory silently
assumes only one GT ramp is nearby. This script checks the committed benchmark
records and finds the opposite:

  - five of six splits contain detection pairs inside one match radius
    (8 pairs across the reviewed cities, 63 in manual_gold);
  - reviewers confirmed BOTH members real on 5 of the 8 reviewed pairs —
    adjacent real ramps genuinely sit 0.67-0.83 R apart in pano space;
  - real and junk pairs occupy the same 15-19 px separation band, so no
    suppression radius can split them;
  - rescoring with NMS applied lowers F1 on four of six splits (bend, whose
    one redundant FP motivated #62, is the only split it helps) and costs
    manual_gold 33 TPs to remove 8 FPs at the deployed 0.55 operating point.

Conclusion: min_distance < R is load-bearing headroom for real adjacent ramps,
not a defect. The matcher already charges redundant second hits as false
positives, which is the right accounting. Full write-up in the #62 discussion.

Pure CPU, no model rerun: NMS only prunes the recorded peak set, and
keep-highest NMS commutes with threshold_abs (a suppressor always outscores
the peak it suppresses, so it survives any threshold the suppressed one does),
so applying it to records.jsonl reproduces exactly what
(threshold, min_distance=10, +NMS-at-R) extraction would emit.

    python scripts/analysis/peak_nms_check.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (
    PANO_SCALE_X, PANO_SCALE_Y, aggregate, build_ground_truth,
    load_yolo_ground_truths, radius_sq_for, score_pano)

VERDICT_CITIES = ("richmond", "bend", "clovis", "morgantown", "budapest_district5")
RSQ = radius_sq_for()
SX, SY = PANO_SCALE_X, PANO_SCALE_Y


def suppress_duplicate_peaks(peaks, radius_sq, scale_x, scale_y):
    """Keep-highest greedy Euclidean NMS over (x_norm, y_norm, confidence) peaks.

    Deliberately local to this script, not in rampnet/: the whole point of the
    analysis is that nothing in the production pipeline should apply it.
    Returns (kept peaks in input order, indices of the pruned peaks).
    """
    order = sorted(range(len(peaks)), key=lambda i: (-peaks[i][2], i))
    kept_scaled, kept_idx = [], set()
    for i in order:
        x, y = peaks[i][0] * scale_x, peaks[i][1] * scale_y
        if all((x - kx) ** 2 + (y - ky) ** 2 >= radius_sq for kx, ky in kept_scaled):
            kept_scaled.append((x, y))
            kept_idx.add(i)
    return ([p for i, p in enumerate(peaks) if i in kept_idx],
            [i for i in range(len(peaks)) if i not in kept_idx])


def load_records(city):
    recs = {}
    with open(os.path.join(REPO, "benchmark", city, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pano"]["panorama_id"]] = r
    return recs


def load_verdicts(city):
    with open(os.path.join(REPO, "benchmark", city, "verdicts.json"), encoding="utf-8") as f:
        return json.load(f)["panos"]


def dets_as_tuples(rec):
    return [(float(d["x_normalized"]), float(d["y_normalized"]), float(d["confidence"]))
            for d in rec["detections"]]


def report(name, scores):
    rep = aggregate(scores)
    ap = f" AP={rep.ap:.4f}" if rep.ap is not None else ""
    print(f"  {name:36s} P={rep.precision:.4f} R={rep.recall:.4f} F1={rep.f1:.4f} "
          f"tp={rep.tp} fp={rep.fp} fn={rep.fn}{ap}")


def main():
    print("=== (a) what NMS at the match radius would prune, and the reviewers' verdicts ===")
    for city in VERDICT_CITIES:
        recs, verdicts = load_records(city), load_verdicts(city)
        for pid, entry in verdicts.items():
            dets = dets_as_tuples(recs[pid])
            _, pruned = suppress_duplicate_peaks(dets, RSQ, SX, SY)
            for i in pruned:
                print(f"  {city:18s} {pid}  pruned det #{i} conf={dets[i][2]:.2f} "
                      f"verdict={entry['dets'][i]!r}")

    print("\n=== (b) reviewed cities, scored as committed vs with NMS ===")
    for city in VERDICT_CITIES:
        recs, verdicts = load_records(city), load_verdicts(city)
        for tag, use_nms in (("as committed", False), ("with NMS", True)):
            scores = []
            for pid, entry in verdicts.items():
                gt = build_ground_truth(recs[pid]["detections"], entry["dets"],
                                        entry["missed"], entry["no_missed"])
                preds = dets_as_tuples(recs[pid])
                if use_nms:
                    preds, _ = suppress_duplicate_peaks(preds, RSQ, SX, SY)
                scores.append(score_pano(preds, gt))
            report(f"{city} {tag}", scores)

    print("\n=== (c) manual_gold (YOLO GT), deployed 0.55 operating point and full sweep ===")
    gts = load_yolo_ground_truths(os.path.join(REPO, "manual_labels"))
    recs = load_records("manual_gold")
    for thr in (0.55, 0.0):
        for tag, use_nms in (("as committed", False), ("with NMS", True)):
            scores, n_pruned = [], 0
            for pid, rec in recs.items():
                preds = [p for p in dets_as_tuples(rec) if p[2] >= thr]
                if use_nms:
                    preds, pruned = suppress_duplicate_peaks(preds, RSQ, SX, SY)
                    n_pruned += len(pruned)
                scores.append(score_pano(preds, gts[pid]))
            report(f"manual_gold thr>={thr} {tag}", scores)
            if use_nms:
                print(f"    ({n_pruned} detections pruned at this threshold)")


if __name__ == "__main__":
    main()
