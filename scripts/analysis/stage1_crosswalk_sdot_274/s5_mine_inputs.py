"""crosswalk_s5 (RampNet#274): turn Jon's Juneau mining verdicts into `export_stage2.py add-panos` inputs.

Rules (docs/stage1_crosswalk_sdot_274.md, "crosswalk_s5 pre-registration", fixed before training):
- "crosswalk" cards become sparse positives at the detection's point. Two positives on one pano closer than the
  scorer's RADIUS (0.05, train.py distance) are merged: the higher-confidence card stays a positive and the other
  becomes an ignore point. The scorer cannot tell them apart, and training two peaks on one wide crossing would
  teach duplicate firing.
- "not_crosswalk" cards become negatives, unless within 0.05 of any "crosswalk" card (train.py refuses those
  near a positive; a merged one is still a crossing) or of an
  "unsure" card (its disc might cover a crossing); those are dropped and counted.
- "unsure" cards within 0.05 of a kept positive become ignore points; elsewhere they are unsupervised anyway.
- `--overrides` (card_id, action, reason), from the review of every negative and positive before training:
  `drop` removes a negative, `ignore` turns a "crosswalk" card into an ignore point. Applied first.

Usage:
    python s5_mine_inputs.py --ratings <ratings_jon_..._juneauMine.csv> --out <dir>
writes positives.csv (label_uid,city,pano_id,x,y), negatives.csv (ratings rows, add-panos --negatives format),
ignore.csv (ignore_id,pano_id,x,y) and summary.json.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from export_stage2 import neg_dist

MERGE = 0.05   # score_274.RADIUS, in train.py's distance (pano widths, y halved, seam wrap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--city", default="juneau-ak")
    ap.add_argument("--overrides", help="CSV card_id,action(drop|ignore),reason")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.ratings, encoding="utf-8", newline="")))
    ov = {r["card_id"]: r["action"] for r in csv.DictReader(open(a.overrides, encoding="utf-8", newline=""))}         if a.overrides else {}
    by_id = {r["card_id"]: r for r in rows}
    for cid, act in ov.items():
        want = {"drop": "not_crosswalk", "ignore": "crosswalk"}[act]
        if by_id[cid]["verdict"] != want:
            raise SystemExit(f"override {cid} {act}: verdict is {by_id[cid]['verdict']}, expected {want}")
    n_ov = {"override_dropped_negatives": 0, "override_positives_to_ignore": 0}
    fields = list(rows[0])
    by = defaultdict(list)
    for r in rows:
        by[r["pano_id"]].append(r)
    pt = lambda r: (float(r["x"]), float(r["y"]))
    pos, neg, ign = [], [], []
    n = defaultdict(int)
    for pid in sorted(by):
        cards = by[pid]
        kept = []
        for r in sorted((r for r in cards if r["verdict"] == "crosswalk"), key=lambda r: -float(r["conf"])):
            if ov.get(r["card_id"]) == "ignore":
                ign.append({"ignore_id": r["card_id"], "pano_id": pid, "x": r["x"], "y": r["y"]})
                n_ov["override_positives_to_ignore"] += 1
            elif any(neg_dist(pt(r), pt(k)) < MERGE for k in kept):
                ign.append({"ignore_id": r["card_id"], "pano_id": pid, "x": r["x"], "y": r["y"]})
                n["positives_merged_to_ignore"] += 1
            else:
                kept.append(r)
        pos += [{"label_uid": f"{a.city}:s5mine:{r['card_id']}", "city": a.city, "pano_id": pid,
                 "x": r["x"], "y": r["y"]} for r in kept]
        unsure = [r for r in cards if r["verdict"] == "unsure"]
        for r in unsure:
            if any(neg_dist(pt(r), pt(k)) < MERGE for k in kept):
                ign.append({"ignore_id": r["card_id"], "pano_id": pid, "x": r["x"], "y": r["y"]})
                n["unsure_to_ignore"] += 1
        for r in cards:
            if r["verdict"] != "not_crosswalk":
                continue
            if ov.get(r["card_id"]) == "drop":
                n_ov["override_dropped_negatives"] += 1
                continue
            if any(neg_dist(pt(r), pt(k)) < MERGE for k in cards if k["verdict"] == "crosswalk"):
                n["negatives_dropped_near_positive"] += 1      # any confirmed crossing, merged ones included
            elif any(neg_dist(pt(r), pt(u)) < MERGE for u in unsure):
                n["negatives_dropped_near_unsure"] += 1
            else:
                neg.append({**r, "city": a.city})
    os.makedirs(a.out, exist_ok=True)
    for name, data, cols in (("positives.csv", pos, ["label_uid", "city", "pano_id", "x", "y"]),
                             ("negatives.csv", neg, fields), ("ignore.csv", ign, ["ignore_id", "pano_id", "x", "y"])):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    summ = {"cards": len(rows), "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in
                                             ("crosswalk", "not_crosswalk", "unsure")},
            "positives": len(pos), "negatives": len(neg), "ignore_points": len(ign), **n, **n_ov,
            "positive_panos": len({p["pano_id"] for p in pos}),
            "negative_only_panos": len({r["pano_id"] for r in neg} - {p["pano_id"] for p in pos})}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
