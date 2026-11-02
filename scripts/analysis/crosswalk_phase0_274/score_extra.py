"""Companion reads for the Vistas crosswalk probe (RampNet#274 item 3), all offline from out/components.

1. Chance floor: the pinned recall@0.05 scores a label by the nearest merged component on ITS pano; with
   several merged components per pano a label can be matched by chance. The floor is the same recall with
   each labeled pano's labels scored against the merged components of a DIFFERENT labeled pano of the same
   split (every derangement sampled, seed 274, 200 rounds): mean and p95.
2. Sensitivity: recall@0.05 after dropping components by mean class probability (< 0.4, < 0.5) and after
   dropping components larger than 5% of a view (the road-surface blobs), each separately -- exploratory,
   never the reading rule.
3. For matched labels, how many views the merged component spanned (a seam-straddling crossing merges).

Usage: python score_extra.py --probe-out ./out --rawlabels ../census/out/rawlabels --rampnet D:/Git/RampNet
       --split paterson=paterson-nj --split gainesville=gainesville-fl
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vistas_crosswalk_probe import load_labels, merge_components, pano_dist, MATCH_R  # noqa: E402


def recall(label_sets, comp_sets, r=MATCH_R):
    hit = tot = 0
    for labs, merged in zip(label_sets, comp_sets):
        for l in labs:
            tot += 1
            if any(pano_dist(m["x"], m["y"], l["x"], l["y"]) <= r for m in merged):
                hit += 1
    return hit / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-out", default="out")
    ap.add_argument("--rawlabels", required=True)
    ap.add_argument("--rampnet", required=True)
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=200)
    a = ap.parse_args()
    out = {}
    all_labels, all_raw = [], []
    for s in a.split:
        split, city = s.split("=", 1)
        pdir = os.path.join(a.rampnet, "benchmark", split, "panos")
        ids = {os.path.splitext(f)[0] for f in os.listdir(pdir)}
        labels = load_labels(a.rawlabels, city, ids)
        cdir = os.path.join(a.probe_out, "components", split)
        recs = {}
        for pid in labels:
            p = os.path.join(cdir, pid + ".json")
            if os.path.exists(p):
                recs[pid] = json.load(open(p))["components"]
        pids = sorted(recs)
        label_sets = [labels[p] for p in pids]
        raw_sets = [recs[p] for p in pids]
        all_labels += label_sets
        all_raw += raw_sets
        out[split] = {"n_panos": len(pids), "n_labels": sum(len(x) for x in label_sets)}
    merged_sets = [merge_components(c) for c in all_raw]
    res = {"pinned_recall@0.05": recall(all_labels, merged_sets),
           "n_labels": sum(len(x) for x in all_labels), "n_panos": len(merged_sets)}
    # The census's human-only margin rule confirms none of these labels (they carry AI-validator votes
    # only), so the server's own `correct` flag (AI-validator dominated) is the only available cut.
    sc = [[l for l in labs if l["server_correct"] == "true"] for labs in all_labels]
    res["server_correct_true"] = {"n_labels": sum(len(x) for x in sc), "recall@0.05": recall(sc, merged_sets)}
    sf = [[l for l in labs if l["server_correct"] == "false"] for labs in all_labels]
    res["server_correct_false"] = {"n_labels": sum(len(x) for x in sf), "recall@0.05": recall(sf, merged_sets)}
    rng = random.Random(274)
    n = len(merged_sets)
    floors = []
    for _ in range(a.rounds):
        perm = list(range(n))
        while True:
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                break
        floors.append(recall(all_labels, [merged_sets[perm[i]] for i in range(n)]))
    res["chance_floor_recall@0.05"] = {"mean": round(statistics.mean(floors), 3),
                                      "p95": round(sorted(floors)[int(0.95 * len(floors)) - 1], 3),
                                      "rounds": a.rounds}
    sens = {}
    for name, keep in (("prob>=0.4", lambda c: c["mean_prob"] >= 0.4),
                       ("prob>=0.5", lambda c: c["mean_prob"] >= 0.5),
                       ("area<=5%view", lambda c: c["area_px"] <= 0.05 * 1024 * 1024),
                       ("prob>=0.4&area<=5%view", lambda c: c["mean_prob"] >= 0.4 and c["area_px"] <= 0.05 * 1024 * 1024)):
        ms = [merge_components([c for c in raw if keep(c)]) for raw in all_raw]
        sens[name] = {"recall@0.05": round(recall(all_labels, ms), 3),
                      "merged_per_pano": round(sum(len(m) for m in ms) / max(1, n), 2)}
    res["sensitivity"] = sens
    res["merged_per_labeled_pano"] = round(sum(len(m) for m in merged_sets) / max(1, n), 2)
    n_views = []
    for labs, merged in zip(all_labels, merged_sets):
        for l in labs:
            best = min(merged, key=lambda m: pano_dist(m["x"], m["y"], l["x"], l["y"]), default=None)
            if best and pano_dist(best["x"], best["y"], l["x"], l["y"]) <= MATCH_R:
                n_views.append(best["n_views"])
    res["matched_component_n_views"] = {str(k): n_views.count(k) for k in sorted(set(n_views))}
    res["splits"] = out
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.probe_out, "summary_extra.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
