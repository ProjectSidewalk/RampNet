"""Ground truth, precision, recall and F1 for RampNet, one row per city.

Two kinds of row, scored the way each kind is scored everywhere else in the repo:

* **Benchmark cities** (``benchmark/<split>/verdicts.json``): ground truth is the reviewer's
  confirmed detections plus their confident missed-ramp marks; ``unsure`` marks abstain.
  Precision, recall and the ground-truth count come from ``rampnet.validation.collect``,
  the scorer behind the table in ``benchmark/README.md``, on the deployed records
  (confidence >= 0.55). Ground truth here is RampNet-anchored (see that README).
* **Manual gold** (``manual_labels/*.txt``, issue #58): independent full manual labels, split
  by city from each panorama's location. RampNet's ``benchmark/manual_gold/records.jsonl``
  detections at confidence >= 0.55 are matched with ``rampnet.detection_eval.score_pano``
  (radius 0.022, x wraps), the matcher behind ``docs/model_comparison.md``.

A city split with no ``verdicts.json`` (juneau, at the time of writing) is listed as not
reviewed rather than silently left out.

    python scripts/analysis/ground_truth_by_city.py                 # table + CSV
    python scripts/analysis/ground_truth_by_city.py --csv out.csv
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from rampnet.detection_eval import aggregate, load_yolo_ground_truths, score_pano  # noqa: E402
from rampnet.validation import collect, total_ramps  # noqa: E402

DEPLOYED_T = 0.55
DEFAULT_CSV = os.path.join(REPO, "docs", "talks", "uchicago_2026", "ground_truth_by_city.csv")

# (split, city, imagery), west to east within the US, then abroad.
CITIES = [
    ("bend", "Bend, OR, US", "Google Street View"),
    ("clovis", "Clovis, NM, US", "Mapillary 360 (GoPro Fusion)"),
    ("laurens_gsv", "Laurens, IA, US", "Google Street View"),
    ("laurens_mapillary", "Laurens, IA, US", "Mapillary 360 (GoPro Max)"),
    ("gainesville", "Gainesville, FL, US", "Google Street View"),
    ("morgantown", "Morgantown, WV, US", "Mapillary 360 (GoPro Max)"),
    ("richmond", "Richmond, VA, US", "Mapillary 360 (iSTAR Pulsar + GoPro Max)"),
    ("annapolis", "Annapolis, MD, US", "Mapillary 360 (Trimble MX7)"),
    ("paterson", "Paterson, NJ, US", "Google Street View"),
    ("sao_paulo", "São Paulo, Brazil", "Google Street View"),
    ("budapest_district5", "Budapest, Hungary", "Mapillary 360 (GoPro Max)"),
    ("bayonne", "Bayonne, France", "Panoramax (GoPro Max)"),
    ("juneau", "Juneau, AK, US", "Mapillary 360"),
]

FIELDS = ["set", "split", "city", "imagery", "panoramas", "gt_curb_ramps", "panoramas_no_ramps",
          "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1",
          "note"]


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def _records(split):
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def benchmark_row(split, city, imagery):
    row = dict(set="benchmark", split=split, city=city, imagery=imagery)
    path = os.path.join(REPO, "benchmark", split, "verdicts.json")
    if not os.path.exists(path):
        row.update(panoramas=len(_records(split)), note="ground truth not reviewed yet")
        return row
    with open(path, encoding="utf-8") as f:
        panos = json.load(f)["panos"]
    confs = {r["pano"]["panorama_id"]: [d["confidence"] for d in r["detections"]]
             for r in _records(split)}
    pools = collect(panos, confs)
    gt = total_ramps(pools)
    tp_r = sum(1 for _, ok in pools.recall_judged if ok)
    tp_p = sum(1 for _, ok in pools.judged if ok)
    fp = len(pools.judged) - tp_p
    no_ramps = sum(
        1 for e in panos.values()
        if not any(d is True for d in e["dets"])
        and not any(not m.get("unsure") for m in e["missed"]))
    p, r = tp_p / len(pools.judged), tp_r / gt
    row.update(panoramas=len(panos), gt_curb_ramps=gt, panoramas_no_ramps=no_ramps,
               true_positives=tp_p, false_positives=fp, false_negatives=pools.missed_total,
               precision=p, recall=r, f1=_f1(p, r), _tp_recall=tp_r,
               note="reviewer-confirmed (RampNet-anchored) ground truth")
    return row


def _gold_city(lat, lng):
    # The gold set is the HF test split of the three training cities (benchmark/README.md).
    if lng > -80:
        return "New York, NY, US"
    return "Bend, OR, US" if lat < 45 else "Portland, OR, US"


def manual_gold_rows():
    gts = load_yolo_ground_truths(os.path.join(REPO, "manual_labels"))
    by_city = {}
    for rec in _records("manual_gold"):
        pid = rec["pano"]["panorama_id"]
        city = _gold_city(*rec["pano"]["pano_coord"])
        preds = [d for d in rec["detections"] if d["confidence"] >= DEPLOYED_T]
        by_city.setdefault(city, []).append((score_pano(preds, gts[pid]), len(gts[pid].gt_points)))
    if sum(len(v) for v in by_city.values()) != len(gts):
        sys.exit("manual_gold records and manual_labels disagree on the panorama set")
    rows = []
    for city in ("New York, NY, US", "Portland, OR, US", "Bend, OR, US"):
        scores = [s for s, _ in by_city[city]]
        rep = aggregate(scores)
        rows.append(dict(set="manual_gold", split="manual_gold", city=city,
                         imagery="Google Street View", panoramas=len(scores),
                         gt_curb_ramps=sum(n for _, n in by_city[city]),
                         panoramas_no_ramps=sum(1 for _, n in by_city[city] if n == 0),
                         true_positives=rep.tp, false_positives=rep.fp, false_negatives=rep.fn,
                         precision=rep.precision, recall=rep.recall, f1=rep.f1,
                         _tp_recall=rep.tp,
                         note="independent manual labels; training city (in-distribution)"))
    return rows


def total_row(rows, label):
    scored = [r for r in rows if "precision" in r]
    tp = sum(r["true_positives"] for r in scored)
    fp = sum(r["false_positives"] for r in scored)
    fn = sum(r["false_negatives"] for r in scored)
    gt = sum(r["gt_curb_ramps"] for r in scored)
    # Recall's TP pool can differ from precision's (panos whose missed-ramp check is
    # unconfirmed), so pool each row's own recall numerator.
    tp_r = sum(r["_tp_recall"] for r in scored)
    p, rr = tp / (tp + fp), tp_r / gt
    return dict(set=label, split="", city=f"Total ({label})", imagery="",
                panoramas=sum(r["panoramas"] for r in scored), gt_curb_ramps=gt,
                panoramas_no_ramps=sum(r["panoramas_no_ramps"] for r in scored),
                true_positives=tp, false_positives=fp, false_negatives=fn,
                precision=p, recall=rr, f1=_f1(p, rr), note="pooled over the rows above")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args(argv)
    bench = [benchmark_row(*c) for c in CITIES]
    gold = manual_gold_rows()
    rows = bench + [total_row(bench, "benchmark")] + gold + [total_row(gold, "manual_gold")]
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        if "precision" in r:
            print(f"{r['city']:<22} {r['split']:<19} {r['panoramas']:>5} {r['gt_curb_ramps']:>5} "
                  f"{r['panoramas_no_ramps']:>4}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
                  f"F1 {r['f1']:.3f}")
        else:
            print(f"{r['city']:<22} {r['split']:<19} {r['panoramas']:>5}  {r['note']}")
    print("wrote", os.path.relpath(args.csv, REPO))


if __name__ == "__main__":
    main()
