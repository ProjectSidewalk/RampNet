"""Stage 1 label agreement against the 1,000-panorama gold set, re-scored (#172, #18).

The paper reports the Stage 1 dataset's agreement with ``manual_labels/`` as precision
0.9403 / recall 0.9245 (``docs/data/rampnet1_stage1_run/stage1_evaluation_results.txt``).
That evaluator (tag ``v1.0-iccv2025``) differed from the matcher every other RampNet
evaluator now uses (``rampnet.metrics.greedy_match``) in two ways:

1. a generated point within radius of an already-claimed ramp was **ignored** (neither TP
   nor FP), where the shared matcher counts it as a false positive (#18);
2. a point in range of several ramps claimed the **first in file order**, where the shared
   matcher claims the **nearest unclaimed** one. This moves recall as well as precision,
   because it changes which ramps end up claimed (#18, closing note).

Counting the 119 ignored points as false positives on the committed output gives 0.9121
(the redundant-only figure). That was once read as an *upper bound* on the corrected
precision; the corrected figure (0.9152) is above it, because nearest-unclaimed matching
mostly turns redundant points into true positives rather than adding false positives. It
is not a bound in either direction: the matching change gains 13 TP and loses 1 across 14
panoramas (``matching_change`` in the result). This script produces the corrected figure by
scoring the same 1,000 panoramas under four conventions on identical inputs:

  published            first-in-order claim, redundant points ignored   (the v1.0 evaluator)
  published_redund_fp  first-in-order claim, redundant points as FP     (the 0.9121 redundant-only arithmetic)
  shared_nowrap        nearest unclaimed, redundant as FP, x not cyclic (rampnet.metrics at #33)
  shared               nearest unclaimed, redundant as FP, x cyclic     (the current
                       stage_one/dataset_evaluation/evaluate.py and the benchmark; #132)

``shared`` is the corrected number of record. Every convention uses the pano radius 0.022
in the 1024x512 matcher space (``rampnet.detection_eval``), the gold points in file order,
and the generated points in the order the dataset stores them. The shared conventions call
``rampnet.metrics.match_points``, the entry point ``stage_one/dataset_evaluation/evaluate.py``
uses, and generated points outside [0, 1] are dropped as that evaluator drops them (none are).

Stage 1 points carry no confidence, so the shared matcher runs in stored order and that order
is arbitrary. ``order_sensitivity`` in the result measures how much it matters: TP over
seeded shuffles of each panorama's points, and the maximum one-to-one matching (the most TP
any order could give).

Inputs
------
* ``manual_labels/*.txt`` (committed): the gold boxes, reduced to centres.
* The Stage 1 points of the same 1,000 panoramas, the ``curb_ramp_points_normalized``
  column of the published dataset's test split (``projectsidewalk/rampnet-dataset``),
  fetched once by ``fetch`` with a pinned revision and committed as
  ``analysis_out/stage1_agreement_172/stage1_gold_labels.json`` (~280 KB; only the two
  label columns are read, over HTTP range requests, not the 44 GB of images).

Run, from the repo root::

    python scripts/analysis/stage1_agreement_172.py fetch   # once; network; writes the labels file
    python scripts/analysis/stage1_agreement_172.py score   # CPU, seconds; writes the result

``score`` writes ``analysis_out/stage1_agreement_172.json`` and
``analysis_out/stage1_agreement_172/summary.md`` deterministically (rounded floats, LF), and
``tests/test_stage1_agreement_172.py`` re-derives the JSON from the committed labels file
byte for byte and asserts the corrected counts as hard-coded constants, so a drift in the
matcher that moves this number fails CI even if both files are regenerated.
"""
import argparse
import hashlib
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rampnet.detection_eval import PANO_RADIUS_NORMALIZED, PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from rampnet.geometry import fold  # noqa: E402
from rampnet.metrics import match_points  # noqa: E402

HF_DATASET = "projectsidewalk/rampnet-dataset"
# The dataset revision `fetch` reads. The published dataset is immutable, so a re-fetch at
# this revision returns the same labels; it is recorded in the labels file as well.
HF_REVISION = "ee882e3f3c779dc13182f307bca616e50d9b8c5c"
LABELS_DIR = os.path.join(REPO, "manual_labels")
OUT_DIR = os.path.join(REPO, "analysis_out", "stage1_agreement_172")
LABELS_JSON = os.path.join(OUT_DIR, "stage1_gold_labels.json")
RESULT_JSON = os.path.join(REPO, "analysis_out", "stage1_agreement_172.json")
SUMMARY_MD = os.path.join(OUT_DIR, "summary.md")

PUBLISHED = {"precision": 0.9403, "recall": 0.9245}   # the paper, and README §"Evaluating the Results"
REDUNDANT_ONLY_PRECISION = 0.9121                      # stage1_generation_cost.md, the redundant-as-FP arithmetic
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 172
ORDER_SHUFFLES = 200
ORDER_SEED = 172

CONVENTIONS = ["published", "published_redund_fp", "shared_nowrap", "shared"]


# --------------------------------------------------------------------------- #
# Pure core (no I/O)
# --------------------------------------------------------------------------- #
def match_first_in_order(pred_points, gt_points, radius_sq, scale_x=PANO_SCALE_X,
                         scale_y=PANO_SCALE_Y):
    """The v1.0-iccv2025 Stage 1 evaluator's matching loop, verbatim in behaviour.

    Each prediction, in input order, claims the **first** unclaimed ground-truth point in
    file order strictly within ``radius`` (not the nearest). A prediction in range only of
    already-claimed points is *ignored*; one in range of nothing is a false positive. The x
    axis is not cyclic. Returns ``(tp, fp, ignored)``.
    """
    claimed = [False] * len(gt_points)
    tp = fp = ignored = 0
    for x_norm, y_norm in pred_points:
        px, py = x_norm * scale_x, y_norm * scale_y
        is_tp = saw_any = False
        for k, (gx, gy) in enumerate(gt_points):
            dx, dy = px - gx * scale_x, py - gy * scale_y
            if dx * dx + dy * dy < radius_sq:
                saw_any = True
                if not claimed[k]:
                    claimed[k] = True
                    is_tp = True
                    break
        if is_tp:
            tp += 1
        elif saw_any:
            ignored += 1
        else:
            fp += 1
    return tp, fp, ignored


def score_pano(pred_points, gt_points, convention, radius_sq=None):
    """``(tp, fp, ignored, n_gt)`` for one pano under a convention (see module docstring)."""
    if radius_sq is None:
        radius_sq = radius_sq_for(PANO_RADIUS_NORMALIZED)
    if convention in ("published", "published_redund_fp"):
        tp, fp, ignored = match_first_in_order(pred_points, gt_points, radius_sq)
        if convention == "published_redund_fp":
            fp, ignored = fp + ignored, 0
    elif convention in ("shared_nowrap", "shared"):
        # rampnet.metrics.match_points, as stage_one/dataset_evaluation/evaluate.py calls it.
        # Its n_redundant is a subset of fp, not a third outcome, so ignored stays 0.
        tp, fp, _n_redundant = match_points(pred_points, gt_points, radius_sq, PANO_SCALE_X,
                                            PANO_SCALE_Y, wrap_x=(convention == "shared"))
        ignored = 0
    else:
        raise ValueError(f"unknown convention {convention!r}")
    return tp, fp, ignored, len(gt_points)


def max_matching_tp(pred_points, gt_points, radius_sq=None, wrap_x=True,
                    scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """The largest number of one-to-one prediction-ramp pairs within radius, over any order.

    Maximum bipartite matching (Kuhn's augmenting paths; a panorama holds at most a few
    dozen points). It is the most TP any ordering of ``pred_points`` could give under the
    shared matcher's radius, so it bounds the order sensitivity from above.
    """
    if radius_sq is None:
        radius_sq = radius_sq_for(PANO_RADIUS_NORMALIZED)
    adj = []
    for x_norm, y_norm in pred_points:
        px, py = x_norm * scale_x, y_norm * scale_y
        row = []
        for k, (gx, gy) in enumerate(gt_points):
            dx = px - gx * scale_x
            if wrap_x:
                dx = fold(dx, scale_x)
            dy = py - gy * scale_y
            if dx * dx + dy * dy < radius_sq:
                row.append(k)
        adj.append(row)
    owner = [-1] * len(gt_points)

    def augment(i, seen):
        for k in adj[i]:
            if not seen[k]:
                seen[k] = True
                if owner[k] < 0 or augment(owner[k], seen):
                    owner[k] = i
                    return True
        return False

    return sum(1 for i in range(len(adj)) if augment(i, [False] * len(gt_points)))


def order_sensitivity(gold, stage1, ids, n=ORDER_SHUFFLES, seed=ORDER_SEED):
    """TP of the ``shared`` convention under seeded per-pano shuffles, and the max matching.

    One ``random.Random(seed)`` drives all ``n`` passes; each pass shuffles every panorama's
    generated points independently. Returns the stored-order TP, the shuffle min / max, and
    the maximum-matching TP with its precision and recall (the number of points is fixed, so
    FP = n_pred - TP).
    """
    rng = random.Random(seed)
    tps = []
    for _ in range(n):
        tp = 0
        for pid in ids:
            pts = list(stage1[pid])
            rng.shuffle(pts)
            tp += score_pano(pts, gold[pid], "shared")[0]
        tps.append(tp)
    stored = sum(score_pano(stage1[pid], gold[pid], "shared")[0] for pid in ids)
    best = sum(max_matching_tp(stage1[pid], gold[pid]) for pid in ids)
    n_pred = sum(len(stage1[pid]) for pid in ids)
    n_gt = sum(len(gold[pid]) for pid in ids)
    p, r, _ = prf(best, n_pred - best, n_gt)
    return {"n_shuffles": n, "seed": seed, "unit": "points within each panorama",
            "tp_stored_order": stored, "tp_shuffle_min": min(tps), "tp_shuffle_max": max(tps),
            "tp_max_matching": best, "precision_max_matching": p, "recall_max_matching": r}


def matching_change(gold, stage1, ids):
    """Per-pano TP difference, ``published_redund_fp`` -> ``shared``: the matching rule alone.

    Both conventions count redundant points as FP, so the difference is first-in-order versus
    nearest-unclaimed (plus the seam wrap, which moves nothing here). Reports how many
    panoramas change, and the TP gained and lost; the net is what moves the totals.
    """
    gained = lost = changed = 0
    for pid in ids:
        a = score_pano(stage1[pid], gold[pid], "published_redund_fp")
        b = score_pano(stage1[pid], gold[pid], "shared")
        if a != b:
            changed += 1
        d = b[0] - a[0]
        gained += max(d, 0)
        lost += max(-d, 0)
    return {"n_panos_changed": changed, "tp_gained": gained, "tp_lost": lost,
            "tp_net": gained - lost}


def prf(tp, fp, n_gt):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate(per_pano):
    """Totals and P/R/F1 from ``[(tp, fp, ignored, n_gt), ...]``."""
    tp = sum(r[0] for r in per_pano)
    fp = sum(r[1] for r in per_pano)
    ignored = sum(r[2] for r in per_pano)
    n_gt = sum(r[3] for r in per_pano)
    precision, recall, f1 = prf(tp, fp, n_gt)
    return {"tp": tp, "fp": fp, "ignored": ignored, "fn": n_gt - tp, "n_gt": n_gt,
            "n_pred": tp + fp + ignored, "precision": precision, "recall": recall, "f1": f1}


def bootstrap_ci(per_pano, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Pano-clustered percentile 95% CI on precision and recall (resample panoramas)."""
    rng = random.Random(seed)
    m = len(per_pano)
    ps, rs = [], []
    for _ in range(n):
        sample = [per_pano[rng.randrange(m)] for _ in range(m)]
        tp = sum(r[0] for r in sample)
        fp = sum(r[1] for r in sample)
        n_gt = sum(r[3] for r in sample)
        p, r, _ = prf(tp, fp, n_gt)
        ps.append(p)
        rs.append(r)
    ps.sort()
    rs.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return {"precision": [ps[lo], ps[hi]], "recall": [rs[lo], rs[hi]]}


def score_all(gold, stage1, conventions=CONVENTIONS):
    """Score every pano present in both sources under each convention.

    Panoramas are taken in sorted id order; gold points in file order; generated points in
    stored order. Returns ``{convention: {...totals, ci}}`` plus the population counts.
    """
    ids = sorted(pid for pid in gold if pid in stage1)
    out = {"n_panos": len(ids), "n_gold_only": len(set(gold) - set(stage1)),
           "n_negative_panos": sum(1 for pid in ids if not gold[pid]), "conventions": {}}
    for conv in conventions:
        per_pano = [score_pano(stage1[pid], gold[pid], conv) for pid in ids]
        agg = aggregate(per_pano)
        agg["ci95"] = bootstrap_ci(per_pano)
        out["conventions"][conv] = agg
    out["matching_change"] = matching_change(gold, stage1, ids)
    out["order_sensitivity"] = order_sensitivity(gold, stage1, ids)
    return out


def rounded(obj, nd=6):
    """Round every float for a byte-stable JSON (numpy-free here, but the rule is the repo's)."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, nd) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_gold(labels_dir=LABELS_DIR):
    """``{pano_id: [(cx, cy), ...]}`` from the YOLO-format gold labels, in file order.

    Same tolerant read as the Stage 1 evaluator (fields beyond the third are ignored;
    out-of-range centres skipped). Empty files are the negative panoramas, kept as ``[]``.
    """
    gold = {}
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".txt"):
            continue
        pts = []
        with open(os.path.join(labels_dir, name), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        pts.append((x, y))
        gold[name[:-4]] = pts
    return gold


def load_stage1(path=LABELS_JSON):
    """``({pano_id: [(x, y), ...]}, doc)`` from the committed labels file, in stored order.

    Points outside [0, 1] are dropped, as ``stage_one/dataset_evaluation/evaluate.py`` drops
    them (0 of 3,972 are; the count is ``n_dropped_out_of_range`` in the result).
    """
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    labels, dropped = {}, 0
    for pid, pts in doc["labels"].items():
        keep = []
        for p in pts:
            x, y = float(p[0]), float(p[1])
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                keep.append((x, y))
            else:
                dropped += 1
        labels[pid] = keep
    doc["n_dropped_out_of_range"] = dropped
    return labels, doc


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args):
    """The gold panoramas' Stage 1 points from the Hub test split, labels columns only."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    want = set(load_gold(args.labels_dir))
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{HF_DATASET}@{args.revision}/**/test/*.parquet"))
    if not shards:
        raise SystemExit(f"no test-split parquet under {HF_DATASET}@{args.revision}")

    def read(path):
        tb = pq.read_table(path, columns=["pano_id", "curb_ramp_points_normalized"], filesystem=fs)
        got = {}
        for pid, pts in zip(tb["pano_id"].to_pylist(), tb["curb_ramp_points_normalized"].to_pylist()):
            if pid in want:
                got[pid] = [[float(p[0]), float(p[1])] for p in (pts or [])]
        return got

    print(f"reading 2 columns of {len(shards)} test shards at {args.revision[:12]}...", flush=True)
    labels = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, got in enumerate(ex.map(read, shards), 1):
            labels.update(got)
            if i % 16 == 0 or i == len(shards):
                print(f"  {i}/{len(shards)} shards, {len(labels)}/{len(want)} gold panos", flush=True)
    missing = sorted(want - set(labels))
    doc = {"dataset": HF_DATASET, "revision": args.revision, "split": "test",
           "column": "curb_ramp_points_normalized", "n_panos": len(labels),
           "n_points": sum(len(v) for v in labels.values()), "missing_gold_panos": missing,
           "labels": {pid: labels[pid] for pid in sorted(labels)}}
    write_json(args.labels_json, doc)
    print(f"{len(labels)} panos / {doc['n_points']} points -> {args.labels_json}"
          + (f"; {len(missing)} gold panos not in the test split" if missing else ""))


def summary_md(result):
    c = result["conventions"]
    rows = [("published (first-in-order, redundant ignored)", "published"),
            ("first-in-order, redundant as FP (redundant-only arithmetic)", "published_redund_fp"),
            ("shared matcher, x not cyclic", "shared_nowrap"),
            ("**shared matcher (corrected, of record)**", "shared")]
    lines = ["| convention | P | R | F1 | TP | FP | ignored | FN |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, key in rows:
        a = c[key]
        lines.append(f"| {label} | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} | "
                     f"{a['tp']} | {a['fp']} | {a['ignored']} | {a['fn']} |")
    s = c["shared"]
    d = result["delta_shared"]
    m = result["matching_change"]
    o = result["order_sensitivity"]
    lines += ["",
              f"{result['n_panos']} panoramas ({result['n_negative_panos']} negative), "
              f"{s['n_gt']} gold ramps, {s['n_pred']} generated points; radius {PANO_RADIUS_NORMALIZED}.",
              "",
              f"Corrected (shared matcher): **P {s['precision']:.4f} / R {s['recall']:.4f}**, "
              f"pano-clustered bootstrap 95% CI P [{s['ci95']['precision'][0]:.4f}, {s['ci95']['precision'][1]:.4f}], "
              f"R [{s['ci95']['recall'][0]:.4f}, {s['ci95']['recall'][1]:.4f}].",
              f"Delta from the published 0.9403 / 0.9245: P {d['precision_vs_published']:+.4f}, "
              f"R {d['recall_vs_published']:+.4f}; from the redundant-only 0.9121: "
              f"P {d['precision_vs_redundant_only']:+.4f} (0.9121 is not a bound on the corrected precision).",
              "",
              f"Matching rule alone (first-in-order -> nearest unclaimed, both redundant-as-FP): "
              f"net {m['tp_net']:+d} TP ({m['tp_gained']} gained, {m['tp_lost']} lost, "
              f"across {m['n_panos_changed']} panoramas).",
              "",
              f"Point order (Stage 1 points carry no confidence): stored order gives TP {o['tp_stored_order']}; "
              f"{o['n_shuffles']} seeded per-pano shuffles give TP {o['tp_shuffle_min']}-{o['tp_shuffle_max']}; "
              f"the maximum one-to-one matching gives TP {o['tp_max_matching']} "
              f"(P {o['precision_max_matching']:.4f} / R {o['recall_max_matching']:.4f}).",
              ""]
    return "\n".join(lines) + "\n"


def cmd_score(args):
    gold = load_gold(args.labels_dir)
    stage1, doc = load_stage1(args.labels_json)
    result = score_all(gold, stage1)
    s = result["conventions"]["shared"]
    result["delta_shared"] = {
        "precision_vs_published": s["precision"] - PUBLISHED["precision"],
        "recall_vs_published": s["recall"] - PUBLISHED["recall"],
        "precision_vs_redundant_only": s["precision"] - REDUNDANT_ONLY_PRECISION,
    }
    result["published"] = dict(PUBLISHED)
    result["redundant_only_precision"] = REDUNDANT_ONLY_PRECISION
    result["n_dropped_out_of_range"] = doc["n_dropped_out_of_range"]
    result["radius_normalized"] = PANO_RADIUS_NORMALIZED
    result["inputs"] = {"labels_json": os.path.relpath(args.labels_json, REPO).replace("\\", "/"),
                        "labels_json_sha256": sha256_of(args.labels_json),
                        "dataset": doc["dataset"], "revision": doc["revision"],
                        "manual_labels_dir": "manual_labels"}
    result["bootstrap"] = {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "panorama"}
    result = rounded(result)
    write_json(args.json_out, result)
    md = summary_md(result)
    with open(args.summary_out, "w", encoding="utf-8", newline="") as fh:
        fh.write(md)
    print(md)
    print(f"-> {args.json_out}\n-> {args.summary_out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="Stage 1 points of the gold panos from the Hub test split (network)")
    p.add_argument("--revision", default=HF_REVISION)
    p.add_argument("--labels-dir", default=LABELS_DIR)
    p.add_argument("--labels-json", default=LABELS_JSON)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("score", help="the four conventions on the committed inputs (CPU)")
    p.add_argument("--labels-dir", default=LABELS_DIR)
    p.add_argument("--labels-json", default=LABELS_JSON)
    p.add_argument("--json-out", default=RESULT_JSON)
    p.add_argument("--summary-out", default=SUMMARY_MD)
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
