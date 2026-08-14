"""Is RampNet's far-field blindness inherited from its LABELS? (#59, experiment E1)

#59 argues that scaling Stage 1 could make the worst failure mode *worse*. The
mechanism: Stage-1 agreement against the manual gold set is P .9403 / R .9245, so
~7.5% of gold-visible ramps are unlabeled inside training panoramas — and in
heatmap regression an unlabeled ramp is not neutral. The target is **zero** at
that location, so the loss actively pushes activations down. Those are implicit
hard negatives.

The hypothesis is that Stage-1's misses are disproportionately **far / small /
occluded** — exactly where GPS->pixel projection error and the crop model degrade.
If so, we have been training the model to suppress detections precisely where it
is now blind, and scaling the same pipeline bakes that in harder.

This measures it. Two recall curves over the **same 1,000 gold panoramas**:

* **Stage-1 label recall** — did the *label pipeline* put a point on each gold ramp?
* **Model recall** — did *RampNet* detect it?

Both stratified on the #25 bins with the identical ``geom()`` estimator used by
``size_analysis.py``, so the curves are directly comparable rather than merely
adjacent.

**Decision rule, fixed in #59 before running:**

* Stage-1's curve **mirrors** the model's cliff -> the blindness is substantially
  *inherited from the labels*. Label quality at range outranks label volume, and
  naive scaling is **contraindicated**.
* Stage-1 stays **flat** while the model collapses -> the ceiling is the
  model/resolution, not the data, and more data is not the thing holding recall back.

No GPU. The Stage-1 labels come from the ``curb_ramp_points_normalized`` column of
``projectsidewalk/rampnet-dataset``'s test split, read **column-selectively over
HTTP range requests** (a few MB, not the 44 GB split), then cached.

    python scripts/analysis/stage1_label_recall.py            # first run fetches
    python scripts/analysis/stage1_label_recall.py --threshold 0.30

The gold set is the right and only venue for this: its labels were produced
manually with no model in the loop (``benchmark/manual_gold/gt_source.json``), so
neither curve is anchored to the other.
"""
import argparse
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for)
from rampnet.metrics import greedy_match  # noqa: E402

HF_DATASET = "projectsidewalk/rampnet-dataset"
LABELS_DIR = os.path.join(REPO, "manual_labels")
GOLD_RECORDS = os.path.join(REPO, "benchmark", "manual_gold", "records.jsonl")
CACHE = os.path.join(OUT, "stage1_gold_labels.json")

# Identical to scripts/analysis/size_analysis.py — a ramp of real width RAMP_W at
# distance d subtends RAMP_W/d radians, and the pano is 4096 px per 2*pi. Shared
# so the Stage-1 curve lands on the same axis as the model curve from #25.
CAM_H, RAMP_W = 2.5, 1.2                    # metres
PX_PER_RAD = 4096.0 / (2 * math.pi)

SIZE_BUCKETS = [(0, 12), (12, 20), (20, 32), (32, 50), (50, 80), (80, 1e9)]
DIST_BUCKETS = [(0, 8), (8, 12), (12, 18), (18, 25), (25, 40), (40, 1e9)]

# The published gold-set operating point (README: P .949 / R .873 at 0.55). The
# Stage-1 curve has no threshold — labels carry no confidence — so this only
# moves the model curve.
DEFAULT_THRESHOLD = 0.55


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_stage1_label_recall.py
# --------------------------------------------------------------------------- #
def geom(y):
    """``(distance_m, apparent_px)`` for a ground point at pano-normalized y.

    Flat-ground estimate, validated against Depth-Anything-3 depth to within
    6.5-8.5% (see ``precision_by_distance.py``). The camera height rescales every
    distance by one factor, so it sets the metre labels but not the comparison.
    """
    depression = (y - 0.5) * math.pi
    if depression <= 1e-4:
        return 150.0, RAMP_W / 150.0 * PX_PER_RAD
    d = min(CAM_H / math.tan(depression), 150.0)
    return d, RAMP_W / d * PX_PER_RAD


def bucket_of(value, buckets):
    """First ``(lo, hi)`` containing ``value`` (half-open), else ``None``."""
    for lo, hi in buckets:
        if lo <= value < hi:
            return (lo, hi)
    return None


def hit_indices(pred_points, gold_points, radius_sq=None):
    """Indices of gold ramps claimed by some prediction.

    Uses the shared greedy matcher, so "was this ramp found?" means exactly what
    it means in every other RampNet evaluator. Predictions must already be in the
    order the caller wants them consumed — by confidence for model detections,
    input order for Stage-1 labels, which carry none.
    """
    if radius_sq is None:
        radius_sq = radius_sq_for()
    assignments = greedy_match(pred_points, gold_points, radius_sq,
                               PANO_SCALE_X, PANO_SCALE_Y)
    return {gi for gi, _ in assignments if gi >= 0}


def build_rows(gold, stage1, model):
    """One row per gold ramp: its geometry, and whether each source found it.

    Ramps in panoramas missing from *either* source are skipped, so both curves
    are computed over an identical population — otherwise a coverage difference
    would masquerade as a recall difference.
    """
    rows = []
    for pid, gold_pts in sorted(gold.items()):
        if pid not in stage1 or pid not in model or not gold_pts:
            continue
        s_hits = hit_indices(stage1[pid], gold_pts)
        m_hits = hit_indices(model[pid], gold_pts)
        for i, (gx, gy) in enumerate(gold_pts):
            dist, px = geom(gy)
            rows.append({"pano": pid, "x": gx, "y": gy, "dist": dist, "px": px,
                         "stage1": i in s_hits, "model": i in m_hits})
    return rows


def recall_table(rows, key, buckets):
    """Per-bucket recall for both sources, plus the gap between them."""
    out = []
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r[key] < hi]
        n = len(b)
        s = sum(1 for r in b if r["stage1"])
        m = sum(1 for r in b if r["model"])
        out.append({
            "lo": lo, "hi": hi, "n": n,
            "stage1_recall": (s / n) if n else float("nan"),
            "model_recall": (m / n) if n else float("nan"),
            "gap": ((s - m) / n) if n else float("nan"),
        })
    return out


MIN_BUCKET_N = 30


def dropoff(table, min_n=MIN_BUCKET_N):
    """Recall in the nearest well-populated bucket minus the farthest one.

    The number the decision rule turns on: a large drop-off for Stage-1 means the
    labels themselves thin out with distance, so the model's cliff is inherited
    rather than its own.

    ``min_n`` exists because the tail buckets are tiny — the gold set has **4**
    ramps beyond 40 m, and at n=4 a single ramp moves recall by 0.25. Letting that
    bucket anchor the comparison inverted the sign of Stage-1's drop-off (to
    -0.041) and would have decided the experiment on four samples.
    """
    pop = [r for r in table if r["n"] >= min_n]
    if len(pop) < 2:
        return {"stage1": float("nan"), "model": float("nan"), "n_buckets": len(pop)}
    return {"stage1": pop[0]["stage1_recall"] - pop[-1]["stage1_recall"],
            "model": pop[0]["model_recall"] - pop[-1]["model_recall"],
            "n_buckets": len(pop),
            "near": (pop[0]["lo"], pop[0]["hi"]), "far": (pop[-1]["lo"], pop[-1]["hi"])}


def verdict(dist_table, tol=0.15):
    """Apply #59's pre-registered decision rule to the distance curves.

    ``tol`` is how much *less* drop-off Stage-1 may show before the labels are
    judged flat relative to the model. Stated here rather than in prose so the
    call cannot drift after seeing the numbers.
    """
    d = dropoff(dist_table)
    if math.isnan(d["stage1"]) or math.isnan(d["model"]):
        return "INCONCLUSIVE — not enough populated buckets."
    if d["stage1"] >= d["model"] - tol:
        return ("MIRRORS — Stage-1 label recall falls off with distance about as "
                "steeply as the model's. The far-field blindness is substantially "
                "INHERITED FROM THE LABELS: label quality at range outranks label "
                "volume, and naive scaling of the same pipeline is contraindicated.")
    return ("FLAT — Stage-1 labels hold up with distance while the model's recall "
            "collapses. The ceiling is the MODEL/RESOLUTION, not the labels, so "
            "scaling is not contraindicated by this mechanism (but neither is it "
            "shown to help — that is E2/E3).")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_gold_labels(labels_dir=LABELS_DIR):
    """``{pano_id: [(cx, cy), ...]}`` from the YOLO-format manual gold labels.

    Empty files are the 207 negative panoramas; they are kept as empty lists so a
    missing file and a deliberately-empty one stay distinguishable.
    """
    gold = {}
    for path in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        pts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    pts.append((float(parts[1]), float(parts[2])))
        gold[os.path.splitext(os.path.basename(path))[0]] = pts
    return gold


def load_stage1_labels(wanted_ids, cache=CACHE, workers=8):
    """Stage-1 auto-labels for the gold panos, from the Hub test split.

    Reads only ``pano_id`` + ``curb_ramp_points_normalized`` via HTTP range
    requests — a few MB rather than the split's ~44 GB — then caches, because the
    published dataset is immutable and this never needs re-fetching.
    """
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return {k: [tuple(p) for p in v] for k, v in json.load(fh).items()}

    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shards = fs.glob(f"datasets/{HF_DATASET}/test/*.parquet")
    if not shards:
        raise SystemExit(f"no test-split parquet found for {HF_DATASET}")
    want = set(wanted_ids)

    def read(path):
        tb = pq.read_table(path, columns=["pano_id", "curb_ramp_points_normalized"],
                           filesystem=fs)
        got = {}
        for pid, pts in zip(tb["pano_id"].to_pylist(),
                            tb["curb_ramp_points_normalized"].to_pylist()):
            if pid in want:
                got[pid] = [(float(p[0]), float(p[1])) for p in (pts or [])]
        return got

    print(f"fetching Stage-1 labels from {len(shards)} shards "
          f"(columns only, not images)...")
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(read, shards), 1):
            out.update(got)
            if i % 32 == 0:
                print(f"  {i}/{len(shards)} shards, {len(out)} gold panos found")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"cached {len(out)} panos -> {cache}")
    return out


def load_model_detections(path=GOLD_RECORDS, threshold=DEFAULT_THRESHOLD):
    """RampNet detections on the gold panos, confidence-ordered and thresholded.

    Ordering matters: the greedy matcher consumes predictions in the order given,
    and every other RampNet evaluator feeds it best-first.
    """
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            dets = [d for d in rec.get("detections", [])
                    if d["confidence"] >= threshold]
            dets.sort(key=lambda d: d["confidence"], reverse=True)
            out[rec["pano"]["panorama_id"]] = [
                (d["x_normalized"], d["y_normalized"]) for d in dets]
    return out


def print_table(title, table, unit):
    print(f"\n{title}")
    print(f"{'bucket':>14} {'n':>6} {'Stage-1':>9} {'model':>9} {'gap':>8}")
    for r in table:
        hi = "+" if r["hi"] > 1e8 else f"{int(r['hi'])}"
        label = f"{int(r['lo'])}-{hi} {unit}"
        if not r["n"]:
            print(f"{label:>14} {0:>6} {'-':>9} {'-':>9} {'-':>8}")
            continue
        print(f"{label:>14} {r['n']:>6} {r['stage1_recall']:>9.3f} "
              f"{r['model_recall']:>9.3f} {r['gap']:>+8.3f}")
    d = dropoff(table)
    if d.get("near"):
        hi = "+" if d["far"][1] > 1e8 else int(d["far"][1])
        print(f"{'drop-off':>14} {'':>6} {d['stage1']:>9.3f} {d['model']:>9.3f}"
              f"   [{int(d['near'][0])}-{int(d['near'][1])} vs "
              f"{int(d['far'][0])}-{hi}; buckets with n>={MIN_BUCKET_N}]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Model confidence threshold (Stage-1 labels have none).")
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--tol", type=float, default=0.15,
                   help="Drop-off tolerance for the pre-registered verdict.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    gold = load_gold_labels()
    stage1 = load_stage1_labels(gold.keys(), cache=args.cache)
    model = load_model_detections(threshold=args.threshold)
    rows = build_rows(gold, stage1, model)

    n_panos = len({r["pano"] for r in rows})
    print(f"\n=== E1: are the model's misses inherited from the labels? (#59) ===")
    print(f"{len(rows)} gold ramps across {n_panos} panos "
          f"(model threshold {args.threshold}); both curves on the same population.")

    dist = recall_table(rows, "dist", DIST_BUCKETS)
    size = recall_table(rows, "px", SIZE_BUCKETS)
    print_table("Recall vs distance", dist, "m")
    print_table("Recall vs apparent size", size, "px")

    overall_s = sum(r["stage1"] for r in rows) / len(rows) if rows else float("nan")
    overall_m = sum(r["model"] for r in rows) / len(rows) if rows else float("nan")
    print(f"\nOverall: Stage-1 {overall_s:.3f}, model {overall_m:.3f}")
    print(f"\nVERDICT: {verdict(dist, args.tol)}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "n_ramps": len(rows),
                       "n_panos": n_panos, "distance": dist, "size": size,
                       "stage1_overall": overall_s, "model_overall": overall_m,
                       "verdict": verdict(dist, args.tol)}, fh, indent=2)
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
