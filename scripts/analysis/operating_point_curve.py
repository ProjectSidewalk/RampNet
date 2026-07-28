"""Operating-point analysis for RampNet: a proper PR curve + AP, plus a
ground-truth-completeness correction on the low-confidence end (issue #54).

The deployment operating point is peak extraction at ``threshold_abs = 0.55`` /
``min_distance = 10``. It has never been characterised below 0.55, and 0.55 is
not even F1-optimal. This produces the standard artifact — a *continuous*
precision-recall curve and Average Precision — the right way:

  1. Run inference ONCE per pano and extract every peak down to a low score
     FLOOR (``--score-floor``, default 0.05, keeping ``min_distance = 10``),
     carrying each peak's height as its confidence.
  2. Sweep the score threshold post-hoc — no re-inference — for the full
     PR curve, AP, and F1-vs-threshold, with the deployed 0.55 point marked.

This also gives RampNet's first *honest* AP: the committed benchmark detections
were extracted at a 0.5 peak, so their AP integrates only the high-confidence
tail. The low-floor extraction here fixes that.

It then addresses a bias the raw curve hides. The benchmark GT was reviewed
largely in the ~0.55 detection regime, so when the threshold drops we surface
faint detections in a band the GT never fully audited — a *real* ramp nobody
labelled scores as a false positive. So the left half of the curve understates
precision. ``gallery`` renders the *incremental* FPs (unmatched predictions with
score in ``[op, 0.55)``) as native-pano crops for a quick A/B spot-check
(A = real ramp the GT missed; B = genuine FP), and turns the tags into a
corrected precision with an error band.

Inference reproduces the deployment path exactly (resize 2048x4096 bilinear,
ImageNet norm, no TTA) via ``threshold_sweep`` — so the sweep at (0.55, 10)
reproduces the committed ``records.jsonl``. Needs the native-res
``benchmark/<city>/panos/`` locally + a GPU for ``extract`` only; ``curve`` and
``gallery`` are CPU-only and read the cache ``extract`` writes.

Run order:

    # GPU, once — writes op_cache/<city>.json
    python scripts/analysis/operating_point_curve.py extract --cities richmond,bend

    # CPU — PR curve + AP + F1-vs-threshold (CSV + PNG)
    python scripts/analysis/operating_point_curve.py curve --cities richmond,bend

    # CPU (needs panos) — incremental-FP gallery for the GT-completeness check
    python scripts/analysis/operating_point_curve.py gallery --city richmond --op-threshold 0.30
    # ...review index.html, export tags.json, then:
    python scripts/analysis/operating_point_curve.py gallery --city richmond --op-threshold 0.30 --tags tags.json
"""
import argparse
import csv
import html
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, GroundTruth, _xy, aggregate, build_ground_truth,
    load_yolo_ground_truths, prediction_confidence, radius_sq_for, score_pano)
from rampnet.metrics import greedy_match  # noqa: E402

DEFAULT_CITIES = ("richmond", "bend")
HF_MODEL_REPO = "projectsidewalk/rampnet-model"
DEPLOYED_THRESHOLD = 0.55
DEFAULT_SCORE_FLOOR = 0.05
DEFAULT_MIN_DISTANCE = 10
CACHE_DIR = os.path.join(OUT, "op_cache")
# 0.15 of pano width reaches +/-3.4 match radii, so a duplicate hit on an already
# labelled ramp is inside the frame used to judge it. The old 0.08 reached +/-1.8 R
# and cropped that evidence out. See visible_radii().
DEFAULT_CROP_FRAC = 0.15
DEFAULT_RENDER_PX = 1024


# --------------------------------------------------------------------------- #
# Pure scoring / curve core (no torch, no I/O) — unit-tested.
# --------------------------------------------------------------------------- #
def f1_of(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _threshold_grid(floor, step):
    """Inclusive grid ``floor, floor+step, ..., 1.0`` without float drift."""
    n = int(round((1.0 - floor) / step))
    return [round(floor + i * step, 10) for i in range(n + 1)]


def classify_predictions(pred_points, gt, radius_sq, scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """Per-prediction outcome WITH coordinates — the location-preserving twin of
    :func:`rampnet.detection_eval.score_pano`.

    Returns a list of ``(x, y, score, outcome, redundant)`` in the same
    confidence-desc order ``score_pano`` matches in, where ``outcome`` is
    ``"tp"`` / ``"fp"`` / ``"ignore"``. Counting outcomes reproduces
    ``score_pano``'s tp/fp/ignored exactly (asserted in the tests), so the
    gallery selects the same FPs the curve counts.

    ``redundant`` is the fourth thing ``greedy_match`` already knows and every
    caller used to throw away: True when this FP had a real GT ramp *inside* the
    match radius that a higher-confidence prediction had already claimed. That is
    a double-count of one ramp, not a detection of something the GT missed — the
    distinction the A/B spot-check depends on (issue #55).
    """
    confs = [prediction_confidence(p) for p in pred_points]
    if any(c is not None for c in confs):
        order = sorted(range(len(pred_points)),
                       key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                       reverse=True)
        preds = [pred_points[i] for i in order]
    else:
        preds = list(pred_points)

    assignments = greedy_match([_xy(p) for p in preds], gt.gt_points,
                               radius_sq, scale_x, scale_y)
    out = []
    for p, (gt_index, saw_in_range) in zip(preds, assignments):
        x, y = _xy(p)
        score = prediction_confidence(p)
        if gt_index >= 0:
            outcome = "tp"
        else:
            px, py = x * scale_x, y * scale_y
            in_ignore = any((px - ix * scale_x) ** 2 + (py - iy * scale_y) ** 2 < radius_sq
                            for ix, iy in gt.ignore_points)
            outcome = "ignore" if in_ignore else "fp"
        out.append((x, y, score, outcome, outcome == "fp" and bool(saw_in_range)))
    return out


def _dist(ax, ay, bx, by, scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """Distance in the same scaled space the matcher uses, so it is directly
    comparable to the match radius."""
    return ((ax - bx) * scale_x) ** 2 + ((ay - by) * scale_y) ** 2


def proximity(x, y, gt_points, neighbours, radius_sq,
              scale_x=PANO_SCALE_X, scale_y=PANO_SCALE_Y):
    """How close this detection sits to an *already-accounted-for* ramp, in units
    of the match radius R.

    A crop centred on a detection cannot show this — at ``crop_frac`` c the frame
    only reaches ``0.5 * c / radius_normalized`` radii (±1.8 R at the old c=0.08),
    so a GT ramp 2.5 R away is off-frame and the reviewer tags a duplicate as a
    discovery. Returned per item and rendered on the card instead.

    ``neighbours`` is ``[(x, y, outcome)]`` for the other predictions kept at this
    operating point. Returns distances as multiples of R (``None`` when there is
    nothing to measure against).
    """
    r = radius_sq ** 0.5
    d_gt = min((_dist(x, y, gx, gy, scale_x, scale_y) ** 0.5 for gx, gy in gt_points),
               default=None)
    near = sorted(((_dist(x, y, nx, ny, scale_x, scale_y) ** 0.5, outcome)
                   for nx, ny, outcome in neighbours), key=lambda t: t[0])
    return {
        "d_gt_r": round(d_gt / r, 3) if d_gt is not None else None,
        "d_pred_r": round(near[0][0] / r, 3) if near else None,
        "neighbour_outcome": near[0][1] if near else None,
    }


def _score_at(panos, threshold, radius_sq):
    """Aggregate ScoreReport over all panos keeping only predictions >= threshold."""
    pano_scores = []
    for pd in panos:
        preds = [(x, y, s) for (x, y, s) in pd["preds"] if s >= threshold]
        pano_scores.append(score_pano(preds, pd["gt"], radius_sq=radius_sq))
    return aggregate(pano_scores)


def sweep_operating_points(panos, floor, step, radius_sq):
    """P/R/F1 (over all panos, deployment-faithful) at each threshold on the grid."""
    rows = []
    for thr in _threshold_grid(floor, step):
        rep = _score_at(panos, thr, radius_sq)
        rows.append({"threshold": thr, "precision": rep.precision, "recall": rep.recall,
                     "f1": rep.f1, "tp": rep.tp, "fp": rep.fp, "fn": rep.fn})
    return rows


def pr_curve_and_ap(panos, radius_sq):
    """AP + PR curve from the full low-floor extraction (recall-confirmed subset).

    Returns the aggregate ScoreReport at the extraction floor: ``.ap`` and
    ``.pr_curve`` are the continuous curve; ``.precision``/``.recall`` are the
    operating point at the floor itself.
    """
    return _score_at(panos, 0.0, radius_sq)


DUP_NEAR_R = 2.0   # within this many match radii of a GT ramp -> treat as a likely duplicate
DUP_MID_R = 3.0    # beyond DUP_NEAR_R but under this -> ambiguous (corner pairs live here)


def duplicate_risk(redundant, d_gt_r):
    """Bucket an incremental FP by how likely it is to be a second hit on a ramp
    that is *already* in the ground truth, rather than a missed one.

    - ``redundant``  a GT ramp is inside the match radius, already claimed by a
      higher-confidence prediction. Strictly a double-count; it can never be an A.
    - ``near``       GT 1-2 R away: almost certainly the same ramp, localisation slop.
    - ``mid``        GT 2-3 R away: ambiguous — a corner's *other* ramp sits here too.
    - ``isolated``   GT >3 R away, or no GT on the pano: the only clean A candidates.
    """
    if redundant:
        return "redundant"
    if d_gt_r is None:
        return "isolated"
    if d_gt_r < DUP_NEAR_R:
        return "near"
    if d_gt_r < DUP_MID_R:
        return "mid"
    return "isolated"


def incremental_fps(panos, op_threshold, upper, radius_sq):
    """The unmatched predictions a lower threshold newly adds: FP with score in
    ``[op_threshold, upper)``. These are what the GT-completeness spot-check audits.

    Each item gets a stable id so tags survive a re-render, plus the duplicate-risk
    geometry (:func:`proximity`, :func:`duplicate_risk`) the reviewer needs to tell
    "the GT missed a ramp" from "the model fired twice on one ramp".
    """
    items = []
    for pd in panos:
        classified = classify_predictions(pd["preds"], pd["gt"], radius_sq)
        for x, y, score, outcome, redundant in classified:
            if outcome != "fp" or not (op_threshold <= score < upper):
                continue
            neighbours = [(nx, ny, nout) for nx, ny, ns, nout, _ in classified
                          if ns >= op_threshold and not (nx == x and ny == y)]
            prox = proximity(x, y, pd["gt"].gt_points, neighbours, radius_sq)
            items.append({
                "id": f'{pd["pano"]}_{round(x, 5)}_{round(y, 5)}',
                "pano": pd["pano"], "x": x, "y": y, "score": score,
                "redundant": redundant,
                "dup_risk": duplicate_risk(redundant, prox["d_gt_r"]),
                # only recall-confirmed panos contribute to recall, so only an A on
                # one of those can move the recall denominator (see corrected_recall)
                "fn_confirmed": bool(pd["gt"].fn_confirmed),
                **prox,
            })
    items.sort(key=lambda it: it["score"])
    return items


TAGS = ("A", "B", "U")


def corrected_precision(tp, fp, items, tags):
    """Corrected precision at the operating point given the incremental-FP tags.

    ``tp``/``fp`` are the raw counts at the operating point; ``items`` are the
    incremental FPs in ``[op, upper)``; ``tags`` maps item id -> ``"A"`` (a real
    ramp the GT missed — becomes a TP), ``"B"`` (a genuine FP) or ``"U"`` (unsure).

    The band is an honest uncertainty interval rather than a formality: the low end
    credits only confirmed As, the high end additionally credits everything the
    reviewer could not call (``U``) and everything not yet looked at. When every
    item is tagged A or B it collapses to a point, which is the correct behaviour —
    there is then nothing left to be uncertain about.

    ``ceiling_all_real`` keeps the old "what if every incremental FP were real"
    reference, and ``n_A_suspect`` counts As that the geometry says are probably a
    second hit on an already-counted ramp (see :func:`duplicate_risk`) — those
    inflate precision without finding anything, so they are surfaced, not hidden.
    """
    denom = tp + fp
    counts = {t: sum(1 for it in items if tags.get(it["id"]) == t) for t in TAGS}
    n_a, n_u = counts["A"], counts["U"]
    n_tagged = sum(counts.values())
    n_untagged = len(items) - n_tagged
    n_a_suspect = sum(1 for it in items if tags.get(it["id"]) == "A"
                      and it.get("dup_risk") in ("redundant", "near"))
    return {
        "uncorrected": tp / denom if denom else 0.0,
        "corrected": (tp + n_a) / denom if denom else 0.0,
        "band_high": (tp + n_a + n_u + n_untagged) / denom if denom else 0.0,
        "ceiling_all_real": (tp + len(items)) / denom if denom else 0.0,
        "tp": tp, "fp": fp, "n_incremental": len(items),
        "n_A": n_a, "n_B": counts["B"], "n_U": n_u,
        "n_tagged": n_tagged, "n_untagged": n_untagged, "n_A_suspect": n_a_suspect,
    }


def corrected_recall(tp_recall, n_gt_recall, items, tags):
    """Recall corrected for the same A tags — the other half of the correction.

    An A is a ramp the GT did not have, so it adds one to the numerator *and* one
    to the denominator: the model found it, and it should always have been counted
    as findable. Correcting precision alone (as this module originally did) reports
    a corrected P against an uncorrected R, which flatters nothing but is simply
    inconsistent — the same relabelling has to be applied to both.

    Only panos whose missed-ramp check is confirmed contribute to recall at all
    (``aggregate`` restricts both sums to those), so an A on an unscanned pano must
    move neither number. ``tp_recall``/``n_gt_recall`` are that confirmed subset;
    from a ScoreReport they are ``rep.n_gt_recall - rep.fn`` and ``rep.n_gt_recall``.
    """
    eligible = [it for it in items if it.get("fn_confirmed", True)]
    n_a = sum(1 for it in eligible if tags.get(it["id"]) == "A")
    n_open = sum(1 for it in eligible
                 if tags.get(it["id"]) == "U" or tags.get(it["id"]) not in TAGS)

    def at(k):
        return (tp_recall + k) / (n_gt_recall + k) if (n_gt_recall + k) else 0.0

    return {
        "uncorrected": at(0),
        "corrected": at(n_a),
        "band_high": at(n_a + n_open),
        "tp_recall": tp_recall, "n_gt_recall": n_gt_recall,
        "n_A_recall": n_a, "n_A_unscanned": sum(
            1 for it in items if tags.get(it["id"]) == "A" and not it.get("fn_confirmed", True)),
    }


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #
def _gt_to_json(gt):
    return {"gt_points": [list(p) for p in gt.gt_points],
            "ignore_points": [list(p) for p in gt.ignore_points],
            "fn_confirmed": gt.fn_confirmed}


def _gt_from_json(d):
    return GroundTruth([tuple(p) for p in d["gt_points"]],
                       [tuple(p) for p in d["ignore_points"]],
                       bool(d["fn_confirmed"]))


def write_cache(path, city, panos, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"city": city, "meta": meta,
               "panos": [{"pano": pd["pano"],
                          "preds": [[x, y, s] for (x, y, s) in pd["preds"]],
                          "gt": _gt_to_json(pd["gt"])} for pd in panos]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def read_cache(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    panos = [{"pano": p["pano"],
              "preds": [tuple(t) for t in p["preds"]],
              "gt": _gt_from_json(p["gt"])} for p in payload["panos"]]
    return panos, payload.get("meta", {})


# --------------------------------------------------------------------------- #
# bundle ground truth (both bundle kinds)
# --------------------------------------------------------------------------- #
def bundle_ground_truths(city, repo=REPO):
    """``({pid: GroundTruth}, panos_dir)`` for either kind of benchmark bundle.

    The city splits carry a human verdict review (``verdicts.json``) that
    :func:`build_ground_truth` turns into GT. ``benchmark/manual_gold`` instead
    carries ``gt_source.json`` pointing at independently-labelled YOLO files —
    no RampNet review to derive from, and so no RampNet anchoring. Resolving both
    here is what lets the sweep cover the in-distribution gold split alongside the
    deployment cities; the layout logic mirrors
    ``scripts/model_comparison/compare.py``, which reads the same two bundle kinds.
    """
    import json as _json
    cdir = os.path.join(repo, "benchmark", city)
    panos_dir = os.path.join(cdir, "panos")
    vpath = os.path.join(cdir, "verdicts.json")
    gpath = os.path.join(cdir, "gt_source.json")

    if os.path.exists(vpath):
        import threshold_sweep as ts
        records, verdicts, panos_dir = ts.load_bundle(city)
        return {pid: build_ground_truth(records[pid]["detections"], entry["dets"],
                                        entry["missed"], entry["no_missed"])
                for pid, entry in verdicts.items()}, panos_dir

    if not os.path.exists(gpath):
        raise SystemExit(f"{cdir}: neither verdicts.json nor gt_source.json — "
                         "not a benchmark bundle")
    with open(gpath, encoding="utf-8") as f:
        src = _json.load(f)
    if src.get("format") != "yolo_points":
        raise SystemExit(f"{gpath}: unsupported format {src.get('format')!r} "
                         "(expected 'yolo_points')")
    labels_dir = os.path.normpath(os.path.join(cdir, src["labels_dir"]))
    gts = load_yolo_ground_truths(labels_dir)
    if not gts:
        raise SystemExit(f"{labels_dir}: no .txt label files found")
    # Only score panos whose imagery is actually in the bundle: manual_labels/ has
    # a label file per gold pano, but the split is unusable for any pano whose jpg
    # was not fetched, and silently scoring it as an all-miss pano would deflate
    # recall rather than report a gap.
    present = {pid for pid in gts
               if os.path.exists(os.path.join(panos_dir, f"{pid}.jpg"))}
    missing = len(gts) - len(present)
    if missing:
        print(f"  [{city}] {missing} of {len(gts)} labelled panos have no imagery "
              f"in the bundle — excluded (not scored as misses)", flush=True)
    return {pid: gts[pid] for pid in sorted(present)}, panos_dir


# --------------------------------------------------------------------------- #
# extract (GPU)
# --------------------------------------------------------------------------- #
def cmd_extract(args):
    import torch  # lazy: only extract needs a GPU / torch
    import threshold_sweep as ts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.load_model().to(device)
    use_fp16 = False
    print(f"device={device} score_floor={args.score_floor} min_distance={args.min_distance}",
          flush=True)
    radius_sq = radius_sq_for()

    for city in args.cities:
        out_path = os.path.join(args.cache, f"{city}.json")
        # Per-city skip, so a preempted or timed-out Slurm job resumes at the split
        # it died in rather than re-running every split before it.
        if os.path.exists(out_path) and not args.force:
            print(f"{city}: cache exists -> skipping (--force to re-extract)", flush=True)
            continue
        gts, panos_dir = bundle_ground_truths(city)
        panos = []
        for i, (pid, gt) in enumerate(gts.items(), 1):
            path = os.path.join(panos_dir, f"{pid}.jpg")
            try:
                h = ts.heatmap_for(model, device, path, use_fp16)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                use_fp16 = True
                print("  OOM -> switching to fp16 autocast", flush=True)
                h = ts.heatmap_for(model, device, path, use_fp16)
            preds = ts.peaks_to_dets(h, args.score_floor, args.min_distance)
            panos.append({"pano": pid, "preds": preds, "gt": gt})
            if i % 50 == 0:
                print(f"  {city}: {i}/{len(gts)}", flush=True)
            del h
        meta = {"score_floor": args.score_floor, "min_distance": args.min_distance,
                "radius_normalized": 0.022, "fp16": use_fp16,
                "n_panos": len(panos), "deployed_threshold": DEPLOYED_THRESHOLD,
                "model": HF_MODEL_REPO, "device": device.type}
        write_cache(out_path, city, panos, meta)
        print(f"{city}: {len(panos)} panos -> {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# curve (CPU)
# --------------------------------------------------------------------------- #
def _save_pr_png(path, pr_curve, ap, city):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping PNG: {e})")
        return
    recalls, precisions = pr_curve
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recalls, precisions, "-", lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"{city}: PR curve (AP={ap:.4f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _save_threshold_png(path, rows, city, mark):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping PNG: {e})")
        return
    thr = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thr, [r["precision"] for r in rows], label="precision")
    ax.plot(thr, [r["recall"] for r in rows], label="recall")
    ax.plot(thr, [r["f1"] for r in rows], label="F1", lw=2)
    ax.axvline(mark, color="gray", ls="--", lw=1, label=f"deployed {mark}")
    best = max(rows, key=lambda r: r["f1"])
    ax.axvline(best["threshold"], color="green", ls=":", lw=1,
               label=f"F1-max {best['threshold']:.2f}")
    ax.set_xlabel("score threshold"); ax.set_ylabel("value")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"{city}: P / R / F1 vs threshold"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def cmd_curve(args):
    os.makedirs(args.out, exist_ok=True)
    radius_sq = radius_sq_for()
    for city in args.cities:
        cache_path = os.path.join(args.cache, f"{city}.json")
        if not os.path.exists(cache_path):
            print(f"[{city}] no cache at {cache_path} — run `extract` first."); continue
        panos, meta = read_cache(cache_path)
        floor = meta.get("score_floor", DEFAULT_SCORE_FLOOR)

        rep = pr_curve_and_ap(panos, radius_sq)
        rows = sweep_operating_points(panos, floor, args.step, radius_sq)
        best = max(rows, key=lambda r: r["f1"])
        deployed = min(rows, key=lambda r: abs(r["threshold"] - DEPLOYED_THRESHOLD))

        # CSVs
        with open(os.path.join(args.out, f"{city}_threshold_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
            w.writeheader()
            w.writerows(rows)
        if rep.pr_curve:
            with open(os.path.join(args.out, f"{city}_pr_curve.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["recall", "precision"])
                w.writerows(zip(*rep.pr_curve))
            _save_pr_png(os.path.join(args.out, f"{city}_pr_curve.png"), rep.pr_curve, rep.ap, city)
        _save_threshold_png(os.path.join(args.out, f"{city}_threshold.png"), rows, city, DEPLOYED_THRESHOLD)

        ap_str = f"{rep.ap:.4f}" if rep.ap is not None else "n/a"
        print(f"\n{'='*66}\n{city.upper()}  (n={meta.get('n_panos', len(panos))}, "
              f"floor={floor}, AP={ap_str})\n{'='*66}")
        print(f"  deployed  thr {deployed['threshold']:.2f}: "
              f"P {deployed['precision']:.3f}  R {deployed['recall']:.3f}  F1 {deployed['f1']:.3f}  "
              f"({deployed['tp']}/{deployed['fp']}/{deployed['fn']})")
        print(f"  F1-optimal thr {best['threshold']:.2f}: "
              f"P {best['precision']:.3f}  R {best['recall']:.3f}  F1 {best['f1']:.3f}  "
              f"({best['tp']}/{best['fp']}/{best['fn']})")
        print(f"  wrote {city}_threshold_sweep.csv, {city}_pr_curve.csv/png, {city}_threshold.png -> {args.out}")


# --------------------------------------------------------------------------- #
# gallery (CPU, needs native panos)
# --------------------------------------------------------------------------- #
def draft_width_for(crop_frac, render_px):
    """The decode width a crop needs to be rendered from *real* pixels.

    ``draft`` is a big win on a 16k-px pano, but it was previously pinned at 2048
    regardless of the crop size, which quietly capped detail below what the review
    needs: at crop_frac 0.08 a 16384-px bend pano drafted to 2048 leaves **163 px**
    of source for a 512-px crop — a 3x upscale of mush, which is why so much of the
    first tagging pass reads "blurry" / "distant" / "ambiguous". Sizing the decode to
    the crop instead means the renderer never upscales."""
    return int(math.ceil(render_px / max(crop_frac, 1e-6)))


def _open_pano_drafted(pano_path, draft_max=2048):
    """Open a pano, decoding the JPEG at a reduced DCT scale (PIL ``draft``) so a
    16k-px native pano doesn't cost a full-resolution decode where the crop can't use
    it (and, over an NFS home, a way to thrash the box). ``draft`` only steps in
    powers of two and never upscales, so asking for more than the file holds is free.
    Returns ``(img, W, H)``; normalised coords map against the drafted size."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(pano_path)
    img.draft("RGB", (draft_max, draft_max // 2))
    img = img.convert("RGB")
    return img, img.width, img.height


def visible_radii(crop_frac, radius_normalized=None):
    """How far, in match radii, a crop of this fraction can actually see.

    Half the crop width over the match radius, both in pano pixels:
    ``0.5 * crop_frac * W / (radius_normalized * W)``. The pano cancels, so it is a
    property of the setting alone — at the original 0.08 the frame reached only
    **±1.8 R**, which is *less* than the 2-3 R where duplicate hits on an already
    labelled ramp live. That is a rendering bug with an epistemic cost: the evidence
    that a detection is a duplicate was cropped out of the picture used to judge it."""
    from rampnet.detection_eval import PANO_RADIUS_NORMALIZED
    return 0.5 * crop_frac / (radius_normalized or PANO_RADIUS_NORMALIZED)


def _crop_and_mark(img, W, H, item, gt, others, crop_frac, render_px=1024,
                   radius_normalized=None):
    """Crop of the (already-open) pano centred on the FP, with the FP (red +), the
    match radius around it (dashed red), GT (green o) and other kept predictions
    (yellow o) drawn. Returns a PIL image.

    The match-radius ring is the point: it shows, in the image, the tolerance the
    scorer actually used, so "would this have matched that ramp?" stops being a
    guess. Markers scale with ``render_px`` so a bigger crop doesn't shrink them
    into invisibility."""
    from PIL import ImageDraw
    from rampnet.detection_eval import PANO_RADIUS_NORMALIZED
    rn = radius_normalized or PANO_RADIUS_NORMALIZED
    cw = max(64, int(crop_frac * W))
    cw = min(cw, W, H)
    cx, cy = item["x"] * W, item["y"] * H
    left, top = int(cx - cw / 2), int(cy - cw / 2)
    left = max(0, min(left, W - cw)); top = max(0, min(top, H - cw))
    crop = img.crop((left, top, left + cw, top + cw))
    scale = render_px / cw                      # no 4x upscale cap: draft sized the source
    crop = crop.resize((int(cw * scale), int(cw * scale)))
    draw = ImageDraw.Draw(crop)
    k = max(1.0, render_px / 512.0)             # marker scale
    w_thin, w_thick = int(2 * k), int(3 * k)

    def to_local(nx, ny):
        return (nx * W - left) * scale, (ny * H - top) * scale

    for gx, gy in gt.gt_points:
        lx, ly = to_local(gx, gy)
        if 0 <= lx < crop.width and 0 <= ly < crop.height:
            r = 9 * k
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], outline=(0, 220, 0), width=w_thick)
    for ox, oy, _s in others:
        lx, ly = to_local(ox, oy)
        if 0 <= lx < crop.width and 0 <= ly < crop.height:
            r = 7 * k
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], outline=(255, 210, 0), width=w_thin)
    lx, ly = to_local(item["x"], item["y"])
    rr = rn * W * scale                          # match radius in local px
    draw.ellipse([lx - rr, ly - rr, lx + rr, ly + rr], outline=(255, 90, 90), width=w_thin)
    arm = 13 * k
    draw.line([lx - arm, ly, lx + arm, ly], fill=(255, 0, 0), width=w_thick)
    draw.line([lx, ly - arm, lx, ly + arm], fill=(255, 0, 0), width=w_thick)
    return crop


_GALLERY_JS = r"""
let tags = {};
let cur = 0;
const cards = () => [...document.querySelectorAll('.card')];

function save(){ try{ localStorage.setItem(KEY, JSON.stringify(tags)); }catch(e){} }

function paint(id){
  const card = document.querySelector('[data-id="'+CSS.escape(id)+'"]');
  if(!card) return;
  card.querySelectorAll('.btns button').forEach(b => b.classList.remove('sel'));
  const v = tags[id];
  if(v){ const b = card.querySelector('.btns button.'+v); if(b) b.classList.add('sel'); }
  card.classList.toggle('done', !!v);
}

function counts(){
  const c = {A:0,B:0,U:0};
  Object.values(tags).forEach(v => { if(c[v]!==undefined) c[v]++; });
  document.getElementById('count').textContent =
    `${c.A} A · ${c.B} B · ${c.U} unsure · ${TOTAL-c.A-c.B-c.U} left`;
}

function tag(id, v){
  if(tags[id] === v) delete tags[id]; else tags[id] = v;   // click again to clear
  paint(id); counts(); save();
}

function focusCard(i){
  const cs = cards(); if(!cs.length) return;
  cur = Math.max(0, Math.min(i, cs.length-1));
  cs.forEach(c => c.classList.remove('cur'));
  const c = cs[cur]; c.classList.add('cur');
  c.scrollIntoView({block:'center', behavior:'smooth'});
}

function zoom(src){
  const o = document.getElementById('lb');
  document.getElementById('lbimg').src = src;
  o.style.display = 'flex';
}
function unzoom(){ document.getElementById('lb').style.display = 'none'; }

function exportTags(){
  const blob = new Blob([JSON.stringify(tags, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = CITY + '_tags.json'; a.click();
}

function importTags(input){
  const f = input.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = () => {
    try{
      tags = JSON.parse(r.result) || {};
      cards().forEach(c => paint(c.dataset.id));
      counts(); save();
    }catch(e){ alert('could not parse that file: ' + e); }
  };
  r.readAsText(f);
}

function onlyUntagged(on){
  cards().forEach(c => { c.style.display = (on && tags[c.dataset.id]) ? 'none' : ''; });
}

document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT') return;
  if(e.key === 'Escape'){ unzoom(); return; }
  const cs = cards();
  if(e.key === 'j' || e.key === 'ArrowDown' || e.key === 'ArrowRight'){ focusCard(cur+1); e.preventDefault(); return; }
  if(e.key === 'k' || e.key === 'ArrowUp'   || e.key === 'ArrowLeft'){ focusCard(cur-1); e.preventDefault(); return; }
  const c = cs[cur]; if(!c) return;
  if(e.key === 'z' || e.key === 'Enter'){ zoom(c.querySelector('img').src); e.preventDefault(); return; }
  const v = {a:'A', b:'B', u:'U'}[e.key.toLowerCase()];
  if(v){ tag(c.dataset.id, v); focusCard(cur+1); e.preventDefault(); }
});

window.addEventListener('DOMContentLoaded', () => {
  try{ tags = JSON.parse(localStorage.getItem(KEY)) || {}; }catch(e){ tags = {}; }
  cards().forEach(c => paint(c.dataset.id));
  counts(); focusCard(0);
});
"""


_DUP_LABEL = {
    "redundant": ("dup", "GT ramp INSIDE the match radius, already claimed — a double-count"),
    "near": ("dup?", "a GT ramp is {d} R away and already detected — probably the same ramp"),
    "mid": ("?", "nearest GT ramp {d} R away — could be the corner's other ramp, or slop"),
    "isolated": ("", ""),
}


def _card_html(it, crop_frac):
    rid = html.escape(it["id"])
    risk = it.get("dup_risk", "isolated")
    d = it.get("d_gt_r")
    badge, tip = _DUP_LABEL.get(risk, ("", ""))
    tip = tip.format(d=d)
    warn = (f'<span class="warn {risk}" title="{html.escape(tip)}">{badge}</span>'
            if badge else "")
    dgt = f'{d:g} R' if d is not None else 'no GT on pano'
    nb = it.get("neighbour_outcome")
    nbtxt = (f' · nearest pred {it["d_pred_r"]:g} R ({nb})'
             if it.get("d_pred_r") is not None and nb else '')
    return (
        f'<figure class="card {risk}" data-id="{rid}">'
        f'<img src="{html.escape(it["img"])}" loading="lazy" '
        f'onclick="zoom(this.src)" title="click to zoom">'
        f'<figcaption>'
        f'<div class="meta"><b>{it["score"]:.3f}</b> {warn}</div>'
        f'<div class="geo">GT {dgt}{nbtxt}</div>'
        f'<div class="pano">{html.escape(it["pano"])}</div>'
        f'<div class="btns">'
        f'<button class="A" onclick="tag(\'{rid}\',\'A\')" title="a">A · real</button>'
        f'<button class="B" onclick="tag(\'{rid}\',\'B\')" title="b">B · FP</button>'
        f'<button class="U" onclick="tag(\'{rid}\',\'U\')" title="u">? · unsure</button>'
        f'</div></figcaption></figure>')


def _write_gallery_html(out_dir, city, op_threshold, upper, items, crop_frac):
    rows = [_card_html(it, crop_frac) for it in items]
    n_dup = sum(1 for it in items if it.get("dup_risk") in ("redundant", "near"))
    reach = visible_radii(crop_frac)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{city} incremental FPs</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}}
h1{{font-size:18px;margin:0 0 6px}}
.bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;z-index:9}}
.bar p{{margin:6px 0;font-size:13px;color:#bbb}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:12px}}
.card{{background:#1c1c1c;border:1px solid #333;border-radius:6px;padding:6px;margin:0}}
.card.cur{{border-color:#6af;box-shadow:0 0 0 2px #6af4}}
.card.done{{opacity:.55}}
.card.redundant,.card.near{{border-color:#c85}}
.card img{{width:100%;border-radius:4px;display:block;cursor:zoom-in}}
.meta{{font-size:13px;color:#ddd;margin:6px 0 2px;display:flex;gap:8px;align-items:center}}
.geo{{font-size:11px;color:#8a8}} .pano{{color:#666;font-size:10px;margin-bottom:5px;word-break:break-all}}
.warn{{font-size:10px;padding:1px 6px;border-radius:3px;background:#c85;color:#111;font-weight:700;cursor:help}}
.warn.mid{{background:#665;color:#ddd}}
.btns{{display:flex;gap:4px}}
button{{flex:1;font-size:11px;padding:6px 4px;border:1px solid #444;background:#222;color:#ccc;border-radius:4px;cursor:pointer}}
button.A.sel{{background:#1a6;color:#fff;border-color:#1a6}}
button.B.sel{{background:#a33;color:#fff;border-color:#a33}}
button.U.sel{{background:#a83;color:#fff;border-color:#a83}}
#export{{background:#357;color:#fff;border-color:#357;padding:6px 12px;flex:none}}
label.imp{{font-size:12px;color:#8ad;cursor:pointer}} label.imp input{{display:none}}
#lb{{display:none;position:fixed;inset:0;background:#000d;z-index:99;align-items:center;
    justify-content:center;cursor:zoom-out}}
#lb img{{max-width:96vw;max-height:96vh;image-rendering:auto}}
kbd{{background:#333;border-radius:3px;padding:1px 5px;font-size:11px}}
</style></head><body>
<div class="bar">
<h1>{city}: incremental FPs, score in [{op_threshold}, {upper}) &mdash; {len(items)} to review</h1>
<p><b>A</b> = a real curb ramp the GT missed (should be a TP) ·
<b>B</b> = a genuine false positive ·
<b>?</b> = unsure (widens the reported error band instead of forcing a guess).</p>
<p>Red ring = the match radius the scorer used. Green o = a GT ramp; yellow o = another kept
prediction. <b>{n_dup} of {len(items)}</b> sit within {DUP_NEAR_R:g} R of an already-detected
ramp and are flagged <span class="warn">dup?</span> — those are second hits on one ramp, not
discoveries. This crop reaches &plusmn;{reach:.1f} R.</p>
<p>Keys: <kbd>a</kbd> <kbd>b</kbd> <kbd>u</kbd> tag &amp; advance ·
<kbd>j</kbd>/<kbd>k</kbd> move · <kbd>z</kbd> zoom · <kbd>Esc</kbd> close.
Clicking the selected tag again clears it. Saved to this browser as you go.</p>
<p style="display:flex;gap:10px;align-items:center">
<span id="count">0 / {len(items)} tagged</span>
<button id="export" onclick="exportTags()">Download {city}_tags.json</button>
<label class="imp">load tags.json<input type="file" accept="application/json"
 onchange="importTags(this)"></label>
<label class="imp"><input type="checkbox" style="display:inline"
 onchange="onlyUntagged(this.checked)"> hide tagged</label>
</p></div>
<div class="grid">{''.join(rows)}</div>
<div id="lb" onclick="unzoom()"><img id="lbimg"></div>
<script>const TOTAL={len(items)}, CITY={json.dumps(city)},
 KEY={json.dumps(f"rampnet-tags-{city}-{op_threshold}-{upper}")};{_GALLERY_JS}</script>
</body></html>"""
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def cmd_gallery(args):
    radius_sq = radius_sq_for()
    cache_path = os.path.join(args.cache, f"{args.city}.json")
    if not os.path.exists(cache_path):
        raise SystemExit(f"no cache at {cache_path} — run `extract` first.")
    panos, meta = read_cache(cache_path)
    by_id = {pd["pano"]: pd for pd in panos}
    items = incremental_fps(panos, args.op_threshold, args.upper, radius_sq)

    rep = _score_at(panos, args.op_threshold, radius_sq)  # tp/fp at the operating point

    if args.tags:
        with open(args.tags, encoding="utf-8") as f:
            tags = json.load(f)
        res = corrected_precision(rep.tp, rep.fp, items, tags)
        rec = corrected_recall(rep.n_gt_recall - rep.fn, rep.n_gt_recall, items, tags)
        risk = {}
        for it in items:
            risk[it["dup_risk"]] = risk.get(it["dup_risk"], 0) + 1
        print(f"[{args.city}] operating point {args.op_threshold} (deployed {args.upper}):")
        print(f"  raw            P {res['uncorrected']:.3f}  R {rec['uncorrected']:.3f}  "
              f"F1 {f1_of(res['uncorrected'], rec['uncorrected']):.3f}"
              f"   ({res['tp']} tp / {res['fp']} fp)")
        print(f"  incremental FPs in band: {res['n_incremental']}  "
              f"(A={res['n_A']}  B={res['n_B']}  unsure={res['n_U']}  "
              f"untagged={res['n_untagged']})")
        print(f"  duplicate risk: " + "  ".join(f"{k}={v}" for k, v in sorted(risk.items())))
        print(f"  corrected      P {res['corrected']:.3f}  R {rec['corrected']:.3f}  "
              f"F1 {f1_of(res['corrected'], rec['corrected']):.3f}")
        print(f"  band           P {res['corrected']:.3f}..{res['band_high']:.3f}  "
              f"R {rec['corrected']:.3f}..{rec['band_high']:.3f}   "
              f"(all-real P ceiling {res['ceiling_all_real']:.3f})")
        if rec["n_A_unscanned"]:
            print(f"  note: {rec['n_A_unscanned']} A-tag(s) sit on panos whose missed-ramp "
                  f"check is unconfirmed — they correct precision but cannot move recall.")
        if res["n_A_suspect"]:
            print(f"  WARNING: {res['n_A_suspect']} of {res['n_A']} A-tags sit within "
                  f"{DUP_NEAR_R:g} R of an already-detected ramp — those are likely second "
                  f"hits on one ramp, not ramps the GT missed. Re-check before publishing.")
        return

    out_dir = args.out or os.path.join(OUT, "op", f"{args.city}_incremental_fp")
    os.makedirs(out_dir, exist_ok=True)
    panos_dir = args.panos or os.path.join(REPO, "benchmark", args.city, "panos")
    # group by pano so each native pano is JPEG-decoded exactly once, not per-FP
    from collections import defaultdict
    by_pano = defaultdict(list)
    for it in items:
        by_pano[it["pano"]].append(it)

    draft_max = args.draft_max or draft_width_for(args.crop_frac, args.render_px)
    kept = []
    for pano_id, pano_items in by_pano.items():
        pd = by_id[pano_id]
        img, W, H = _open_pano_drafted(os.path.join(panos_dir, f"{pano_id}.jpg"), draft_max)
        for it in pano_items:
            others = [(x, y, s) for (x, y, s) in pd["preds"]
                      if s >= args.op_threshold and not (x == it["x"] and y == it["y"])]
            crop = _crop_and_mark(img, W, H, it, pd["gt"], others, args.crop_frac,
                                  args.render_px)
            img_name = f'{it["id"]}.png'
            crop.save(os.path.join(out_dir, img_name))
            kept.append({**it, "img": img_name})
        img.close()
    kept.sort(key=lambda it: it["score"])

    with open(os.path.join(out_dir, "incremental_fps.json"), "w", encoding="utf-8") as f:
        json.dump({"city": args.city, "op_threshold": args.op_threshold,
                   "upper": args.upper, "crop_frac": args.crop_frac,
                   "render_px": args.render_px, "items": kept}, f, indent=2)
    html_path = _write_gallery_html(out_dir, args.city, args.op_threshold, args.upper, kept,
                                    args.crop_frac)
    n_dup = sum(1 for it in kept if it["dup_risk"] in ("redundant", "near"))
    print(f"[{args.city}] {len(kept)} incremental FPs in [{args.op_threshold}, {args.upper}) "
          f"-> {html_path}")
    print(f"  crop {args.crop_frac:g} of pano width = +/-{visible_radii(args.crop_frac):.1f} "
          f"match radii, rendered at {args.render_px}px (decode width {draft_max})")
    print(f"  {n_dup} flagged as likely duplicates of an already-detected ramp")
    print(f"  review, tag A/B/unsure, Download {args.city}_tags.json, "
          f"then re-run with --tags <that file>")


# --------------------------------------------------------------------------- #
def _csv_cities(s):
    return tuple(c.strip() for c in s.split(",") if c.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="GPU: run inference once, extract peaks to a low floor, cache")
    e.add_argument("--cities", type=_csv_cities, default=DEFAULT_CITIES)
    e.add_argument("--score-floor", type=float, default=DEFAULT_SCORE_FLOOR)
    e.add_argument("--min-distance", type=int, default=DEFAULT_MIN_DISTANCE)
    e.add_argument("--cache", default=CACHE_DIR)
    e.add_argument("--force", action="store_true",
                   help="re-extract splits that already have a cache (default: skip them, "
                        "so a preempted job resumes where it stopped)")
    e.set_defaults(func=cmd_extract)

    c = sub.add_parser("curve", help="CPU: PR curve + AP + F1-vs-threshold from the cache")
    c.add_argument("--cities", type=_csv_cities, default=DEFAULT_CITIES)
    c.add_argument("--cache", default=CACHE_DIR)
    c.add_argument("--out", default=os.path.join(OUT, "op"))
    c.add_argument("--step", type=float, default=0.05)
    c.set_defaults(func=cmd_curve)

    g = sub.add_parser("gallery", help="CPU: incremental-FP gallery for the GT-completeness check")
    g.add_argument("--city", required=True)
    g.add_argument("--op-threshold", type=float, required=True)
    g.add_argument("--upper", type=float, default=DEPLOYED_THRESHOLD)
    g.add_argument("--cache", default=CACHE_DIR)
    g.add_argument("--panos", default=None)
    g.add_argument("--out", default=None)
    g.add_argument("--crop-frac", type=float, default=DEFAULT_CROP_FRAC,
                   help=f"crop width as a fraction of pano width (default "
                        f"{DEFAULT_CROP_FRAC}, reaching +/-"
                        f"{0.5 * DEFAULT_CROP_FRAC / 0.022:.1f} match radii). Must exceed "
                        f"{0.022 * 2 * DUP_MID_R:.2f} to show a duplicate at {DUP_MID_R:g} R")
    g.add_argument("--render-px", type=int, default=DEFAULT_RENDER_PX,
                   help=f"rendered crop edge in px (default {DEFAULT_RENDER_PX})")
    g.add_argument("--draft-max", type=int, default=None,
                   help="decode panos at a reduced DCT scale >= this width (PIL draft). "
                        "Default: derived from --crop-frac/--render-px so crops are never "
                        "upscaled; set explicitly only to trade detail for speed")
    g.add_argument("--tags", default=None, help="tags.json -> print corrected precision instead of rendering")
    g.set_defaults(func=cmd_gallery)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
