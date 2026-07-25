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
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)

from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, GroundTruth, _xy, aggregate, build_ground_truth,
    prediction_confidence, radius_sq_for, score_pano)
from rampnet.metrics import greedy_match  # noqa: E402

DEFAULT_CITIES = ("richmond", "bend")
DEPLOYED_THRESHOLD = 0.55
DEFAULT_SCORE_FLOOR = 0.05
DEFAULT_MIN_DISTANCE = 10
CACHE_DIR = os.path.join(OUT, "op_cache")


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

    Returns a list of ``(x, y, score, outcome)`` in the same confidence-desc order
    ``score_pano`` matches in, where ``outcome`` is ``"tp"`` / ``"fp"`` / ``"ignore"``.
    Counting outcomes reproduces ``score_pano``'s tp/fp/ignored exactly (asserted
    in the tests), so the gallery selects the same FPs the curve counts.
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
    for p, (gt_index, _) in zip(preds, assignments):
        x, y = _xy(p)
        score = prediction_confidence(p)
        if gt_index >= 0:
            outcome = "tp"
        else:
            px, py = x * scale_x, y * scale_y
            in_ignore = any((px - ix * scale_x) ** 2 + (py - iy * scale_y) ** 2 < radius_sq
                            for ix, iy in gt.ignore_points)
            outcome = "ignore" if in_ignore else "fp"
        out.append((x, y, score, outcome))
    return out


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


def incremental_fps(panos, op_threshold, upper, radius_sq):
    """The unmatched predictions a lower threshold newly adds: FP with score in
    ``[op_threshold, upper)``. These are what the GT-completeness spot-check audits.

    Each item gets a stable id so tags survive a re-render.
    """
    items = []
    for pd in panos:
        for x, y, score, outcome in classify_predictions(pd["preds"], pd["gt"], radius_sq):
            if outcome == "fp" and op_threshold <= score < upper:
                items.append({
                    "id": f'{pd["pano"]}_{round(x, 5)}_{round(y, 5)}',
                    "pano": pd["pano"], "x": x, "y": y, "score": score,
                })
    items.sort(key=lambda it: it["score"])
    return items


def corrected_precision(tp, fp, items, tags):
    """Corrected precision at the operating point given the incremental-FP A/B tags.

    ``tp``/``fp`` are the raw counts at the operating point; ``items`` are the
    incremental FPs in ``[op, upper)``; ``tags`` maps item id -> "A"/"B". An A
    (real ramp the GT missed) becomes a TP. The band spans no-correction (lower)
    to every-incremental-FP-real (upper) so the untagged residual is explicit.
    """
    denom = tp + fp
    n_a = sum(1 for it in items if tags.get(it["id"]) == "A")
    n_tagged = sum(1 for it in items if tags.get(it["id"]) in ("A", "B"))
    return {
        "uncorrected": tp / denom if denom else 0.0,
        "corrected": (tp + n_a) / denom if denom else 0.0,
        "upper_bound": (tp + len(items)) / denom if denom else 0.0,
        "tp": tp, "fp": fp, "n_incremental": len(items), "n_A": n_a, "n_tagged": n_tagged,
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
        records, verdicts, panos_dir = ts.load_bundle(city)
        panos = []
        for i, (pid, entry) in enumerate(verdicts.items(), 1):
            gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                    entry["missed"], entry["no_missed"])
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
            if i % 20 == 0:
                print(f"  {city}: {i}/{len(verdicts)}", flush=True)
            del h
        out_path = os.path.join(args.cache, f"{city}.json")
        meta = {"score_floor": args.score_floor, "min_distance": args.min_distance,
                "radius_normalized": 0.022, "fp16": use_fp16,
                "n_panos": len(panos), "deployed_threshold": DEPLOYED_THRESHOLD}
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
def _open_pano_drafted(pano_path, draft_max=2048):
    """Open a pano, decoding the JPEG at a reduced DCT scale (PIL ``draft``) so a
    16k-px native pano doesn't cost a full-resolution decode — the crops are shown
    at ~512 px, so full res is wasted work (and, over an NFS home, a way to thrash
    the box). Returns ``(img, W, H)``; normalised coords map against the drafted size."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(pano_path)
    img.draft("RGB", (draft_max, draft_max // 2))
    img = img.convert("RGB")
    return img, img.width, img.height


def _crop_and_mark(img, W, H, item, gt, others, crop_frac):
    """Crop of the (already-open) pano centred on the FP, with the FP (red +),
    GT (green o) and other kept predictions (yellow o) drawn. Returns a PIL image."""
    from PIL import ImageDraw
    cw = max(64, int(crop_frac * W))
    cx, cy = item["x"] * W, item["y"] * H
    left, top = int(cx - cw / 2), int(cy - cw / 2)
    left = max(0, min(left, W - cw)); top = max(0, min(top, H - cw))
    crop = img.crop((left, top, left + cw, top + cw))
    scale = min(4.0, 512 / cw)
    crop = crop.resize((int(cw * scale), int(cw * scale)))
    draw = ImageDraw.Draw(crop)

    def to_local(nx, ny):
        return (nx * W - left) * scale, (ny * H - top) * scale

    for gx, gy in gt.gt_points:
        lx, ly = to_local(gx, gy)
        if 0 <= lx < crop.width and 0 <= ly < crop.height:
            draw.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], outline=(0, 220, 0), width=3)
    for ox, oy, _s in others:
        lx, ly = to_local(ox, oy)
        if 0 <= lx < crop.width and 0 <= ly < crop.height:
            draw.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], outline=(255, 210, 0), width=2)
    lx, ly = to_local(item["x"], item["y"])
    draw.line([lx - 13, ly, lx + 13, ly], fill=(255, 0, 0), width=3)
    draw.line([lx, ly - 13, lx, ly + 13], fill=(255, 0, 0), width=3)
    return crop


_GALLERY_JS = """
const tags = {};
function tag(id, v){
  tags[id] = v;
  document.querySelectorAll('[data-id="'+id+'"] button').forEach(b=>b.classList.remove('sel'));
  document.querySelector('[data-id="'+id+'"] button.'+v).classList.add('sel');
  document.getElementById('count').textContent = Object.keys(tags).length + ' / ' + TOTAL + ' tagged';
}
function exportTags(){
  const blob = new Blob([JSON.stringify(tags, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'tags.json'; a.click();
}
"""


def _write_gallery_html(out_dir, city, op_threshold, upper, items):
    rows = []
    for it in items:
        rid = html.escape(it["id"])
        rows.append(
            f'<div class="card" data-id="{rid}">'
            f'<img src="{html.escape(it["img"])}" loading="lazy">'
            f'<div class="meta">score {it["score"]:.3f}<br><span class="pano">{html.escape(it["pano"])}</span></div>'
            f'<div class="btns">'
            f'<button class="A" onclick="tag(\'{rid}\',\'A\')">A · real (GT missed)</button>'
            f'<button class="B" onclick="tag(\'{rid}\',\'B\')">B · genuine FP</button>'
            f'</div></div>')
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{city} incremental FPs</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}}
h1{{font-size:18px}} .bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;z-index:9}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:12px}}
.card{{background:#1c1c1c;border:1px solid #333;border-radius:6px;padding:6px}}
.card img{{width:100%;border-radius:4px;display:block}}
.meta{{font-size:12px;color:#aaa;margin:4px 0}} .pano{{color:#666;font-size:10px}}
.btns{{display:flex;gap:4px}} button{{flex:1;font-size:11px;padding:5px;border:1px solid #444;background:#222;color:#ccc;border-radius:4px;cursor:pointer}}
button.A.sel{{background:#1a6;color:#fff;border-color:#1a6}} button.B.sel{{background:#a33;color:#fff;border-color:#a33}}
#export{{background:#357;color:#fff;border-color:#357;padding:6px 12px}}
</style></head><body>
<div class="bar"><h1>{city}: incremental FPs, score in [{op_threshold}, {upper}) &mdash; {len(items)} to review</h1>
<p>A = a real curb ramp the GT missed (should be a TP) &nbsp;·&nbsp; B = a genuine false positive.
<span id="count">0 / {len(items)} tagged</span>
<button id="export" onclick="exportTags()">Download tags.json</button></p></div>
<div class="grid">{''.join(rows)}</div>
<script>const TOTAL={len(items)};{_GALLERY_JS}</script></body></html>"""
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
        print(f"[{args.city}] operating point {args.op_threshold} (deployed {args.upper}):")
        print(f"  raw            P {res['uncorrected']:.3f}  ({res['tp']} tp / {res['fp']} fp)")
        print(f"  incremental FPs in band: {res['n_incremental']}  "
              f"(tagged {res['n_tagged']}, A={res['n_A']})")
        print(f"  corrected      P {res['corrected']:.3f}   "
              f"(band {res['uncorrected']:.3f}..{res['upper_bound']:.3f})")
        return

    out_dir = args.out or os.path.join(OUT, "op", f"{args.city}_incremental_fp")
    os.makedirs(out_dir, exist_ok=True)
    panos_dir = args.panos or os.path.join(REPO, "benchmark", args.city, "panos")
    # group by pano so each native pano is JPEG-decoded exactly once, not per-FP
    from collections import defaultdict
    by_pano = defaultdict(list)
    for it in items:
        by_pano[it["pano"]].append(it)

    kept = []
    for pano_id, pano_items in by_pano.items():
        pd = by_id[pano_id]
        img, W, H = _open_pano_drafted(os.path.join(panos_dir, f"{pano_id}.jpg"), args.draft_max)
        for it in pano_items:
            others = [(x, y, s) for (x, y, s) in pd["preds"]
                      if s >= args.op_threshold and not (x == it["x"] and y == it["y"])]
            crop = _crop_and_mark(img, W, H, it, pd["gt"], others, args.crop_frac)
            img_name = f'{it["id"]}.png'
            crop.save(os.path.join(out_dir, img_name))
            kept.append({**it, "img": img_name})
        img.close()
    kept.sort(key=lambda it: it["score"])

    with open(os.path.join(out_dir, "incremental_fps.json"), "w", encoding="utf-8") as f:
        json.dump({"city": args.city, "op_threshold": args.op_threshold,
                   "upper": args.upper, "items": kept}, f, indent=2)
    html_path = _write_gallery_html(out_dir, args.city, args.op_threshold, args.upper, kept)
    print(f"[{args.city}] {len(kept)} incremental FPs in [{args.op_threshold}, {args.upper}) "
          f"-> {html_path}\n  review, click A/B, Download tags.json, then re-run with --tags tags.json")


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
    g.add_argument("--crop-frac", type=float, default=0.08)
    g.add_argument("--draft-max", type=int, default=2048,
                   help="decode panos at a reduced DCT scale >= this width (PIL draft); "
                        "keeps huge native panos cheap to render")
    g.add_argument("--tags", default=None, help="tags.json -> print corrected precision instead of rendering")
    g.set_defaults(func=cmd_gallery)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
