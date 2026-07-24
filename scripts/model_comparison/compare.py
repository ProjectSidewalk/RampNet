"""Compare curb-ramp detectors on a benchmark bundle.

Scores each selected model against the same model-agnostic ground truth derived
from the human review (see rampnet/detection_eval.py), so RampNet and the VLMs
are compared on equal footing. RampNet's verdict-based numbers are re-printed as
a cross-check.

    python scripts/model_comparison/compare.py benchmark/richmond --models rampnet
    python scripts/model_comparison/compare.py benchmark/richmond \
        --models rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash

Each --models token is a provider (rampnet/gemini/qwen/owlv2/gdino/molmo) or
provider:model_id to pin a variant, so several models from the same provider
compare side by side. Detectors that emit calibrated scores (RampNet, OWLv2,
Grounding DINO) additionally get AP, a PR curve (--pr-out) and a threshold sweep
(--sweep); chat VLMs have no score to rank by, so they get one operating point.
See docs/model_comparison.md.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))          # rampnet.* (editable install fallback)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # local detectors.py

from rampnet.detection_eval import (  # noqa: E402
    build_ground_truth, score_pano, aggregate, radius_sq_for, PANO_RADIUS_NORMALIZED,
)
from rampnet.validation import collect, format_report  # noqa: E402
from detectors import (  # noqa: E402
    GDINO_QUERY, OWLV2_QUERY, PanoSample, build_detector, parse_model_spec,
)


def load_dotenv(root):
    """Load KEY=VALUE lines from a repo-root .env into os.environ (without
    overriding already-set vars), so a Gemini key can live in a git-ignored file
    instead of the shell/transcript. Minimal parser — no python-dotenv dependency."""
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def cache_key(label, signature, city, pano_id):
    """Stable hash over everything that determines a detector's output for one
    pano, so re-runs reuse cached detections and don't re-pay the API."""
    blob = json.dumps({"label": label, "sig": signature, "city": city, "pid": pano_id},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class DetectionCache:
    """On-disk cache of per-pano detection points (a paid VLM call is expensive;
    scoring/radius changes are free, so we cache the detector output, not the
    score). Sharded by key prefix. A no-op when disabled."""

    def __init__(self, root, enabled=True):
        self.root = root
        self.enabled = enabled

    def _path(self, key):
        return os.path.join(self.root, key[:2], f"{key}.json")

    def get(self, key):
        if not self.enabled:
            return None
        path = self._path(key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["points"]
        return None

    def put(self, key, points):
        if not self.enabled:
            return
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"points": [list(p) for p in points]}, f)


def load_bundle(bundle_dir):
    """Return (records_by_pid, verdicts_panos, panos_dir) for a benchmark bundle."""
    records = {}
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    with open(os.path.join(bundle_dir, "verdicts.json"), encoding="utf-8") as f:
        verdicts = json.load(f)["panos"]
    return records, verdicts, os.path.join(bundle_dir, "panos")


def validate_bundle(records, verdicts):
    """Fail fast on a structurally broken bundle, *before* any (paid) detector call.

    ``score_model`` builds each pano's ground truth from ``records[pid]`` + the
    verdict entry outside its per-pano failure guard (that guard is for transient
    detect() errors, not data integrity). Without this pre-flight a reviewed pano
    missing from records.jsonl, a missing verdict field, or detections/verdicts
    that don't line up would surface as a raw KeyError/ValueError partway through a
    long VLM run — after spend, and aborting models already scored. Catch it here
    with a clear message instead. Raises SystemExit listing every offending pano.

    (Legacy verdicts.json without ``no_missed`` are intentionally rejected here
    rather than silently defaulted — the current/planned bundles are new-schema;
    see docs/model_comparison.md.)"""
    problems = []
    for pid, entry in verdicts.items():
        rec = records.get(pid)
        if rec is None:
            problems.append(f"{pid}: reviewed in verdicts.json but absent from records.jsonl")
            continue
        missing = [k for k in ("dets", "missed", "no_missed") if k not in entry]
        if missing:
            problems.append(f"{pid}: verdict entry missing field(s) {missing}")
            continue
        n_det, n_ver = len(rec.get("detections", [])), len(entry["dets"])
        if n_det != n_ver:
            problems.append(f"{pid}: {n_det} detections vs {n_ver} verdicts (misaligned)")
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise SystemExit(f"Bundle validation failed ({len(problems)} pano(s)):\n  {shown}{more}")


# scored: [(pred_points, GroundTruth)] for every pano that was successfully
# scored, kept so the threshold sweep and PR curves can re-score from memory
# instead of re-running the detector.
ModelRun = namedtuple("ModelRun", ["report", "failures", "scored"])


def score_model(detector, records, verdicts, panos_dir, radius_sq, label, city, cache,
                max_consecutive_failures=10):
    """Run one detector over every reviewed pano and aggregate the score.

    Returns a ``ModelRun``. ``detector.prepare()`` runs before the pano loop
    (outside the per-pano guard) so credential / dependency / not-wired errors
    propagate to the caller and skip the whole model — but it is skipped entirely
    when every pano is already cached, so a ``.model_cache`` copied back from a GPU
    cluster scores on a machine that can't load the model at all. Each pano's
    detections are cached, so re-runs don't re-pay the API. A transient per-pano
    failure is recorded and skipped rather than crashing the run;
    ``max_consecutive_failures`` aborts the model early during an outage instead of
    burning budget."""
    sig = detector.signature() if hasattr(detector, "signature") else None
    keys = {pid: (cache_key(label, sig, city, pid) if sig is not None else None)
            for pid in verdicts}
    cached = {pid: (cache.get(k) if k else None) for pid, k in keys.items()}
    if not cached or any(p is None for p in cached.values()):
        detector.prepare()
    else:
        print(f"[{label}] all {len(cached)} panos already cached; model load skipped")
    pano_scores, failures, consecutive, scored = [], [], 0, []
    for pid, entry in verdicts.items():
        rec = records[pid]
        gt = build_ground_truth(rec["detections"], entry["dets"], entry["missed"], entry["no_missed"])
        key, preds = keys[pid], cached[pid]
        if preds is None:
            sample = PanoSample(
                pano_id=pid,
                image_path=os.path.join(panos_dir, f"{pid}.jpg"),
                width=rec["pano"].get("width"),
                height=rec["pano"].get("height"),
                meta=rec["pano"],
            )
            try:
                preds = detector.detect(sample)
            except Exception as e:  # transient API/network failure: isolate this pano
                failures.append((pid, f"{type(e).__name__}: {str(e)[:120]}"))
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    failures.append(("<abort>", f"{consecutive} consecutive failures; stopped early"))
                    break
                continue
            consecutive = 0
            if key:
                cache.put(key, preds)
        scored.append((preds, gt))
        pano_scores.append(score_pano(preds, gt, radius_sq=radius_sq))
    return ModelRun(aggregate(pano_scores), failures, scored)


def rescore(scored, radius_sq, min_confidence=0.0):
    """Re-aggregate a finished run with predictions below ``min_confidence`` dropped.

    Detections are cached with their scores, so every operating point of a
    confidence-carrying detector is a free re-score — no second model run. A
    prediction with no confidence (chat VLMs) is never dropped: there is nothing to
    threshold on."""
    return aggregate([
        score_pano([p for p in preds if _conf_of(p) is None or _conf_of(p) >= min_confidence],
                   gt, radius_sq=radius_sq)
        for preds, gt in scored])


def _conf_of(p):
    return p[2] if len(p) > 2 else None


def has_confidences(scored):
    """True when every prediction in the run carries a score (so AP / a sweep mean
    something). An empty run counts as no confidences."""
    preds = [p for ps, _ in scored for p in ps]
    return bool(preds) and all(_conf_of(p) is not None for p in preds)


SWEEP_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def sweep_rows(scored, radius_sq, thresholds=SWEEP_THRESHOLDS):
    """(threshold, ScoreReport) for each threshold that still keeps a prediction."""
    top = max((_conf_of(p) for ps, _ in scored for p in ps if _conf_of(p) is not None),
              default=0.0)
    return [(t, rescore(scored, radius_sq, t)) for t in thresholds if t <= top]


def _pct(x):
    return f"{x:.3f}"


def _ci(lo_hi):
    return f"({lo_hi[0]:.3f}-{lo_hi[1]:.3f})"


def _ap(r):
    # Blank for chat VLMs: no calibrated per-box score, so no curve to integrate.
    return f"{r.ap:.3f}" if r.ap is not None else "  -  "


def print_table(rows):
    # Wide enough for the longest HF id in play (google/owlv2-large-patch14-ensemble).
    header = (f"{'model':<36} {'P':>6} {'95% CI':>15} {'R':>6} {'95% CI':>15} "
              f"{'F1':>6} {'AP':>6}   {'tp/fp/fn/ign':>16}")
    print(header)
    print("-" * len(header))
    for name, r in rows:
        counts = f"{r.tp}/{r.fp}/{r.fn}/{r.ignored}"
        print(f"{name:<36} {_pct(r.precision):>6} {_ci(r.precision_ci):>15} "
              f"{_pct(r.recall):>6} {_ci(r.recall_ci):>15} {_pct(r.f1):>6} {_ap(r):>6}   "
              f"{counts:>16}")


def print_sweep(label, rows):
    """Threshold sweep for one model: what tuning the score cutoff buys.

    This is the point of a real detector over a chat VLM — the recall-first
    direction needs a knob, and a model pinned at one operating point has none.
    The best-F1 row is flagged; it is chosen *on the benchmark itself*, so it is an
    optimistic, tune-on-test number and must be quoted as such."""
    if not rows:
        return
    best = max(range(len(rows)), key=lambda i: rows[i][1].f1)
    print(f"\n[{label}] threshold sweep (re-scored from cached detections)")
    print(f"  {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}   {'tp/fp/fn':>14}")
    for i, (t, r) in enumerate(rows):
        mark = " <- best F1" if i == best else ""
        print(f"  {t:>5.2f} {_pct(r.precision):>6} {_pct(r.recall):>6} {_pct(r.f1):>6}   "
              f"{f'{r.tp}/{r.fp}/{r.fn}':>14}{mark}")


def write_pr_curves(out_dir, curves):
    """Write each model's PR curve to JSON, and a combined PNG if matplotlib is
    around (it is not a harness dependency). ``curves``: [(label, ScoreReport)]."""
    os.makedirs(out_dir, exist_ok=True)
    for label, r in curves:
        recalls, precisions = r.pr_curve
        safe = label.replace("/", "_")
        with open(os.path.join(out_dir, f"pr_{safe}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": label, "ap": r.ap, "n_gt": r.n_gt_recall,
                       "recalls": recalls, "precisions": precisions}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"PR curves written to {out_dir} (JSON only; matplotlib not installed)")
        return
    plt.figure(figsize=(7, 6))
    for label, r in curves:
        recalls, precisions = r.pr_curve
        plt.plot(recalls, precisions, marker=".", markersize=3,
                 label=f"{label} (AP {r.ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curb-ramp detection PR curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    path = os.path.join(out_dir, "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curves written to {out_dir} (JSON + {os.path.basename(path)})")


def main():
    # Windows consoles are cp1252; avoid UnicodeEncodeError on stray bytes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="Compare curb-ramp detectors on a benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir (e.g. benchmark/richmond) with records.jsonl + verdicts.json.")
    ap.add_argument("--models", default="rampnet",
                    help="Comma-separated detectors. Each is a provider (rampnet/gemini/qwen/"
                         "owlv2/gdino/molmo, using its default model) or provider:model_id to "
                         "pin a variant, e.g. 'rampnet,gemini:gemini-2.5-flash,owlv2'.")
    ap.add_argument("--gemini-model", default="gemini-3.6-flash")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--qwen-coord-space", choices=["auto", "norm1000", "pixels"], default="auto",
                    help="Box convention the Qwen checkpoint emits: 'norm1000' (Qwen3-VL, "
                         "0-1000) or 'pixels' (Qwen2/2.5-VL, absolute). 'auto' infers it "
                         "from the model id.")
    ap.add_argument("--owlv2-model", default="google/owlv2-large-patch14-ensemble")
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--molmo-model", default="allenai/Molmo2-8B")
    ap.add_argument("--owlv2-query", help=f"OWLv2 text query (default {OWLV2_QUERY!r}).")
    ap.add_argument("--gdino-query", help=f"Grounding DINO category text (default {GDINO_QUERY!r}); "
                                          "lowercase, period-terminated.")
    ap.add_argument("--gdino-text-threshold", type=float,
                    help="Grounding DINO token-alignment threshold (default 0.2).")
    ap.add_argument("--score-threshold", type=float,
                    help="Score floor for the open-vocabulary detectors (owlv2/gdino), default "
                         "0.05. This is a CACHE floor, not the operating point: it is part of "
                         "the detector signature, so lowering it re-runs the model, while every "
                         "higher operating point is a free re-score (--op-threshold, --sweep).")
    ap.add_argument("--molmo-coord-scale", choices=["auto", "100", "1000"], default="auto",
                    help="Divisor for Molmo point coordinates: Molmo 1 emits percentages "
                         "(100), Molmo 2 emits 0-1000. 'auto' infers it from the tag syntax.")
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective",
                    help="VLM input: 'perspective' reprojects the pano into rectilinear "
                         "views (fair); 'none' uses one whole-pano call (lower bound). "
                         "No effect on rampnet.")
    ap.add_argument("--radius", type=float, default=PANO_RADIUS_NORMALIZED,
                    help=f"Normalized match radius (default {PANO_RADIUS_NORMALIZED}).")
    ap.add_argument("--op-threshold", type=float, default=0.0,
                    help="Drop predictions scoring below this before the main table, so the "
                         "reported operating point is comparable across models. Free (re-scores "
                         "the cache); models without confidences are unaffected.")
    ap.add_argument("--sweep", action="store_true",
                    help="Also print a threshold sweep for every model whose detections carry "
                         "confidences (RampNet, owlv2, gdino) — the tunable operating range.")
    ap.add_argument("--pr-out", help="Directory to write PR curves to (JSON per model, plus a "
                                     "combined PNG when matplotlib is installed).")
    ap.add_argument("--limit", type=int,
                    help="Score at most N panos (smoke test / cost control for VLM runs).")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / ".model_cache"),
                    help="Where to cache per-pano detections (keyed by model + rig + pano). "
                         "Re-runs reuse hits and don't re-pay the API.")
    ap.add_argument("--no-cache", action="store_true", help="Disable the detection cache.")
    args = ap.parse_args()

    load_dotenv(str(REPO_ROOT))
    records, verdicts, panos_dir = load_bundle(args.bundle)
    if args.limit:
        verdicts = dict(list(verdicts.items())[:args.limit])
    validate_bundle(records, verdicts)  # fail fast before any (paid) detector call
    radius_sq = radius_sq_for(args.radius)
    specs = [parse_model_spec(t) for t in args.models.split(",") if t.strip()]
    city = os.path.basename(os.path.normpath(args.bundle))
    cache = DetectionCache(args.cache_dir, enabled=not args.no_cache)

    print(f"Bundle: {args.bundle}  ({len(verdicts)} reviewed panos)  "
          f"match radius {args.radius}  ground truth: reviewer-confirmed ramps + missed marks")
    print(f"Detection cache: {'off' if args.no_cache else args.cache_dir}\n")

    rows, runs = [], []
    seen = {}
    for provider, model_id in specs:
        label, detector = build_detector(provider, model_id, records, args)
        # Disambiguate if the same label appears twice (e.g. same model, two configs).
        if label in seen:
            seen[label] += 1
            label = f"{label}#{seen[label]}"
        else:
            seen[label] = 1
        try:
            run = score_model(
                detector, records, verdicts, panos_dir, radius_sq, label, city, cache)
        except Exception as e:
            # Missing client lib, missing credentials, a checkpoint whose remote
            # code won't load on this transformers version: skip the whole model
            # with a clear note rather than crashing a multi-model cluster run that
            # has already paid for the models before it. Per-pano faults are
            # isolated inside score_model; data-integrity problems are caught by
            # validate_bundle before any of this. The type is printed so a genuine
            # bug here is still diagnosable rather than silently "not runnable".
            print(f"[{label}] not runnable: {type(e).__name__}: {e}\n")
            continue
        report = (rescore(run.scored, radius_sq, args.op_threshold)
                  if args.op_threshold > 0 else run.report)
        rows.append((label, report))
        runs.append((label, run))
        if run.failures:
            print(f"[{label}] {len(run.failures)} pano failure(s) isolated "
                  "(excluded from the score):")
            for pid, msg in run.failures[:5]:
                print(f"    {pid}: {msg}")
            if len(run.failures) > 5:
                print(f"    ... and {len(run.failures) - 5} more")
            print()

    if rows:
        if args.op_threshold > 0:
            print(f"Operating point: predictions with confidence < {args.op_threshold} dropped "
                  "(models without confidences are unaffected).")
        print_table(rows)
        # AP is over the recall-confirmed panos only (one consistent GT denominator);
        # the P/R columns count every pano. See rampnet/detection_eval.aggregate.
        if any(r.ap is not None for _, r in rows):
            print("AP: all-point interpolated, over the recall-confirmed panos; "
                  "'-' = no calibrated per-box score.")

    if args.sweep:
        for label, run in runs:
            if has_confidences(run.scored):
                print_sweep(label, sweep_rows(run.scored, radius_sq))
        print()

    if args.pr_out:
        curves = [(label, r) for label, r in rows if r.pr_curve]
        if curves:
            write_pr_curves(args.pr_out, curves)
        else:
            print("No model produced a PR curve (needs per-detection confidences).")

    # Cross-check: RampNet's own verdict-based P/R (the published definition).
    confs_by_pid = {pid: [d["confidence"] for d in records[pid]["detections"]] for pid in verdicts}
    pools = collect(verdicts, confs_by_pid)
    print()
    print(format_report("RampNet verdict-based cross-check", pools))
    # collect() records verdict/results mismatches as notes and leaves surfacing to
    # the caller; print them so a silently-skipped pano is visible, not swallowed.
    for w in pools.warnings:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
