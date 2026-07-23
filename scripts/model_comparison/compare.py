"""Compare curb-ramp detectors on a benchmark bundle.

Scores each selected model against the same model-agnostic ground truth derived
from the human review (see rampnet/detection_eval.py), so RampNet and the VLMs
are compared on equal footing. RampNet's verdict-based numbers are re-printed as
a cross-check.

    python scripts/model_comparison/compare.py benchmark/richmond --models rampnet
    python scripts/model_comparison/compare.py benchmark/richmond \
        --models rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash

Each --models token is a provider (rampnet/gemini/qwen) or provider:model_id to
pin a variant, so several models from the same provider compare side by side.
See docs/model_comparison.md.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))          # rampnet.* (editable install fallback)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # local detectors.py

from rampnet.detection_eval import (  # noqa: E402
    build_ground_truth, score_pano, aggregate, radius_sq_for, PANO_RADIUS_NORMALIZED,
)
from rampnet.validation import collect, format_report  # noqa: E402
from detectors import PanoSample, build_detector, parse_model_spec  # noqa: E402


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


def score_model(detector, records, verdicts, panos_dir, radius_sq, label, city, cache,
                max_consecutive_failures=10):
    """Run one detector over every reviewed pano and aggregate the score.

    Returns ``(report, failures)``. ``detector.prepare()`` runs first (outside the
    per-pano guard) so credential / dependency / not-wired errors propagate to the
    caller and skip the whole model. Each pano's detections are cached, so re-runs
    don't re-pay the API. A transient per-pano failure is recorded and skipped
    rather than crashing the run; ``max_consecutive_failures`` aborts the model
    early during an outage instead of burning budget."""
    detector.prepare()
    sig = detector.signature() if hasattr(detector, "signature") else None
    pano_scores, failures, consecutive = [], [], 0
    for pid, entry in verdicts.items():
        rec = records[pid]
        gt = build_ground_truth(rec["detections"], entry["dets"], entry["missed"], entry["no_missed"])
        key = cache_key(label, sig, city, pid) if sig is not None else None
        preds = cache.get(key) if key else None
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
        pano_scores.append(score_pano(preds, gt, radius_sq=radius_sq))
    return aggregate(pano_scores), failures


def _pct(x):
    return f"{x:.3f}"


def _ci(lo_hi):
    return f"({lo_hi[0]:.3f}-{lo_hi[1]:.3f})"


def print_table(rows):
    header = f"{'model':<22} {'P':>6} {'95% CI':>15} {'R':>6} {'95% CI':>15} {'F1':>6}   {'tp/fp/fn/ign':>16}"
    print(header)
    print("-" * len(header))
    for name, r in rows:
        counts = f"{r.tp}/{r.fp}/{r.fn}/{r.ignored}"
        print(f"{name:<22} {_pct(r.precision):>6} {_ci(r.precision_ci):>15} "
              f"{_pct(r.recall):>6} {_ci(r.recall_ci):>15} {_pct(r.f1):>6}   {counts:>16}")


def main():
    # Windows consoles are cp1252; avoid UnicodeEncodeError on stray bytes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="Compare curb-ramp detectors on a benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir (e.g. benchmark/richmond) with records.jsonl + verdicts.json.")
    ap.add_argument("--models", default="rampnet",
                    help="Comma-separated detectors. Each is a provider (rampnet/gemini/qwen, "
                         "using its default model) or provider:model_id to pin a variant, e.g. "
                         "'rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash'.")
    ap.add_argument("--gemini-model", default="gemini-3.6-flash")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL")
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective",
                    help="VLM input: 'perspective' reprojects the pano into rectilinear "
                         "views (fair); 'none' uses one whole-pano call (lower bound). "
                         "No effect on rampnet.")
    ap.add_argument("--radius", type=float, default=PANO_RADIUS_NORMALIZED,
                    help=f"Normalized match radius (default {PANO_RADIUS_NORMALIZED}).")
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

    rows = []
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
            report, failures = score_model(
                detector, records, verdicts, panos_dir, radius_sq, label, city, cache)
        except (NotImplementedError, ImportError, RuntimeError) as e:
            # Not-wired detector, missing client lib, or missing credentials: skip
            # the whole model with a clear note rather than crashing the run.
            print(f"[{label}] not runnable: {e}\n")
            continue
        rows.append((label, report))
        if failures:
            print(f"[{label}] {len(failures)} pano failure(s) isolated (excluded from the score):")
            for pid, msg in failures[:5]:
                print(f"    {pid}: {msg}")
            if len(failures) > 5:
                print(f"    ... and {len(failures) - 5} more")
            print()

    if rows:
        print_table(rows)

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
