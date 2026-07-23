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


def load_bundle(bundle_dir):
    """Return (records_by_pid, verdicts_panos, panos_dir) for a benchmark bundle."""
    records = {}
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    verdicts = json.load(open(os.path.join(bundle_dir, "verdicts.json"), encoding="utf-8"))["panos"]
    return records, verdicts, os.path.join(bundle_dir, "panos")


def score_model(detector, records, verdicts, panos_dir, radius_sq):
    """Run one detector over every reviewed pano and aggregate the score."""
    pano_scores = []
    for pid, entry in verdicts.items():
        rec = records[pid]
        dets = rec["detections"]
        gt = build_ground_truth(dets, entry["dets"], entry["missed"], entry["no_missed"])
        sample = PanoSample(
            pano_id=pid,
            image_path=os.path.join(panos_dir, f"{pid}.jpg"),
            width=rec["pano"].get("width"),
            height=rec["pano"].get("height"),
            meta=rec["pano"],
        )
        preds = detector.detect(sample)
        pano_scores.append(score_pano(preds, gt, radius_sq=radius_sq))
    return aggregate(pano_scores)


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
    args = ap.parse_args()

    load_dotenv(str(REPO_ROOT))
    records, verdicts, panos_dir = load_bundle(args.bundle)
    if args.limit:
        verdicts = dict(list(verdicts.items())[:args.limit])
    radius_sq = radius_sq_for(args.radius)
    specs = [parse_model_spec(t) for t in args.models.split(",") if t.strip()]

    print(f"Bundle: {args.bundle}  ({len(verdicts)} reviewed panos)  "
          f"match radius {args.radius}  ground truth: reviewer-confirmed ramps + missed marks\n")

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
            report = score_model(detector, records, verdicts, panos_dir, radius_sq)
        except (NotImplementedError, ImportError, RuntimeError) as e:
            # Scaffolded VLM detectors, a missing client lib, or a missing API key:
            # skip the model with a clear note rather than crashing the whole run.
            print(f"[{label}] not runnable yet: {e}\n")
            continue
        rows.append((label, report))

    if rows:
        print_table(rows)

    # Cross-check: RampNet's own verdict-based P/R (the published definition).
    confs_by_pid = {pid: [d["confidence"] for d in records[pid]["detections"]] for pid in verdicts}
    pools = collect(verdicts, confs_by_pid)
    print()
    print(format_report("RampNet verdict-based cross-check", pools))


if __name__ == "__main__":
    main()
