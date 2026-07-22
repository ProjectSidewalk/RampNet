"""Score a validation benchmark bundle: precision/recall + a threshold sweep.

Reads a bundle directory (``benchmark/<city>/`` with ``records.jsonl`` +
``verdicts.json``) — the self-contained, image-free scoring data — and reports
precision, recall, and a confidence-threshold sweep via :mod:`rampnet.validation`,
both overall and on the unbiased subset (excluding the always-included densest
"top" panos). This is the CLI around the scorer; the gallery that produces
``verdicts.json`` is tracked in issue #26.

    python scripts/score_validation.py benchmark/richmond
    python scripts/score_validation.py benchmark/bend --assume-scanned
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root on the path so `rampnet` imports without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rampnet.validation import collect, format_report  # noqa: E402


def load_bundle(bundle_dir):
    d = Path(bundle_dir)
    records_path, verdicts_path = d / "records.jsonl", d / "verdicts.json"
    if not records_path.exists() or not verdicts_path.exists():
        sys.exit(f"Bundle must contain records.jsonl and verdicts.json: {d}")

    confs_by_pid = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            confs_by_pid[r["pano"]["panorama_id"]] = [d["confidence"] for d in r["detections"]]

    verdicts = json.load(open(verdicts_path, encoding="utf-8"))
    return confs_by_pid, verdicts["panos"]


def main():
    ap = argparse.ArgumentParser(description="Score a validation benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir with records.jsonl + verdicts.json (e.g. benchmark/richmond).")
    ap.add_argument("--assume-scanned", action="store_true",
                    help="Count every fully-judged pano toward recall (reviewer attestation).")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):  # tolerate cp1252 consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    confs_by_pid, panos = load_bundle(args.bundle)

    pools = collect(panos, confs_by_pid, assume_scanned=args.assume_scanned)
    for w in pools.warnings:
        print(f"! {w}")
    print(format_report("All reviewed panos", pools))
    print()

    unbiased = collect(panos, confs_by_pid, exclude_top=True, assume_scanned=args.assume_scanned)
    if unbiased.n_seen != pools.n_seen:  # top panos existed
        print(format_report("Unbiased subset (random + empty samples only)", unbiased))
        print()

    print("Recall = per-pano-comprehensive, as judged by the reviewer on the sampled panos.")


if __name__ == "__main__":
    main()
