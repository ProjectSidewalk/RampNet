"""Score a validation benchmark bundle: precision/recall + a threshold sweep.

Reads a bundle directory (``benchmark/<city>/`` with ``records.jsonl`` +
``verdicts.json``) — the self-contained, image-free scoring data — and reports
precision, recall, and a confidence-threshold sweep via :mod:`rampnet.validation`,
both overall and on the unbiased subset (excluding the always-included densest
"top" panos). This is the CLI around the scorer; the gallery that produces
``verdicts.json`` is tracked in issue #26.

If the verdicts file carries a ``review_notes`` block (the reviewer's caveats about
the review itself — see :mod:`rampnet.validation`), it is printed **before** the
numbers, and any per-pano ``note`` after them. Neither affects scoring; they are here
so nobody reads a precision figure off this output without the caveat attached to it.

    python scripts/score_validation.py benchmark/richmond
    python scripts/score_validation.py benchmark/bend --assume-scanned
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root on the path so `rampnet` imports without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rampnet.validation import collect, format_report, format_review_notes  # noqa: E402


def load_bundle(bundle_dir):
    """Returns (confs_by_pid, panos, review_notes) for a benchmark bundle dir."""
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
    return confs_by_pid, verdicts["panos"], verdicts.get("review_notes")


def main():
    ap = argparse.ArgumentParser(description="Score a validation benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir with records.jsonl + verdicts.json (e.g. benchmark/richmond).")
    ap.add_argument("--assume-scanned", action="store_true",
                    help="Count every fully-judged pano toward recall (reviewer attestation).")
    ap.add_argument("--lenient-duplicates", action="store_true",
                    help="Score 'duplicate' detections as redundant (abstained) instead of "
                         "the default false positive. The headline number uses the default.")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):  # tolerate cp1252 consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    confs_by_pid, panos, review_notes = load_bundle(args.bundle)
    lenient = args.lenient_duplicates

    # Caveats first: whoever reads a precision number off this output has already read
    # the reviewer's warning about it. Scoring itself never looks at these notes.
    banner = format_review_notes(review_notes)
    if banner:
        print(banner)
        print()

    pools = collect(panos, confs_by_pid, assume_scanned=args.assume_scanned,
                    lenient_duplicates=lenient)
    for w in pools.warnings:
        print(f"! {w}")
    print(format_report("All reviewed panos", pools))
    print()

    unbiased = collect(panos, confs_by_pid, exclude_top=True,
                       assume_scanned=args.assume_scanned, lenient_duplicates=lenient)
    if unbiased.n_seen != pools.n_seen:  # top panos existed
        print(format_report("Unbiased subset (random + empty samples only)", unbiased))
        print()

    if pools.n_duplicate:
        mode = ("lenient: duplicates abstain (redundant, excluded)" if lenient
                else "default: duplicates scored as false positives")
        print(f"Duplicate scoring — {mode}. "
              f"Re-run with{'out' if lenient else ''} --lenient-duplicates for the other variant.")
    print("Recall = per-pano-comprehensive, as judged by the reviewer on the sampled panos.")

    # Per-pano notes: the reviewer's record of individual judgment calls. Like
    # review_notes they never touch the metrics, but they explain them.
    noted = [(pid, e["note"]) for pid, e in panos.items() if e.get("note")]
    if noted:
        print(f"\n--- Reviewer notes on individual panos ({len(noted)}) ---")
        for pid, note in noted:
            print(f"  {pid}: {note}")


if __name__ == "__main__":
    main()
