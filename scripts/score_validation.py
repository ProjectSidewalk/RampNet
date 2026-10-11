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

``--exclude-train-overlap`` drops the split's panoramas that are also in RampNet's
training data, as listed in ``benchmark/train_overlap.json`` (written by
``scripts/analysis/train_overlap_check.py``; #127). Only ``bend`` has any. It is not the
default: the published headline scores every reviewed panorama.

    python scripts/score_validation.py benchmark/richmond
    python scripts/score_validation.py benchmark/bend --assume-scanned
    python scripts/score_validation.py benchmark/bend --exclude-train-overlap
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root on the path so `rampnet` imports without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from rampnet.validation import collect, format_report, format_review_notes  # noqa: E402

TRAIN_OVERLAP = REPO_ROOT / "benchmark" / "train_overlap.json"


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


def drop_train_overlap(panos, split, overlap_path=TRAIN_OVERLAP):
    """(panos without the split's training-overlap ids, number dropped).

    Exits if the overlap file is missing or has no entry for the split: silently scoring
    everything would print a number labelled as overlap-free that is not.
    """
    path = Path(overlap_path)
    if not path.is_file():
        sys.exit(f"--exclude-train-overlap: {path} not found "
                 "(run scripts/analysis/train_overlap_check.py)")
    overlap = json.loads(path.read_text(encoding="utf-8")).get("overlap", {})
    if split not in overlap:
        sys.exit(f"--exclude-train-overlap: {path} has no entry for {split!r}")
    ids = set(overlap[split])
    kept = {pid: entry for pid, entry in panos.items() if pid not in ids}
    return kept, len(panos) - len(kept)


def main():
    ap = argparse.ArgumentParser(description="Score a validation benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir with records.jsonl + verdicts.json (e.g. benchmark/richmond).")
    ap.add_argument("--assume-scanned", action="store_true",
                    help="Count every fully-judged pano toward recall (reviewer attestation).")
    ap.add_argument("--lenient-duplicates", action="store_true",
                    help="Score 'duplicate' detections as redundant (abstained) instead of "
                         "the default false positive. The headline number uses the default.")
    ap.add_argument("--exclude-train-overlap", action="store_true",
                    help="Drop panos that are also in rampnet-dataset's train/validation splits, "
                         "per benchmark/train_overlap.json. Not the published headline.")
    ap.add_argument("--train-overlap", type=Path, default=TRAIN_OVERLAP,
                    help="Overlap file for --exclude-train-overlap (default: %(default)s).")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):  # tolerate cp1252 consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    confs_by_pid, panos, review_notes = load_bundle(args.bundle)
    lenient = args.lenient_duplicates
    split = Path(args.bundle).resolve().name
    n_excluded = None
    if args.exclude_train_overlap:
        panos, n_excluded = drop_train_overlap(panos, split, args.train_overlap)

    # Caveats first: whoever reads a precision number off this output has already read
    # the reviewer's warning about it. Scoring itself never looks at these notes.
    banner = format_review_notes(review_notes)
    if banner:
        print(banner)
        print()
    if n_excluded is not None:
        print(f"Excluded {n_excluded} training-overlap panoramas ({split})")
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
