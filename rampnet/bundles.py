"""What a directory under ``benchmark/`` is, for scripts that discover splits by listing.

Three kinds of directory carry a ``records.jsonl``:

* a **scored split**: it has a review (``verdicts.json``) or independent labels
  (``gt_source.json``);
* a **borrowing bundle** (#48's neighbourhood bundles): its ``bundle.json`` borrows
  another split's verdicts, so counting it would score those judged panos twice;
* a **staged bundle** (bayonne, #159): imagery and detections exported ahead of the
  ground-truth review. Nothing about it can be scored yet.

Only the first is a split. A script that lists ``benchmark/*/`` and keeps the first
kind only stays reproducible when a city is staged before its review; one that keeps
everything with a ``records.jsonl`` exits on the staged bundle (PR #234 review, B1).

    >>> from rampnet.bundles import is_scored_split
    >>> is_scored_split("benchmark/richmond"), is_scored_split("benchmark/bayonne")
    (True, False)
"""
import os


def is_scored_split(bundle_dir):
    """True for a ``benchmark/<split>/`` directory that can be scored (see module doc)."""
    has = lambda name: os.path.exists(os.path.join(bundle_dir, name))  # noqa: E731
    return (has("records.jsonl") and not has("bundle.json")
            and (has("verdicts.json") or has("gt_source.json")))


def scored_splits(benchmark_dir):
    """Sorted names of every scored split under ``benchmark_dir``."""
    return sorted(n for n in os.listdir(benchmark_dir)
                  if is_scored_split(os.path.join(benchmark_dir, n)))
