"""Unit tests for the heatmap-cache dump script (#135, #138 finding 2).

Three regressions this pins:

- The label a dump gets must come from ``--label-prefix``, not a hardcoded
  ``run_a_epoch_``. A second run's ``--summary-csv`` (the #135 cosine rung's) used to
  produce ``run_a_epoch_N`` labels regardless, which land beside and overwrite Run A's
  committed dumps.
- ``--verify`` must fail loudly -- non-zero exit, a printed message -- when a cached
  checkpoint's fingerprint is not in the summary table, rather than silently skipping
  the check.
- The ``exclude_border`` recorded in a dump's signature must be the value the
  extractor actually used, read from ``stage_two/evaluate.py``, not a literal typed
  here that could drift from it (the #132 class of bug).
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "stage_two"))

import dump_peaks_from_cache as dump  # noqa: E402
import evaluate as ev  # noqa: E402


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "f1_at_protocol", "max_f1",
                                            "checkpoint_fingerprint"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_fingerprint_labels_uses_the_given_prefix(tmp_path):
    """A second run's summary must not be labelled with Run A's prefix.

    This is finding 2's core bug: ``fingerprint_labels`` used to hardcode
    ``run_a_epoch_{N}`` no matter which --summary-csv was passed, so a cosine-rung
    dump would be mislabelled as a Run A epoch and land in Run A's file.
    """
    summary = tmp_path / "summary.csv"
    write_summary(summary, [
        {"epoch": 1, "f1_at_protocol": 0.9, "max_f1": 0.91, "checkpoint_fingerprint": "aaa"},
        {"epoch": 2, "f1_at_protocol": 0.9, "max_f1": 0.91, "checkpoint_fingerprint": "bbb"},
    ])

    run_a = dump.fingerprint_labels(str(summary), "run_a_epoch_")
    rung = dump.fingerprint_labels(str(summary), "cosine_rung_epoch_")

    assert run_a == {"aaa": "run_a_epoch_1", "bbb": "run_a_epoch_2"}
    assert rung == {"aaa": "cosine_rung_epoch_1", "bbb": "cosine_rung_epoch_2"}


def make_cache(tmp_path, fingerprint, pano_ids=("pano_a", "pano_b")):
    """A minimal evaluate.py-shaped cache: <cache_dir>/heatmaps/<key>/<pano>_heatmap.npy."""
    cache_dir = tmp_path / "evaluate_cache"
    key_dir = cache_dir / "heatmaps" / f"{fingerprint}_manual_notta"
    key_dir.mkdir(parents=True)
    h, w = ev.MODEL_HEATMAP_SIZE
    for pano in pano_ids:
        heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap[h // 2, w // 2] = 0.9
        np.save(key_dir / f"{pano}_heatmap.npy", heatmap)
    return cache_dir


def test_verify_fails_loudly_on_an_unknown_fingerprint(tmp_path, capsys):
    """A cached checkpoint absent from --summary-csv must fail --verify, not skip it.

    Before this fix, ``if args.verify and fingerprint in committed`` silently did
    nothing when the fingerprint was not found -- a wrong --summary-csv (or a stray
    cache entry) produced a dump that nobody had verified and nothing said so.
    """
    cache_dir = make_cache(tmp_path, "deadbeef0000")
    summary = tmp_path / "summary.csv"
    # A summary that does not know about "deadbeef0000" at all.
    write_summary(summary, [
        {"epoch": 1, "f1_at_protocol": 0.9, "max_f1": 0.91, "checkpoint_fingerprint": "aaa"},
    ])
    out_dir = tmp_path / "out"

    rc = dump.main([
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--summary-csv", str(summary),
        "--label-prefix", "cosine_rung_epoch_",
        "--verify",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "NO MATCH" in captured.out
    assert "deadbeef0000" in captured.out
    # The dump is still written -- --verify reports a problem, it does not withhold
    # the file -- but under the unmatched-fingerprint label since none was found.
    written = list(out_dir.glob("ckpt_deadbeef0000__manual_gold.json"))
    assert len(written) == 1


def test_dump_matching_a_committed_fingerprint_verifies_ok(tmp_path, capsys):
    """The normal path: a fingerprint present in --summary-csv verifies and exits 0."""
    cache_dir = make_cache(tmp_path, "cafef00dfeed", pano_ids=())
    # No detections at all -> F1 and max-F1 are both 0, which is what the summary
    # below must also say, since --verify checks scored numbers against it.
    summary = tmp_path / "summary.csv"
    write_summary(summary, [
        {"epoch": 3, "f1_at_protocol": 0.0, "max_f1": 0.0,
         "checkpoint_fingerprint": "cafef00dfeed"},
    ])
    out_dir = tmp_path / "out"

    rc = dump.main([
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--summary-csv", str(summary),
        "--label-prefix", "cosine_rung_epoch_",
        "--verify",
    ])

    assert rc == 0
    payload = json.loads((out_dir / "cosine_rung_epoch_3__manual_gold.json").read_text())
    assert payload["model"] == "cosine_rung_epoch_3"
    # exclude_border must come from the extractor, not a literal copied here -- pin
    # it against evaluate.py's own constant so the two cannot silently diverge.
    assert payload["signature"]["exclude_border"] == ev.PEAK_EXCLUDE_BORDER


def test_exclude_border_constant_matches_peak_local_max_call():
    """evaluate.py's own extractor must use the constant dump_peaks_from_cache reads.

    Guards against the constant and the actual peak_local_max() call drifting apart
    -- the same class of bug #132 found when exclude_border was a bare literal.
    """
    import inspect
    src = inspect.getsource(ev.extract_peaks_from_heatmap)
    assert "exclude_border=PEAK_EXCLUDE_BORDER" in src
