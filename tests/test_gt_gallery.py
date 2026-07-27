"""Guards for the GT labeler's note round-trip (scripts/gt_gallery.py).

The reviewer's caveats — a split-level ``review_notes`` block and per-pano ``note``
strings — live in verdicts.json alongside the verdicts. Re-reviewing a city re-runs
this gallery over the existing file, so the notes have to survive the trip
bundle -> viewer -> export. A silent drop would be invisible (the numbers are
unchanged) and would destroy exactly the context the notes exist to preserve.

    pytest tests/test_gt_gallery.py -v
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from gt_gallery import build_html, initial_verdicts, load_bundle  # noqa: E402

BUNDLE = {
    "run_key": "testville", "run_name": "testville",
    "review_notes": {"reviewer": "jonf", "confidence": "low",
                     "summary": "Different city, different rubric.",
                     "caveats": ["Corner aprons were counted as two ramps."]},
    "panos": {
        "P1": {"group": "random", "dets": [True], "missed": [], "no_missed": True,
               "note": "one sweeping apron, counted as two"},
        "P2": {"group": "empty", "dets": [], "missed": [], "no_missed": True},
    },
}


def _write_bundle(tmp_path, with_verdicts=True):
    d = tmp_path / "testville"
    (d / "panos").mkdir(parents=True)
    (d / "records.jsonl").write_text(
        json.dumps({"pano": {"panorama_id": "P1"}, "detections": [{"confidence": 0.9}]}) + "\n"
        + json.dumps({"pano": {"panorama_id": "P2"}, "detections": []}) + "\n",
        encoding="utf-8")
    if with_verdicts:
        (d / "verdicts.json").write_text(json.dumps(BUNDLE), encoding="utf-8")
    return d


def test_load_bundle_reads_review_notes(tmp_path):
    *_, review_notes = load_bundle(_write_bundle(tmp_path))
    assert review_notes["confidence"] == "low"
    assert review_notes["caveats"] == BUNDLE["review_notes"]["caveats"]


def test_load_bundle_without_verdicts_returns_empty_notes(tmp_path):
    *_, review_notes = load_bundle(_write_bundle(tmp_path, with_verdicts=False))
    assert review_notes == {}


def test_initial_verdicts_carries_per_pano_note(tmp_path):
    initial = initial_verdicts(BUNDLE["panos"])
    assert initial["P1"]["note"] == "one sweeping apron, counted as two"
    assert initial["P2"]["note"] == ""  # absent note prefills blank, never None


def test_build_html_embeds_notes_and_leaves_no_placeholders(tmp_path):
    records, panos_dir, verdicts_panos, run_key, run_name, notes = load_bundle(
        _write_bundle(tmp_path))
    html = build_html([], initial_verdicts(verdicts_panos), run_key, run_name,
                      "records.jsonl", notes)
    assert "Different city, different rubric." in html
    assert "one sweeping apron, counted as two" in html
    assert "__" not in html.replace("__pycache__", "")  # every placeholder substituted


def test_build_html_without_notes_embeds_empty_object():
    html = build_html([], {}, "k", "n", "records.jsonl")
    assert "const INITIAL_NOTES = {};" in html
