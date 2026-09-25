"""Which detections docs/detection_recall_analysis.md's recall tables come from (#171).

The doc gave two different recalls at threshold 0.55 for the same distance bands: §1 reads
the committed deployment detections (``records.jsonl``), §5 reads a re-run of inference
(``overlap_test.py`` -> ``analysis_out/overlap.json``). The two differ on bend only. These
tests pin that, from committed files, so the doc's statement of it cannot drift.

The distance bands themselves need ``gt_depth_da3.json`` (Depth Anything 3 depths), which is
not committed, so the per-band numbers are not checked here; the doc says so beside them.

CPU only, no network, committed files only.
"""
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAP = os.path.join(REPO, "analysis_out", "overlap.json")
DEPTH_112 = os.path.join(REPO, "analysis_out", "recall_by_depth_112.json")
DOC = os.path.join(REPO, "docs", "detection_recall_analysis.md")
THRESHOLDS = ("0.55", "0.35", "0.25", "0.15")


@pytest.fixture(scope="module")
def overlap():
    with open(OVERLAP, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def records_points():
    """Richmond + bend GT points with the committed-records hit, from #112's rows."""
    with open(DEPTH_112, encoding="utf-8") as fh:
        data = json.load(fh)
    return data, [p for p in data["points"] if p["city"] in ("richmond", "bend")]


def _key(city, pano, x, y):
    return (city, pano, round(x, 4), round(y, 4))


def test_section_1_population_is_the_committed_records(records_points):
    data, points = records_points
    rep = data["tables"]["published_reproduction"]
    assert (rep["hit"], rep["n"]) == (487, 637)
    assert rep["per_city"] == {"bend": {"hit": 249, "n": 327},
                               "richmond": {"hit": 238, "n": 310}}
    assert sum(p["hit"] for p in points) == 487 and len(points) == 637


def test_section_3_recalls_reproduce_from_overlap_json(overlap):
    """Every recall in §3's table, per city and threshold, from the committed per-ramp hits."""
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    rows = re.findall(r"^\| \**(0\.\d\d)\** (?:\(deployed\) )?\| [\d.]+ / ([\d.]+) / \S+ \| "
                      r"[\d.]+ / ([\d.]+) / \S+ \|$", doc, flags=re.M)
    table = {thr: (float(rich), float(bend)) for thr, rich, bend in rows}
    assert set(table) == set(THRESHOLDS), table
    for thr in THRESHOLDS:
        for i, city in enumerate(("richmond", "bend")):
            pts = [r for r in overlap if r["city"] == city]
            recall = sum(r[f"t{thr}"] for r in pts) / len(pts)
            assert round(recall, 3) == table[thr][i], (city, thr, recall)


def test_the_rerun_differs_from_the_records_on_bend_only(overlap, records_points):
    """The §1-vs-§5 disagreement, ramp by ramp: 10 bend ramps, 6 lost and 4 gained."""
    _, points = records_points
    rerun = {_key(r["city"], r["pid"], r["x"], r["y"]): r["t0.55"] for r in overlap}
    assert len(rerun) == len(overlap) == 637
    changed = [(p["city"], p["hit"]) for p in points
               if rerun[_key(p["city"], p["pano"], p["x"], p["y"])] != p["hit"]]
    assert len(changed) == 10
    assert {c for c, _ in changed} == {"bend"}
    assert sum(1 for _, was in changed if was) == 6          # lost by the re-run
    assert sum(1 for _, was in changed if not was) == 4      # gained by the re-run
    assert sum(rerun.values()) == 485


def test_the_doc_states_the_resolution():
    with open(DOC, encoding="utf-8") as fh:
        prose = re.sub(r"\s+", " ", fh.read())
    assert "10 GT ramps change state (6 lost, 4 gained)" in prose
    assert "The 0.55 column here is not §1's table" in prose
    assert "reproduces the committed `records.jsonl` exactly." not in prose
