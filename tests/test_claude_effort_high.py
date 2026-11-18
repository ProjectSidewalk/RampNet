"""The effort=high write-up's numbers, re-derived from committed files.

``docs/claude_effort_high.md`` pairs the Opus 5.5 effort=high leg against the effort=low leg.
Two checks, CPU only, no network, no ``.model_cache``:

1. every per-split P / R / F1 and boxes-per-pano figure in ``results.json`` re-scores from the
   published detections and the bundles (the bootstrap is skipped: it takes about a minute;
   ``python scripts/analysis/claude_effort_high.py --check`` re-derives everything);
2. every table row the doc prints is exactly what ``--markdown`` renders from ``results.json``,
   so the doc and the committed results cannot drift apart.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import claude_effort_high as M  # noqa: E402

DOC = os.path.join(REPO, "docs", "claude_effort_high.md")


@pytest.fixture(scope="module")
def results():
    with open(M.OUT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("split", M.SPLITS)
def test_scores_and_box_counts_rescore_from_published_detections(results, split):
    rows, panos, emitted = M.load_split(split)
    want = results["per_split"][split]
    assert len(panos) == want["n_panos"]
    for m in rows:
        assert M.cell(rows[m], panos) == want["scores"][m], (split, m)
    for leg in M.LEGS:
        assert round(emitted[leg] / len(panos), 3) == want["boxes_per_pano"][leg], (split, leg)


def test_every_table_row_in_the_doc_matches_results_json(results):
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    rows = [line for line in M.markdown(results).splitlines() if line.startswith("| ")]
    assert len(rows) > 30
    missing = [r for r in rows if r not in doc]
    assert not missing, missing
