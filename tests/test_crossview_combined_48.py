"""The combined cross-family table (#48) re-derives from committed predictions.

Checks medians and fallback rates only (fast); the bootstrap CIs are re-derived by running
``scripts/analysis/crossview_combined_48.py``.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import crossview_align_48 as H  # noqa: E402
import crossview_combined_48 as C  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(C.OUT_JSON), reason="no committed table")


def _table():
    with open(C.OUT_JSON, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw
    return json.loads(raw)


def test_combined_table_rederives():
    tab = _table()
    assert tab["config"]["pairs_sha256"] == H.PAIRS_SHA256
    assert [r["arm"] for r in tab["rows"]] == [a for a, *_ in C.ROWS]
    pairs = H.read_frozen_pairs()
    strata = {"all": list(range(len(pairs))),
              "gsv": [i for i, p in enumerate(pairs) if p["imagery"] == "gsv"],
              "mapillary": [i for i, p in enumerate(pairs) if p["imagery"] == "mapillary"]}
    for row in tab["rows"]:
        errs = H.arm_errors(pairs, C.load(row["arm"]))
        for s, idx in strata.items():
            if s not in row:
                continue
            assert round(float(np.median([errs[i][0] for i in idx])), 4) == \
                pytest.approx(row[s]["median_deg"], abs=1e-4), (row["arm"], s)
            assert float(np.mean([errs[i][3] for i in idx])) == \
                pytest.approx(row[s]["fallback_rate"], abs=1e-4), (row["arm"], s)


def test_richmond_only_arms_are_never_ranked_on_all_pairs():
    """A Richmond-only arm scored over all 300 would fall back on the 240 GSV pairs and be
    ranked as if it had run everywhere; the table must carry the Richmond stratum only."""
    for row in _table()["rows"]:
        if row["arm"] in C.RICHMOND_ONLY:
            assert row["richmond_only"] and set(row) & {"all", "gsv"} == set(), row["arm"]
            assert row["mapillary"]["n"] == 60
