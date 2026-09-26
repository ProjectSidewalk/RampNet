"""Guards on the generated tables in the data-sourcing docs (#145).

``scripts/analysis/sourcing_tables.py`` owns two blocks; these tests fail when either is
stale, when the tolerance curve drifts from the numbers the §5g prose was written
against, when a frozen inventory is missing from the §9 list, and when §5j's
hand-maintained bearing-residual table disagrees with its committed JSON.

Pure: reads only committed files. No network, no GPU.
"""
import json
import math
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import sourcing_tables as st  # noqa: E402
from scoreboard_render import splice  # noqa: E402


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def test_committed_docs_are_current():
    for path, tables in st.blocks().items():
        text = _read(path)
        for name in tables:
            assert st.BEGIN.format(name=name) in text, f"{name} block missing from {path}"
        assert splice(text, tables, begin=st.BEGIN, end=st.END) == text, \
            f"{path} is stale: re-run scripts/analysis/sourcing_tables.py"


def test_check_mode_passes_on_the_committed_docs():
    done = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "analysis",
                                                        "sourcing_tables.py"), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_tolerance_curve_reproduces_the_published_rows():
    """The eight rows §5g's prose was written against, pinned, so a regenerated JSON
    that drifts fails here rather than silently rewriting the table."""
    table = st.offset_tolerance_table(json.loads(_read(st.OFFSET_JSON)))
    rows = [line for line in table.splitlines()[2:]]
    assert rows == [
        "| 0.29 m *(Denver)* | **0.25%** |",
        "| 0.58 m | 2.1% |",
        "| 0.87 m | 5.5% |",
        "| 1.16 m | 9.4% |",
        "| 1.74 m | 16.5% |",
        "| 2.32 m | 22.9% |",
        "| 3.48 m | 32.5% |",
        "| 4.64 m | 39.7% |",
    ]


def test_the_snapshot_block_lists_every_manifest():
    manifests = st.load_manifests()
    assert len(manifests) >= 13
    text = _read(st.SOURCING_DOC)
    start = text.index(st.BEGIN.format(name="inventory-snapshots"))
    block = text[start:text.index(st.END.format(name="inventory-snapshots"), start)]
    for stem, m in manifests.items():
        assert f"`{stem}`" in block, stem
        assert f"`{m['sha256'][:12]}`" in block, stem


# --------------------------------------------------------------------------- #
# §5j stays hand-maintained -- so check it against its JSON instead
# --------------------------------------------------------------------------- #
BEARING_JSON = os.path.join(REPO, "analysis_out", "stage1_bearing_residual.json")
BEARING_ROW = re.compile(r"^\| \*\*(NYC|Portland|Bend)\*\* \|")
# The one cell the 4-dp JSON cannot reproduce: Bend's mean_deg is stored as 0.0355, a
# rounding tie, and the published table (rendered from the unrounded value) printed
# +0.036 while Python's formatting gives +0.035. This is why the table is not generated;
# see sourcing_tables.py. Bend's se_mean_deg (0.1055, also a tie) is recoverable, so it is
# checked below from sd_deg / sqrt(n_residuals) instead of being excused here.
KNOWN_TIES = {("bend", "mean_deg"): "+0.036"}


BEARING_HEADER = "| City | panos | gov records | matched | mean | s.e. |"


def _bearing_rows():
    """The city rows of §5j's table only. The §5 endpoint table has an NYC row too, which
    a whole-file match used to read and then silently overwrite with this one."""
    text = _read(st.LOCATION_DOC)
    start = text.index("## 5j. ")
    start = text.index(BEARING_HEADER, start)
    table = text[start:].split("\n\n", 1)[0]
    rows = {}
    for line in table.splitlines():
        if BEARING_ROW.match(line):
            cells = [c.strip().replace("**", "") for c in line.split("|")[1:-1]]
            assert cells[0].lower() not in rows, f"§5j: duplicate {cells[0]} row"
            rows[cells[0].lower()] = cells
    return rows


def _fmt(value, places, signed=False):
    s = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    return s.replace("-", "−")


def test_the_hand_bearing_table_agrees_with_its_json():
    cities = json.loads(_read(BEARING_JSON))["cities"]
    rows = _bearing_rows()
    assert set(rows) == {"nyc", "portland", "bend"}
    for city, cells in rows.items():
        c = cities[city]
        want = {
            "n_panos": f"{c['n_panos']:,}",
            "n_gov": f"{c['n_gov']:,}",
            "matched_frac": _fmt(c["matched_frac"], 3),
            "mean_deg": _fmt(c["mean_deg"], 3, signed=True) + "°",
            # From sd / sqrt(n), not the 4-dp se, which sits on a tie for Bend.
            "se_mean_deg": _fmt(c["sd_deg"] / math.sqrt(c["n_residuals"]), 3),
            "abs_median_deg": _fmt(c["abs_median_deg"], 2) + "°",
            "abs_p90_deg": _fmt(c["abs_p90_deg"], 2) + "°",
            "abs_median_m_at_median_range": _fmt(c["abs_median_m_at_median_range"], 2) + " m",
        }
        # A dropped or added column would otherwise make zip() skip checks silently.
        assert len(cells) == 1 + len(want) == 9, (city, cells)
        assert abs(c["sd_deg"] / math.sqrt(c["n_residuals"]) - c["se_mean_deg"]) < 5e-5, city
        for (key, expected), have in zip(want.items(), cells[1:]):
            if (city, key) in KNOWN_TIES:
                assert have.rstrip("°") == KNOWN_TIES[(city, key)], (city, key, have)
                continue
            assert have == expected, f"§5j {city} {key}: doc {have} vs JSON {expected}"


def test_the_nyc_bearing_row_is_pinned():
    """Pinned, so a JSON regeneration that drifts is caught even if the doc follows it."""
    assert _bearing_rows()["nyc"][1:] == [
        "10,273", "62,132", "0.847", "+0.055°", "0.026", "3.30°", "9.07°", "0.64 m"]


def test_first_sentence_skips_abbreviations():
    """The §9 note column keeps a note whole up to its first real sentence end."""
    assert st._first_sentence("U.S. Census data. More.") == "U.S. Census data."
    assert st._first_sentence("Near Main St. and 4th. More.") == "Near Main St. and 4th."
    assert st._first_sentence("One. Two.") == "One."
    assert st._first_sentence("no period") == "no period"
    assert st._first_sentence(None) == ""
