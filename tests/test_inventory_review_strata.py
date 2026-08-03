"""Unit tests for the review sheet's date strata (issue #96 §5k, §5l).

The strata exist to test one claim at a useful n: that records postdating their
survey are positionally worse (Charlotte, Fisher p = 3.9e-06 at n=5). That test
is only valid if the partition is honest — a record silently filed as "dated"
because a null sentinel parsed as a year would bias the very comparison the
strata were built to make. CPU only, no network.
"""
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import inventory_review_sheet as s  # noqa: E402


def rows_from(values, field="INSTALL_DATE"):
    return [{field: v, "lon": -122.3 + i * 1e-4, "lat": 47.6 + i * 1e-4}
            for i, v in enumerate(values)]


def test_partition_splits_on_the_cutoff_year():
    rows = rows_from(["2015-06-01", "2019-01-01", "2020-01-01", "2024-12-31"])
    out = s.year_strata(rows, range(len(rows)), "INSTALL_DATE", 2019)
    assert out["dated_before"] == [0, 1]      # cutoff year itself is "before"
    assert out["dated_after"] == [2, 3]
    assert out["undated"] == []


def test_missing_and_empty_are_undated():
    rows = rows_from([None, "", "   ", "2018-01-01"])
    out = s.year_strata(rows, range(4), "INSTALL_DATE", 2019)
    assert out["undated"] == [0, 1, 2]
    assert out["dated_before"] == [3]


def test_null_sentinels_are_undated_not_dated():
    """§5c paid for this twice. Boston's CONST_DATE is uniformly "18991230" --
    the spreadsheet zero date -- and the old placeholder was 2000-01. Either one
    read as a real install year files a record in the wrong stratum and biases
    the dated-vs-undated comparison these strata exist to make."""
    rows = rows_from(["18991230", "2000-01-01", "2018-06-01"])
    out = s.year_strata(rows, range(3), "INSTALL_DATE", 2019)
    assert out["undated"] == [0, 1]
    assert out["dated_before"] == [2]


def test_epoch_milliseconds_are_understood():
    """ArcGIS FeatureServer emits epoch ms -- Seattle's INSTALL_DATE is this."""
    rows = rows_from([1529366400000, 1700000000000])   # 2018, 2023
    out = s.year_strata(rows, range(2), "INSTALL_DATE", 2019)
    assert out["dated_before"] == [0]
    assert out["dated_after"] == [1]


def test_the_partition_is_a_partition():
    rows = rows_from(["2015", None, "2024", "18991230", 1529366400000, ""])
    out = s.year_strata(rows, range(len(rows)), "INSTALL_DATE", 2019)
    seen = sorted(i for v in out.values() for i in v)
    assert seen == list(range(len(rows)))
    assert sum(len(v) for v in out.values()) == len(rows)


def test_a_frame_subset_is_respected():
    """Strata must apply to the FRAME, not the whole inventory -- otherwise a
    --where-not exclusion silently comes back."""
    rows = rows_from(["2015", "2024", "2016"])
    out = s.year_strata(rows, [0, 2], "INSTALL_DATE", 2019)
    assert out["dated_before"] == [0, 2]
    assert out["dated_after"] == []


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def test_equal_allocation_across_strata():
    rows = rows_from(["2015"] * 50 + ["2024"] * 50 + [None] * 50)
    picked, stratum_of, sizes = s.sample_year_strata(
        rows, range(150), "INSTALL_DATE", 2019, 60, seed=1)
    assert sizes == {"dated_before": 50, "dated_after": 50, "undated": 50}
    got = {}
    for i in picked:
        got[stratum_of[i]] = got.get(stratum_of[i], 0) + 1
    assert got == {"dated_before": 20, "dated_after": 20, "undated": 20}


def test_a_short_stratum_contributes_all_it_has_and_is_not_redistributed():
    """A stratum that cannot be filled is a finding about the city. Topping it up
    from another stratum would hide that and silently unbalance the design."""
    rows = rows_from(["2015"] * 50 + ["2024"] * 3 + [None] * 50)
    picked, stratum_of, _ = s.sample_year_strata(
        rows, range(103), "INSTALL_DATE", 2019, 60, seed=1)
    got = {}
    for i in picked:
        got[stratum_of[i]] = got.get(stratum_of[i], 0) + 1
    assert got["dated_after"] == 3
    assert got["dated_before"] == 20 and got["undated"] == 20
    assert len(picked) == 43


def test_sampling_is_deterministic_and_seed_sensitive():
    rows = rows_from(["2015"] * 40 + ["2024"] * 40 + [None] * 40)
    a = s.sample_year_strata(rows, range(120), "INSTALL_DATE", 2019, 30, seed=7)[0]
    b = s.sample_year_strata(rows, range(120), "INSTALL_DATE", 2019, 30, seed=7)[0]
    c = s.sample_year_strata(rows, range(120), "INSTALL_DATE", 2019, 30, seed=8)[0]
    assert a == b
    assert a != c


def test_strata_draw_independently_rather_than_sharing_one_shuffle():
    """Each stratum gets its own derived seed, so one stratum's contents cannot
    shift which records another stratum draws."""
    base = ["2015"] * 40 + ["2024"] * 40 + [None] * 40
    a = s.sample_year_strata(rows_from(base), range(120),
                             "INSTALL_DATE", 2019, 30, seed=3)
    changed = list(base)
    changed[45] = "2026"                       # still dated_after, different row
    b = s.sample_year_strata(rows_from(changed), range(120),
                             "INSTALL_DATE", 2019, 30, seed=3)
    before_a = sorted(i for i in a[0] if a[1][i] == "dated_before")
    before_b = sorted(i for i in b[0] if b[1][i] == "dated_before")
    assert before_a == before_b


def test_strata_names_are_stable():
    assert s.YEAR_STRATA == ("dated_before", "dated_after", "undated")


# --------------------------------------------------------------------------- #
# the export path
# --------------------------------------------------------------------------- #
def test_the_exported_record_carries_its_stratum():
    """The in-page export rebuilds each record from CHIPS, so a field present
    only in the verdicts template is DROPPED on export. That happened once: the
    whole point of stratifying is lost if the reviewer's own file cannot say
    which stratum a verdict came from, and recovering it needs a re-join against
    a template the reviewer does not have."""
    src = io.open(os.path.join(REPO, "scripts", "analysis",
                            "inventory_review_sheet.py"), encoding="utf-8").read()
    export = src.split('document.getElementById("export")')[1]
    assert "stratum" in export.split("URL.createObjectURL")[0]


def test_the_chip_dict_carries_the_stratum_not_just_the_verdict():
    src = io.open(os.path.join(REPO, "scripts", "analysis",
                            "inventory_review_sheet.py"), encoding="utf-8").read()
    chip_append = src.split("chips.append({")[1].split("})")[0]
    assert "stratum" in chip_append
