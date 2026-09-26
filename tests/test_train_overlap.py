"""Guards for #127: the reviewer caveats and Bend's training overlap travel with the published rows.

The `records` config is what someone who never opens this repo loads. Before #127 it carried the
verdicts but not the reviewer's own caveats about them, and not the four Bend panoramas that are
also in RampNet's training data -- both lived only in git. These tests pin:

- the overlap list itself (`benchmark/train_overlap.json`): every split present, only reviewed ids,
  Bend's four, and LF/sorted bytes so a re-run can be diffed;
- what `build_records` writes from the committed bundles: the review_notes columns per split, the
  per-pano notes, and `train_overlap`;
- the refusal to export a split that has no overlap entry;
- the four-row Bend table quoted in the card and benchmark/README.md, re-derived from the bundle.

CPU only, no network: reads committed fixtures and writes to tmp_path.
"""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from export_benchmark import (  # noqa: E402
    BENCHMARK_SPLITS, MODEL_RES, NATIVE, RECORDS, RECORDS_FEATURES, RECORDS_SCHEMA,
    TRAIN_OVERLAP_NAME, build_records, records_only_note, records_only_splits)
from rampnet.validation import collect, total_ramps  # noqa: E402
from score_validation import drop_train_overlap, load_bundle  # noqa: E402
from train_overlap_check import dump, overlap_report  # noqa: E402

BENCHMARK = REPO_ROOT / "benchmark"
OVERLAP_PATH = BENCHMARK / TRAIN_OVERLAP_NAME
BEND_OVERLAP = {"6WC0hdAYRsSAcluKSs5iRg", "9kW9cxpuj7q8DMzf-ClrQQ",
                "DJ8Zp111zu6KnMZz-0PHgQ", "VgWpqFkTwCIROvM0z-DkOw"}
REVIEW_COLUMNS = ("reviewer", "reviewed_at", "review_confidence", "review_summary",
                  "review_caveats")


# ------------------------------------------------------------------- the committed overlap file

def _overlap():
    return json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))


def test_train_overlap_file_covers_every_split_and_only_reviewed_panos():
    overlap = _overlap()["overlap"]
    assert set(overlap) == set(BENCHMARK_SPLITS)
    for split, ids in overlap.items():
        reviewed = json.loads((BENCHMARK / split / "verdicts.json").read_text(
            encoding="utf-8"))["panos"]
        assert set(ids) <= set(reviewed), split
        assert ids == sorted(ids), split
    assert set(overlap["bend"]) == BEND_OVERLAP
    assert all(not ids for split, ids in overlap.items() if split != "bend")


def test_train_overlap_json_is_byte_stable():
    """LF, sorted keys, indent 2, trailing newline -- what train_overlap_check.dump writes."""
    raw = OVERLAP_PATH.read_bytes()
    assert b"\r" not in raw
    assert dump(json.loads(raw.decode("utf-8"))).encode("utf-8") == raw
    assert (json.dumps(json.loads(raw.decode("utf-8")), indent=2, sort_keys=True)
            + "\n").encode("utf-8") == raw


def test_overlap_report_intersects_every_split_against_the_union_of_dataset_splits():
    report = overlap_report({"a": {"x", "y"}, "b": {"z"}},
                            {"train": {"y", "q"}, "validation": {"z"}}, "2026-01-01",
                            ["train", "validation"])
    assert report["overlap"] == {"a": ["y"], "b": ["z"]}
    assert report["reviewed_counts"] == {"a": 2, "b": 1}
    assert report["dataset_split_sizes"] == {"train": 2, "validation": 1}


# ------------------------------------------------------------------------ the records schema

def test_records_schema_and_features_declare_the_same_columns():
    """Two parallel definitions; a column added to one only is published half-described."""
    assert RECORDS_SCHEMA.names == list(RECORDS_FEATURES)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("records")
    written = build_records(BENCHMARK, out)
    tables = dict((city, pq.read_table(str(out / "data" / RECORDS / "{}.parquet".format(city))))
                  for city, *_ in written)
    return written, tables


def test_every_split_is_built(built):
    written, tables = built
    assert sorted(tables) == sorted(BENCHMARK_SPLITS)


def test_build_records_carries_the_reviewer_caveats(built):
    _, tables = built
    budapest = tables["budapest_district5"].to_pylist()
    assert {r["review_confidence"] for r in budapest} == {"low"}
    assert all(r["review_caveats"] for r in budapest)
    assert all(r["review_summary"] for r in budapest)
    for split in ("gainesville", "paterson", "sao_paulo"):
        assert {r["review_confidence"] for r in tables[split].to_pylist()} == {"high"}, split
    # The Laurens sentinel travels verbatim rather than being coerced to a level.
    for split in ("laurens_gsv", "laurens_mapillary"):
        levels = {r["review_confidence"] for r in tables[split].to_pylist()}
        assert len(levels) == 1 and levels.pop().startswith("unrecorded"), split
    # No block => null, never [] or "" -- "not recorded", not "no caveats".
    for split in ("annapolis", "bend", "clovis", "morgantown", "richmond"):
        for row in tables[split].to_pylist():
            assert all(row[c] is None for c in REVIEW_COLUMNS), split


def test_per_pano_notes_travel_only_where_they_were_written(built):
    _, tables = built
    counts = dict((city, sum(1 for n in t.column("note").to_pylist() if n is not None))
                  for city, t in tables.items())
    assert counts.pop("laurens_mapillary") == 3
    assert set(counts.values()) == {0}


def test_build_records_flags_exactly_bends_four_overlap_panos(built):
    written, tables = built
    for city, table in tables.items():
        flags = table.column("train_overlap").to_pylist()
        assert None not in flags, city
        flagged = {p for p, f in zip(table.column("pano_id").to_pylist(), flags) if f}
        assert flagged == (BEND_OVERLAP if city == "bend" else set()), city
    assert dict((row[0], row[4]) for row in written)["bend"] == 4
    assert tables["bend"].num_rows == 110          # flagged, not dropped


def _minimal_benchmark(tmp_path, overlap):
    split = tmp_path / "bend"
    split.mkdir()
    (split / "records.jsonl").write_text(json.dumps(
        {"pano": {"panorama_id": "p1"}, "detections": []}) + "\n", encoding="utf-8")
    (split / "verdicts.json").write_text(json.dumps(
        {"panos": {"p1": {"group": "random", "dets": [], "missed": [], "no_missed": True}}}),
        encoding="utf-8")
    if overlap is not None:
        (tmp_path / TRAIN_OVERLAP_NAME).write_text(json.dumps({"overlap": overlap}),
                                                   encoding="utf-8")
    return tmp_path


def test_a_split_missing_from_the_overlap_file_is_refused(tmp_path):
    bench = _minimal_benchmark(tmp_path, {"clovis": []})
    with pytest.raises(SystemExit) as excinfo:
        build_records(bench, tmp_path / "out")
    assert "bend" in str(excinfo.value) and "train_overlap_check.py" in str(excinfo.value)


def test_a_missing_overlap_file_is_refused(tmp_path):
    bench = _minimal_benchmark(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        build_records(bench, tmp_path / "out")
    assert TRAIN_OVERLAP_NAME in str(excinfo.value)


def test_a_split_with_an_empty_entry_builds(tmp_path):
    bench = _minimal_benchmark(tmp_path, {"bend": []})
    (row,) = build_records(bench, tmp_path / "out")
    assert row[:5] == ("bend", 1, 0, 0, 0)


# ------------------------------------------------------------------------------ the card

def test_records_only_splits_are_named_on_the_card():
    index = {RECORDS: ["a", "b", "c"], NATIVE: ["a", "b"], MODEL_RES: ["a"]}
    assert records_only_splits(index) == ["c"]
    assert "`c`" in records_only_note(["c"])
    assert records_only_splits({RECORDS: ["a"], NATIVE: ["a"]}) == []
    assert records_only_note([]) == ""


# -------------------------------------------------------- the Bend table, from the fixtures

def _pr(panos, confs, exclude_top):
    pools = collect(panos, confs, exclude_top=exclude_top)
    correct = sum(1 for _, ok in pools.judged if ok)
    recall_tp = sum(1 for _, ok in pools.recall_judged if ok)
    return correct, len(pools.judged), recall_tp / total_ramps(pools)


def test_bend_numbers_with_and_without_overlap():
    """The four-row table in the card and benchmark/README.md."""
    confs, panos, _ = load_bundle(BENCHMARK / "bend")
    kept, n_dropped = drop_train_overlap(panos, "bend", OVERLAP_PATH)
    assert n_dropped == 4 and len(kept) == 106
    rows = [_pr(panos, confs, False), _pr(kept, confs, False),
            _pr(panos, confs, True), _pr(kept, confs, True)]
    assert [(c, n) for c, n, _ in rows] == [(248, 260), (241, 252), (208, 214), (201, 206)]
    assert [round(c / n, 3) for c, n, _ in rows] == [0.954, 0.956, 0.972, 0.976]
    assert [round(r, 3) for _, _, r in rows] == [0.758, 0.753, 0.738, 0.731]


def test_exclude_train_overlap_refuses_a_split_the_file_does_not_cover(tmp_path):
    path = tmp_path / TRAIN_OVERLAP_NAME
    path.write_text(json.dumps({"overlap": {"bend": []}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        drop_train_overlap({"p": {}}, "clovis", path)
    assert drop_train_overlap({"p": {}}, "bend", path) == ({"p": {}}, 0)

