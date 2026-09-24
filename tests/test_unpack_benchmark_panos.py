"""`scripts/unpack_benchmark_panos.py`: the Hub Parquet -> `benchmark/<city>/panos/` step.

CPU only, no network: the tests build a two-row Parquet in the export's own schema and
unpack it, so what is checked is the layout, the two hash checks and the resume path.
Skipped when pyarrow is not installed (it is not in requirements-dev.txt's minimum).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import unpack_benchmark_panos as up  # noqa: E402

SCHEMA = pa.schema([
    pa.field("pano_id", pa.string()),
    pa.field("city", pa.string()),
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("sha256", pa.string()),
])

PANOS = {"p1": b"\xff\xd8 one", "p2": b"\xff\xd8 two"}


def _row(pano, data, city="testville", sha=None):
    return {"pano_id": pano, "city": city, "image": {"bytes": data, "path": f"{pano}.jpg"},
            "width": 8, "height": 4, "sha256": sha or hashlib.sha256(data).hexdigest()}


def _write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), str(path))


def _write_manifest(benchmark, city, panos):
    d = benchmark / city
    d.mkdir(parents=True)
    body = {"city": city, "n": len(panos), "panos": {
        p: {"file": f"{p}.jpg", "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
            "width": 8, "height": 4} for p, b in panos.items()}}
    (d / "imagery_manifest.json").write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    parquet = tmp_path / "testville.parquet"
    _write_parquet(parquet, [_row(p, b) for p, b in PANOS.items()])
    benchmark = tmp_path / "benchmark"
    _write_manifest(benchmark, "testville", PANOS)
    return parquet, benchmark, tmp_path / "out"


def test_unpacks_into_a_checkout_shaped_root(fixture):
    parquet, benchmark, out = fixture
    assert up.main(["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
                    "--benchmark", str(benchmark)]) == 0
    for p, b in PANOS.items():
        assert (out / "benchmark" / "testville" / "panos" / f"{p}.jpg").read_bytes() == b
    record = json.loads((out / "unpack_manifest.json").read_text(encoding="utf-8"))
    assert record["cities"]["testville"]["n"] == 2
    assert record["cities"]["testville"]["written"] == 2
    assert record["cities"]["testville"]["checked_against_manifest"] is True
    assert record["cities"]["testville"]["panos"]["p1"]["sha256"] == \
        hashlib.sha256(PANOS["p1"]).hexdigest()


def test_a_rerun_rewrites_nothing_that_already_matches(fixture):
    parquet, benchmark, out = fixture
    args = ["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
            "--benchmark", str(benchmark)]
    up.main(args)
    # corrupt one file on disk: it must be rewritten, the other left alone
    (out / "benchmark" / "testville" / "panos" / "p2.jpg").write_bytes(b"garbage")
    up.main(args)
    record = json.loads((out / "unpack_manifest.json").read_text(encoding="utf-8"))
    assert record["cities"]["testville"]["written"] == 1
    assert record["cities"]["testville"]["already_present"] == 1
    assert (out / "benchmark" / "testville" / "panos" / "p2.jpg").read_bytes() == PANOS["p2"]


def test_bytes_that_do_not_hash_to_the_parquets_own_column_are_refused(tmp_path):
    parquet = tmp_path / "bad.parquet"
    _write_parquet(parquet, [_row("p1", PANOS["p1"], sha="0" * 64)])
    with pytest.raises(ValueError, match="corrupt"):
        up.unpack_parquet(str(parquet), str(tmp_path / "out"), "testville")


def test_bytes_that_differ_from_the_committed_manifest_are_refused(tmp_path):
    parquet = tmp_path / "other.parquet"
    _write_parquet(parquet, [_row("p1", b"\xff\xd8 not the reviewed image"), _row("p2", PANOS["p2"])])
    benchmark = tmp_path / "benchmark"
    _write_manifest(benchmark, "testville", PANOS)
    with pytest.raises(ValueError, match="different image"):
        up.main(["--out", str(tmp_path / "out"), "--cities", "testville",
                 "--parquet", str(parquet), "--benchmark", str(benchmark)])


def test_a_parquet_short_of_the_manifest_is_an_error(tmp_path):
    parquet = tmp_path / "short.parquet"
    _write_parquet(parquet, [_row("p1", PANOS["p1"])])
    benchmark = tmp_path / "benchmark"
    _write_manifest(benchmark, "testville", PANOS)
    with pytest.raises(ValueError, match="absent from the Parquet"):
        up.main(["--out", str(tmp_path / "out"), "--cities", "testville",
                 "--parquet", str(parquet), "--benchmark", str(benchmark)])


def test_without_a_manifest_only_the_parquets_own_hash_is_checked(tmp_path):
    parquet = tmp_path / "t.parquet"
    _write_parquet(parquet, [_row("p1", PANOS["p1"])])
    out = tmp_path / "out"
    assert up.main(["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
                    "--benchmark", str(tmp_path / "no_such_dir")]) == 0
    record = json.loads((out / "unpack_manifest.json").read_text(encoding="utf-8"))
    assert record["cities"]["testville"]["checked_against_manifest"] is False
