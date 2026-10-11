"""`scripts/unpack_benchmark_panos.py`: the Hub Parquet -> `benchmark/<city>/panos/` step.

CPU only, no network: the tests build a two-row Parquet in the export's own schema and
unpack it, so what is checked is the layout, the two hash checks and the resume path.
requirements-dev.txt lists pyarrow, so CI runs these; the importorskip only keeps an env
without it from erroring at collection.
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


def test_a_row_for_another_city_is_refused(tmp_path):
    parquet = tmp_path / "mixed.parquet"
    _write_parquet(parquet, [_row("p1", PANOS["p1"]), _row("p2", PANOS["p2"], city="elsewhere")])
    with pytest.raises(ValueError, match="city='elsewhere', expected 'testville'"):
        up.unpack_parquet(str(parquet), str(tmp_path / "out"), "testville")


def test_the_4096_config_checks_membership_not_the_native_hashes(fixture, tmp_path):
    """The 4096x2048 re-render has different bytes from the reviewed native files, so the
    committed manifest can only say which panos belong to the split."""
    _, benchmark, _ = fixture
    rerendered = {p: b + b" resized" for p, b in PANOS.items()}
    parquet = tmp_path / "t4096.parquet"
    _write_parquet(parquet, [_row(p, b) for p, b in rerendered.items()])
    out = tmp_path / "out4096"
    assert up.main(["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
                    "--benchmark", str(benchmark), "--config", "4096x2048"]) == 0
    assert (out / "benchmark" / "testville" / "panos" / "p1.jpg").read_bytes() == rerendered["p1"]
    # ...but a pano the split was never reviewed on is still refused
    stray = tmp_path / "stray.parquet"
    _write_parquet(stray, [_row("p9", b"\xff\xd8 stray")])
    with pytest.raises(ValueError, match="not in the committed imagery_manifest"):
        up.main(["--out", str(tmp_path / "out_stray"), "--cities", "testville",
                 "--parquet", str(stray), "--benchmark", str(benchmark), "--config", "4096x2048"])


def test_one_root_holds_one_config(fixture, tmp_path):
    """Both configs write benchmark/<city>/panos/, so unpacking the other config into a
    root would overwrite the reviewed native images one by one; it is refused instead."""
    parquet, benchmark, out = fixture
    assert up.main(["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
                    "--benchmark", str(benchmark)]) == 0
    with pytest.raises(SystemExit):
        up.main(["--out", str(out), "--cities", "testville", "--parquet", str(parquet),
                 "--benchmark", str(benchmark), "--config", "4096x2048"])
    assert (out / "benchmark" / "testville" / "panos" / "p1.jpg").read_bytes() == PANOS["p1"]


def test_manual_gold_is_refused_before_any_download(tmp_path, capsys):
    with pytest.raises(SystemExit):
        up.main(["--out", str(tmp_path / "out"), "--cities", "manual_gold"])
    assert "rampnet-dataset" in capsys.readouterr().err


def test_a_split_missing_from_the_hub_names_the_ones_that_are_there(tmp_path, monkeypatch):
    """No network: the two Hub calls are replaced. laurens_mapillary was not uploaded at
    63d5ffd0; the raw EntryNotFoundError said nothing about what to ask for instead."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import EntryNotFoundError

    def not_found(*a, **k):
        raise EntryNotFoundError("404")
    monkeypatch.setattr(hub, "hf_hub_download", not_found)
    monkeypatch.setattr(hub.HfApi, "list_repo_files", lambda self, *a, **k: [
        "data/native/bend.parquet", "data/native/richmond.parquet",
        "data/4096x2048/bend.parquet", "README.md"])
    with pytest.raises(up.SplitNotOnHub) as e:
        up.download(up.REPO_ID, "main", "native", "laurens_mapillary", str(tmp_path))
    msg = str(e.value)
    assert "data/native/laurens_mapillary.parquet" in msg
    assert "for native: bend, richmond." in msg
