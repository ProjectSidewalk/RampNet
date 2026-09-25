"""The committed Croissant files (croissant/*.json, issue #150) stay valid and in step with git.

Offline and dependency-free: parses both files as JSON-LD documents, checks the required
Croissant / RAI / GeoCroissant keys, and re-derives the benchmark's per-split extents from the
committed bundles. The MLCommons reference validator runs only when ``mlcroissant`` is installed;
the Hub hash check (``scripts/validate_croissant.py --hub``) needs the network and is not run here.
"""
import copy
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_croissant",
                                               REPO / "scripts" / "validate_croissant.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


@pytest.fixture(scope="module", params=sorted(vc.FILES))
def named_doc(request):
    return request.param, vc.load(vc.FILES[request.param])


def test_structure_is_clean(named_doc):
    name, doc = named_doc
    assert vc.check_structure(doc) == [], name


def test_benchmark_extents_match_committed_bundles():
    doc = vc.load(vc.FILES["rampnet-benchmark"])
    assert vc.check_benchmark_extents(doc) == []


def test_benchmark_splits_are_the_published_nine():
    # The two Laurens splits are in git but not on the Hub yet (benchmark/README.md); the
    # metadata describes the published repo, so they must not appear until they are pushed.
    doc = vc.load(vc.FILES["rampnet-benchmark"])
    names = [r["splits/name"] for r in vc.record_set(doc, "splits")["data"]]
    assert names == ["annapolis", "bend", "budapest_district5", "clovis", "gainesville",
                     "morgantown", "paterson", "richmond", "sao_paulo"]
    manifest = [r["file_manifest/path"] for r in vc.record_set(doc, "file_manifest")["data"]]
    assert len(manifest) == 4 * 9


def test_dataset_manifest_covers_every_shard():
    doc = vc.load(vc.FILES["rampnet-dataset"])
    paths = {r["file_manifest/path"] for r in vc.record_set(doc, "file_manifest")["data"]}
    for split in ("train", "val", "test"):
        want = {"{}/data-{:05d}-of-00128.parquet".format(split, i) for i in range(128)}
        assert want <= paths, split
    assert len(paths) == 384


def test_both_files_pin_a_hub_revision(named_doc):
    name, doc = named_doc
    assert vc.pinned_revision(doc) is not None, name


def test_doi_is_an_obvious_placeholder_until_minted(named_doc):
    _, doc = named_doc
    assert doc["identifier"] == vc.DOI_PLACEHOLDER or vc.DOI_RE.match(doc["identifier"])


def test_check_catches_planted_problems():
    doc = vc.load(vc.FILES["rampnet-benchmark"])
    broken = copy.deepcopy(doc)
    del broken["rai:dataBiases"]
    broken["conformsTo"] = ["http://mlcommons.org/croissant/1.1"]
    broken["identifier"] = "10.1234/not-a-url"
    problems = vc.check_structure(broken)
    assert any("rai:dataBiases" in p for p in problems)
    assert any("geo/1.0" in p for p in problems)
    assert any("identifier" in p for p in problems)

    shifted = copy.deepcopy(doc)
    vc.record_set(shifted, "split_extents")["data"][0]["split_extents/max_lat"] += 0.001
    assert any("max_lat" in p for p in vc.check_benchmark_extents(shifted))


def test_mlcroissant_reference_validator(named_doc):
    pytest.importorskip("mlcroissant")
    name, _ = named_doc
    assert vc.check_mlcroissant(vc.FILES[name]) == []
