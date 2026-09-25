"""The committed Croissant files (croissant/*.json, issue #150) stay valid and in step with git.

Offline and dependency-free: parses both files as JSON-LD documents, checks the required
Croissant / RAI / GeoCroissant keys, and re-derives the benchmark's per-split extents from the
committed bundles and the dataset's city boxes from the committed government inventories. The
MLCommons reference validator runs when ``mlcroissant`` is installed (``requirements-dev.txt`` carries
it, so CI runs it). The Hub check (``scripts/validate_croissant.py --hub``) needs the network and is
not run here; its drift and pagination logic is tested below against canned responses.
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


def test_content_url_is_cloneable_never_a_bare_sha(named_doc):
    # mlcroissant 1.1.0 clones a Hub URL only as the bare repository or `.../tree/refs%2F<ref>`;
    # `.../tree/<sha>` reaches `git clone` verbatim and fails (PR #190 re-review, R1).
    name, doc = named_doc
    repo = next(d for d in doc["distribution"] if d["@id"] == "repo")
    base = "https://huggingface.co/datasets/projectsidewalk/" + name
    url = repo["contentUrl"]
    assert url == base or url.startswith(base + "/tree/refs%2F"), url
    assert vc.content_ref(doc) == "refs/heads/main"

    for good, ref in ((base, "refs/heads/main"),
                      (base + "/tree/refs%2Ftags%2Fv1.0.0", "refs/tags/v1.0.0")):
        ok = copy.deepcopy(doc)
        next(d for d in ok["distribution"] if d["@id"] == "repo")["contentUrl"] = good
        assert not any("contentUrl" in p for p in vc.check_structure(ok))
        assert vc.content_ref(ok) == ref
    bad = copy.deepcopy(doc)
    next(d for d in bad["distribution"] if d["@id"] == "repo")["contentUrl"] = (
        base + "/tree/" + vc.pinned_revision(doc))
    assert any("contentUrl" in p for p in vc.check_structure(bad))


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


def test_dataset_boxes_match_committed_location_data():
    doc = vc.load(vc.FILES["rampnet-dataset"])
    assert vc.check_dataset_boxes(doc) == []

    shifted = copy.deepcopy(doc)
    shifted["spatialCoverage"][0]["geo"]["box"] = "40.5 -74.25478 40.91256 -73.70028"
    shifted["geocr:spatialBias"] = shifted["geocr:spatialBias"].replace("78.8%", "78.2%")
    problems = vc.check_dataset_boxes(shifted)
    assert any("New York City" in p and "box" in p for p in problems)
    assert any("geocr:spatialBias" in p for p in problems)


def test_context_is_appendix_1_plus_geocr(named_doc):
    _, doc = named_doc
    assert doc["@context"] == vc.CONTEXT
    broken = copy.deepcopy(doc)
    broken["@context"]["samplingRate"] = "cr:samplingRate"
    assert any("@context" in p and "samplingRate" in p for p in vc.check_structure(broken))


def test_links_are_pinned_and_no_bare_repo_paths(named_doc):
    _, doc = named_doc
    assert vc.check_links(doc) == []
    broken = copy.deepcopy(doc)
    broken["rai:dataBiases"] += (" See https://github.com/ProjectSidewalk/RampNet/blob/main/docs/seam.md"
                                 " and docs/data_provenance.md section 3.")
    problems = vc.check_links(broken)
    assert any("unpinned" in p and "'main'" in p for p in problems)
    assert any("docs/data_provenance.md" in p for p in problems)


def test_benchmark_record_sets_share_one_key_rule():
    doc = vc.load(vc.FILES["rampnet-benchmark"])
    broken = copy.deepcopy(doc)
    vc.record_set(broken, "native")["key"] = {"@id": "native/pano_id"}
    assert any("native is keyed" in p for p in vc.check_benchmark_extents(broken))


def test_release_gate_rejects_placeholders(named_doc):
    name, doc = named_doc
    problems = vc.check_release(doc)            # not minted yet: must fail
    assert any("identifier" in p for p in problems), name
    assert any("NOT-YET" in p for p in problems), name

    minted = copy.deepcopy(doc)
    minted["identifier"] = "https://doi.org/10.57967/hf/0000000"
    minted["citeAs"] = minted["citeAs"].replace(
        "DOI: forthcoming", "DOI: 10.57967/hf/0000000").replace(
        vc.DOI_PLACEHOLDER, "DOI: 10.57967/hf/0000000")
    assert vc.check_release(minted) == []
    minted["version"] = ""
    assert any("version" in p for p in vc.check_release(minted))


def test_hub_check_fails_on_drift_and_follows_pagination(monkeypatch):
    doc = vc.load(vc.FILES["rampnet-benchmark"])
    rev = vc.pinned_revision(doc)
    rows = vc.record_set(doc, "file_manifest")["data"]
    entries = [{"type": "file", "path": r["file_manifest/path"], "size": r["file_manifest/bytes"],
                "lfs": {"oid": r["file_manifest/sha256"]}} for r in rows]
    tree_url = (vc.HUB_API + "/tree/{}?recursive=true").format("rampnet-benchmark", rev)
    pages = {tree_url: (entries[:10], "page2"), "page2": (entries[10:], None)}

    def fake(url, main_sha):
        if url == (vc.HUB_API + "/refs").format("rampnet-benchmark"):
            return {"branches": [{"ref": "refs/heads/main", "targetCommit": main_sha}],
                    "tags": [{"ref": "refs/tags/v1.0.0", "targetCommit": rev}]}, None
        return pages[url]

    monkeypatch.setattr(vc, "_get_json", lambda url: fake(url, rev))
    assert vc.check_hub("rampnet-benchmark", doc) == []            # needs both pages to pass

    monkeypatch.setattr(vc, "_get_json", lambda url: fake(url, "f" * 40))
    assert any("has moved" in p for p in vc.check_hub("rampnet-benchmark", doc))

    # The tag route: contentUrl names a tag, and --hub resolves the tag, not main.
    tagged = copy.deepcopy(doc)
    next(d for d in tagged["distribution"] if d["@id"] == "repo")["contentUrl"] = (
        "https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark/tree/refs%2Ftags%2Fv1.0.0")
    assert vc.check_hub("rampnet-benchmark", tagged) == []


def test_mlcroissant_reference_validator(named_doc):
    pytest.importorskip("mlcroissant")
    name, _ = named_doc
    assert vc.check_mlcroissant(vc.FILES[name]) == []
