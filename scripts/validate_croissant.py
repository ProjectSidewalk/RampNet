"""Check the hand-authored Croissant 1.1 + GeoCroissant 1.0 + RAI files in ``croissant/``.

Levels, cheapest first. The default is offline and needs only the standard library, so
``tests/test_croissant.py`` runs it on every PR:

* **structure** -- each file is JSON, uses the Croissant 1.1 Appendix 1 ``@context`` (see
  ``CONTEXT``), declares the three specs in ``conformsTo``, carries the required Croissant
  properties and the RAI / GeoCroissant keys this repo commits to, has unique ``@id`` values, and
  every field source points at a distribution that exists. The ``repo`` FileObject's
  ``contentUrl`` must be the bare Hub repository URL or a named ``tree/refs%2F...`` ref -- the only
  forms mlcroissant 1.1.0 can clone -- and the 40-hex revision the file describes is named in the
  ``repo`` and ``file_manifest`` descriptions.
* **links** -- every link into this GitHub repository is pinned to a commit or a release tag, and
  no text cites a bare repo-relative path, which would not resolve once the file is on the Hub.
* **derived numbers** -- the benchmark's ``split_extents`` record set (bounding box, capture-month
  range, panorama count and imagery source per split) is re-derived from the committed
  ``benchmark/<city>/records.jsonl`` + ``verdicts.json`` with the same "reviewed panoramas only"
  filter that ``scripts/export_benchmark.py`` applies, and ``(split, pano_id)`` is checked to be
  unique there. The dataset's three city boxes, record counts and NYC share are re-derived from the
  committed ``stage_one/dataset_generation/location_data/`` files with the parsers in
  ``combine_location_data.py``, after checking each file's sha256 against
  ``docs/data_provenance.md`` section 3.

Optional levels:

* ``--mlcroissant`` -- also run the MLCommons reference validator (``pip install mlcroissant``).
* ``--hub`` -- also (a) fail if the ref ``contentUrl`` resolves to on the Hub (``main``, or the named
  ref) is not the revision each file pins, and
  (b) fetch the tree listing at the pinned revision and check that every Parquet file's size and
  sha256 equal the ``file_manifest`` record set (network).
* ``--rebuild-records`` -- rebuild the benchmark's ``records`` config from the committed bundles
  with ``scripts/export_benchmark.py``'s own ``build_records`` into a temporary directory and
  compare each file's sha256 with ``file_manifest`` (needs ``pyarrow``; Parquet bytes depend on
  the pyarrow version, see ``docs/fair_metadata_150.md`` section 4).
* ``--load`` -- load data through the Croissant files with ``mlcroissant``: the benchmark
  ``records`` record set from that rebuild, and the dataset ``panoramas`` record set from a
  one-row synthetic shard with the Hub's schema (needs ``mlcroissant``, ``pyarrow``, ``Pillow``).
* ``--load-hub`` -- load the benchmark ``records`` record set exactly as a consumer would: no local
  mapping, so mlcroissant clones the Hub repository (Git LFS pointers only) and fetches just the nine
  ``records`` Parquet files (about 180 KB) into a temporary cache. Needs ``mlcroissant``,
  ``gitpython`` and ``git lfs``; downloads no imagery.
* ``--release`` -- the pre-upload gate: also fail on the DOI placeholder, on any "forthcoming" or
  "NOT-YET" text, and on an empty ``version``. The committed files fail it until the DOI is minted.

Usage::

    python scripts/validate_croissant.py                     # offline checks
    python scripts/validate_croissant.py --mlcroissant       # + reference validator
    python scripts/validate_croissant.py --hub               # + Hub drift and file hashes
    python scripts/validate_croissant.py --rebuild-records --load
    python scripts/validate_croissant.py --release --hub --mlcroissant   # before any upload

Exits non-zero on any failure. See ``docs/fair_metadata_150.md``.
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CROISSANT_DIR = REPO / "croissant"
FILES = {
    "rampnet-dataset": CROISSANT_DIR / "rampnet-dataset.json",
    "rampnet-benchmark": CROISSANT_DIR / "rampnet-benchmark.json",
}
HUB_API = "https://huggingface.co/api/datasets/projectsidewalk/{}"

CONFORMS_TO = (
    "http://mlcommons.org/croissant/1.1",
    "http://mlcommons.org/croissant/RAI/1.0",
    "http://mlcommons.org/croissant/geo/1.0",
)
# Croissant 1.1 "Required" dataset properties, plus the ones this repo commits to.
REQUIRED = ("@context", "@type", "conformsTo", "name", "description", "license", "url", "creator",
            "datePublished", "distribution", "recordSet", "citeAs", "citation", "version",
            "isLiveDataset", "identifier", "spatialCoverage", "geocr:coordinateReferenceSystem",
            "geocr:samplingStrategy", "geocr:spatialBias")
REQUIRED_RAI = ("rai:dataCollection", "rai:dataCollectionType", "rai:dataCollectionTimeframe",
                "rai:dataCollectionRawData", "rai:dataCollectionMissingData", "rai:dataBiases",
                "rai:dataLimitations", "rai:dataUseCases", "rai:personalSensitiveInformation",
                "rai:annotationsPerItem", "rai:dataReleaseMaintenancePlan")

# The JSON-LD context: Appendix 1 of the Croissant 1.1 spec, term for term, with two deliberate
# differences. (1) schema.org is https: Appendix 1 writes http://schema.org/, but mlcroissant 1.1.0
# rejects that form ("The current JSON-LD doesn't extend https://schema.org/Dataset"), and the spec's
# own later examples use https. (2) One extra term, the `geocr` prefix GeoCroissant 1.0 requires.
# mlcroissant's built-in context also carries the Croissant 1.0 terms path, repeated, replace and
# samplingRate; these files use none of them, so they are left out and mlcroissant logs a
# "context is not standard" warning naming those four. The warning is expected.
SCHEMA_ORG = "https://schema.org/"
CONTEXT = {
    "@language": "en",
    "@vocab": SCHEMA_ORG,
    "sc": SCHEMA_ORG,
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
    "geocr": "http://mlcommons.org/croissant/geo/",
}

# The DOI is not minted yet (issue #150). This exact string is what to search for and replace;
# the check below fails if it is changed to anything that is neither this nor a DOI URL, and
# --release fails while it is still there.
DOI_PLACEHOLDER = "DOI-NOT-YET-MINTED (issue #150): replace with the DataCite DOI once minted"
DOI_RE = re.compile(r"^https://doi\.org/10\.\d{4,9}/\S+$")
UNRELEASED_RE = re.compile(r"forthcoming|NOT-YET", re.IGNORECASE)

# mlcroissant 1.1.0's extract_git_info() understands a Hub URL only as the bare repository or as
# `.../tree/refs%2F<ref>`; a `.../tree/<sha>` URL is passed to `git clone` verbatim and fails. So
# contentUrl is one of those two forms, and the revision lives in the descriptions (REVISION_RE).
CONTENT_URL_RE = re.compile(
    r"^https://huggingface\.co/datasets/projectsidewalk/([\w.-]+)(?:/tree/refs%2F([\w.%-]+))?$")
REVISION_RE = re.compile(r"revision ([0-9a-f]{40})")
SOURCE_NAMES = {"launch": "Google Street View", "mapillary": "Mapillary"}

# Links into this repository must name a commit or a release tag, never a branch.
GITHUB_LINK_RE = re.compile(r"https://github\.com/ProjectSidewalk/RampNet/(?:blob|tree)/([^/\s#]+)/?([^\s#),;]*)")
PINNED_REF_RE = re.compile(r"^(?:[0-9a-f]{40}|v\d[\w.-]*)$")
# A path into this repo written without a URL. Preceded by '/' means it is part of a URL.
BARE_PATH_RE = re.compile(
    r"(?<![\w/.-])((?:docs|benchmark|stage_one|stage_two|scripts|rampnet|analysis_out|manual_labels|"
    r"croissant)/[\w./<>-]*)")
# The validator is added by the same PR as these files, so no earlier commit can pin it.
UNPINNED_OK = ("scripts/validate_croissant.py",)

# Committed government inventories (docs/data_provenance.md section 3): file, parser, date
# field, record count, sha256, and the spatialCoverage place name the box belongs to.
LOCATION_DATA = REPO / "stage_one" / "dataset_generation" / "location_data"
LOCATION_FILES = (
    ("New York City, NY, USA", "nyc.csv", "parse_csv", "GeoCyclora", 217679,
     "beea2b323d00d82192dd18ace3f257cef30ce3b579544d4e607fe7abe5e57f8c"),
    ("Portland, OR, USA", "portland.geojson", "parse_geojson", "InstallDate", 45035,
     "d5366a7e0d18f09f9ba49f1cbf7a26b99ee90633689dbe94cbde2a21bd395dbe"),
    ("Bend, OR, USA", "bend.geojson", "parse_geojson", "InstallDate", 13357,
     "a0da4e016474c2c8fddcc6f77a7dd4a3aa5caaea455c839fad762d66a7af948e"),
)


def load(path):
    """Parse one Croissant file as JSON (JSON-LD is JSON)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk_ids(node, out):
    if isinstance(node, dict):
        if "@id" in node and ("@type" in node):
            out.append(node["@id"])
        for key, value in node.items():
            if key != "data":                     # inline rows hold no node ids
                _walk_ids(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_ids(value, out)


def _strings(node):
    """Every string value in a document, skipping ``@context`` and inline data rows."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in ("@context", "data"):
                yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def _fields(record_set):
    for field in record_set.get("field", []):
        yield field
        yield from field.get("subField", [])


def _keys(record_set):
    key = record_set.get("key")
    key = [key] if isinstance(key, dict) else (key or [])
    return [k.get("@id") for k in key]


def check_structure(doc):
    """Return a list of problems with one parsed Croissant document (empty when fine)."""
    problems = []
    for key in REQUIRED + REQUIRED_RAI:
        if key not in doc or doc[key] in (None, "", []):
            problems.append("missing required key {!r}".format(key))
    ctx = doc.get("@context", {})
    if ctx != CONTEXT:
        extra = sorted(set(ctx) - set(CONTEXT))
        missing = sorted(set(CONTEXT) - set(ctx))
        changed = sorted(k for k in set(ctx) & set(CONTEXT) if ctx[k] != CONTEXT[k])
        problems.append("@context differs from Appendix 1 + geocr: extra {}, missing {}, changed {}"
                        .format(extra, missing, changed))
    conforms = doc.get("conformsTo", [])
    conforms = [conforms] if isinstance(conforms, str) else conforms
    for uri in CONFORMS_TO:
        if uri not in conforms:
            problems.append("conformsTo lacks {}".format(uri))
    if doc.get("@type") != "sc:Dataset":
        problems.append("@type must be sc:Dataset")
    if doc.get("isLiveDataset") is not False:
        problems.append("isLiveDataset must be false: both datasets are static releases")
    ident = doc.get("identifier")
    if ident != DOI_PLACEHOLDER and not (isinstance(ident, str) and DOI_RE.match(ident)):
        problems.append("identifier is neither the DOI placeholder nor a https://doi.org/ URL")
    for date in doc.get("rai:dataCollectionTimeframe", []):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date)):
            problems.append("rai:dataCollectionTimeframe value {!r} is not YYYY-MM-DD".format(date))

    ids = []
    _walk_ids(doc, ids)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append("duplicate @id values: {}".format(dupes))

    dist_ids = {d.get("@id") for d in doc.get("distribution", [])}
    for dist in doc.get("distribution", []):
        parent = (dist.get("containedIn") or {}).get("@id")
        if parent and parent not in dist_ids:
            problems.append("{} is containedIn unknown {}".format(dist.get("@id"), parent))
        if dist.get("encodingFormat") not in ("git+https", "application/x-parquet"):
            problems.append("{} has unexpected encodingFormat {!r}".format(
                dist.get("@id"), dist.get("encodingFormat")))
    repo = next((d for d in doc.get("distribution", []) if d.get("@id") == "repo"), {})
    if not CONTENT_URL_RE.match(repo.get("contentUrl", "")):
        problems.append("repo contentUrl {!r} is neither the bare Hub repository URL nor a "
                        ".../tree/refs%2F<ref> URL (mlcroissant cannot clone a bare sha)".format(
                            repo.get("contentUrl")))
    rev = pinned_revision(doc)
    if rev is None:
        problems.append("repo description names no single 40-hex revision")
    else:
        manifest = record_set(doc, "file_manifest") or {}
        if REVISION_RE.findall(manifest.get("description", "")) != [rev]:
            problems.append("file_manifest description does not name the pinned revision {}".format(rev))

    field_ids = set()
    for rs in doc.get("recordSet", []):
        field_ids.update(f["@id"] for f in _fields(rs))
    for rs in doc.get("recordSet", []):
        for key in _keys(rs):
            if key not in field_ids:
                problems.append("{} is keyed on unknown field {}".format(rs["@id"], key))
        for field in _fields(rs):
            src = field.get("source", {})
            ref = (src.get("fileSet") or src.get("fileObject") or {}).get("@id")
            if ref is not None and ref not in dist_ids:
                problems.append("{} reads from unknown distribution {}".format(field["@id"], ref))
            target = ((field.get("references") or {}).get("field") or {}).get("@id")
            if target is not None and target not in field_ids:
                problems.append("{} references unknown field {}".format(field["@id"], target))
        if "data" in rs:
            names = {f["@id"] for f in rs.get("field", [])}
            for row in rs["data"]:
                if set(row) != names:
                    problems.append("{}: inline row keys {} != fields {}".format(
                        rs["@id"], sorted(row), sorted(names)))
                    break
    return problems


def _pinned_path_exists(ref, path):
    """True/False when git can answer for ``ref:path``; None when the ref is not available locally
    (a shallow CI checkout, or a tag that was not fetched)."""
    try:
        known = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", ref + "^{commit}"],
                               capture_output=True).returncode == 0
        if not known:
            return None
        spec = "{}:{}".format(ref, path.rstrip("/")) if path else ref
        return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", spec],
                              capture_output=True).returncode == 0
    except OSError:
        return None


def check_links(doc):
    """Links into this repo are pinned (and resolve, where git can tell); no bare repo paths."""
    problems = []
    seen = set()
    for text in _strings(doc):
        for match in GITHUB_LINK_RE.finditer(text):
            ref, path = match.group(1), match.group(2).rstrip(".")
            if not PINNED_REF_RE.match(ref):
                problems.append("unpinned GitHub link (ref {!r}): {}".format(ref, match.group(0)))
            elif (ref, path) not in seen:
                seen.add((ref, path))
                if _pinned_path_exists(ref, path) is False:
                    problems.append("{} does not exist at {}".format(path, ref))
        for match in BARE_PATH_RE.finditer(text):
            if not match.group(1).rstrip(".").startswith(UNPINNED_OK):
                problems.append("bare repo-relative path {!r} (use a pinned GitHub URL) in: {}...".format(
                    match.group(1), text[:60]))
    return problems


def record_set(doc, rs_id):
    """The record set with this @id, or None."""
    return next((r for r in doc.get("recordSet", []) if r.get("@id") == rs_id), None)


def _reviewed_panos(bundle):
    judged = json.loads((bundle / "verdicts.json").read_text(encoding="utf-8")).get("panos", {})
    panos = []
    for line in (bundle / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            pano = json.loads(line).get("pano", {})
            if pano.get("panorama_id") in judged:
                panos.append(pano)
    return panos


def derive_split_extents(splits, benchmark=REPO / "benchmark"):
    """Recompute ``split_extents`` rows from committed benchmark bundles.

    Only panoramas that appear in ``verdicts.json`` count, which is the filter
    ``scripts/export_benchmark.py`` uses to build the published ``records`` config.
    """
    rows = []
    for split in splits:
        panos = _reviewed_panos(Path(benchmark) / split)
        sources = sorted({p.get("source") for p in panos})
        dates = sorted(p["capture_date"] for p in panos)
        rows.append({
            "split": split,
            "imagery_source": " + ".join(SOURCE_NAMES.get(s, s) for s in sources),
            "num_panoramas": len(panos),
            "min_lat": round(min(p["lat"] for p in panos), 5),
            "min_lng": round(min(p["lng"] for p in panos), 5),
            "max_lat": round(max(p["lat"] for p in panos), 5),
            "max_lng": round(max(p["lng"] for p in panos), 5),
            "capture_date_min": dates[0],
            "capture_date_max": dates[-1],
        })
    return rows


def check_benchmark_extents(doc, benchmark=REPO / "benchmark"):
    """Compare the committed ``split_extents`` rows with a fresh derivation, and check the key."""
    problems = []
    rs = record_set(doc, "split_extents")
    splits_rs = record_set(doc, "splits")
    if rs is None or splits_rs is None:
        return ["benchmark file lacks the split_extents or splits record set"]
    declared = [r["splits/name"] for r in splits_rs["data"]]
    stored = {r["split_extents/split"]: r for r in rs["data"]}
    if sorted(stored) != sorted(declared):
        problems.append("split_extents covers {} but splits declares {}".format(
            sorted(stored), sorted(declared)))
    for fresh in derive_split_extents(declared, benchmark):
        row = stored.get(fresh["split"], {})
        for key, value in fresh.items():
            if row.get("split_extents/" + key) != value:
                problems.append("split_extents[{}].{}: file has {!r}, committed data gives {!r}".format(
                    fresh["split"], key, row.get("split_extents/" + key), value))
    boxes = {p["description"].split("`")[1]: p["geo"]["box"] for p in doc["spatialCoverage"]}
    for split, row in stored.items():
        want = "{} {} {} {}".format(row["split_extents/min_lat"], row["split_extents/min_lng"],
                                    row["split_extents/max_lat"], row["split_extents/max_lng"])
        if boxes.get(split) != want:
            problems.append("spatialCoverage box for {} is {!r}, split_extents gives {!r}".format(
                split, boxes.get(split), want))
    lo = min(r["split_extents/capture_date_min"] for r in stored.values())
    hi = max(r["split_extents/capture_date_max"] for r in stored.values())
    if doc.get("temporalCoverage") != "{}/{}".format(lo, hi):
        problems.append("temporalCoverage {!r} != {}/{}".format(doc.get("temporalCoverage"), lo, hi))

    # One key rule for every per-panorama record set: (split, pano_id). Check it holds in git.
    for rid, want_key in (("records", ["records/split", "records/pano_id"]),
                          ("native", ["native/split", "native/pano_id"]),
                          ("px4096x2048", ["px4096x2048/split", "px4096x2048/pano_id"]),
                          ("galleries", ["galleries/split", "galleries/crop_id"])):
        got = _keys(record_set(doc, rid) or {})
        if got != want_key:
            problems.append("{} is keyed on {}, expected {}".format(rid, got, want_key))
    for split in declared:
        ids = [p.get("panorama_id") for p in _reviewed_panos(Path(benchmark) / split)]
        if len(ids) != len(set(ids)):
            problems.append("{}: pano_id is not unique among reviewed panoramas".format(split))
    return problems


@functools.lru_cache(maxsize=None)
def derive_dataset_boxes(location_data=LOCATION_DATA):
    """Parse the committed government inventories with ``combine_location_data.py``'s own parsers.

    Returns ``{place name: (record count, box, sha256)}`` with the box formatted as a GeoShape box
    string, rounded to 5 decimals.
    """
    cld = _import("combine_location_data", REPO / "stage_one" / "dataset_generation" /
                  "combine_location_data.py")
    out = {}
    for place, filename, parser, date_field, _, _ in LOCATION_FILES:
        path = Path(location_data) / filename
        with warnings.catch_warnings():         # utcfromtimestamp, once per dated geojson record
            warnings.simplefilter("ignore", DeprecationWarning)
            rows = getattr(cld, parser)(str(path), date_field)
        lats = [r["latitude"] for r in rows]
        lngs = [r["longitude"] for r in rows]
        box = "{} {} {} {}".format(round(min(lats), 5), round(min(lngs), 5),
                                   round(max(lats), 5), round(max(lngs), 5))
        out[place] = (len(rows), box, hashlib.sha256(path.read_bytes()).hexdigest())
    return out


def check_dataset_boxes(doc, location_data=LOCATION_DATA):
    """The dataset's city boxes, record counts and NYC share, re-derived from committed inputs."""
    problems = []
    derived = derive_dataset_boxes(Path(location_data))
    stored = {p["name"]: p["geo"]["box"] for p in doc["spatialCoverage"]}
    if sorted(stored) != sorted(derived):
        problems.append("spatialCoverage places {} != {}".format(sorted(stored), sorted(derived)))
    for place, filename, _, _, count, sha in LOCATION_FILES:
        n, box, digest = derived[place]
        if digest != sha:
            problems.append("{}: sha256 {} != docs/data_provenance.md {}".format(filename, digest, sha))
        if n != count:
            problems.append("{}: {} records parsed, docs/data_provenance.md says {}".format(
                filename, n, count))
        if stored.get(place) != box:
            problems.append("spatialCoverage box for {} is {!r}, {} gives {!r}".format(
                place, stored.get(place), filename, box))
    total = sum(n for n, _, _ in derived.values())
    share = "{:.1f}%".format(100.0 * derived["New York City, NY, USA"][0] / total)
    for key in ("geocr:spatialBias", "rai:dataBiases"):
        if share not in doc.get(key, "") or "{:,}".format(total) not in doc.get(key, ""):
            problems.append("{} should quote NYC's {} of the {:,} committed records".format(
                key, share, total))
    return problems


def check_release(doc):
    """The pre-upload gate: no placeholder identifier, no 'forthcoming' text, a version set."""
    problems = []
    ident = doc.get("identifier")
    if not (isinstance(ident, str) and DOI_RE.match(ident)):
        problems.append("release: identifier must be the minted https://doi.org/ DOI, not {!r}".format(
            ident))
    for text in _strings(doc):
        match = UNRELEASED_RE.search(text)
        if match:
            start = max(0, match.start() - 40)
            problems.append("release: unreleased marker {!r} in ...{}...".format(
                match.group(0), text[start:match.end() + 40].replace("\n", " ")))
    if not str(doc.get("version", "")).strip():
        problems.append("release: version is not set")
    return problems


def pinned_revision(doc):
    """The 40-hex Hub revision the ``repo`` FileObject's description names, or None."""
    repo = next((d for d in doc.get("distribution", []) if d.get("@id") == "repo"), {})
    found = set(REVISION_RE.findall(repo.get("description", "")))
    return found.pop() if len(found) == 1 else None


def content_ref(doc):
    """The git ref ``contentUrl`` resolves to: ``refs/heads/main`` for the bare repository URL, or
    the named ``refs/...`` ref (e.g. ``refs/tags/v1.0.0``)."""
    repo = next((d for d in doc.get("distribution", []) if d.get("@id") == "repo"), {})
    match = CONTENT_URL_RE.match(repo.get("contentUrl", ""))
    if not match:
        return None
    return "refs/" + urllib.parse.unquote(match.group(2)) if match.group(2) else "refs/heads/main"


def resolve_ref(name, ref):
    """The commit a Hub ref points at, from ``/api/datasets/<id>/refs``; None if it does not exist."""
    refs = _get_json((HUB_API + "/refs").format(name))[0]
    for entry in refs.get("branches", []) + refs.get("tags", []) + refs.get("converts", []):
        if entry.get("ref") == ref:
            return entry.get("targetCommit")
    return None


def _get_json(url):
    """GET a JSON document; returns (parsed body, URL of the next page or None)."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        body = json.load(resp)
        link = resp.headers.get("Link", "")
    nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return body, (nxt.group(1) if nxt else None)


def hub_tree(name, rev):
    """Every entry of the Hub tree at ``rev``, following the listing's pagination."""
    url = (HUB_API + "/tree/{}?recursive=true").format(name, rev)
    entries = []
    while url:
        page, url = _get_json(url)
        entries.extend(page)
    return entries


def check_hub(name, doc):
    """The ref ``contentUrl`` names is the pinned revision, and ``file_manifest`` matches the tree
    at the pin."""
    rev = pinned_revision(doc)
    if rev is None:
        return ["repo description names no revision"]
    problems = []
    ref = content_ref(doc)
    got = resolve_ref(name, ref) if ref else None
    if got != rev:
        problems.append("{} of projectsidewalk/{} is {} but this file describes {}: the Hub has moved. "
                        "Re-pin the repo and file_manifest descriptions, the file_manifest rows and "
                        "dateModified together.".format(ref, name, got, rev))
    tree = hub_tree(name, rev)
    hub = {e["path"]: (e["size"], (e.get("lfs") or {}).get("oid"))
           for e in tree if e.get("type") == "file" and e["path"].endswith(".parquet")}
    ours = {r["file_manifest/path"]: (r["file_manifest/bytes"], r["file_manifest/sha256"])
            for r in record_set(doc, "file_manifest")["data"]}
    for path in sorted(set(hub) | set(ours)):
        if hub.get(path) != ours.get(path):
            problems.append("{}: Hub {} vs file_manifest {}".format(path, hub.get(path), ours.get(path)))
    return problems


def rebuild_records(doc, out, benchmark=REPO / "benchmark"):
    """Build the ``records`` config into ``out`` with the exporter's own ``build_records``.

    The exporter also builds splits that are in git but not on the Hub (the two Laurens arms);
    those files are removed so the rebuild holds exactly the splits ``doc`` declares.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        exporter = _import("export_benchmark", REPO / "scripts" / "export_benchmark.py")
    finally:
        sys.path.pop(0)
    exporter.build_records(Path(benchmark), Path(out))
    published = {r["splits/name"] for r in record_set(doc, "splits")["data"]}
    for path in (Path(out) / "data" / "records").glob("*.parquet"):
        if path.stem not in published:
            path.unlink()


def check_rebuild_records(doc, out):
    """sha256 of each rebuilt ``data/records/<split>.parquet`` against ``file_manifest``."""
    import pyarrow                                 # optional dependency
    rebuild_records(doc, out)
    ours = {r["file_manifest/path"]: r["file_manifest/sha256"]
            for r in record_set(doc, "file_manifest")["data"]
            if r["file_manifest/path"].startswith("data/records/")}
    problems = []
    for path, want in sorted(ours.items()):
        built = Path(out) / path
        got = hashlib.sha256(built.read_bytes()).hexdigest() if built.is_file() else None
        if got != want:
            problems.append("rebuild (pyarrow {}): {} sha256 {} != file_manifest {}".format(
                pyarrow.__version__, path, got, want))
    return problems


def _posix_fullpaths(mlc):
    """mlcroissant 1.1.0 builds FileSet-relative paths with os.sep, so on Windows an ``includes``
    glob like ``data/records/*.parquet`` matches nothing ("No objects to concatenate"). Validation
    is unaffected. Loading on Windows needs POSIX relative paths; on Linux/macOS this is a no-op."""
    if os.sep == "/":
        return
    import pathlib
    from mlcroissant._src.operation_graph.operations import filter as mlc_filter

    def get_fullpath(file, data_dir):
        rel = os.fspath(file).replace(os.fspath(data_dir), "").replace("\\", "/").lstrip("/")
        return pathlib.PurePosixPath(rel)
    mlc_filter.get_fullpath = get_fullpath

    # download_git_lfs_file() finds the clone's working dir by splitting the OS path on the POSIX
    # relative path, which never matches on Windows; compute it from the path parts instead.
    from mlcroissant._src.core.optional import deps
    from mlcroissant._src.operation_graph.operations import read as mlc_read

    def download_git_lfs_file(file):
        full = pathlib.PurePath(os.fspath(file.filepath))
        rel = pathlib.PurePosixPath(os.fspath(file.fullpath))
        working_dir = full.parents[len(rel.parts) - 1]
        deps.git.Git(str(working_dir)).execute(["git", "lfs", "pull", "--include", str(rel)])
    mlc_read.download_git_lfs_file = download_git_lfs_file


def check_load_records(doc_path, out, benchmark=REPO / "benchmark"):
    """Load the benchmark ``records`` record set through the Croissant file from a local rebuild."""
    import mlcroissant as mlc                      # optional dependency
    _posix_fullpaths(mlc)
    ds = mlc.Dataset(jsonld=str(doc_path), mapping={"repo": str(out)})
    per_split, dets, missed = {}, 0, 0
    for row in ds.records("records"):
        split = row["records/split"]
        split = split.decode() if isinstance(split, bytes) else split
        per_split[split] = per_split.get(split, 0) + 1
        dets += len(row["records/detections"] or [])
        missed += len(row["records/missed"] or [])
    want_split, want_dets, want_missed = {}, 0, 0
    published = [r["splits/name"] for r in record_set(load(doc_path), "splits")["data"]]
    for bundle in sorted(Path(benchmark).glob("*/verdicts.json")):
        split = bundle.parent.name
        if split not in published:
            continue
        judged = json.loads(bundle.read_text(encoding="utf-8"))["panos"]
        want_split[split] = len(_reviewed_panos(bundle.parent))
        want_missed += sum(len(v.get("missed", [])) for v in judged.values())
        for line in (bundle.parent / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("pano", {}).get("panorama_id") in judged:
                    want_dets += len(rec.get("detections", []))
    problems = []
    if per_split != want_split:
        problems.append("load: records rows per split {} != committed {}".format(per_split, want_split))
    if (dets, missed) != (want_dets, want_missed):
        problems.append("load: {} detections / {} missed, committed bundles give {} / {}".format(
            dets, missed, want_dets, want_missed))
    return problems, sum(per_split.values()), dets, missed


def check_load_hub(doc_path, cache):
    """Load ``records`` through the Croissant file with no mapping: mlcroissant clones the Hub repo
    named by ``contentUrl`` itself. Only the nine small ``records`` files are fetched from LFS."""
    import mlcroissant as mlc                      # optional dependency
    from mlcroissant._src.core import constants
    _posix_fullpaths(mlc)
    constants.DOWNLOAD_PATH = Path(cache) / "download"     # a fresh clone, not a stale cache
    doc = load(doc_path)
    ds = mlc.Dataset(jsonld=str(doc_path))
    per_split, dets, missed = {}, 0, 0
    for row in ds.records("records"):
        split = row["records/split"]
        split = split.decode() if isinstance(split, bytes) else split
        per_split[split] = per_split.get(split, 0) + 1
        dets += len(row["records/detections"] or [])
        missed += len(row["records/missed"] or [])
    clones = [p for p in (Path(cache) / "download").glob("croissant-*") if (p / ".git").exists()]
    head = subprocess.run(["git", "-C", str(clones[0]), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip() if clones else None
    want = {r["split_extents/split"]: r["split_extents/num_panoramas"]
            for r in record_set(doc, "split_extents")["data"]}
    problems = []
    if head != pinned_revision(doc):
        problems.append("load-hub: cloned HEAD {} is not the pinned revision".format(head))
    if per_split != want:
        problems.append("load-hub: records rows per split {} != split_extents {}".format(per_split, want))
    return problems, sum(per_split.values()), dets, missed, head


def write_synthetic_shard(out):
    """One row with the Hub's ``rampnet-dataset`` schema (read from datasets-server /info)."""
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), (90, 120, 150)).save(buf, format="JPEG")
    schema = pa.schema([
        pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("pano_id", pa.string()),
        pa.field("record_creation_time", pa.int64()),
        pa.field("curb_ramp_points_normalized", pa.list_(pa.list_(pa.float32()))),
        pa.field("pano_coord", pa.list_(pa.float64())),
        pa.field("curb_ramp_coords", pa.list_(pa.list_(pa.float64()))),
        pa.field("pano_azimuth", pa.float64()),
    ])
    row = {"image": {"bytes": buf.getvalue(), "path": None}, "pano_id": "SYNTHETIC_PANO_ID",
           "record_creation_time": 1750000000,
           "curb_ramp_points_normalized": [[0.25, 0.625], [0.75, 0.5]],
           "pano_coord": [40.7, -74.0], "curb_ramp_coords": [[40.70001, -74.00002]],
           "pano_azimuth": -12.5}
    path = Path(out) / "train" / "data-00000-of-00128.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=schema), str(path))
    return row


def check_load_dataset(doc_path, out):
    """Load the dataset ``panoramas`` record set through the Croissant file from a synthetic shard."""
    import mlcroissant as mlc                      # optional dependency
    _posix_fullpaths(mlc)
    want = write_synthetic_shard(out)
    ds = mlc.Dataset(jsonld=str(doc_path), mapping={"repo": str(out)})
    rows = list(ds.records("panoramas"))
    if len(rows) != 1:
        return ["load: synthetic shard gave {} panoramas rows, expected 1".format(len(rows))]
    row = {k.split("/", 1)[1]: v for k, v in rows[0].items()}
    problems = []
    expected = {"split", "image", "pano_id", "record_creation_time", "curb_ramp_points_normalized",
                "pano_coord", "curb_ramp_coords", "pano_azimuth"}
    if set(row) != expected:
        problems.append("load: panoramas fields {} != {}".format(sorted(row), sorted(expected)))
    text = lambda v: v.decode() if isinstance(v, bytes) else v          # noqa: E731
    if text(row.get("split")) != "train" or text(row.get("pano_id")) != want["pano_id"]:
        problems.append("load: split/pano_id {!r}/{!r}".format(row.get("split"), row.get("pano_id")))
    points = [[round(float(x), 4) for x in p] for p in row.get("curb_ramp_points_normalized", [])]
    if points != want["curb_ramp_points_normalized"]:
        problems.append("load: curb_ramp_points_normalized {} != {}".format(
            points, want["curb_ramp_points_normalized"]))
    if [float(x) for x in row.get("pano_coord", [])] != want["pano_coord"]:
        problems.append("load: pano_coord {}".format(row.get("pano_coord")))
    if getattr(row.get("image"), "size", None) != (64, 32):
        problems.append("load: image did not decode to a 64x32 image ({!r})".format(row.get("image")))
    return problems


def check_mlcroissant(path):
    """Run the MLCommons reference validator; returns a list of problems."""
    import mlcroissant as mlc                     # optional dependency
    try:
        mlc.Dataset(jsonld=str(path))
    except mlc.ValidationError as err:            # raised with every issue in the message
        return [str(err)]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mlcroissant", action="store_true",
                        help="also run the mlcroissant reference validator")
    parser.add_argument("--hub", action="store_true",
                        help="also check the pin against the Hub's main, and file sizes and sha256 "
                             "(network)")
    parser.add_argument("--rebuild-records", action="store_true",
                        help="rebuild the benchmark records config and compare sha256 (pyarrow)")
    parser.add_argument("--load", action="store_true",
                        help="load records / a synthetic dataset shard through mlcroissant")
    parser.add_argument("--load-hub", action="store_true",
                        help="load benchmark records by cloning the Hub repo (~180 KB of Parquet)")
    parser.add_argument("--release", action="store_true",
                        help="pre-upload gate: reject the DOI placeholder and 'forthcoming' text")
    args = parser.parse_args(argv)

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in FILES.items():
            doc = load(path)
            problems = check_structure(doc) + check_links(doc)
            notes = []
            if name == "rampnet-benchmark":
                problems += check_benchmark_extents(doc)
                if args.rebuild_records or args.load:
                    records_out = Path(tmp) / "benchmark"
                    if args.rebuild_records:
                        problems += check_rebuild_records(doc, records_out)
                    else:
                        rebuild_records(doc, records_out)
                    if args.load:
                        more, rows, dets, missed = check_load_records(path, records_out)
                        problems += more
                        notes.append("loaded {:,} records rows, {:,} detections, {:,} missed".format(
                            rows, dets, missed))
                if args.load_hub:
                    more, rows, dets, missed, head = check_load_hub(path, Path(tmp) / "cache")
                    problems += more
                    notes.append("loaded from the Hub (clone at {}): {:,} records rows, {:,} "
                                 "detections, {:,} missed".format((head or "?")[:7], rows, dets, missed))
            else:
                problems += check_dataset_boxes(doc)
                if args.load:
                    problems += check_load_dataset(path, Path(tmp) / "dataset")
                    notes.append("loaded a synthetic one-row shard")
            if args.mlcroissant:
                problems += check_mlcroissant(path)
            if args.hub:
                problems += check_hub(name, doc)
            if args.release:
                problems += check_release(doc)
            rows = sum(len(r.get("data", [])) for r in doc["recordSet"] if r["@id"] == "file_manifest")
            status = "ok" if not problems else "FAIL"
            print("{:<20} {:<4} {} record sets, {} files in manifest, revision {}, identifier: {}".format(
                name, status, len(doc["recordSet"]), rows, (pinned_revision(doc) or "none")[:7],
                "placeholder (DOI not minted)" if doc.get("identifier") == DOI_PLACEHOLDER
                else doc.get("identifier")))
            for note in notes:
                print("    " + note)
            for problem in problems:
                print("    - " + problem)
            failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
