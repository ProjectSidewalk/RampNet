"""Check the hand-authored Croissant 1.1 + GeoCroissant 1.0 + RAI files in ``croissant/``.

Three levels, cheapest first. The default is offline and has no dependency beyond the standard
library, so ``tests/test_croissant.py`` runs it on every PR:

* **structure** -- each file is JSON, declares the three specs in ``conformsTo``, carries the
  required Croissant properties and the RAI / GeoCroissant keys this repo commits to, has unique
  ``@id`` values, and every field source points at a distribution that exists.
* **derived numbers** -- the benchmark's ``split_extents`` record set (bounding box, capture-month
  range, panorama count and imagery source per split) is re-derived from the committed
  ``benchmark/<city>/records.jsonl`` + ``verdicts.json`` with the same "reviewed panoramas only"
  filter that ``scripts/export_benchmark.py`` applies, and must match exactly.
* ``--mlcroissant`` -- also run the MLCommons reference validator (``pip install mlcroissant``).
* ``--hub`` -- also fetch the Hugging Face tree listing at the revision each file pins and check
  that every Parquet file's size and sha256 equal the ``file_manifest`` record set (network).

Usage::

    python scripts/validate_croissant.py                 # offline checks
    python scripts/validate_croissant.py --mlcroissant   # + reference validator
    python scripts/validate_croissant.py --hub           # + Hub file hashes

Exits non-zero on any failure. See ``docs/fair_metadata_150.md``.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CROISSANT_DIR = REPO / "croissant"
FILES = {
    "rampnet-dataset": CROISSANT_DIR / "rampnet-dataset.json",
    "rampnet-benchmark": CROISSANT_DIR / "rampnet-benchmark.json",
}

CONFORMS_TO = (
    "http://mlcommons.org/croissant/1.1",
    "http://mlcommons.org/croissant/RAI/1.0",
    "http://mlcommons.org/croissant/geo/1.0",
)
# Croissant 1.1 "Required" dataset properties, plus the ones this repo commits to.
REQUIRED = ("@context", "@type", "conformsTo", "name", "description", "license", "url", "creator",
            "datePublished", "distribution", "recordSet", "citeAs", "version", "isLiveDataset",
            "identifier", "spatialCoverage", "geocr:coordinateReferenceSystem",
            "geocr:samplingStrategy", "geocr:spatialBias")
REQUIRED_RAI = ("rai:dataCollection", "rai:dataCollectionType", "rai:dataBiases",
                "rai:dataLimitations", "rai:dataUseCases", "rai:personalSensitiveInformation",
                "rai:annotationsPerItem", "rai:dataReleaseMaintenancePlan")

# The DOI is not minted yet (issue #150). This exact string is what to search for and replace;
# the check below fails if it is changed to anything that is neither this nor a DOI URL.
DOI_PLACEHOLDER = "DOI-NOT-YET-MINTED (issue #150): replace with the DataCite DOI once minted"
DOI_RE = re.compile(r"^https://doi\.org/10\.\d{4,9}/\S+$")

REVISION_RE = re.compile(r"revision ([0-9a-f]{40})")
SOURCE_NAMES = {"launch": "Google Street View", "mapillary": "Mapillary"}


def load(path):
    """Parse one Croissant file as JSON (JSON-LD is JSON)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _fields(record_set):
    for field in record_set.get("field", []):
        yield field
        yield from field.get("subField", [])


def check_structure(doc):
    """Return a list of problems with one parsed Croissant document (empty when fine)."""
    problems = []
    for key in REQUIRED + REQUIRED_RAI:
        if key not in doc or doc[key] in (None, "", []):
            problems.append("missing required key {!r}".format(key))
    ctx = doc.get("@context", {})
    for prefix in ("cr", "rai", "geocr", "sc", "dct"):
        if prefix not in ctx:
            problems.append("@context lacks the {!r} prefix".format(prefix))
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
    field_ids = set()
    for rs in doc.get("recordSet", []):
        field_ids.update(f["@id"] for f in _fields(rs))
    for rs in doc.get("recordSet", []):
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


def record_set(doc, rs_id):
    """The record set with this @id, or None."""
    return next((r for r in doc.get("recordSet", []) if r.get("@id") == rs_id), None)


def derive_split_extents(splits, benchmark=REPO / "benchmark"):
    """Recompute ``split_extents`` rows from committed benchmark bundles.

    Only panoramas that appear in ``verdicts.json`` count, which is the filter
    ``scripts/export_benchmark.py`` uses to build the published ``records`` config.
    """
    rows = []
    for split in splits:
        bundle = Path(benchmark) / split
        judged = json.loads((bundle / "verdicts.json").read_text(encoding="utf-8")).get("panos", {})
        panos = []
        for line in (bundle / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                pano = json.loads(line).get("pano", {})
                if pano.get("panorama_id") in judged:
                    panos.append(pano)
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
    """Compare the committed ``split_extents`` rows with a fresh derivation."""
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
    return problems


def pinned_revision(doc):
    """The 40-hex Hub revision the ``repo`` FileObject's description pins."""
    repo = next(d for d in doc["distribution"] if d.get("@id") == "repo")
    match = REVISION_RE.search(repo.get("description", ""))
    return match.group(1) if match else None


def check_hub(name, doc):
    """Compare ``file_manifest`` with the Hub's tree listing at the pinned revision (network)."""
    rev = pinned_revision(doc)
    if rev is None:
        return ["repo FileObject pins no revision"]
    url = "https://huggingface.co/api/datasets/projectsidewalk/{}/tree/{}?recursive=true".format(
        name, rev)
    with urllib.request.urlopen(url, timeout=60) as resp:
        tree = json.load(resp)
    hub = {e["path"]: (e["size"], (e.get("lfs") or {}).get("oid"))
           for e in tree if e.get("type") == "file" and e["path"].endswith(".parquet")}
    ours = {r["file_manifest/path"]: (r["file_manifest/bytes"], r["file_manifest/sha256"])
            for r in record_set(doc, "file_manifest")["data"]}
    problems = []
    for path in sorted(set(hub) | set(ours)):
        if hub.get(path) != ours.get(path):
            problems.append("{}: Hub {} vs file_manifest {}".format(path, hub.get(path), ours.get(path)))
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
                        help="also check file sizes and sha256 against the Hub (network)")
    args = parser.parse_args(argv)

    failed = False
    for name, path in FILES.items():
        doc = load(path)
        problems = check_structure(doc)
        if name == "rampnet-benchmark":
            problems += check_benchmark_extents(doc)
        if args.mlcroissant:
            problems += check_mlcroissant(path)
        if args.hub:
            problems += check_hub(name, doc)
        rows = sum(len(r.get("data", [])) for r in doc["recordSet"] if r["@id"] == "file_manifest")
        status = "ok" if not problems else "FAIL"
        print("{:<20} {:<4} {} record sets, {} files in manifest, identifier: {}".format(
            name, status, len(doc["recordSet"]), rows,
            "placeholder (DOI not minted)" if doc.get("identifier") == DOI_PLACEHOLDER
            else doc.get("identifier")))
        for problem in problems:
            print("    - " + problem)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
