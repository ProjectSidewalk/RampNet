"""What a directory under ``benchmark/`` is, for scripts that discover splits by listing.

Three kinds of directory carry a ``records.jsonl``:

* a **scored split**: it has a review (``verdicts.json``) or independent labels
  (``gt_source.json``);
* a **borrowing bundle** (#48's neighbourhood bundles): its ``bundle.json`` borrows
  another split's verdicts, so counting it would score those judged panos twice;
* a **staged bundle** (bayonne, #159): imagery and detections exported ahead of the
  ground-truth review. Nothing about it can be scored yet.

Only the first is a split. A script that lists ``benchmark/*/`` and keeps the first
kind only stays reproducible when a city is staged before its review; one that keeps
everything with a ``records.jsonl`` exits on the staged bundle (PR #234 review, B1).

    >>> from rampnet.bundles import is_scored_split
    >>> is_scored_split("benchmark/richmond"), is_scored_split("benchmark/bayonne")
    (True, False)

It also holds the bundle loaders the scorers share (#150; moved here from
``scripts/model_comparison/compare.py``, which re-exports them unchanged):
:func:`load_bundle`, :func:`ground_truths_from_verdicts`,
:func:`load_manual_ground_truths`, :func:`validate_bundle`,
:func:`validate_manual_bundle`, plus :func:`ground_truths` (either kind of bundle, the
way every committed scorer selects its panos), :func:`rampnet_predictions` (RampNet's
own detections from ``records.jsonl``, in the published prediction-file shape) and
:func:`split_pins` (content hashes naming the exact split files a result was scored
against).
"""
import hashlib
import json
import os

from rampnet.detection_eval import build_ground_truth, load_yolo_ground_truths


def is_scored_split(bundle_dir):
    """True for a ``benchmark/<split>/`` directory that can be scored (see module doc)."""
    has = lambda name: os.path.exists(os.path.join(bundle_dir, name))  # noqa: E731
    return (has("records.jsonl") and not has("bundle.json")
            and (has("verdicts.json") or has("gt_source.json")))


def scored_splits(benchmark_dir):
    """Sorted names of every scored split under ``benchmark_dir``."""
    return sorted(n for n in os.listdir(benchmark_dir)
                  if is_scored_split(os.path.join(benchmark_dir, n)))


#: A bundle that borrows its verdicts from another bundle instead of carrying a copy
#: (#48). Its ``records.jsonl`` holds the judged panos of ``verdicts_from`` plus
#: unjudged neighbours; only the judged ones are scored, and ``--detect-unjudged``
#: runs the detector over the rest so their detections land in the cache.
BUNDLE_SPEC = "bundle.json"


def verdicts_from_spec(bundle_dir):
    """The verdict panos a ``bundle.json`` bundle points at, read from that bundle.

    Borrowing rather than copying keeps one verdicts.json per review, so the judged
    panos cannot drift from the published split they came from."""
    with open(os.path.join(bundle_dir, BUNDLE_SPEC), encoding="utf-8") as f:
        spec = json.load(f)
    src = spec.get("verdicts_from")
    if not src:
        raise SystemExit(f"{bundle_dir}/{BUNDLE_SPEC}: no 'verdicts_from'")
    vpath = os.path.normpath(os.path.join(bundle_dir, src, "verdicts.json"))
    if not os.path.exists(vpath):
        raise SystemExit(f"{bundle_dir}/{BUNDLE_SPEC}: verdicts_from {src!r} has no "
                         f"verdicts.json ({vpath})")
    with open(vpath, encoding="utf-8") as f:
        return json.load(f)["panos"]


#: Any of these makes a bundle scoreable, so ``--unreviewed`` must refuse it.
REVIEW_FILES = ("verdicts.json", BUNDLE_SPEC, "gt_source.json")


def refuse_unreviewed_if_reviewed(bundle_dir):
    """Exit if ``--unreviewed`` was given for a bundle that has ground truth of any kind:
    a review (``verdicts.json``), borrowed verdicts (``bundle.json``) or independent
    labels (``gt_source.json``, manual_gold). Decided from the files on disk, so an
    empty or manual-GT review cannot slip through as "no verdicts" (PR #234, S2)."""
    found = [n for n in REVIEW_FILES if os.path.exists(os.path.join(bundle_dir, n))]
    if found:
        raise SystemExit(f"{bundle_dir}: --unreviewed given, but the bundle has "
                         f"{', '.join(found)}; drop the flag and score it.")


def load_bundle(bundle_dir, unreviewed=False):
    """Return (records_by_pid, verdicts_panos, panos_dir) for a benchmark bundle.

    ``unreviewed=True`` accepts a bundle that has only ``records.jsonl`` and
    ``panos/`` -- a city staged ahead of its ground-truth review (#159) -- and
    returns ``{}`` for its verdicts, so every pano is detect-only. It is refused for
    a bundle that already has a review, so the flag can never hide one.

    ``verdicts_panos`` is None for a manual-GT bundle (``gt_source.json`` instead
    of ``verdicts.json`` — see ``load_manual_ground_truths``); the city bundles
    always carry a verdict review. A ``bundle.json`` bundle (#48) borrows the
    verdicts of another bundle (``verdicts_from_spec``). A directory with none of
    the three is rejected here so a mistyped path fails with one clear message
    instead of a downstream KeyError.
    """
    records = {}
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    vpath = os.path.join(bundle_dir, "verdicts.json")
    verdicts = None
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            verdicts = json.load(f)["panos"]
    elif os.path.exists(os.path.join(bundle_dir, BUNDLE_SPEC)):
        verdicts = verdicts_from_spec(bundle_dir)
    elif unreviewed and not os.path.exists(os.path.join(bundle_dir, "gt_source.json")):
        return records, {}, os.path.join(bundle_dir, "panos")
    elif not os.path.exists(os.path.join(bundle_dir, "gt_source.json")):
        raise SystemExit(f"{bundle_dir}: neither verdicts.json, {BUNDLE_SPEC} nor "
                         "gt_source.json — not a benchmark bundle")
    return records, verdicts, os.path.join(bundle_dir, "panos")


def ground_truths_from_verdicts(records, verdicts):
    """{pid: GroundTruth} derived from a bundle's human review (the city path)."""
    return {pid: build_ground_truth(records[pid]["detections"], entry["dets"],
                                    entry["missed"], entry["no_missed"])
            for pid, entry in verdicts.items()}


def load_manual_ground_truths(bundle_dir):
    """{pid: GroundTruth} for a manual-GT bundle (``benchmark/manual_gold``).

    The bundle's ``gt_source.json`` points at a directory of YOLO-format label
    files that were produced by independent manual labeling — no RampNet review
    to derive from, hence no verdicts and no RampNet anchoring. Box centers
    become GT points, there are no ignore points, and every pano is
    recall-confirmed (see ``rampnet.detection_eval.yolo_ground_truth``).
    """
    with open(os.path.join(bundle_dir, "gt_source.json"), encoding="utf-8") as f:
        src = json.load(f)
    if src.get("format") != "yolo_points":
        raise SystemExit(f"{bundle_dir}/gt_source.json: unsupported format "
                         f"{src.get('format')!r} (expected 'yolo_points')")
    labels_dir = os.path.normpath(os.path.join(bundle_dir, src["labels_dir"]))
    gts = load_yolo_ground_truths(labels_dir)
    if not gts:
        raise SystemExit(f"{labels_dir}: no .txt label files found")
    return gts


def validate_bundle(records, verdicts):
    """Fail fast on a structurally broken bundle, *before* any (paid) detector call.

    ``score_model`` builds each pano's ground truth from ``records[pid]`` + the
    verdict entry outside its per-pano failure guard (that guard is for transient
    detect() errors, not data integrity). Without this pre-flight a reviewed pano
    missing from records.jsonl, a missing verdict field, or detections/verdicts
    that don't line up would surface as a raw KeyError/ValueError partway through a
    long VLM run — after spend, and aborting models already scored. Catch it here
    with a clear message instead. Raises SystemExit listing every offending pano.

    (Legacy verdicts.json without ``no_missed`` are intentionally rejected here
    rather than silently defaulted — the current/planned bundles are new-schema;
    see docs/model_comparison.md.)"""
    problems = []
    for pid, entry in verdicts.items():
        rec = records.get(pid)
        if rec is None:
            problems.append(f"{pid}: reviewed in verdicts.json but absent from records.jsonl")
            continue
        missing = [k for k in ("dets", "missed", "no_missed") if k not in entry]
        if missing:
            problems.append(f"{pid}: verdict entry missing field(s) {missing}")
            continue
        n_det, n_ver = len(rec.get("detections", [])), len(entry["dets"])
        if n_det != n_ver:
            problems.append(f"{pid}: {n_det} detections vs {n_ver} verdicts (misaligned)")
    if problems:
        _fail_validation(problems)


def validate_manual_bundle(records, gts, need_detections=False):
    """Pre-flight for a manual-GT bundle, mirroring ``validate_bundle``'s job.

    The label files and ``records.jsonl`` are built by different tools
    (``manual_labels/`` is hand-curated; records come from ``fetch_manual_gold`` +
    ``export_gold_records``), so catch any drift between them before a paid
    detector call. ``need_detections`` is set when the rampnet baseline was
    requested: its detections live in the records, and a bundle whose exporter
    hasn't run yet must say so instead of scoring RampNet as all-misses.
    """
    problems = []
    for pid in gts:
        rec = records.get(pid)
        if rec is None:
            problems.append(f"{pid}: labeled but absent from records.jsonl "
                            "(re-run scripts/fetch_manual_gold.py)")
        elif need_detections and "detections" not in rec:
            problems.append(f"{pid}: no RampNet detections in records.jsonl "
                            "(run scripts/export_gold_records.py first)")
    for pid in records:
        if pid not in gts:
            problems.append(f"{pid}: in records.jsonl but has no label file")
    if problems:
        _fail_validation(problems)


def _fail_validation(problems):
    shown = "\n  ".join(problems[:10])
    more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
    raise SystemExit(f"Bundle validation failed ({len(problems)} pano(s)):\n  {shown}{more}")


def ground_truths(bundle_dir):
    """``(records, {pano_id: GroundTruth}, kind)`` for a scored bundle of either kind.

    This is the pano selection every committed scorer uses (``scoreboard.load_split``,
    ``rescore_benchmark_eval.load_split``), in one place:

    * a **verdict** bundle (the city splits, or a ``bundle.json`` bundle that borrows
      them) is validated, and its GT is derived from the review, in ``verdicts.json``
      order. ``kind`` is ``"verdicts"``.
    * a **manual-GT** bundle (``benchmark/manual_gold``) reads its independent labels and
      keeps the panos present in ``records.jsonl``, in records order. ``kind`` is
      ``"manual_labels"``.

    The order matters only for ties in AP's confidence sort, and it is the order the
    published numbers were computed in.
    """
    records, verdicts, _ = load_bundle(bundle_dir)
    if verdicts is not None:
        validate_bundle(records, verdicts)
        return records, ground_truths_from_verdicts(records, verdicts), "verdicts"
    gts = load_manual_ground_truths(bundle_dir)
    validate_manual_bundle(records, gts)
    return records, {pid: gts[pid] for pid in records if pid in gts}, "manual_labels"


def rampnet_predictions(records, pano_ids=None):
    """RampNet's own detections from ``records.jsonl``, as ``{pano_id: [[x, y, conf]]}``.

    The same shape as a ``benchmark/model_detections`` file's ``detections`` field, so
    RampNet is scored through exactly the path every other model is. ``pano_ids``
    restricts (and orders) the output; a pano with no ``detections`` key gets ``[]``.
    The floats are copied, not re-rounded, so the scores are identical to scoring the
    record dicts directly.
    """
    pids = list(records) if pano_ids is None else list(pano_ids)
    return {pid: [[d["x_normalized"], d["y_normalized"], d["confidence"]]
                  for d in records[pid].get("detections", [])]
            for pid in pids}


def normalized_sha256(path):
    """Full sha256 hex of a file's bytes with CRLF folded to LF.

    LF-normalised so a ``core.autocrlf=true`` Windows checkout and a Linux clone agree
    (same convention as ``laurens_paired_151.input_hashes`` and the scorer fingerprint).
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def labels_digest(labels_dir):
    """One 16-hex digest over every ``*.txt`` label file, sorted by stem.

    Built the way ``scripts/analysis/imagery_manifest.digest_of`` builds an imagery
    digest (``stem|sha256`` joined by ``;``), over LF-normalised file bytes."""
    stems = sorted(n[:-4] for n in os.listdir(labels_dir) if n.endswith(".txt"))
    spec = ";".join(f"{s}|{normalized_sha256(os.path.join(labels_dir, s + '.txt'))}"
                    for s in stems)
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16], len(stems)


def split_pins(bundle_dir):
    """Content hashes that name the exact split files a score was computed against.

    Returns ``{records_sha256, n_records, gt_kind, n_reviewed, imagery_digest}`` plus
    ``verdicts_sha256`` for a verdict bundle, or ``gt_source_sha256`` and
    ``labels_digest`` for a manual-GT bundle. ``imagery_digest`` is read from the
    committed ``imagery_manifest.json`` (``None`` if the bundle has none); the images
    themselves are not re-hashed, because they are not in the repo.

    ``n_reviewed`` is the number of verdict entries (city bundles) or label files
    (manual-GT bundles). Example::

        >>> split_pins("benchmark/richmond")["n_records"]
        124
    """
    pins = {"records_sha256": normalized_sha256(os.path.join(bundle_dir, "records.jsonl"))}
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as fh:
        pins["n_records"] = sum(1 for line in fh if line.strip())
    vpath = os.path.join(bundle_dir, "verdicts.json")
    gpath = os.path.join(bundle_dir, "gt_source.json")
    if os.path.exists(vpath):
        pins["gt_kind"] = "verdicts"
        pins["verdicts_sha256"] = normalized_sha256(vpath)
        with open(vpath, encoding="utf-8") as fh:
            pins["n_reviewed"] = len(json.load(fh)["panos"])
    elif os.path.exists(gpath):
        pins["gt_kind"] = "manual_labels"
        pins["gt_source_sha256"] = normalized_sha256(gpath)
        with open(gpath, encoding="utf-8") as fh:
            labels_dir = os.path.normpath(os.path.join(bundle_dir, json.load(fh)["labels_dir"]))
        pins["labels_digest"], pins["n_reviewed"] = labels_digest(labels_dir)
    else:
        raise SystemExit(f"{bundle_dir}: no verdicts.json or gt_source.json -- split pins "
                         "cover scored splits only")
    mpath = os.path.join(bundle_dir, "imagery_manifest.json")
    pins["imagery_digest"] = None
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            pins["imagery_digest"] = json.load(fh).get("digest")
    return pins
