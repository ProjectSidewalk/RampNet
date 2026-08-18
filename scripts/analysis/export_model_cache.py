"""Publish the challenger detections so they stop living on one machine (#46, replication).

``.model_cache/`` holds every challenger's per-panorama detections — Gemini x2, Qwen x2,
Molmo, OWLv2, Grounding DINO across nine splits. They cost GPU-hours on Hyak and paid
API calls to produce, they are the input to ``fp_taxonomy.py``, ``silent_witness.py``,
``complementarity.py`` and ``null_recall.py``, and they are **git-ignored and
unpublished**. Every number those scripts produce is therefore reproducible only where
that directory happens to exist. ``docs/replication.md`` calls this blocker 1.

The cache is 18.8 MB — the same size class as ``analysis_out/op_cache``, which we
already commit for exactly this reason. The only real objection is its shape: 12,951
single-panorama shards keyed by an opaque SHA-1 of (label, signature, city, pano),
which is fine as a working cache and hostile as a published artifact — slow to check
out, and unreadable without reconstructing detector signatures.

So this consolidates it into **one JSON per (model, split)** — about 60 human-readable
files under ``benchmark/model_detections/`` — keyed by panorama id, with the detector's
signature recorded inside so the provenance survives.

    python scripts/analysis/export_model_cache.py --out benchmark/model_detections
    python scripts/analysis/export_model_cache.py --verify   # exported == cached

``--verify`` re-scores every split from the exported files and from ``.model_cache``
and asserts identical per-pano (TP, FP), because a published artifact that silently
differs from what produced the paper's numbers is worse than none. It fails rather
than passes when it had nothing to compare — a green run must mean "checked and
matched", never "found no cache and said nothing".

**``--models`` defaults to ``CHALLENGERS``, so a model outside that roster is neither
exported nor verified by the commands above.** Publishing an off-roster leg means
naming it explicitly, e.g.::

    python scripts/analysis/export_model_cache.py --models gemini:gemini-3.7-flash
    python scripts/analysis/export_model_cache.py --verify --models gemini:gemini-3.7-flash

and the exact command belongs in ``docs/replication.md`` beside the result, because
the default command will silently skip those files forever otherwise.

Downstream code reads the export through :func:`load_detections`, preferring it over
``.model_cache`` when present, so a fresh clone works with no cache at all.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet import roster  # noqa: E402

from fp_taxonomy import CHALLENGERS, _compare_args  # noqa: E402
from miss_decomposition import ALL_SPLITS  # noqa: E402

PUBLISHED_DIR = os.path.join(REPO, "benchmark", "model_detections")


#: Filesystem-safe model id, e.g. ``IDEA-Research/grounding-dino-base`` ->
#: ``IDEA-Research__grounding-dino-base``. Defined in the roster so the test that
#: checks this directory against the registry can spell a filename without importing
#: the exporter; re-exported here because that is where callers look for it.
slug = roster.slug


def spec_label(spec, cargs):
    """The label a ``--models`` spec resolves to, WITHOUT building a detector.

    ``build_detector`` would give the same answer, but it imports the detector stack
    (torch, transformers). The whole point of publishing the detections is that a
    fresh clone can score them with neither, so the published path must not drag that
    import in. ``roster.label_for`` is that torch-free resolution, shared with every
    other caller; ``cargs`` is still consulted so a run with an overridden provider
    model labels itself with the model actually used.
    """
    return roster.label_for(spec, cargs)


def load_detections(label, city, published_dir=PUBLISHED_DIR, publish_as=None):
    """``{pano_id: [points]}`` from the published export, or ``None`` if absent.

    The published files are the replication path; ``.model_cache`` remains the working
    cache for runs that are still producing detections. Callers try this first so a
    clean clone needs no cache.
    """
    path = published_path(label, city, published_dir, publish_as)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["detections"]


def published_path(label, city, out_dir=PUBLISHED_DIR, publish_as=None):
    """Where one (model, split) export lives.

    ``publish_as`` exists because a model id is NOT always enough to name a leg.
    Claude (#122) was the first provider where one id yields several distinct
    legs — `claude-sonnet-5` at effort `low` and at effort `high` are different
    detections with different cache signatures — and both would land on
    ``claude-sonnet-5__annapolis.json``, the second silently overwriting the
    first. The cache LABEL must stay the bare model id (it is baked into the
    already-paid cache keys), so the distinguishing name belongs here, at
    publication time, and nowhere else.

    Callers should prefer ``publication_name`` to fill ``publish_as``: the registry
    already records every leg's published name, and a flag that has to be typed
    from memory is a flag that will one day not be."""
    return os.path.join(out_dir, f"{slug(publish_as or label)}__{city}.json")


def publication_name(spec, cargs, publish_as=None):
    """What this leg publishes under: the explicit flag, else the registry, else
    the plain label.

    Without this, re-exporting a pinned leg and forgetting ``--publish-as`` writes
    the bare model id — `claude-opus-5__annapolis.json` — which collides with
    nothing, so the overwrite guard stays quiet, and surfaces only later as a file
    that belongs to no registered leg. The registry knows the answer; this is the
    one place that writes the filename, so this is where it should ask.
    """
    if publish_as:
        return publish_as
    leg = roster.leg_for(spec, cargs)
    return roster.published_name(leg) if leg is not None else spec_label(spec, cargs)


def export(cache_dir, out_dir, splits, specs, allow_partial=False, overrides=None,
           publish_as=None):
    """Consolidate ``.model_cache`` into one file per (model, split).

    A split whose cache is incomplete is REFUSED unless ``allow_partial``. A
    partial export is the dangerous artifact here: it looks like a finished split
    everywhere downstream (``load_detections`` returns only ``detections`` and
    drops ``n_uncached``; ``silent_witness`` narrows to the panos present without
    reporting a count), and ``--verify`` cannot catch it either, because a pano
    uncached at export time is absent from both sides and never compared. So the
    decision to publish a partial leg has to be made deliberately, not by default.

    ``overrides`` patches fields of the ``compare.py``-defaults namespace. Every
    field feeds the detector's cache signature, so an export must be run with the
    same settings as the run that produced the detections — e.g. the supervised
    YOLO pano arms (#51) ran ``--tiling none --yolo-imgsz 1280``, and an export at
    the defaults would rebuild a different signature and silently find no cache.
    """
    import compare as C
    from detectors import build_detector, parse_model_spec

    os.makedirs(out_dir, exist_ok=True)
    cache = C.DetectionCache(cache_dir, enabled=True)
    cargs = _compare_args(cache_dir)
    for k, v in (overrides or {}).items():
        setattr(cargs, k, v)
    written, skipped, partial, collisions = [], [], [], []
    if publish_as and len(specs) > 1:
        raise ValueError("--publish-as names ONE leg, so it cannot be combined with "
                         f"several --models specs (got {len(specs)}): every spec "
                         "would write to the same file.")
    for spec in specs:
        name = publication_name(spec, cargs, publish_as)
        for city in splits:
            bundle = os.path.join(REPO, "benchmark", city)
            if not os.path.exists(os.path.join(bundle, "records.jsonl")):
                continue
            records, verdicts, _ = C.load_bundle(bundle)
            gts = (C.load_manual_ground_truths(bundle) if verdicts is None
                   else C.ground_truths_from_verdicts(records, verdicts))
            provider, model_id = parse_model_spec(spec)
            label, det = build_detector(provider, model_id, records, cargs)
            sig = det.signature() if hasattr(det, "signature") else None
            if sig is None:
                continue
            dets, missing = {}, 0
            for pid in gts:
                pts = cache.get(C.cache_key(label, sig, city, pid))
                if pts is None:
                    missing += 1
                    continue
                dets[pid] = pts
            if not dets:
                skipped.append((label, city))
                continue
            if missing and not allow_partial:
                partial.append((label, city, len(dets), missing))
                continue
            path = published_path(label, city, out_dir, name)
            # Refuse to overwrite a DIFFERENT leg that happens to share this name.
            # Two legs of one model id (Claude at two effort levels) resolve to the
            # same filename, and a silent overwrite is the worst outcome available:
            # the file still looks complete, --verify still passes against whichever
            # leg was written last, and the other leg's numbers are simply gone.
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh).get("signature")
                if existing is not None and existing != sig:
                    collisions.append((label, city, path))
                    continue
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"model": label, "published_as": name,
                           "city": city, "signature": sig,
                           "n_panos": len(dets), "n_uncached": missing,
                           "detections": dets}, fh, separators=(",", ":"), sort_keys=True)
            written.append((label, city, len(dets), missing, os.path.getsize(path)))
    return written, skipped, partial, collisions


def verify(cache_dir, out_dir, splits, specs, overrides=None, publish_as=None):
    """Do the exported detections score identically to the cached ones?

    Returns ``(compared, problems, vacuous, unpublished)`` where ``compared``
    counts (model, split) pairs that actually had panos on both sides. A pair with
    nothing to compare is NOT a pass — see the ``vacuous`` bookkeeping below.
    ``overrides`` carries the producing run's signature knobs, exactly as in
    ``export``: verifying at the wrong ones finds no cache and compares nothing."""
    import compare as C
    from detectors import build_detector, parse_model_spec
    from rampnet.detection_eval import radius_sq_for, score_pano

    cache = C.DetectionCache(cache_dir, enabled=True)
    cargs = _compare_args(cache_dir)
    for k, v in (overrides or {}).items():
        setattr(cargs, k, v)
    rsq = radius_sq_for()
    problems, compared, vacuous, unpublished = [], 0, [], []
    for spec in specs:
        name = publication_name(spec, cargs, publish_as)
        for city in splits:
            bundle = os.path.join(REPO, "benchmark", city)
            if not os.path.exists(os.path.join(bundle, "records.jsonl")):
                continue
            records, verdicts, _ = C.load_bundle(bundle)
            gts = (C.load_manual_ground_truths(bundle) if verdicts is None
                   else C.ground_truths_from_verdicts(records, verdicts))
            provider, model_id = parse_model_spec(spec)
            label, det = build_detector(provider, model_id, records, cargs)
            sig = det.signature() if hasattr(det, "signature") else None
            if sig is None:
                continue
            pub = load_detections(label, city, out_dir, name)
            if pub is None:
                unpublished.append((label, city))
                continue
            # (tp, fp) per pano, compared as tuples. The old tp*1000+fp encoding
            # aliases once a split's FP count reaches 1000, and OWLv2 / Grounding
            # DINO run 7,300-9,700 detections per split -- an order of magnitude
            # past the modulus, so a compensating difference would have verified
            # as identical on exactly the legs with the most to hide.
            a, b, n = [], [], 0
            for pid, gt in gts.items():
                cached = cache.get(C.cache_key(label, sig, city, pid))
                if cached is None:
                    continue
                sa = score_pano(cached, gt, radius_sq=rsq)
                sb = score_pano(pub.get(pid, []), gt, radius_sq=rsq)
                a.append((pid, sa.tp, sa.fp))
                b.append((pid, sb.tp, sb.fp))
                n += 1
            if n == 0:
                # Nothing on the cache side: the old code counted this as a pass,
                # so a missing .model_cache printed a clean bill of health having
                # compared nothing at all.
                vacuous.append((label, city))
                continue
            compared += 1
            if a != b:
                diffs = sum(1 for x, y in zip(a, b) if x != y)
                problems.append(f"{label} / {city}: cached != published "
                                f"({diffs} of {n} panos differ)")
    return compared, problems, vacuous, unpublished


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--out", default=PUBLISHED_DIR)
    p.add_argument("--splits", default=",".join(ALL_SPLITS))
    p.add_argument("--models", default=",".join(CHALLENGERS))
    p.add_argument("--verify", action="store_true",
                   help="Only check that the export scores identically to the cache.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Publish a split whose cache is incomplete. Off by default: a "
                        "partial export looks complete to every downstream reader.")
    p.add_argument("--tiling", default="perspective", choices=["perspective", "none"],
                   help="Tiling mode of the run that produced the detections. It is part "
                        "of every detector's cache signature, so it must match the "
                        "producing run (the #51 YOLO pano arms ran --tiling none).")
    p.add_argument("--yolo-imgsz", type=int, default=1024,
                   help="YOLO inference imgsz of the producing run (signature field; "
                        "the #51 pano arms ran 1280).")
    p.add_argument("--claude-effort", default="low",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude reasoning effort of the producing run (signature field). "
                        "One model id yields a DIFFERENT leg per effort level, so this "
                        "must match the run or the export finds no cache.")
    p.add_argument("--claude-tool-choice", default="auto", choices=["auto", "forced"],
                   help="Claude tool choice of the producing run (signature field).")
    p.add_argument("--publish-as",
                   help="Filename stem for this leg, when the model id alone does not "
                        "identify it — e.g. claude-sonnet-5 at two effort levels are two "
                        "legs that would otherwise both write claude-sonnet-5__<split>.json. "
                        "Names ONE leg, so it takes a single --models spec. The SAME value "
                        "must be passed to --verify.")
    args = p.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    overrides = {"tiling": args.tiling, "yolo_imgsz": args.yolo_imgsz,
                 "claude_effort": args.claude_effort,
                 "claude_tool_choice": args.claude_tool_choice}

    if args.verify:
        compared, problems, vacuous, unpublished = verify(
            args.cache_dir, args.out, splits, specs, overrides, args.publish_as)
        print(f"compared {compared} (model, split) pair(s) against {args.cache_dir}")
        for msg in problems:
            print(f"  ✗ {msg}")
        for label, city in vacuous:
            print(f"  ! {label} / {city}: published, but the cache has none of its "
                  f"panos — NOTHING was compared")
        for label, city in unpublished:
            print(f"  - {label} / {city}: no published export to check")
        if problems:
            print(f"{len(problems)} MISMATCH(ES)")
            return 1
        if compared == 0:
            # The whole point of --verify is proving the published files match what
            # produced the numbers. With nothing compared it proves nothing, and
            # printing a pass here is worse than printing nothing.
            print("NOTHING VERIFIED: no (model, split) pair had panos on both sides. "
                  "Point --cache-dir at the cache that produced the export.")
            return 1
        if vacuous:
            print(f"{len(vacuous)} published pair(s) had NO cached panos to check "
                  f"against — they are unverified, not verified.")
        print(f"{compared} pair(s): published detections score IDENTICALLY to the cache")
        return 1 if vacuous else 0

    written, skipped, partial, collisions = export(
        args.cache_dir, args.out, splits, specs, allow_partial=args.allow_partial,
        overrides=overrides, publish_as=args.publish_as)
    total = sum(w[4] for w in written)
    print(f"wrote {len(written)} file(s), {total/1e6:.1f} MB total -> {args.out}\n")
    print(f"{'model':>42} {'split':>20} {'panos':>6} {'uncached':>9} {'KB':>7}")
    for label, city, n, missing, size in written:
        flag = "  <-- PARTIAL" if missing else ""
        print(f"{label:>42} {city:>20} {n:>6} {missing:>9} {size/1024:>7.0f}{flag}")
    for label, city in skipped:
        print(f"  (no cache: {label} / {city})")
    for label, city, n, missing in partial:
        print(f"  REFUSED (incomplete): {label} / {city} — {n} cached, {missing} "
              f"uncached. Finish the leg, or pass --allow-partial to publish it "
              f"anyway and say so where the numbers are quoted.")
    for label, city, path in collisions:
        print(f"  REFUSED (name collision): {label} / {city} — {path} already holds a "
              f"DIFFERENT leg's detections (its recorded signature does not match this "
              f"run's). One model id can be several legs; give this one a distinct "
              f"--publish-as instead of overwriting the other.")
    if collisions:
        return 1
    if any(w[3] for w in written):
        print("\nNOTE: a partial export is indistinguishable from a complete one "
              "downstream (load_detections drops n_uncached, and --verify never "
              "compares a pano the cache lacks). Record the gap next to the result.")
    print(f"\nNow run --verify before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
