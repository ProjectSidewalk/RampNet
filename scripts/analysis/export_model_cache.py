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
and asserts identical TP/FP counts, because a published artifact that silently differs
from what produced the paper's numbers is worse than none.

Downstream code reads the export through :func:`load_detections`, preferring it over
``.model_cache`` when present, so a fresh clone works with no cache at all.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from fp_taxonomy import CHALLENGERS, _compare_args  # noqa: E402
from miss_decomposition import ALL_SPLITS  # noqa: E402

PUBLISHED_DIR = os.path.join(REPO, "benchmark", "model_detections")


def slug(label):
    """Filesystem-safe model id. ``IDEA-Research/grounding-dino-base`` -> ``IDEA-Research__grounding-dino-base``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", label)


def spec_label(spec, cargs):
    """The label a ``--models`` spec resolves to, WITHOUT building a detector.

    ``build_detector`` would give the same answer, but it imports the detector stack
    (torch, transformers). The whole point of publishing the detections is that a
    fresh clone can score them with neither, so the published path must not drag that
    import in. Provider defaults come from ``_compare_args``, which a test already
    cross-checks against ``compare.py``'s parser, so this cannot drift on its own.
    """
    provider, _, model_id = spec.partition(":")
    if model_id.strip():
        return model_id.strip()
    return getattr(cargs, f"{provider.strip()}_model", provider.strip())


def load_detections(label, city, published_dir=PUBLISHED_DIR):
    """``{pano_id: [points]}`` from the published export, or ``None`` if absent.

    The published files are the replication path; ``.model_cache`` remains the working
    cache for runs that are still producing detections. Callers try this first so a
    clean clone needs no cache.
    """
    path = os.path.join(published_dir, f"{slug(label)}__{city}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["detections"]


def export(cache_dir, out_dir, splits, specs):
    """Consolidate ``.model_cache`` into one file per (model, split)."""
    import compare as C
    from detectors import build_detector, parse_model_spec

    os.makedirs(out_dir, exist_ok=True)
    cache = C.DetectionCache(cache_dir, enabled=True)
    cargs = _compare_args(cache_dir)
    written, skipped = [], []
    for spec in specs:
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
            path = os.path.join(out_dir, f"{slug(label)}__{city}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"model": label, "city": city, "signature": sig,
                           "n_panos": len(dets), "n_uncached": missing,
                           "detections": dets}, fh, separators=(",", ":"), sort_keys=True)
            written.append((label, city, len(dets), missing, os.path.getsize(path)))
    return written, skipped


def verify(cache_dir, out_dir, splits, specs):
    """Do the exported detections score identically to the cached ones?"""
    import compare as C
    from detectors import build_detector, parse_model_spec
    from rampnet.detection_eval import radius_sq_for, score_pano

    cache = C.DetectionCache(cache_dir, enabled=True)
    cargs = _compare_args(cache_dir)
    rsq = radius_sq_for()
    problems, checked = [], 0
    for spec in specs:
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
            pub = load_detections(label, city, out_dir)
            if pub is None:
                continue
            a = b = 0
            for pid, gt in gts.items():
                cached = cache.get(C.cache_key(label, sig, city, pid))
                if cached is None:
                    continue
                sa = score_pano(cached, gt, radius_sq=rsq)
                sb = score_pano(pub.get(pid, []), gt, radius_sq=rsq)
                a += sa.tp * 1000 + sa.fp
                b += sb.tp * 1000 + sb.fp
            checked += 1
            if a != b:
                problems.append(f"{label} / {city}: cached != published")
    return checked, problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--out", default=PUBLISHED_DIR)
    p.add_argument("--splits", default=",".join(ALL_SPLITS))
    p.add_argument("--models", default=",".join(CHALLENGERS))
    p.add_argument("--verify", action="store_true",
                   help="Only check that the export scores identically to the cache.")
    args = p.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    specs = [s.strip() for s in args.models.split(",") if s.strip()]

    if args.verify:
        checked, problems = verify(args.cache_dir, args.out, splits, specs)
        print(f"verified {checked} (model, split) pair(s)")
        for msg in problems:
            print(f"  ✗ {msg}")
        print("published detections score IDENTICALLY to the cache"
              if not problems else f"{len(problems)} MISMATCH(ES)")
        return 1 if problems else 0

    written, skipped = export(args.cache_dir, args.out, splits, specs)
    total = sum(w[4] for w in written)
    print(f"wrote {len(written)} file(s), {total/1e6:.1f} MB total -> {args.out}\n")
    print(f"{'model':>42} {'split':>20} {'panos':>6} {'uncached':>9} {'KB':>7}")
    for label, city, n, missing, size in written:
        print(f"{label:>42} {city:>20} {n:>6} {missing:>9} {size/1024:>7.0f}")
    for label, city in skipped:
        print(f"  (no cache: {label} / {city})")
    print(f"\nNow run --verify before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
