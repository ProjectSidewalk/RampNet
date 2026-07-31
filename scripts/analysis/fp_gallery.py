"""Crops for the false positives geometry cannot explain (#46, the FP gallery).

``fp_taxonomy.py`` bucketed every model's false positives and found that 60-81% of
them land in ``isolated`` — not on a real ramp, not on the ego vehicle, just
somewhere. That bucket is explicitly an **upper bound on hallucination**, because a
driveway apron, crosswalk paint and a flight of stairs all fall into it and geometry
cannot tell them apart. #46 asks for the gallery that can.

It also sets the ceiling on the #35 cascade. If OWLv2's isolated false positives are
obvious junk, an arbiter kills them cheaply; if they are ambiguous concrete, the
arbiter struggles exactly where the detector did. Nothing in the cached detections
answers that — only looking does.

This renders the same three-panel crop ``miss_gallery.py`` uses (context / detail /
**as the model saw it**), through the same instrument and into the same manifest
format, so FP verdicts and miss verdicts stay comparable and joinable. The
resolution-parity reasoning carries over unchanged and matters just as much: on a 4x
split a reviewer calling a false positive "obviously not a ramp" may be using detail
the model never had.

Sampling is deliberate and reported, never silent. The full isolated population is
~41,000 boxes for OWLv2 alone, so a gallery is necessarily a sample; ``--sample``
takes the model's **highest-confidence** isolated false positives, which are the ones
an arbiter would have to overrule and the ones #46 means by "worst cases". Chat VLMs
emit no score, so for them the order is the stable cache order and the sheet says so.

    python scripts/analysis/fp_gallery.py --models owlv2 --sample 24 \\
        --render analysis_out/fp_gallery --panos-root .

No GPU, no model load — ``.model_cache`` plus the committed bundles, exactly as
``fp_taxonomy.py``.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet.detection_eval import radius_sq_for  # noqa: E402

import fp_taxonomy as fx  # noqa: E402
import miss_gallery as mg  # noqa: E402
from miss_decomposition import US_SPLITS  # noqa: E402
from stage1_label_recall import geom  # noqa: E402

DEFAULT_SAMPLE = 24


def rank_key(row):
    """Sort key putting the highest-confidence false positives first.

    A scoreless prediction (chat VLM) sorts last rather than first, so a model that
    emits no confidence never displaces a scored model's genuinely-worst cases when
    several are sampled together. Ties fall back to a stable pano/coordinate order so
    the sample is reproducible without a seed.
    """
    conf = row.get("confidence")
    return (0 if conf is not None else 1,
            -(conf if conf is not None else 0.0),
            row["city"], row["pano"], row["x"], row["y"])


def to_gallery_items(rows, panos_root, sample=None):
    """Attach the imagery facts ``miss_gallery.render`` needs, then take the worst N.

    ``px`` here is the apparent width a ramp *would* have at this point's distance —
    a false positive has no true size, but the question "could a reviewer tell what
    the model latched onto?" is still governed by how far away it is, so the same
    judgeability floor applies with the same meaning.
    """
    items = []
    for r in sorted(rows, key=rank_key):
        w = mg.pano_width(r["city"], r["pano"], panos_root)
        if w is None:
            continue
        dist, px = geom(r["y"])
        spx = mg.source_px(px, w)
        items.append({**r, "dist": dist, "px": px, "source_width": w,
                      "source_px": spx, "parity": mg.parity_class(w),
                      "judgeable": mg.judgeable(spx)})
        if sample and len(items) >= sample:
            break
    return items


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default="owlv2",
                   help="Comma-separated compare.py specs (default owlv2 — the "
                        "largest FP population and the #35 cascade's blocker).")
    p.add_argument("--bucket", default="isolated", choices=list(fx.BUCKETS))
    p.add_argument("--cities", default=",".join(US_SPLITS))
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                   help="Worst-N per model (0 = all). Always reported, never silent.")
    p.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    p.add_argument("--panos-root", default=REPO)
    p.add_argument("--render", default=None, metavar="DIR")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    import compare as C
    cache = C.DetectionCache(args.cache_dir, enabled=True)
    cargs = fx._compare_args(args.cache_dir)
    radius_sq = radius_sq_for()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    specs = [s.strip() for s in args.models.split(",") if s.strip()]

    print(f"=== FP gallery: '{args.bucket}' false positives (#46) ===\n")
    summary = {}
    all_items = []
    for spec in specs:
        rows = []
        label = spec
        for city in cities:
            got = fx.model_rows(city, spec, cache, cargs, radius_sq, fx.HOOD_Y)
            if got is None:
                continue
            label, city_rows, _ = got
            rows.extend(r for r in city_rows if r["bucket"] == args.bucket)
        scored = sum(1 for r in rows if r.get("confidence") is not None)
        items = to_gallery_items(rows, args.panos_root,
                                 args.sample or None)
        for it in items:
            it["model"] = label
        all_items.extend(items)
        s = mg.summarize_feasibility(items)
        summary[label] = {"population": len(rows), "sampled": len(items),
                          "scored": scored, "feasibility": s}
        print(f"{label}")
        print(f"  population {len(rows)} {args.bucket} FPs "
              f"({'confidence-ranked' if scored else 'no confidence — stable cache order'})")
        print(f"  SAMPLED {len(items)}"
              + (f" of {len(rows)} — the other {len(rows) - len(items)} are NOT shown"
                 if args.sample and len(rows) > len(items) else ""))
        print(f"  judgeable {s['judgeable']}/{s['total']}   "
              f"parity {s['parity']['n']} / advantaged {s['advantaged']['n']}")

    print(f"\n{'-'*76}")
    print("The same instrument caveat as the miss gallery applies, and it cuts the")
    print("other way here: on an 'advantaged' pano a reviewer calling a false positive")
    print("'obviously not a ramp' may be using detail the model never had. The third")
    print("panel shows the model's own pixel budget so that call can be made fairly.")

    if args.render:
        n, skipped = mg.render(all_items, args.render, True, args.panos_root,
                               extra_fields=("model", "confidence", "bucket"))
        print(f"\nWrote {n} crops to {args.render} "
              f"({skipped} skipped below the {mg.JUDGEABLE_SOURCE_PX:.0f} source-px floor)")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"bucket": args.bucket, "sample": args.sample,
                       "cities": cities, "per_model": summary}, fh, indent=2)
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
