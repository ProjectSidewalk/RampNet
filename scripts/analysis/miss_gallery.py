"""Crops for the misses the cached detections cannot explain (#46, the gallery half).

``miss_taxonomy.py`` bucketed every miss from the cached peaks alone and left one
bucket geometry cannot open: **silent** — 128 pooled misses where the model produced
nothing at all, even at the 0.05 floor. That bucket still contains occlusion, deep
shadow, surface debris and GT disagreement mixed together, and it is the number the
sourcing programme is sized against. Separating those causes needs the imagery.

This renders the crops a reviewer needs, and — before it renders anything — says
**which misses are adjudicable at all**. That order matters. A confident label on a
17-pixel ramp is worse than no label, and this benchmark has already been burned once
by a visual pass made before the instrument was checked.

## The instrument check, and why it changes the method

``geom()`` reports apparent size at the **model's** 4096-px-wide input. The stored
panoramas are a different size per split, so the reviewer's pixels and the model's
pixels are not the same thing:

* **4x (bend, paterson, gainesville)** — the reviewer sees four times the linear
  detail the model had. "I can see the ramp" therefore does **not** imply the model
  should have: those pixels never reached it.
* **1x (morgantown)** — stored at exactly the model's input width. The reviewer and
  the model see *identical* pixels, so here "I can see it clearly" **does** imply an
  appearance failure rather than a resolution one. Morgantown is the cleanest split
  in the benchmark for that inference, precisely because it has no resolution
  headroom.
* **richmond is mixed within the split** (4096-12288 px wide; iSTAR Pulsar and GoPro
  Max panos in one bundle), so parity is decided per pano, never per split.

So each crop carries its own parity class, and a verdict of "appearance failure" is
only sound on the ``parity`` ones. That is a real restriction on what the gallery can
conclude, and it is better stated up front than discovered afterwards.

## What it renders

Three panels per miss. The first two are sampled at **native source resolution** (view
width is derived from the field of view and the panorama's own width, so the crop
neither invents detail nor throws it away):

* **context** — wide enough to show a parked car, a shadow, a construction barrier;
* **detail** — tight on the ramp;
* **as the model saw it** — the same detail view rendered from the panorama
  downsampled to the model's 4096-px input, blown back up with nearest-neighbour so
  it looks exactly as coarse as it really was.

That third panel is what makes the ``advantaged`` splits usable. Without it a
reviewer looking at 4x imagery cannot tell "the model should have caught this" from
"the model never had those pixels"; with it, the two readings sit side by side and
the comparison is direct rather than inferred.

Each view is a gnomonic reprojection centred on the ramp's own longitude and
latitude, which removes the equirectangular vertical stretch that is worst exactly
where ramps sit, and centres every subject identically so framing cannot bias a
judgment.

    python scripts/analysis/miss_gallery.py                        # feasibility only
    python scripts/analysis/miss_gallery.py --render analysis_out/miss_gallery
    python scripts/analysis/miss_gallery.py --render out --bucket merged --field near

Writes ``manifest.json`` alongside the crops, keyed ``<pano>_<x>_<y>`` exactly like
``benchmark/<city>/incremental_fp_tags.json`` so reviewer verdicts commit the same
way and stay joinable. Note the same fragility that file has: **the key encodes
coordinates**, so a re-extraction that moves a point orphans the tag.

Needs ``benchmark/<city>/panos`` locally (git-ignored). No GPU, no network.
"""
import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from miss_decomposition import US_SPLITS  # noqa: E402
import miss_taxonomy as mt  # noqa: E402

# The model's input width (rampnet.model.PANO_INPUT_SIZE is (2048, 4096)), which is
# the scale geom() reports apparent size at. Every parity judgment is relative to it.
MODEL_WIDTH = 4096.0

# Field of view for the two panels. Context is wide enough to carry the occluder;
# detail is tight enough that a 20-px ramp still fills a usable fraction of the frame.
CONTEXT_FOV_DEG = 60.0
DETAIL_FOV_DEG = 16.0

# Cap on rendered panel width. Native sampling on a 16384-px pano at 60 deg would be
# 2731 px per panel; past this the extra pixels cost disk without helping a reviewer.
MAX_PANEL_PX = 1400

# Common on-screen size for all three panels. Every panel is scaled to this with
# NEAREST, so what differs between them is only their real information content --
# the native width each was rendered at is printed on the panel itself.
DISPLAY_PX = 640

# Below this the ramp is too few pixels for anyone — reviewer or model — to call
# occlusion apart from shadow apart from debris. Items under it are still rendered
# (so the claim is checkable) but are marked unjudgeable and excluded from any rate.
JUDGEABLE_SOURCE_PX = 30.0

# How much source resolution over the model's counts as an advantage. Anything at or
# under this is parity: the reviewer is looking at the model's own pixels.
PARITY_MAX_RATIO = 1.05


# --------------------------------------------------------------------------- #
# Pure core (no I/O) — unit-tested in tests/test_miss_gallery.py
# --------------------------------------------------------------------------- #
def native_panel_width(fov_deg, source_width, max_px=MAX_PANEL_PX):
    """Panel width that samples the source at ~1:1 for this field of view.

    A panorama ``source_width`` px around covers 360 degrees, so ``fov`` degrees is
    ``fov/360 * source_width`` px of real imagery. Rendering wider than that
    upsamples — it invents detail a reviewer may mistake for evidence. Rendering
    narrower discards detail the imagery actually has. Both are failures of an
    instrument, so the width is derived rather than chosen.
    """
    return max(64, min(int(max_px), int(round(fov_deg / 360.0 * source_width))))


def source_px(model_px, source_width, model_width=MODEL_WIDTH):
    """Apparent ramp width in the STORED imagery, from its size at the model's input."""
    return model_px * (source_width / model_width)


def parity_class(source_width, model_width=MODEL_WIDTH, parity_max=PARITY_MAX_RATIO):
    """``'parity'`` or ``'advantaged'`` — what the reviewer sees relative to the model.

    The distinction is the whole methodological point of this script. On a
    ``parity`` pano, a ramp a reviewer can see is a ramp the model had every pixel of
    and still missed, so "appearance failure" is a sound reading. On an
    ``advantaged`` pano it is not: the reviewer is looking at detail that never
    reached the model, and the same verdict would confound appearance with
    resolution.
    """
    return "parity" if source_width / model_width <= parity_max else "advantaged"


def judgeable(src_px, floor=JUDGEABLE_SOURCE_PX):
    """Is there enough imagery here for any reviewer to attribute a cause?"""
    return src_px >= floor


def views_for(x, y, source_width, context_fov=CONTEXT_FOV_DEG,
              detail_fov=DETAIL_FOV_DEG):
    """``(context_view, detail_view)`` centred on a pano-normalized point.

    Yaw and pitch come from the point itself — ``lon = (x-0.5)*360``,
    ``lat = (0.5-y)*180``, matching ``equirect_tiling``'s convention — so the subject
    lands dead centre in both panels. Centring on the subject rather than on a fixed
    view ring is what makes crops comparable to each other: nothing is judged at the
    edge of a frame in one case and the middle in another.
    """
    from equirect_tiling import View
    yaw = (x - 0.5) * 360.0
    pitch = (0.5 - y) * 180.0
    out = []
    for fov in (context_fov, detail_fov):
        w = native_panel_width(fov, source_width)
        out.append(View(yaw, pitch, fov, fov, w, w))
    return tuple(out)


def tag_key(pano, x, y):
    """Match ``benchmark/<city>/incremental_fp_tags.json``'s key exactly.

    Same construction, same fragility: the coordinates are part of the identity, so a
    re-extraction that nudges a point orphans the reviewer's verdict rather than
    silently mis-attaching it. ``low_floor_sweep.py tagcheck`` exists for that reason
    and the same check applies here.
    """
    return f"{pano}_{_short(x)}_{_short(y)}"


def _short(v):
    return f"{round(float(v), 5):g}"


def summarize_feasibility(items):
    """Counts by parity class and judgeability, which bound what the gallery can say."""
    out = {}
    for cls in ("parity", "advantaged"):
        sel = [i for i in items if i["parity"] == cls]
        out[cls] = {
            "n": len(sel),
            "judgeable": sum(1 for i in sel if i["judgeable"]),
            "median_source_px": (sorted(i["source_px"] for i in sel)[len(sel) // 2]
                                 if sel else None),
        }
    out["total"] = len(items)
    out["judgeable"] = sum(1 for i in items if i["judgeable"])
    # Every judgeable item licenses an appearance verdict, because the third panel
    # shows the model's own pixel budget next to the source detail — the reviewer
    # compares rather than infers. Without that panel this would be the parity count
    # alone, which is why the panel exists. `parity_only` is kept as the subset where
    # the reading needs no comparison at all.
    out["appearance_licensed"] = out["judgeable"]
    out["parity_only"] = sum(
        1 for i in items if i["judgeable"] and i["parity"] == "parity")
    return out


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def pano_path(city, pano, panos_root=REPO):
    return os.path.join(panos_root, "benchmark", city, "panos", f"{pano}.jpg")


def pano_width(city, pano, panos_root=REPO, cache={}):
    """Stored width of one panorama, or ``None`` if the image is not on disk.

    ``benchmark/*/panos`` is git-ignored, so in a **worktree** it lives only in the
    main checkout. ``--panos-root`` exists for that case; defaulting to ``REPO`` is
    right for an ordinary clone.
    """
    key = (panos_root, city, pano)
    if key in cache:
        return cache[key]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        cache[key] = Image.open(pano_path(city, pano, panos_root)).size[0]
    except Exception:
        cache[key] = None
    return cache[key]


def load_queue(path):
    """``{(city, pano, x, y)}`` from ``silent_witness.py``'s unwitnessed list.

    The witnessed misses are already explained — some other model detected a ramp
    there, so the imagery demonstrably contains one — and putting them in front of a
    reviewer costs time for an answer already in hand. Restricting the gallery to the
    unwitnessed remainder is the difference between tagging 128 crops and 59.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {(r["city"], r["pano"], round(float(r["x"]), 6), round(float(r["y"]), 6))
            for r in payload.get("unwitnessed", [])}


def collect(bucket, field, threshold, cities, panos_root=REPO, queue=None):
    """Every miss in ``bucket`` (and optionally ``field``), with its imagery facts.

    ``queue``, when given, restricts the population to those points — see
    :func:`load_queue`.
    """
    items = []
    for city in cities:
        loaded = mt.load_rows(city, threshold, rng=None)
        if loaded is None:
            continue
        rows, _ = loaded
        for r in rows:
            if r["hit"] or r["bucket"] != bucket:
                continue
            if field and r["field"] != field:
                continue
            if queue is not None and (city, r["pano"], round(r["x"], 6),
                                      round(r["y"], 6)) not in queue:
                continue
            w = pano_width(city, r["pano"], panos_root)
            if w is None:
                items.append({**r, "source_width": None, "source_px": 0.0,
                              "parity": "missing", "judgeable": False})
                continue
            spx = source_px(r["px"], w)
            items.append({**r, "source_width": w, "source_px": spx,
                          "parity": parity_class(w), "judgeable": judgeable(spx)})
    return items


def render(items, outdir, only_judgeable=True, panos_root=REPO, extra_fields=()):
    """Write a three-panel crop per item plus the tagging manifest.

    Generic over what the point *is*: each item needs ``city``, ``pano``, ``x``,
    ``y``, ``source_width``, ``source_px``, ``parity`` and ``judgeable``. Misses
    carry ``bucket``/``field``; ``fp_gallery.py`` feeds false positives through the
    same path with ``model``/``confidence`` named in ``extra_fields``, so both
    galleries share one instrument and one manifest format rather than drifting.
    """
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    from equirect_tiling import equirect_to_perspective

    os.makedirs(outdir, exist_ok=True)
    manifest, skipped = {}, 0
    by_pano = {}
    for it in items:
        if only_judgeable and not it["judgeable"]:
            skipped += 1
            continue
        by_pano.setdefault((it["city"], it["pano"]), []).append(it)

    for (city, pano), group in sorted(by_pano.items()):
        img = Image.open(pano_path(city, pano, panos_root)).convert("RGB")
        # The pano as the MODEL received it. Downsampling the whole panorama once is
        # the faithful simulation — the model never saw a crop, it saw a 4096-px-wide
        # equirectangular image — and it is what turns an 'advantaged' pano into
        # usable evidence: the reviewer can compare full detail against the model's
        # actual pixel budget instead of guessing what was available.
        model_img = (img if img.width <= MODEL_WIDTH else
                     img.resize((int(MODEL_WIDTH), int(MODEL_WIDTH / 2)), Image.LANCZOS))
        for it in group:
            ctx, det = views_for(it["x"], it["y"], it["source_width"])
            # Third panel: the same detail view rendered at the MODEL's native
            # sampling. Its lower pixel count is the whole point, so it is never
            # rendered larger than the model's imagery supports.
            _, model_det = views_for(it["x"], it["y"], MODEL_WIDTH,
                                     detail_fov=DETAIL_FOV_DEG)
            panels = []
            for src, view, label in ((img, ctx, f"context {int(CONTEXT_FOV_DEG)}deg"),
                                     (img, det, f"detail {int(DETAIL_FOV_DEG)}deg source"),
                                     (model_img, model_det, "as the model saw it")):
                p = equirect_to_perspective(src, view)
                # Every panel is displayed at the same size so a reviewer can compare
                # them at a glance, and all three are scaled with NEAREST so the
                # difference between them stays exactly their true information
                # content. Interpolating the source panels would make the model panel
                # look worse than it is by contrast alone.
                if p.size != (DISPLAY_PX, DISPLAY_PX):
                    p = p.resize((DISPLAY_PX, DISPLAY_PX), Image.NEAREST)
                d = ImageDraw.Draw(p)
                # The subject is dead centre by construction; ring it without
                # covering it, so the marker cannot be mistaken for the ramp. Drawn
                # after scaling, so the ring is identical in every panel and cannot
                # be read as a size cue.
                c = DISPLAY_PX / 2
                r = int(DISPLAY_PX * 0.045)
                d.ellipse([c - r, c - r, c + r, c + r], outline=(60, 220, 90), width=3)
                d.rectangle([0, 0, DISPLAY_PX - 1, 13], fill=(16, 16, 16))
                d.text((4, 3), f"{label}  [{view.width}px native]", fill=(210, 210, 210))
                panels.append(p)
            h = DISPLAY_PX
            sheet = Image.new("RGB", (sum(p.width for p in panels) + 16, h), (16, 16, 16))
            xo = 0
            for p in panels:
                sheet.paste(p, (xo, 0))
                xo += p.width + 8
            key = tag_key(pano, it["x"], it["y"])
            name = f"{city}__{key}.jpg"
            sheet.save(os.path.join(outdir, name), quality=92)
            entry = {
                "city": city, "pano": pano, "x": it["x"], "y": it["y"],
                "file": name,
                "dist_m": round(it["dist"], 1),
                "model_px": round(it["px"], 1),
                "source_px": round(it["source_px"], 1),
                "source_width": it["source_width"],
                "parity": it["parity"],
            }
            for f in ("bucket", "field") + tuple(extra_fields):
                if f in it:
                    entry[f] = it[f]
            manifest[key] = entry
        img.close()

    # Rendering is grouped by panorama (opening a 16k-px JPEG is the expensive part),
    # but the REVIEW order is near-field first: those are the crops that close the
    # sourcing bracket, and a reviewer who runs out of time should have spent it on
    # them rather than on far-field ones that #59 already attributes to pixel
    # starvation. The manifest's order is the page's order, so it is sorted here.
    manifest = dict(sorted(manifest.items(),
                           key=lambda kv: (kv[1].get("field") != "near",
                                           kv[1].get("city", ""), kv[0])))

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"n": len(manifest), "skipped_unjudgeable": skipped,
                   "context_fov_deg": CONTEXT_FOV_DEG, "detail_fov_deg": DETAIL_FOV_DEG,
                   "judgeable_source_px": JUDGEABLE_SOURCE_PX,
                   "items": manifest}, fh, indent=2)
    return len(manifest), skipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bucket", default="silent", choices=list(mt.BUCKETS))
    p.add_argument("--field", default=None, choices=["near", "far"],
                   help="Restrict to one distance population (default: both).")
    p.add_argument("--threshold", type=float, default=mt.DEFAULT_THRESHOLD)
    p.add_argument("--cities", default=",".join(US_SPLITS))
    p.add_argument("--queue", default=None, metavar="SILENT_WITNESS_JSON",
                   help="Restrict to silent_witness.py's UNWITNESSED list — the misses "
                        "no other model explained, i.e. the ones that actually need eyes.")
    p.add_argument("--panos-root", default=REPO,
                   help="Checkout holding benchmark/<city>/panos (git-ignored, so in "
                        "a worktree it lives in the main checkout instead).")
    p.add_argument("--render", default=None, metavar="DIR",
                   help="Also write crops + manifest.json here.")
    p.add_argument("--include-unjudgeable", action="store_true",
                   help="Render items below the source-pixel floor too.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    queue = load_queue(args.queue) if args.queue else None
    items = collect(args.bucket, args.field, args.threshold, cities, args.panos_root,
                    queue)
    if not items:
        print(f"no '{args.bucket}' misses matched")
        return 0
    if queue is not None:
        print(f"[queue] restricted to the {len(queue)} UNWITNESSED misses from "
              f"{os.path.basename(args.queue)}; {len(items)} of them have imagery.\n")

    print(f"=== Gallery feasibility: '{args.bucket}' misses"
          f"{', ' + args.field + '-field' if args.field else ''} (#46) ===\n")
    print("The reviewer's pixels are not the model's pixels. geom() sizes a ramp at the")
    print(f"model's {int(MODEL_WIDTH)}-px input; the stored panos differ per split, so each")
    print("crop is classified by what it gives the reviewer OVER the model.\n")
    print(f"{'split':>14} {'n':>4} {'src/model':>10} {'parity':>9} "
          f"{'med src px':>11} {'judgeable':>10}")
    for city in cities:
        sel = [i for i in items if i["city"] == city]
        if not sel:
            continue
        widths = [i["source_width"] for i in sel if i["source_width"]]
        ratios = sorted(w / MODEL_WIDTH for w in widths) or [0]
        med_ratio = ratios[len(ratios) // 2]
        classes = {i["parity"] for i in sel}
        cls = "mixed" if len(classes) > 1 else next(iter(classes))
        spx = sorted(i["source_px"] for i in sel)
        print(f"{city:>14} {len(sel):>4} {med_ratio:>9.1f}x {cls:>9} "
              f"{spx[len(spx)//2]:>11.0f} "
              f"{sum(1 for i in sel if i['judgeable']):>4}/{len(sel):<5}")

    s = summarize_feasibility(items)
    print(f"\n{'-'*74}")
    print(f"  total {s['total']}   judgeable (>= {JUDGEABLE_SOURCE_PX:.0f} source px) "
          f"{s['judgeable']}")
    for cls in ("parity", "advantaged"):
        c = s[cls]
        med = f"{c['median_source_px']:.0f}" if c["median_source_px"] else "—"
        print(f"    {cls:>11}: {c['n']:>3}  judgeable {c['judgeable']:>3}  "
              f"median {med} source px")
    print(f"\n  APPEARANCE VERDICTS ARE LICENSED ON {s['appearance_licensed']} OF "
          f"{s['total']} ITEMS — every judgeable one.")
    print(f"  On an 'advantaged' pano the reviewer sees detail the model never had, so")
    print(f"  'I can see the ramp' would confound appearance with resolution. The third")
    print(f"  panel removes that: it renders the same view from the pano downsampled to")
    print(f"  the model's {int(MODEL_WIDTH)} px, so the reviewer COMPARES the two budgets instead")
    print(f"  of inferring. Without that panel only the {s['parity_only']} parity items would")
    print(f"  support the verdict, which is the reason the panel exists.")
    print(f"  The {s['total'] - s['judgeable']} sub-floor items are excluded from any rate rather")
    print(f"  than labelled — under {JUDGEABLE_SOURCE_PX:.0f} source px nobody can tell occlusion from")
    print(f"  shadow from debris, and a confident label there would be worse than none.")

    if args.render:
        n, skipped = render(items, args.render, not args.include_unjudgeable,
                            args.panos_root)
        print(f"\nWrote {n} crops to {args.render} "
              f"({skipped} skipped below the pixel floor)")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"bucket": args.bucket, "field": args.field,
                       "summary": s,
                       "items": [{k: v for k, v in i.items()
                                  if k not in ("null_supra", "null_sub")}
                                 for i in items]}, fh, indent=2)
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
