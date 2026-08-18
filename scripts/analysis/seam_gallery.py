"""See the two real seam defects (#132).

Everything in #132 is a number. This renders the two findings that survived scrutiny.

**Section 1 - the ground truth double-marks the same ramp (#130).** One physical ramp
straddling the 360 seam was labelled once per edge, because the labelling viewer
(``gt_gallery.py``) clamped its crop window instead of wrapping it: the two halves landed
in separate crops most of a panorama apart, so no reviewer could see them as one object.
Both rings in these panels are the same ramp.

**Section 2 - our cached detections drop peaks beside the seam.** ``peaks_to_dets``
omitted ``exclude_border``, and skimage defaults it to True, so every peak within
``min_distance``=10 of the array edge was discarded when the committed op_caches were
built. ``stage_two/evaluate.py`` passes ``exclude_border=False``, so production was never
affected. These panels show the model responding clearly at a ramp whose detection the
cache threw away.

RETRACTED, and deliberately not rendered here: an earlier version of this gallery showed
an "original vs rolled" comparison as evidence that the model cannot see across the seam.
That result was the ``exclude_border`` bug above, not a model property - the model detects
24 of 25 seam-band ramps. Rolling moved a ramp off the array edge so the extractor stopped
discarding it, which is why the comparison looked so convincing.

Rendering is at the model's own 4096x2048 input resolution, never the native panorama
(#26 fairness rule), which also makes the heatmap overlay exact: the heatmap is 1024 wide,
so four image columns per heatmap column.

Usage:
    python scripts/analysis/seam_gallery.py --panos-root benchmark --out analysis_out/seam_gallery
"""
import argparse
import base64
import glob
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from operating_point_curve import CACHE_DIR, read_cache  # noqa: E402
from miss_decomposition import ALL_SPLITS  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from rampnet.geometry import crop_left, dist_to_seam, merge_seam_duplicates  # noqa: E402
from seam_roll_diagnostic import (  # noqa: E402
    SEAM_BAND, dedup_seam_only, find_pano_image, heatmap, peaks, wrapped_match)
from threshold_sweep import PRE  # noqa: E402
from skimage.feature import peak_local_max  # noqa: E402


def peaks_nb(h, threshold, md=10):
    """Peaks with exclude_border=False -- what stage_two/evaluate.py actually does.

    ``peaks`` (via threshold_sweep) used skimage's default until #132, dropping every
    peak within ``md`` of the array edge. Both are kept here so the gallery can show the
    difference rather than assert it.
    """
    pk = peak_local_max(np.clip(h, 0, 1), min_distance=md, threshold_abs=threshold,
                        exclude_border=False)
    H, W = h.shape
    return [(float(c / W), float(r / H), float(h[r][c])) for r, c in pk]

Image.MAX_IMAGE_PIXELS = None

MODEL_W, MODEL_H = 4096, 2048          # the model's input; the heatmap is a 1/4 of this
HEAT_W, HEAT_H = 1024, 512
FOV_DEG = 60                           # crop width in degrees of panorama longitude
RSQ = radius_sq_for()
R = RSQ ** 0.5


def cut_wrapped(img, center_x_px, side, height_px=None):
    """A ``side``-wide crop centred on ``center_x_px``, joined across the seam.

    The whole point: a clamping crop (what ``gt_gallery.py`` does) shows a seam ramp as
    two objects at opposite ends of the frame. This shows it as one object.
    """
    w, h = img.size
    height_px = height_px or side
    left = crop_left(center_x_px, w, side)
    top = max(0, min(int(round(h / 2 - height_px / 2)), h - height_px))
    if left + side <= w:
        return img.crop((left, top, left + side, top + height_px)), left, top
    out = Image.new(img.mode, (side, height_px))
    first = w - left
    out.paste(img.crop((left, top, w, top + height_px)), (0, 0))
    out.paste(img.crop((0, top, side - first, top + height_px)), (first, 0))
    return out, left, top


def heat_overlay(crop, heat, left, top, side, height_px):
    """Blend the heatmap over the crop, cut with the same wrapping window."""
    hm = np.clip(heat, 0, 1)
    big = Image.fromarray((hm * 255).astype(np.uint8), mode="L").resize(
        (MODEL_W, MODEL_H), Image.BILINEAR)
    win, _, _ = cut_wrapped(big, left + side / 2, side, height_px)
    a = np.asarray(win).astype(np.float32) / 255.0
    base = np.asarray(crop.convert("RGB")).astype(np.float32)
    # red where the model responds, alpha rising with activation so a weak response
    # stays visibly weak rather than being flattened by a hard colormap
    tint = np.zeros_like(base)
    tint[..., 0] = 255.0
    alpha = (a ** 0.7)[..., None] * 0.85
    return Image.fromarray((base * (1 - alpha) + tint * alpha).astype(np.uint8))


def crop_local_xy(x_norm, y_norm, left, top, side, pano_w, pano_h, scale):
    """Where a pano-normalized point lands in a rendered crop, in rendered pixels.

    Pure so it can be tested: the marker offset was wrong in the first version of this
    gallery (a hardcoded 0.25 stood in for the crop's real ``top``) and the only symptom
    was rings sitting in the roadway below the ramp, which no assertion would have caught.

    x wraps -- the crop may straddle the seam, which is the whole point of this page.
    y does not: the window is clamped vertically by :func:`cut_wrapped`.
    """
    dx = ((x_norm * pano_w) - left) % pano_w
    return dx * scale, (y_norm * pano_h - top) * scale


def annotate(img, marks, left, top, side, pano_w):
    """Draw the seam line, GT rings at the match radius, and detections.

    ``top`` is the crop's own first row and MUST come from the same cut that produced
    ``img``. An earlier version assumed the crop started a fixed quarter of the way down
    the panorama; the real top is ``MODEL_H/2 - side/2``, so every marker landed ~190
    rendered pixels low -- far enough to sit in the roadway instead of on the ramp.
    """
    d = ImageDraw.Draw(img)
    scale = img.size[0] / side                      # crop px -> rendered px

    def to_local(x_norm):
        return crop_local_xy(x_norm, 0.0, left, top, side, pano_w, MODEL_H, scale)[0]

    # the seam itself — where the panorama's own edge falls inside this crop
    sx = to_local(0.0)
    if 0 <= sx <= img.size[0]:
        for yy in range(0, img.size[1], 24):
            d.line([(sx, yy), (sx, yy + 12)], fill=(0, 229, 255), width=3)
        d.text((sx + 6, 6), "360 seam", fill=(0, 229, 255))

    for kind, x_norm, y_norm in marks:
        cx, cy = crop_local_xy(x_norm, y_norm, left, top, side, pano_w, MODEL_H, scale)
        if kind == "gt":
            r = R / PANO_SCALE_X * pano_w * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 212, 0), width=4)
        elif kind == "drop":
            s = 16
            d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=(255, 61, 61), width=6)
            d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=(255, 61, 61), width=6)
        elif kind == "det":
            s = 14
            d.line([(cx - s, cy), (cx + s, cy)], fill=(26, 156, 62), width=5)
            d.line([(cx, cy - s), (cx, cy + s)], fill=(26, 156, 62), width=5)
    return img


def b64(img, max_w=760):
    if img.size[0] > max_w:
        img = img.resize((max_w, int(img.size[1] * max_w / img.size[0])), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def roll_image(img):
    w, h = img.size
    out = Image.new(img.mode, img.size)
    half = w // 2
    out.paste(img.crop((half, 0, w, h)), (0, 0))
    out.paste(img.crop((0, 0, half, h)), (w - half, 0))
    return out


def collect():
    """Seam-band GT, plus the #130 duplicate pairs, keyed by pano."""
    items, dupes = [], []
    for city in dict.fromkeys(ALL_SPLITS):
        path = os.path.join(CACHE_DIR, f"{city}.json")
        if not os.path.exists(path):
            continue
        for rec in read_cache(path)[0]:
            raw = list(rec["gt"].gt_points)
            gt = dedup_seam_only(raw)
            if len(gt) != len(raw):
                dupes.append((city, rec["pano"], raw))
            for g in gt:
                if dist_to_seam(g[0], PANO_SCALE_X) < SEAM_BAND:
                    items.append((city, rec["pano"], g, gt))
    return items, dupes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panos-root", default=os.path.join(REPO, "benchmark"))
    ap.add_argument("--out", default=os.path.join(REPO, "analysis_out", "seam_gallery"))
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items, dupes = collect()
    if args.limit:
        items = items[:args.limit]
        dupes = dupes[:args.limit]
    print(f"seam-band GT: {len(items)}   #130 duplicate panos: {len(dupes)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from threshold_sweep import load_model
    model = load_model().to(device)

    side = int(FOV_DEG / 360.0 * MODEL_W)
    height = side
    cards, cache = [], {}
    for i, (city, pid, g, gt) in enumerate(items, 1):
        path = find_pano_image(args.panos_root, city, pid)
        if path is None:
            continue
        if pid not in cache:
            img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
            h = heatmap(model, device, img)
            # The two extractors, differing ONLY in exclude_border. Everything else --
            # model, weights, heatmap, min_distance, threshold -- is identical.
            kept = peaks_nb(h, args.threshold)
            cached = peaks(h, args.threshold)
            cache[pid] = (img, h, kept, cached)
        img, h, kept, cached = cache[pid]

        cx = g[0] * MODEL_W
        crop, left, top = cut_wrapped(img, cx, side, height)
        in_cache = bool(wrapped_match([(d[0], d[1]) for d in cached], [g]))
        in_prod = bool(wrapped_match([(d[0], d[1]) for d in kept], [g]))
        dropped = [d for d in kept if d not in cached]
        marks = ([("gt", g[0], g[1])] + [("det", d[0], d[1]) for d in cached]
                 + [("drop", d[0], d[1]) for d in dropped])
        cy = int(g[1] * HEAT_H)
        cols = [(int(g[0] * HEAT_W) + k) % HEAT_W for k in range(-10, 11)]
        peak_here = float(np.max(h[max(0, cy - 10):cy + 11][:, cols]))
        cards.append({
            "city": city, "pano": pid,
            "seam_px": round(dist_to_seam(g[0], PANO_SCALE_X), 1),
            "x": round(g[0], 5), "heat": round(peak_here, 3),
            "in_cache": in_cache, "in_prod": in_prod,
            "img": b64(annotate(crop.copy(), marks, left, top, side, MODEL_W)),
            "heat_img": b64(annotate(heat_overlay(crop, h, left, top, side, height),
                                     [("gt", g[0], g[1])], left, top, side, MODEL_W)),
        })
        print(f"  {i}/{len(items)} {city}:{pid} seam={cards[-1]['seam_px']}px "
              f"heat={peak_here:.2f} cache={'HIT' if in_cache else 'DROPPED'} "
              f"prod={'HIT' if in_prod else 'miss'}", flush=True)

    dupe_cards = []
    for city, pid, raw in dupes:
        path = find_pano_image(args.panos_root, city, pid)
        if path is None:
            continue
        img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
        near = [p for p in raw if dist_to_seam(p[0], PANO_SCALE_X) < R * 1.5]
        if not near:
            continue
        crop, left, top = cut_wrapped(img, near[0][0] * MODEL_W, side, height)
        dupe_cards.append({
            "city": city, "pano": pid,
            "marks": [[round(p[0], 5), round(p[1], 5)] for p in near],
            "img": b64(annotate(crop.copy(), [("gt", p[0], p[1]) for p in near],
                                left, top, side, MODEL_W)),
        })

    os.makedirs(args.out, exist_ok=True)
    html = render(cards, dupe_cards, args.threshold)
    out = os.path.join(args.out, "index.html")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(html)
    with open(os.path.join(args.out, "manifest.json"), "w", newline="") as f:
        json.dump([{k: v for k, v in c.items() if not k.startswith(("img", "heat"))}
                   for c in cards], f, indent=2)
    print(f"\nwrote {out}")


def render(cards, dupe_cards, threshold):
    dropped = sum(1 for c in cards if c["in_prod"] and not c["in_cache"])
    seen = sum(1 for c in cards if c["heat"] >= threshold)
    head = f"""<!-- generated by scripts/analysis/seam_gallery.py (#132) -->
<meta charset="utf-8"><title>The 360 seam, seen</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#14161a;color:#e8e8ea}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px 20px 80px}}
 h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:20px;margin:40px 0 8px}}
 .sub{{color:#9aa0a8;margin:0 0 20px}}
 .key{{display:flex;gap:20px;flex-wrap:wrap;background:#1c1f25;border:1px solid #2c313a;
   border-radius:10px;padding:12px 16px;margin:0 0 24px}}
 .key span{{display:inline-flex;align-items:center;gap:7px;font-size:14px}}
 .sw{{width:15px;height:15px;border-radius:3px;display:inline-block}}
 .card{{background:#1c1f25;border:1px solid #2c313a;border-radius:12px;padding:16px;margin:0 0 20px}}
 .card h3{{margin:0 0 4px;font-size:15px;font-weight:600}}
 .meta{{color:#9aa0a8;font-size:13px;margin:0 0 12px}}
 .panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}
 figure{{margin:0}} figcaption{{font-size:13px;color:#9aa0a8;margin-top:6px}}
 img{{width:100%;border-radius:8px;display:block}}
 .tag{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:700}}
 .bad{{background:#5b1a1a;color:#ffb4b4}} .ok{{background:#14401f;color:#9ff0b6}}
 .note{{background:#241d10;border:1px solid #5c4415;border-radius:10px;padding:14px 16px;margin:0 0 24px}}
</style>
<div class="wrap">
<h1>The 360&deg; seam, seen</h1>
<p class="sub">Issue #132. Every panel is at the model's own input resolution
(4096&times;2048), and each crop is cut with a <b>wrapping</b> window, so a ramp sitting on
the panorama's edge appears whole instead of split in two.</p>
<div class="note"><b>What this does not show.</b> An earlier version of this page compared
an ordinary pass against a 180&deg;-rolled pass, as evidence that the model cannot see
across the seam. <b>That result was retracted</b> &mdash; it was our own extractor bug
(section 2), not a model property. The model responds at <b>{seen} of {len(cards)}</b> of
the ramps below. Those panels are deliberately gone rather than quietly corrected.</div>
<div class="key">
  <span><i class="sw" style="background:#ffd400"></i> ground truth, ringed at the match radius</span>
  <span><i class="sw" style="background:#1a9c3e"></i> detection kept by the cache</span>
  <span><i class="sw" style="background:#ff3d3d"></i> detection the cache dropped</span>
  <span><i class="sw" style="background:#00e5ff"></i> the 360&deg; seam</span>
  <span><i class="sw" style="background:#ff2d2d"></i> heatmap response</span>
</div>
"""
    body = []
    if dupe_cards:
        body.append("""<h2>1. The ground truth double-marks one ramp &mdash; issue #130</h2>
<p class="sub">One physical ramp straddling the seam, labelled once per edge. The
labelling viewer clamped its crop window instead of wrapping it, so the two halves landed
in separate crops most of a panorama apart and no reviewer could see them as one object.
Both rings below are the same ramp. Note this is <b>not</b> settled for every pair: about
three of the eleven may be genuinely adjacent far-field ramps, which is why they need
human adjudication rather than an automatic merge.</p>""")
        for c in dupe_cards:
            xs = ", ".join(f"x={m[0]}" for m in c["marks"])
            body.append(f"""<div class="card">
 <h3>{c['city']} &middot; {c['pano']}</h3>
 <p class="meta">{len(c['marks'])} marks: {xs}</p>
 <div class="panels"><figure><img src="{c['img']}">
 <figcaption>two ground-truth marks, one ramp</figcaption></figure></div></div>""")

    body.append(f"""<h2>2. Our cached detections drop peaks beside the seam</h2>
<p class="sub"><code>peaks_to_dets</code> omitted <code>exclude_border</code>, and skimage
defaults it to <b>True</b>, discarding every peak within <code>min_distance</code>=10 of
the array edge &mdash; a 3.5&deg; strip either side of the seam. That is how every
committed <code>op_cache</code> was built. <code>stage_two/evaluate.py</code> passes
<code>exclude_border=False</code>, so <b>production was never affected</b>. Below, the
model's heatmap is plainly lit at the ramp and the detection still went missing from the
cache: <b>{dropped} of {len(cards)}</b> shown here.</p>""")
    for c in cards:
        a = ('<span class="tag ok">kept</span>' if c["in_cache"]
             else '<span class="tag bad">dropped</span>')
        b = ('<span class="tag ok">found</span>' if c["in_prod"]
             else '<span class="tag bad">missed</span>')
        body.append(f"""<div class="card">
 <h3>{c['city']} &middot; {c['pano']}</h3>
 <p class="meta">{c['seam_px']} px from the seam &middot; x={c['x']} &middot;
   peak heatmap response at the ramp <b>{c['heat']}</b> &middot;
   cache {a} &nbsp; production setting {b}</p>
 <div class="panels">
  <figure><img src="{c['img']}"><figcaption>image &mdash; ramp, seam, and both extractors' detections</figcaption></figure>
  <figure><img src="{c['heat_img']}"><figcaption>the model's heatmap &mdash; it sees the ramp</figcaption></figure>
 </div></div>""")
    return head + "\n".join(body) + "\n</div>\n"


if __name__ == "__main__":
    main()
