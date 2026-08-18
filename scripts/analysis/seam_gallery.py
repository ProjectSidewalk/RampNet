"""See the seam defect: the same ramp, the same pixels, found only when rolled (#132).

Everything in #132 is a number. This renders the mechanism.

For each ground-truth ramp sitting within ~4 deg of the 360 seam, three panels of the
*same world region*, cut with a wrapping window so the ramp appears whole rather than
split across the panorama's two edges:

  1. **image**   the crop, with the seam drawn as a line, the ramp ringed at the match
                 radius the scorer actually uses, and every detection the model emitted
  2. **original** the model's heatmap over that same crop, from an ordinary forward pass
  3. **rolled**   the model's heatmap over that same crop, from a pass on a panorama
                 rolled by half its width — so the ramp sat at the centre of the frame
                 instead of on its border, with every other pixel unchanged

The comparison is controlled by construction: panels 2 and 3 differ only in where the
image border fell. Same model, same weights, same ramp, same pixels.

A second section renders the #130 ground-truth duplicates — one physical ramp carrying
two marks, one per edge — which is the annotation half of the same defect.

Rendering is at the model's own 4096x2048 input resolution, never the native panorama
(#26 fairness rule), which also makes the heatmap overlay exact: the heatmap is 1024 wide,
so four image columns per heatmap column, with no resampling guesswork.

Usage:
    python scripts/analysis/seam_gallery.py --panos-root benchmark --out analysis_out/seam_gallery
    python scripts/analysis/seam_gallery.py --limit 4        # smoke test
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


def annotate(img, marks, left, side, pano_w):
    """Draw the seam line, GT rings at the match radius, and detections."""
    d = ImageDraw.Draw(img)
    scale = img.size[0] / side                      # crop px -> rendered px

    def to_local(x_norm):
        px = x_norm * pano_w
        dx = (px - left) % pano_w
        return dx * scale

    # the seam itself — where the panorama's own edge falls inside this crop
    sx = to_local(0.0)
    if 0 <= sx <= img.size[0]:
        for yy in range(0, img.size[1], 24):
            d.line([(sx, yy), (sx, yy + 12)], fill=(0, 229, 255), width=3)
        d.text((sx + 6, 6), "360 seam", fill=(0, 229, 255))

    for kind, x_norm, y_norm in marks:
        cx, cy = to_local(x_norm), (y_norm - 0.25) * MODEL_H * scale
        if kind == "gt":
            r = R / PANO_SCALE_X * pano_w * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 212, 0), width=4)
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
            h_o = heatmap(model, device, img)
            # Un-roll the HEATMAP back into original coordinates so the overlay lines up
            # with the crop. Peaks are then already in original coordinates -- do NOT also
            # shift them, which would be a second correction of the same half-width and
            # put every rolled detection half a panorama away. (seam_roll_diagnostic.py
            # shifts the peaks instead, because it never needs the heatmap itself.)
            h_r = np.roll(heatmap(model, device, roll_image(img)), HEAT_W // 2, axis=1)
            det_o = peaks(h_o, args.threshold)
            det_r = peaks(h_r, args.threshold)
            cache[pid] = (img, h_o, h_r, det_o, det_r)
        img, h_o, h_r, det_o, det_r = cache[pid]

        cx = g[0] * MODEL_W
        crop, left, top = cut_wrapped(img, cx, side, height)
        marks = [("gt", g[0], g[1])] + [("det", d[0], d[1]) for d in det_o]
        found_o = bool(wrapped_match([(d[0], d[1]) for d in det_o], [g]))
        found_r = bool(wrapped_match([(d[0], d[1]) for d in det_r], [g]))
        cards.append({
            "city": city, "pano": pid,
            "seam_px": round(dist_to_seam(g[0], PANO_SCALE_X), 1),
            "x": round(g[0], 5), "found_original": found_o, "found_rolled": found_r,
            "img": b64(annotate(crop.copy(), marks, left, side, MODEL_W)),
            "heat_o": b64(annotate(heat_overlay(crop, h_o, left, top, side, height),
                                   [("gt", g[0], g[1])], left, side, MODEL_W)),
            "heat_r": b64(annotate(heat_overlay(crop, h_r, left, top, side, height),
                                   [("gt", g[0], g[1])], left, side, MODEL_W)),
        })
        print(f"  {i}/{len(items)} {city}:{pid} seam={cards[-1]['seam_px']}px "
              f"orig={'HIT' if found_o else 'MISS'} rolled={'HIT' if found_r else 'MISS'}",
              flush=True)

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
                                left, side, MODEL_W)),
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
    rec = sum(1 for c in cards if not c["found_original"] and c["found_rolled"])
    head = f"""<!-- generated by scripts/analysis/seam_gallery.py (#132) -->
<meta charset="utf-8"><title>The 360 seam, seen</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#14161a;color:#e8e8ea}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px 20px 80px}}
 h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:38px 0 10px}}
 .sub{{color:#9aa0a8;margin:0 0 22px}}
 .key{{display:flex;gap:20px;flex-wrap:wrap;background:#1c1f25;border:1px solid #2c313a;
   border-radius:10px;padding:12px 16px;margin:0 0 26px}}
 .key span{{display:inline-flex;align-items:center;gap:7px}}
 .sw{{width:15px;height:15px;border-radius:3px;display:inline-block}}
 .card{{background:#1c1f25;border:1px solid #2c313a;border-radius:12px;padding:16px;margin:0 0 22px}}
 .card h3{{margin:0 0 4px;font-size:15px;font-weight:600}}
 .meta{{color:#9aa0a8;font-size:13px;margin:0 0 12px}}
 .panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
 figure{{margin:0}} figcaption{{font-size:13px;color:#9aa0a8;margin-top:6px}}
 img{{width:100%;border-radius:8px;display:block}}
 .tag{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:700}}
 .miss{{background:#5b1a1a;color:#ffb4b4}} .hit{{background:#14401f;color:#9ff0b6}}
 code{{background:#2a2f38;padding:1px 5px;border-radius:4px;font-size:13px}}
</style>
<div class="wrap">
<h1>The 360&deg; seam, seen</h1>
<p class="sub">Issue #132. Every panel below is the model's own input resolution
(4096&times;2048) and the crop is cut with a <b>wrapping</b> window, so a ramp sitting on
the panorama's edge appears whole instead of split in two. Detections at threshold
{threshold}.</p>
<div class="key">
  <span><i class="sw" style="background:#ffd400"></i> ground truth, ringed at the match radius</span>
  <span><i class="sw" style="background:#1a9c3e"></i> model detection</span>
  <span><i class="sw" style="background:#00e5ff"></i> the 360&deg; seam</span>
  <span><i class="sw" style="background:#ff2d2d"></i> heatmap response</span>
</div>
<h2>1. Ramps on the seam &mdash; the same pixels, twice</h2>
<p class="sub">The two heatmap panels differ in <b>one thing only</b>: whether the
panorama's border fell on the ramp. Same model, same weights, same ramp. Where the
&ldquo;rolled&rdquo; panel lights up and the &ldquo;original&rdquo; does not, the model
could see the ramp perfectly well &mdash; it just could not see <i>across the border</i>.
<b>{rec} of {len(cards)}</b> shown here are recovered by the roll.</p>
"""
    body = []
    for c in cards:
        o = ('<span class="tag hit">found</span>' if c["found_original"]
             else '<span class="tag miss">missed</span>')
        r = ('<span class="tag hit">found</span>' if c["found_rolled"]
             else '<span class="tag miss">missed</span>')
        body.append(f"""<div class="card">
 <h3>{c['city']} &middot; {c['pano']}</h3>
 <p class="meta">{c['seam_px']} px from the seam &middot; x={c['x']} &middot;
   original {o} &nbsp; rolled {r}</p>
 <div class="panels">
  <figure><img src="{c['img']}"><figcaption>image &mdash; ramp, seam, and what the model emitted</figcaption></figure>
  <figure><img src="{c['heat_o']}"><figcaption><b>original</b> &mdash; ramp on the border</figcaption></figure>
  <figure><img src="{c['heat_r']}"><figcaption><b>rolled 180&deg;</b> &mdash; same ramp, mid-frame</figcaption></figure>
 </div></div>""")

    if dupe_cards:
        body.append("""<h2>2. The ground truth double-marks the same ramp</h2>
<p class="sub">Issue #130. One physical ramp straddling the seam was labelled once per
edge, because the labelling viewer clamped its crop window instead of wrapping it &mdash;
the two halves landed in separate crops most of a panorama apart, so no reviewer could
see them as one object. Both rings below are the same ramp.</p>""")
        for c in dupe_cards:
            xs = ", ".join(f"x={m[0]}" for m in c["marks"])
            body.append(f"""<div class="card">
 <h3>{c['city']} &middot; {c['pano']}</h3>
 <p class="meta">{len(c['marks'])} marks: {xs}</p>
 <div class="panels"><figure><img src="{c['img']}">
 <figcaption>two ground-truth marks, one ramp</figcaption></figure></div></div>""")
    return head + "\n".join(body) + "\n</div>\n"


if __name__ == "__main__":
    main()
