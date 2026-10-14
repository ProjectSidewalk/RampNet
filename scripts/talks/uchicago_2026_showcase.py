"""A short video of RampNet heatmaps on benchmark panoramas where it was near perfect.

Each frame: the raw panorama on top, and below it the same panorama dimmed with the model's
heatmap and a ring on every peak at the deployed threshold, with the city, imagery source and
the reviewer's verdict in a caption. The selection is mechanical, from the committed verdicts:
panoramas where every detection was confirmed and the reviewer found nothing missed, plus a few
with no ramps and no detections (true negatives).

    python scripts/talks/uchicago_2026_showcase.py select
    python scripts/talks/uchicago_2026_showcase.py infer  --pano-root benchmark      # GPU
    python scripts/talks/uchicago_2026_showcase.py render --pano-root benchmark
    python scripts/talks/uchicago_2026_showcase.py video                              # ffmpeg

City-grouped variants, with and without a locator map in the caption bar. The map marks every
benchmark city and highlights the one on screen; it is drawn from a trimmed Natural Earth file
that ``basemap`` writes once (network) and that is committed beside the manifest:

    python scripts/talks/uchicago_2026_showcase.py basemap
    python scripts/talks/uchicago_2026_showcase.py render --pano-root benchmark --order city \
        --frames-dir <out>/frames_by_city_map --map
    python scripts/talks/uchicago_2026_showcase.py video --frames-dir <out>/frames_by_city_map \
        --out <out>/showcase_by_city_map.mp4

``--panels-from <frames dir>`` replaces ``--pano-root``: it lifts the two image panels from
frames already rendered in manifest order (the default ``render``), so a re-ordered or
re-captioned cut needs neither the panoramas nor a GPU. The heatmap PNGs still supply the
detection count in the caption.

Committed: the manifest (which panoramas, why), one 8-bit heatmap PNG per panorama and the
basemap, so the frames and the mp4 regenerate on CPU without the checkpoint. Frames and the mp4
are not committed (``.gitignore`` in the showcase folder); ``video`` rebuilds them in a minute.
"""
import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHOW_DIR = os.path.join(REPO, "docs", "talks", "uchicago_2026", "showcase")
MANIFEST = os.path.join(SHOW_DIR, "manifest.json")
FRAMES_DIR = os.path.join(SHOW_DIR, "frames")
BASEMAP = os.path.join(SHOW_DIR, "basemap_ne110m.json")
HF_MODEL = "projectsidewalk/rampnet-model"
DEPLOYED_T = 0.55

# Imagery source and place per split, from benchmark/README.md.
SPLITS = {
    "bend":               ("Bend, OR, US",            "Google Street View"),
    "paterson":           ("Paterson, NJ, US",        "Google Street View"),
    "gainesville":        ("Gainesville, FL, US",     "Google Street View"),
    "sao_paulo":          ("São Paulo, Brazil",       "Google Street View"),
    "laurens_gsv":        ("Laurens, IA, US",         "Google Street View"),
    "richmond":           ("Richmond, VA, US",        "Mapillary 360° (iSTAR Pulsar / GoPro Max)"),
    "annapolis":          ("Annapolis, MD, US",       "Mapillary 360° (Trimble MX7)"),
    "morgantown":         ("Morgantown, WV, US",      "Mapillary 360° (GoPro Max)"),
    "clovis":             ("Clovis, NM, US",          "Mapillary 360° (GoPro Fusion)"),
    "laurens_mapillary":  ("Laurens, IA, US",         "Mapillary 360° (GoPro Max)"),
    "budapest_district5": ("Budapest, Hungary",       "Mapillary 360° (GoPro Max)"),
    # No showcase panoramas (it is not in the selection), but it is a benchmark city, so the
    # locator map marks it.
    "bayonne":            ("Bayonne, France",         "Panoramax (GoPro Max)"),
}
# Splits ``select`` draws from. Bayonne was added to the benchmark after the 2026-10-05
# selection; leaving it out keeps a re-run of ``select`` identical to the committed manifest.
SHOWCASE_SPLITS = [s for s in SPLITS if s != "bayonne"]
# ``--order city``: a west-to-east sweep, so the highlight on the map travels rather than jumps.
# Within a city the manifest order is kept, with its true negative (if any) last.
CITY_ORDER = ["bend", "clovis", "laurens_gsv", "laurens_mapillary", "gainesville", "morgantown",
              "richmond", "annapolis", "paterson", "sao_paulo", "budapest_district5", "bayonne"]
PER_SPLIT = 3          # top-2 by ramp count plus one seeded random, for variety of scenes
NEGATIVES = 4          # true-negative panoramas, spread across splits
SEED = 2026


def load_verdicts(split):
    with open(os.path.join(REPO, "benchmark", split, "verdicts.json"), encoding="utf-8") as f:
        return json.load(f)["panos"]


def load_records(split):
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        return {json.loads(l)["pano"]["panorama_id"]: json.loads(l) for l in f}


def cmd_select_append(split, n_pick):
    """Add one split's picks to the existing manifest, leaving every existing item as it is.

    Used for bayonne, whose verdicts landed after the 2026-10-05 selection, and which has only
    one panorama meeting ``select``'s rule (every detection confirmed, nothing missed, >= 3
    ramps). So the rule here is wider, to give a choice of exemplars: the ``n_pick`` panoramas
    with no wrong or unsure detection and >= 2 confirmed ramps, fewest confident misses first,
    then most ramps. One with no misses is 'all_found'; one with a confident miss is
    'near_perfect', and its caption says so ("8 curb ramps, 7 detected"). Plus one true
    negative from a per-split RNG, so appending does not reshuffle the cities already chosen.
    """
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if any(it["split"] == split for it in manifest["items"]):
        sys.exit(f"{split} is already in the manifest")
    rng = random.Random(f"{SEED}:{split}")
    v = load_verdicts(split)
    cands = []
    for pid, vv in v.items():
        found = sum(1 for d in vv["dets"] if d is True)
        missed = sum(1 for m in vv.get("missed", []) if not m.get("unsure"))
        if found >= 2 and all(d is True for d in vv["dets"]):
            cands.append((missed, -found, pid, found))
    added = []
    for missed, _, pid, found in sorted(cands)[:n_pick]:
        n = found + missed
        added.append(dict(split=split, pano_id=pid, n_ramps=n,
                          kind="all_found" if missed == 0 else "near_perfect",
                          why=(f"{n} ramps, every detection confirmed" +
                               (", nothing missed" if missed == 0 else f", {missed} missed"))))
    tn = sorted(pid for pid, vv in v.items() if not vv["dets"] and not vv.get("missed"))
    if tn:
        added.append(dict(split=split, pano_id=rng.choice(tn), n_ramps=0, kind="true_negative",
                          why="no curb ramps, and the model fired nowhere"))
    for it in added:
        it["appended"] = datetime.date.today().isoformat()
    manifest["items"] += added
    manifest.setdefault("appended", []).append(dict(
        split=split, date=datetime.date.today().isoformat(),
        rule=(f"select --append {split} --n-pick {n_pick}: no wrong/unsure detection, >= 2 "
              f"confirmed, fewest confident misses then most ramps; true negative RNG seeded "
              f"'{SEED}:{split}'")))
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for it in added:
        print("appended", it["split"], it["pano_id"], it["kind"], it["n_ramps"])


def cmd_select(args):
    if args.append:
        return cmd_select_append(args.append, args.n_pick)
    rng = random.Random(SEED)
    items, negatives = [], []
    for split in SHOWCASE_SPLITS:
        v = load_verdicts(split)
        perfect = sorted(((sum(1 for d in vv["dets"] if d is True), pid) for pid, vv in v.items()
                          if vv["dets"] and all(d is True for d in vv["dets"])
                          and not vv.get("missed")), reverse=True)
        perfect = [(n, pid) for n, pid in perfect if n >= 3]
        pick = perfect[:2]
        rest = perfect[2:]
        if rest:
            pick.append(rng.choice(rest))
        for n, pid in pick[:PER_SPLIT]:
            items.append(dict(split=split, pano_id=pid, n_ramps=n, kind="all_found",
                              why=f"{n} ramps, every detection confirmed, nothing missed"))
        tn = sorted(pid for pid, vv in v.items() if not vv["dets"] and not vv.get("missed"))
        if tn:
            negatives.append(dict(split=split, pano_id=rng.choice(tn), n_ramps=0,
                                  kind="true_negative",
                                  why="no curb ramps, and the model fired nowhere"))
    # Order: shuffled so the video does not run city by city; the true negatives come last.
    rng.shuffle(items)
    rng.shuffle(negatives)
    items += negatives[:NEGATIVES]
    os.makedirs(SHOW_DIR, exist_ok=True)
    manifest = dict(selected=datetime.date.today().isoformat(), rule=(
        f"per split: the two 'all_found' panoramas with the most confirmed ramps (>= 3) plus one "
        f"seeded-random other; then {NEGATIVES} true negatives (no ramps, no detections), "
        f"seed {SEED}; from benchmark/<split>/verdicts.json"), items=items)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(SHOW_DIR, ".gitignore"), "w", newline="\n") as f:
        f.write("frames*/\n*.mp4\n")
    print(f"{len(items)} panoramas -> {os.path.relpath(MANIFEST, REPO)}")


def _heat_path(it):
    return os.path.join(SHOW_DIR, f"heat_{it['split']}_{it['pano_id']}.png")


def _pano_path(root, it):
    return os.path.join(root, it["split"], "panos", f"{it['pano_id']}.jpg")


def cmd_infer(args):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    ckpt = f"{args.checkpoint} @ {getattr(model.config, '_commit_hash', 'unknown')}"
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    Image.MAX_IMAGE_PIXELS = None
    run = dict(checkpoint=ckpt, device=device.type, autocast_fp16=device.type == "cuda",
               input_size=[2048, 4096], tta=False, run_date=datetime.date.today().isoformat())
    for it in manifest["items"]:
        out = _heat_path(it)
        if os.path.exists(out) and not args.force:
            continue
        it["inference"] = run           # per item: a later append runs on a different day
        img = Image.open(_pano_path(args.pano_root, it))
        img.draft("RGB", (4096, 2048))
        img = img.convert("RGB").resize((4096, 2048), Image.LANCZOS)
        x = norm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    y = model(x)
            else:
                y = model(x)
        heat = np.clip(y.float().squeeze().cpu().numpy(), 0, 1)
        Image.fromarray((heat * 255).round().astype(np.uint8)).save(out)
        it["max"] = round(float(heat.max()), 3)
        print(it["split"], it["pano_id"], "max", it["max"], flush=True)
    manifest.setdefault("inference", run)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


# Frame geometry: 1920x1080. Two panels of the panorama's street band (the sky above and the
# vehicle roof below carry no ramps), each 1920 x 442, with a 196 px caption block.
FRAME_W, FRAME_H = 1920, 1080
BAND = (0.30, 0.76)       # fraction of the equirect height kept
PANEL_H = int(FRAME_W / 2 * (BAND[1] - BAND[0]))
CAPTION_H = FRAME_H - 2 * PANEL_H


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _peaks(heat_path):
    """Peaks of the committed heatmap at the deployed threshold, as (row, col) on 512 x 1024."""
    import numpy as np
    from PIL import Image
    from skimage.feature import peak_local_max

    heat = np.asarray(Image.open(heat_path), dtype=np.float32) / 255.0      # 512 x 1024
    return heat, peak_local_max(heat, min_distance=10, threshold_abs=DEPLOYED_T,
                                exclude_border=False)


def render_panels(pano_path, heat_path):
    """The two image panels (raw band on top, heatmap band below), FRAME_W x 2*PANEL_H."""
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import cm

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(pano_path)
    img.draft("RGB", (4096, 2048))
    img = img.convert("RGB").resize((4096, 2048), Image.BILINEAR)
    y0, y1 = int(BAND[0] * 2048), int(BAND[1] * 2048)
    band = img.crop((0, y0, 4096, y1)).resize((FRAME_W, PANEL_H), Image.LANCZOS)

    heat, peaks = _peaks(heat_path)

    # Bottom panel: the panorama dimmed and cooled, the heatmap in jet, alpha from its value.
    base = np.asarray(band, dtype=np.float32)
    tint = base * 0.42 + np.array([18, 22, 90], dtype=np.float32) * 0.58
    hb = heat[int(BAND[0] * 512):int(BAND[1] * 512)]
    hb_img = Image.fromarray((hb * 255).astype(np.uint8)).resize((FRAME_W, PANEL_H), Image.BICUBIC)
    hb = np.asarray(hb_img, dtype=np.float32) / 255.0
    rgba = cm.jet(hb)[..., :3] * 255
    alpha = np.clip(hb * 1.6, 0, 0.92)[..., None]
    comp = tint * (1 - alpha) + rgba * alpha
    bottom = Image.fromarray(comp.round().astype(np.uint8))
    draw = ImageDraw.Draw(bottom)
    for r, c in peaks:
        x = (c + 0.5) / 1024 * FRAME_W
        y = ((r + 0.5) / 512 - BAND[0]) / (BAND[1] - BAND[0]) * PANEL_H
        if 0 <= y <= PANEL_H:
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline=(90, 240, 255), width=5)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(90, 240, 255))

    panels = Image.new("RGB", (FRAME_W, 2 * PANEL_H))
    panels.paste(band, (0, 0))
    panels.paste(bottom, (0, PANEL_H))
    return panels


# ---- Locator map ---------------------------------------------------------------------------
# Two insets in the caption bar: the lower 48 (eight of the eleven places are there, four of
# them within a few degrees of each other) and a world view for the rest, with a thin box on
# the world view marking the US inset. Plain equirectangular; the US inset squeezes longitude
# by cos(37 deg) so the country is not stretched.
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
          "{}.geojson")
US_BOX = (-125.0, -66.5, 24.0, 49.6)       # lon0, lon1, lat0, lat1
WORLD_BOX = (-130.0, 32.0, -40.0, 62.0)
MAP_H = 160
MAP_PAD = 19                               # (CAPTION_H - MAP_H) / 2, rounded
MAP_BG = (12, 12, 14)
MAP_LAND = (44, 46, 52)
MAP_EDGE = (70, 72, 80)
DOT_IDLE = (165, 165, 158)
DOT_ON = (90, 240, 255)                    # same cyan as the detection rings


def _ring_list(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def _clip_rings(rings, box, margin=8.0):
    lon0, lon1, lat0, lat1 = box
    out = []
    for ring in rings:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        if max(lons) < lon0 - margin or min(lons) > lon1 + margin:
            continue
        if max(lats) < lat0 - margin or min(lats) > lat1 + margin:
            continue
        out.append([[round(x, 2), round(y, 2)] for x, y in ring])
    return out


def cmd_basemap(args):
    """Fetch Natural Earth 1:110m countries + US states and keep only rings near the insets."""
    def fetch(name):
        with urllib.request.urlopen(NE_URL.format(name), timeout=60) as r:
            return json.load(r)

    countries = fetch("ne_110m_admin_0_countries")
    states = fetch("ne_110m_admin_1_states_provinces_lakes")
    world = [r for f in countries["features"] for r in _ring_list(f["geometry"])]
    us = [r for f in states["features"] if f["properties"].get("adm0_a3") == "USA"
          for r in _ring_list(f["geometry"])]
    out = dict(source="Natural Earth 1:110m (public domain), admin_0_countries and "
                      "admin_1_states_provinces_lakes, via github.com/nvkelso/natural-earth-vector",
               fetched=datetime.date.today().isoformat(),
               world=_clip_rings(world, WORLD_BOX), us_states=_clip_rings(us, US_BOX),
               us_neighbours=_clip_rings(world, US_BOX))
    with open(BASEMAP, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    print("wrote", os.path.relpath(BASEMAP, REPO), f"({os.path.getsize(BASEMAP) / 1e3:.0f} kB)")


def split_centres():
    """Median panorama location per split, from the committed records."""
    import statistics
    out = {}
    for split in SPLITS:
        lat, lng = [], []
        with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)["pano"]
                lat.append(p["lat"])
                lng.append(p["lng"])
        out[split] = (statistics.median(lng), statistics.median(lat))
    return out


class LocatorMap:
    """Pre-rasterised insets; ``draw(frame, x_right, y_top, split)`` adds the dots."""

    def __init__(self, scale=4):
        from PIL import Image, ImageDraw

        with open(BASEMAP, encoding="utf-8") as f:
            bm = json.load(f)
        centres = split_centres()
        # One dot per place: Laurens has two splits (GSV and Mapillary) at the same town.
        self.place_of = {s: SPLITS[s][0] for s in SPLITS}
        self.places = {}
        for s, (lng, lat) in centres.items():
            self.places.setdefault(self.place_of[s], (lng, lat))
        self.insets = []
        for box, layers, squeeze in ((US_BOX, ("us_neighbours", "us_states"), 0.8),
                                     (WORLD_BOX, ("world",), 1.0)):
            lon0, lon1, lat0, lat1 = box
            w = int(round(MAP_H * (lon1 - lon0) * squeeze / (lat1 - lat0)))
            big = Image.new("RGB", (w * scale, MAP_H * scale), MAP_BG)
            d = ImageDraw.Draw(big)

            def proj(lng, lat, box=box, w=w, s=scale):
                return ((lng - box[0]) / (box[1] - box[0]) * w * s,
                        (box[3] - lat) / (box[3] - box[2]) * MAP_H * s)

            for layer in layers:
                for ring in bm[layer]:
                    pts = [proj(x, y) for x, y in ring]
                    d.polygon(pts, fill=MAP_LAND, outline=MAP_EDGE, width=scale)
            if box is WORLD_BOX:
                a, b = proj(US_BOX[0], US_BOX[3]), proj(US_BOX[1], US_BOX[2])
                d.rectangle((a, b), outline=(110, 110, 105), width=scale)
            img = big.resize((w, MAP_H), Image.LANCZOS)
            self.insets.append((box, w, img))
        self.width = sum(w for _, w, _ in self.insets) + 24 * (len(self.insets) - 1)

    def draw(self, frame, x_right, y_top, split):
        from PIL import ImageDraw

        d = ImageDraw.Draw(frame)
        on = self.place_of[split]
        x = x_right - self.width
        for box, w, img in self.insets:
            frame.paste(img, (x, y_top))
            lon0, lon1, lat0, lat1 = box
            world = box is WORLD_BOX
            r_idle, r_on, ring = (3, 5, 10) if world else (4.5, 6, 13)
            hit = None
            for place, (lng, lat) in self.places.items():
                if not (lon0 <= lng <= lon1 and lat0 <= lat <= lat1):
                    continue
                px = x + (lng - lon0) / (lon1 - lon0) * w
                py = y_top + (lat1 - lat) / (lat1 - lat0) * MAP_H
                if place == on:
                    hit = (px, py)
                else:
                    d.ellipse((px - r_idle, py - r_idle, px + r_idle, py + r_idle), fill=DOT_IDLE)
            if hit:                           # drawn last so it sits on top of its neighbours
                px, py = hit
                d.ellipse((px - ring, py - ring, px + ring, py + ring), outline=DOT_ON, width=3)
                d.ellipse((px - r_on, py - r_on, px + r_on, py + r_on), fill=DOT_ON)
            x += w + 24


def draw_caption(frame, it, n_peaks, index, total, locator=None, y_cap=2 * PANEL_H):
    from PIL import ImageDraw

    d = ImageDraw.Draw(frame)
    place, imagery = SPLITS[it["split"]]
    if it["kind"] == "true_negative":
        verdict = "no curb ramps, no detections"
    else:
        verdict = f"{it['n_ramps']} curb ramps, {n_peaks} detected"
    x_right = FRAME_W - 40
    if locator is not None:
        locator.draw(frame, x_right, y_cap + MAP_PAD, it["split"])
        x_right -= locator.width + 48
    d.text((40, y_cap + 42), place, font=_font(54), fill=(245, 245, 240))
    d.text((40, y_cap + 112), imagery, font=_font(30), fill=(160, 160, 155))
    d.text((x_right, y_cap + 44), verdict, font=_font(34), fill=(130, 215, 255), anchor="ra")
    d.text((x_right, y_cap + 112), f"{index + 1} / {total}", font=_font(28),
           fill=(130, 130, 125), anchor="ra")
    return frame


def render_frame(it, pano_path, heat_path, index, total, locator=None, panels=None,
                 caption="bottom"):
    """One 1920x1080 frame. ``panels`` (an already-rendered panel pair) skips the panorama.

    ``caption`` puts the caption bar (and map) below the two panels, or above them."""
    from PIL import Image

    if panels is None:
        panels = render_panels(pano_path, heat_path)
    frame = Image.new("RGB", (FRAME_W, FRAME_H), (12, 12, 14))
    top = caption == "top"
    frame.paste(panels, (0, CAPTION_H if top else 0))
    return draw_caption(frame, it, len(_peaks(heat_path)[1]), index, total, locator,
                        y_cap=0 if top else 2 * PANEL_H)


def ordered(items, order, per_city=None):
    """Manifest order, or grouped by city (CITY_ORDER) with true negatives last in each city.

    ``per_city`` keeps at most that many panoramas per city: the ones with the most ramps, a
    true negative only when the city has fewer positives than that (Laurens has one). The
    survivors keep their full-cut order, so the abbreviated cut is a subsequence of the full one.
    """
    idx = list(range(len(items)))
    if per_city is not None:
        keep = []
        for split in dict.fromkeys(it["split"] for it in items):
            mine = [i for i in idx if items[i]["split"] == split]
            mine.sort(key=lambda i: (items[i]["kind"] == "true_negative",
                                     items[i]["kind"] != "all_found", -items[i]["n_ramps"], i))
            keep += mine[:per_city]
        idx = sorted(keep)
    if order == "city":
        idx.sort(key=lambda i: (CITY_ORDER.index(items[i]["split"]),
                                items[i]["kind"] == "true_negative", i))
    return idx


def cmd_render(args):
    from PIL import Image

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if not (args.pano_root or args.panels_from):
        sys.exit("render needs --pano-root, --panels-from, or both")
    frames_dir = args.frames_dir
    os.makedirs(frames_dir, exist_ok=True)
    items = manifest["items"]
    locator = LocatorMap() if args.map else None
    for old in os.listdir(frames_dir):          # a shorter cut must not inherit stale frames
        if old.startswith("frame_") and old.endswith(".png"):
            os.remove(os.path.join(frames_dir, old))
    order = ordered(items, args.order, args.per_city)
    for k, i in enumerate(order):
        it = items[i]
        panels = None
        prev = os.path.join(args.panels_from, f"frame_{i:03d}.png") if args.panels_from else None
        if prev and os.path.exists(prev) and "appended" not in it:
            # Frames from a default render are in manifest order: frame_<i> is items[i].
            src = Image.open(prev).convert("RGB")
            if src.size != (FRAME_W, FRAME_H):
                sys.exit(f"{src.size} frame in --panels-from, expected {(FRAME_W, FRAME_H)}")
            panels = src.crop((0, 0, FRAME_W, 2 * PANEL_H))
        frame = render_frame(it, None if panels else _pano_path(args.pano_root, it),
                             _heat_path(it), k, len(order), locator, panels, args.caption)
        frame.save(os.path.join(frames_dir, f"frame_{k:03d}.png"))
        print("frame", k, it["split"], it["pano_id"], flush=True)


def cmd_video(args):
    frames_dir = args.frames_dir
    items = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png"))
    lst = os.path.join(frames_dir, "frames.txt")
    with open(lst, "w", newline="\n") as f:
        for i in range(len(items)):
            f.write(f"file 'frame_{i:03d}.png'\nduration {args.seconds}\n")
        f.write(f"file 'frame_{len(items) - 1:03d}.png'\n")
    out = args.out
    # The repeated last entry is for older ffmpeg, which otherwise drops the last duration;
    # ffmpeg 8 honours both and shows the last frame twice as long, so cap the length.
    total = f"{len(items) * args.seconds:.3f}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-t", total, "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--append", metavar="SPLIT",
                   help="add this split's picks to the existing manifest instead of reselecting")
    p.add_argument("--n-pick", type=int, default=5, help="with --append: exemplars to add")
    sub.add_parser("basemap", help="fetch and trim the Natural Earth outlines for --map")
    p = sub.add_parser("infer")
    p.add_argument("--pano-root", required=True, help="directory holding <split>/panos/")
    p.add_argument("--checkpoint", default=HF_MODEL)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("--pano-root", help="directory holding <split>/panos/")
    p.add_argument("--panels-from", help="frames dir from a default (manifest-order) render; "
                     "reuses its image panels, so no panoramas are needed; items it has no "
                     "frame for (appended later) fall back to --pano-root")
    p.add_argument("--frames-dir", default=FRAMES_DIR)
    p.add_argument("--order", choices=("manifest", "city"), default="manifest")
    p.add_argument("--map", action="store_true", help="locator map in the caption bar")
    p.add_argument("--per-city", type=int, help="at most this many panoramas per city")
    p.add_argument("--caption", choices=("bottom", "top"), default="bottom",
                   help="caption bar (and map) below or above the panels")
    p = sub.add_parser("video")
    p.add_argument("--seconds", type=float, default=2.5, help="seconds per panorama")
    p.add_argument("--frames-dir", default=FRAMES_DIR)
    p.add_argument("--out", default=os.path.join(SHOW_DIR, "showcase.mp4"))
    args = ap.parse_args(argv)
    {"select": cmd_select, "basemap": cmd_basemap, "infer": cmd_infer, "render": cmd_render,
     "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
