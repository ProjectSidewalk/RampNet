#!/usr/bin/env python3
"""Do the auto-placed crosswalk training positives sit on visible paint? (#285, A4 follow-up 2)

``crosswalk_s6`` added 3,040 positives placed automatically from OSM marked crossings (Sacramento,
Kaysville, Brussels) and failed guard 6: many more false alarms on plain road surface. The
hypothesis under test is that some of those positives sit where no crosswalk paint is visible, so
the model learns "road surface at a junction". The pre-registration and the result are in
docs/crosswalk_followups_285.md, section "A4 follow-up 2".

``build`` draws a seeded, stratified sample of TRAINING positives, exactly as the trainer sees them
(the 4096x2048 JPEGs of ``root_v6_full``):

- ``auto``: ``osm_vistas_auto`` positives (``root_v6/positives_full.csv``), AUTO_PER_CITY per city;
- ``human``: crowd-labelled positives of ``root_v4`` (sources ``crowd`` and ``ps_crowd_jon_confirmed``),
  the base rate;
- ``sdot_b2``: ``root_v4``'s SDOT positives, placed by the same Vistas-centroid snap as ``auto`` (the
  snap's own effect, in data s4 already trained on).

Each positive gets a tight crop (the label point, the 1-sigma circle of its training Gaussian and its
3-sigma training disc) and a context crop. The sample table goes to the repo; imagery (crops, blind
review montages, the page) goes to ``--out`` outside it. Montage cells are numbered in a seeded order
that mixes the strata, so a rater tags cells without seeing which stratum a cell is from (the imagery
itself can still hint at the source).

``score`` reads a tags file keyed by cell number, writes the counts with Wilson intervals and the
pre-registered verdict to the repo, and rebuilds the page with the tags and a short skim section.
Tags from a model are a pre-pass, never ground truth.

Usage::

    S=scripts/analysis/stage1_crosswalk_sdot_274
    python $S/pos_sheet.py build --root <keep>/crosswalk274-root_v6_full --out <keep>/osm285/pos_sheet
    python $S/pos_sheet.py score --out <keep>/osm285/pos_sheet \
        --tags analysis_out/crosswalk_followups_285/osm_arm/pos_sheet/ai_prerater_tags_claude-opus-5-5.json
"""
import argparse
import csv
import hashlib
import html as H
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recall_check  # noqa: E402  (light: stdlib only at import time)

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OSM = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "osm_arm")
OUT = os.path.join(OSM, "pos_sheet")

SEED = 285
CITIES = ("sacramento-ca", "kaysville-ut", "brussels-be")
AUTO_PER_CITY = 70
N_HUMAN = 50
N_SDOT = 30
HUMAN_SOURCES = ("crowd", "ps_crowd_jon_confirmed")
ROOT_V4_TRAIN_POSITIVES = 8034    # docs/crosswalk_followups_285.md, A4: root_v4's train positives

# The trainer's target (stage_two/train.py): a Gaussian of sigma 10 px on a 1024-wide heatmap; a sparse
# pano trains only inside the 3-sigma disc around each positive.
SIGMA_W = 10.0 / 1024.0
DISC_W = math.ceil(3 * 10.0) / 1024.0
TIGHT_W = 0.08          # tight crop width, share of the pano width (holds the 3-sigma disc)
TIGHT_ASPECT = 0.75
CONTEXT_W = 0.25
CONTEXT_ASPECT = 0.5
OUT_PX = 360
GRID = (3, 3)           # montage columns, rows

PAINT = ("clear", "faint", "none", "cant_tell")
POINT = ("on_paint", "between", "off", "na")

# Pre-registered rule (docs, "A4 follow-up 2"), fixed before any crop was looked at.
SUPPORT_MIN = 0.10      # auto no-paint share (training-weighted) at least this ...
NOT_SUPPORTED_MARGIN = 0.03
POINT_FLAG_MARGIN = 0.15


def wilson(k, n, z=1.96):
    """Wilson score interval.

    >>> [round(v, 3) for v in wilson(0, 10)]
    [0.0, 0.278]
    """
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def digest(uids):
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()[:12]


def draw(auto_rows, v4_points, seed=SEED, per_city=AUTO_PER_CITY, n_human=N_HUMAN, n_sdot=N_SDOT):
    """The stratified sample, then one seeded order over all strata (the blind cell order).

    ``auto_rows`` are ``positives_full.csv`` rows; ``v4_points`` are dicts with uid, source, city,
    pano_id, x, y. Returns rows with ``stratum`` and ``cell`` (1-based).

    >>> auto = [{"label_uid": f"a{i}", "city": c, "pano_id": f"p{i}", "x": "0.5", "y": "0.55"}
    ...         for i, c in enumerate(["c1"] * 5 + ["c2"] * 5)]
    >>> v4 = [{"uid": f"h{i}", "source": s, "city": "s", "pano_id": f"q{i}", "x": 0.5, "y": 0.55}
    ...       for i, s in enumerate(["crowd"] * 6 + ["sdot_b2"] * 4)]
    >>> rows = draw(auto, v4, per_city=2, n_human=3, n_sdot=2)
    >>> Counter(r["stratum"] for r in rows) == Counter({"auto": 4, "human": 3, "sdot_b2": 2})
    True
    >>> sorted(r["cell"] for r in rows) == list(range(1, 10))
    True
    >>> rows == draw(auto, v4, per_city=2, n_human=3, n_sdot=2)
    True
    """
    out = []
    by_city = defaultdict(list)
    for r in auto_rows:
        by_city[r["city"]].append(r)
    for city in sorted(by_city):
        pool = sorted(by_city[city], key=lambda r: r["label_uid"])
        for r in random.Random(f"{seed}:auto:{city}").sample(pool, min(per_city, len(pool))):
            out.append({"stratum": "auto", "uid": r["label_uid"], "source": "osm_vistas_auto", "city": city,
                        "pano_id": r["pano_id"], "x": float(r["x"]), "y": float(r["y"])})
    for stratum, sources, n in (("human", HUMAN_SOURCES, n_human), ("sdot_b2", ("sdot_b2",), n_sdot)):
        pool = sorted((p for p in v4_points if p["source"] in sources), key=lambda p: p["uid"])
        for p in random.Random(f"{seed}:{stratum}").sample(pool, min(n, len(pool))):
            out.append({"stratum": stratum, **p})
    order = list(range(len(out)))
    random.Random(f"{seed}:order").shuffle(order)
    for cell, i in enumerate(order, 1):
        out[i]["cell"] = cell
    return sorted(out, key=lambda r: r["cell"])


def v4_points(root):
    """Every non-auto train positive of a root (root_v6_full holds root_v4's files unchanged)."""
    pts = []
    tdir = os.path.join(root, "train")
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(tdir, fn), encoding="utf-8"))
        s = d.get("supervision") or {}
        for (x, y), src, pid in zip(d["curb_ramp_points_normalized"], s.get("sources", []), s.get("point_ids", [])):
            if src != "osm_vistas_auto":
                pts.append({"uid": f"{pid}@{d['pano_id']}", "source": src, "city": s.get("city", ""),
                            "pano_id": d["pano_id"], "x": float(x), "y": float(y)})
    return pts


def auto_meta():
    """(pano_id, node_id) -> placement fields and Jon's A3 verdict where the placement was a gate card."""
    meta = {}
    for city in CITIES:
        cdir = os.path.join(OSM, "cities", city)
        key = lambda r: (r["pano_id"], round(float(r["x"]), 4), round(float(r["y"]), 4))  # noqa: E731
        cand = {key(r): r["node_id"] for r in read_csv(os.path.join(cdir, "a3_candidates.csv"))}
        verdict = {}
        for fn in os.listdir(os.path.join(cdir, "ratings")):
            if fn.endswith(".csv"):
                for r in read_csv(os.path.join(cdir, "ratings", fn)):
                    verdict[(r["pano_id"], cand[key(r)])] = r["verdict"]
        for r in read_csv(os.path.join(cdir, "placements_kinds.csv")):
            k = (r["pano_id"], r["node_id"])
            meta[k] = {f: r[f] for f in ("range_m", "comp_area_px", "on_paint", "shift", "basis",
                                         "camera_model", "capture_date")}
            meta[k]["a3_verdict"] = verdict.get(k, "")
    return meta


def _annotate_tight(crop, x, y, box, w, h):
    from PIL import ImageDraw
    left, top, cw, _ = box
    s = crop.size[0] / cw
    px, py = ((x * w - left) % w) * s, (y * h - top) * s
    d = ImageDraw.Draw(crop)
    for r, col in ((DISC_W * w * s, (0, 220, 255)), (SIGMA_W * w * s, (255, 0, 255))):
        d.ellipse([px - r, py - r, px + r, py + r], outline=col, width=1)
    g, arm = 4, 12                     # a cross with a gap, so the pixels under the point stay visible
    for a, b in (((px - arm, py), (px - g, py)), ((px + g, py), (px + arm, py)),
                 ((px, py - arm), (px, py - g)), ((px, py + g), (px, py + arm))):
        d.line([a, b], fill=(255, 0, 255), width=2)
    return crop


def render(img, x, y):
    """(tight crop, context crop) for one positive, annotated."""
    from PIL import ImageDraw
    w, h = img.size
    tight, box = recall_check._crop(img, x, y, TIGHT_W, TIGHT_ASPECT, OUT_PX)
    _annotate_tight(tight, x, y, box, w, h)
    ctx, cbox = recall_check._crop(img, x, y, CONTEXT_W, CONTEXT_ASPECT, OUT_PX)
    s = OUT_PX / cbox[2]
    d = ImageDraw.Draw(ctx)
    tl = ((box[0] - cbox[0]) % w) * s
    d.rectangle([tl, (box[1] - cbox[1]) * s, tl + box[2] * s, (box[1] - cbox[1] + box[3]) * s],
                outline=(255, 0, 255), width=1)
    return tight, ctx


def cmd_build(a):
    from PIL import Image, ImageDraw
    auto_rows = read_csv(os.path.join(OSM, "root_v6", "positives_full.csv"))
    pts = v4_points(a.root)
    n_v4 = Counter(p["source"] for p in pts)
    if sum(n_v4.values()) != ROOT_V4_TRAIN_POSITIVES:
        raise SystemExit(f"{a.root}: {sum(n_v4.values())} non-auto train positives, expected {ROOT_V4_TRAIN_POSITIVES}")
    rows = draw(auto_rows, pts)
    meta = auto_meta()
    for r in rows:
        node = r["uid"].split(":")[-1].split("@")[0] if r["stratum"] == "auto" else ""
        r.update(meta.get((r["pano_id"], node), {}) if node else {})
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "review"), exist_ok=True)
    tiles = []
    for r in rows:
        img = Image.open(os.path.join(a.root, "train", f"{r['pano_id']}.jpg")).convert("RGB")
        tight, ctx = render(img, r["x"], r["y"])
        r["img_tight"], r["img_context"] = f"img/{r['cell']:03d}_tight.jpg", f"img/{r['cell']:03d}_context.jpg"
        tight.save(os.path.join(a.out, r["img_tight"]), quality=90)
        ctx.save(os.path.join(a.out, r["img_context"]), quality=85)
        tile = Image.new("RGB", (OUT_PX, tight.size[1] + ctx.size[1] + 22), (0, 0, 0))
        tile.paste(tight, (0, 22))
        tile.paste(ctx, (0, 22 + tight.size[1]))
        ImageDraw.Draw(tile).text((6, 5), f"cell {r['cell']}", fill=(255, 255, 0))
        tiles.append(tile)
    cols, rws = GRID
    th = tiles[0].size[1]
    for m in range(0, len(tiles), cols * rws):
        sheet = Image.new("RGB", (cols * (OUT_PX + 8), rws * (th + 8)), (40, 40, 40))
        for k, t in enumerate(tiles[m:m + cols * rws]):
            sheet.paste(t, ((k % cols) * (OUT_PX + 8), (k // cols) * (th + 8)))
        sheet.save(os.path.join(a.out, "review", f"montage_{m // (cols * rws) + 1:03d}.jpg"), quality=90)
    os.makedirs(OUT, exist_ok=True)
    fields = ["cell", "stratum", "uid", "source", "city", "pano_id", "x", "y", "range_m", "comp_area_px",
              "on_paint", "shift", "basis", "camera_model", "capture_date", "a3_verdict", "img_tight", "img_context"]
    with open(os.path.join(OUT, "sample.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    info = {"seed": SEED, "auto_per_city": AUTO_PER_CITY, "n_human": N_HUMAN, "n_sdot": N_SDOT,
            "human_sources": list(HUMAN_SOURCES), "root": os.path.basename(os.path.normpath(a.root)),
            "pools": {"auto": len(auto_rows), **{f"v4_{k}": v for k, v in sorted(n_v4.items())}},
            "sample": dict(Counter(f"{r['stratum']} | {r['city']}" if r["stratum"] == "auto" else r["stratum"]
                                   for r in rows)),
            "sample_digest": digest([r["uid"] for r in rows]), "n": len(rows),
            "crop": {"tight_w": TIGHT_W, "context_w": CONTEXT_W, "sigma_w": round(SIGMA_W, 6),
                     "disc_w": round(DISC_W, 6)},
            "montages": math.ceil(len(rows) / (cols * rws))}
    with open(os.path.join(OUT, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1)
    write_page(a.out, rows, None)
    print(json.dumps(info, indent=1))


def tally(rows, tags):
    """Counts and the pre-registered verdict from tags keyed by cell.

    >>> rows = [{"cell": i, "stratum": s, "city": c} for i, (s, c) in enumerate(
    ...     [("auto", "a")] * 10 + [("auto", "b")] * 10 + [("human", "h")] * 10 + [("sdot_b2", "s")] * 10, 1)]
    >>> tags = {str(i): {"paint": "clear", "point": "on_paint"} for i in range(1, 41)}
    >>> tally(rows, tags, )["verdict"]
    'not_supported'
    >>> for i in range(1, 9): tags[str(i)] = {"paint": "none", "point": "off"}
    >>> for i in range(11, 19): tags[str(i)] = {"paint": "none", "point": "off"}
    >>> tally(rows, tags)["verdict"]
    'supported'
    """
    weights = {}
    pos = read_csv(os.path.join(OSM, "root_v6", "positives_full.csv")) if os.path.exists(
        os.path.join(OSM, "root_v6", "positives_full.csv")) else []
    n_pos = Counter(r["city"] for r in pos)
    res = {"strata": {}, "auto_cities": {}}

    def block(rs):
        paint = Counter(tags[str(r["cell"])]["paint"] for r in rs)
        point = Counter(tags[str(r["cell"])]["point"] for r in rs)
        decided = sum(paint[k] for k in ("clear", "faint", "none"))
        pdec = sum(point[k] for k in ("on_paint", "between", "off"))
        off = point["between"] + point["off"]
        return {"n": len(rs), "paint": {k: paint[k] for k in PAINT}, "point": {k: point[k] for k in POINT},
                "no_paint_share": paint["none"] / decided if decided else None,
                "no_paint_wilson": wilson(paint["none"], decided),
                "no_paint_or_cant_tell_share": (paint["none"] + paint["cant_tell"]) / len(rs) if rs else None,
                "faint_share": paint["faint"] / decided if decided else None,
                "peak_off_paint_share": off / pdec if pdec else None, "peak_off_paint_wilson": wilson(off, pdec)}

    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    for s, rs in sorted(by.items()):
        res["strata"][s] = block(rs)
    auto_city = defaultdict(list)
    for r in by.get("auto", []):
        auto_city[r["city"]].append(r)
    for c, rs in sorted(auto_city.items()):
        res["auto_cities"][c] = block(rs)
        weights[c] = n_pos.get(c, len(rs))
    tot = sum(weights.values())
    shares = {c: res["auto_cities"][c]["no_paint_share"] for c in weights}
    p_auto = sum(weights[c] * (shares[c] or 0.0) for c in weights) / tot if tot else None
    res["auto_no_paint_training_weighted"] = p_auto
    res["weights"] = weights
    lo, hi = res["strata"]["auto"]["no_paint_wilson"]
    p_h = res["strata"].get("human", {}).get("no_paint_share") or 0.0
    if p_auto is not None and p_auto >= SUPPORT_MIN and lo > p_h:
        v = "supported"
    elif hi < SUPPORT_MIN or (p_auto is not None and p_auto <= p_h + NOT_SUPPORTED_MARGIN):
        v = "not_supported"
    else:
        v = "inconclusive"
    res["verdict"] = v
    q = {s: res["strata"].get(s, {}).get("peak_off_paint_share") for s in ("auto", "human", "sdot_b2")}
    res["point_flag"] = bool(q["auto"] is not None and q["human"] is not None and q["sdot_b2"] is not None
                             and q["auto"] >= q["human"] + POINT_FLAG_MARGIN
                             and q["auto"] >= q["sdot_b2"] + POINT_FLAG_MARGIN)
    res["rule"] = {"support_min": SUPPORT_MIN, "not_supported_margin": NOT_SUPPORTED_MARGIN,
                   "point_flag_margin": POINT_FLAG_MARGIN}
    return res


def skim(rows, tags, k=20):
    """Up to ``k`` cells for the top of the page: every auto no-paint, then auto faint, then auto peak-off-paint,
    then human/sdot no-paint, each in cell order."""
    t = lambda r: tags[str(r["cell"])]  # noqa: E731
    tiers = [lambda r: r["stratum"] == "auto" and t(r)["paint"] == "none",
             lambda r: r["stratum"] == "auto" and t(r)["paint"] == "faint",
             lambda r: r["stratum"] == "auto" and t(r)["point"] in ("between", "off"),
             lambda r: r["stratum"] != "auto" and t(r)["paint"] == "none",
             lambda r: r["stratum"] != "auto" and t(r)["point"] in ("between", "off")]
    out, seen = [], set()
    for f in tiers:
        for r in rows:
            if r["cell"] not in seen and f(r):
                out.append(r)
                seen.add(r["cell"])
    nopaint = [r for r in out if r["stratum"] == "auto" and t(r)["paint"] == "none"]
    return out[:max(k, len(nopaint))]


def _card(r, tags):
    tg = tags.get(str(r["cell"])) if tags else None
    extra = ""
    if r["stratum"] == "auto":
        extra = (f"{H.escape(r.get('camera_model', ''))}, {r.get('range_m', '')} m, region {r.get('comp_area_px', '')} px"
                 + (f", A3 verdict: {H.escape(r['a3_verdict'])}" if r.get("a3_verdict") else ""))
    tagtxt = (f'<br><strong>AI pre-pass:</strong> paint <em>{H.escape(tg["paint"])}</em>, peak '
              f'<em>{H.escape(tg["point"])}</em>' + (f' &middot; {H.escape(tg["note"])}' if tg.get("note") else "")
              ) if tg else ""
    data = f'data-stratum="{r["stratum"]}"' + (f' data-paint="{tg["paint"]}" data-point="{tg["point"]}"' if tg else "")
    return (f'<figure class="pos" {data}><img src="{r["img_tight"]}" width="{OUT_PX}" alt="Tight crop of cell '
            f'{r["cell"]}: magenta cross = label point, magenta circle = 1 sigma, cyan circle = training disc">'
            f'<img src="{r["img_context"]}" width="{OUT_PX}" alt="Context around cell {r["cell"]}; the box is the '
            f'tight crop"><figcaption><strong>Cell {r["cell"]}</strong> &middot; {r["stratum"]} &middot; '
            f'{H.escape(r["city"])}<br>{extra}<br><code>{H.escape(r["pano_id"])}</code>{tagtxt}</figcaption></figure>')


def write_page(out, rows, tags, res=None):
    rows = [dict(r, cell=int(r["cell"])) for r in rows]
    summary = ""
    if res:
        def pct(v):
            return "&ndash;" if v is None else f"{100 * v:.1f}%"

        def ci(v):
            return "" if v[0] is None else f" ({100 * v[0]:.1f}&ndash;{100 * v[1]:.1f})"
        trs = []
        for name, b in [(f"auto &middot; {c}", b) for c, b in res["auto_cities"].items()] + \
                       [(s, b) for s, b in res["strata"].items()]:
            trs.append(f"<tr><th scope='row'>{name}</th><td>{b['n']}</td>"
                       + "".join(f"<td>{b['paint'][k]}</td>" for k in PAINT)
                       + f"<td>{pct(b['no_paint_share'])}{ci(b['no_paint_wilson'])}</td>"
                       + f"<td>{pct(b['peak_off_paint_share'])}{ci(b['peak_off_paint_wilson'])}</td></tr>")
        summary = (f"<p><strong>Pre-registered verdict: {res['verdict'].replace('_', ' ')}.</strong> Auto no-paint "
                   f"share, weighted by each city's training positives: {pct(res['auto_no_paint_training_weighted'])}. "
                   f"Point-placement flag: {'raised' if res['point_flag'] else 'not raised'}. Tags are an AI pre-pass "
                   f"(claude-opus-5-5), not ground truth.</p>"
                   "<table><tr><th scope='col'>Stratum</th><th scope='col'>n</th>"
                   + "".join(f"<th scope='col'>{k}</th>" for k in PAINT)
                   + "<th scope='col'>No paint (95% CI)</th><th scope='col'>Peak off paint (95% CI)</th></tr>"
                   + "".join(trs) + "</table>")
    sk = ""
    if tags:
        sel = skim(rows, tags)
        sk = (f"<section><h2>Skim first: {len(sel)} cells</h2><p>Every auto positive tagged no visible paint, then "
              f"faint paint, then a label point off the paint; then the same from the human and SDOT strata.</p>"
              f"<div class='grid'>{''.join(_card(r, tags) for r in sel)}</div></section>")
    filt = ("<fieldset><legend>Show</legend>"
            + "".join(f'<label><input type="radio" name="f" value="{v}"{" checked" if v == "all" else ""}> {v}</label> '
                      for v in ("all", "auto", "human", "sdot_b2"))
            + "</fieldset>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training positives: paint check</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#555; --line:#ddd; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161616; --fg:#eee; --muted:#aaa; --line:#333; }} }}
  body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 16px 48px; }}
  h1 {{ font-size:1.4em; }} h2 {{ font-size:1.15em; }}
  table {{ border-collapse:collapse; margin:8px 0 16px; }} td, th {{ border:1px solid var(--line); padding:3px 8px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .pos {{ margin:0; max-width:{OUT_PX}px; }} .pos img {{ max-width:100%; height:auto; display:block; }}
  figcaption {{ color:var(--muted); font-size:.88em; margin-top:4px; }}
  fieldset {{ border:1px solid var(--line); margin:12px 0; }}
</style></head><body>
<h1>Do the crosswalk training positives sit on visible paint? (#285, A4 follow-up 2)</h1>
<p>Training positives as the trainer sees them (4096&times;2048). Top crop: magenta cross = the label point, magenta
circle = 1 sigma of its target, cyan circle = the 3-sigma disc a sparse pano trains inside. Bottom: context, with the
top crop boxed. Strata: <em>auto</em> = OSM + Vistas placements added in crosswalk_s6; <em>human</em> = crowd-labelled
positives of root_v4; <em>sdot_b2</em> = root_v4's SDOT positives, placed by the same Vistas snap. Nothing to rate.</p>
{summary}{sk}<section><h2>All {len(rows)} cells</h2>{filt}<div class="grid" id="all">
{''.join(_card(r, tags) for r in rows)}</div></section>
<script>
  document.querySelectorAll('input[name=f]').forEach(function (r) {{
    r.addEventListener('change', function () {{
      document.querySelectorAll('#all .pos').forEach(function (p) {{
        p.hidden = !(r.value === 'all' || p.dataset.stratum === r.value);
      }});
    }});
  }});
</script></body></html>
"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


def cmd_score(a):
    rows = read_csv(os.path.join(OUT, "sample.csv"))
    doc = json.load(open(a.tags, encoding="utf-8"))
    if not doc.get("not_ground_truth"):
        raise SystemExit(f"{a.tags}: a pre-pass tags file must say not_ground_truth: true")
    tags = doc["tags"]
    missing = [r["cell"] for r in rows if r["cell"] not in tags]
    bad = [c for c, t in tags.items() if t.get("paint") not in PAINT or t.get("point") not in POINT]
    if missing or bad:
        raise SystemExit(f"untagged cells {missing[:10]}..., invalid tags {bad[:10]}")
    for r in rows:
        r["cell"] = int(r["cell"])
    res = tally(rows, tags)
    agree = Counter((r["a3_verdict"], tags[str(r["cell"])]["paint"]) for r in rows if r.get("a3_verdict"))
    res["a3_overlap"] = {f"{v} | {p}": n for (v, p), n in sorted(agree.items())}
    res["tags_file"] = os.path.basename(a.tags)
    res["rater"] = {k: doc.get(k) for k in ("rater", "model", "date", "not_ground_truth")}
    res["sample_digest"] = json.load(open(os.path.join(OUT, "sample.json"), encoding="utf-8"))["sample_digest"]
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    write_page(a.out, rows, tags, res)
    print(json.dumps(res, indent=1))
    print(f"-> {os.path.join(a.out, 'index.html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="seeded sample, crops, blind review montages and the untagged page")
    b.add_argument("--root", required=True, help="root_v6_full (its train/ holds every sampled pano at 4096 px)")
    b.add_argument("--out", required=True, help="page dir with crops and montages; outside the repo")
    s = sub.add_parser("score", help="counts, Wilson intervals and the pre-registered verdict from a tags file")
    s.add_argument("--out", required=True, help="the page dir written by build")
    s.add_argument("--tags", required=True, help="tags JSON keyed by cell: {paint, point, note}")
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
