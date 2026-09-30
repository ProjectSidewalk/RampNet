"""Example figures for the perspective-photo test (issue #218).

Reads only committed outputs of ``perspective_photos_218.py`` / ``seoul_photos_218.py``
(detections, census, pano captures) to pick examples by a stated rule, then draws them
over the source images, which are not committed (they are re-fetched by id and checked
against the committed sha256s).

Subcommands::

    # CPU, committed files only: every candidate for every figure slot, in rule order
    python scripts/analysis/perspective_figures_218.py candidates
    # CPU: draw the figures (needs the images)
    python scripts/analysis/perspective_figures_218.py render \\
        --flat-dir IMGDIR --pano-dir PANODIR --seoul-dir SEOUL_DISPLAY_DIR

``IMGDIR`` holds the Richmond 2048-px thumbnails (``perspective_photos_218.py fetch``),
``PANODIR`` the Richmond 360 panos of ``benchmark/richmond_neighbourhood`` (as used by
``docs/multiview_48.md``), and ``SEOUL_DISPLAY_DIR`` the 1,400-px display copies that
``seoul_photos_218.py gallery`` writes to ``benchmark/seoul_presence_218/img/``.

Selection rules (docs/perspective_photos_218.md, "Examples"); every pair is scored with
the canvas_level arm at 0.30 under the primary bearing test, exactly as ``score`` does:

- hits: in-view pairs the canvas arm hits, ordered by the claiming detection's score,
  highest first, one per ramp and per image; the first two;
- misses: in-view pairs at 6-12 m, the ramp in the central half of the frame
  (|bearing - heading| <= HFOV / 4), missed by every flat arm at 0.30, whose ramp has
  a non-source pano capture that RampNet hits at 0.55; ordered by |bearing - heading|,
  one per ramp and per image. The two shown are the first two in that order in which the ramp is
  plainly visible by eye (``--misses``; the candidates looked at are listed in
  ``MISSES_VIEWED``);
- unmatched: gallery card ``d001`` of ``benchmark/richmond_flat_fp_218``, the one card
  already viewed before rating (docs section 7), so showing it unblinds no other card;
- stretch: in-view pairs the stretch hits at 0.30 and neither canvas arm hits, ordered
  by the stretch detection's score; the first;
- Seoul: canvas_level max score ranks 1 and 2, the median photo among those that fired
  at 0.30, and one photo with no detection at 0.30 drawn with seed 218.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet import perspective as P  # noqa: E402
import perspective_photos_218 as PP  # noqa: E402

FIG_DIR = os.path.join(REPO, "docs", "figures", "perspective_photos_218")
SEOUL_DETS = os.path.join(PP.OUT, "seoul", "dets_canvas_level.jsonl")
SEOUL_FETCHED = os.path.join(PP.OUT, "seoul", "fetched.csv")
FP_MANIFEST = os.path.join(REPO, "benchmark", "richmond_flat_fp_218", "manifest.json")
THR = PP.PRIMARY_THR
PANO_THR = 0.55
UNMATCHED_CARD = "d001"
SEED = 218
MAX_BYTES = 1_500_000
MAPILLARY = "Imagery: Mapillary contributors, CC BY-SA"
SEOUL_CREDIT = "Photos: Seoul Sidewalk Accessibility Image Dataset (Lieu et al.), CC0"


# --------------------------------------------------------------------------- #
# selection (committed files only)
# --------------------------------------------------------------------------- #
def pano_best():
    """{ramp uid: best non-source 3-18 m pano capture}: highest world_conf, then nearest.
    Only captures whose pano is in the neighbourhood records (so detections can be drawn)."""
    have = set()
    with open(PP.NEIGHBOURHOOD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                have.add(json.loads(line)["pano"]["panorama_id"])
    best = {}
    for r in PP.read_csv(PP.CAPTURES):
        if r["city"] != "richmond" or r["is_source"] != "0" or r["pano_id"] not in have:
            continue
        d = float(r["dist_m"])
        if not (PP.RANGE_MIN <= d <= PP.RANGE_MAX):
            continue
        wc = float(r["world_conf"]) if r["world_conf"] not in ("", "nan") else 0.0
        key = (-wc, d, r["pano_id"])
        if r["ramp_uid"] not in best or key < best[r["ramp_uid"]][0]:
            best[r["ramp_uid"]] = (key, {"pano_id": r["pano_id"], "dist_m": d,
                                         "world_conf": wc, "x_proj": float(r["x_proj"]),
                                         "y_proj": float(r["y_proj"])})
    return {u: v[1] for u, v in best.items()}


def flat_pairs():
    """Every in-view (image, ramp) pair with the bearing-test claim of each arm at THR."""
    rows = {r["image_id"]: r for r in PP.read_csv(PP.IMAGES_CSV)}
    ramps = PP.ramp_table()
    recs = {a: PP.load_dets(a) for a in ("canvas_level", "canvas_sfm", "stretch")}
    out = []
    for iid in sorted(recs["canvas_level"]):
        rc = recs["canvas_level"][iid]
        g = PP.image_geometry(rows[iid], rc["width"], rc["height"], ramps)
        if not g["positive"]:
            continue
        claims = {}
        for arm, rr in recs.items():
            dets = rr[iid]["dets"]
            b, dep, _ = PP.det_world(dets, g["cam"], g["R_wc"])
            claims[arm] = PP.claim_bearing(dets, b, dep, g["near"], THR)
        hfov = g["cam"].hfov_deg()
        for r in g["near"]:
            if not r["in_view"]:
                continue
            p = {"image_id": iid, "ramp": r["uid"], "range": round(r["range"], 2),
                 "dbear": round(r["dbear"], 2), "hfov": round(hfov, 1),
                 "camera": f'{rows[iid]["make"]} {rows[iid]["model"]}'.strip(),
                 "capture_date": rows[iid]["capture_date"]}
            for arm in recs:
                i = claims[arm].get(r["uid"])
                p[f"{arm}_det"] = i
                p[f"{arm}_score"] = None if i is None else recs[arm][iid]["dets"][i]["score"]
            out.append(p)
    return out


def one_per_ramp(pairs):
    """Keep the first pair of each ramp and of each image, in the given order."""
    seen, out = set(), []
    for p in pairs:
        if p["ramp"] not in seen and p["image_id"] not in seen:
            seen.update((p["ramp"], p["image_id"]))
            out.append(p)
    return out


def candidates():
    pairs = flat_pairs()
    pano = pano_best()
    hits = one_per_ramp(sorted((p for p in pairs if p["canvas_level_det"] is not None),
                               key=lambda p: (-p["canvas_level_score"], p["image_id"])))
    misses = one_per_ramp(sorted(
        (p for p in pairs
         if all(p[f"{a}_det"] is None for a in ("canvas_level", "canvas_sfm", "stretch"))
         and 6.0 <= p["range"] < 12.0 and abs(p["dbear"]) <= p["hfov"] / 4
         and p["ramp"] in pano and pano[p["ramp"]]["world_conf"] >= PANO_THR),
        key=lambda p: (abs(p["dbear"]), p["image_id"])))
    stretch = sorted((p for p in pairs if p["stretch_det"] is not None
                      and p["canvas_level_det"] is None and p["canvas_sfm_det"] is None),
                     key=lambda p: (-p["stretch_score"], p["image_id"]))
    for lst in (hits, misses, stretch):
        for p in lst:
            p["pano"] = pano.get(p["ramp"])
    seoul = [json.loads(line) for line in open(SEOUL_DETS, encoding="utf-8") if line.strip()]
    ranked = sorted(seoul, key=lambda r: (-r["max_score"], r["filename"]))
    fired = [r for r in ranked if r["max_score"] >= THR]
    quiet = sorted(r["filename"] for r in seoul if r["max_score"] < THR)
    rng = np.random.default_rng(SEED)
    seoul_pick = [ranked[0]["filename"], ranked[1]["filename"],
                  fired[(len(fired) - 1) // 2]["filename"],
                  quiet[int(rng.integers(0, len(quiet)))]]
    return {"n_pairs": len(pairs), "hits": hits, "misses": misses, "stretch": stretch,
            "seoul": seoul_pick, "seoul_n_fired": len(fired)}


def cmd_candidates(args):
    c = candidates()
    print(f"{c['n_pairs']} in-view pairs")
    for k in ("hits", "misses", "stretch"):
        print(f"\n{k}: {len(c[k])}")
        for p in c[k][:args.show]:
            pn = p["pano"] or {}
            print(f"  {p['image_id']} {p['ramp']:>13} {p['range']:5.1f} m dbear {p['dbear']:6.1f}"
                  f" hfov {p['hfov']:5.1f} canvas {p['canvas_level_score']} stretch "
                  f"{p['stretch_score']} | {p['camera']} {p['capture_date']} | pano "
                  f"{pn.get('pano_id')} {pn.get('dist_m')} m conf {pn.get('world_conf')}")
    print(f"\nseoul ({c['seoul_n_fired']} fired at {THR}):", c["seoul"])


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
# The miss candidates looked at by eye, in candidate order, with what was seen at the
# GT ramp's bearing (the dotted locus). The first two marked "visible" are drawn.
MISSES_VIEWED = [
    ("320533466155073", "richmond:22", "not visible: locus is mid-road, no ramp there"),
    ("301315184774353", "richmond:31", "not plainly visible: locus is on the far crosswalk"),
    ("590984823847247", "richmond:93", "visible: tactile-paved ramp right of the locus"),
    ("1301363080261966", "richmond:8", "not visible: locus is mid-road"),
    ("648022270132445", "richmond:195", "unclear: plaza edge at the crosswalk, no clear ramp"),
    ("952482502230245", "richmond:145", "visible: tactile-paved ramp at the right edge"),
    ("323753172499410", "richmond:144", "visible: tactile-paved ramp at the right edge"),
    ("477911040083348", "richmond:188", "not visible: locus is mid-road"),
    ("2907449746136160", "richmond:19", "not visible: night, motion blur"),
    ("496296311587433", "richmond:108", "not visible: locus falls on a porch (pose error)"),
    ("271541614653055", "richmond:221", "not visible: locus is mid-road"),
    ("1195329198857725", "richmond:24", "unclear: locus at a crosswalk end, curb not "
                                        "clearly ramped"),
    ("1791267528389167", "richmond:77", "not visible: occluded by a vehicle"),
    ("1092779572545829", "richmond:74", "not visible at the locus: crosswalk edge by the "
                                        "median"),
    ("532146831289058", "richmond:52", "not visible: locus is mid-road"),
]
DEFAULT_MISSES = "590984823847247:richmond:93,952482502230245:richmond:145"

MAG = "#d6168b"      # GT locus
HIT = "#00a86b"      # the claiming detection
DET = "#ff8c00"      # other detections >= THR
WEAK = "#ffd24d"     # peaks between the 0.10 floor and THR


def gt_locus(g, r, heights=np.linspace(0.5, 4.0, 36)):
    """World points at the ramp seen from camera heights 0.5-4 m -> camera-frame rays.
    This is the locus the bearing test accepts a detection on (docs section 3)."""
    return np.stack([g["R_wc"] @ np.array([r["e"], r["n"], -h]) for h in heights])


def draw_dets(ax, pts, thr=THR, hit=None, sx=1.0, sy=1.0, floor=True, label=True,
              extent=None):
    """pts: [(px, py, score)] in image pixels, drawn at (px * sx, py * sy). Ring size is
    a fixed share of the panel width ``extent`` (w, h), and points outside it are skipped."""
    import matplotlib.patches as mp
    w, h = extent
    rad = 0.028 * w
    for k, (x, y, s) in enumerate(pts):
        if s < thr and not floor:
            continue
        x, y = x * sx, y * sy
        if not (0 <= x <= w and 0 <= y <= h):
            continue
        if k == hit:
            col, lw, ls = HIT, 3.0, "-"
        elif s >= thr:
            col, lw, ls = DET, 2.5, "-"
        else:
            col, lw, ls = WEAK, 1.5, "--"
        ax.add_patch(mp.Circle((x, y), rad if s >= thr else 0.7 * rad, fill=False, ec=col,
                               lw=lw, ls=ls))
        if label and (s >= thr or k == hit):
            ax.text(x + 1.1 * rad, y - 1.1 * rad, f"{s:.2f}" + (" hit" if k == hit else ""),
                    color=col, fontsize=9, fontweight="bold", clip_on=True,
                    bbox=dict(fc="black", alpha=0.55, pad=1.5, lw=0))


def panel(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def load_photo(flat_dir, iid, fetched):
    from PIL import Image
    path = os.path.join(flat_dir, f"{iid}.jpg")
    if PP.sha256_file(path) != fetched[iid]["sha256"]:
        raise SystemExit(f"{path}: sha256 differs from fetched.csv")
    return Image.open(path).convert("RGB")


def flat_panel(ax, img, dets, g, r, hit, arm_name, disp_w=1000):
    s = disp_w / img.size[0]
    small = img.resize((disp_w, round(img.size[1] * s)))
    panel(ax, small, f"flat photo, {arm_name} detections")
    if r is not None:
        u, v = P.project_cam(g["cam"], gt_locus(g, r))
        ok = np.isfinite(u) & (v < img.size[1]) & (u >= 0) & (u < img.size[0])
        ax.plot(u[ok] * s, v[ok] * s, ":", color=MAG, lw=2.5)
    draw_dets(ax, [(d["u"], d["v"], d["score"]) for d in dets], hit=hit, sx=s, sy=s,
              extent=small.size)
    ax.set_xlim(0, small.size[0])
    ax.set_ylim(small.size[1], 0)


def stretch_panel(ax, img, dets, g, r, hit, disp_w=1000):
    """The stretch arm's actual input (photo resized to 2048x4096), shown at 2:1."""
    small = img.resize((disp_w, disp_w // 2))
    panel(ax, small, "stretch arm (b) input, photo resized to 2048x4096")
    sx, sy = disp_w / img.size[0], (disp_w / 2) / img.size[1]
    if r is not None:
        u, v = P.project_cam(g["cam"], gt_locus(g, r))
        ok = np.isfinite(u) & (v < img.size[1])
        ax.plot((u[ok] + 0.5) * sx, (v[ok] + 0.5) * sy, ":", color=MAG, lw=2.5)
    draw_dets(ax, [(d["u"] + 0.5, d["v"] + 0.5, d["score"]) for d in dets], hit=hit,
              sx=sx, sy=sy, extent=(disp_w, disp_w // 2))
    ax.set_xlim(0, disp_w)
    ax.set_ylim(disp_w // 2, 0)


def canvas_rgb(img, cam, M=None):
    """The canvas arm's input as RGB uint8 (2048x4096), exactly as infer built it."""
    t, inside = PP.build_canvas_tensor(img, cam, np.eye(3) if M is None else M,
                                       P.CANVAS_H, P.CANVAS_W)
    a = t.numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) \
        + np.array([0.485, 0.456, 0.406])
    return (np.clip(a, 0, 1) * 255).astype(np.uint8), inside


def canvas_panel(ax, img, dets, g, r, hit):
    cam = g["cam"]
    rgb, inside = canvas_rgb(img, cam)
    rows_, cols_ = np.nonzero(inside)
    r0, r1, c0, c1 = rows_.min(), rows_.max() + 1, cols_.min(), cols_.max() + 1
    pad_c, pad_r = 60, 40
    c0, c1 = max(0, c0 - pad_c), min(P.CANVAS_W, c1 + pad_c)
    r0, r1 = max(0, r0 - pad_r), min(P.CANVAS_H, r1 + pad_r)
    crop = rgb[r0:r1, c0:c1]
    deg = 360.0 / P.CANVAS_W
    panel(ax, crop, f"canvas arm (a): crop of the 2048x4096 equirect input, "
                    f"{(c1 - c0) * deg:.0f}° x {(r1 - r0) * deg:.0f}° shown")
    if r is not None:
        x, y = P.ray_to_canvas_norm(gt_locus(g, r))
        ax.plot(x * P.CANVAS_W - c0, y * P.CANVAS_H - r0, ":", color=MAG, lw=2.5)
    draw_dets(ax, [(d["x"] * P.CANVAS_W - c0, d["y"] * P.CANVAS_H - r0, d["score"])
                   for d in dets], hit=hit, extent=(c1 - c0, r1 - r0))
    ax.set_xlim(0, c1 - c0)
    ax.set_ylim(r1 - r0, 0)


def pano_dets():
    out = {}
    with open(PP.NEIGHBOURHOOD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["pano"]["panorama_id"]] = r
    return out


def pano_panel(ax, pano_dir, pano, rec, half_w_deg=40.0, half_h_deg=20.0):
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(pano_dir, f"{pano['pano_id']}.jpg"))
                    .convert("RGB"))
    H, W = im.shape[:2]
    cx = int(round(pano["x_proj"] * W))
    im = np.roll(im, W // 2 - cx, axis=1)            # ramp column to the centre
    hw, hh = int(W * half_w_deg / 360), int(H * half_h_deg / 180)
    cy = int(round(pano["y_proj"] * H))
    y0, y1 = max(0, cy - hh), min(H, cy + hh)
    crop = im[y0:y1, W // 2 - hw:W // 2 + hw]
    s = 1000 / crop.shape[1]
    crop = np.asarray(Image.fromarray(crop).resize((1000, round(crop.shape[0] * s))))
    date = rec["pano"].get("capture_date", "")
    panel(ax, crop, f"360 pano {pano['pano_id']} ({date}), {pano['dist_m']:.1f} m; "
                    f"detections ≥ {PANO_THR}")
    ax.plot([(hw) * s], [(cy - y0) * s], marker="+", ms=22, mew=2.5, color=MAG)
    pts = []
    for d in rec["detections"]:
        if d["confidence"] < PANO_THR:
            continue
        dx = ((d["x_normalized"] - pano["x_proj"] + 0.5) % 1.0 - 0.5) * W
        pts.append(((dx + hw) * s, (d["y_normalized"] * H - y0) * s, d["confidence"]))
    near = [k for k, (x, y, _) in enumerate(pts) if 0 <= x <= crop.shape[1]]
    hit = min(near, key=lambda k: abs(pts[k][0] - hw * s)) if pano["world_conf"] >= PANO_THR \
        and near else None
    draw_dets(ax, pts, thr=PANO_THR, hit=hit, floor=False,
              extent=(crop.shape[1], crop.shape[0]))
    ax.set_xlim(0, crop.shape[1])
    ax.set_ylim(crop.shape[0], 0)


def zoom_panel(ax, img, it):
    box = PP.crop_box(it["u"], it["v"], it["width"], it["height"])
    crop = img.crop(box)
    panel(ax, crop, f"zoom: gallery card {it['item']} (crop the rater sees)")
    draw_dets(ax, [(it["u"] - box[0], it["v"] - box[1], it["score"])], label=True,
              extent=crop.size)
    ax.set_xlim(0, crop.size[0])
    ax.set_ylim(crop.size[1], 0)


def legend(fig, y=0.005, pano=True):
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=MAG, ls=":", lw=2.5,
                   label="GT pool ramp: bearing, for camera heights 0.5-4 m"),
         ml.Line2D([], [], color=HIT, marker="o", mfc="none", ls="", ms=12, mew=3,
                   label="detection that hits it (bearing test)"),
         ml.Line2D([], [], color=DET, marker="o", mfc="none", ls="", ms=12, mew=2.5,
                   label=f"other detection ≥ {THR}"),
         ml.Line2D([], [], color=WEAK, marker="o", mfc="none", ls="", ms=9, mew=1.5,
                   label=f"peak 0.10-{THR} (below the operating point)")]
    if pano:
        h.append(ml.Line2D([], [], color=MAG, marker="+", ls="", ms=14, mew=2.5,
                           label="GT ramp in the 360 pano"))
    fig.legend(handles=h, loc="lower left", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.01, y + 0.018))
    fig.text(0.99, y, MAPILLARY, ha="right", va="bottom", fontsize=8.5, color="#444")


def save(fig, name, out):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 82, "optimize": True})
    n = os.path.getsize(path)
    if n > MAX_BYTES:
        raise SystemExit(f"{path}: {n} bytes > {MAX_BYTES}")
    print(f"{path}: {n / 1e6:.2f} MB")
    return {"file": name, "bytes": n, "sha256": PP.sha256_file(path)}


def pair_row(fig, axes, kind, p, ctx, arm="canvas_level"):
    """One image-ramp row: flat photo | canvas | pano (or stretch input | canvas | pano)."""
    img = load_photo(ctx["flat_dir"], p["image_id"], ctx["fetched"])
    row = ctx["rows"][p["image_id"]]
    g = PP.image_geometry(row, img.size[0], img.size[1], ctx["ramps"])
    r = next(x for x in g["near"] if x["uid"] == p["ramp"])
    recs = ctx["dets"]
    cdets = recs["canvas_level"][p["image_id"]]["dets"]
    if arm == "stretch":
        sd = recs["stretch"][p["image_id"]]["dets"]
        stretch_panel(axes[0], img, sd, g, r, p["stretch_det"])
    else:
        flat_panel(axes[0], img, cdets, g, r, p["canvas_level_det"], "canvas arm (a)")
    canvas_panel(axes[1], img, cdets, g, r, p["canvas_level_det"])
    pano_panel(axes[2], ctx["pano_dir"], p["pano"], ctx["pano_recs"][p["pano"]["pano_id"]])
    sc = p["canvas_level_score"]
    win = math.degrees(math.atan(PP.LATERAL_M / r["range"]))

    def off(dets, i):
        b, _, _ = PP.det_world([dets[i]], g["cam"], g["R_wc"])
        return abs(float(P.wrap_deg(b[0] - r["bearing"])))
    if kind == "hit":
        what = (f"HIT, canvas arm {sc:.2f}, {off(cdets, p['canvas_level_det']):.0f}° from "
                f"the GT bearing (window ±{win:.0f}°)")
    elif kind == "miss":
        what = "MISS by every flat arm at 0.30"
        if cdets:
            i = max(range(len(cdets)), key=lambda k: cdets[k]["score"])
            what += (f"; top canvas peak {cdets[i]['score']:.2f} is {off(cdets, i):.0f}° from "
                     f"the GT bearing (window ±{win:.0f}°)")
    else:
        what = (f"stretch arm {p['stretch_score']:.2f} hits at "
                f"{off(sd, p['stretch_det']):.0f}° from the GT bearing (window ±{win:.0f}°); "
                f"canvas arms do not")
    camera = "unnamed camera" if p["camera"] == "none none" else p["camera"]
    return (f"{what}\n{p['ramp']}, {p['range']:.1f} m, {p['dbear']:+.0f}° from heading  |  "
            f"{camera} {p['capture_date']}, image {p['image_id']}  |  360 pano: RampNet "
            f"{p['pano']['world_conf']:.2f} on the same ramp")


def row_headers(fig, ax, heads):
    """Row captions above each row's grid cells (after subplots_adjust)."""
    for k, h in enumerate(heads):
        top = max(a.get_position().y1 for a in ax[k])
        fig.text(0.01, top + 0.03, h, fontsize=10.5, fontweight="bold", va="bottom",
                 linespacing=1.5)


def render_richmond(ctx, sel, out):
    import matplotlib.pyplot as plt
    files = []
    groups = [("richmond_hits.jpg", [("hit", p) for p in sel["hits"]]),
              ("richmond_misses.jpg", [("miss", p) for p in sel["misses"]])]
    for name, rows_ in groups:
        fig, ax = plt.subplots(len(rows_), 3, figsize=(19, 5.2 * len(rows_)),
                               gridspec_kw={"width_ratios": [1, 1, 1]})
        heads = [pair_row(fig, ax[k], kind, p, ctx) for k, (kind, p) in enumerate(rows_)]
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, hspace=0.5,
                            wspace=0.04)
        row_headers(fig, ax, heads)
        legend(fig)
        files.append(save(fig, name, out))
        plt.close(fig)
    # unmatched card + stretch example
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.4))
    it = sel["unmatched"]
    img = load_photo(ctx["flat_dir"], it["image_id"], ctx["fetched"])
    row = ctx["rows"][it["image_id"]]
    g = PP.image_geometry(row, img.size[0], img.size[1], ctx["ramps"])
    cdets = ctx["dets"]["canvas_level"][it["image_id"]]["dets"]
    flat_panel(ax[0][0], img, cdets, g, None, None, "canvas arm (a)")
    canvas_panel(ax[0][1], img, cdets, g, None, None)
    zoom_panel(ax[0][2], img, it)
    heads = [f"UNMATCHED, not yet rated: canvas arm {it['score']:.2f} matches no pool ramp "
             f"(gallery card {it['item']})\n{row['make']} {row['model']} "
             f"{row['capture_date']}, image {it['image_id']}",
             pair_row(fig, ax[1], "stretch", sel["stretch"], ctx, arm="stretch")]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, hspace=0.5,
                        wspace=0.04)
    row_headers(fig, ax, heads)
    legend(fig)
    files.append(save(fig, "richmond_unmatched_stretch.jpg", out))
    plt.close(fig)
    return files


def render_seoul(seoul_dir, names, out):
    import matplotlib.pyplot as plt
    from PIL import Image
    recs = {json.loads(line)["filename"]: json.loads(line)
            for line in open(SEOUL_DETS, encoding="utf-8") if line.strip()}
    fig, ax = plt.subplots(2, 2, figsize=(16, 12.4))
    for a, name in zip(ax.flat, names):
        rec = recs[name]
        im = Image.open(os.path.join(seoul_dir, os.path.splitext(name)[0] + ".jpg"))
        s = im.size[0] / rec["width"]
        panel(a, im, f"{os.path.splitext(name)[0]}: max score {rec['max_score']:.2f}")
        draw_dets(a, [(d["u"], d["v"], d["score"]) for d in rec["dets"]], sx=s, sy=s,
                  floor=False, extent=im.size)
        a.set_xlim(0, im.size[0])
        a.set_ylim(im.size[1], 0)
    fig.suptitle("Seoul pedestrian photos, canvas arm (a) at an assumed 70° FOV: "
                 f"detections ≥ {THR} ringed. Not rated; no presence judgment.",
                 fontsize=12, x=0.01, ha="left")
    fig.text(0.99, 0.005, SEOUL_CREDIT, ha="right", va="bottom", fontsize=9, color="#444")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, hspace=0.12,
                        wspace=0.03)
    f = save(fig, "seoul_strip.jpg", out)
    plt.close(fig)
    return f


def render_geometry(out):
    """Where a level 70° photo lands in the 360x180 canvas (plus the p5 / p95 FOVs)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.add_patch(plt.Rectangle((-180, -90), 360, 180, fc="#7c7468", ec="black", lw=1.2))
    ax.text(-172, 80, "canvas outside the photo: ImageNet mean grey (0 after "
                      "normalisation); peaks here are dropped", color="white", fontsize=10)
    styles = {70: dict(fc="#4f8fd0", alpha=0.85, ec="black", lw=1.5),
              48: dict(fc="none", ec="#ffe08a", lw=1.8, ls="--"),
              103: dict(fc="none", ec="#ffffff", lw=1.8, ls="--")}
    for hfov in (103, 70, 48):
        W, H = 4000, 3000
        cam = P.Camera(W, H, P.pinhole_focal_for_hfov(hfov, W, H))
        n = 200
        t = np.linspace(0, 1, n)
        us = np.concatenate([t * W, np.full(n, W), (1 - t) * W, np.zeros(n)]) - 0.5
        vs = np.concatenate([np.zeros(n), t * H, np.full(n, H), (1 - t) * H]) - 0.5
        rays = P.unproject_cam(cam, us, vs)
        x, y = P.ray_to_canvas_norm(rays)
        lon, lat = x * 360 - 180, 90 - y * 180
        ax.add_patch(plt.Polygon(np.c_[lon, lat], closed=True, **styles[hfov]))
        cols = (x.max() - x.min()) * P.CANVAS_W
        if hfov == 70:
            ax.annotate(f"70° x {cam.vfov_deg():.0f}° photo (4:3), level, heading on the "
                        f"centre column:\n{cols:.0f} of 4,096 canvas columns "
                        f"({cols / P.CANVAS_W:.0%} of the width)",
                        xy=(35, 20), xytext=(40, 50), fontsize=10.5, color="white",
                        arrowprops=dict(arrowstyle="->", color="white"))
        else:
            ax.text(lon.max() + 2, -lat.max() + (8 if hfov == 103 else -2),
                    f"{hfov}°" + (" (p95 FOV)" if hfov == 103 else " (p5 FOV)"),
                    color=styles[hfov]["ec"], fontsize=10)
    ax.axhline(0, color="white", lw=0.8, ls=":")
    ax.text(-178, 2, "horizon", color="white", fontsize=9)
    # trained scale
    ax.annotate("", xy=(-150, -60), xytext=(-60, -60),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
    ax.text(-105, -56, "90° = 1,024 px", color="white", ha="center", fontsize=10)
    ax.text(-172, -78, "Trained angular scale: 4,096 px / 360° = 11.4 px per degree, "
                       "the same in every pano RampNet saw.\nA 1.2 m ramp at 10 m spans "
                       f"{2 * math.degrees(math.atan(0.6 / 10)):.1f}° = "
                       f"{2 * math.degrees(math.atan(0.6 / 10)) * P.CANVAS_W / 360:.0f} px "
                       "across. At the photo's centre the canvas keeps that scale; the "
                       "stretch arm shows it 3.6-6.8x larger (p10-p90).",
            color="white", fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 45))
    ax.set_yticks(range(-90, 91, 45))
    ax.set_xlabel("longitude from the photo heading (°); canvas column = (lon + 180) / 360 x "
                  "4,096")
    ax.set_ylabel("latitude (°); canvas row = (90 - lat) / 180 x 2,048")
    ax.set_title("Canvas embed (arm a): a perspective photo reprojected into RampNet's "
                 "2048x4096 equirect input", loc="left", fontsize=12)
    fig.tight_layout()
    f = save(fig, "canvas_geometry.jpg", out)
    plt.close(fig)
    return f


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    c = candidates()
    want = [tuple(s.split(":", 1)) for s in args.misses.split(",")]
    misses = []
    for iid, ramp in want:
        m = [p for p in c["misses"] if p["image_id"] == iid and p["ramp"] == ramp]
        if not m:
            raise SystemExit(f"{iid} {ramp} is not a miss candidate")
        misses.append(m[0])
    items = json.load(open(FP_MANIFEST, encoding="utf-8"))["items"]
    sel = {"hits": c["hits"][:2], "misses": misses, "stretch": c["stretch"][0],
           "unmatched": next(i for i in items if i["item"] == UNMATCHED_CARD),
           "seoul": c["seoul"]}
    ctx = {"flat_dir": args.flat_dir, "pano_dir": args.pano_dir,
           "fetched": {r["image_id"]: r for r in PP.read_csv(PP.FETCHED_CSV)},
           "rows": {r["image_id"]: r for r in PP.read_csv(PP.IMAGES_CSV)},
           "ramps": PP.ramp_table(),
           "dets": {a: PP.load_dets(a) for a in ("canvas_level", "stretch")},
           "pano_recs": pano_dets()}
    os.makedirs(args.out, exist_ok=True)
    files = render_richmond(ctx, sel, args.out)
    files.append(render_seoul(args.seoul_dir, sel["seoul"], args.out))
    files.append(render_geometry(args.out))
    total = sum(f["bytes"] for f in files)
    print(f"total {total / 1e6:.2f} MB")
    if total > 8_000_000:
        raise SystemExit("figures exceed 8 MB in total")
    PP.write_json(os.path.join(args.out, "figures.json"), {
        "selection": sel, "misses_viewed": [
            {"image_id": i, "ramp": r, "seen": s} for i, r, s in MISSES_VIEWED],
        "files": files,
        "note": "sha256 of each JPEG as rendered here; matplotlib / Pillow versions can "
                "change the bytes, not the selection"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--show", type=int, default=12)
    r = sub.add_parser("render")
    r.add_argument("--flat-dir", required=True)
    r.add_argument("--pano-dir", required=True)
    r.add_argument("--seoul-dir", required=True)
    r.add_argument("--misses", default=DEFAULT_MISSES,
                   help="image_id:ramp,... of the misses to draw (see MISSES_VIEWED)")
    r.add_argument("--out", default=FIG_DIR)
    args = ap.parse_args(argv)
    {"candidates": cmd_candidates, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
