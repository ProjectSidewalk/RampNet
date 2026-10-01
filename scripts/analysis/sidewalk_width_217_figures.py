"""Example figures for the sidewalk-width estimator (#217, arm 1).

Two figures, both from the held-out half B only:

* ``examples_contact_sheet.jpg`` -- eight photos spanning the clear-width error
  distribution, each with the walkable mask, the detected clear-span edges, the
  measurement band, the vanishing-point horizon and a caption.
* ``diagnostics.jpg`` -- predicted vs GT clear width with the 1.2 / 1.5 m class lines,
  and MAE per GT-width bin for clear and total width.

Selection (``select_examples``) reads only committed files (``results.json`` and the GT
table), so the same eight photos come out of every run. The rule, on half B, clear width:

    best            smallest |error|
    median          signed error closest to the median signed error
    worst over      largest signed error
    worst under     most negative signed error
    true <1.2 m     flagged <1.2 m and GT <1.2 m: the one with the median |error|
    false <1.2 m    flagged <1.2 m but GT >=1.2 m: the one with the largest GT
    no estimate     no clear estimate: the first by filename
    2nd worst over  the second-largest signed error (fills the eighth slot)

Ties go to the lower filename; a photo already picked is skipped and the next in the
same order is taken. The overlays are recomputed from the Mask2Former label maps with
the functions in ``sidewalk_width_217.py`` and the recomputed width is asserted equal to
the committed per-image estimate, so the drawing is the measurement, not a sketch of it.

Usage::

    # which photos to fetch (no images needed)
    python scripts/analysis/sidewalk_width_217_figures.py select

    # render; --images holds the Seoul JPEGs, --seg the label maps (makelab2:
    # /homes/gws/jonf/seoul_sidewalk/images and /homes/gws/jonf/sw217_seg). Only the
    # selected photos need to be present.
    python scripts/analysis/sidewalk_width_217_figures.py render \
        --images IMGDIR --seg SEGDIR --out docs/figures/sidewalk_width_217
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sidewalk_width_217 as sw  # noqa: E402

RESULTS = os.path.join(sw.REPO, "analysis_out", "sidewalk_width_217", "results.json")
JPEG_QUALITY = 85
PANEL_W = 720                 # each contact-sheet photo is drawn at this width (px)
BLUE, ORANGE = "#2a78d6", "#eb6834"   # dataviz default categorical slots 1 and 2
TEXT, TEXT2 = "#0b0b0b", "#52514e"


def load(results_path=RESULTS):
    """(names, half, gt, clear, total, cfg).

    clear/total are {filename: width or nan}; cfg is {"clear": config, "total": config},
    the chosen config of each measure from ``results.json``."""
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)
    gt = sw._read_gt()
    names = sorted(gt)
    G = {n: float(gt[n]["width"]) for n in names}
    est = {m: {n: (np.nan if res["measures"][m]["per_image"][n] is None
                   else res["measures"][m]["per_image"][n]) for n in names}
           for m in ("clear", "total")}
    cfg = {m: res["measures"][m]["config"] for m in ("clear", "total")}
    return names, res["half"], G, est["clear"], est["total"], cfg


def select_examples(names, half, G, E):
    """[(slot label, filename)] by the rule in the module docstring."""
    B = [n for n in names if half[n] == "B"]
    est = [n for n in B if np.isfinite(E[n])]
    err = {n: E[n] - G[n] for n in est}
    med = float(np.median(list(err.values())))
    tp = [n for n in est if E[n] < 1.2 and G[n] < 1.2]
    tp_by_abs = sorted(tp, key=lambda n: (abs(err[n]), n))
    rules = [
        ("best", sorted(est, key=lambda n: (abs(err[n]), n))),
        ("median", sorted(est, key=lambda n: (abs(err[n] - med), n))),
        ("worst over", sorted(est, key=lambda n: (-err[n], n))),
        ("worst under", sorted(est, key=lambda n: (err[n], n))),
        ("true <1.2 m", tp_by_abs[len(tp_by_abs) // 2:] + tp_by_abs[:len(tp_by_abs) // 2]),
        ("false <1.2 m", sorted([n for n in est if E[n] < 1.2 and G[n] >= 1.2],
                                key=lambda n: (-G[n], n))),
        ("no estimate", sorted(n for n in B if not np.isfinite(E[n]))),
        ("2nd worst over", sorted(est, key=lambda n: (-err[n], n))[1:]),
    ]
    picked, out = set(), []
    for label, order in rules:
        n = next(x for x in order if x not in picked)
        picked.add(n)
        out.append((label, n))
    return out


def find_photo(images_dir, name):
    stem = os.path.splitext(name)[0].lower()
    for dirpath, _, files in os.walk(images_dir):
        for fn in files:
            if os.path.splitext(fn)[0].lower() == stem and not fn.startswith("._"):
                return os.path.join(dirpath, fn)
    raise FileNotFoundError(f"{name} not under {images_dir}")


def overlay_geometry(lab, cfg):
    """Everything drawn on one photo, recomputed with the scoring code.

    Returns a dict: walk mask, clear left/right edges, used rows, band rows, pitch (rad or
    None), horizon row, VP edge lines (a, b) or None, the clear width, focal length, and
    ``band_z`` = (nearest, farthest depth actually measured, number of band rows).

    Only the ``vp`` horizon is drawn: the chosen clear config uses it, and ``level`` /
    ``vp_prior`` would need a different pitch here."""
    assert cfg["horizon"] == "vp", (
        f"overlay_geometry draws the VP horizon only; config has horizon={cfg['horizon']!r}")
    Hh, Ww = lab.shape
    cx, cy = (Ww - 1) / 2.0, (Hh - 1) / 2.0
    f = sw.focal_px(sw.F35_MM, Ww, Hh)
    spans = sw.image_spans(lab)
    vps = sw.image_vp(spans, f, lab.shape)
    vk = (cfg["walk"], cfg["obst"], cfg["mark"])
    pitch = vps[vk]
    left, right = spans[vk + ("clear",)]
    out = {"mask": np.isin(lab, sw.WALK_SETS[cfg["walk"]]), "left": left, "right": right,
           "pitch": pitch, "f": f, "cy": cy, "width": np.nan, "used": np.zeros(Hh, bool),
           "band": np.zeros(Hh, bool), "lines": None}
    # the fitted edge lines the VP came from (total spans, same rows as image_vp)
    tl, tr = spans[vk + ("total",)]
    below = np.arange(int(math.ceil(cy + f * sw.CAMERA_H_M / sw.Z_MAX_M)), Hh)
    rows = below[tl[below] >= 0]
    if rows.size:
        aL, bL, nL = sw.robust_line(rows, tl[rows])
        aR, bR, nR = sw.robust_line(rows, tr[rows])
        if min(nL, nR) >= 20:
            out["lines"] = ((aL, bL), (aR, bR))
    if pitch is None:
        return out
    Z, Wd, _ = sw.ground_widths(left, right, f, cx, cy, sw.CAMERA_H_M, pitch)
    m = np.isfinite(Z) & np.isfinite(Wd)
    out["used"] = m
    mz = m & (Z >= cfg["zmin"])
    if mz.any():
        z0 = Z[mz].min()
        out["band"] = mz & (Z <= z0 + cfg["band"])
        zb = Z[out["band"]]
        out["band_z"] = (float(zb.min()), float(zb.max()), int(out["band"].sum()))
    out["width"] = sw.band_stat(Z, Wd, cfg["zmin"], cfg["band"], cfg["stat"])
    out["horizon"] = sw.horizon_row(f, cy, pitch)
    return out


def draw_panel(ax, photo, g, title, caption):
    from matplotlib.patches import Polygon
    Hh, Ww = g["mask"].shape
    ax.imshow(photo, extent=(-0.5, Ww - 0.5, Hh - 0.5, -0.5))
    tint = np.zeros((Hh, Ww, 4))
    tint[g["mask"]] = (0.10, 0.85, 0.35, 0.33)
    ax.imshow(tint, extent=(-0.5, Ww - 0.5, Hh - 0.5, -0.5), interpolation="nearest")
    rows = np.arange(Hh)
    u = g["used"]
    if u.any():
        ax.plot(g["left"][u] - 0.5, rows[u], ".", ms=1.2, color="#00e5ff")
        ax.plot(g["right"][u] + 0.5, rows[u], ".", ms=1.2, color="#ff3df5")
    if g["band"].any():
        br = rows[g["band"]]
        poly = np.r_[np.c_[g["left"][br] - 0.5, br], np.c_[g["right"][br][::-1] + 0.5, br[::-1]]]
        ax.add_patch(Polygon(poly, closed=True, fc=(1, 0.85, 0, 0.45), ec="#ffd400", lw=1.2))
    if g["lines"] is not None:
        top = g.get("horizon", g["cy"])
        vv = np.array([max(top, 0), Hh - 1])
        for a, b in g["lines"]:
            ax.plot(a + b * vv, vv, "--", color="white", lw=0.9, alpha=0.9)
    if g["pitch"] is not None:
        ax.axhline(g["horizon"], color="#ffd400", lw=1.4, ls="-")
        ax.text(8, g["horizon"] - 8, f"VP horizon (pitch {pitch_deg(g['pitch']):+.1f}°)",
                color="#ffd400", fontsize=7, va="bottom",
                bbox=dict(fc=(0, 0, 0, 0.55), ec="none", pad=1.5))
    else:
        ax.axhline(g["cy"], color="#bbbbbb", lw=1.0, ls=":")
        ax.text(8, g["cy"] - 8, "no vanishing point found: no clear estimate", color="white",
                fontsize=7, va="bottom", bbox=dict(fc=(0, 0, 0, 0.55), ec="none", pad=1.5))
    ax.set_xlim(-0.5, Ww - 0.5)
    ax.set_ylim(Hh - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=TEXT, loc="left", fontweight="bold")
    ax.text(0, -0.02, caption, transform=ax.transAxes, fontsize=8, color=TEXT, va="top",
            family="monospace")


def pitch_deg(p):
    """Pitch in degrees rounded to 0.1, without a negative zero (-0.04 -> 0.0)."""
    return round(math.degrees(p), 1) + 0.0


def fmt(x):
    return "  --  " if not np.isfinite(x) else f"{x:5.2f} m"


def fmt_err(e):
    """Signed error in metres to 0.01, without a negative zero (-0.001 -> "+0.00 m")."""
    return "  --" if not np.isfinite(e) else f"{round(e, 2) + 0.0:+.2f} m"


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    names, half, G, C, T, cfg = load(args.results)
    picks = select_examples(names, half, G, C)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- contact sheet ----------------
    ncol = 4
    nrow = math.ceil(len(picks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.3 * nrow), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    for ax, (label, n) in zip(axes.ravel(), picks):
        lab = np.array(Image.open(os.path.join(args.seg, os.path.splitext(n)[0] + ".png")))
        g = overlay_geometry(lab, cfg["clear"])
        want = C[n]
        if np.isfinite(want) or np.isfinite(g["width"]):
            assert abs(g["width"] - want) < 5e-3, (n, g["width"], want)
        photo = Image.open(find_photo(args.images, n)).convert("RGB")
        photo = photo.resize((lab.shape[1] // 2, lab.shape[0] // 2), Image.LANCZOS)
        err = C[n] - G[n]
        cap = (f"{n}  (CC0)\n"
               f"GT {G[n]:.2f} m   clear {fmt(C[n])}   total {fmt(T[n])}\n"
               f"clear error {fmt_err(err)}")
        if "band_z" in g:
            cap += f"   band {g['band_z'][0]:.2f}-{g['band_z'][1]:.2f} m ({g['band_z'][2]} rows)"
        draw_panel(ax, np.asarray(photo), g, label, cap)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("Sidewalk width from one photo, held-out half B (#217): green = walkable "
                 "mask, cyan/magenta = clear-span edges per row, yellow band = rows measured, "
                 "yellow line = VP horizon, dashed = fitted edges", fontsize=10, color=TEXT,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=4.6, w_pad=0.6)
    p1 = os.path.join(args.out, "examples_contact_sheet.jpg")
    fig.savefig(p1, format="jpeg", pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
    plt.close(fig)

    # ---------------- diagnostics ----------------
    B = [n for n in names if half[n] == "B"]
    gb = np.array([G[n] for n in B])
    cb = np.array([C[n] for n in B])
    tb = np.array([T[n] for n in B])
    ok = np.isfinite(cb)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=args.dpi,
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    lim = float(np.ceil(max(gb.max(), np.nanmax(cb)) + 0.3))
    for t in sw.THRESH_M:
        a1.axvline(t, color=TEXT2, lw=0.8, ls=":")
        a1.axhline(t, color=TEXT2, lw=0.8, ls=":")
    a1.plot([0, lim], [0, lim], color=TEXT2, lw=1)
    a1.scatter(gb[ok], cb[ok], s=22, color=BLUE, alpha=0.75, edgecolor="white", lw=0.6,
               label=f"clear width, half B ({ok.sum()} of {len(B)} estimated)")
    # the four picks inside the dense low-left cloud get a leader line from open space,
    # stacked in the same order as their points so the lines do not cross
    leader = {"best": (8.3, 5.0), "median": (8.3, 4.2), "true <1.2 m": (8.3, 3.4),
              "false <1.2 m": (8.3, 2.6)}
    for label, n in picks:
        if np.isfinite(C[n]):
            if label in leader:
                a1.annotate(label, (G[n], C[n]), xytext=leader[label], textcoords="data",
                            fontsize=7.5, color=TEXT, va="center",
                            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.6,
                                            shrinkA=2, shrinkB=4))
            else:
                a1.annotate(label, (G[n], C[n]), xytext=(6, 4), textcoords="offset points",
                            fontsize=7.5, color=TEXT)
            a1.scatter([G[n]], [C[n]], s=46, facecolor="none", edgecolor=TEXT, lw=1.1)
    # 1.2 m left of its line and 1.5 m right of its own, so neither line runs through text
    a1.text(1.2, lim - 0.15, "1.2 m ", fontsize=7.5, color=TEXT2, va="top", ha="right")
    a1.text(1.5, lim - 0.15, " 1.5 m", fontsize=7.5, color=TEXT2, va="top", ha="left")
    a1.set_xlim(0, lim)
    a1.set_ylim(0, lim)
    a1.set_aspect("equal")
    a1.set_xlabel("GT clear width, laser (m)")
    a1.set_ylabel("estimated clear width (m)")
    a1.set_title("Estimated vs GT (circled = contact-sheet photos)", fontsize=10, loc="left")
    a1.legend(loc="lower right", fontsize=8, frameon=False)

    bins = ((0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 99.0))
    xs = np.arange(len(bins))
    wbar = 0.38
    for j, (lab_, e, col) in enumerate((("clear", cb, BLUE), ("total", tb, ORANGE))):
        maes, ns = [], []
        for lo, hi in bins:
            m = (gb >= lo) & (gb < hi) & np.isfinite(e)
            maes.append(float(np.abs(e[m] - gb[m]).mean()))
            ns.append(int(m.sum()))
        x = xs + (j - 0.5) * (wbar + 0.02)
        a2.bar(x, maes, width=wbar, color=col, label=f"{lab_} width")
        for xi, v, k in zip(x, maes, ns):
            a2.text(xi, v + 0.02, f"{v:.2f}\nn={k}", ha="center", va="bottom", fontsize=7.5,
                    color=TEXT)
    a2.set_xticks(xs, ["<1.5 m", "1.5-3 m", "3-5 m", ">=5 m"])
    a2.set_xlabel("GT width bin")
    a2.set_ylabel("mean absolute error (m)")
    a2.set_title("MAE by GT width, half B", fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    a2.set_ylim(0, max(a2.get_ylim()[1], 1.9))
    fig.tight_layout()
    p2 = os.path.join(args.out, "diagnostics.jpg")
    fig.savefig(p2, format="jpeg", pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
    plt.close(fig)
    for p in (p1, p2):
        print(f"{p}  {os.path.getsize(p) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=RESULTS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    p = sub.add_parser("render")
    p.add_argument("--images", required=True)
    p.add_argument("--seg", required=True)
    p.add_argument("--out", default=os.path.join(sw.REPO, "docs", "figures",
                                                 "sidewalk_width_217"))
    p.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    if args.cmd == "select":
        names, half, G, C, T, _ = load(args.results)
        for label, n in select_examples(names, half, G, C):
            print(f"{label:15s} {n:16s} GT {G[n]:.2f}  clear {C[n]:.3f}  total {T[n]:.3f}")
    else:
        render(args)


if __name__ == "__main__":
    main()
