"""The two figures for docs/vlm_prompt_276.md, from the committed Step 5 JSON.

Both regenerate from ``analysis_out/prompt_opt_276/step5_results.json`` alone: no cache,
no network, no model. ``prompt_276_heldout.py`` regenerates that JSON from the detection
caches.

1. ``arm_vs_zs.png`` -- every arm minus zero-shot (confidence on, both at the dev_select-
   selected threshold) as macro dF1 with its paired pano-bootstrap 95% interval, one row
   per (model, arm), one panel per pool (reference, open, all 12). The primary endpoint,
   Flash alt on the reference pool, is the filled marker; everything else is exploratory.
   This is the null and the one exploratory gain in one picture.
2. ``dev_to_heldout.png`` -- what the dev_select gain of each arm became on held-out
   cities: x = dF1 on dev_select (65 panos, the number the gate and the "best of K" pick
   saw), y = macro dF1 on the reference pool with its interval. Points below the diagonal
   lost in transfer; the gate line at 0.02 is drawn.

**Colour.** One hue per model (blue for Flash, orange for Argon: validated categorical
slots 1 and 2, an adjacent pair that clears the CVD floor), plus the neutral inks. Arm is
carried by row position and marker shape, never by colour. Every plotted number is also
in ``step5_results.md`` and the tables of the doc, which discharges the palette's
contrast warning.

Usage::

    python scripts/analysis/prompt_276_figures.py            # writes docs/figures/prompt_276/
    python scripts/analysis/prompt_276_figures.py --out-dir /tmp/f276
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STEP5 = os.path.join(REPO, "analysis_out", "prompt_opt_276", "step5_results.json")
FIG_DIR = os.path.join(REPO, "docs", "figures", "prompt_276")

# references/palette.md: categorical slots 1 and 2, the neutral inks, the surface.
MODEL_COLOR = {"flash38": "#2a78d6", "argon": "#eb6834"}
MODEL_NAME = {"flash38": "Gemini 3.8 Flash", "argon": "Gemini 4 Argon"}
ARM_NAME = {"alt": "optimized (alt)", "para3": "paraphrase 3", "expert": "expert"}
ARM_MARKER = {"alt": "o", "para3": "s", "expert": "D"}
POOL_NAME = {"reference": "reference pool (7 GSV US cities)",
             "open": "open pool (Mapillary + Panoramax)",
             "all12": "all 12 held-out splits"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

ROWS = [("flash38", "alt"), ("flash38", "para3"), ("flash38", "expert"),
        ("argon", "para3"), ("argon", "expert")]
POOLS = ["reference", "open", "all12"]
GATE = 0.02
# Label offsets (points) in dev_to_heldout.png, where two Flash points sit 0.005 apart.
LABEL_NUDGE = {("flash38", "alt"): (-8, 16), ("flash38", "para3"): (8, -12)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)


def _tests(step5):
    out = {}
    for t in step5["tests"]:
        if t.get("vs") == "zs":
            out[(t["pool"], t["short"], t["arm"])] = t
    return out


def _dev_delta(step5):
    """dF1 on dev_select, each definition minus zero-shot, confidence on, both at the
    selected threshold -- the number the gate judged."""
    by = {(t["short"], t["arm"]): t["dev_select_f1"] for t in step5["thresholds"]
          if t["status"] == "selected"}
    return {k: v - by[(k[0], "zs")] for k, v in by.items() if k[1] != "zs"}


def _p_text(t):
    p = t["p_holm"]
    star = " *" if p < 0.05 else ""
    if p < 0.001:
        return f"p < .001{star}"
    if p < 0.01:
        return f"p {p:.3f}{star}".replace("p 0.", "p .")
    if p >= 0.995:
        return "p 1.0"
    return f"p {p:.2f}{star}".replace("p 0.", "p .")


def fig_arm_vs_zs(step5, path, plt):
    tests = _tests(step5)
    fig, axes = plt.subplots(1, len(POOLS), figsize=(11.4, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(ROWS)))[::-1]
    for ax, pool in zip(axes, POOLS):
        _style(ax)
        ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        for y, (short, arm) in zip(ys, ROWS):
            t = tests.get((pool, short, arm))
            if t is None:
                continue
            lo, hi = t["ci95"]
            d = t["macro_delta_f1"]
            primary = t.get("family") == "primary"
            c = MODEL_COLOR[short]
            ax.plot([lo, hi], [y, y], color=c, lw=2.0, solid_capstyle="round", zorder=2)
            ax.plot(d, y, marker=ARM_MARKER[arm], ms=8.5 if primary else 7.5,
                    color=c if primary else SURFACE, mec=c, mew=1.8, zorder=3)
            ax.text(hi + 0.004, y, f"{d:+.3f}  {_p_text(t)}",
                    fontsize=7.6, color=INK_SECONDARY, va="center", ha="left")
        ax.set_xlim(-0.075, 0.16)
        ax.set_title(POOL_NAME[pool], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("macro ΔF1, arm − zero-shot", fontsize=8.5, color=INK_SECONDARY)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{MODEL_NAME[s]}: {ARM_NAME[a]}" for s, a in ROWS],
                            fontsize=8.6, color=INK)
    fig.suptitle("Held-out gain of each prompt arm over zero-shot (#276 Step 5): the primary "
                 "endpoint, Flash's optimized definition, is null",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Paired pano bootstrap, 2,000 draws, 95% interval; p is "
             "Holm-corrected within each (model, pool) family, * p < .05. Filled marker = "
             "pre-registered primary endpoint; every other row is exploratory. Both arms "
             "of each row use elicited confidence at the dev_select-selected threshold.",
             fontsize=7.2, color=INK_MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_dev_to_heldout(step5, path, plt):
    tests = _tests(step5)
    dev = _dev_delta(step5)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    lim = (-0.06, 0.075)
    ax.plot(lim, lim, color=BASELINE, lw=1.0, ls="--", zorder=1)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    ax.axvline(GATE, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.text(GATE + 0.002, lim[0] + 0.004, "gate: ≥ 0.02 on dev_select", fontsize=7.4,
            color=INK_MUTED, ha="left", va="bottom", rotation=90)
    for short, arm in ROWS:
        t = tests.get(("reference", short, arm))
        if t is None or (short, arm) not in dev:
            continue
        x = dev[(short, arm)]
        y = t["macro_delta_f1"]
        lo, hi = t["ci95"]
        c = MODEL_COLOR[short]
        primary = t.get("family") == "primary"
        ax.plot([x, x], [lo, hi], color=c, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot(x, y, marker=ARM_MARKER[arm], ms=9 if primary else 8,
                color=c if primary else SURFACE, mec=c, mew=1.8, zorder=3)
        dx, dy = LABEL_NUDGE.get((short, arm), (7, 4))
        ax.annotate(f"{MODEL_NAME[short].split()[-1]} {ARM_NAME[arm]}", (x, y),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("ΔF1 on dev_select (65 panos), arm − zero-shot", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_ylabel("macro ΔF1 on the reference pool, arm − zero-shot", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_title("What a dev_select gain became on held-out cities (#276)", fontsize=11,
                 color=INK, loc="left", pad=30)
    ax.text(0, 1.015, "Below the dashed diagonal the gain shrank in transfer. The one arm "
            "that passed the gate, Flash's optimized\ndefinition (filled), lost all of it.",
            transform=ax.transAxes, fontsize=7.6, color=INK_SECONDARY, ha="left", va="bottom")
    fig.text(0.01, 0.005, "Bars: paired pano-bootstrap 95% interval of the held-out Δ. "
             "Both axes: confidence on, dev_select-selected threshold.\nBlue = Flash, "
             "orange = Argon.",
             fontsize=7.2, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


FIGURES = {"arm_vs_zs.png": fig_arm_vs_zs, "dev_to_heldout.png": fig_dev_to_heldout}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step5", default=STEP5)
    ap.add_argument("--out-dir", default=FIG_DIR)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(args.step5, encoding="utf-8") as f:
        step5 = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, fn in FIGURES.items():
        path = os.path.join(args.out_dir, name)
        fn(step5, path, plt)
        print(f"wrote {os.path.relpath(path, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
