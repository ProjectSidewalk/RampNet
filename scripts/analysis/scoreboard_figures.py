"""The four figures for docs/model_scoreboard.md — the findings that read faster as a picture.

Called by ``scoreboard.py``; importable on its own for iterating on a figure without
re-scoring. Matplotlib is imported inside ``render_all`` so ``--no-figures`` needs no
plotting stack at all.

**Colour.** One hue does all the work here: the validated categorical slot 1
(``#2a78d6``) plus the palette's neutral inks, with the sequential blue ramp for the
heatmap. That is a deliberate reduction, not laziness. The natural design — a hue per
model class — cannot ship: three of the four figures are scatter/matrix forms, which are
scored on the **all-pairs** pairlist, and five categorical slots fail it outright
(``validate_palette.js``: magenta↔orange normal-vision ΔE 12.9, below the hard floor of
15). A normal-vision FAIL is the one result secondary encoding does not excuse, so the
documented remedy is to cut series or facet rather than to add a legend and hope. Model
class is therefore carried by **position** (bar order, group headers) and **marker
shape** — channels with no CVD failure mode — and colour is freed to carry emphasis.

The palette's contrast WARN (slot 1 sits above 3:1, but the light neutrals do not) is
discharged the way ``plot_operating_point.py`` discharges it: every plotted number is
also in a table, in ``docs/model_scoreboard.md`` and ``analysis_out/scoreboard.json``.
"""
import os

# Palette: validated categorical slot 1, the sequential blue ramp, and the neutral inks
# (references/palette.md). Same values plot_operating_point.py uses, so the two figures
# families sit together in one document without looking like two documents.
BLUE = "#2a78d6"
BLUE_DEEP = "#184f95"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
MUTED_FILL = "#c9c8c2"      # the un-emphasised bars: present, recessive, not competing
# Sequential blue, light -> dark (palette.md steps 100..700). Monotone in lightness, one
# hue: the documented form for a magnitude encoding.
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Class -> marker, the CVD-free channel that replaces a per-class hue in the scatters.
CLASS_MARKER = {
    "purpose-trained": "o",
    "supervised": "s",
    "supervised-transfer": "P",
    "chat-vlm": "^",
    "pointing": "D",
    "open-vocab": "v",
    "unclassified": "X",
}


# Label placement is deterministic, not automatic: a solver would move labels whenever
# a number moved, and these figures are committed artifacts that get diffed. Each entry
# is (dx, dy) in points plus the horizontal anchor, and the default is down-and-right.
# Only points that would collide need one.
PR_NUDGE = {
    "y11x_pano_h200": (11, 5, "left"),
    "y11l_pano": (11, -8, "left"),
    "google/owlv2-large-patch14-ensemble": (-9, 11, "right"),
    "IDEA-Research/grounding-dino-base": (-9, -2, "right"),
}
GEN_NUDGE = {
    "rampnet": (-14, -13, "right"),   # up-left runs into the diagonal caption
    "y11l_pano": (11, 6, "left"),
    "y11x_pano_h200": (11, -7, "left"),
    "y26_pano": (-11, -10, "right"),
    "gemini-3.7-flash": (11, 3, "left"),
    "Qwen/Qwen3-VL-32B-Instruct": (11, 6, "left"),
    "Qwen/Qwen3-VL-8B-Instruct": (11, -8, "left"),
    "google/owlv2-large-patch14-ensemble": (12, 7, "left"),
    "IDEA-Research/grounding-dino-base": (12, -9, "left"),
}


def _titles(ax, title, subtitle):
    """Title above, subtitle below it, both above the axes and left-aligned.

    set_title() alone puts the title flush to the axes, so a subtitle placed in axes
    coordinates lands ABOVE it and the two read in the wrong order.
    """
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=30)
    ax.text(0.0, 1.018, subtitle, transform=ax.transAxes, fontsize=8.6,
            color=INK_SECONDARY, va="bottom")


def _style(ax):
    """Recessive chrome: hairline solid grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)


def _seq_color(value, vmax):
    """A step of the sequential ramp for ``value`` in [0, vmax]."""
    if value is None or vmax <= 0:
        return SURFACE
    idx = int(round((value / vmax) * (len(SEQ) - 1)))
    return SEQ[max(0, min(len(SEQ) - 1, idx))]


def fig_headline(result, path, plt):
    """Pooled F1 per model, sorted — the board, and the size of the lead.

    Emphasis rather than a hue per class: the story here is one number (how far clear
    RampNet is), which is the case the anti-pattern list says to solve with highlight-one
    rather than with a full categorical palette.
    """
    from scoreboard import CLASS_LABEL

    # Complete rows only: figures 1, 2 and 4 all plot a pooled mean, and a one-city mean
    # drawn beside a seven-city one is the comparison the partial table exists to avoid.
    # Those legs appear in the matrix (fig 3), which is per split and needs no averaging.
    models = [m for m in result["models"] if m["complete"] and m["f1"] is not None]
    models.sort(key=lambda m: m["f1"])
    # Class rides in the tick label rather than in a second text column: drawn
    # separately it lands on top of the tick labels, because matplotlib sizes the left
    # margin from the ticks alone.
    labels = [f"{m['display']}  ·  {CLASS_LABEL[m['class']]}" for m in models]
    values = [m["f1"] for m in models]
    is_ref = [m["model"] == "rampnet" for m in models]

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.barh(range(len(models)), values, height=0.62, zorder=3,
                   color=[BLUE if r else MUTED_FILL for r in is_ref])
    for i, (bar, m) in enumerate(zip(bars, models)):
        ax.text(bar.get_width() + 0.012, i, f"{m['f1']:.3f}", va="center",
                fontsize=8.6, color=INK if is_ref[i] else INK_SECONDARY,
                fontweight="bold" if is_ref[i] else "normal", zorder=4)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9)
    for tick, ref in zip(ax.get_yticklabels(), is_ref):
        tick.set_color(INK if ref else INK_SECONDARY)
        if ref:
            tick.set_fontweight("bold")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xlabel("F1, macro-mean over the seven pooled US city splits",
                  fontsize=9.5, color=INK_SECONDARY)

    # The lead goes in the subtitle rather than into an annotated arrow: the gap between
    # the top two bars is 0.38 of a row, which cannot hold a rule and a caption without
    # colliding with one of them.
    runner_up = max((m for m in models if m["model"] != "rampnet"), key=lambda m: m["f1"])
    ref_f1 = next(m["f1"] for m in models if m["model"] == "rampnet")
    _titles(ax, "RampNet leads every off-the-shelf and supervised baseline tested",
            f"+{ref_f1 - runner_up['f1']:.3f} F1 clear of the best challenger "
            f"({runner_up['display']}, {runner_up['f1']:.3f}).")
    # Two lines: at 7.4pt one line of this runs past the right edge and gets clipped.
    n_partial = len([m for m in result["models"] if not m["complete"]])
    fig.text(0.008, 0.030,
             "Operating points differ by model class: RampNet 0.55, YOLO 0.25, "
             "open-vocab 0.05 floor, chat VLMs emit no score.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.text(0.008, 0.008,
             (f"{n_partial} single-split legs are reported per split instead — "
              if n_partial else "") + "see docs/model_scoreboard.md.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.052, 1, 1))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fig_precision_recall(result, path, plt):
    """Where each model sits in the P/R plane — the shape of how it fails, not just how far.

    F1 iso-contours are the reason this beats the bar chart for diagnosis: two models on
    the same contour score identically and are nothing alike.
    """
    from matplotlib.lines import Line2D
    import numpy as np

    from scoreboard import CLASS_LABEL

    models = [m for m in result["models"]
              if m["complete"] and m["precision"] is not None
              and m["recall"] is not None]

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    grid = np.linspace(0.001, 1.0, 400)
    rr, pp = np.meshgrid(grid, grid)
    f1 = 2 * rr * pp / (rr + pp)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(rr, pp, f1, levels=levels, colors=GRID, linewidths=0.9, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"F1 {v:.1f}", fontsize=6.8, colors=INK_MUTED)

    for m in models:
        ref = m["model"] == "rampnet"
        ax.plot(m["recall"], m["precision"], CLASS_MARKER[m["class"]],
                ms=13 if ref else 9.5, color=BLUE,
                alpha=1.0 if ref else 0.72, mec=SURFACE, mew=2.0, zorder=5)
        dx, dy, ha = PR_NUDGE.get(m["model"], (11, -3.5, "left"))
        ax.annotate(m["display"], (m["recall"], m["precision"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=8.4, color=INK if ref else INK_SECONDARY,
                    fontweight="bold" if ref else "normal", zorder=6)

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("recall", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("precision", fontsize=9.5, color=INK_SECONDARY)
    _titles(ax, "The same F1 hides opposite failures",
            "Macro-mean over the seven pooled US city splits. Marker shape is model class.")

    seen, handles = set(), []
    for m in models:
        if m["class"] in seen:
            continue
        seen.add(m["class"])
        handles.append(Line2D([], [], ls="", marker=CLASS_MARKER[m["class"]], ms=8,
                              color=BLUE, mec=SURFACE, mew=1.6,
                              label=CLASS_LABEL[m["class"]]))
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.2,
                    labelcolor=INK_SECONDARY, handletextpad=0.5,
                    bbox_to_anchor=(0.005, 0.005))
    leg.set_zorder(7)

    fig.text(0.008, 0.012,
             "Open-vocabulary detectors buy their recall with ~65–72 false positives per "
             "panorama; Qwen-32B buys precision by barely firing.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.028, 1, 0.96))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fig_by_split(result, path, plt):
    """F1 for every (model, split) cell — consistency across cities, at a glance.

    Sequential single-hue ramp (magnitude), values printed in every cell so the figure is
    also its own table, and the three held-out splits pushed to the right of a gap so
    they are never read as part of the pooled headline.
    """
    from scoreboard import US_SPLITS

    splits = list(result["all_splits"])
    held = [s for s in splits if s not in US_SPLITS]
    ordered = list(US_SPLITS) + held
    models = result["models"]
    per = result["per_split"]

    # A one-column gutter between the pooled splits and the held-out ones.
    gap_at = len(US_SPLITS)
    ncols = len(ordered) + 1

    # Height tracks the row count so adding a leg does not silently squash the cells.
    fig, ax = plt.subplots(figsize=(11.6, 1.5 + 0.42 * len(models)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    vmax = 1.0
    for row, m in enumerate(models):
        for i, split in enumerate(ordered):
            col = i if i < gap_at else i + 1
            cell = per[m["model"]].get(split)
            value = cell["f1"] if cell else None
            color = _seq_color(value, vmax) if value is not None else "#f4f3f0"
            ax.add_patch(plt.Rectangle((col + 0.03, row + 0.03), 0.94, 0.94,
                                       facecolor=color, edgecolor="none", zorder=2))
            text = "—" if value is None else f"{value:.2f}"
            # White ink only where the fill is dark enough to carry it.
            ink = "#ffffff" if (value is not None and value >= 0.62) else INK
            ax.text(col + 0.5, row + 0.5, text, ha="center", va="center",
                    fontsize=8.2, color=ink if value is not None else INK_MUTED,
                    zorder=3)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, len(models))
    ax.invert_yaxis()
    ax.set_xticks([(i if i < gap_at else i + 1) + 0.5 for i in range(len(ordered))])
    ax.set_xticklabels([s if s in US_SPLITS else f"{s} †" for s in ordered],
                       rotation=38, ha="right", fontsize=8.2, color=INK_SECONDARY)
    ax.set_yticks([r + 0.5 for r in range(len(models))])
    ax.set_yticklabels([m["display"] for m in models], fontsize=8.8,
                       color=INK_SECONDARY)
    for tick, m in zip(ax.get_yticklabels(), models):
        if m["model"] == "rampnet":
            tick.set_color(INK)
            tick.set_fontweight("bold")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    # No gutter caption: the gap plus the daggered column headers already carry it, and a
    # rotated label there runs into the title band.
    #
    # The claim is bounded on purpose. RampNet does NOT have the flattest row outright --
    # OWLv2 (0.028) and Grounding DINO (0.039) are flatter, by being pinned near zero in
    # every city. Consistency only means something above the floor, so the comparison is
    # drawn against the models that clear it, and the threshold is stated rather than
    # implied.
    working = [m for m in models
               if m["complete"] and m["f1"] is not None and m["f1"] >= 0.1]
    spreads = [m["f1_max"] - m["f1_min"] for m in working if m["model"] != "rampnet"]
    ref = next(m for m in models if m["model"] == "rampnet")
    _titles(ax, "F1 by model and split",
            f"RampNet varies by {ref['f1_max'] - ref['f1_min']:.2f} across the seven "
            f"pooled cities; every challenger above F1 0.1 varies by "
            f"{min(spreads):.2f}–{max(spreads):.2f}.")
    fig.text(0.008, 0.012,
             "† held out of the pooled headline: budapest (single-rater GT at low reviewer "
             "confidence), sao_paulo (non-US), manual_gold (in-distribution reference).",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fig_generalization(result, path, plt):
    """In-distribution F1 against deployed F1 — what the architecture buys.

    The diagonal is "no domain advantage". A zero-shot model has no training distribution
    to be inside, so it lands on or near the line; a model trained on the RampNet dataset
    starts above the line on manual_gold and falls by however much it fails to generalize.
    The vertical drop to the diagonal is that penalty, and it is the whole #51 ablation in
    one distance.
    """
    from matplotlib.lines import Line2D

    from scoreboard import CLASS_LABEL

    models = [m for m in result["models"]
              if m["complete"] and m["f1"] is not None
              and m["manual_gold_f1"] is not None]

    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, zorder=1)
    ax.text(0.965, 0.985, "no domain advantage", rotation=45, rotation_mode="anchor",
            ha="right", va="bottom", fontsize=7.8, color=INK_MUTED, zorder=2)

    for m in models:
        ref = m["model"] == "rampnet"
        x, y = m["manual_gold_f1"], m["f1"]
        ax.plot([x, x], [y, x], color=INK_MUTED, lw=0.9, ls=(0, (2, 2)), zorder=3)
        ax.plot(x, y, CLASS_MARKER[m["class"]], ms=13 if ref else 9.5, color=BLUE,
                alpha=1.0 if ref else 0.72, mec=SURFACE, mew=2.0, zorder=5)
        # Signed: deployed minus in-distribution. A model ABOVE the line gains, and
        # hardcoding a minus rendered those as "--0.05".
        dx, dy, ha = GEN_NUDGE.get(m["model"], (11, -3.5, "left"))
        ax.annotate(f"{m['display']}  {y - x:+.2f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.4,
                    color=INK if ref else INK_SECONDARY,
                    fontweight="bold" if ref else "normal", zorder=6)

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("in-distribution F1  (manual_gold, 1,000 GSV panoramas)",
                  fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("deployed F1  (macro-mean, seven US cities)",
                  fontsize=9.5, color=INK_SECONDARY)
    _titles(ax, "The drop from in-distribution to deployed is what generalization costs",
            "Labels give deployed F1 minus in-distribution F1. Marker shape is model class.")

    seen, handles = set(), []
    for m in models:
        if m["class"] in seen:
            continue
        seen.add(m["class"])
        handles.append(Line2D([], [], ls="", marker=CLASS_MARKER[m["class"]], ms=8,
                              color=BLUE, mec=SURFACE, mew=1.6,
                              label=CLASS_LABEL[m["class"]]))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.2,
              labelcolor=INK_SECONDARY, handletextpad=0.5)

    fig.text(0.008, 0.012,
             "Both Gemini legs with city numbers are absent: their manual_gold detections "
             "were never published (docs/model_scoreboard.md, 'What is missing').",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.028, 1, 0.96))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# Four validated categorical hues for the curve families (validate_palette.js, adjacent
# pairlist -- the documented one for lines: ALL PASS, worst adjacent CVD dE 9.1). The three
# YOLO arms share one hue and separate by dash: they are one family, and spending three
# slots on them would push the set past what the all-pairs floors allow.
CURVE_COLOR = {
    "rampnet": "#2a78d6",
    "y11l_pano": "#eb6834",
    "y11x_pano_h200": "#eb6834",
    "y26_pano": "#eb6834",
    "google/owlv2-large-patch14-ensemble": "#1baf7a",
    "IDEA-Research/grounding-dino-base": "#eda100",
}
CURVE_DASH = {
    "y11l_pano": (0, (5, 2)),
    "y11x_pano_h200": (0, (1, 1.6)),
    "y26_pano": (0, (4, 1.4, 1, 1.4)),
}


def _decimate(xs, ys, keep=1500):
    """Thin a PR curve for plotting; the open detectors carry ~60k points each."""
    if len(xs) <= keep:
        return xs, ys
    step = len(xs) / keep
    idx = sorted({int(i * step) for i in range(keep)} | {0, len(xs) - 1})
    return [xs[i] for i in idx], [ys[i] for i in idx]


def fig_pr_curves(result, path, plt):
    """The trade-off curve, pooled over the seven US splits — how to choose a threshold.

    The headline table reports one point per model. This is the surface that point sits on,
    which is what a threshold decision actually needs: a model with a calibrated score can
    be moved along its curve for free, and a chat VLM cannot be moved at all. Both facts
    are visible here and neither is visible in a table of F1.

    Pooling here is MICRO (concatenate every panorama, then integrate once), unlike the
    macro-mean headline: a PR curve is an integral over ranked predictions and has no
    natural macro form. Said on the figure so the two cannot be silently compared.
    """
    from matplotlib.lines import Line2D

    curves = result.get("curves") or {}
    by_name = {m["model"]: m for m in result["models"]}

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Scoreless models first, so the curves draw over them.
    for m in result["models"]:
        if m["model"] in curves or not m["complete"]:
            continue
        if m["precision"] is None:
            continue
        ax.plot(m["recall"], m["precision"], "o", ms=7, color=INK_MUTED,
                mec=SURFACE, mew=1.6, zorder=4)
        ax.annotate(m["display"], (m["recall"], m["precision"]),
                    textcoords="offset points", xytext=(9, -3.5), fontsize=8,
                    color=INK_MUTED, zorder=5)

    handles = []
    for name, curve in sorted(curves.items(),
                              key=lambda kv: -(kv[1]["ap"] or 0)):
        colour = CURVE_COLOR.get(name, INK_MUTED)
        dash = CURVE_DASH.get(name, "solid")
        ref = name == "rampnet"
        xs, ys = _decimate(curve["recalls"], curve["precisions"])
        ax.plot(xs, ys, color=colour, lw=2.6 if ref else 1.7, ls=dash,
                zorder=6 if ref else 5, solid_capstyle="round")
        label = by_name.get(name, {}).get("display", name)
        handles.append(Line2D([], [], color=colour, lw=2.6 if ref else 1.7, ls=dash,
                              label=f"{label}  AP {curve['ap']:.3f}"))

    # The two thresholds the project has argued about, on RampNet's curve.
    marks = (curves.get("rampnet") or {}).get("marks") or {}
    for thr, mk in sorted(marks.items()):
        deployed = abs(float(thr) - 0.55) < 1e-9
        ax.plot(mk["recall"], mk["precision"], "o", ms=9,
                mfc=CURVE_COLOR["rampnet"] if not deployed else SURFACE,
                mec=CURVE_COLOR["rampnet"], mew=2.2, zorder=8)
        ax.annotate(f"{float(thr):.2f}" + ("  deployed" if deployed else "  recommended (#54)"),
                    (mk["recall"], mk["precision"]), textcoords="offset points",
                    xytext=(-10, 11 if deployed else -16), ha="right", fontsize=8.2,
                    color=INK, zorder=9)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("precision", fontsize=9.5, color=INK_SECONDARY)
    _titles(ax, "A calibrated score is a dial; a chat VLM is a dot",
            "Pooled over the seven US splits (micro — every panorama counts once). "
            "Grey dots emit no confidence and cannot be moved.")
    # Mid-left: the lower-left corner is where the two open-detector curves live, and a
    # legend there sits on top of them.
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.2,
                    labelcolor=INK_SECONDARY, handlelength=2.6,
                    bbox_to_anchor=(0.01, 0.30))
    leg.set_zorder(10)
    fig.text(0.008, 0.030,
             "RampNet's curve is read from analysis_out/op_cache — the #54 low-floor "
             "re-extraction of the same run, no TTA.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.text(0.008, 0.008,
             "Below 0.55 it is a LOWER bound: the ground truth was assembled from "
             "detections at or above that floor, so a real ramp nobody marked scores as a "
             "false positive.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.052, 1, 0.96))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


FIGURES = {
    "scoreboard_pr_curves.png": fig_pr_curves,
    "scoreboard_f1.png": fig_headline,
    "scoreboard_pr.png": fig_precision_recall,
    "scoreboard_by_split.png": fig_by_split,
    "scoreboard_generalization.png": fig_generalization,
}


def render_all(result, figure_dir):
    """Write every figure; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figure_dir, exist_ok=True)
    written = []
    for name, fn in FIGURES.items():
        written.append(fn(result, os.path.join(figure_dir, name), plt))
    return written
