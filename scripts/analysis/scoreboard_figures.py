"""The five figures for docs/model_scoreboard.md — the findings that read faster as a picture.

Called by ``scoreboard.py``; importable on its own for iterating on a figure without
re-scoring. Matplotlib is imported inside ``render_all`` so ``--no-figures`` needs no
plotting stack at all.

**Colour.** The four aggregate figures run on one hue: the validated categorical slot 1
(``#2a78d6``) plus the palette's neutral inks, with the sequential blue ramp for the
heatmap. That is a deliberate reduction, not laziness. The natural design — a hue per
model class — cannot ship: three of those four are scatter/matrix forms, which are
scored on the **all-pairs** pairlist, and five categorical slots fail it outright
(``validate_palette.js``: magenta↔orange normal-vision ΔE 12.9, below the hard floor of
15). A normal-vision FAIL is the one result secondary encoding does not excuse, so the
documented remedy is to cut series or facet rather than to add a legend and hope. Model
class is therefore carried by **position** (bar order, group headers) and **marker
shape** — channels with no CVD failure mode — and colour is freed to carry emphasis.

The PR-curve figure is the exception and gets five hues, because overlapping *lines* are
scored on the **adjacent** pairlist, which that set passes outright (worst adjacent CVD
ΔE 9.1); see ``CURVE_COLOR``. Marker shape cannot separate curves, so there is no
position-or-shape substitute to fall back on there.

The palette's contrast WARN (slot 1 sits above 3:1, but the light neutrals do not) is
discharged the way ``plot_operating_point.py`` discharges it: every plotted number is
also in a table, in ``docs/model_scoreboard.md`` and ``analysis_out/scoreboard.json``.
"""
import os
import re

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
    # Left: running right put this label past the right spine.
    "mask2former-vistas-curb-cut-1024x1024": (-11, -3.5, "right"),
    # Raised: at the default height it printed over Gemini 3.1 Pro's label, 0.05 to its left.
    "claude-fable-5-1-effort-low-anthropic": (11, 6, "left"),
}
# The scoreless dots on the PR-curve figure; the default is right of the dot.
CURVE_DOT_NUDGE = {
    # These two dots sit 0.05 recall apart at the same precision, so running both labels
    # right printed one over the other. Gemini's goes left; Fable's stays right, raised so
    # it clears the Gemini dot and still starts at its own.
    "gemini-3.1-pro-preview": (-9, -3.5, "right"),
    "claude-fable-5-1-effort-low-anthropic": (9, 5, "left"),
}
GEN_NUDGE = {
    # RampNet sits close to the diagonal in the top-right, so its label crosses that line
    # whichever side it goes; the halo carries it rather than a nudge that cannot fit.
    "rampnet": (-14, -13, "right"),
    "y11l_pano": (11, 6, "left"),
    "y11x_pano_h200": (11, -7, "left"),
    # Below-right, not left: running left put this label over the Vistas marker 0.09 away.
    "y26_pano": (6, -18, "left"),
    # Left, under the diagonal: right of it is the YOLO26 / YOLO11x label stack.
    "mask2former-vistas-curb-cut-1024x1024": (-11, -10, "right"),
    # Left: running right put this label across YOLO11l's at the same height.
    "claude-fable-5-1-effort-low-anthropic": (-11, 4, "right"),
    # Left: running right put this label's number on YOLO26's marker.
    "gemini-3.7-flash": (-11, 6, "right"),
    # Left, not right: running right puts this label across Qwen-8B's marker 0.10 away,
    # and a reader then attaches it to the wrong point.
    "Qwen/Qwen3-VL-32B-Instruct": (-11, 4, "right"),
    "Qwen/Qwen3-VL-8B-Instruct": (11, -8, "left"),
    "google/owlv2-large-patch14-ensemble": (12, 7, "left"),
    "IDEA-Research/grounding-dino-base": (12, -9, "left"),
}


def _halo(width=2.6):
    """A SURFACE-coloured stroke behind label text.

    These are dense scatters over contours, a reference diagonal and each other, and a
    label that crosses a hairline reads as damaged even though it is legible. A halo
    fixes every such crossing at once, which nudging one label at a time does not — and
    unlike a nudge it cannot push a label off the axes.
    """
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=SURFACE)]


def _titles(ax, title, subtitle):
    """Title above, subtitle below it, both above the axes and left-aligned.

    set_title() alone puts the title flush to the axes, so a subtitle placed in axes
    coordinates lands ABOVE it and the two read in the wrong order.
    """
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=30)
    ax.text(0.0, 1.018, subtitle, transform=ax.transAxes, fontsize=8.6,
            color=INK_SECONDARY, va="bottom")


def _n_pooled(result):
    """The pooled-split count as a word ("eight"), read off the board rather than typed (#171)."""
    from scoreboard_render import number_word
    return number_word(len(result["pooled_splits"]))


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


def _partial_footer(result):
    """The footer clause for the legs left off the pooled figures, or "" if there are none.

    Counted off the board rather than typed: the partial legs are not all single-split (one
    covers eight of the nine pooled cities), so "single-split legs" was wrong as soon as one
    of them grew. ``pooled_splits`` on a model row names (or, in the JSON, counts) the
    pooled cities it covers.

    >>> _partial_footer({"pooled_splits": list("abcdefghi"), "models": [
    ...     {"complete": False, "pooled_splits": 1},
    ...     {"complete": False, "pooled_splits": 8}]})
    '2 legs that cover 1–8 of the nine pooled splits are reported per split instead — '
    """
    partial = [m for m in result["models"] if not m["complete"]]
    if not partial:
        return ""
    # A list of split names in build()'s result, a count once json_payload has written it.
    counts = [len(v) if isinstance(v, (list, tuple)) else v
              for v in (m["pooled_splits"] for m in partial)]
    lo, hi = min(counts), max(counts)
    span = f"{lo}" if lo == hi else f"{lo}–{hi}"
    noun = "leg that covers" if len(partial) == 1 else "legs that cover"
    return (f"{len(partial)} {noun} {span} of the {_n_pooled(result)} pooled splits "
            f"{'is' if len(partial) == 1 else 'are'} reported per split instead — ")


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

    # 10.8 wide, not 9.6: the title starts at the axes' left edge, and the longest tick
    # label (the Vistas arm, pooled since 2026-10-20) pushed that edge far enough right at
    # 9.6 that the title ran off the figure.
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
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
    ax.set_xlabel(f"F1, macro-mean over the {_n_pooled(result)} pooled US city splits",
                  fontsize=9.5, color=INK_SECONDARY)

    # The lead goes in the subtitle rather than into an annotated arrow: the gap between
    # the top two bars is 0.38 of a row, which cannot hold a rule and a caption without
    # colliding with one of them.
    #
    # Both lookups tolerate absence: a --models subset run may hold no RampNet row, or no
    # challenger, and a figure helper is the wrong place to die on that.
    runner_up = max((m for m in models if m["model"] != "rampnet"),
                    key=lambda m: m["f1"], default=None)
    ref_f1 = next((m["f1"] for m in models if m["model"] == "rampnet"), None)
    if ref_f1 is not None and runner_up is not None:
        subtitle = (f"+{ref_f1 - runner_up['f1']:.3f} F1 clear of the best challenger "
                    f"({runner_up['display']}, {runner_up['f1']:.3f}).")
    else:
        subtitle = f"Macro-mean over the {_n_pooled(result)} pooled US city splits."
    _titles(ax, "RampNet leads every off-the-shelf and supervised baseline tested",
            subtitle)
    # Two lines: at 7.4pt one line of this runs past the right edge and gets clipped.
    fig.text(0.008, 0.030,
             "Operating points differ by model class: RampNet 0.55, YOLO 0.25, "
             "open-vocab 0.05 floor, chat VLMs emit no score.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.text(0.008, 0.008, _partial_footer(result) + "see docs/model_scoreboard.md.",
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
    # Contour labels go in the right margin, not wherever the auto-placer picks. Left to
    # itself it put "F1 0.6" at recall 0.44 / precision 0.99 — on top of the two YOLO
    # markers and their labels. Every iso-F1 curve crosses recall 1.0 at p = F1/(2 - F1),
    # and nothing on this board reaches recall 1.0, so that column is always free.
    ax.clabel(cs, fmt=lambda v: f"F1 {v:.1f}", fontsize=6.8, colors=INK_MUTED,
              manual=[(1.0, lv / (2 - lv)) for lv in levels])

    for m in models:
        ref = m["model"] == "rampnet"
        ax.plot(m["recall"], m["precision"], CLASS_MARKER[m["class"]],
                ms=13 if ref else 9.5, color=BLUE,
                alpha=1.0 if ref else 0.72, mec=SURFACE, mew=2.0, zorder=5)
        dx, dy, ha = PR_NUDGE.get(m["model"], (11, -3.5, "left"))
        ax.annotate(m["display"], (m["recall"], m["precision"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=8.4, color=INK if ref else INK_SECONDARY,
                    fontweight="bold" if ref else "normal", zorder=6,
                    path_effects=_halo())

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("recall", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("precision", fontsize=9.5, color=INK_SECONDARY)
    _titles(ax, "The same F1 hides opposite failures",
            f"Macro-mean over the {_n_pooled(result)} pooled US city splits. "
            "Marker shape is model class.")

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


# How far a pooled mean recomputed from the per-split cells may sit from the board's own.
# scoreboard.json rounds every float to six places (scoreboard_render.JSON_PRECISION), so
# a mean of rounded cells and the rounded mean can differ by up to ~1e-6; the in-memory
# board agrees to float precision.
POOLED_MEAN_TOL = 2e-6


def by_split_rows(result):
    """Row order for the by-split heatmap, as ``(full, partial)``.

    Each list holds ``(summary, pooled_mean, n_pooled_run)`` tuples. ``pooled_mean`` is the
    board's own pooled F1 (``summary["f1"]``, the number the headline quotes) -- read, not
    recomputed -- and it is cross-checked here against the mean of the row's per-split
    cells, so the aggregate column can never disagree with the cells printed beside it.

    * ``full``: legs that ran every pooled city, sorted by pooled mean, descending.
    * ``partial``: every other leg, in its own group, sorted by how many pooled cities it
      ran (most first), then by the mean over those cities (a leg with none sorts last).
      Kept apart so a one-city leg is never ranked against a nine-city mean, and ordered by
      coverage first so a one-city mean never sits above an eight-city one either.

    Works on the in-memory board and on ``analysis_out/scoreboard.json`` alike. Ties keep
    the board's order (complete, then class, then F1), because ``sort`` is stable.

    Example::

        full, partial = by_split_rows(json.load(open("analysis_out/scoreboard.json")))
        [m["model"] for m, _, _ in full][:2]   # ['rampnet', 'claude-opus-5-5-effort-low-anthropic']
    """
    pooled = list(result["pooled_splits"])
    per = result["per_split"]
    full, partial = [], []
    for m in result["models"]:
        cells = per[m["model"]]
        f1s = [cells[s]["f1"] for s in pooled if cells.get(s)]
        n_run = len(f1s)
        assert len(m["pooled_splits"]) == n_run, (m["model"], m["pooled_splits"], n_run)
        assert m["complete"] == (n_run == len(pooled)), m["model"]
        if f1s:
            recomputed = sum(f1s) / n_run
            assert m["f1"] is not None and abs(recomputed - m["f1"]) <= POOLED_MEAN_TOL, (
                f"{m['model']}: pooled F1 {m['f1']} != mean of its cells {recomputed}")
        else:
            assert m["f1"] is None, m["model"]
        (full if m["complete"] else partial).append((m, m["f1"], n_run))
    full.sort(key=lambda r: -r[1])
    partial.sort(key=lambda r: (r[1] is None, -r[2], -(r[1] or 0.0)))
    return full, partial


def held_out_footer(result, held):
    """The by-split figure's dagger footnote, built from ``result["held_out"]``.

    Each reason is cut to its first clause (up to the first comma, parenthesis or dash),
    and splits that share a short reason are named together, so the footnote follows the
    board's own split names and reasons rather than a hand-typed copy of them.

    Example::

        held_out_footer({"held_out": {"sao_paulo": "non-US city -- ...",
                                      "bayonne": "non-US city -- ..."}},
                        ["sao_paulo", "bayonne"])
        # '† held out of the pooled headline: sao_paulo and bayonne (non-US city).'
    """
    groups = {}
    for split in held:
        reason = result["held_out"].get(split, "")
        short = re.split(r",| \(| — | -- ", reason, maxsplit=1)[0].strip()
        groups.setdefault(short, []).append(split)
    def names(splits):
        return splits[0] if len(splits) == 1 else \
            ", ".join(splits[:-1]) + " and " + splits[-1]

    parts = [names(splits) + (f" ({short})" if short else "")
             for short, splits in groups.items()]
    return "† held out of the pooled headline: " + ", ".join(parts) + "."


def fig_by_split(result, path, plt):
    """F1 for every (model, split) cell — consistency across cities, at a glance.

    Sequential single-hue ramp (magnitude), values printed in every cell so the figure is
    also its own table. Left to right: the pooled cities, the **pooled mean** (the headline
    number, which the rows are sorted by), a gutter, then the held-out splits, so those are
    never read as part of the pooled headline. Top to bottom: the legs that ran every
    pooled city, a labelled rule, then the partial-coverage legs (``by_split_rows``).
    """
    pooled = list(result["pooled_splits"])
    held = [s for s in result["all_splits"] if s not in pooled]
    per = result["per_split"]
    full, partial = by_split_rows(result)
    n_pool = len(pooled)

    # Columns, in data units: cities, a narrow gap, the wider aggregate column (it has to
    # hold e.g. "0.568 (8/9)"), the gutter, then the held-out splits.
    agg_gap, agg_w, gutter = 0.3, 1.6, 1.0
    col_x = {s: float(i) for i, s in enumerate(pooled)}
    agg_x = n_pool + agg_gap
    held_x0 = agg_x + agg_w + gutter
    col_x.update({s: held_x0 + i for i, s in enumerate(held)})
    width = held_x0 + len(held)

    # Rows: the full-coverage group, a separator band, the partial group. head is the band
    # above the grid that carries the column-group labels.
    sep, head = 0.9, 0.95
    placed = [(r, float(i)) for i, r in enumerate(full)]
    sep_y = float(len(full))
    placed += [(r, sep_y + sep + i) for i, r in enumerate(partial)]
    height = len(full) + (sep + len(partial) if partial else 0.0)

    # Height tracks the row count and width the column count, so adding a leg or a split
    # does not silently squash the cells. The constant terms hold the row labels, titles
    # and footer; at fourteen splits this is the 13-inch width the figure was drawn at.
    fig_h = 1.9 + 0.42 * (height + head)
    fig_w = 3.0 + 0.59 * width
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    def cell(x, y, w, value, text, bold=False):
        color = _seq_color(value, 1.0) if value is not None else "#f4f3f0"
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), w - 0.06, 0.94,
                                   facecolor=color, edgecolor="none", zorder=2))
        # White ink only where the fill is dark enough to carry it.
        ink = "#ffffff" if (value is not None and value >= 0.62) else INK
        ax.text(x + w / 2, y + 0.5, text, ha="center", va="center", fontsize=8.2,
                color=ink if value is not None else INK_MUTED,
                fontweight="bold" if bold else "normal", zorder=3)

    for (m, mean_f1, n_run), y in placed:
        cells = per[m["model"]]
        for split, x in col_x.items():
            c = cells.get(split)
            value = c["f1"] if c else None
            cell(x, y, 1.0, value, "—" if value is None else f"{value:.2f}")
        # Three places, like the headline it repeats; partial rows carry their coverage.
        if mean_f1 is None:
            text = "—"
        elif m["complete"]:
            text = f"{mean_f1:.3f}"
        else:
            text = f"{mean_f1:.3f} ({n_run}/{n_pool})"
        cell(agg_x, y, agg_w, mean_f1, text, bold=True)

    # One frame per row group around the aggregate column: it is a summary of the cells to
    # its left, not another city, and the frame says so without spending a colour.
    for y0, n in ((0.0, len(full)), (sep_y + sep, len(partial))):
        if n:
            ax.add_patch(plt.Rectangle((agg_x - 0.02, y0 - 0.02), agg_w + 0.04, n + 0.04,
                                       facecolor="none", edgecolor=INK, lw=1.1, zorder=4))

    # Column-group labels, with a hairline under each, so the gutter reads as a boundary
    # between two groups rather than as a missing column.
    def group(x0, x1, label, **kw):
        ax.plot([x0 + 0.05, x1 - 0.05], [-0.14, -0.14], color=BASELINE, lw=1.0,
                solid_capstyle="butt", clip_on=False, zorder=3)
        ax.text((x0 + x1) / 2, -0.30, label, ha="center", va="bottom", fontsize=9,
                clip_on=False, **kw)

    group(0.0, float(n_pool), f"Pooled US cities ({_n_pooled(result)})",
          color=INK_SECONDARY)
    group(agg_x, agg_x + agg_w, "Pooled mean ▼", color=INK, fontweight="bold")
    if held:
        group(held_x0, width, "Held out †", color=INK_SECONDARY)

    if partial:
        ax.plot([0.0, width], [sep_y + 0.16, sep_y + 0.16], color=BASELINE, lw=1.0,
                zorder=3)
        # The "rows above" clause only makes sense when there are rows above.
        not_ranked = (", and these rows are not ranked against the rows above."
                      if full else ".")
        ax.text(0.04, sep_y + sep / 2 + 0.1,
                "Partial pooled coverage: the mean is over the pooled cities each leg ran "
                f"(n/{n_pool}){not_ranked}",
                ha="left", va="center", fontsize=8.4, color=INK_SECONDARY,
                fontstyle="italic", zorder=3)

    ax.set_xlim(0, width)
    # A hair past the last row, or the bottom edge of the aggregate frame is clipped.
    ax.set_ylim(height + 0.06, -head)
    xt = [col_x[s] + 0.5 for s in pooled] + [agg_x + agg_w / 2] + \
         [col_x[s] + 0.5 for s in held]
    ax.set_xticks(xt)
    ax.set_xticklabels(pooled + ["pooled mean"] + [f"{s} †" for s in held],
                       rotation=38, ha="right", fontsize=8.2, color=INK_SECONDARY)
    agg_tick = ax.get_xticklabels()[n_pool]
    agg_tick.set_color(INK)
    agg_tick.set_fontweight("bold")
    ax.set_yticks([y + 0.5 for _, y in placed])
    ax.set_yticklabels([m["display"] for (m, _, _), _ in placed], fontsize=8.8,
                       color=INK_SECONDARY)
    for tick, ((m, _, _), _) in zip(ax.get_yticklabels(), placed):
        if m["model"] == "rampnet":
            tick.set_color(INK)
            tick.set_fontweight("bold")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    # The claim is bounded on purpose. RampNet does NOT have the flattest row outright --
    # OWLv2 (0.028) and Grounding DINO (0.039) are flatter, by being pinned near zero in
    # every city. Consistency only means something above the floor, so the comparison is
    # drawn against the models that clear it, and the threshold is stated rather than
    # implied.
    models = result["models"]
    working = [m for m in models
               if m["complete"] and m["f1"] is not None and m["f1"] >= 0.1]
    spreads = [m["f1_max"] - m["f1_min"] for m in working if m["model"] != "rampnet"]
    ref = next((m for m in models if m["model"] == "rampnet"), None)
    if ref is not None and ref["f1_max"] is not None and spreads:
        subtitle = (f"RampNet varies by {ref['f1_max'] - ref['f1_min']:.2f} across the "
                    f"{_n_pooled(result)} pooled cities; every challenger above F1 0.1 varies by "
                    f"{min(spreads):.2f}–{max(spreads):.2f}.")
    else:
        subtitle = "F1 per (model, split). Held-out splits sit right of the gutter."
    _titles(ax, "F1 by model and split", subtitle)

    # Two footer lines, spaced in points so they stay apart however tall the figure is.
    line = 11.0 / (fig_h * 72.0)
    fig.text(0.008, 0.008 + line,
             f"Pooled mean: macro-mean F1 over the {_n_pooled(result)} pooled US cities, the "
             "headline number (docs/model_scoreboard.md); full-coverage rows are sorted by it, "
             "partial rows by coverage, then by it.",
             fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    if held:
        fig.text(0.008, 0.008, held_out_footer(result, held),
                 fontsize=7.4, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.008 + 2.4 * line, 1, 1))
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
                    fontweight="bold" if ref else "normal", zorder=6,
                    path_effects=_halo())

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("in-distribution F1  (manual_gold, 1,000 GSV panoramas)",
                  fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(f"deployed F1  (macro-mean, {_n_pooled(result)} US cities)",
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


# Five validated categorical hues for the curve families (validate_palette.js, adjacent
# pairlist -- the documented one for lines: ALL PASS, worst adjacent CVD dE 9.1). The three
# YOLO arms share one hue and separate by dash: they are one family, and spending three
# slots on them would push the set past what the all-pairs floors allow.
#
# The Vistas arm takes slot 7 (violet). Its curve crosses the YOLO, OWLv2 and Grounding DINO
# curves, and violet passes against each of those (worst CVD dE 27.1, aqua). It fails
# against RampNet's blue (CVD dE 4.5, normal 10.5), but the two curves meet only at the
# (0, 1) anchor: at every recall above 0 that the Vistas arm reaches, RampNet's curve is at
# least 0.37 precision above it. Magenta (slot 5) was the other candidate and fails normal
# vision against the YOLO orange (dE 12.9), which the Vistas curve does cross. Before
# 2026-10-20 the arm had no entry and drew in INK_MUTED, the grey of the scoreless dots.
CURVE_COLOR = {
    "rampnet": "#2a78d6",
    "y11l_pano": "#eb6834",
    "y11x_pano_h200": "#eb6834",
    "y26_pano": "#eb6834",
    "mask2former-vistas-curb-cut-1024x1024": "#6250d6",
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
    """The trade-off curve, pooled over the US splits — how to choose a threshold.

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
    #
    # MICRO precision/recall, not the headline table's macro-mean: these axes are the
    # micro-pooled curve's, and a macro point plotted on them is a different aggregation
    # under the same subtitle. The shift is small (max 0.010 in P, Qwen-32B) but "small"
    # is not "labelled", and this figure's whole subject is what a point on a curve means.
    for m in result["models"]:
        if m["model"] in curves or not m["complete"]:
            continue
        pr = m.get("micro_precision"), m.get("micro_recall")
        if pr[0] is None or pr[1] is None:
            continue
        precision, recall = pr
        ax.plot(recall, precision, "o", ms=7, color=INK_MUTED,
                mec=SURFACE, mew=1.6, zorder=4)
        # Labels draw ABOVE the curves (zorder 5-6), so where one crosses a curve its halo
        # clears the line instead of the line striking through the text.
        dx, dy, ha = CURVE_DOT_NUDGE.get(m["model"], (9, -3.5, "left"))
        ax.annotate(m["display"], (recall, precision),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8,
                    color=INK_MUTED, zorder=7, path_effects=_halo())

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
                              label=f"{label}  {curve['ap']:.3f}"))

    # The two thresholds the project has argued about, on RampNet's curve.
    from scoreboard import DEPLOYED_THRESHOLD

    marks = (curves.get("rampnet") or {}).get("marks") or {}
    for thr, mk in sorted(marks.items()):
        deployed = abs(float(thr) - DEPLOYED_THRESHOLD) < 1e-9
        ax.plot(mk["recall"], mk["precision"], "o", ms=9,
                mfc=CURVE_COLOR["rampnet"] if not deployed else SURFACE,
                mec=CURVE_COLOR["rampnet"], mew=2.2, zorder=8)
        ax.annotate(f"{float(thr):.2f}" + ("  deployed" if deployed else "  recommended (#54)"),
                    (mk["recall"], mk["precision"]), textcoords="offset points",
                    xytext=(-10, 11 if deployed else -16), ha="right", fontsize=8.2,
                    color=INK, zorder=9, path_effects=_halo())

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("precision", fontsize=9.5, color=INK_SECONDARY)
    _titles(ax, "A calibrated score is a dial; a chat VLM is a dot",
            f"Pooled over the {_n_pooled(result)} US splits, micro — every panorama counts once, dots "
            "included. Grey dots emit no confidence and cannot be moved.")
    # Mid-left: the lower-left corner is where the two open-detector curves live, and a
    # legend there sits on top of them.
    #
    # "micro" belongs in the legend TITLE, not repeated on six rows: the headline table's
    # AP column is the macro-mean of the same data and lands a few thousandths away, so
    # the family has to be named — but naming it six times widened the block until its
    # last row ran into the Qwen-8B dot label.
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.2,
                    labelcolor=INK_SECONDARY, handlelength=2.6,
                    title="AP, micro-pooled", alignment="left",
                    bbox_to_anchor=(0.01, 0.30))
    leg.get_title().set(color=INK_MUTED, fontsize=7.8)
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
