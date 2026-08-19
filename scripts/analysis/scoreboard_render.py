"""Markdown tables for docs/model_scoreboard.md, and the splice that keeps them current.

Split out of ``scoreboard.py`` so the scoring is testable without a doc on disk and the
rendering is testable without re-scoring ten splits.

The doc holds the prose; this holds the numbers. Each generated block sits between a
matched pair of HTML comments and is replaced wholesale on every run, so a number can
only ever enter the doc by coming out of the scorer. Prose written *outside* the markers
is never touched. ``scoreboard.py --check`` re-renders and compares, which is what makes a
stale summary a test failure rather than a thing someone notices six months later.
"""
import json
import os
import re

BEGIN = "<!-- BEGIN GENERATED: {name} (scripts/analysis/scoreboard.py) -->"
END = "<!-- END GENERATED: {name} -->"

# Column header for each split, short enough that the by-split matrix stays readable.
SPLIT_HEADER = {
    "richmond": "rich",
    "bend": "bend",
    "clovis": "clovis",
    "morgantown": "morg",
    "annapolis": "annap",
    "paterson": "pater",
    "gainesville": "gaines",
    "budapest_district5": "budapest †",
    "sao_paulo": "sao_paulo †",
    "manual_gold": "manual_gold †",
}


def num(value, places=3, dash="–"):
    return dash if value is None else f"{value:.{places}f}"


def bold(text, on=True):
    return f"**{text}**" if on else text


def _table(header, rows, align=None):
    align = align or (["---"] * len(header))
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def headline_table(result):
    """Rows are models, columns are metrics, pooled over the seven US city splits.

    Only legs with all seven are here. A one-city mean in the same column as a
    seven-city one is the exact confusion the coverage field exists to prevent, so the
    partial legs get their own table instead of a footnote.
    """
    from scoreboard import CLASS_LABEL, RAMPNET

    models = [m for m in result["models"] if m["complete"]]
    best = {k: max((m[k] for m in models if m[k] is not None), default=None)
            for k in ("precision", "recall", "f1", "ap")}
    ref = next((m["f1"] for m in models if m["model"] == RAMPNET), None)

    rows = []
    for m in models:
        lead = None if (ref is None or m["f1"] is None) else m["f1"] - ref
        if m["model"] == RAMPNET:
            delta = "—"
        elif lead is None:
            delta = "–"
        else:
            delta = f"{lead:+.3f}"
        span = ("–" if m["f1_min"] is None
                else f"{m['f1_min']:.2f}–{m['f1_max']:.2f}")
        rows.append([
            bold(m["display"], m["model"] == RAMPNET),
            CLASS_LABEL[m["class"]],
            m["operating_point_note"],
            bold(num(m["precision"]), m["precision"] == best["precision"]),
            bold(num(m["recall"]), m["recall"] == best["recall"]),
            bold(num(m["f1"]), m["f1"] == best["f1"]),
            delta,
            num(m["ap"]) + ("&nbsp;†" if m.get("ap_is_substituted") and m["ap"] else ""),
            num(m["fp_per_pano"], 1),
            span,
        ])
    # Every metric here is the macro-mean over the seven pooled splits, AP included. The
    # PR-curve figure's legend is the MICRO-pooled AP of the same data and reads a few
    # thousandths different; the two are labelled wherever both appear so a reader never
    # has to guess which family a number belongs to.
    header = ["model", "class", "op", "P", "R", "F1", "ΔF1 vs RampNet", "AP (macro)",
              "FP/pano", "F1 range"]
    align = ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", ":-:"]
    return _table(header, rows, align)


def partial_table(result):
    """Legs that have not run every pooled split — reported per split, never averaged.

    Returns a one-line note instead of a table when there are none, so the block never
    renders as an empty header that reads like a missing result.
    """
    from scoreboard import CLASS_LABEL

    partial = [m for m in result["models"] if not m["complete"]]
    if not partial:
        return "*Every registered leg has run all seven pooled splits.*"

    per = result["per_split"]
    rows = []
    for m in partial:
        for split in result["all_splits"]:
            cell = per[m["model"]].get(split)
            if not cell:
                continue
            rows.append([
                m["display"],
                CLASS_LABEL[m["class"]],
                f"`{split}`",
                num(cell["precision"]),
                num(cell["recall"]),
                num(cell["f1"]),
                num(cell["ap"]),
                num(cell["fp_per_pano"], 1),
                f"{cell['tp']}/{cell['fp']}/{cell['fn']}",
            ])
    header = ["model", "class", "split", "P", "R", "F1", "AP", "FP/pano", "tp/fp/fn"]
    align = ["---", "---", "---", "--:", "--:", "--:", "--:", "--:", "--:"]
    return _table(header, rows, align)


def by_split_table(result):
    """F1 for every (model, split) pair — the matrix the per-split tables never form."""
    from scoreboard import RAMPNET, US_SPLITS

    splits = result["all_splits"]
    per = result["per_split"]
    # Best model per split, so each column's winner is visible without arithmetic.
    best = {}
    for s in splits:
        vals = [cells[s]["f1"] for cells in per.values() if cells.get(s)]
        best[s] = max(vals) if vals else None

    best_pooled = max((m["f1"] for m in result["models"]
                       if m["complete"] and m["f1"] is not None), default=None)
    rows = []
    for m in result["models"]:
        cells = per[m["model"]]
        row = [bold(m["display"], m["model"] == RAMPNET)]
        for s in splits:
            if s in US_SPLITS:
                cell = cells.get(s)
                row.append("–" if not cell else bold(num(cell["f1"], 3),
                                                     cell["f1"] == best[s]))
            if s == US_SPLITS[-1]:
                # A partial row has a pooled mean, but it is a mean over a different set
                # of cities, so printing it in this column would invite the comparison
                # the partial table exists to prevent.
                row.append("–" if not m["complete"]
                           else bold(num(m["f1"]), m["f1"] == best_pooled))
        for s in splits:
            if s not in US_SPLITS:
                cell = cells.get(s)
                row.append("–" if not cell else bold(num(cell["f1"], 3),
                                                     cell["f1"] == best[s]))
        rows.append(row)

    header = ["model"] + [SPLIT_HEADER[s] for s in US_SPLITS] + ["**pooled**"] + \
             [SPLIT_HEADER[s] for s in splits if s not in US_SPLITS]
    align = ["---"] + ["--:"] * (len(header) - 1)
    return _table(header, rows, align)


def threshold_table(result):
    """RampNet at the two thresholds the project has argued about, pooled over the US splits.

    The headline table reports one operating point per model because most of the roster
    has only one. RampNet does not, and the difference is the whole of #54, so it gets the
    two rows rather than a sentence.
    """
    marks = ((result.get("curves") or {}).get("rampnet") or {}).get("marks") or {}
    if not marks:
        return "*No low-floor cache available — run `operating_point_curve.py extract`.*"
    note = {"0.55": "deployed today (`OPERATIONAL_CONFIDENCE`, auto-labeler)",
            "0.30": "recommended by #54; **not yet adopted** (labeler#20 open)"}
    rows = []
    for thr in sorted(marks, reverse=True):
        m = marks[thr]
        rows.append([f"**{thr}**", num(m["precision"]), num(m["recall"]), num(m["f1"]),
                     note.get(thr, "")])
    return _table(["peak threshold", "P", "R", "F1", ""], rows,
                  ["---", "--:", "--:", "--:", "---"])


def ap_provenance_table(result):
    """Where RampNet's AP on each split comes from, and what the log prints for it.

    This is the one column where the two documents disagree by design, so the mapping is
    generated rather than described: ``model_comparison.md`` prints the bundle AP, this
    page prints the low-floor one, and both are here side by side with the reason. The
    test asserts the middle column against the log, so the correspondence is a gate.
    """
    from scoreboard import RAMPNET

    cells = result["per_split"].get(RAMPNET) or {}
    rows = []
    for split in result["all_splits"]:
        cell = cells.get(split)
        if not cell:
            continue
        substituted = cell["ap_source"] != "bundle"
        rows.append([
            f"`{split}`",
            num(cell["ap_bundle"]),
            bold(num(cell["ap"]), substituted),
            "`op_cache` (0.05 floor)" if substituted else "bundle — already at 0.05",
            "truncated at the deployed 0.55" if substituted
            else "no truncation to undo; flip-TTA export",
        ])
    return _table(["split", "AP in `model_comparison.md`", "AP here", "read from", "why"],
                  rows, ["---", "--:", "--:", "---", "---"])


def coverage_note(result):
    """What each split is, how big it is, and why a held-out one is held out."""
    rows = []
    for split, info in result["splits"].items():
        why = result["held_out"].get(split)
        rows.append([
            f"`{split}`",
            "pooled" if info["pooled"] else "held out †",
            str(info["n_panos"]),
            str(info["n_gt"]),
            why or "US deployment city, verdict-grade GT",
        ])
    return _table(["split", "role", "panos", "GT ramps", "note"], rows,
                  ["---", "---", "--:", "--:", "---"])


def render_tables(result):
    """{block name: markdown} for every generated block in the doc."""
    return {
        "headline": headline_table(result),
        "thresholds": threshold_table(result),
        "partial": partial_table(result),
        "by-split": by_split_table(result),
        "ap-provenance": ap_provenance_table(result),
        "coverage": coverage_note(result),
    }


def splice(text, tables):
    """Replace each generated block in ``text``; leave everything else byte-identical.

    A block present in ``tables`` but absent from the doc is a silent no-op by design:
    the doc decides which tables it wants and where, the script only decides what they
    say.
    """
    for name, body in tables.items():
        pattern = re.compile(
            re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)),
            re.S)
        replacement = (BEGIN.format(name=name) + "\n\n" + body + "\n\n"
                       + END.format(name=name))
        text = pattern.sub(lambda _m: replacement, text)
    return text


def json_payload(result):
    """The committed JSON, as a string — the result minus the plot-only curve arrays.

    A PR curve is one point per ranked prediction, and the two open detectors carry
    ~120k between them: serialized they are 7.7 MB, 98% of the file, for something no
    reader diffs and ``scoreboard.py`` rebuilds from the same committed detections in
    about three seconds. What the page actually cites — the AP, RampNet's marked
    thresholds, and how many points the curve had — is kept.

    LF on every platform: this is byte-compared by ``--check``, and Python's default
    newline translation on Windows would emit CRLF and make a re-run look like a change
    (the imagery_manifest fix, 22dd536).
    """
    slim = dict(result)
    slim["curves"] = {
        name: {k: v for k, v in curve.items() if k not in ("recalls", "precisions")}
        for name, curve in (result.get("curves") or {}).items()
    }
    return json.dumps(slim, indent=2, sort_keys=False) + "\n"


def write_json(path, result):
    """Write the machine-readable scoreboard (see ``json_payload``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(json_payload(result))
