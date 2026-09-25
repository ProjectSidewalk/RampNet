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
    "laurens_mapillary": "laur_mly",
    "laurens_gsv": "laur_gsv †",
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
        floor = cell.get("bundle_floor")
        low_floor_bundle = floor is not None and floor < 0.4
        if substituted:
            read_from, why = ("`op_cache` (0.05 floor)",
                              "truncated at the deployed 0.55")
        elif low_floor_bundle:
            read_from, why = ("bundle — already at 0.05",
                              "no truncation to undo; flip-TTA export")
        else:
            # Truncated like the substituted rows, but with no low-floor cache to
            # swap in -- a held-out split never went through the #54 re-extraction.
            read_from, why = ("bundle — 0.55 floor, no `op_cache`",
                              "**truncated**; not comparable with the rows above")
        rows.append([
            f"`{split}`",
            num(cell["ap_bundle"]),
            bold(num(cell["ap"]), substituted),
            read_from,
            why,
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


# Decimals kept for every float in the committed JSON. Six is ~3,000x finer than
# anything the page reports (three decimals) and still coarse enough to be identical on
# every platform this runs on -- which is the whole point, see _round_floats.
JSON_PRECISION = 6


def _round_floats(value, places=JSON_PRECISION):
    """Round every float in a nested structure, so the artifact is byte-reproducible.

    Full-precision floats do NOT survive the trip between environments: AP comes out of
    numpy, and a different numpy build reorders the last bits of an accumulation, which
    changes ``repr`` and therefore the file. That made a byte-compare of this file fail
    on CI's Python 3.10 while passing on 3.12 and on the author's machine -- the artifact
    was not reproducible, and the check that was supposed to prove it was reproducible
    was the thing that noticed.

    Rounding fixes the artifact rather than weakening the check: seventeen significant
    digits of accumulation noise were never meaningful in a file whose purpose is to be
    diffed by a reviewer, and at six decimals a real change is still visible thousands of
    times before the page's three decimals would move.
    """
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round_floats(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, places) for v in value]
    return value


def json_payload(result):
    """The committed JSON, as a string — the result minus the plot-only curve arrays.

    A PR curve is one point per ranked prediction, and the two open detectors carry
    ~120k between them: serialized they are 7.7 MB, 98% of the file, for something no
    reader diffs and ``scoreboard.py`` rebuilds from the same committed detections in
    about three seconds. What the page actually cites — the AP, RampNet's marked
    thresholds, and how many points the curve had — is kept.

    Floats are rounded (``_round_floats``) so the file is identical on every platform,
    and written LF-only for the same reason: this is byte-compared by ``--check``, and
    Python's default newline translation on Windows would emit CRLF and make a re-run
    look like a change (the imagery_manifest fix, 22dd536).
    """
    slim = dict(result)
    slim["curves"] = {
        name: {k: v for k, v in curve.items() if k not in ("recalls", "precisions")}
        for name, curve in (result.get("curves") or {}).items()
    }
    return json.dumps(_round_floats(slim), indent=2, sort_keys=False) + "\n"


def write_json(path, result):
    """Write the machine-readable scoreboard (see ``json_payload``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(json_payload(result))


# --------------------------------------------------------------------------- #
# the prose around the tables
# --------------------------------------------------------------------------- #
# The generated blocks cannot drift, but the sentences around them can, and did: the page
# said "Eighteen model legs, ten splits" and "the seven pooled US city splits" for weeks
# after the board had grown to 21 legs, twelve splits and eight pooled cities (#171). The
# counts those sentences quote are all facts of the board, so they are checked against it
# here, and scoreboard.py --check fails when one is wrong.

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40}


def number_word(n):
    """0..49 as an English word, lower case: ``number_word(21) == "twenty-one"``."""
    if n < 20:
        return _UNITS[n]
    tens = next(w for w, v in _TENS.items() if v == n - n % 10)
    return tens if n % 10 == 0 else f"{tens}-{_UNITS[n % 10]}"


def word_number(word):
    """Inverse of :func:`number_word` (digits accepted too), or None for a non-number."""
    if word.isdigit():
        return int(word)
    word = word.lower()
    if word in _UNITS:
        return _UNITS.index(word)
    if word in _TENS:
        return _TENS[word]
    head, _, tail = word.partition("-")
    if head in _TENS and tail in _UNITS[1:10]:
        return _TENS[head] + _UNITS.index(tail)
    return None


# A count is digits or a number word, and nothing else. An earlier version accepted any
# word, so a later sentence such as "over all pooled cities" would have read "all" as the
# count and failed with a confusing message. Now a phrase with a non-number where the count
# goes does not match the rule at all.
_NUMBER_WORDS = sorted(set(_UNITS) | set(_TENS)
                       | {f"{t}-{u}" for t in _TENS for u in _UNITS[1:10]},
                       key=len, reverse=True)
_NUM = r"\b(\d+|(?i:" + "|".join(_NUMBER_WORDS) + r"))\b"   # "twenty-one" or "21"
_DEC = r"(\d\.\d{3})"                                          # a three-decimal value

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_GEOMETRY_DIR = os.path.join(REPO, "docs", "data", "yolo_geometry_51")


def _yolo_tiles_splits():
    """How many splits y11x_tiles was scored on: one ``<split>_tiles.txt`` report each."""
    if not os.path.isdir(YOLO_GEOMETRY_DIR):
        return None
    return sum(1 for f in os.listdir(YOLO_GEOMETRY_DIR) if f.endswith("_tiles.txt"))


def prose_facts(result):
    """The counts and values the hand-written prose quotes, each read off the board.

    Ints are compared as counts (``word_number``), floats at three decimals, strings
    exactly. ``yolo_tiles_splits`` is the one fact not on the board: it counts the committed
    reports in ``docs/data/yolo_geometry_51/``.
    """
    per = result["per_split"]
    pooled = result["pooled_splits"]
    rampnet_top = sum(
        1 for s in result["all_splits"]
        if "rampnet" in per and s in per["rampnet"]
        and all(per["rampnet"][s]["f1"] >= cells[s]["f1"]
                for cells in per.values() if s in cells))

    def f1s(key, splits):
        return [per[key][s]["f1"] for s in splits]

    def spread(key, splits):
        values = f1s(key, splits)
        return max(values) - min(values)

    # Finding 2: RampNet's range over the pool, and over the pool without its worst split.
    worst = min(pooled, key=lambda s: per["rampnet"][s]["f1"])
    rest = [s for s in pooled if s != worst]
    challengers = [m for m in result["models"] if m["complete"] and m["model"] != "rampnet"]
    strong = [m["model"] for m in challengers if m["f1"] > 0.1]
    open_vocab = [m["model"] for m in challengers if m["class"] == "open-vocab"]
    best = max(challengers, key=lambda m: m["f1"])
    rampnet = next(m for m in result["models"] if m["model"] == "rampnet")
    curves = result.get("curves") or {}
    splits = result["splits"]
    city_gt = sum(splits[s]["n_gt"] for s in result["city_splits"])
    gold_gt = splits[result["in_distribution_split"]]["n_gt"]

    facts = {
        "legs": len(result["models"]),
        "splits": len(result["all_splits"]),
        "pooled": len(pooled),
        "pooled_minus_one": len(pooled) - 1,
        "held_out": len(result["held_out"]),
        "complete": sum(1 for m in result["models"] if m["complete"]),
        "rampnet_top": rampnet_top,
        "city_splits": len(result["city_splits"]),
        "city_gt": city_gt,
        "manual_gold_share": round(100 * gold_gt / (gold_gt + city_gt)),
        "yolo_tiles_splits": _yolo_tiles_splits(),
        # the headline lead, over the best challenger with full pooled coverage
        "best_challenger": best["display"].split(" (")[0],
        "lead": rampnet["f1"] - best["f1"],
        # the two AP families (macro: the table's column; micro: the PR-curve legend)
        "ap_gap_macro": rampnet["ap"] - max(m["ap"] for m in challengers
                                            if m.get("ap") is not None),
        "ap_gap_micro": curves["rampnet"]["ap"] - max(
            c["ap"] for k, c in curves.items() if k != "rampnet"),
        # Finding 2
        "rampnet_min": min(f1s("rampnet", pooled)),
        "rampnet_max": max(f1s("rampnet", pooled)),
        "rampnet_range": spread("rampnet", pooled),
        "rampnet_worst_split": worst,
        "rampnet_worst_f1": per["rampnet"][worst]["f1"],
        "worst_share": spread("rampnet", pooled) - spread("rampnet", rest),
        "rest_min": min(f1s("rampnet", rest)),
        "rest_max": max(f1s("rampnet", rest)),
        "rest_range": spread("rampnet", rest),
        "strong_range_min": min(spread(k, pooled) for k in strong),
        "strong_range_max": max(spread(k, pooled) for k in strong),
        "strong_rest_min": min(spread(k, rest) for k in strong),
        "strong_rest_max": max(spread(k, rest) for k in strong),
    }
    for i, key in enumerate(open_vocab):
        facts[f"open_vocab_range_{i}"] = spread(key, pooled)
    return facts


def _agrees(word, value):
    """Does the prose's ``word`` state the board's ``value``?"""
    if isinstance(value, float):
        return word == f"{value:.3f}"
    if isinstance(value, int):
        return word_number(word.replace(",", "")) == value
    return word == value


# (name, regex, [fact per captured group], required). Names are unique, one per sentence,
# and matches are tracked per rule: a required rule is satisfied only by its own sentence,
# so rewording that sentence reports it missing even when a similar sentence elsewhere
# still matches a sibling rule. Unrequired rules fire wherever the phrase occurs, which is
# what catches a stale sentence pasted back in from an older copy of the page.
PROSE_RULES = (
    # --- counts, each anchored to the one sentence that states it -----------------------
    ("intro: legs", _NUM + r" model legs\b", ["legs"], True),
    ("intro: splits and pooled", _NUM + r" splits \(" + _NUM + r" of them pooled\)",
     ["splits", "pooled"], True),
    ("board: pooled", r"\bMacro-mean over the " + _NUM + r" pooled US city splits\b",
     ["pooled"], True),
    ("how to read: pool size",
     r"\bThe pool is " + _NUM + r" cities, not all " + _NUM + r" splits\b",
     ["pooled", "splits"], True),
    ("how to read: held out", _NUM + r" held-out splits\b", ["held_out"], True),
    ("matrix: top score", r"\btop score in " + _NUM + r" of the " + _NUM + r" splits\b",
     ["rampnet_top", "splits"], True),
    ("matrix: complete legs", r"\bof the " + _NUM + r" models with full pooled coverage\b",
     ["complete"], True),
    ("macro: manual_gold share",
     r"\boutnumber all " + _NUM + r" city splits combined \(([\d,]+)\), so a count-pooled "
     r"headline over all " + _NUM + r" would be (\d+)% one split\b",
     ["city_splits", "city_gt", "splits", "manual_gold_share"], True),
    ("missing: y11x_tiles scope",
     r"\bscored on 2026-08-30 on the " + _NUM + r" splits registered then\b",
     ["yolo_tiles_splits"], True),
    # --- values, each anchored to the one sentence that states it -----------------------
    ("headline: lead",
     r"\bRampNet leads the best challenger with full pooled coverage \(([^)]+)\) by "
     + _DEC + r" F1\b", ["best_challenger", "lead"], True),
    ("finding 2: RampNet range",
     r"\bOver the " + _NUM + r" its F1 spans " + _DEC + "–" + _DEC + r", a range of " + _DEC,
     ["pooled", "rampnet_min", "rampnet_max", "rampnet_range"], True),
    ("finding 2: challenger range",
     r"\bevery challenger with full pooled coverage scoring above 0\.1 spans " + _DEC
     + r" \([^)]+\) to " + _DEC + r" \(", ["strong_range_min", "strong_range_max"], True),
    ("finding 2: worst split",
     r"\bMost of its range \(" + _DEC + " of " + _DEC + r"\) is `(\w+)` \(" + _DEC + r"\)",
     ["worst_share", "rampnet_range", "rampnet_worst_split", "rampnet_worst_f1"], True),
    ("finding 2: without the worst split",
     r"\bover the other " + _NUM + r" cities it spans " + _DEC + "–" + _DEC
     + r", a range of " + _DEC + r", against " + _DEC + r" \([^)]+\) to " + _DEC + r" \(",
     ["pooled_minus_one", "rest_min", "rest_max", "rest_range",
      "strong_rest_min", "strong_rest_max"], True),
    ("finding 2: open-vocab range",
     r"\bThe two open-vocabulary detectors \*are\* flatter \(" + _DEC + ", " + _DEC + r"\)",
     ["open_vocab_range_0", "open_vocab_range_1"], True),
    ("AP: family gaps",
     r"\bmacro-to-macro the gap is " + _DEC + r", micro-to-micro " + _DEC,
     ["ap_gap_macro", "ap_gap_micro"], True),
    # --- the same facts in any other sentence, including a restored older one -----------
    # "the other seven pooled splits" counts the pool minus one, so "other" is excluded.
    ("anywhere: N pooled splits",
     r"(?<!other )" + _NUM + r" pooled (?:US )?(?:city )?(?:splits|cities)\b",
     ["pooled"], False),
    ("anywhere: over the N US splits", r"\bover the " + _NUM + r" US (?:city )?splits\b",
     ["pooled"], False),
    ("anywhere: across the N cities",
     r"\bacross the " + _NUM + r" (?:pooled )?(?:US )?(?:city )?(?:cities|splits)\b",
     ["pooled"], False),
    ("anywhere: all N cities combined", r"\ball " + _NUM + r" cit(?:y splits|ies) combined\b",
     ["city_splits"], False),
    ("anywhere: RampNet wins by", r"\bRampNet wins by " + _DEC + r" F1\b", ["lead"], False),
    ("anywhere: challengers above 0.1",
     r"\bscoring above 0\.1 (?:spans|swings between) " + _DEC + r" \([^)]+\) (?:to|and) "
     + _DEC + r" \(", ["strong_range_min", "strong_range_max"], False),
    ("anywhere: macro-to-macro gap", r"\bmacro-to-macro the gap is " + _DEC,
     ["ap_gap_macro"], False),
    # --- retired forms. The first two cannot be true while there are held-out splits and
    # y11x_tiles is off the board; the third is false as long as a strong challenger's range
    # is narrower than RampNet's (0.182 against 0.311 on 2026-09-25).
    ("retired: top score in all splits", r"\btop score in all " + _NUM + r" splits\b",
     [], False),
    ("retired: scored on all splits", r"\bscored on all " + _NUM + r" splits\b", [], False),
    ("retired: only stable strong model",
     r"\bthe only strong model that is also stable\b", [], False),
)
assert len({name for name, *_ in PROSE_RULES}) == len(PROSE_RULES), "rule names must be unique"


def prose_problems(text, result):
    """Every count or value in the hand-written prose that disagrees with the board.

    Generated blocks are stripped first (they are checked by the splice), and whitespace
    is collapsed because the doc wraps its prose. Returns [] when the prose is current.

    >>> prose_problems("Eighteen model legs, ten splits.", board)   # doctest: +SKIP
    ['intro: legs: "Eighteen model legs" says Eighteen, the board has 21 (legs)', ...]
    """
    prose = re.sub(r"<!-- BEGIN GENERATED: (\S+) .*?<!-- END GENERATED: \1 -->", "",
                   text, flags=re.S)
    prose = re.sub(r"\s+", " ", prose)
    facts = prose_facts(result)
    problems = []
    for name, pattern, keys, required in PROSE_RULES:
        found = list(re.finditer(pattern, prose))
        if required and not found:
            problems.append(f"{name}: no sentence quotes it any more -- restore it, or "
                            "drop the rule in scoreboard_render.PROSE_RULES")
        for m in found:
            if not keys:
                problems.append(f'{name}: "{m.group(0)}" is no longer true of the board')
                continue
            for word, key in zip(m.groups(), keys):
                if not _agrees(word, facts[key]):
                    shown = (f"{facts[key]:.3f}" if isinstance(facts[key], float)
                             else facts[key])
                    problems.append(f'{name}: "{m.group(0)}" says {word}, the board has '
                                    f"{shown} ({key})")
    return sorted(set(problems))
