"""One table with every model on it: the aggregated scoreboard behind docs/model_scoreboard.md.

``docs/model_comparison.md`` is the comprehensive log — per-split tables, the mechanism
behind each number, the caveats, the negative results. It is the right document for "why
does Qwen-32B invert on budapest". It is the wrong document for "which model is best, and
by how much", because that answer is spread across ten per-split tables in reading order
rather than model order.

This script produces the other view: **rows are models, columns are metrics**, aggregated
across splits, plus figures for the findings that are easier to see than to read. It is
generated rather than hand-maintained so the summary cannot drift from the log it
summarizes — ``--check`` fails when the committed doc no longer matches the committed
data, which is a failure mode a hand-copied summary table has and a generated one does not.

Reads **only committed artifacts** — the benchmark bundles, the published detections in
``benchmark/model_detections/``, and ``manual_labels/``. No ``.model_cache``, no GPU, no
credentials, no network, so a fresh clone reproduces every number here::

    python scripts/analysis/scoreboard.py                 # JSON + doc tables + figures
    python scripts/analysis/scoreboard.py --check         # doc AND JSON current?
    python scripts/analysis/scoreboard.py --no-figures    # tables only (no matplotlib)

The scoring path needs **numpy and pillow only** -- not ``requirements-dev.txt``, which
pulls the whole training stack for the rest of the suite.

**Nothing here may disagree with ``docs/model_comparison.md``.** Both documents score the
same committed detections, so a difference is a bug in one of them, not a choice.
``tests/test_scoreboard.py::test_every_number_matches_model_comparison`` parses every
per-split table out of the log and asserts that every (model, split) row in it agrees on
P, R, F1 and AP. The single deliberate exception is RampNet's AP -- see below -- and that
is asserted too, against ``ap_bundle``, so the exception cannot silently widen.

**Aggregation.** The headline is the **macro-mean over the seven US city splits** — the
pool is ``low_floor_sweep.US_SPLITS``, imported rather than restated, because a third copy
of that registry is how a split ends up silently in one headline and out of another. The
three held-out splits keep their own columns and their documented reasons travel with
them: ``docs/model_comparison.md`` states outright that budapest's numbers "must not be
pooled with the US splits or averaged into a headline", ``sao_paulo`` is held out for
geography rather than GT quality, and ``manual_gold`` is the in-distribution reference,
not a deployment city.

Macro rather than micro even within the pool: pooling counts would weight paterson (395 GT
ramps) twice as heavily as clovis (195) for no reason anyone would defend, and pooling
``manual_gold`` in would be worse still — its 3,919 GT points outnumber all nine cities
combined, so a pooled headline would be 59% one split that is in-distribution for exactly
one model on the roster.

**Operating points** are per model class, and are the ones the log already committed to
rather than new choices (see ``OPERATING_POINT``): RampNet at its deployed 0.55, the
supervised YOLO arms at the pre-registered conf 0.25 (#71), the open-vocabulary detectors
at their 0.05 export floor, and the chat VLMs wherever they are — they emit no confidence,
so there is nothing to threshold and the setting is a no-op for them.

**AP comes from a different file than P/R/F1, and only for RampNet.** The city bundles hold
RampNet's detections only down to its deployed 0.55, because they *are* a production run
and that is where production stops — so an AP computed from them integrates a curve that
has been cut off at the operating point. Read that way RampNet's pooled AP is 0.720 and
sits *below* the YOLO arms' 0.730, which is an artifact of the floor and not a result.
``analysis_out/op_cache/`` (committed, 928 KB, all ten splits) holds the same panoramas
re-extracted down to 0.05, which is the floor every other scored model is exported at, and
scoring RampNet's curve from it gives a pooled AP of **0.849**. So on the city splits:
P/R/F1 stay on the committed ``records.jsonl`` — the published, deployment-faithful
operating point — and AP and the PR curve come from the low-floor cache. Both are stated
wherever the number appears, because two sources for one row is exactly the kind of thing
that reads as an error later if it is not said out loud.

**This is the one place the scoreboard and the log print different numbers under the same
column heading**, so the truncated value is carried on every row as ``ap_bundle``, printed
in the page's AP-provenance table beside the substituted one, and asserted against the log
in the tests. A reader who finds 0.876 here and 0.763 there can see in one table why.

The substitution is gated on *measured* truncation (bundle floor more than 0.1 above the
cache floor), not on a list of split names. ``manual_gold``'s bundle is already exported at
0.05, so it keeps its own AP: there is no truncation to undo there, and swapping in the
cache would quietly trade that split's flip-TTA export for a no-TTA one — a different
change, and one that would leave a single row's AP and P/R/F1 describing two different
inference configurations.

One consequence worth carrying: the sub-0.55 half of that curve is scored against ground
truth assembled from detections at or above 0.55, so a real ramp nobody marked counts as a
false positive. #55 measured that 27.2% of the incremental false positives in [0.30, 0.55)
are ground-truth-completeness artifacts. RampNet's untruncated AP is therefore itself a
lower bound.
"""
import argparse
import functools
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import compare as C  # noqa: E402  (torch-free: detectors.py imports torch lazily)
from rampnet import roster  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    aggregate, prediction_confidence, radius_sq_for, score_pano)
from export_model_cache import PUBLISHED_DIR, load_detections  # noqa: E402
# The split registry, imported not restated. test_registries_agree_with_low_floor_sweep
# already keeps low_floor_sweep and miss_decomposition in step; a third private copy here
# is exactly how a split ends up pooled in one headline and held out of another.
from low_floor_sweep import (  # noqa: E402
    ALL_SPLITS, CACHE_DIR, CITY_SPLITS, DEPLOYED_THRESHOLD, HELD_OUT, US_SPLITS,
    load_split as load_low_floor)
from operating_point_curve import pr_curve_and_ap  # noqa: E402

IN_DISTRIBUTION = "manual_gold"
# Held-out splits in a stable reading order: the two cities, then the reference split.
HELD_OUT_ORDER = tuple(s for s in ALL_SPLITS if s in HELD_OUT)

RAMPNET = "rampnet"
# #54's recommendation, not yet adopted by the deployment consumer
# (sidewalk-auto-labeler#20 is open). Marked on the PR figure beside the deployed point so
# the choice is visible rather than asserted.
RECOMMENDED_THRESHOLD = 0.30

# Confidence floor each model class is reported at. These are not new choices -- each is
# the operating point its own write-up committed to, restated here so one table decides it
# for every row and figure instead of each caller deciding again. A model whose
# predictions carry no confidence (the chat VLMs) is unaffected either way: rescore()
# never drops an unscored prediction.
OPERATING_POINT = {
    "purpose-trained": DEPLOYED_THRESHOLD,   # RampNet as deployed (docs/operating_point.md)
    "supervised": 0.25,          # pre-registered YOLO headline (#71)
    "supervised-transfer": 0.0,  # Vistas mask components, at the floor #126 exported
    "open-vocab": 0.0,           # the 0.05 export floor, as the log reports them
    "chat-vlm": 0.0,             # no confidences -- no-op
    "pointing": 0.0,             # no confidences -- no-op
    "unclassified": 0.0,         # discovered leg: report it where it sits, and say so
}

# How to say that operating point in a table cell. "0.05 floor" and "no score" are not
# the same thing and the distinction is load-bearing: a floored detector could be tuned
# to a better point and is not being, whereas a chat VLM has no confidence to tune on, so
# its single row IS the model. Collapsing both to "0.00" invites the reading that the
# chat VLMs were handed an unfairly low threshold.
OPERATING_POINT_NOTE = {
    "purpose-trained": "0.55",
    "supervised": "0.25",
    "supervised-transfer": "export floor",
    "open-vocab": "0.05 floor",
    "chat-vlm": "no score",
    "pointing": "no score",
    "unclassified": "export floor",
}

# Model class per roster provider. The registry knows WHO has been run; this only says
# what kind of thing each provider is, which is the one fact a results table needs and
# the roster does not carry. An unknown provider falls through to "unclassified" and is
# still scored -- silently dropping a leg someone paid for is the failure this file
# exists to prevent.
PROVIDER_CLASS = {
    "rampnet": "purpose-trained",
    "yolo": "supervised",
    "vistas": "supervised-transfer",
    "gemini": "chat-vlm",
    "claude": "chat-vlm",
    "qwen": "chat-vlm",
    "molmo": "pointing",
    "owlv2": "open-vocab",
    "gdino": "open-vocab",
}

CLASS_ORDER = ("purpose-trained", "supervised", "supervised-transfer", "chat-vlm",
               "pointing", "open-vocab", "unclassified")
CLASS_LABEL = {
    "purpose-trained": "purpose-trained",
    "supervised": "supervised baseline",
    "supervised-transfer": "supervised transfer",
    "chat-vlm": "chat VLM",
    "pointing": "pointing model",
    "open-vocab": "open-vocab detector",
    "unclassified": "unclassified",
}

# Short, readable stand-ins for the long published ids, keyed by published name (unique
# by construction -- it is a filename stem). Anything absent falls back to the id itself,
# so a new leg is readable-but-verbose rather than missing.
DISPLAY = {
    "rampnet": "RampNet",
    "y11x_pano_h200": "YOLO11x (pano)",
    "y11l_pano": "YOLO11l (pano)",
    "y26_pano": "YOLO26 (pano)",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "allenai/Molmo2-8B": "Molmo2-8B",
    "google/owlv2-large-patch14-ensemble": "OWLv2-large",
    "IDEA-Research/grounding-dino-base": "Grounding DINO",
    "claude-opus-5-effort-low": "Claude Opus 5 (low)",
    "claude-opus-5-effort-high": "Claude Opus 5 (high)",
    "claude-sonnet-5-effort-low": "Claude Sonnet 5 (low)",
    "claude-sonnet-5-effort-high": "Claude Sonnet 5 (high)",
    "mask2former-vistas-curb-cut": "Mask2Former Vistas (curb cut)",
    "mask2former-vistas-curb-cut+curb": "Mask2Former Vistas (+curb)",
}

DEFAULT_JSON = os.path.join(REPO, "analysis_out", "scoreboard.json")
DEFAULT_DOC = os.path.join(REPO, "docs", "model_scoreboard.md")
FIGURE_DIR = os.path.join(REPO, "docs", "figures")


def class_of(leg):
    """Model class for a roster leg; an unregistered provider is named, never dropped."""
    return PROVIDER_CLASS.get(leg.provider, "unclassified")


def display_of(leg):
    """Short readable name, falling back to the published id."""
    name = roster.published_name(leg)
    return DISPLAY.get(name, name)


def legs():
    """Every registered leg, in roster order (which is results-table order).

    Driven by ``rampnet.roster`` rather than by scanning ``benchmark/model_detections/``:
    filenames are ambiguous now that one model id can be several legs
    (``claude-opus-5`` at two efforts publishes as two stems) and that a published name
    can itself contain the ``__`` separator (``mask2former-vistas-curb-cut__curb``).
    ``test_roster.py`` already asserts the directory and the registry agree, so the
    registry is the safe side of that pair to read from.
    """
    return list(roster.ROSTER)


def unregistered_exports():
    """Published detection files no roster entry claims — reported, never scored silently.

    ``test_roster.py`` should make this empty. It is computed anyway so that if the two
    ever drift, the scoreboard says so instead of quietly omitting a paid-for run.
    """
    known = {roster.published_filename(leg, split)
             for leg in roster.ROSTER for split in ALL_SPLITS}
    return sorted(name for name in os.listdir(PUBLISHED_DIR)
                  if name.endswith(".json") and name not in known)


@functools.lru_cache(maxsize=None)
def low_floor_panos(split):
    """The low-floor cache for one split, or None. Memoized: read up to 4x per build.

    ``score`` needs it twice per split (floor + AP) and ``pooled_curve`` /
    ``rampnet_marks`` need it again, and it is the same file every time.
    """
    if not os.path.exists(os.path.join(CACHE_DIR, f"{split}.json")):
        return None
    return load_low_floor(split)[0]


def low_floor_floor(split):
    """The lowest confidence the low-floor cache stored for a split, or None."""
    panos = low_floor_panos(split)
    if panos is None:
        return None
    return min((c for pd in panos for *_xy, c in pd["preds"]), default=None)


def low_floor_report(split, radius_sq):
    """RampNet's untruncated PR curve + AP for one split, or None if uncached.

    ``op_cache`` is the #54 extraction: the same panoramas, same preprocessing, same
    no-TTA deployment path, every heatmap peak down to 0.05. Its >=0.55 slice is gated to
    reproduce the committed records (``low_floor_sweep.py parity``), so it is the same run
    seen further down rather than a different one.
    """
    panos = low_floor_panos(split)
    return None if panos is None else pr_curve_and_ap(panos, radius_sq)


def bundle_floor(preds):
    """The lowest confidence present in a bundle's detections, or None if unscored."""
    return min((c for pts in preds.values() for c in
                (prediction_confidence(d) for d in pts) if c is not None), default=None)


def uses_low_floor_cache(split, preds):
    """Is RampNet's bundle for ``split`` truncated far enough above the cache to swap?

    Decided by *measuring* both floors rather than by naming splits. One function so the
    AP column (``score``) and the PR curve (``pooled_curve``) can never disagree about
    which source a split is read from -- a divergence that would put a curve and the AP
    printed beside it on two different runs.
    """
    if low_floor_panos(split) is None:
        return False
    bf, cf = bundle_floor(preds), low_floor_floor(split)
    return bf is not None and cf is not None and bf - cf > 0.1


def load_split(split):
    """(records, {pano_id: GroundTruth}) for one benchmark bundle."""
    bundle = os.path.join(REPO, "benchmark", split)
    records, verdicts, _ = C.load_bundle(bundle)
    if verdicts is not None:
        C.validate_bundle(records, verdicts)
        return records, C.ground_truths_from_verdicts(records, verdicts)
    gts = C.load_manual_ground_truths(bundle)
    return records, {pid: gts[pid] for pid in records if pid in gts}


def score(leg, split, records, gts, radius_sq, op):
    """One (leg, split) cell, or None when the leg was never run on that split.

    P/R/F1 are read at the operating point ``op``; AP is read from the full-range run,
    because ``--op-threshold`` truncates the curve it integrates. That split is
    ``compare.operating_report``'s, reused here so the scoreboard and the log cannot
    disagree about what a row means.
    """
    if leg.provider == "rampnet":
        preds = {pid: records[pid]["detections"] for pid in gts}
    else:
        # publish_as, not label: a pinned leg's detections live under its own stem, and
        # loading by label alone silently returns the sibling leg's file.
        preds = load_detections(leg.label, split,
                                publish_as=roster.published_name(leg))
        if preds is None:
            return None
    scored = [(preds.get(pid, []), gt) for pid, gt in gts.items()]
    full = aggregate([score_pano(p, g, radius_sq=radius_sq) for p, g in scored])
    rep = C.operating_report(full, scored, radius_sq, op)

    # RampNet's bundle curve is truncated at its deployed threshold; the committed
    # low-floor cache carries the same run down to 0.05. See the module docstring.
    #
    # Substituted only where truncation actually exists, decided by measuring the two
    # floors rather than by naming splits. manual_gold's bundle is ALREADY at 0.05, so
    # there is nothing to fix there -- and swapping in the cache would silently trade a
    # flip-TTA export for a no-TTA one, which is a different change from un-truncating a
    # curve and would leave that row's AP and its P/R/F1 describing two different
    # inference configs.
    ap, ap_source = rep.ap, "bundle"
    if leg.provider == "rampnet" and uses_low_floor_cache(split, preds):
        low = low_floor_report(split, radius_sq)
        if low is not None and low.ap is not None:
            ap, ap_source = low.ap, "op_cache (0.05 floor)"
    return {
        "split": split,
        "precision": rep.precision, "recall": rep.recall, "f1": rep.f1, "ap": ap,
        "ap_source": ap_source,
        # The AP as computed from the bundle alone -- i.e. exactly the number
        # docs/model_comparison.md's per-split tables print. Carried on every row, not
        # just the substituted ones, so the two documents can be diffed mechanically
        # (test_every_number_matches_model_comparison) instead of by eye.
        "ap_bundle": rep.ap,
        # The lowest confidence the bundle actually carries. Three states have to be
        # told apart downstream and "was it substituted?" only distinguishes two:
        # substituted (truncated, cache swapped in), not substituted because the
        # bundle is already at 0.05 (manual_gold), and not substituted because no
        # low-floor cache exists for that split at all (laurens_gsv). The last one is
        # still a truncated AP, and reporting it as "already at 0.05" is a false
        # provenance claim -- which is what the generated table did until this existed.
        "bundle_floor": bundle_floor(preds),
        "tp": rep.tp, "fp": rep.fp, "fn": rep.fn,
        "n_panos": rep.n_panos, "n_gt_recall": rep.n_gt_recall,
        "fp_per_pano": rep.fp / rep.n_panos if rep.n_panos else None,
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(leg, cells):
    """Macro-mean over the pooled US splits the leg actually ran on, + the held-out ones.

    ``coverage`` travels with every aggregate so a model averaged over one city can never
    be read as one averaged over seven — and ``complete`` decides which table it lands
    in, because a one-city mean has no business sitting in the same column as a
    seven-city one. The held-out splits are carried individually, never folded in, for
    the reasons in ``HELD_OUT``.
    """
    klass = class_of(leg)
    pooled = [cells[s] for s in US_SPLITS if cells.get(s)]
    f1s = [c["f1"] for c in pooled]
    have_ap = bool(pooled) and all(c["ap"] is not None for c in pooled)
    have_bundle_ap = bool(pooled) and all(c["ap_bundle"] is not None for c in pooled)
    # Micro (count-pooled) P/R alongside the macro-mean. The PR-curve figure integrates
    # over concatenated panoramas, so a point drawn on those axes has to be micro too --
    # plotting the macro-mean there would put the dots and the lines in two different
    # aggregations under one subtitle. The headline table stays macro.
    tp = sum(c["tp"] for c in pooled)
    fp = sum(c["fp"] for c in pooled)
    fn = sum(c["fn"] for c in pooled)
    out = {
        "model": roster.published_name(leg),
        "label": leg.label,
        "spec": leg.spec,
        "provider": leg.provider,
        "standing": leg.standing,
        "display": display_of(leg),
        "class": klass,
        "operating_point": OPERATING_POINT[klass],
        "operating_point_note": OPERATING_POINT_NOTE[klass],
        "coverage": f"{len(pooled)}/{len(US_SPLITS)}",
        "complete": len(pooled) == len(US_SPLITS),
        "pooled_splits": [s for s in US_SPLITS if cells.get(s)],
        "precision": mean(c["precision"] for c in pooled),
        "recall": mean(c["recall"] for c in pooled),
        "f1": mean(f1s),
        "ap": mean(c["ap"] for c in pooled) if have_ap else None,
        # Macro-mean of the per-split BUNDLE AP -- the aggregate of exactly the numbers
        # docs/model_comparison.md prints. For every model but RampNet it equals "ap";
        # for RampNet it is the 0.55-truncated figure the substitution exists to replace,
        # kept so the page can show what it replaced rather than assert it.
        "ap_bundle": mean(c["ap_bundle"] for c in pooled) if have_bundle_ap else None,
        "ap_is_substituted": any(c["ap_source"] != "bundle" for c in pooled),
        "fp_per_pano": mean(c["fp_per_pano"] for c in pooled),
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "f1_min": min(f1s) if f1s else None,
        "f1_max": max(f1s) if f1s else None,
        "f1_min_split": min(pooled, key=lambda c: c["f1"])["split"] if pooled else None,
        "f1_max_split": max(pooled, key=lambda c: c["f1"])["split"] if pooled else None,
        "n_splits_run": len([s for s in ALL_SPLITS if cells.get(s)]),
    }
    for split in HELD_OUT_ORDER:
        cell = cells.get(split)
        for metric in ("f1", "precision", "recall", "ap"):
            out[f"{split}_{metric}"] = cell[metric] if cell else None
    return out


# Curves are pooled over the seven US splits so the figure has one line per model rather
# than seven. Unlike the headline table this pooling is MICRO (every panorama counts once,
# so a bigger split pulls harder) -- a PR curve is an integral over predictions and has no
# natural macro form. Stated on the figure.
def pooled_curve(leg, splits, radius_sq):
    """(ScoreReport, marked operating points) pooled across US_SPLITS, or None.

    None when the leg carries no confidences (a chat VLM has one operating point, not a
    curve) or has not run every pooled split.
    """
    pano_scores, has_conf, n_seen = [], True, 0
    for split in US_SPLITS:
        records, gts = splits[split]
        if leg.provider == "rampnet":
            bundle = {pid: records[pid]["detections"] for pid in gts}
            # Same gate the AP column uses, so the curve and the AP printed in its legend
            # are always read from the same source.
            if uses_low_floor_cache(split, bundle):
                for pd in low_floor_panos(split):
                    pano_scores.append(score_pano(pd["preds"], pd["gt"],
                                                  radius_sq=radius_sq))
            else:
                for pid, gt in gts.items():
                    pano_scores.append(score_pano(bundle[pid], gt, radius_sq=radius_sq))
            n_seen += 1
            continue
        preds = load_detections(leg.label, split, publish_as=roster.published_name(leg))
        if preds is None:
            return None
        n_seen += 1
        for pid, gt in gts.items():
            pts = preds.get(pid, [])
            if any(prediction_confidence(q) is None for q in pts):
                has_conf = False
            pano_scores.append(score_pano(pts, gt, radius_sq=radius_sq))
    if n_seen != len(US_SPLITS) or not has_conf:
        return None
    report = aggregate(pano_scores)
    return report if report.pr_curve else None


def rampnet_marks(splits, radius_sq, thresholds=(DEPLOYED_THRESHOLD, RECOMMENDED_THRESHOLD)):
    """(recall, precision) on the pooled curve at each named threshold.

    The point of drawing the curve is to choose a point on it, so the two the project has
    actually argued about are marked: the deployed 0.55 and #54's recommended 0.30.
    """
    panos = []
    for split in US_SPLITS:
        cached = low_floor_panos(split)
        if cached is None:
            return {}
        panos.extend(cached)
    out = {}
    for t in thresholds:
        rep = aggregate([score_pano([q for q in pd["preds"] if q[2] >= t], pd["gt"],
                                    radius_sq=radius_sq) for pd in panos])
        out[f"{t:.2f}"] = {"recall": rep.recall, "precision": rep.precision, "f1": rep.f1}
    return out


def build(models=None):
    """Score every (leg, split) pair from committed data. Returns the result dict.

    ``models`` filters by published name, so ``--models claude-opus-5-effort-low`` names
    one leg rather than one model id.
    """
    radius_sq = radius_sq_for()
    wanted = set(models) if models else None
    chosen = [leg for leg in legs()
              if wanted is None or roster.published_name(leg) in wanted]
    splits = {s: load_split(s) for s in ALL_SPLITS}

    per_model, summaries = {}, {}
    for leg in chosen:
        op = OPERATING_POINT[class_of(leg)]
        cells = {}
        for split, (records, gts) in splits.items():
            cell = score(leg, split, records, gts, radius_sq, op)
            if cell is not None:
                cells[split] = cell
        name = roster.published_name(leg)
        per_model[name] = cells
        summaries[name] = summarize(leg, cells)

    # Complete rows first, then class (so the table reads as a taxonomy), then F1.
    # A leg with no pooled coverage at all sorts last within its class rather than
    # ranking as if it scored zero.
    def key(name):
        s = summaries[name]
        return (not s["complete"], CLASS_ORDER.index(s["class"]),
                -(s["f1"] if s["f1"] is not None else -1))

    order = sorted(summaries, key=key)

    # Pooled PR curves for the models that carry confidences. The point arrays are
    # PLOT-ONLY and are dropped before the JSON is written (scoreboard_render.
    # json_payload): the two open detectors alone carry ~120k points, which is 7.7 MB of
    # committed artifact for something no reader diffs and the figure rebuilds in 3 s.
    # The AP, the marks and the point count survive, which is what the page cites.
    curves = {}
    for leg in chosen:
        rep = pooled_curve(leg, splits, radius_sq)
        if rep is not None:
            curves[roster.published_name(leg)] = {
                "recalls": list(rep.pr_curve[0]), "precisions": list(rep.pr_curve[1]),
                "ap": rep.ap,
                "n_points": len(rep.pr_curve[0]),
                "marks": (rampnet_marks(splits, radius_sq)
                          if leg.provider == "rampnet" else {}),
            }
    return {
        "pooled_splits": list(US_SPLITS),
        "city_splits": list(CITY_SPLITS),
        "all_splits": list(ALL_SPLITS),
        "held_out": dict(HELD_OUT),
        "in_distribution_split": IN_DISTRIBUTION,
        "unregistered_exports": unregistered_exports(),
        "curves": curves,
        "splits": {s: {"n_panos": len(gts),
                       "n_gt": sum(len(g.gt_points) for g in gts.values()),
                       "pooled": s in US_SPLITS}
                   for s, (_, gts) in splits.items()},
        "models": [summaries[n] for n in order],
        "per_split": {n: per_model[n] for n in order},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", default=DEFAULT_JSON)
    ap.add_argument("--doc", default=DEFAULT_DOC)
    ap.add_argument("--figure-dir", default=FIGURE_DIR)
    ap.add_argument("--models",
                    help="Comma-separated PUBLISHED names (roster.published_name), e.g. "
                         "claude-opus-5-effort-low. Scores a SUBSET, so it will not "
                         "write the committed doc, JSON or figures unless you also pass "
                         "--doc/--json-out/--figure-dir explicitly. Default: every leg.")
    ap.add_argument("--no-figures", action="store_true", help="Skip matplotlib entirely.")
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed doc AND analysis_out/scoreboard.json match "
                         "the committed data; write nothing and exit non-zero on drift.")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    # Refused before scoring, not after: --check on a subset would compare a partial
    # board against the full committed page and report the page as stale, which is a
    # false alarm and an expensive one to sit through.
    if args.check and models:
        print("--check scores every leg; drop --models")
        raise SystemExit(2)

    # A subset run produces a partial board. Splicing that into the committed page leaves
    # a one-row headline table sitting above prose about a twelve-model field, and the
    # next --check then reports the REAL page as stale -- the signal inverted. So a
    # subset run is read-only against the committed artifacts unless the caller names
    # different destinations.
    explicit = {a.split("=")[0] for a in sys.argv[1:]}
    subset_guard = bool(models) and not args.check
    write_doc = not subset_guard or "--doc" in explicit
    write_json_out = not subset_guard or "--json-out" in explicit
    write_figs = not args.no_figures and (not subset_guard or "--figure-dir" in explicit)

    result = build(models)

    from scoreboard_render import (  # noqa: E402
        json_payload, render_tables, splice, write_json)
    tables = render_tables(result)

    if args.check:
        problems = []
        if not os.path.exists(args.doc):
            problems.append(f"{args.doc}: missing")
        else:
            with open(args.doc, encoding="utf-8", newline="") as fh:
                current = fh.read()
            if splice(current, tables) != current:
                problems.append(f"{args.doc}: generated tables are stale "
                                "(re-run scripts/analysis/scoreboard.py)")
        # The JSON is a committed artifact too, and nothing else checks it. Compared as
        # bytes, which also catches a CRLF flip that a value-level compare would miss.
        if not os.path.exists(args.json_out):
            problems.append(f"{args.json_out}: missing")
        else:
            with open(args.json_out, "rb") as fh:
                on_disk = fh.read()
            if on_disk != json_payload(result).encode("utf-8"):
                problems.append(f"{args.json_out}: stale or reformatted "
                                "(re-run scripts/analysis/scoreboard.py)")
        if problems:
            print("\n".join(problems))
            raise SystemExit(1)
        print(f"{args.doc}: current")
        print(f"{args.json_out}: current")
        return

    if subset_guard:
        print(f"--models given ({', '.join(models)}): scoring a subset, so the committed "
              "doc, JSON and figures are left alone.\nPass --doc/--json-out/--figure-dir "
              "to write a partial board somewhere else.")

    if write_json_out:
        write_json(args.json_out, result)
        print(f"wrote {args.json_out}")

    if write_doc and os.path.exists(args.doc):
        with open(args.doc, encoding="utf-8", newline="") as fh:
            current = fh.read()
        updated = splice(current, tables)
        if updated != current:
            with open(args.doc, "w", encoding="utf-8", newline="") as fh:
                fh.write(updated)
            print(f"updated {args.doc}")
        else:
            print(f"{args.doc}: already current")
    elif write_doc:
        print(f"{args.doc}: not found -- write the prose first, then re-run to fill "
              "the generated blocks")

    if write_figs:
        import scoreboard_figures
        for path in scoreboard_figures.render_all(result, args.figure_dir):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
