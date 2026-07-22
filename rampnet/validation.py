"""Precision/recall from human validation verdicts.

The counterpart to :mod:`rampnet.metrics`: where ``metrics`` derives its
``(confidence, is_true_positive)`` flags by spatially *matching* predicted peaks
against ground-truth keypoints, this module derives the same flags from a human
reviewer's per-detection verdicts and their marks for ramps the model *missed*.
Both feed the identical precision/recall definitions (precision = TP / (TP + FP),
recall = TP / total_gt), so a "corrected" model-eval number and a human-validation
number mean the same thing and can be reported side by side.

It is the scoring half of the deployment spot-check loop (a gallery renders
sampled detections, a reviewer judges each crop and marks missed ramps, exporting
a ``verdicts.json``); the deployment pipeline owns only the thin I/O adapter that
loads its ``results.jsonl`` + verdicts and calls :func:`collect` / :func:`format_report`.

Verdict schema (one entry per reviewed pano, keyed by pano id):
  - ``dets``:   per-detection verdict, aligned with the pano's detections —
                ``True`` (correct), ``False`` (incorrect), ``"unsure"``
                (abstains from both metrics), or ``None`` (not yet judged; the
                pano is then partially judged and unusable for either metric).
  - ``missed``: reviewer marks for ramps the model missed. Each is a point dict;
                ``{"unsure": True}`` abstains (kept out of the recall denominator).
  - ``no_missed``: present on entries exported by a gallery with the missed-ramp
                confirmation feature — the pano's false-negative check is
                *confirmed* only if ``no_missed`` is set or a missed mark exists.
                Entries without the key are legacy and trusted (see :func:`collect`).
  - ``group``:  sampling group; ``"top"`` are the always-included densest panos,
                excludable from unbiased estimates via ``exclude_top``.

Pure-Python (no torch); ``pytest tests/test_validation.py``.
"""
import math
from collections import namedtuple

# Confidence thresholds for the human-readable sweep. The lowest matches the
# detector's own peak threshold (peak_local_max threshold_abs), so the first row
# is "every emitted detection".
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# The verdict pools produced by collect(). judged/recall_judged are lists of
# (confidence, is_correct) — the same shape rampnet.metrics.match_predictions
# returns — so the precision/recall math below is shared between the two sources.
Pools = namedtuple('Pools', [
    'judged',         # PRECISION pool: decided (conf, is_correct) on fully-judged panos
    'recall_judged',  # RECALL pool: the subset from panos whose missed-ramp check is confirmed
    'missed_total',   # confident missed-ramp marks on the recall-pool panos (the FN count)
    'n_seen',         # panos whose verdicts line up with results.jsonl
    'n_judged',       # of those, fully judged (no None verdicts)
    'n_unconfirmed',  # fully-judged panos held out of recall (missed-ramp check unconfirmed)
    'n_unsure',       # detections marked 'unsure' (abstained)
    'missed_unsure',  # missed marks flagged unsure (abstained)
    'warnings',       # human-readable notes (e.g. verdict/results mismatch) — presentation-free
])


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def collect(panos, confs_by_pid, exclude_top=False, assume_scanned=False):
    """Turn reviewer verdicts into precision/recall pools.

    panos:        {pano_id: verdict entry} (see the schema in the module docstring).
    confs_by_pid: {pano_id: [detection confidence, ...]} from the run's results.

    Precision counts every *decided* detection on panos whose detections are all
    judged. Recall additionally requires the pano's missed-ramp check to be
    confirmed, so its numerator (correct detections) and denominator (those plus
    confident missed marks) cover the same panos.

    ``assume_scanned=True`` treats every fully-judged pano as false-negative
    checked, regardless of the per-pano flag — reviewer attestation that they
    scanned each pano for misses without clicking "no missed ramps" on the clean
    ones. It is an explicit override; the gate otherwise exists so panos nobody
    scanned can't inflate recall.

    'unsure' abstention: an 'unsure' crop verdict and a missed mark with
    ``{'unsure': True}`` are the reviewer saying "can't tell from this imagery" —
    dropped from both metrics (forcing a guess would bias either) and reported as
    separate counts. A pano with unsure marks still counts as fully judged.

    The missed-ramp check is per entry: new-schema entries carry ``no_missed`` and
    must have it set or have a missed mark; legacy entries (no key) are trusted.
    Mixed old/new files therefore score correctly.

    This gate mirrors reviewed()/fnChecked() in the spot-check viewer JS — keep the
    two in sync. Returns a :class:`Pools`.
    """
    judged, recall_judged, missed_total, missed_unsure = [], [], 0, 0
    n_seen = n_judged = n_unconfirmed = n_unsure = 0
    warnings = []
    for pid, entry in panos.items():
        if exclude_top and entry.get('group') == 'top':
            continue
        confs = confs_by_pid.get(pid)
        if confs is None or len(confs) != len(entry['dets']):
            warnings.append(f"skipping {pid}: verdicts don't match results detections")
            continue
        n_seen += 1
        if any(d is None for d in entry['dets']):
            continue  # partially judged: unusable for either metric
        n_judged += 1
        n_unsure += sum(1 for d in entry['dets'] if d == 'unsure')
        # Decided verdicts only (True/False); 'unsure' abstains from both metrics.
        pano_judged = [(c, d) for c, d in zip(confs, entry['dets']) if d != 'unsure']
        judged += pano_judged
        fn_checked = True if assume_scanned else (
            (entry['no_missed'] or entry['missed']) if 'no_missed' in entry else True)
        if fn_checked:
            recall_judged += pano_judged
            missed_total += sum(1 for m in entry['missed'] if not m.get('unsure'))
            missed_unsure += sum(1 for m in entry['missed'] if m.get('unsure'))
        else:
            n_unconfirmed += 1
    return Pools(judged, recall_judged, missed_total, n_seen, n_judged,
                 n_unconfirmed, n_unsure, missed_unsure, warnings)


# --- Shared precision/recall primitives -----------------------------------
# One definition each, used by both the headline numbers and the threshold
# sweep so they can't diverge; identical in form to the precision (TP/(TP+FP))
# and recall (TP/total_gt) that rampnet.metrics computes for the matched path.

def _precision_at(judged, threshold=0.0):
    """(#correct, #kept) among detections with confidence >= threshold."""
    kept = [ok for c, ok in judged if c >= threshold]
    return sum(kept), len(kept)


def _recall_tp_at(recall_judged, threshold=0.0):
    """#correct detections in the recall pool with confidence >= threshold."""
    return sum(1 for c, ok in recall_judged if c >= threshold and ok)


def total_ramps(pools):
    """Recall denominator: correct detections in the recall pool + confident missed marks."""
    return _recall_tp_at(pools.recall_judged) + pools.missed_total


def format_report(title, pools, thresholds=THRESHOLDS):
    """Render one titled precision/recall block (headline + threshold sweep) as text.

    Returns the block as a string (no printing), so callers own presentation. Any
    per-pool notes in ``pools.warnings`` are the caller's to surface.
    """
    lines = [f"--- {title} ---",
             f"Panos fully judged:   {pools.n_judged} (of {pools.n_seen} seen)"]
    if pools.n_unconfirmed:
        lines.append(
            f"! {pools.n_unconfirmed} of those excluded from RECALL only: their "
            f"missed-ramp check was never confirmed\n  (mark the missed ramps in "
            f"the gallery, then re-export).")

    tp, n_judged_dets = _precision_at(pools.judged)
    fp = n_judged_dets - tp
    lines.append(f"Detections judged:    {n_judged_dets}  (correct {tp}, incorrect {fp})")
    if pools.n_unsure:
        lines.append(f"Detections unsure:    {pools.n_unsure}  (abstained — not in precision or recall)")
    lines.append(f"Missed ramps marked:  {pools.missed_total}"
                 + (f"  (+{pools.missed_unsure} unsure, abstained)" if pools.missed_unsure else ""))

    if not pools.judged:
        lines.append("Nothing judged yet.")
        return "\n".join(lines)

    p = tp / n_judged_dets
    lo, hi = wilson_interval(tp, n_judged_dets)
    lines.append(f"Precision: {p:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    denom = total_ramps(pools)
    rtp = _recall_tp_at(pools.recall_judged)
    if denom:
        r = rtp / denom
        rlo, rhi = wilson_interval(rtp, denom)
        lines.append(f"Recall:    {r:.3f}  (95% CI {rlo:.3f}-{rhi:.3f})  "
                     f"[vs ramps visible in the {pools.n_judged - pools.n_unconfirmed} recall-pool panos]")

    lines.append("")
    lines.append(f"{'threshold':>9}  {'kept':>5}  {'precision':>9}  {'recall':>7}")
    for t in thresholds:
        ktp, kept = _precision_at(pools.judged, t)
        if not kept:
            break
        prec = ktp / kept
        rtp_t = _recall_tp_at(pools.recall_judged, t)
        rec = rtp_t / denom if denom else float('nan')
        lines.append(f"{t:>9.2f}  {kept:>5}  {prec:>9.3f}  {rec:>7.3f}")
    return "\n".join(lines)
