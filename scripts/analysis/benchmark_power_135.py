"""Can the benchmark resolve Run B? A power analysis of the tie bar (#135, #84).

Run A (#84) answered its question with a plateau: `manual_gold` max-F1 for epochs
2-8 spans 0.008, inside the pre-registered 0.01 tie bar, so no epoch in that range
is distinguishable from any other. Run B is 30-60 epochs with cosine decay --
1,675-3,350 GPU-hours -- and the obvious risk is that it returns *another*
unreadable plateau. Before spending the hours it is worth asking what effect size
this benchmark can actually resolve, which is answerable from committed data alone.

The pre-registered 0.01 tie bar is an **unpaired** bar: it comes from the counting
noise on one model's recall over 3,919 instances. But "is the Run B checkpoint
better than the Run A checkpoint" is a **paired** question -- both are scored on the
same 1,000 panoramas against the same ground truth, so the pano-to-pano difficulty
that dominates the unpaired noise is common to both and cancels. The quantity that
governs a paired comparison is the *discordance*: how many ramps one model finds and
the other misses. This script measures both, so the difference between them stops
being an argument and becomes a number.

What it computes, for every committed split and for pooled combinations:

- **Inventory** -- panos, recall-confirmed panos, ground-truth instances.
- **Unpaired noise floor** -- a cluster bootstrap that resamples *panoramas* (the
  sampling unit; ramps cluster within a pano, so an instance-level binomial
  understates the spread). Reports the design effect against the naive binomial that
  the 0.01 bar was derived from.
- **Paired noise floor** -- the same bootstrap applied to the *difference* between
  two detectors, plus the McNemar decomposition (b = A misses / B finds,
  c = A finds / B misses) that explains it.
- **MDE** -- the minimum detectable effect at 80% power, two-sided alpha 0.05,
  i.e. 2.80 x s.e. Also the 1.96 x s.e. bar, which is the "is this CI clear of zero"
  reading that the tie bar is closer to in spirit.

Every input is committed: `manual_labels/` and `benchmark/*/records.jsonl` +
`verdicts.json` for ground truth, `benchmark/model_detections/*.json` for the
challengers. No cluster access, no `.model_cache`, no GPU, no network.

    python scripts/analysis/benchmark_power_135.py
    python scripts/analysis/benchmark_power_135.py --bootstrap 50000 \
        --out-json docs/data/benchmark_power_135.json

A note on what the paired numbers stand in for. The comparison Run B would actually
make -- two Stage 2 checkpoints from one lineage -- has no committed per-pano data
(Run A's committed artifacts are downsampled PR curves, aggregate only). So the
paired s.e. is bracketed by real pairs that *are* committed, chosen so the bracket
holds by construction:

- `y11l_pano` vs `y11x_pano_h200` is the #51 training-budget/anneal comparison, the
  very result cited as the argument *for* Run B. Two different architectures trained
  to different budgets are **less** correlated than two epochs of one run, so their
  paired s.e. is an upper bound on the epoch-vs-epoch case.
- The same RampNet detections read at two nearby confidence thresholds are **more**
  correlated than two checkpoints, so they give the lower bound -- **on F1 only**.
  max-F1 re-picks its own threshold, so moving the read-out point cannot move it and
  its delta is identically zero for every row. That block therefore bounds F1 and is
  emitted as null for max-F1, rather than as a zero that reads like a measurement.

The true epoch-vs-epoch s.e. lies between. Both ends are reported rather than one
point estimate, because the bracket is the honest form of the answer.

ONE PROPERTY OF THE COMMITTED DATA THAT EVERY UNPAIRED ROW INHERITS. The nine city
bundles under `benchmark/*/records.jsonl` were published with a 0.55 peak floor;
`manual_gold` reaches 0.05. So on those nine splits the #54 operating point of 0.30
is BELOW every committed detection: filtering at it is a no-op, and the max-F1 sweep
peaks on an already-truncated curve (which is why `max_f1 == f1` exactly on those
rows). The inventory records `reference_min_confidence` and
`protocol_threshold_binds` per split so this is visible in the artifact instead of
having to be inferred, the unpaired table marks affected rows with `*`, and the
pooling comparison is a LOWER bound on what pooling would buy against untruncated
city curves. There is no clean uniform-0.30 RampNet arm in committed data;
`--reference rampnet_1pass` reads `analysis_out/op_cache` down to 0.05 on all ten
splits but is single-pass and missing seam detections (#132), so it bounds the size
of the effect rather than correcting it.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "model_comparison"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import compare  # noqa: E402  (torch-free: detectors.py defers every heavy import)
from export_model_cache import load_detections  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, prediction_confidence, radius_sq_for, score_pano,
)
from rampnet.geometry import dist_sq  # noqa: E402
from rampnet.metrics import greedy_match  # noqa: E402

#: Panorama x is cyclic, so the matcher must wrap at the 360 seam (#130/#140).
#: ``score_pano`` defaults to ``wrap_x=True`` for exactly this reason, and
#: ``match_detail`` below has to make the same choice or ``score_model``'s runtime
#: assertion fires -- which is how this was caught: the assertion is the whole point
#: of keeping two matchers, and it did its job.
WRAP_X = True

#: Two-sided alpha 0.05, 80% power. z_{0.975} + z_{0.80}.
Z_MDE = 2.8015854
#: The "95% CI clear of zero" bar, which is what a tie bar is closer to in spirit.
Z_CI = 1.959964

#: Protocol operating points. RampNet is the #54 uniform 0.30 recommendation; the
#: YOLO arms are the #71 pre-registered headline conf 0.25. A paired comparison uses
#: each model at its own protocol point, and reports the calibration-free max-F1
#: alongside -- the #84 amendment's reason for demanding both columns applies here
#: for the same reason: a fixed threshold confounds calibration with capability.
PROTOCOL_THRESHOLD = {"rampnet": 0.30, "rampnet_1pass": 0.30}
YOLO_THRESHOLD = 0.25
#: Every RampNet-family arm reads at the #54 point, including the Run A epoch dumps
#: (``run_a_epoch_3`` and friends, written by dump_peaks_from_cache.py). A prefix rule
#: rather than eight enumerated names, so a ninth epoch cannot silently pick up 0.25 --
#: which would surface as a large fake capability gap rather than as an error.
RAMPNET_PREFIXES = ("rampnet", "run_a_epoch_")
#: Label of the Run A epoch-N dump, for the measured epoch-pair matrix.
RUN_A_EPOCH = "run_a_epoch_{}"


def protocol_threshold(model):
    if model in PROTOCOL_THRESHOLD:
        return PROTOCOL_THRESHOLD[model]
    if model.startswith(RAMPNET_PREFIXES):
        return PROTOCOL_THRESHOLD["rampnet"]
    return YOLO_THRESHOLD


def discover_splits(repo):
    """Every committed benchmark bundle, in sorted order."""
    bench = Path(repo) / "benchmark"
    return sorted(d.name for d in bench.iterdir()
                  if (d / "records.jsonl").exists())


def load_split(repo, split):
    """(records_by_pid, {pid: GroundTruth}) for one bundle, via compare.py's loaders."""
    bundle = os.path.join(repo, "benchmark", split)
    records, verdicts, _ = compare.load_bundle(bundle)
    gts = (compare.load_manual_ground_truths(bundle) if verdicts is None
           else compare.ground_truths_from_verdicts(records, verdicts))
    return records, gts


def detections_for(repo, split, model, records):
    """{pid: [points]} for one model on one split, or None if it was never run.

    Three sources, because the RampNet arms predate the published-export format:

    - ``rampnet`` -- the bundle's own records, i.e. flip-TTA, which is what the
      human review was performed on.
    - ``rampnet_1pass`` -- ``analysis_out/op_cache/<split>.json``, the **same
      released checkpoint** scored single-pass for the #54 operating-point study.
      Pairing it with ``rampnet`` gives the one RampNet-against-RampNet comparison
      the repo can make from committed data, which is the closest available analogue
      to two checkpoints of one run.
    - anything else -- the published challenger export.
    """
    if model == "rampnet":
        return {pid: rec["detections"] for pid, rec in records.items()}
    if model.startswith("run_a_epoch_"):
        # NOT benchmark/model_detections/: that directory is the challenger roster's,
        # and rampnet/roster.py asserts every file in it belongs to a registered leg
        # (#122). Run A's epochs are internal checkpoints of one experiment, not
        # entries in the RampNet-vs-VLM comparison, so they live beside the rest of
        # the #84 data instead of being registered as challengers they are not.
        path = os.path.join(repo, "docs", "data", "run_a_84_detections",
                            f"{model}__{split}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["detections"]
    if model == "rampnet_1pass":
        path = os.path.join(repo, "analysis_out", "op_cache", f"{split}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return {p["pano"]: p["preds"] for p in json.load(fh)["panos"]}
    return load_detections(model, split,
                           os.path.join(repo, "benchmark", "model_detections"))


def match_detail(pred_points, gt, radius_sq, wrap_x=WRAP_X):
    """Per-prediction (confidence, is_tp, gt_index) for the *scored* predictions.

    Reproduces ``score_pano``'s ordering, seam wrapping and ignore-point fallback,
    but keeps the ground-truth index each true positive claimed, which ``score_pano``
    discards and which the McNemar decomposition needs (it asks *which* ramps a model
    found, not how many). ``score_pano`` remains the authority on the counts -- the
    caller asserts the two agree, so this cannot silently drift from it.

    Every geometry decision here is ``score_pano``'s, including ``WRAP_X``: matching
    and the ignore fallback both go through ``rampnet.geometry.dist_sq`` rather than
    an inline distance, which is the mistake #132 4 found in ``score_pano`` itself.

    Predictions are ordered by descending confidence, and greedy matching in that
    order means the assignment of the top-k predictions is identical for every k.
    So thresholding *after* matching is exact, and one match per (model, split)
    serves every threshold and every bootstrap replicate.
    """
    confs = [prediction_confidence(p) for p in pred_points]
    if any(c is not None for c in confs):
        order = sorted(range(len(pred_points)),
                       key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                       reverse=True)
        preds = [pred_points[i] for i in order]
    else:
        preds = list(pred_points)

    xy = [(float(p["x_normalized"]), float(p["y_normalized"])) if isinstance(p, dict)
          else (float(p[0]), float(p[1])) for p in preds]
    assignments = greedy_match(xy, gt.gt_points, radius_sq,
                              PANO_SCALE_X, PANO_SCALE_Y, wrap_x)

    scored = []
    for (px_n, py_n), p, (gt_index, _) in zip(xy, preds, assignments):
        if gt_index >= 0:
            scored.append((prediction_confidence(p), True, gt_index))
            continue
        in_ignore = any(
            dist_sq(px_n, py_n, ix, iy, PANO_SCALE_X, PANO_SCALE_Y, wrap_x) < radius_sq
            for ix, iy in gt.ignore_points)
        if not in_ignore:
            scored.append((prediction_confidence(p), False, -1))
    return scored


class Scored:
    """One (model, split) reduced to flat arrays the bootstrap can resample.

    Everything downstream is a weighted sum over panoramas, so the per-pano
    bookkeeping is: which pano each scored prediction belongs to, whether it was a
    true positive, its confidence, and each pano's ground-truth count (zero unless
    the pano's missed-ramp check is confirmed, mirroring ``aggregate``'s recall gate).
    """

    def __init__(self, split, model, pids, conf, is_tp, det_pano, n_gt, recall_ok,
                 hit_gt, gt_x):
        self.split, self.model = split, model
        self.pids = pids                    # (n_panos,) pano ids, the bootstrap unit
        self.conf = conf                    # (n_dets,) descending within pano
        self.is_tp = is_tp                  # (n_dets,) bool
        self.det_pano = det_pano            # (n_dets,) index into pids
        self.n_gt = n_gt                    # (n_panos,) GT instances, recall-gated
        self.recall_ok = recall_ok          # (n_panos,) bool
        self.hit_gt = hit_gt                # (n_instances,) confidence that found it,
                                            #   -inf if never found
        self.gt_x = gt_x                    # (n_instances,) normalized x, for the seam check


def score_model(repo, split, model, records, gts, radius_sq, wrap_x=WRAP_X):
    """Reduce one (model, split) to a :class:`Scored`, or None if it was never run.

    ``wrap_x`` is a parameter rather than a constant for one reason: the committed
    #84 curve in ``docs/data/run_a_84_manual_gold/summary.csv`` was measured before
    #140 sealed the seam, so reproducing *that file* needs the matcher of the day.
    Analysis always runs at ``WRAP_X``; only the regression test passes False.
    """
    dets = detections_for(repo, split, model, records)
    if dets is None:
        return None

    pids = sorted(gts)
    pid_index = {pid: i for i, pid in enumerate(pids)}
    conf, is_tp, det_pano = [], [], []
    n_gt = np.zeros(len(pids), dtype=np.float64)
    recall_ok = np.zeros(len(pids), dtype=bool)
    hit_gt, gt_x = [], []

    for pid in pids:
        i = pid_index[pid]
        gt = gts[pid]
        points = dets.get(pid, [])
        detail = match_detail(points, gt, radius_sq, wrap_x)
        # score_pano stays the authority on the counts; this asserts match_detail
        # did not drift from it. A mismatch means the two matchers disagree, which
        # would invalidate every number below -- so it fails rather than warns.
        ref = score_pano(points, gt, radius_sq=radius_sq, wrap_x=wrap_x)
        assert sum(1 for _, tp, _ in detail if tp) == ref.tp, (split, model, pid, "tp")
        assert sum(1 for _, tp, _ in detail if not tp) == ref.fp, (split, model, pid, "fp")

        recall_ok[i] = gt.fn_confirmed
        if gt.fn_confirmed:
            n_gt[i] = len(gt.gt_points)

        found = {}
        for c, tp, k in detail:
            conf.append(-math.inf if c is None else float(c))
            is_tp.append(tp)
            det_pano.append(i)
            if tp:
                # Greedy matching is 1:1, so a GT index is claimed at most once; the
                # confidence that claimed it is the threshold above which it counts
                # as found.
                found[k] = -math.inf if c is None else float(c)
        if gt.fn_confirmed:
            for k, (gx, _) in enumerate(gt.gt_points):
                hit_gt.append(found.get(k, -math.inf))
                gt_x.append(gx)

    return Scored(split, model, pids,
                  np.asarray(conf, dtype=np.float64),
                  np.asarray(is_tp, dtype=bool),
                  np.asarray(det_pano, dtype=np.int64),
                  n_gt, recall_ok,
                  np.asarray(hit_gt, dtype=np.float64),
                  np.asarray(gt_x, dtype=np.float64))


def stack(scoreds):
    """Concatenate several splits into one pooled :class:`Scored` (micro-averaged).

    Pano indices are re-based so the bootstrap still resamples panoramas, and the
    caller resamples *within* each split (stratified) so the pooled statistic keeps
    each split's fixed size rather than letting a large split randomly dominate.
    """
    pids, conf, is_tp, det_pano, n_gt, recall_ok, hit_gt, gt_x = (
        [] for _ in range(8))
    offset = 0
    for s in scoreds:
        pids.extend(f"{s.split}/{p}" for p in s.pids)
        conf.append(s.conf)
        is_tp.append(s.is_tp)
        det_pano.append(s.det_pano + offset)
        n_gt.append(s.n_gt)
        recall_ok.append(s.recall_ok)
        hit_gt.append(s.hit_gt)
        gt_x.append(s.gt_x)
        offset += len(s.pids)
    return Scored("+".join(s.split for s in scoreds), scoreds[0].model, pids,
                  np.concatenate(conf), np.concatenate(is_tp), np.concatenate(det_pano),
                  np.concatenate(n_gt), np.concatenate(recall_ok),
                  np.concatenate(hit_gt), np.concatenate(gt_x))


def _f1(p, r):
    denom = p + r
    return np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)


def metrics(s, weights, threshold):
    """(precision, recall, f1, max_f1) at pano weights ``weights`` -- (B, n_panos).

    ``weights`` is a bootstrap replicate's multiplicity per panorama; the observed
    sample is the all-ones row. Precision counts every pano and recall only the
    recall-confirmed ones, exactly as ``rampnet.detection_eval.aggregate`` does.
    """
    w_det = weights[:, s.det_pano]                        # (B, n_dets)
    keep = s.conf >= threshold
    tp_all = (w_det * (keep & s.is_tp)).sum(axis=1)
    n_pred = (w_det * keep).sum(axis=1)
    rec_det = keep & s.is_tp & s.recall_ok[s.det_pano]
    tp_rec = (w_det * rec_det).sum(axis=1)
    n_gt = weights @ s.n_gt

    precision = np.divide(tp_all, n_pred, out=np.zeros_like(tp_all), where=n_pred > 0)
    recall = np.divide(tp_rec, n_gt, out=np.zeros_like(tp_rec), where=n_gt > 0)
    return precision, recall, _f1(precision, recall), max_f1(s, weights, n_gt)


def max_f1(s, weights, n_gt):
    """The calibration-free peak of the F1-vs-confidence curve, per replicate.

    Swept by walking the detections in descending confidence and reading F1 at every
    prefix -- which is the same curve ``--threshold 0.0`` writes, evaluated under the
    replicate's panorama weights. Ties in confidence are handled by only reading F1
    at the last detection of each tied block, so a threshold that cannot be set
    between two equal scores is never reported as achievable.
    """
    order = np.argsort(-s.conf, kind="stable")
    w_det = weights[:, s.det_pano[order]]                 # (B, n_dets)
    tp = s.is_tp[order]
    rec = tp & s.recall_ok[s.det_pano[order]]
    cum_tp = np.cumsum(w_det * tp, axis=1)
    cum_pred = np.cumsum(w_det, axis=1)
    cum_rec = np.cumsum(w_det * rec, axis=1)

    c = s.conf[order]
    block_end = np.ones(len(c), dtype=bool)
    block_end[:-1] = c[:-1] > c[1:]
    if not block_end.any():
        return np.zeros(weights.shape[0])

    p = np.divide(cum_tp, cum_pred, out=np.zeros_like(cum_tp), where=cum_pred > 0)
    ngt = n_gt[:, None]
    r = np.divide(cum_rec, np.broadcast_to(ngt, cum_rec.shape),
                  out=np.zeros_like(cum_rec), where=ngt > 0)
    return _f1(p, r)[:, block_end].max(axis=1)


def bootstrap_weights(rng, split_sizes, n_reps):
    """Stratified cluster-bootstrap multiplicities -- (n_reps, sum(split_sizes)).

    Panoramas are resampled with replacement *within* each split, so a pooled
    statistic keeps every split at its observed size. Resampling panoramas rather
    than instances is the point: ramps cluster within a panorama (a pano with one
    hard ramp often has several), so an instance-level resample understates the
    spread by the design effect this script reports.
    """
    blocks = []
    for n in split_sizes:
        blocks.append(rng.multinomial(n, np.full(n, 1.0 / n), size=n_reps).astype(np.float64))
    return np.hstack(blocks) if len(blocks) > 1 else blocks[0]


def observed_and_se(s, split_sizes, threshold, rng, n_reps, chunk=500, paired=None):
    """Observed metrics plus bootstrap s.e.; with ``paired``, of the *difference*.

    ``paired`` is a second :class:`Scored` over the same panoramas in the same order.
    The identical weight matrix is applied to both, so the panorama-difficulty
    component that dominates the unpaired spread cancels -- which is the entire
    reason a paired comparison resolves smaller effects than an unpaired one.

    That shared order is a precondition, not a convention: column j of the weight
    matrix means a different panorama in each :class:`Scored` if it is violated, and
    the result is a plausible-looking wrong standard error rather than an error. So
    it is asserted here, the way ``mcnemar`` asserts its own.
    """
    if paired is not None:
        assert list(s.pids) == list(paired.pids),             "paired comparison needs the same panoramas in the same order"
    ones = np.ones((1, len(s.pids)))
    obs = np.array(metrics(s, ones, threshold)).ravel()
    if paired is not None:
        obs = obs - np.array(metrics(paired, ones, threshold)).ravel()

    draws = []
    done = 0
    while done < n_reps:
        b = min(chunk, n_reps - done)
        w = bootstrap_weights(rng, split_sizes, b)
        m = np.array(metrics(s, w, threshold))
        if paired is not None:
            m = m - np.array(metrics(paired, w, threshold))
        draws.append(m)
        done += b
    d = np.hstack(draws)                                  # (4, n_reps)
    names = ("precision", "recall", "f1", "max_f1")
    return {n: {"observed": float(o), "se": float(np.std(v, ddof=1)),
                "ci_lo": float(np.percentile(v, 2.5)),
                "ci_hi": float(np.percentile(v, 97.5))}
            for n, o, v in zip(names, obs, d)}


def mcnemar(a, b, threshold_a, threshold_b):
    """(b_count, c_count, n_instances) -- the discordance between two detectors.

    ``b_count`` is ramps A misses and B finds, ``c_count`` the reverse. The paired
    difference in recall is (b - c)/n, and its variance depends on b + c alone -- not
    on n x p x (1-p), which is what the unpaired bar is built from. So the
    discordance rate is the parameter that decides whether a paired comparison can
    read a small effect, and it is measurable without running anything.
    """
    assert len(a.hit_gt) == len(b.hit_gt), "paired split must enumerate the same GT"
    hit_a = a.hit_gt >= threshold_a
    hit_b = b.hit_gt >= threshold_b
    return int((~hit_a & hit_b).sum()), int((hit_a & ~hit_b).sum()), len(hit_a)


#: A ground-truth ramp this close to x=0 or x=1 sits on the panorama seam.
SEAM_MARGIN = 0.03


def seam_enrichment(a, b, threshold_a, threshold_b, margin=SEAM_MARGIN):
    """Is a pair's disagreement concentrated at the panorama seam?

    A guard against reading an artifact as a model difference. Issue #132 established
    that the committed ``analysis_out/op_cache`` detections were extracted with
    ``skimage.feature.peak_local_max``'s default ``exclude_border``, so they carry
    **no** detections at the seam, while ``benchmark/*/records.jsonl`` does. Any
    pair that straddles those two sources therefore disagrees at the seam for
    reasons that have nothing to do with the detectors.

    Returns (n_discordant, n_at_seam, baseline_rate). A seam share far above the
    baseline rate means part of the measured discordance is that artifact -- which
    inflates the pair's s.e., so a bound built from it stays conservative rather
    than becoming wrong.
    """
    hit_a = a.hit_gt >= threshold_a
    hit_b = b.hit_gt >= threshold_b
    disc = hit_a != hit_b
    at_seam = (a.gt_x < margin) | (a.gt_x > 1 - margin)
    baseline = float(at_seam.mean()) if len(at_seam) else float("nan")
    return int(disc.sum()), int((disc & at_seam).sum()), baseline


def naive_binomial_se(recall, n_gt):
    """The instance-level binomial s.e. the 0.01 tie bar was derived from."""
    return math.sqrt(recall * (1 - recall) / n_gt) if n_gt else float("nan")


def required_discordance(delta_recall, n_gt, deff, z=Z_CI):
    """How discordant two detectors must be for a recall gap to be UNREADABLE.

    The paired difference in recall is (b - c)/n and its variance is
    ``deff * (b + c) / n^2`` -- McNemar, inflated by the panorama clustering the
    bootstrap measures. Inverting for the b + c that would push a *given* observed
    gap down to non-significance turns "is this benchmark sharp enough" into a
    question with a checkable answer: compare the required discordance against the
    discordance real detector pairs actually exhibit.

    Returns (required_b_plus_c, required_rate). A required rate far above anything
    observed means the gap is resolvable for any plausible pair.
    """
    if not delta_recall or not n_gt:
        return float("nan"), float("nan")
    need = (abs(delta_recall) * n_gt / z) ** 2 / deff
    return need, need / n_gt


def read_run_a_summary(repo):
    """Run A's committed per-epoch `manual_gold` numbers, or None if absent."""
    path = os.path.join(repo, "docs", "data", "run_a_84_manual_gold", "summary.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        for line in fh:
            if line.strip():
                rows.append(dict(zip(header, line.rstrip("\n").split(","))))
    return rows


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--splits", default=None,
                    help="Comma-separated bundle names (default: every committed bundle).")
    ap.add_argument("--reference", default="rampnet",
                    help="Detector used for the unpaired noise floor.")
    ap.add_argument("--pairs",
                    default="rampnet_1pass:rampnet,y11l_pano:y11x_pano_h200,"
                            "y26_pano:y11x_pano_h200",
                    help="Comma-separated A:B pairs for the paired analysis. The "
                         "default spans the range that brackets an epoch-vs-epoch "
                         "comparison: one checkpoint under two inference protocols "
                         "(most similar) through two YOLO architectures (least).")
    ap.add_argument("--self-pair-deltas", default="0.02,0.05",
                    help="Confidence offsets used to build the maximally-correlated "
                         "lower-bound pair from the reference detector alone.")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--matrix-bootstrap", type=int, default=5000,
                    help="Bootstrap size for the 28-pair Run A epoch matrix. Smaller "
                         "than --bootstrap on purpose: 5,000 pins a standard error to "
                         "about 1% relative, which is far finer than any conclusion "
                         "drawn from it, and it keeps 28 paired resamples affordable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    repo = args.repo
    splits = ([s.strip() for s in args.splits.split(",") if s.strip()]
              if args.splits else discover_splits(repo))
    rsq = radius_sq_for()
    # ONE Generator for the whole run, threaded through every group and pair in the
    # order they are computed. That makes the committed artifact exactly reproducible
    # from the full invocation, and it makes a SUBSET only reproducible to bootstrap
    # noise: `--splits manual_gold` draws a different stream than the same split does
    # inside the ten-split run. Documented in the doc's Reproduce section rather than
    # re-seeded per group, because re-seeding would move every published s.e.
    rng = np.random.default_rng(args.seed)

    bundles, scored = {}, {}
    for split in splits:
        records, gts = load_split(repo, split)
        bundles[split] = (records, gts)

    def get(split, model):
        key = (split, model)
        if key not in scored:
            records, gts = bundles[split]
            scored[key] = score_model(repo, split, model, records, gts, rsq)
        return scored[key]

    out = {"splits": splits, "reference": args.reference, "bootstrap": args.bootstrap,
           "seed": args.seed, "inventory": {}, "unpaired": {}, "paired": {},
           "z_mde": Z_MDE, "z_ci": Z_CI}

    # ---- inventory -------------------------------------------------------------
    print("=" * 96)
    print("INVENTORY -- what the committed benchmark actually holds")
    print("=" * 96)
    print(f"{'split':>20} {'panos':>7} {'recall panos':>13} {'GT instances':>13} "
          f"{'ramps/pano':>11} {'GT source':>12}")
    for split in splits:
        records, gts = bundles[split]
        s = get(split, args.reference)
        n_panos = len(gts)
        n_rec = int(s.recall_ok.sum())
        n_inst = int(s.n_gt.sum())
        src = "manual" if (Path(repo) / "benchmark" / split / "gt_source.json").exists() \
            else "anchored"
        # THE TRUNCATION, RECORDED RATHER THAN INFERRED. The nine city bundles were
        # published with a 0.55 peak floor while manual_gold reaches 0.05, so on those
        # splits a "keep conf >= 0.30" filter is a no-op and the max-F1 sweep runs off
        # the end of a curve that was already cut. Without these two fields the only
        # evidence is indirect (max_f1 == f1 exactly, self_pair b = c = 0), which is
        # what let the effect sit unnoticed in this artifact for two review passes.
        floor = float(s.conf.min()) if len(s.conf) else float("nan")
        out["inventory"][split] = {
            "n_panos": n_panos, "n_recall_panos": n_rec,
            "n_gt_instances": n_inst, "gt_source": src,
            "reference_min_confidence": floor,
            # True when the protocol threshold sits below every committed detection,
            # i.e. the operating point named in the header was never actually applied.
            "protocol_threshold_binds": bool(
                len(s.conf) and floor < protocol_threshold(args.reference)),
        }
        print(f"{split:>20} {n_panos:>7} {n_rec:>13} {n_inst:>13} "
              f"{n_inst / max(n_rec, 1):>11.2f} {src:>12}")

    cities = [s for s in splits if out["inventory"][s]["gt_source"] == "anchored"]
    groups = {s: [s] for s in splits}
    if cities:
        groups["POOLED cities"] = cities
    groups["POOLED all"] = list(splits)
    for name, members in groups.items():
        if len(members) > 1:
            n = sum(out["inventory"][m]["n_gt_instances"] for m in members)
            print(f"{name:>20} {sum(out['inventory'][m]['n_panos'] for m in members):>7} "
                  f"{sum(out['inventory'][m]['n_recall_panos'] for m in members):>13} "
                  f"{n:>13}")

    # ---- unpaired noise floor --------------------------------------------------
    thr = protocol_threshold(args.reference)
    print()
    print("=" * 96)
    print(f"UNPAIRED NOISE FLOOR -- {args.reference} at conf {thr:.2f}, "
          f"cluster bootstrap over panoramas (B={args.bootstrap})")
    print("=" * 96)
    truncated = [s_ for s_ in splits
                 if not out["inventory"][s_]["protocol_threshold_binds"]]
    if truncated:
        print(f"  !! {len(truncated)} of {len(splits)} split(s) hold NO detection below "
              f"the {thr:.2f} threshold, so it is a no-op there and the effective")
        print(f"     operating point is that split's own floor: "
              f"{', '.join(sorted(truncated))}.")
        print("     max-F1 on those rows is also the peak of a TRUNCATED curve, not of "
              "the full one. Read every row below with that caveat.")
    for name, members in groups.items():
        parts = [get(m, args.reference) for m in members]
        if any(p is None for p in parts):
            continue
        s = parts[0] if len(parts) == 1 else stack(parts)
        sizes = [len(p.pids) for p in parts]
        res = observed_and_se(s, sizes, thr, rng, args.bootstrap)
        n_inst = int(s.n_gt.sum())
        naive = naive_binomial_se(res["recall"]["observed"], n_inst)
        deff = (res["recall"]["se"] / naive) ** 2 if naive else float("nan")
        cut = sorted(m for m in members if m in truncated)
        out["unpaired"][name] = {"threshold": thr, "n_gt_instances": n_inst,
                                 "naive_binomial_se_recall": naive,
                                 "design_effect_recall": deff,
                                 # Which members of this row were measured above the
                                 # named threshold rather than at it.
                                 "truncated_members": cut, **res}
        mark = " *" if cut else ""
        print(f"{name:>20} {fmt(res['f1']['observed']):>8} {fmt(res['f1']['se']):>8} "
              f"{fmt(res['recall']['se']):>8} {fmt(naive):>12} {deff:>11.2f} "
              f"{fmt(Z_MDE * res['f1']['se']):>9}{mark}")
    if truncated:
        print("  * row includes at least one truncated split; see the warning above.")

    # ---- paired noise floor ----------------------------------------------------
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p.strip()]
    print()
    print("=" * 96)
    print("PAIRED NOISE FLOOR -- the same panoramas, so pano difficulty cancels")
    print("=" * 96)
    for a_model, b_model in pairs:
        rows = []
        for name, members in groups.items():
            pa = [get(m, a_model) for m in members]
            pb = [get(m, b_model) for m in members]
            if any(x is None for x in pa + pb):
                continue
            sa = pa[0] if len(pa) == 1 else stack(pa)
            sb = pb[0] if len(pb) == 1 else stack(pb)
            sizes = [len(p.pids) for p in pa]
            ta, tb = protocol_threshold(a_model), protocol_threshold(b_model)
            if ta != tb:
                # A paired difference needs one threshold, or the "difference" also
                # carries the operating-point gap. Both YOLO arms share 0.25.
                raise SystemExit(
                    f"{a_model} (conf {ta}) and {b_model} (conf {tb}) have different "
                    "protocol operating points, so their difference would mix a "
                    "capability gap with a calibration gap. Pair like with like.")
            # NOTE -- res_a and res_b run a full 20,000-replicate bootstrap each to
            # produce ONE scalar between them (unpaired_se_f1, below). About half the
            # work in this block is therefore redundant, and y11x_pano_h200's unpaired
            # result is recomputed for each of the two pairs it appears in. That is
            # DELIBERATELY not optimised: one Generator is threaded through this loop
            # in order, so removing a draw shifts the stream for everything after it
            # and moves every standard error the doc quotes. The cost is ~10 minutes
            # of CPU on a run that happens when the analysis changes; the benefit of
            # touching it is zero. If it is ever reworked, regenerate the artifact and
            # re-read every number out of it -- do not assume they held.
            res_d = observed_and_se(sb, sizes, tb, rng, args.bootstrap, paired=sa)
            res_a = observed_and_se(sa, sizes, ta, rng, args.bootstrap)
            res_b = observed_and_se(sb, sizes, tb, rng, args.bootstrap)
            bc, cc, n_inst = mcnemar(sa, sb, ta, tb)
            unpaired_se = math.hypot(res_a["f1"]["se"], res_b["f1"]["se"])
            # How much the panorama clustering inflates the textbook McNemar s.e.
            # sqrt(b+c)/n. Measured here so the inversion below can use a real
            # factor instead of assuming instances are independent.
            mcnemar_se = math.sqrt(bc + cc) / n_inst if n_inst else float("nan")
            paired_deff = ((res_d["recall"]["se"] / mcnemar_se) ** 2
                           if mcnemar_se else float("nan"))
            n_disc, n_seam, seam_base = seam_enrichment(sa, sb, ta, tb)
            rows.append((name, res_d, res_a, res_b, bc, cc, n_inst, unpaired_se))
            out["paired"].setdefault(f"{a_model}:{b_model}", {})[name] = {
                "threshold": tb, "delta": res_d, "a": res_a, "b": res_b,
                "mcnemar_b": bc, "mcnemar_c": cc, "n_gt_instances": n_inst,
                "unpaired_se_f1": unpaired_se, "mcnemar_se_recall": mcnemar_se,
                "paired_design_effect_recall": paired_deff,
                "n_discordant": n_disc, "n_discordant_at_seam": n_seam,
                "seam_baseline_rate": seam_base,
            }
        if not rows:
            print(f"  ({a_model} vs {b_model}: not both published on any split)")
            continue
        print(f"\n  {b_model}  minus  {a_model}   (conf {protocol_threshold(b_model):.2f})")
        print(f"  {'split':>18} {'dF1':>8} {'se(dF1)':>9} {'unpaired':>9} {'gain':>6} "
              f"{'MDE 80%':>9} | {'dmaxF1':>8} {'se':>8} | {'b':>5} {'c':>5} {'discord%':>9}")
        for name, d, a, b, bc, cc, n_inst, unp in rows:
            gain = unp / d["f1"]["se"] if d["f1"]["se"] else float("nan")
            print(f"  {name:>18} {fmt(d['f1']['observed']):>8} {fmt(d['f1']['se']):>9} "
                  f"{fmt(unp):>9} {gain:>5.1f}x {fmt(Z_MDE * d['f1']['se']):>9} | "
                  f"{fmt(d['max_f1']['observed']):>8} {fmt(d['max_f1']['se']):>8} | "
                  f"{bc:>5} {cc:>5} {100 * (bc + cc) / max(n_inst, 1):>8.1f}%")
        entry = out["paired"][f"{a_model}:{b_model}"].get("manual_gold")
        if entry and entry["n_discordant"]:
            share = entry["n_discordant_at_seam"] / entry["n_discordant"]
            flag = "  <-- #132 seam artifact, not a model difference" \
                if share > 3 * entry["seam_baseline_rate"] else ""
            print(f"  seam check (manual_gold): {entry['n_discordant_at_seam']} of "
                  f"{entry['n_discordant']} disagreements ({100 * share:.0f}%) sit within "
                  f"{SEAM_MARGIN} of the seam, against a {100 * entry['seam_baseline_rate']:.1f}% "
                  f"baseline.{flag}")

    # ---- maximally-correlated lower bound --------------------------------------
    print()
    print("=" * 96)
    print(f"LOWER BOUND -- {args.reference} against itself at a shifted operating point")
    print("=" * 96)
    print("  Two checkpoints of one run are less correlated than one checkpoint read at")
    print("  two thresholds, so this brackets the paired s.e. from below.")
    print("  F1 ONLY. max-F1 re-picks its own threshold, so it is not a function of the")
    print("  shift being varied and its delta is identically zero by construction --")
    print("  which is a degenerate statistic, not a measurement of zero uncertainty.")
    print("  It is therefore recorded as null here rather than as 0.0 +/- 0.0.")
    print(f"  {'split':>18} {'shift':>7} {'dF1':>9} {'se(dF1)':>9} {'MDE 80%':>9} "
          f"{'b':>5} {'c':>5} {'discord%':>9}")
    out["self_pair"] = {}
    for delta in [float(x) for x in args.self_pair_deltas.split(",") if x.strip()]:
        for name, members in groups.items():
            parts = [get(m, args.reference) for m in members]
            if any(p is None for p in parts):
                continue
            s = parts[0] if len(parts) == 1 else stack(parts)
            sizes = [len(p.pids) for p in parts]
            base, shifted = thr, thr + delta
            ones = np.ones((1, len(s.pids)))
            draws = []
            done = 0
            obs = (np.array(metrics(s, ones, shifted)).ravel()
                   - np.array(metrics(s, ones, base)).ravel())
            while done < args.bootstrap:
                nb = min(500, args.bootstrap - done)
                w = bootstrap_weights(rng, sizes, nb)
                draws.append(np.array(metrics(s, w, shifted)) - np.array(metrics(s, w, base)))
                done += nb
            d = np.hstack(draws)
            bc, cc, n_inst = mcnemar(s, s, base, shifted)
            key = f"{name}@+{delta}"
            out["self_pair"][key] = {
                "delta_conf": delta, "n_gt_instances": n_inst,
                "mcnemar_b": bc, "mcnemar_c": cc,
                "f1": {"observed": float(obs[2]), "se": float(np.std(d[2], ddof=1))},
                # max_f1 sweeps its own threshold, so shifting the read-out point
                # cannot move it: obs[3] and its bootstrap spread are exactly 0 for
                # every row. Emitting {"observed": 0.0, "se": 0.0} reads as a
                # measured zero with zero uncertainty; null plus the note below says
                # what it actually is.
                "max_f1": None,
                "max_f1_note": ("not a function of the operating-point shift -- "
                                "max-F1 re-picks its own threshold, so this block "
                                "bounds F1 only"),
            }
            print(f"  {name:>18} {delta:>+7.2f} {fmt(obs[2]):>9} "
                  f"{fmt(float(np.std(d[2], ddof=1))):>9} "
                  f"{fmt(Z_MDE * float(np.std(d[2], ddof=1))):>9} "
                  f"{bc:>5} {cc:>5} {100 * (bc + cc) / max(n_inst, 1):>8.1f}%")

    # ---- MEASURED epoch-to-epoch discordance: the bracket, closed ---------------
    # Everything above brackets the paired s.e. for two Stage 2 checkpoints between
    # two proxies, because Run A's committed artifacts are aggregate curves. Once
    # dump_peaks_from_cache.py has recovered the per-panorama detections from the
    # surviving heatmap cache, the real thing is measurable and the proxies are only
    # a sanity check on it.
    epochs = [n for n in range(1, 9)
              if get("manual_gold", RUN_A_EPOCH.format(n)) is not None]
    measured = {}
    if len(epochs) >= 2:
        s0 = get("manual_gold", RUN_A_EPOCH.format(epochs[0]))
        sizes = [len(s0.pids)]
        thr = protocol_threshold(RUN_A_EPOCH.format(epochs[0]))
        n_inst = int(s0.n_gt.sum())
        print()
        print("=" * 96)
        print(f"MEASURED epoch-to-epoch discordance, Run A on manual_gold "
              f"(B={args.matrix_bootstrap})")
        print("=" * 96)
        print("  The real quantity the bracket above was standing in for. Every pair of")
        print("  Run A checkpoints, scored on the same 1,000 panoramas.")
        print(f"\n  {'pair':>8} {'gap':>4} {'d maxF1':>9} {'se':>8} {'MDE 80%':>9} "
              f"{'d F1@.30':>9} {'se':>8} | {'b':>5} {'c':>5} {'discord%':>9}")
        for i, ea in enumerate(epochs):
            for eb in epochs[i + 1:]:
                sa = get("manual_gold", RUN_A_EPOCH.format(ea))
                sb = get("manual_gold", RUN_A_EPOCH.format(eb))
                d = observed_and_se(sb, sizes, thr, rng, args.matrix_bootstrap, paired=sa)
                bc, cc, _ = mcnemar(sa, sb, thr, thr)
                n_disc, n_seam, seam_base = seam_enrichment(sa, sb, thr, thr)
                measured[f"{ea}v{eb}"] = {
                    "epoch_a": ea, "epoch_b": eb, "gap": eb - ea, "delta": d,
                    "mcnemar_b": bc, "mcnemar_c": cc, "n_gt_instances": n_inst,
                    "discordance_rate": (bc + cc) / n_inst,
                    "n_discordant": n_disc, "n_discordant_at_seam": n_seam,
                    "seam_baseline_rate": seam_base,
                }
                print(f"  {f'{ea} vs {eb}':>8} {eb - ea:>4} "
                      f"{fmt(d['max_f1']['observed']):>9} {fmt(d['max_f1']['se']):>8} "
                      f"{fmt(Z_MDE * d['max_f1']['se']):>9} "
                      f"{fmt(d['f1']['observed']):>9} {fmt(d['f1']['se']):>8} | "
                      f"{bc:>5} {cc:>5} {100 * (bc + cc) / max(n_inst, 1):>8.1f}%")

        ses = sorted(v["delta"]["max_f1"]["se"] for v in measured.values())
        far = sorted(v["delta"]["max_f1"]["se"] for v in measured.values()
                     if v["gap"] >= 3)
        rates = sorted(v["discordance_rate"] for v in measured.values())
        out["measured_epoch_pairs"] = {
            "threshold": thr, "n_gt_instances": n_inst, "bootstrap": args.matrix_bootstrap,
            "se_max_f1_min": ses[0], "se_max_f1_median": ses[len(ses) // 2],
            "se_max_f1_max": ses[-1],
            "se_max_f1_median_gap_ge_3": (far[len(far) // 2] if far else None),
            "discordance_min": rates[0], "discordance_max": rates[-1],
            "pairs": measured,
        }
        head = far or ses
        headline = head[len(head) // 2]
        print(f"\n  se(d max-F1) across {len(ses)} pairs: {ses[0]:.4f} to {ses[-1]:.4f}, "
              f"median {ses[len(ses) // 2]:.4f}")
        print(f"  Discordance: {100 * rates[0]:.1f}% to {100 * rates[-1]:.1f}%.")
        print(f"  **Checkpoints >= 3 epochs apart -- the closest analogue to a Run B "
              f"checkpoint vs a Run A one --")
        print(f"    have median se(d max-F1) {headline:.4f}, i.e. MDE "
              f"{Z_MDE * headline:.4f} at 80% power.**")

    # ---- what Run A's own plateau looks like under a paired read ---------------
    summary = read_run_a_summary(repo)
    if summary and "manual_gold" in out["unpaired"]:
        n_inst = out["unpaired"]["manual_gold"]["n_gt_instances"]
        # Paired design effects measured above, across every real pair on
        # manual_gold. Using the largest is the conservative choice: it makes the
        # required discordance smallest, i.e. hardest to call a gap resolvable.
        deffs = [v["manual_gold"]["paired_design_effect_recall"]
                 for v in out["paired"].values()
                 if "manual_gold" in v and not math.isnan(
                     v["manual_gold"]["paired_design_effect_recall"])]
        deff = max(deffs) if deffs else 1.0
        observed = [v["manual_gold"]["mcnemar_b"] + v["manual_gold"]["mcnemar_c"]
                    for v in out["paired"].values() if "manual_gold" in v]
        obs_rates = sorted(100 * o / n_inst for o in observed)
        print()
        print("=" * 96)
        print("RUN A's PLATEAU, RE-READ AS A PAIRED COMPARISON (#84 -> #135)")
        print("=" * 96)
        print(f"  Paired design effect used: {deff:.2f} (largest measured on manual_gold).")
        print(f"  Discordance actually observed between real detector pairs here: "
              f"{obs_rates[0]:.1f}%-{obs_rates[-1]:.1f}%.")
        print("  'Required discordance' is how far apart two checkpoints would have to")
        print("  be for the observed recall gap to be UNREADABLE at 95%. Above the")
        print("  observed range means the gap is resolvable for any plausible pair.")
        print(f"\n  {'epochs':>10} {'recall A':>9} {'recall B':>9} {'d recall':>9} "
              f"{'need b+c':>9} {'need rate':>10} {'verdict':>14}")
        by_ep = {int(r["epoch"]): r for r in summary}
        # Chosen to show the SHAPE: epoch 1 against the plateau, the plateau against
        # itself, and the plateau against the tail.
        interesting = [(1, 2), (1, 3), (2, 6), (3, 6), (3, 7), (3, 8), (5, 8),
                       (6, 7), (1, 8)]
        out["run_a_paired"] = {"design_effect": deff, "n_gt_instances": n_inst,
                               "observed_discordance_pct": obs_rates, "pairs": {}}
        for ea, eb in interesting:
            if ea not in by_ep or eb not in by_ep:
                continue
            ra = float(by_ep[ea]["recall_at_protocol"])
            rb = float(by_ep[eb]["recall_at_protocol"])
            need, rate = required_discordance(rb - ra, n_inst, deff)
            verdict = ("resolvable" if rate > obs_rates[-1] / 100
                       else "not resolvable" if rate < obs_rates[0] / 100
                       else "borderline")
            out["run_a_paired"]["pairs"][f"{ea}v{eb}"] = {
                "recall_a": ra, "recall_b": rb, "delta_recall": rb - ra,
                "required_b_plus_c": need, "required_rate": rate, "verdict": verdict}
            # A required rate above 100% cannot be reached by any pair -- there are
            # only n instances to disagree about -- so print it as the impossibility
            # it is rather than as a number someone might read as attainable.
            shown = ">100%" if rate > 1 else f"{100 * rate:.1f}%"
            print(f"  {f'{ea} vs {eb}':>10} {ra:>9.4f} {rb:>9.4f} {rb - ra:>+9.4f} "
                  f"{need:>9.0f} {shown:>10} {verdict:>14}")

        # The gate column. max-F1 has no clean McNemar form (it re-picks its own
        # threshold), so it is read against the measured paired s.e. directly,
        # bracketed rather than point-estimated: the low end is the most-correlated
        # real pair on manual_gold, the high end the least-correlated.
        ses = sorted(v["manual_gold"]["delta"]["max_f1"]["se"]
                     for v in out["paired"].values() if "manual_gold" in v)
        se_lo, se_hi = ses[0], ses[-1]
        source = "proxy bracket (Run A epoch dumps absent)"
        if measured:
            # The real epoch-to-epoch s.e., now that it exists: the smallest and
            # largest MEASURED values replace a bracket built from stand-ins. Every
            # verdict below therefore rests on the comparison actually being made
            # rather than on two pairs chosen to straddle it.
            m = sorted(v["delta"]["max_f1"]["se"] for v in measured.values())
            se_lo, se_hi = m[0], m[-1]
            source = f"MEASURED across {len(m)} Run A epoch pairs"
        print(f"\n  Gate column (max-F1). Paired s.e. -- {source}:")
        print(f"  {se_lo:.4f} to {se_hi:.4f}; MDE {Z_MDE * se_lo:.4f} to "
              f"{Z_MDE * se_hi:.4f}.")
        out["run_a_paired"] = out.get("run_a_paired", {})
        out["run_a_paired"]["max_f1_se_source"] = source
        # With the epoch dumps present each pair has its OWN measured s.e., which is
        # strictly better than reading every pair against one global bracket: the
        # s.e. grows with epoch separation (discordance does), so a bracket is loose
        # at one end and tight at the other. Fall back to the bracket only for pairs
        # with no dump.
        print(f"\n  {'epochs':>10} {'max-F1 A':>9} {'max-F1 B':>9} {'delta':>9} "
              f"{'se':>8} {'z':>6} {'verdict':>16}")
        out["run_a_paired"]["max_f1_se_bracket"] = [se_lo, se_hi]
        out["run_a_paired"]["max_f1_pairs"] = {}
        out["run_a_paired"]["wrap_x"] = WRAP_X
        # The point estimates must come from the SAME scoring as the standard errors
        # standing beside them. Reading them off summary.csv instead mixes provenance:
        # that file was written before #140 sealed the seam, so its epoch-7 max-F1 is
        # the non-wrapping matcher's while every s.e. in the row is the wrapping one's.
        # The gap is only 0.000264, but a table that quietly averages two matchers is
        # the failure this repo keeps finding, not a rounding question. summary.csv
        # stays the provenance record and the regression test's target; it is not the
        # analysis's input when the per-panorama dumps are present.
        def epoch_max_f1(e):
            scored = get("manual_gold", RUN_A_EPOCH.format(e))
            if scored is None:
                return float(by_ep[e]["max_f1"]), "summary.csv"
            weights = np.ones((1, len(scored.pids)))
            thr_e = protocol_threshold(RUN_A_EPOCH.format(e))
            value = float(np.array(metrics(scored, weights, thr_e)).ravel()[3])
            return value, "epoch dump"

        sources = set()
        for ea, eb in interesting:
            if ea not in by_ep or eb not in by_ep:
                continue
            (ma, sa_src), (mb, sb_src) = epoch_max_f1(ea), epoch_max_f1(eb)
            sources.update((sa_src, sb_src))
            pair = measured.get(f"{ea}v{eb}")
            if pair is not None:
                se = pair["delta"]["max_f1"]["se"]
                z = abs(mb - ma) / se if se else 0.0
                verdict = "resolvable" if z >= Z_CI else "not resolvable"
                note = ""
            else:
                se = None
                z_lo, z_hi = abs(mb - ma) / se_hi, abs(mb - ma) / se_lo
                z = z_lo
                verdict = ("resolvable" if z_lo >= Z_CI
                           else "not resolvable" if z_hi < Z_CI else "borderline")
                note = " (bracketed)"
            out["run_a_paired"]["max_f1_pairs"][f"{ea}v{eb}"] = {
                "max_f1_a": ma, "max_f1_b": mb, "delta": mb - ma,
                "se": se, "z": z, "verdict": verdict, "measured": pair is not None,
                "max_f1_source": sa_src if sa_src == sb_src else f"{sa_src}/{sb_src}"}
            print(f"  {f'{ea} vs {eb}':>10} {ma:>9.4f} {mb:>9.4f} {mb - ma:>+9.4f} "
                  f"{fmt(se) if se else 'n/a':>8} {z:>6.1f} {verdict + note:>16}")

    if args.out_json:
        path = args.out_json if os.path.isabs(args.out_json) else os.path.join(repo, args.out_json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # newline="" so a Windows re-run writes the same bytes as a Linux one; a
        # committed JSON that flips to CRLF breaks byte-comparison silently.
        with open(path, "w", encoding="utf-8", newline="") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
