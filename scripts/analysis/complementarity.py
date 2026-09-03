"""Do RampNet and a challenger miss *different* ramps? (issue #35 gate)

For each GT ramp on a recall-eligible pano, record whether RampNet found it, the
challenger found it, both, or neither -> oracle-union recall, and the set that
matters: **RampNet-miss n challenger-hit**.

Read-only: RampNet's side comes from the bundle's committed detections and the
challenger's from ``.model_cache``, so this never runs a model or spends anything.

**Which RampNet, though?** The committed bundle detections are the *shipped* operating
point — on richmond every one scores >= 0.5519 — while this document's own
recommendation since #54/#55 (PR #79) is **0.30**. Those are different models for this
purpose: at 0.30 RampNet finds 257 of richmond's 310 ramps instead of 238, so 19 of the
"misses" a challenger gets credit for recovering are ramps RampNet already has and the
shipped threshold is discarding. ``--rampnet-op-threshold`` re-sources RampNet's side
from the committed ``analysis_out/op_cache/<split>.json`` floor peaks at a given
threshold, which is how you ask "what is the complementary gain at the operating point
we would actually deploy?". Default is the bundle records, so the published roster
numbers and the #35 gate's committed results are unchanged.

**The oracle-union recall is a ceiling, not a proposal.** It assumes you could
keep every right call and discard every wrong one, which no combiner can do. The
FP arithmetic printed at the end is the counterweight, and for a low-precision
challenger it is usually decisive -- see the union precision line. Pair this with
``null_recall.py`` before believing any union number from a model that emits many
boxes per pano: at high density a share of "hits" are what the match radius hands
out for free, and that share inflates the complementary set too.

Usage -- the positional form is the one three call sites in
``docs/model_comparison.md`` use, and it still means what it did:

    python scripts/analysis/complementarity.py                       # gemini-3.6-flash, richmond
    python scripts/analysis/complementarity.py gemini-3.1-pro-preview paterson
    python scripts/analysis/complementarity.py vistas:curb-cut richmond \
        --vistas-input-size 1024 1024

The first positional is a **model spec** (``provider`` or ``provider:model_id``,
as ``compare.py --models`` takes them). A bare token that is not a known provider
is read as a Gemini model id, which is how the #35 gate was invoked before this
script grew past one provider.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

from rampnet import roster                                                # noqa: E402
from rampnet.detection_eval import (                                      # noqa: E402
    build_ground_truth, score_pano, radius_sq_for, PANO_SCALE_X, PANO_SCALE_Y,
    _xy, prediction_confidence)
from rampnet.metrics import greedy_match                                  # noqa: E402
from compare import load_bundle, DetectionCache, cache_key                # noqa: E402
from detectors import build_detector, parse_model_spec, PROVIDERS         # noqa: E402
from operating_point_curve import CACHE_DIR, read_cache                   # noqa: E402


def matched_gt(preds, gt_points, radius_sq):
    """Which GT ramps a model covers: greedy 1:1, exactly as ``score_pano`` matches.

    Ordering and geometry are ``score_pano``'s -- highest confidence first when the
    predictions carry one, else input order, then ``rampnet.metrics.greedy_match``
    with ``wrap_x=True``, which measures the x separation the shorter way round the
    panorama (``rampnet.geometry.fold``, #132). Calling the shared matcher rather than
    re-deriving the distance here is what keeps the four cells below and the FP counts
    from ``score_pano`` on the same matcher: this script prints both in one table, and
    ``score_pano`` wraps by default, so an inline non-wrapping distance here would put
    two matchers in one output.

    Measured effect of the wrap on this script's cells: zero flips on RampNet's side
    at >= 0.5519, >= 0.30 and >= 0.05, and zero on the published Vistas 384 arm
    (richmond); the paterson #35 gate reproduces unchanged. The parity 1024 arm could
    not be re-checked -- its detections are not published (see the note beside the
    parity table in ``docs/model_comparison.md``) -- so the bound there is the doc's
    own seam count, 1 site in 38.
    """
    confs = [prediction_confidence(p) for p in preds]
    order = (sorted(range(len(preds)),
                    key=lambda i: confs[i] if confs[i] is not None else float("-inf"),
                    reverse=True)
             if any(c is not None for c in confs) else range(len(preds)))
    assignments = greedy_match([_xy(preds[i]) for i in order], gt_points,
                               radius_sq, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
    return {gt_index for gt_index, _ in assignments if gt_index >= 0}


def complementary_null(rows, radius_sq):
    """How many of rampnet's misses would the challenger "recover" by coincidence?

    Same null as ``null_recall.py`` -- score pano A's ground truth against pano
    B's predictions, averaged over every non-identity cyclic shift -- but
    restricted to the GT rampnet MISSED, because that is the subset the
    complementary-gain headline is about. Both sides stay real model output on
    real imagery, so box count and spatial clustering are preserved; only the
    pairing is wrong, so every match is chance.

    Applying the whole-split null from ``null_recall.py`` to this subset instead
    would assume the coincidence rate is uniform across GT. It need not be:
    rampnet's misses are a biased sample (far-field, adjacent pairs), and those
    are exactly the places box density differs. Hence measuring it here.

    Returns (mean, max) as a fraction of the missed GT.
    """
    preds = [p for p, _ in rows]
    missed = [m for _, m in rows]
    n = len(rows)
    total = sum(len(m) for m in missed)
    if not total or n < 2:
        return 0.0, 0.0
    shifted = []
    for k in range(1, n):
        hit = sum(len(matched_gt(preds[(i + k) % n], missed[i], radius_sq))
                  for i in range(n))
        shifted.append(hit / total)
    return sum(shifted) / len(shifted), max(shifted)


def model_spec(token):
    """``provider``/``provider:model_id``, or a legacy bare Gemini model id."""
    provider, model_id = parse_model_spec(token)
    if provider in PROVIDERS:
        return provider, model_id
    # Legacy positional form: a bare model id meant gemini. Keep it working --
    # reading it as a provider would raise on strings that used to be valid.
    return "gemini", token


def compare_args(args):
    """A namespace matching ``compare.py``'s parser defaults, so the cache key this
    script reconstructs is the one ``compare.py`` wrote under.

    Provider defaults come from ``rampnet.roster.PROVIDER_DEFAULTS`` -- one
    definition, the same source ``fp_taxonomy``'s shim and ``null_recall`` read --
    because a wrong default here does not crash, it silently misses every cache
    entry and reports zero detections. The deviation-only knobs
    (``vistas_input_size``, ``vistas_revision``) are threaded through from the CLI:
    they enter the signature ONLY when set, so leaving them off reproduces the
    published arm and setting one addresses a distinct cache entry.
    """
    import argparse as _a
    ns = _a.Namespace()
    for k, v in dict(
            roster.PROVIDER_DEFAULTS,
            owlv2_query=None, gdino_query=None,
            gdino_text_threshold=None, score_threshold=None,
            yolo_model=None, tiling=args.tiling,
            radius=args.radius, op_threshold=0.0, limit=None,
            cache_dir=args.cache_dir, no_cache=False,
            vistas_input_size=args.vistas_input_size,
            vistas_revision=args.vistas_revision).items():
        setattr(ns, k, v)
    return ns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=roster.PROVIDER_DEFAULTS["gemini_model"],
                    help="Model spec (provider or provider:model_id). A bare "
                         "non-provider token is read as a Gemini model id.")
    ap.add_argument("split", nargs="?", default="richmond",
                    help="Benchmark split name (default richmond).")
    ap.add_argument("--cache-dir", default=os.path.join(REPO, ".model_cache"))
    ap.add_argument("--rampnet-op-threshold", type=float, default=None,
                    help="Score RampNet from analysis_out/op_cache/<split>.json floor "
                         "peaks at this threshold instead of the bundle's committed "
                         "detections. The bundle is the SHIPPED point (>=0.5519 on "
                         "richmond); this document recommends 0.30, and the gap "
                         "changes who gets credit for a recovery. Default: the bundle, "
                         "so published numbers are unchanged.")
    ap.add_argument("--radius", type=float, default=0.022)
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective")
    ap.add_argument("--vistas-input-size", type=int, nargs=2, metavar=("H", "W"),
                    default=None,
                    help="Override what the Vistas checkpoint actually sees. In the "
                         "signature only when set, so this addresses a DIFFERENT "
                         "cache entry than the published 384x384 arm (#126).")
    ap.add_argument("--vistas-revision", default=None)
    args = ap.parse_args()

    provider, model_id = model_spec(args.model)
    bundle = os.path.join(REPO, "benchmark", args.split)
    records, verdicts, _ = load_bundle(bundle)
    if verdicts is None:
        sys.exit(f"{bundle}: no verdicts.json -- this gate needs a reviewed split.")
    label, detector = build_detector(provider, model_id, records, compare_args(args))
    sig = detector.signature()
    cache = DetectionCache(args.cache_dir)
    radius_sq = radius_sq_for(args.radius)

    floor_peaks = None
    if args.rampnet_op_threshold is not None:
        cached, _ = read_cache(os.path.join(CACHE_DIR, f"{args.split}.json"))
        floor_peaks = {pd["pano"]: pd["preds"] for pd in cached}
        print(f"rampnet re-sourced from op_cache at >= {args.rampnet_op_threshold} "
              f"(bundle records are the shipped point and are NOT used)\n")

    n = both = r_only = c_only = neither = 0
    r_fp = c_fp = 0
    panos = missing = 0
    # (challenger preds, GT points rampnet MISSED) per pano, for the null below
    shift_rows = []
    for pid, entry in verdicts.items():
        gt = build_ground_truth(records[pid]["detections"], entry["dets"],
                                entry["missed"], entry["no_missed"])
        if not gt.fn_confirmed:
            continue
        cp = cache.get(cache_key(label, sig, args.split, pid))
        if cp is None:
            missing += 1
            continue
        if floor_peaks is not None:
            rp = [p for p in floor_peaks.get(pid, [])
                  if p[2] >= args.rampnet_op_threshold]
        else:
            rp = [(d["x_normalized"], d["y_normalized"], d["confidence"])
                  for d in records[pid]["detections"]]
        mr, mc = matched_gt(rp, gt.gt_points, radius_sq), matched_gt(cp, gt.gt_points, radius_sq)
        for i in range(len(gt.gt_points)):
            r, c = i in mr, i in mc
            both += r and c
            r_only += r and not c
            c_only += c and not r
            neither += not r and not c
        n += len(gt.gt_points)
        r_fp += score_pano(rp, gt).fp
        c_fp += score_pano(cp, gt).fp
        shift_rows.append((cp, [g for i, g in enumerate(gt.gt_points) if i not in mr]))
        panos += 1

    if not n:
        sys.exit("No recall-eligible panos with cached detections -- nothing to compare. "
                 "Run compare.py for this model/split first (and pass the SAME "
                 "--vistas-input-size, which is part of the cache key).")

    r_tp, c_tp, union = both + r_only, both + c_only, both + r_only + c_only
    r_miss = c_only + neither
    rn = ("rampnet" if floor_peaks is None
          else f"rampnet@{args.rampnet_op_threshold:g}")
    print(f"{args.split} complementarity — {rn} vs {label}  "
          f"({panos} recall-eligible panos, {n} GT ramps"
          + (f"; {missing} panos missing from cache" if missing else "") + ")\n")
    print(f"  {rn:18s} recall {r_tp / n:.3f}   ({r_tp}/{n})")
    print(f"  {label[:18]:18s} recall {c_tp / n:.3f}   ({c_tp}/{n})")
    print(f"  ORACLE-UNION recall {union / n:.3f}   ({union}/{n})   "
          f"<- ceiling if you could keep every right call")
    print()
    print(f"  found by BOTH        {both:4d}  ({both / n:.1%})")
    print(f"  {rn[:14]:14s} ONLY   {r_only:4d}  ({r_only / n:.1%})")
    print(f"  {label[:14]:14s} ONLY   {c_only:4d}  ({c_only / n:.1%})   "
          f"<- complementary gain (rampnet-miss n challenger-hit)")
    print(f"  found by NEITHER     {neither:4d}  ({neither / n:.1%})   "
          f"<- hard misses, no model helps")
    print()
    print(f"  Union recall lift over {rn}:  +{(union - r_tp) / n:.3f}  ({c_only} ramps)")
    if r_miss:
        print(f"  Of {rn}'s {r_miss} misses, {label} recovers {c_only} "
              f"({c_only / r_miss:.0%}); {neither} nobody finds")
        mean_null, max_null = complementary_null(shift_rows, radius_sq)
        exp = mean_null * r_miss
        print(f"    null (same boxes, wrong pano): {mean_null:.3f} "
              f"=> ~{exp:.0f} of those {c_only} are coincidence, "
              f"~{c_only - exp:.0f} attributable  (worst shift {max_null:.3f})")

    # The counterweight to the oracle number. A naive union keeps every box from
    # both models, so it pays both FP bills; co-located FPs would dedup, which is
    # why this is an upper bound on the cost and therefore a LOWER bound on the
    # union's precision.
    u_p = union / (union + r_fp + c_fp) if union + r_fp + c_fp else 0.0
    u_r = union / n
    u_f1 = 2 * u_p * u_r / (u_p + u_r) if u_p + u_r else 0.0
    r_p = r_tp / (r_tp + r_fp) if r_tp + r_fp else 0.0
    r_f1 = 2 * r_p * (r_tp / n) / (r_p + r_tp / n) if r_p + r_tp / n else 0.0
    print()
    print(f"  FP cost on these panos:  {rn} {r_fp}  |  {label} {c_fp}"
          f"   (a naive union pays ~both)")
    print(f"  {rn} alone:  P {r_p:.3f}  R {r_tp / n:.3f}  F1 {r_f1:.3f}")
    print(f"  NAIVE UNION:    P {u_p:.3f}  R {u_r:.3f}  F1 {u_f1:.3f}"
          f"   <- precision is a lower bound (no FP dedup)")
    print(f"  => a naive union {'BEATS' if u_f1 > r_f1 else 'LOSES TO'} {rn} alone "
          f"on F1 ({u_f1:.3f} vs {r_f1:.3f})")


if __name__ == "__main__":
    main()
