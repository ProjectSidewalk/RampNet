"""Does agreement with Gemini 4 Argon add anything beyond RampNet's own confidence? (#35)

``cascade_cost_35.py`` never ran the Gemini 4 Argon legs: its ``summary.json`` lists every
(leg, split) pair under ``gaps``. This script runs that analysis, unchanged, for two of them
and compares the result with the open-weight YOLO pano arms the study already scored. The
question behind it is whether a paid "Argon as teacher" distillation run is worth doing: if
gating on Argon buys less than gating on an open-weight detector on the open-imagery splits,
it is not.

**Nothing here is a new method.** Every per-pair file is ``cascade_cost_35.run_pair`` with the
study's arguments (T_hi 0.30, the default grid, seed 0, 2,000 resamples; every null shift
on richmond and 20 evenly spaced ones elsewhere, as in the study's cross-split sweep). Every interval is ``cascade_cost_35.bootstrap_delta`` (pano
resample, paired, setting held fixed). What is new is only *which* two prediction sets it is
pointed at: the study pairs a cascade row with the 0.30 baseline; here the same function also
pairs it with the best single threshold, with the threshold at matched recall, and with the
best open-weight arm's row.

Legs:

* ``gemini-4-argon`` -- the standing zero-shot leg. It emits no score, so its grid is the 15
  score-less settings (c_min = 0 only), like every chat VLM in the study.
* ``gemini-4-argon-confidence-on`` -- the #276 study leg (same prompt plus an elicited
  confidence). It has scores, so it gets the full 60-setting grid with c_min at 0 and its own
  score quartiles.
* open-weight: ``y11l_pano``, ``y11x_pano_h200``, ``y26_pano``. Their committed files in
  ``analysis_out/cascade_cost_35/`` are read where they exist and never rewritten; the pairs
  that study did not run (juneau, bayonne, laurens_gsv) are run here, into this script's own
  directory.

Three reads of each (split, leg), all from the per-pair file:

(Outputs go to ``analysis_out/cascade_cost_35/argon/``, a subdirectory like ``transfer/``, so
the study's own ``--check`` and ``summary.json`` -- which read the ``*.json`` files directly
under ``cascade_cost_35/`` -- are untouched; ``summary.json`` therefore still lists these
pairs under ``gaps``.)

* ``best_viable`` -- the study's rule (attributable dR >= 0.020, F1 >= baseline, F1 above
  threshold-only at the same T_lo), best by (F1, attributable dR). ``None`` when no row passes.
* ``best_by_f1`` -- the best grid row by F1, viable or not. **Chosen in sample**, from 15
  settings for the score-less leg and 60 for the others, so it favours a scored arm.
* ``fixed`` -- the study's one post-hoc setting (T_lo 0.05, r_gate R/2, c_min = the arm's
  median score, or all boxes for a score-less arm). Chosen on richmond x the Vistas arm, so it
  is not selected on any split or arm read here.

**Pools.** The cascade study has no pool. Two are reported here as count sums (TP, FP, recall
TP, recall GT added across splits, then P/R/F1 as ``aggregate`` computes them): ``open_276``
(``laurens_mapillary``, ``bayonne`` -- ``prompt_276_heldout.OPEN_POOL``) and ``open_rig``
(those two plus ``juneau`` and ``budapest_district5``). A pooled row uses ONE arm on every
split, and every arm is reported on its own. ``best_open_weight`` (per split and per pool)
is a further in-sample max over the three YOLO arms. No pooled interval is computed: the
study has no pooled bootstrap to reuse.

**Argon-only points.** An Argon box with no RampNet floor peak within the scoring radius
(0.022, wrapped, the scorer's distance) at a floor of 0.05 (everything the op_cache holds)
and 0.10 (the labeler's storage floor). Each group is scored on its own by ``score_pano``,
on the recall-confirmed panos only, because a pano whose missed-ramp scan was not attested
has no GT for a ramp RampNet never fired on. ``tp_new`` counts the matched ramps that the
0.30 baseline does not already match.

**Unique points, every arm** (PR #289 review). The same grouping for all five arms at the
0.05 floor, YOLO also at its headline conf 0.25, plus the "RampNet-blind" GT ramps (no
floor peak within the group radius) and how many of them each arm puts any box on. Three
configurations (``UNIQUE_CONFIGS``): R (primary), 2R regrouped and rescored, and R-group /
2R-score; both 2R reads are sensitivity rows. **Scoreboard reconciliation:** the op_cache
peaks at the scoreboard's RampNet operating point, beside the scoreboard's own row.

Inputs, all committed: ``benchmark/<split>/`` (GT), ``analysis_out/op_cache/<split>.json``,
``benchmark/model_detections/<leg>__<split>.json``, the YOLO pair files in
``analysis_out/cascade_cost_35/``, and ``analysis_out/scoreboard.json`` (read only). CPU only; no network, no ``.model_cache``, no model call.

    python scripts/analysis/cascade_argon_35.py            # run missing pairs, write argon_pairs.json
    python scripts/analysis/cascade_argon_35.py --check    # regenerate everything + byte-compare
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cascade_cost_35 as cc  # noqa: E402
from rampnet.detection_eval import radius_sq_for, score_pano  # noqa: E402
from rampnet.geometry import dist_sq  # noqa: E402
from complementarity import load_floor_peaks, matched_gt  # noqa: E402

ARGON = ("gemini-4-argon", "gemini-4-argon-confidence-on")
OPEN_WEIGHT = ("y11l_pano", "y11x_pano_h200", "y26_pano")
#: ``prompt_276_heldout.OPEN_POOL`` (not imported: that module pulls in the #276 harness).
OPEN_276 = ("laurens_mapillary", "bayonne")
OPEN_RIG = ("laurens_mapillary", "juneau", "bayonne", "budapest_district5")
POOLS = {"open_276": OPEN_276, "open_rig": OPEN_RIG}
#: The YOLO arms' headline confidence (``compare.py``: "headline F1 at conf 0.25"; the
#: scoreboard's ``operating_point`` for all three). Their detections are published down
#: to 0.05, so "all boxes" is the 0.05 floor.
YOLO_HEADLINE_CONF = 0.25
#: The scoreboard read for the threshold reconciliation (read only, never written).
SCOREBOARD = os.path.join(cc.REPO, "analysis_out", "scoreboard.json")
#: The unique-point comparison's floor: every RampNet peak the op_cache holds.
UNIQUE_FLOOR = 0.05
#: Unique-point configurations, as (group radius, score radius) in multiples of R.
#: ``R`` is the scorer's radius and the primary read. ``2R`` regroups AND rescores at
#: twice the radius, a sensitivity row for the looser box centres of the VLM arms (PR
#: #289 review, finding 2); 2R is generous to every arm, so it never replaces ``R``.
#: ``R_group_2R_score`` keeps the R grouping and rescores at 2R: the reviewer's read,
#: which isolates localisation on a fixed set of boxes.
UNIQUE_CONFIGS = {"R": (1, 1), "2R": (2, 2), "R_group_2R_score": (1, 2)}


def null_shifts(split):
    """The study's: every shift on richmond, 20 evenly spaced ones elsewhere
    (cascade_cost_35.md, Method)."""
    return "all" if split == cc.PRIMARY[0] else "20"


READS = ("best_viable", "best_by_f1", "fixed")
FLOORS = (0.05, 0.10)
OUT_DIR = os.path.join(cc.DEFAULT_OUT, "argon")
RESULT = "argon_pairs.json"


def pair_path(split, leg, out_dir):
    """This script's file for the pair, else the study's committed one, else ``None``."""
    name = cc.out_name(split, leg, cc.T_HI)
    for d in (out_dir, cc.DEFAULT_OUT):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def run_missing(out_dir):
    """Run every (split, leg) of ``ARGON + OPEN_WEIGHT`` that has published detections and
    no pair file yet, with the cross-split sweep's arguments. Returns the names written."""
    wrote = []
    for split in splits():
        gts = peaks = None
        for leg in ARGON + OPEN_WEIGHT:
            if pair_path(split, leg, out_dir):
                continue
            if gts is None:
                gts, peaks = cc.load_gts(split), load_floor_peaks(split)
            p, run = cc.run_pair(split, leg, null_shifts=null_shifts(split), gts=gts, peaks=peaks)
            if p is None:
                continue
            name = cc.out_name(split, leg, cc.T_HI)
            cc.write_json(os.path.join(out_dir, name), p)
            cc.record_run(out_dir, name, run)
            wrote.append(name)
            print(f"{split:20s} {leg:32s} {p['verdict']:10s} best F1 "
                  f"{p['best_by_f1']['F1']:.4f} (base {p['baseline']['F1']:.4f}) "
                  f"{run['elapsed_s']:.0f}s", flush=True)
    return wrote


def splits():
    have = set(cc.splits_with_op_cache())
    return [s for s in cc.all_benchmark_splits() if s in have]


def _counts(r):
    """``(tp, fp, recall tp, recall n_gt)`` of a report row (``fn`` is recall-side)."""
    return (r["tp"], r["fp"], r["n_gt"] - r["fn"], r["n_gt"])


def _prf(c):
    f1, r = cc._f1(*c)
    return {"tp": c[0], "fp": c[1], "fn": c[3] - c[2], "n_gt": c[3],
            "P": c[0] / (c[0] + c[1]) if c[0] + c[1] else 0.0, "R": r, "F1": f1}


def _brief(r):
    if r is None:
        return None
    m = r.get("threshold_only_at_matched_recall")
    return {"setting": [r["t_lo"], r["r_gate"], r["c_min"]],
            **{k: r[k] for k in ("tp", "fp", "fn", "n_gt", "P", "R", "F1", "dF1", "dR",
                                 "attributable_dR", "promoted_tp", "promoted_fp",
                                 "fp_per_attributable_ramp")},
            "thr_at_matched_recall": None if m is None else
            {k: m[k] for k in ("t", "tp", "fp", "fn", "F1")}}


def _thr_idx(peaks, pids, t):
    return {pid: tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t)
            for pid in pids}


def argon_only(cands, peaks, gts, kept_idx, radius_sq, floor, score_radius_sq=None):
    """Boxes by the best RampNet floor peak (>= ``floor``) within ``radius_sq`` (wrapped):
    ``none`` (no peak) / ``sub`` (best peak below T_hi) / ``kept`` (>= T_hi). Each group is
    scored alone by ``score_pano`` at ``score_radius_sq`` (default ``radius_sq``), on the
    recall-confirmed panos only. ``tp_new`` counts the GT ramps a group matches that the
    T_hi baseline (``kept_idx``, matched at the same score radius) does not.

    Named for its first use (Argon boxes), but nothing in it is Argon-specific: the
    unique-point comparison runs it on every arm.

    Example: one pano, peaks ``[(0.5, 0.5, 0.40), (0.2, 0.5, 0.15)]``, boxes at
    ``(0.5, 0.5)``, ``(0.2, 0.5)`` and ``(0.8, 0.5)`` -> one box each in ``kept``, ``sub``
    and ``none``.
    """
    score_radius_sq = radius_sq if score_radius_sq is None else score_radius_sq
    groups = {"none": {}, "sub": {}, "kept": {}}
    out = {g: {"boxes": 0, "tp": 0, "fp": 0, "ignored": 0, "tp_new": 0} for g in groups}
    n_panos = 0
    for pid, gt in gts.items():
        if not gt.fn_confirmed:
            continue
        n_panos += 1
        pk = [p for p in peaks.get(pid, []) if p[2] >= floor]
        by = {g: [] for g in groups}
        for c in cands.get(pid, []):
            near = [p[2] for p in pk if dist_sq(c[0], c[1], p[0], p[1], cc.PANO_SCALE_X,
                                                cc.PANO_SCALE_Y, wrap_x=True) < radius_sq]
            by["none" if not near else "kept" if max(near) >= cc.T_HI else "sub"].append(c)
        base = matched_gt([peaks.get(pid, [])[i] for i in kept_idx[pid]], gt.gt_points,
                          score_radius_sq)
        for g, boxes in by.items():
            s = score_pano(boxes, gt, score_radius_sq)
            o = out[g]
            o["boxes"] += len(boxes)
            o["tp"] += s.tp
            o["fp"] += s.fp
            o["ignored"] += s.ignored
            o["tp_new"] += len(matched_gt(boxes, gt.gt_points, score_radius_sq) - base)
    for o in out.values():
        o["precision"] = o["tp"] / (o["tp"] + o["fp"]) if o["tp"] + o["fp"] else None
    return {"floor": floor, "recall_confirmed_panos": n_panos, **out}


def _near(x, y, pts, radius_sq):
    return any(dist_sq(x, y, p[0], p[1], cc.PANO_SCALE_X, cc.PANO_SCALE_Y, wrap_x=True)
               < radius_sq for p in pts)


def blind_gt(peaks, gts, radius_sq, floor):
    """``{pid: [gt point, ...]}``: the "RampNet-blind" GT ramps -- no RampNet peak
    (>= ``floor``) within ``radius_sq``, wrapped -- on the recall-confirmed panos. These
    are the ramps no cascade setting can recover, since the cascade only promotes peaks.

    Example: GT at ``(0.5, 0.5)`` and ``(0.9, 0.5)``, one peak at ``(0.5, 0.5, 0.06)``,
    floor 0.05 -> only ``(0.9, 0.5)`` is blind; at floor 0.10 both are.
    """
    out = {}
    for pid, gt in gts.items():
        if not gt.fn_confirmed:
            continue
        pk = [p for p in peaks.get(pid, []) if p[2] >= floor]
        out[pid] = [g for g in gt.gt_points if not _near(g[0], g[1], pk, radius_sq)]
    return out


def blind_reached(cands, blind, radius_sq):
    """How many blind GT ramps have at least one of ``cands`` within ``radius_sq``
    (wrapped). Any box counts, not a 1:1 match: this asks whether the arm put a box on
    the ramp at all."""
    return sum(1 for pid, pts in blind.items() for g in pts
               if _near(g[0], g[1], cands.get(pid, []), radius_sq))


def _conf_filter(cands, t):
    return {pid: [c for c in cs if len(c) > 2 and c[2] is not None and c[2] >= t]
            for pid, cs in cands.items()}


def arm_variants(leg, cands):
    """``{variant: cands}``: every arm with all its boxes; a YOLO arm also at its headline
    confidence (``YOLO_HEADLINE_CONF``). Argon zero-shot has no score, and the confidence
    leg's elicited score has no shipped operating point, so both Argon legs are all-boxes."""
    out = {"all": cands}
    if leg in OPEN_WEIGHT:
        out[f"conf_ge_{YOLO_HEADLINE_CONF:g}"] = _conf_filter(cands, YOLO_HEADLINE_CONF)
    return out


def unique_points(cands, peaks, gts, kept_idx, floor=UNIQUE_FLOOR):
    """Per unique-point configuration (``UNIQUE_CONFIGS``): the blind GT count and, per arm
    and variant, the ``none`` group of ``argon_only`` (TP / FP / ignored / tp_new /
    precision) plus ``blind_gt_reached``. Blind GT is defined at the group radius and
    reached at the score radius."""
    res = {}
    for name, (g, s) in UNIQUE_CONFIGS.items():
        g_sq, s_sq = radius_sq_for(g * cc.RADIUS), radius_sq_for(s * cc.RADIUS)
        blind = blind_gt(peaks, gts, g_sq, floor)
        arms = {}
        for leg, cs in cands.items():
            arms[leg] = {}
            for var, vc in arm_variants(leg, cs).items():
                n = argon_only(vc, peaks, gts, kept_idx, g_sq, floor, s_sq)["none"]
                arms[leg][var] = {**n, "blind_gt_reached": blind_reached(vc, blind, s_sq)}
        res[name] = {"group_radius": g * cc.RADIUS, "score_radius": s * cc.RADIUS,
                     "floor": floor, "n_blind_gt": sum(len(v) for v in blind.values()),
                     "arms": arms}
    return res


def scoreboard_operating_point(path=SCOREBOARD):
    """``(RampNet's scoreboard operating point, its per-split rows)`` from
    ``scoreboard.json``, or ``(None, {})`` if the file is absent."""
    if not os.path.exists(path):
        return None, {}
    with open(path, encoding="utf-8") as f:
        sb = json.load(f)
    m = next(m for m in sb["models"] if m["model"] == "rampnet")
    return m["operating_point"], sb["per_split"]["rampnet"]


def compare_split(split, out_dir, scoreboard=None):
    """One split: every arm's three reads, the paired intervals, the Argon-only counts,
    the unique-point comparison across all arms, and the scoreboard reconciliation.
    ``scoreboard`` is ``scoreboard_operating_point()``'s result (read here if omitted)."""
    pay = {}
    for leg in ARGON + OPEN_WEIGHT:
        path = pair_path(split, leg, out_dir)
        if path:
            with open(path, encoding="utf-8") as f:
                pay[leg] = json.load(f)
            pay[leg]["_file"] = os.path.relpath(path, cc.REPO).replace(os.sep, "/")
    if not any(leg in pay for leg in ARGON):
        return None
    gts, peaks = cc.load_gts(split), load_floor_peaks(split)
    pids = list(gts)
    radius_sq = radius_sq_for(cc.RADIUS)
    scorer = cc._Scorer(gts, peaks, radius_sq)
    kept = _thr_idx(peaks, pids, cc.T_HI)
    cands = {leg: cc.load_challenger(split, leg)[0] for leg in pay}
    ref = pay[next(iter(pay))]
    for p in pay.values():      # every arm saw the same RampNet side
        assert p["baseline"] == ref["baseline"], split
        assert p["threshold_only_best"] == ref["threshold_only_best"], split
        assert (p["t_hi"], p["radius"], p["args"]["null_shifts"]) == (
            cc.T_HI, cc.RADIUS, null_shifts(split)), (split, p["challenger"])

    def idx_of(leg, row):
        idx = cc._setting_indices(scorer, cands[leg], cc.T_HI, row["t_lo"], row["r_gate"],
                                  row["c_min"])
        rep = cc._fast_report([scorer.pano(pid, idx[pid]) for pid in pids])
        assert (rep["tp"], rep["fp"], rep["fn"]) == (row["tp"], row["fp"], row["fn"]), (
            split, leg, rep, row)
        return idx

    def boot(a, b):
        d = cc.bootstrap_delta(scorer, a, b)
        return {"dF1_ci95": d["dF1_ci95"], "dR_ci95": d["dR_ci95"]}

    tb = ref["threshold_only_best"]
    thr_best_idx = _thr_idx(peaks, pids, tb["t"])
    arms = {leg: {"file": p["_file"], "verdict": p["verdict"],
                  "scored": len(p["c_min_grid"]) > 1, "grid_settings": len(p["grid"]),
                  "boxes": sum(len(v) for v in cands[leg].values()),
                  "false_viable_rate": (p["calibration"] or {}).get("false_viable_rate"),
                  "selection_aware_viable_frac":
                      (p["bootstrap_selection_aware"] or {}).get("viable_frac"),
                  "best_viable": _brief(p["best_viable"]),
                  "best_by_f1": _brief(p["best_by_f1"]),
                  "fixed": _brief(p["fixed_setting"]),
                  "vs_baseline": {k: {"dF1_ci95": v["dF1_ci95"], "dR_ci95": v["dR_ci95"]}
                                  for k, v in p["bootstrap_vs_baseline"].items()}}
            for leg, p in pay.items()}
    row_of = {"best_viable": "best_viable", "best_by_f1": "best_by_f1",
              "fixed": "fixed_setting"}
    best_open, paired = {}, {}
    for read in READS:
        have = [leg for leg in OPEN_WEIGHT if leg in pay and pay[leg][row_of[read]]]
        best_open[read] = (max(have, key=lambda leg: (pay[leg][row_of[read]]["F1"], leg))
                           if have else None)
    for leg in ARGON:
        if leg not in pay:
            continue
        paired[leg] = {}
        for read in READS:
            row = pay[leg][row_of[read]]
            if row is None:
                paired[leg][read] = None
                continue
            a = idx_of(leg, row)
            m = row.get("threshold_only_at_matched_recall")
            bo = best_open[read]
            orow = pay[bo][row_of[read]] if bo else None
            paired[leg][read] = {
                "minus_threshold_best": {
                    "dF1": row["F1"] - tb["F1"], **boot(thr_best_idx, a)},
                "minus_threshold_at_matched_recall": None if m is None else {
                    "t": m["t"], "dF1": row["F1"] - m["F1"],
                    **boot(_thr_idx(peaks, pids, m["t"]), a)},
                "minus_best_open_weight": None if orow is None else {
                    "arm": bo, "dF1": row["F1"] - orow["F1"], "dR": row["R"] - orow["R"],
                    **boot(idx_of(bo, orow), a)}}
    only = {leg: [argon_only(cands[leg], peaks, gts, kept, radius_sq, fl) for fl in FLOORS]
            for leg in ARGON if leg in pay}
    op, sb_rows = scoreboard if scoreboard is not None else scoreboard_operating_point()
    sb_row = sb_rows.get(split)
    sb = None if op is None else {
        "operating_point": op,
        "op_cache_at_operating_point": {k: v for k, v in cc._fast_report(
            [scorer.pano(pid, i) for pid, i in _thr_idx(peaks, pids, op).items()]).items()
            if k in ("tp", "fp", "fn", "n_gt", "F1")},
        "scoreboard": None if sb_row is None else {
            k: sb_row[k] for k in ("tp", "fp", "fn", "n_gt_recall", "f1", "bundle_floor")
            if k in sb_row}}
    return {"split": split, "n_panos": ref["n_panos"], "n_gt_recall": ref["n_gt_recall"],
            "null_shifts": ref["null"]["shifts"],
            "baseline": ref["baseline"],
            "threshold_only_best": {k: tb[k] for k in ("t", "tp", "fp", "fn", "n_gt", "P",
                                                       "R", "F1")},
            "arms": arms, "best_open_weight": best_open, "paired": paired,
            "argon_only": only,
            "unique_points": unique_points(cands, peaks, gts, kept),
            "scoreboard_reconciliation": sb}


def pool(rows, name, members):
    """Count-pooled F1 over ``members``; ``None`` for an arm missing on any of them."""
    rs = [r for r in rows if r["split"] in members]
    if len(rs) != len(members):
        return None

    def add(get):
        cs = [get(r) for r in rs]
        if any(c is None for c in cs):
            return None
        return _prf(tuple(sum(c[i] for c in cs) for i in range(4)))

    out = {"splits": list(members),
           "baseline": add(lambda r: _counts(r["baseline"])),
           "threshold_only_best_per_split": add(lambda r: _counts(r["threshold_only_best"])),
           "arms": {}}
    for leg in ARGON + OPEN_WEIGHT:
        out["arms"][leg] = {
            read: add(lambda r, leg=leg, read=read:
                      _counts(r["arms"][leg][read])
                      if leg in r["arms"] and r["arms"][leg][read] else None)
            for read in ("best_by_f1", "fixed")}
    for read in ("best_by_f1", "fixed"):
        have = [leg for leg in OPEN_WEIGHT if out["arms"][leg][read]]
        out[f"best_open_weight_{read}"] = (
            max(have, key=lambda leg: (out["arms"][leg][read]["F1"], leg)) if have else None)
    return out


def pool_unique(rows, members):
    """Count sums of ``unique_points`` over ``members`` per configuration, arm and
    variant; ``None`` for an arm/variant missing on any member split."""
    rs = [r for r in rows if r["split"] in members]
    if len(rs) != len(members):
        return None
    keys = ("boxes", "tp", "fp", "ignored", "tp_new", "blind_gt_reached")
    out = {}
    for name in UNIQUE_CONFIGS:
        arms = {}
        for leg in ARGON + OPEN_WEIGHT:
            variants = {v for r in rs for v in r["unique_points"][name]["arms"].get(leg, {})}
            arms[leg] = {}
            for var in sorted(variants):
                per = [r["unique_points"][name]["arms"].get(leg, {}).get(var) for r in rs]
                if any(x is None for x in per):
                    arms[leg][var] = None
                    continue
                o = {k: sum(x[k] for x in per) for k in keys}
                o["precision"] = o["tp"] / (o["tp"] + o["fp"]) if o["tp"] + o["fp"] else None
                arms[leg][var] = o
        out[name] = {"n_blind_gt": sum(r["unique_points"][name]["n_blind_gt"] for r in rs),
                     "arms": arms}
    return out


def build(out_dir):
    sb = scoreboard_operating_point()
    rows = [r for r in (compare_split(s, out_dir, sb) for s in splits()) if r]
    return {"question": "does gating RampNet's sub-threshold peaks on Gemini 4 Argon add "
                        "more than gating on an open-weight YOLO pano arm? (#35)",
            "method": "cascade_cost_35.run_pair and bootstrap_delta, unchanged",
            "t_hi": cc.T_HI, "radius": cc.RADIUS,
            "null_shifts": "all on richmond, 20 evenly spaced elsewhere (the study's)",
            "argon_legs": list(ARGON), "open_weight_legs": list(OPEN_WEIGHT),
            "op_cache_commit_note": cc.OP_CACHE_NOTE,
            "selection_note": ("best_viable and best_by_f1 are chosen in sample (15 settings "
                               "for the score-less zero-shot leg, 60 for a scored arm); "
                               "fixed is the study's post-hoc setting, selected on neither "
                               "these splits nor these arms. 'best open-weight' is a "
                               "further in-sample max over the three YOLO arms, per split "
                               "(best_open_weight) and per pool (best_open_weight_<read>): "
                               "up to 3 x 60 = 180 candidate rows against Argon zero-shot's "
                               "15 and the confidence leg's 60, so each YOLO arm is also "
                               "reported on its own"),
            "unique_points_note": ("boxes with no RampNet op_cache peak >= floor within the "
                                   "group radius, scored alone by score_pano at the score "
                                   "radius on recall-confirmed panos; blind GT = GT ramps "
                                   "with no such peak within the group radius; reached = "
                                   "any box of the arm within the score radius. YOLO also "
                                   f"at its headline conf >= {YOLO_HEADLINE_CONF:g}. R is "
                                   "primary; 2R and R_group_2R_score are sensitivity "
                                   "reads, generous to every arm"),
            "splits": rows,
            "pools": {n: pool(rows, n, m) for n, m in POOLS.items()},
            "pools_unique_points": {n: pool_unique(rows, m) for n, m in POOLS.items()}}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _f(v, fmt=".4f"):
    return "–" if v is None else format(v, fmt)


def _ci(d):
    return "–" if d is None else (f"{d['dF1']:+.4f} [{d['dF1_ci95'][0]:+.3f}, "
                                  f"{d['dF1_ci95'][1]:+.3f}]")


def tables(res):
    L = []
    for read in ("best_by_f1", "fixed", "best_viable"):
        L += [f"**{read}**", "",
              "| split | n GT | baseline F1 @0.30 | best thr-only F1 (t) | best open-weight "
              "(arm) | Argon zero-shot | Argon + confidence |",
              "|---|---:|---:|---:|---|---:|---:|"]
        for r in res["splits"]:
            bo = r["best_open_weight"][read]
            a = r["arms"]
            cell = lambda leg: _f((a.get(leg, {}).get(read) or {}).get("F1"))   # noqa: E731
            tb = r["threshold_only_best"]
            L.append(f"| {r['split']} | {r['n_gt_recall']} | {r['baseline']['F1']:.4f} | "
                     f"{tb['F1']:.4f} ({tb['t']:g}) | "
                     + (f"{cell(bo)} ({bo})" if bo else "–")
                     + f" | {cell(ARGON[0])} | {cell(ARGON[1])} |")
        if read != "best_viable":
            for name, p in res["pools"].items():
                if p is None:
                    continue
                bo = p[f"best_open_weight_{read}"]
                cell = lambda leg: _f((p["arms"][leg][read] or {}).get("F1"))   # noqa: E731
                L.append(f"| **pool {name}** | {p['baseline']['n_gt']} | "
                         f"{p['baseline']['F1']:.4f} | "
                         f"{p['threshold_only_best_per_split']['F1']:.4f} (per split) | "
                         + (f"{cell(bo)} ({bo})" if bo else "–")
                         + f" | {cell(ARGON[0])} | {cell(ARGON[1])} |")
        L.append("")
    L += ["**Paired pano-bootstrap ΔF1 (Argon cascade minus the comparator), 95% interval**",
          "", "| split | leg | read | vs best threshold | vs threshold at matched recall | "
          "vs best open-weight (arm) |", "|---|---|---|---|---|---|"]
    for r in res["splits"]:
        for leg, reads in r["paired"].items():
            for read, d in reads.items():
                if d is None:
                    continue
                bo = d["minus_best_open_weight"]
                L.append(f"| {r['split']} | {leg} | {read} | {_ci(d['minus_threshold_best'])} | "
                         f"{_ci(d['minus_threshold_at_matched_recall'])} | "
                         + (f"{_ci(bo)} ({bo['arm']})" if bo else "–") + " |")
    L += ["", "**Argon boxes by the RampNet floor peak within the scoring radius "
          "(recall-confirmed panos; each group scored alone)**", "",
          "| split | leg | floor | no peak: boxes | TP / FP / ignored | precision | TP not in "
          "the 0.30 baseline | sub-threshold peak: TP / FP | peak >= 0.30: TP / FP |",
          "|---|---|---:|---:|---|---:|---:|---|---|"]
    for r in res["splits"]:
        for leg, per in r["argon_only"].items():
            for o in per:
                n, s, k = o["none"], o["sub"], o["kept"]
                L.append(f"| {r['split']} | {leg} | {o['floor']:g} | {n['boxes']} | "
                         f"{n['tp']} / {n['fp']} / {n['ignored']} | {_f(n['precision'], '.3f')} "
                         f"| {n['tp_new']} | {s['tp']} / {s['fp']} | {k['tp']} / {k['fp']} |")
    L += ["", "**Each arm on its own, pooled F1 (no in-sample max over the YOLO arms)**", "",
          "| pool | read | " + " | ".join(ARGON + OPEN_WEIGHT) + " |",
          "|---|---|" + "---:|" * len(ARGON + OPEN_WEIGHT)]
    for name, p in res["pools"].items():
        if p is None:
            continue
        for read in ("best_by_f1", "fixed"):
            L.append(f"| {name} | {read} | " + " | ".join(
                _f((p["arms"][leg][read] or {}).get("F1")) for leg in ARGON + OPEN_WEIGHT)
                + " |")
    L += unique_tables(res)
    L += ["", "**Scoreboard reconciliation: op_cache peaks at the scoreboard's RampNet "
          "operating point vs the scoreboard row**", "",
          "| split | operating point | op_cache F1 (TP / FP / FN) | scoreboard F1 (TP / FP / "
          "FN) | scoreboard bundle floor |", "|---|---:|---|---|---:|"]
    for r in res["splits"]:
        sb = r.get("scoreboard_reconciliation")
        if not sb:
            continue
        o, b = sb["op_cache_at_operating_point"], sb["scoreboard"]
        L.append(f"| {r['split']} | {sb['operating_point']:g} | {o['F1']:.4f} ({o['tp']} / "
                 f"{o['fp']} / {o['fn']}) | "
                 + ("–" if b is None else f"{b['f1']:.4f} ({b['tp']} / {b['fp']} / {b['fn']})")
                 + f" | {_f((b or {}).get('bundle_floor'))} |")
    return "\n".join(L) + "\n"


def _u(o):
    """``TP / FP (precision), reached`` for one unique-point cell."""
    if o is None:
        return "–"
    return f"{o['tp']} / {o['fp']} ({_f(o['precision'], '.2f')}), {o['blind_gt_reached']}"


def unique_tables(res):
    """The unique-point comparison: one table per configuration, every split plus pools.
    A cell is ``TP / FP (precision), blind GT reached``."""
    conf = f"conf_ge_{YOLO_HEADLINE_CONF:g}"
    cols = ([(leg, "all") for leg in ARGON] + [(leg, conf) for leg in OPEN_WEIGHT]
            + [(leg, "all") for leg in OPEN_WEIGHT])
    head = ([f"{leg} (all)" for leg in ARGON]
            + [f"{leg} @{YOLO_HEADLINE_CONF:g}" for leg in OPEN_WEIGHT]
            + [f"{leg} (all)" for leg in OPEN_WEIGHT])
    L = []
    for name in UNIQUE_CONFIGS:
        L += ["", f"**Unique points, {name}: boxes with no RampNet peak >= "
              f"{UNIQUE_FLOOR:g} within the group radius; cell = TP / FP (precision), "
              "RampNet-blind GT reached**", "",
              "| split | blind GT | " + " | ".join(head) + " |",
              "|---|---:|" + "---|" * len(cols)]
        rows = [(r["split"], r["unique_points"][name]) for r in res["splits"]]
        rows += [(f"**pool {n}**", pu[name]) for n, pu in res["pools_unique_points"].items()
                 if pu is not None]
        for label, u in rows:
            L.append(f"| {label} | {u['n_blind_gt']} | " + " | ".join(
                _u(u["arms"].get(leg, {}).get(var)) for leg, var in cols) + " |")
    return L


def own_pair_files(out_dir):
    return [n for n in cc.pair_files(out_dir) if n != RESULT]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="Regenerate every per-pair file in --out from its recorded args, "
                         "rebuild argon_pairs.json, and byte-compare; exit 1 on any "
                         "difference. Writes nothing.")
    ap.add_argument("--tables", action="store_true",
                    help="Print the markdown tables from the committed argon_pairs.json.")
    args = ap.parse_args()
    path = os.path.join(args.out, RESULT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # the tables print a Greek delta
    if args.tables:
        with open(path, encoding="utf-8") as f:
            print(tables(json.load(f)))
        return
    if args.check:
        bad = cc.check(args.out, own_pair_files(args.out))
        with open(path, encoding="utf-8", newline="") as f:
            same = f.read() == cc.dumps(build(args.out))
        print(f"{'same' if same else 'DIFF'}  {RESULT}")
        if not same:
            bad.append(RESULT)
        print(f"{len(bad)} differ" + ("" if not bad else ": " + ", ".join(bad)))
        sys.exit(1 if bad else 0)
    run_missing(args.out)
    res = build(args.out)
    cc.write_json(path, res)
    print(tables(json.loads(cc.dumps(res))))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
