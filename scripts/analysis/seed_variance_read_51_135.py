"""Apply the pre-registered seed-variance read (#51, #135) to the scored replicates.

The question: is the 0.039 pooled-US7 F1 gap between RampNet and the ``y11x_tiles``
baseline (``docs/operating_point_parity_51.md``) bigger than run-to-run seed noise? Two
campaigns of three replicates each were trained to answer it -- Campaign A (YOLO, seeds
1/2/3, Tillicum) and Campaign B (RampNet Stage 2, seeds 1/2/3, klone). Their reading was
fixed before any replicate was scored, in ``docs/seed_variance_51_135.md`` (Amendment 1,
ratified 2026-09-04). This script IS that reading, as code:

  * the statistic is the macro-mean F1 over ``POOLED_SPLITS`` (the seven US splits), each
    replicate at its own uniform threshold selected on ``sao_paulo`` -- the parity
    protocol, so ``select_threshold``/``macro_at`` are imported from
    ``operating_point_parity_51.py`` rather than re-implemented;
  * a Campaign A replicate is read at its best ``metrics/mAP50-95(B)`` epoch <= 44 from
    its own ``results.csv`` -- that column, not the 0.1/0.9 fitness blend (they do not
    always peak at the same epoch; s2 is the case in point);
  * ``s_A`` and ``s_B`` are the sample SDs over each campaign's three replicates,
    ``s_gap = sqrt(s_A^2 + s_B^2)``, and the A1.1 bands are applied to ``s_gap``;
  * ``s_B`` is also read against #135's paired MDE of 0.0063 (A1.2).

Two SECONDARY reads travel in the same artifact, fenced off and never fed into ``s_gap``:
the YOLO replicates at their as-saved ``best.pt`` (<= 60 epochs), and both arms on
``manual_gold``. They were added on 2026-09-15, after the campaigns finished but before
any number was looked at; they are descriptive, not decisions.

Inputs, all committed under ``docs/data/seed_variance_51_135/``:

  y11x_tiles_s{1,2,3}/results.csv     Ultralytics per-epoch metrics (the epoch pick)
  yolo/<split>_tiles.txt              compare.py --sweep report, six YOLO legs per file
  rampnet_s{1,2,3}/<split>.json       op_cache written by operating_point_curve.py
                                      extract --checkpoint, floor 0.05, no TTA
  env.txt, driver.log                 the makelab2 run that produced the above

Usage:
    python scripts/analysis/seed_variance_read_51_135.py            # writes the JSON
    python scripts/analysis/seed_variance_read_51_135.py --check    # exit 1 if stale
    python scripts/analysis/seed_variance_read_51_135.py --markdown # tables for the doc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)

from analysis.operating_point_parity_51 import (  # noqa: E402
    DEFAULT_DEV, OUT_JSON as PARITY_JSON, POOLED_SPLITS, RAMPNET, collect_rampnet,
    macro_at, parse_sweeps, rampnet_grid, select_threshold)
from rampnet.detection_eval import GroundTruth, aggregate, radius_sq_for, score_pano  # noqa: E402

DATA = os.path.join(REPO, "docs", "data", "seed_variance_51_135")
OUT_JSON = os.path.join(REPO, "docs", "data", "seed_variance_51_135.json")
RUN_A_EPOCH1 = os.path.join(REPO, "docs", "data", "run_a_84_detections",
                            "run_a_epoch_1__manual_gold.json")

SEEDS = (1, 2, 3)
MAX_EPOCH = 44                        # the seed-0 arm's best.pt came from ~ep44
MAP5095 = "metrics/mAP50-95(B)"       # the selection column, per the pre-registration
MAP50 = "metrics/mAP50(B)"
SPLITS_AS_RUN = POOLED_SPLITS + ("sao_paulo", "manual_gold")
YOLO_ARM = "y11x_tiles"
SEED0_YOLO_LEG = "y11x_tiles"         # the n=1 arm the 0.039 was measured on

# Amendment 1.1: bands on s_gap, disjoint, each endpoint in exactly one row.
BAND_REAL, BAND_AMBIGUOUS, BAND_INDISTINGUISHABLE = 0.010, 0.020, math.inf
# Amendment 1.2: s_B against the paired epoch-to-epoch MDE measured in #135.
PAIRED_MDE_135 = 0.0063
# The published operating points, for the manual_gold secondary only.
PROTOCOL_THRESHOLD = {"rampnet": 0.30, "yolo": 0.25}


# --------------------------------------------------------------------------- #
# Campaign A: which epoch each replicate is read at
# --------------------------------------------------------------------------- #
def read_results_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Ultralytics pads column names with spaces in some versions.
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def pick_epoch(results_csv, max_epoch=MAX_EPOCH, column=MAP5095):
    """``(epoch, value)`` of the best ``column`` among epochs <= ``max_epoch``.

    Ties go to the earlier epoch, which is what Ultralytics' own best.pt logic does
    (it only overwrites on a strict improvement).
    """
    rows = [r for r in read_results_csv(results_csv) if int(r["epoch"]) <= max_epoch]
    if not rows:
        raise ValueError(f"{results_csv}: no epochs <= {max_epoch}")
    best = max(rows, key=lambda r: (float(r[column]), -int(r["epoch"])))
    return int(best["epoch"]), float(best[column])


def fitness_epoch(results_csv, max_epoch=MAX_EPOCH):
    """The epoch the 0.1/0.9 fitness blend would pick. NOT the read -- kept so the
    test can show the two columns disagree on s2, which is why the column is named."""
    rows = [r for r in read_results_csv(results_csv) if int(r["epoch"]) <= max_epoch]
    best = max(rows, key=lambda r: (0.1 * float(r[MAP50]) + 0.9 * float(r[MAP5095]),
                                    -int(r["epoch"])))
    return int(best["epoch"])


def primary_leg(seed, data=DATA):
    ep, _ = pick_epoch(os.path.join(data, f"{YOLO_ARM}_s{seed}", "results.csv"))
    return f"{YOLO_ARM}_s{seed}_ep{ep}"


def secondary_leg(seed):
    return f"{YOLO_ARM}_s{seed}_best"


# --------------------------------------------------------------------------- #
# collecting the scored sweeps
# --------------------------------------------------------------------------- #
def collect_yolo_seed(data=DATA):
    """``{leg: {split: {threshold: metrics}}}`` over ``yolo/<split>_tiles.txt``."""
    cells = {}
    for split in SPLITS_AS_RUN:
        path = os.path.join(data, "yolo", f"{split}_tiles.txt")
        if not os.path.exists(path):
            continue
        for model, sweep in parse_sweeps(path).items():
            cells.setdefault(model, {})[split] = sweep
    return cells


def collect_rampnet_seed(seed, grid, data=DATA):
    return collect_rampnet(grid, op_cache=os.path.join(data, f"rampnet_s{seed}"),
                           splits=SPLITS_AS_RUN)


def read_replicate(per_split, dev_split=DEFAULT_DEV):
    """One replicate through the parity protocol: threshold on the dev split, macro F1
    over the pool. None if the replicate is missing the dev split or any pooled split."""
    thr = select_threshold(per_split, dev_split)
    if thr is None:
        return None
    pooled = macro_at(per_split, thr)
    if pooled is None:
        return None
    return {
        "selected_threshold": thr,
        "at_floor": thr <= min(per_split[dev_split]),
        "pooled": pooled,
        "per_split": {s: per_split[s][thr] for s in SPLITS_AS_RUN
                      if s in per_split and thr in per_split[s]},
    }


def manual_gold_read(per_split, protocol_thr):
    """The manual_gold secondary: F1 at the arm's published operating point, and the
    max F1 over the sweep with the threshold that gave it."""
    sweep = per_split.get("manual_gold")
    if not sweep:
        return None
    best_thr, best = max(sweep.items(), key=lambda kv: (kv[1]["f1"], kv[0]))
    at = sweep.get(protocol_thr)
    return {
        "protocol_threshold": protocol_thr,
        "f1_at_protocol": None if at is None else at["f1"],
        "max_f1": best["f1"],
        "max_f1_threshold": best_thr,
        "p_at_max": best["p"], "r_at_max": best["r"],
    }


def run_a_epoch1_manual_gold(grid, gts_from, radius_sq=None):
    """Run A epoch 1 (seed 42, same 1-epoch recipe) re-scored on manual_gold with the
    same function as the replicates, for the secondary table. Its committed dump carries
    detections only, so GT comes from a replicate's op_cache -- ``extract`` wrote that
    GT from the same bundle. Reported beside the seeds, never pooled into s_B."""
    if not os.path.exists(RUN_A_EPOCH1) or not os.path.exists(gts_from):
        return None
    if radius_sq is None:
        radius_sq = radius_sq_for()
    with open(RUN_A_EPOCH1, encoding="utf-8") as fh:
        dump = json.load(fh)
    with open(gts_from, encoding="utf-8") as fh:
        cache = json.load(fh)
    gts = {p["pano"]: GroundTruth([tuple(q) for q in p["gt"]["gt_points"]],
                                  [tuple(q) for q in p["gt"]["ignore_points"]],
                                  bool(p["gt"]["fn_confirmed"]))
           for p in cache["panos"]}
    dets = dump["detections"]
    if set(dets) != set(gts):
        raise SystemExit("run_a_epoch_1 dump and the replicate op_cache cover different "
                         f"manual_gold panos ({len(dets)} vs {len(gts)})")
    sweep = {}
    for thr in grid:
        rep = aggregate([
            score_pano([tuple(q) for q in dets[pid] if q[2] >= thr], gt, radius_sq=radius_sq)
            for pid, gt in gts.items()])
        sweep[thr] = {"p": rep.precision, "r": rep.recall, "f1": rep.f1,
                      "tp": rep.tp, "fp": rep.fp, "fn": rep.fn}
    out = manual_gold_read({"manual_gold": sweep}, PROTOCOL_THRESHOLD["rampnet"])
    out["label"] = dump["model"]
    out["checkpoint_fingerprint"] = dump["signature"]["checkpoint_fingerprint"]
    return out


# --------------------------------------------------------------------------- #
# the statistics and the bands
# --------------------------------------------------------------------------- #
def sd(values):
    """Sample SD (n-1). None unless all three replicates are present."""
    vals = [v for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def band_gap(s_gap):
    """Amendment 1.1. Endpoints belong to exactly one row."""
    if s_gap is None:
        return None
    if s_gap < BAND_REAL:
        return "real"
    if s_gap < BAND_AMBIGUOUS:
        return "ambiguous"
    return "indistinguishable"


def band_b(s_b):
    """Amendment 1.2."""
    if s_b is None:
        return None
    return "below_paired_mde" if s_b < PAIRED_MDE_135 else "dominates_paired_mde"


def published_reference():
    """The n=1 numbers the campaigns put error bars on, read from the parity artifact so
    they cannot drift from docs/operating_point_parity_51.md."""
    with open(PARITY_JSON, encoding="utf-8") as fh:
        parity = json.load(fh)
    rn = parity["models"][RAMPNET]
    yo = parity["models"][SEED0_YOLO_LEG]
    return {
        "rampnet_f1": rn["pooled"]["f1"], "rampnet_threshold": rn["selected_threshold"],
        "yolo_f1": yo["pooled"]["f1"], "yolo_threshold": yo["selected_threshold"],
        "gap": rn["pooled"]["f1"] - yo["pooled"]["f1"],
        "rampnet_manual_gold_f1_at_selected": rn["per_split"].get("manual_gold", {}).get("f1"),
        "yolo_manual_gold_f1_at_selected": yo["per_split"].get("manual_gold", {}).get("f1"),
    }


def build(data=DATA, dev_split=DEFAULT_DEV):
    grid = rampnet_grid()
    yolo = collect_yolo_seed(data)

    # --- Campaign A, primary ---------------------------------------------------
    camp_a = {}
    for seed in SEEDS:
        csv_path = os.path.join(data, f"{YOLO_ARM}_s{seed}", "results.csv")
        ep, val = pick_epoch(csv_path)
        leg = f"{YOLO_ARM}_s{seed}_ep{ep}"
        row = read_replicate(yolo.get(leg, {}), dev_split)
        camp_a[f"s{seed}"] = {
            "leg": leg, "epoch": ep, "map5095_at_epoch": val,
            "fitness_epoch_would_be": fitness_epoch(csv_path),
            "read": row,
            "manual_gold": manual_gold_read(yolo.get(leg, {}), PROTOCOL_THRESHOLD["yolo"]),
        }

    # --- Campaign B, primary ---------------------------------------------------
    camp_b = {}
    for seed in SEEDS:
        per_split = collect_rampnet_seed(seed, grid, data)
        cache = os.path.join(data, f"rampnet_s{seed}", "manual_gold.json")
        label = None
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                label = json.load(fh)["meta"].get("model")
        camp_b[f"s{seed}"] = {
            "leg": f"rampnet_s{seed}", "model_label": label,
            "read": read_replicate(per_split, dev_split),
            "manual_gold": manual_gold_read(per_split, PROTOCOL_THRESHOLD["rampnet"]),
        }

    # --- the pre-registered statistics ----------------------------------------
    f1_a = [r["read"]["pooled"]["f1"] if r["read"] else None for r in camp_a.values()]
    f1_b = [r["read"]["pooled"]["f1"] if r["read"] else None for r in camp_b.values()]
    s_a, s_b = sd(f1_a), sd(f1_b)
    complete_a, complete_b = all(v is not None for v in f1_a), all(v is not None for v in f1_b)
    s_gap = math.sqrt(s_a ** 2 + s_b ** 2) if (complete_a and complete_b) else None
    ref = published_reference()

    stats = {
        "campaign_a_f1": f1_a, "campaign_b_f1": f1_b,
        "mean_a": statistics.fmean(f1_a) if complete_a else None,
        "mean_b": statistics.fmean(f1_b) if complete_b else None,
        "s_A": s_a, "s_B": s_b, "s_gap": s_gap,
        "gap_published": ref["gap"],
        "gap_of_means": (statistics.fmean(f1_b) - statistics.fmean(f1_a)
                         if complete_a and complete_b else None),
        "z_gap": (ref["gap"] / s_gap if s_gap else None),
        "band_gap": band_gap(s_gap),
        "band_gap_edges": {"real_below": BAND_REAL, "ambiguous_below": BAND_AMBIGUOUS},
        "s_A_only_band": band_gap(s_a) if s_a is not None and s_gap is None else None,
        "band_b": band_b(s_b), "paired_mde_135": PAIRED_MDE_135,
    }

    # --- secondary: YOLO at as-saved best.pt ------------------------------------
    secondary_best = {}
    for seed in SEEDS:
        leg = secondary_leg(seed)
        secondary_best[f"s{seed}"] = {
            "leg": leg,
            "read": read_replicate(yolo.get(leg, {}), dev_split),
            "manual_gold": manual_gold_read(yolo.get(leg, {}), PROTOCOL_THRESHOLD["yolo"]),
        }
    f1_best = [r["read"]["pooled"]["f1"] if r["read"] else None
               for r in secondary_best.values()]

    # --- secondary: manual_gold ------------------------------------------------
    mg_a = [r["manual_gold"] for r in camp_a.values()]
    mg_b = [r["manual_gold"] for r in camp_b.values()]
    mg_best = [r["manual_gold"] for r in secondary_best.values()]

    def _sd_of(rows, key):
        return sd([r[key] if r else None for r in rows])

    secondary = {
        "note": ("Post-hoc reads added 2026-09-15 before any number was seen. Reported "
                 "beside the pre-registered statistic; never enter s_gap or the bands."),
        "yolo_best_pt": {
            "replicates": secondary_best,
            "f1": f1_best, "mean": statistics.fmean(f1_best) if all(v is not None for v in f1_best) else None,
            "sd": sd(f1_best),
        },
        "manual_gold": {
            "yolo_ep_le44": {"sd_f1_at_protocol": _sd_of(mg_a, "f1_at_protocol"),
                             "sd_max_f1": _sd_of(mg_a, "max_f1"),
                             "mean_max_f1": (statistics.fmean([r["max_f1"] for r in mg_a])
                                             if all(mg_a) else None)},
            "yolo_best_pt": {"sd_f1_at_protocol": _sd_of(mg_best, "f1_at_protocol"),
                             "sd_max_f1": _sd_of(mg_best, "max_f1"),
                             "mean_max_f1": (statistics.fmean([r["max_f1"] for r in mg_best])
                                             if all(mg_best) else None)},
            "rampnet": {"sd_f1_at_protocol": _sd_of(mg_b, "f1_at_protocol"),
                        "sd_max_f1": _sd_of(mg_b, "max_f1"),
                        "mean_max_f1": (statistics.fmean([r["max_f1"] for r in mg_b])
                                        if all(mg_b) else None)},
            "run_a_epoch_1_reference": run_a_epoch1_manual_gold(
                grid, os.path.join(data, "rampnet_s1", "manual_gold.json")),
            "run_b_reopen_condition_max_f1_sd": 0.002,   # docs/stage2_cosine_rung_135.md
        },
    }

    return {
        "dev_split": dev_split, "pool": list(POOLED_SPLITS), "max_epoch": MAX_EPOCH,
        "selection_column": MAP5095, "grid_rampnet": grid,
        "published_reference": ref,
        "campaign_a": camp_a, "campaign_b": camp_b,
        "statistics": stats, "secondary": secondary,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def rnd(o, n=5):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rnd(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v, n) for v in o]
    return o


def _f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def _rel(path):
    try:
        return os.path.relpath(path, REPO)
    except ValueError:        # Windows: a different drive from the repo
        return path


def render_markdown(result):
    st, ref = result["statistics"], result["published_reference"]
    L = []
    L.append("### Primary read (pre-registered)\n")
    L.append("| replicate | leg | epoch | mAP50-95 (val) | thr on sao_paulo | pooled P | pooled R | **pooled F1** |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, r in result["campaign_a"].items():
        rd = r["read"]
        L.append(f"| A {k} | `{r['leg']}` | {r['epoch']} | {_f(r['map5095_at_epoch'])} | "
                 f"{_f(rd['selected_threshold'], 2) if rd else '-'}{'*' if rd and rd['at_floor'] else ''} | "
                 f"{_f(rd['pooled']['p'], 3) if rd else '-'} | {_f(rd['pooled']['r'], 3) if rd else '-'} | "
                 f"**{_f(rd['pooled']['f1']) if rd else '-'}** |")
    for k, r in result["campaign_b"].items():
        rd = r["read"]
        L.append(f"| B {k} | `{r['leg']}` | 1 | - | "
                 f"{_f(rd['selected_threshold'], 2) if rd else '-'}{'*' if rd and rd['at_floor'] else ''} | "
                 f"{_f(rd['pooled']['p'], 3) if rd else '-'} | {_f(rd['pooled']['r'], 3) if rd else '-'} | "
                 f"**{_f(rd['pooled']['f1']) if rd else '-'}** |")
    L.append(f"| seed-0 arm (n=1, published) | `{SEED0_YOLO_LEG}` | ~44 | - | {_f(ref['yolo_threshold'], 2)} | | | {_f(ref['yolo_f1'])} |")
    L.append(f"| RampNet (n=1, published) | `{RAMPNET}` | 1 | - | {_f(ref['rampnet_threshold'], 2)} | | | {_f(ref['rampnet_f1'])} |")
    L.append("")
    L.append("| statistic | value |")
    L.append("|---|---|")
    L.append(f"| mean F1, Campaign A / B | {_f(st['mean_a'])} / {_f(st['mean_b'])} |")
    L.append(f"| `s_A` | {_f(st['s_A'])} |")
    L.append(f"| `s_B` | {_f(st['s_B'])} |")
    L.append(f"| `s_gap = sqrt(s_A² + s_B²)` | **{_f(st['s_gap'])}** |")
    L.append(f"| published gap (n=1) | {_f(st['gap_published'])} |")
    L.append(f"| gap of replicate means (B − A) | {_f(st['gap_of_means'])} |")
    L.append(f"| published gap / `s_gap` | {_f(st['z_gap'], 2)} σ |")
    L.append(f"| **A1.1 band** | **{st['band_gap']}** |")
    L.append(f"| A1.2: `s_B` vs paired MDE {PAIRED_MDE_135} | {st['band_b']} |")
    L.append("")
    L.append("### Secondary reads (post-hoc, descriptive only)\n")
    sb = result["secondary"]["yolo_best_pt"]
    L.append("| YOLO at as-saved `best.pt` (≤60) | thr | pooled F1 |")
    L.append("|---|---|---|")
    for k, r in sb["replicates"].items():
        rd = r["read"]
        L.append(f"| `{r['leg']}` | {_f(rd['selected_threshold'], 2) if rd else '-'} | {_f(rd['pooled']['f1']) if rd else '-'} |")
    L.append(f"| mean / SD | | {_f(sb['mean'])} / {_f(sb['sd'])} |")
    L.append("")
    mg = result["secondary"]["manual_gold"]
    L.append("| manual_gold | leg | F1 at protocol thr | max F1 | at thr |")
    L.append("|---|---|---|---|---|")
    for group, rows in (("A ≤44", result["campaign_a"]), ("A best.pt", sb["replicates"]),
                        ("B", result["campaign_b"])):
        for k, r in rows.items():
            m = r["manual_gold"]
            if m:
                L.append(f"| {group} {k} | `{r['leg']}` | {_f(m['f1_at_protocol'])} (@{m['protocol_threshold']:.2f}) | "
                         f"{_f(m['max_f1'])} | {m['max_f1_threshold']:.2f} |")
    ra = mg["run_a_epoch_1_reference"]
    if ra:
        L.append(f"| Run A ep1 (seed 42, reference) | `{ra['label']}` | {_f(ra['f1_at_protocol'])} (@{ra['protocol_threshold']:.2f}) | "
                 f"{_f(ra['max_f1'])} | {ra['max_f1_threshold']:.2f} |")
    L.append("")
    L.append("| manual_gold SD (n=3) | SD F1 at protocol | SD max F1 | mean max F1 |")
    L.append("|---|---|---|---|")
    for k in ("yolo_ep_le44", "yolo_best_pt", "rampnet"):
        g = mg[k]
        L.append(f"| {k} | {_f(g['sd_f1_at_protocol'])} | {_f(g['sd_max_f1'])} | {_f(g['mean_max_f1'])} |")
    L.append("")
    L.append("\\* selected threshold is the cache floor: the true optimum may be lower and unmeasured, so that F1 is a lower bound.")
    return "\n".join(L)


def _render_text(result):
    st = result["statistics"]
    lines = [f"Dev split (selection only): {result['dev_split']}",
             f"Pool: {', '.join(result['pool'])}; Campaign A read at best {MAP5095} epoch <= {MAX_EPOCH}",
             ""]
    for camp in ("campaign_a", "campaign_b"):
        for k, r in result[camp].items():
            rd = r["read"]
            ep = f" ep{r['epoch']}" if "epoch" in r else ""
            if rd:
                lines.append(f"  {camp[-1].upper()} {k:<3} {r['leg']:<22}{ep:<6} thr {rd['selected_threshold']:.2f}"
                             f"{'*' if rd['at_floor'] else ' '}  F1 {rd['pooled']['f1']:.4f}")
            else:
                lines.append(f"  {camp[-1].upper()} {k:<3} {r['leg']:<22}{ep:<6} NOT SCORED")
    lines.append("")
    lines.append(f"  s_A {_f(st['s_A'])}   s_B {_f(st['s_B'])}   s_gap {_f(st['s_gap'])}"
                 f"   published gap {_f(st['gap_published'])} = {_f(st['z_gap'], 2)} sigma")
    lines.append(f"  A1.1 band: {st['band_gap']}    A1.2: {st['band_b']}")
    if st["s_A_only_band"]:
        lines.append(f"  (Campaign B incomplete: on s_A alone the band would be {st['s_A_only_band']} -- "
                     "an upper bound on significance, not the finding)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DATA, help="directory holding the scored inputs")
    ap.add_argument("--json", default=OUT_JSON, help="artifact path")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed artifact does not match a fresh build")
    ap.add_argument("--markdown", action="store_true",
                    help="print the doc tables instead of the summary")
    args = ap.parse_args()

    result = build(args.data)
    payload = rnd(result)
    print(render_markdown(result) if args.markdown else _render_text(result))

    if args.check:
        if not os.path.exists(args.json):
            print(f"\n--check: {args.json} does not exist", file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            if json.load(fh) != payload:
                print(f"\n--check: {args.json} is stale", file=sys.stderr)
                return 1
        print(f"\n--check: {_rel(args.json)} is current")
        return 0

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {_rel(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
