"""Does the richmond cascade setting transfer to other splits? (#35, the out-of-sample read)

``cascade_cost_35.py`` found the gated cascade VIABLE on richmond with the Vistas parity arm
(``mask2former-vistas-curb-cut-1024x1024``) as the challenger, at a setting chosen in sample
from 60. This script reads that setting, unchanged, on splits it was not chosen on.

**The fixed setting** (``cascade_cost_35.FIXED_SETTING``): T_hi 0.30, T_lo 0.05, r_gate 0.011
(R/2), and c_min = the challenger's median box score **on the split being scored** (the rule
is a rank, so it transfers across splits whose score distributions differ; the richmond value
0.641 is also read, as a labelled sensitivity row, not part of the rule).

**The verdict rule, stated in the plan (#35, 2026-09-27) before any transfer number existed.**
On a split the cascade *transfers* when all three hold for the fixed setting:

1. ``verdict_of`` applied to the fixed row alone reads VIABLE: attributable ΔR >= 0.020 after
   the wrong-pano null, F1 >= the baseline at 0.30, and F1 above threshold-only at T_lo 0.05;
2. the pano-bootstrap 95% interval of the attributable ΔR lies above 0 (lower bound > 0);
3. fewer FPs per attributable ramp than the matched-recall threshold pays per ramp, where the
   matched-recall threshold is the best-F1 single threshold (0.05-0.95 in 0.01 steps) whose
   recall reaches the cascade's, and its price is (its FP - baseline FP) / (its recall ramps -
   baseline recall ramps). If no single threshold reaches the cascade's recall, criterion 3
   holds (a threshold cannot buy that recall at any price) and that is recorded.

**The cascade transfers if it transfers on at least two of the three GSV splits** (bend,
paterson, gainesville). annapolis (a second Mapillary rig) is scored the same way and reported
beside them, but is not part of the count. richmond is shown as the in-sample reference.

**The attributable-ΔR interval.** 2,000 pano resamples (``numpy.default_rng(0)``), setting held
fixed. In each resample the attributable ΔR is the cascade's recall minus the mean recall of
the wrong-pano copies on the same resampled panos (every cyclic shift k = 1..n-1, the same
null ``cascade_cost_35`` uses), i.e. real ΔR minus null-mean ΔR, recomputed per resample from
per-pano integer counts. It is conditional on the setting, which here was fixed before these
splits were scored, so there is no selection to repeat.

**Calibration of the fixed row.** Each wrong-pano copy of the challenger is run through
criterion 1 exactly as ``cascade_cost_35.calibrate`` does for the grid (its null = the true
alignment plus the other shifts). The fraction reading VIABLE is how often the fixed row would
read VIABLE with no challenger aligned to its panos.

Inputs, all committed: ``benchmark/<split>/`` (GT), ``analysis_out/op_cache/<split>.json``
(RampNet floor peaks; written before the seam fix f4c71c8), and
``benchmark/model_detections/mask2former-vistas-curb-cut-1024x1024__<split>.json``. CPU only,
no network.

    python scripts/analysis/cascade_transfer_35.py            # writes transfer.json, prints tables
    python scripts/analysis/cascade_transfer_35.py --check    # regenerate + byte-compare
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cascade_cost_35 as cc  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402
from complementarity import load_floor_peaks  # noqa: E402

CHALLENGER = cc.PRIMARY[1]
GSV_SPLITS = ("bend", "paterson", "gainesville")
OTHER_SPLITS = ("annapolis",)
REFERENCE = "richmond"
SPLITS = GSV_SPLITS + OTHER_SPLITS + (REFERENCE,)
#: The rule's count: transfers on at least this many of GSV_SPLITS.
MIN_GSV_PASSES = 2
#: The richmond median box score of the parity arm (cascade_cost_35 table (a)): read as a
#: sensitivity row on the other splits, not part of the rule.
RICHMOND_C_MIN = 0.640738
BOOTSTRAP = 2000
SEED = 0
OUT_PATH = os.path.join(cc.DEFAULT_OUT, "transfer", "transfer.json")


def _ramps(r, base):
    """Recall ramps gained over baseline (integer, recall-confirmed panos only)."""
    return round((r["R"] - base["R"]) * base["n_gt"])


def read_setting(split, c_min, gts=None, peaks=None, bootstrap=BOOTSTRAP, seed=SEED):
    """One split x the fixed (T_lo, r_gate) at ``c_min``: the row, its null, its controls,
    the attributable-ΔR interval and the three criteria. Returns a JSON-able dict."""
    gts = cc.load_gts(split) if gts is None else gts
    peaks = load_floor_peaks(split) if peaks is None else peaks
    cands, header = cc.load_challenger(split, CHALLENGER)
    if cands is None:
        return None
    t_hi, (t_lo, r_gate, _) = cc.T_HI, cc.FIXED_SETTING
    pids = list(gts)
    n = len(pids)
    scorer = cc._Scorer(gts, peaks, radius_sq_for(cc.RADIUS))
    kept = {pid: tuple(i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t_hi)
            for pid in pids}
    n_kept = {pid: len(v) for pid, v in kept.items()}
    base_scores = [scorer.pano(pid, kept[pid]) for pid in pids]
    base = cc._fast_report(base_scores)

    def thr(t):
        return cc._fast_report([scorer.pano(pid, tuple(
            i for i, p in enumerate(peaks.get(pid, [])) if p[2] >= t)) for pid in pids])

    thr_tlo = thr(t_lo)
    sweep = {t: thr(t) for t in cc.THR_SWEEP}
    t_best = max(sweep, key=lambda t: (sweep[t]["F1"], -t))

    rsq = radius_sq_for(r_gate)
    filt = {pid: cc.filter_cands(cs, c_min) for pid, cs in cands.items()}
    rep, n_prom, sc = cc._eval_cascade(scorer, filt, t_hi, t_lo, rsq, n_kept)
    ks = list(range(1, n))
    null_reps, null_tpr = [], np.zeros(n, dtype=np.int64)
    for k in ks:
        nrep, _, nsc = cc._eval_cascade(scorer, cc.shifted(filt, pids, k), t_hi, t_lo, rsq,
                                        n_kept)
        null_reps.append(nrep)
        null_tpr += cc._counts(nsc)[:, 2]
    null_dR = [r["R"] - base["R"] for r in null_reps]
    att = (rep["R"] - base["R"]) - sum(null_dR) / len(null_dR)
    att_ramps = att * base["n_gt"]
    promoted_fp = rep["fp"] - base["fp"]
    row = {"t_lo": t_lo, "r_gate": r_gate, "c_min": c_min, **rep, "n_promoted": n_prom,
           "promoted_tp": rep["tp"] - base["tp"], "promoted_fp": promoted_fp,
           "dR": rep["R"] - base["R"], "dF1": rep["F1"] - base["F1"],
           "dR_ramps": _ramps(rep, base),
           "null_dR_mean": sum(null_dR) / len(null_dR), "null_dR_max": max(null_dR),
           "null_dFP_mean": sum(r["fp"] - base["fp"] for r in null_reps) / len(null_reps),
           "attributable_dR": att, "attributable_ramps": att_ramps,
           "fp_per_attributable_ramp": promoted_fp / att_ramps if att_ramps > 0 else None,
           "threshold_only_F1": thr_tlo["F1"]}

    # criterion 1: the pre-stated rule on this one row
    verdict = cc.verdict_of(base, {cc._tkey(t_lo): thr_tlo}, [row])

    # criterion 2: attributable dR interval, setting fixed, pano bootstrap
    base_c, casc_c = cc._counts(base_scores), cc._counts(sc)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(bootstrap, n))
    W = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.int64)
    ngt = (W @ base_c[:, 3]).astype(np.float64)
    r_casc = np.divide((W @ casc_c[:, 2]).astype(np.float64), ngt, out=np.zeros(bootstrap),
                       where=ngt > 0)
    r_null = np.divide((W @ null_tpr).astype(np.float64), len(ks) * ngt,
                       out=np.zeros(bootstrap), where=ngt > 0)
    att_ci = cc._ci95(r_casc - r_null)
    # the same resamples, raw dR and dF1 against baseline (paired)
    _, rb, fb = cc._prf(W @ base_c)
    _, rc, fc = cc._prf(W @ casc_c)
    boot = {"resamples": bootstrap, "seed": seed, "rng": "numpy.default_rng", "unit": "pano",
            "attr_dR_ci95": att_ci, "dR_ci95": cc._ci95(rc - rb), "dF1_ci95": cc._ci95(fc - fb),
            "conditioning": "setting fixed before these splits were scored"}

    # criterion 3: FP per attributable ramp vs the matched-recall threshold's FP per ramp
    ok = [t for t in sweep if sweep[t]["R"] >= rep["R"]]
    matched = None
    if ok:
        t = max(ok, key=lambda t: (sweep[t]["F1"], t))
        m = sweep[t]
        g = _ramps(m, base)
        matched = {"t": t, **m, "extra_fp": m["fp"] - base["fp"], "extra_ramps": g,
                   "fp_per_ramp": (m["fp"] - base["fp"]) / g if g > 0 else None}
    fpr = row["fp_per_attributable_ramp"]
    if att_ramps <= 0:
        c3, c3_note = False, "no attributable gain"
    elif matched is None:
        c3, c3_note = True, "no single threshold reaches the cascade's recall"
    elif matched["fp_per_ramp"] is None:
        c3, c3_note = False, "the matched-recall threshold gains no ramps over baseline"
    else:
        c3, c3_note = fpr < matched["fp_per_ramp"], None

    # calibration of the fixed row: each wrong-pano copy through criterion 1
    calib = []
    thr_map = {cc._tkey(t_lo): thr_tlo}
    for i, nrep in enumerate(null_reps):
        pool = [row["dR"]] + [d for j, d in enumerate(null_dR) if j != i]
        wrong = {"t_lo": t_lo, "F1": nrep["F1"],
                 "attributable_dR": null_dR[i] - sum(pool) / len(pool)}
        calib.append(cc.verdict_of(base, thr_map, [wrong]))

    criteria = {"viable": verdict == "VIABLE", "attr_dR_ci_above_0": att_ci[0] > 0,
                "fewer_fp_per_ramp_than_matched_threshold": c3}
    return {
        "split": split, "challenger": CHALLENGER, "t_hi": t_hi, "n_panos": n,
        "n_gt_recall": base["n_gt"], "challenger_boxes": sum(len(v) for v in cands.values()),
        "challenger_signature": header.get("signature"),
        "baseline": base, "threshold_only_at_t_lo": thr_tlo,
        "threshold_only_best": {"t": t_best, **sweep[t_best]},
        "fixed": row, "verdict": verdict, "bootstrap": boot,
        "threshold_only_at_matched_recall": matched, "criterion_3_note": c3_note,
        "null": {"mode": "shift", "shifts": len(ks)},
        "calibration": {"challengers": len(calib),
                        "verdicts": {v: calib.count(v) for v in ("VIABLE", "PARTIAL",
                                                                 "NOT VIABLE")},
                        "false_viable_rate": calib.count("VIABLE") / len(calib)},
        "criteria": criteria, "transfers": all(criteria.values()),
    }


def build(bootstrap=BOOTSTRAP, seed=SEED):
    """The whole transfer read: primary (median c_min per split), the richmond-c_min
    sensitivity, the in-sample grid verdict per split (from the committed per-pair files),
    and the overall verdict against the pre-stated count."""
    primary, sensitivity, grid = {}, {}, {}
    for split in SPLITS:
        gts, peaks = cc.load_gts(split), load_floor_peaks(split)
        cands, _ = cc.load_challenger(split, CHALLENGER)
        if cands is None:
            primary[split] = None
            continue
        q50 = cc.c_min_grid(cands)[2]
        primary[split] = read_setting(split, q50, gts, peaks, bootstrap, seed)
        primary[split]["c_min_quartiles"] = cc.c_min_grid(cands)[1:]
        if split != REFERENCE:
            sensitivity[split] = read_setting(split, RICHMOND_C_MIN, gts, peaks, bootstrap, seed)
        path = os.path.join(cc.DEFAULT_OUT, cc.out_name(split, CHALLENGER, cc.T_HI))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
            bv = p["best_viable"]
            grid[split] = {
                "file": os.path.relpath(path, cc.REPO).replace(os.sep, "/"),
                "null_shifts": p["null"]["shifts"], "verdict": p["verdict"],
                "n_viable_rows": len(p["viable_rows"]), "n_settings": len(p["grid"]),
                "best_viable": None if bv is None else {
                    k: bv[k] for k in ("t_lo", "r_gate", "c_min", "F1", "R", "dR",
                                       "attributable_dR", "promoted_fp",
                                       "fp_per_attributable_ramp")},
                "best_by_f1": {k: p["best_by_f1"][k] for k in
                               ("t_lo", "r_gate", "c_min", "F1", "attributable_dR")},
                "threshold_only_best": {k: p["threshold_only_best"][k] for k in ("t", "F1")},
                "calibration_false_viable_rate": p["calibration"]["false_viable_rate"],
                "selection_aware_viable_frac": p["bootstrap_selection_aware"]["viable_frac"],
                "naive_union_F1": p["naive_union"]["aggregate"]["F1"],
            }
    passes = [s for s in GSV_SPLITS if primary.get(s) and primary[s]["transfers"]]
    missing = [s for s in GSV_SPLITS if not primary.get(s)]
    return {
        "challenger": CHALLENGER,
        "fixed_setting": {"t_hi": cc.T_HI, "t_lo": cc.FIXED_SETTING[0],
                          "r_gate": cc.FIXED_SETTING[1],
                          "c_min": "median box score of the challenger on that split"},
        "rule": {"per_split": ["verdict_of on the fixed row reads VIABLE",
                               "attributable dR bootstrap 95% CI lower bound > 0",
                               "FP per attributable ramp < the matched-recall threshold's "
                               "FP per ramp"],
                 "overall": f"transfers on >= {MIN_GSV_PASSES} of {list(GSV_SPLITS)}",
                 "stated": "issue #35 plan comment, 2026-09-27, before scoring"},
        "gsv_splits": list(GSV_SPLITS), "other_splits": list(OTHER_SPLITS),
        "reference_split": REFERENCE,
        "gsv_passes": passes, "gsv_missing": missing,
        "overall": (None if missing else
                    ("TRANSFERS" if len(passes) >= MIN_GSV_PASSES else "DOES NOT TRANSFER")),
        "primary": primary,
        "sensitivity_richmond_c_min": {"c_min": RICHMOND_C_MIN, "splits": sensitivity},
        "in_sample_grid": grid,
        "op_cache_commit_note": cc.OP_CACHE_NOTE,
    }


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def print_tables(out):
    print(f"overall: {out['overall']}  (GSV passes: {out['gsv_passes']})\n")
    print("| split | c_min (q50) | baseline TP/FP/FN, F1 | cascade TP/FP/FN | P | R | F1 "
          "| dR (ramps) | null dR mean / max | attributable dR [95% CI] | promoted FP "
          "| FP / attr ramp | thr-only F1 @0.05 | matched-R threshold (t: +FP / +ramps = FP/ramp, F1) "
          "| best single threshold (t, F1) | verdict | CI > 0 | cheaper | transfers |")
    print("|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|---|---|---|---|")
    for split in SPLITS:
        p = out["primary"].get(split)
        if p is None:
            print(f"| {split} | no published detections |")
            continue
        b, r, m, bo = p["baseline"], p["fixed"], p["threshold_only_at_matched_recall"], p["bootstrap"]
        ci = bo["attr_dR_ci95"]
        ms = ("none reaches" if m is None else
              f"{m['t']:g}: +{m['extra_fp']} / +{m['extra_ramps']} = "
              f"{_f(m['fp_per_ramp'], '.2f')}, {m['F1']:.4f}")
        c = p["criteria"]
        yn = {True: "yes", False: "no"}
        print(f"| {split} | {r['c_min']:.3f} | {b['tp']}/{b['fp']}/{b['fn']}, {b['F1']:.4f} | "
              f"{r['tp']}/{r['fp']}/{r['fn']} | {r['P']:.4f} | {r['R']:.4f} | {r['F1']:.4f} | "
              f"{r['dR']:+.4f} ({r['dR_ramps']:+d}) | {r['null_dR_mean']:+.4f} / "
              f"{r['null_dR_max']:+.4f} | {r['attributable_dR']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
              f"{r['promoted_fp']} | {_f(r['fp_per_attributable_ramp'], '.2f')} | "
              f"{r['threshold_only_F1']:.4f} | {ms} | "
              f"{p['threshold_only_best']['t']:g}, {p['threshold_only_best']['F1']:.4f} | "
              f"{p['verdict']} | {yn[c['attr_dR_ci_above_0']]} | "
              f"{yn[c['fewer_fp_per_ramp_than_matched_threshold']]} | {yn[p['transfers']]} |")
    print("\nsensitivity: richmond's absolute c_min", out["sensitivity_richmond_c_min"]["c_min"])
    for split, p in out["sensitivity_richmond_c_min"]["splits"].items():
        if p is None:
            continue
        r = p["fixed"]
        print(f"  {split}: {r['tp']}/{r['fp']}/{r['fn']} F1 {r['F1']:.4f} attr "
              f"{r['attributable_dR']:+.4f} CI {p['bootstrap']['attr_dR_ci95']} FP/attr "
              f"{_f(r['fp_per_attributable_ramp'], '.2f')} verdict {p['verdict']} "
              f"transfers {p['transfers']}")
    print("\nin-sample grid (secondary):")
    for split, g in out["in_sample_grid"].items():
        print(f"  {split}: {g['verdict']} ({g['n_viable_rows']}/{g['n_settings']} viable), "
              f"best viable {g['best_viable']}, calib {g['calibration_false_viable_rate']}")
    print("\ncalibration of the fixed row (wrong-pano copies reading VIABLE):")
    for split in SPLITS:
        p = out["primary"].get(split)
        if p:
            print(f"  {split}: {p['calibration']['verdicts']} "
                  f"rate {p['calibration']['false_viable_rate']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--check", action="store_true",
                    help="Regenerate and byte-compare against --out; write nothing.")
    args = ap.parse_args()
    out = build()
    text = cc.dumps(out)
    if args.check:
        with open(args.out, encoding="utf-8", newline="") as f:
            same = f.read() == text
        print("same" if same else "DIFF", args.out)
        sys.exit(0 if same else 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print_tables(out)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
