"""Read the #82 paired fine-tune screen (Step 3) from the committed per-checkpoint caches.

Inputs: ``analysis_out/aug_transfer_82/finetune/<label>/<split>.json``, one
``operating_point_curve.py extract`` cache per checkpoint x split (written on makelab2 by
``scripts/analysis/aug82_score_ckpts.sh``). Labels: ``released`` (the Hub checkpoint) and
``<arm>_s<seed>`` for arm in control / res / photo / both and the screen's seeds.

What it reports, per split and never only pooled, at 0.30 and 0.55:

* every checkpoint's P / R / F1 (and AP);
* each augmented arm minus the SAME-SEED control, paired pano bootstrap;
* control minus released (what the extra steps alone do), per seed;
* the control-seed spread printed beside every contrast: control_s2 minus control_s1,
  with its own paired interval -- the noise floor a contrast has to clear;
* pooled reads for the transfer splits (laurens_mapillary, clovis, richmond), the
  in-domain splits (manual_gold, bend) and the US7 pool, stratified by split;
* the Laurens rig effect per checkpoint on the 47 corners both rigs saw
  (``analysis_out/laurens_paired_151.json`` pairs; GSV minus GoPro, pair bootstrap).

    python scripts/analysis/aug_finetune_82.py            # -> finetune_results.json / .md
    python scripts/analysis/aug_finetune_82.py --check    # exit 1 if the committed output is stale
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet.detection_eval import radius_sq_for, score_pano  # noqa: E402
from operating_point_curve import _score_at, pr_curve_and_ap, read_cache  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "aug_transfer_82")
FT_ROOT = os.path.join(OUT_DIR, "finetune")
RESULTS = os.path.join(OUT_DIR, "finetune_results.json")
RESULTS_MD = os.path.join(OUT_DIR, "finetune_results.md")
PAIRS = os.path.join(REPO, "analysis_out", "laurens_paired_151.json")

SPLITS = ("annapolis", "bend", "budapest_district5", "clovis", "gainesville", "laurens_gsv",
          "laurens_mapillary", "manual_gold", "morgantown", "paterson", "richmond",
          "sao_paulo")
ARMS = ("control", "res", "photo", "both")
SEEDS = (1, 2)
THRESHOLDS = (0.30, 0.55)
POOLS = {"transfer (laurens_mapillary+clovis+richmond)": ("laurens_mapillary", "clovis",
                                                          "richmond"),
         "in-domain (manual_gold+bend)": ("manual_gold", "bend"),
         "US7 (miss_decomposition.US_SPLITS)": ("richmond", "bend", "clovis", "morgantown",
                                                "annapolis", "paterson", "gainesville",
                                                "laurens_mapillary")}
N_REPS = 2000
SEED = 82
ND = 4


def rnd(v, nd=ND):
    if v is None:
        return None
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)


def labels_present(root=FT_ROOT):
    out = []
    if os.path.isdir(os.path.join(root, "released")):
        out.append("released")
    for a in ARMS:
        for s in SEEDS:
            if os.path.isdir(os.path.join(root, f"{a}_s{s}")):
                out.append(f"{a}_s{s}")
    return out


def load(root=FT_ROOT):
    data = {}
    for lab in labels_present(root):
        for c in SPLITS:
            p = os.path.join(root, lab, f"{c}.json")
            if os.path.exists(p):
                panos, _ = read_cache(p)
                data[(lab, c)] = sorted(panos, key=lambda d: d["pano"])
    return data


def _scored(city, panos, rsq):
    records = {p["pano"]: {"detections": [list(t) for t in p["preds"]]} for p in panos}
    return bp.score_model(REPO, city, "rampnet", records, {p["pano"]: p["gt"] for p in panos},
                          rsq)


def _metrics(panos, rsq):
    out = {}
    for t in THRESHOLDS:
        s = _score_at(panos, t, rsq)
        out[f"{t:.2f}"] = {"P": rnd(s.precision), "R": rnd(s.recall), "F1": rnd(s.f1),
                           "tp": s.tp, "fp": s.fp, "fn": s.fn}
    out["AP"] = rnd(pr_curve_and_ap(panos, rsq).ap)
    return out


def _contrast(sa, sb, sizes, t):
    rng = np.random.default_rng(SEED)
    r = bp.observed_and_se(sa, sizes, t, rng, N_REPS, paired=sb)
    return {k: {"observed": rnd(v["observed"]), "ci_lo": rnd(v["ci_lo"]),
                "ci_hi": rnd(v["ci_hi"])} for k, v in r.items() if k != "max_f1"}


def contrasts_for(scored, members, pairs_of_labels):
    """{name: {thr: contrast}} for (name, a, b) over ``members`` (stacked if >1)."""
    out = {}
    for name, a, b in pairs_of_labels:
        if not all((a, c) in scored and (b, c) in scored for c in members):
            continue
        sa = bp.stack([scored[(a, c)] for c in members]) if len(members) > 1 else scored[(a, members[0])]
        sb = bp.stack([scored[(b, c)] for c in members]) if len(members) > 1 else scored[(b, members[0])]
        sizes = [len(scored[(a, c)].pids) for c in members]
        out[name] = {f"{t:.2f}": _contrast(sa, sb, sizes, t) for t in THRESHOLDS}
    return out


def contrast_list(labels):
    pl = []
    if "control_s1" in labels and "control_s2" in labels:
        pl.append(("spread: control_s2 - control_s1", "control_s2", "control_s1"))
    for s in SEEDS:
        if f"control_s{s}" in labels and "released" in labels:
            pl.append((f"control_s{s} - released", f"control_s{s}", "released"))
    for a in ARMS[1:]:
        for s in SEEDS:
            if f"{a}_s{s}" in labels and f"control_s{s}" in labels:
                pl.append((f"{a}_s{s} - control_s{s}", f"{a}_s{s}", f"control_s{s}"))
    return pl


def laurens_rig_effect(data, rsq, labels):
    """GSV minus GoPro on the paired corners, per checkpoint, pair bootstrap."""
    with open(PAIRS, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    out = {}
    rng0 = np.random.default_rng(SEED)
    w = rng0.multinomial(len(pairs), np.full(len(pairs), 1.0 / len(pairs)), size=N_REPS)
    for lab in labels:
        if (lab, "laurens_gsv") not in data or (lab, "laurens_mapillary") not in data:
            continue
        g = {p["pano"]: p for p in data[(lab, "laurens_gsv")]}
        m = {p["pano"]: p for p in data[(lab, "laurens_mapillary")]}
        ent = {}
        for t in THRESHOLDS:
            cnt = np.zeros((len(pairs), 2, 4))     # pair x arm x (tp, fp, tp_r, n_gt)
            for i, pr in enumerate(pairs):
                for j, d in enumerate((g[pr["gsv"]], m[pr["mly"]])):
                    s = score_pano([q for q in d["preds"] if q[2] >= t], d["gt"],
                                   radius_sq=rsq)
                    n_gt = len(d["gt"].gt_points) if d["gt"].fn_confirmed else 0
                    cnt[i, j] = (s.tp, s.fp, s.tp if d["gt"].fn_confirmed else 0, n_gt)

            def prf(weights):
                c = np.tensordot(weights, cnt, axes=(1, 0))   # (B, 2, 4)
                tp, fp, tpr, n = (c[..., k] for k in range(4))
                p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
                r = np.where(n > 0, tpr / np.maximum(n, 1e-12), 0.0)
                f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
                return p, r, f
            obs = prf(np.ones((1, len(pairs))))
            bs = prf(w.astype(np.float64))
            row = {}
            for k, name in enumerate(("precision", "recall", "f1")):
                d_obs = obs[k][0, 0] - obs[k][0, 1]
                d_bs = bs[k][:, 0] - bs[k][:, 1]
                row[name] = {"gsv": rnd(obs[k][0, 0]), "gopro": rnd(obs[k][0, 1]),
                             "gsv_minus_gopro": rnd(d_obs),
                             "ci_lo": rnd(np.percentile(d_bs, 2.5)),
                             "ci_hi": rnd(np.percentile(d_bs, 97.5))}
            ent[f"{t:.2f}"] = row
        out[lab] = ent
    return {"n_pairs": len(pairs), "per_checkpoint": out}


def build(root=FT_ROOT):
    rsq = radius_sq_for()
    data = load(root)
    labels = labels_present(root)
    scored = {k: _scored(k[1], v, rsq) for k, v in data.items()}
    pl = contrast_list(labels)
    rep = {"protocol": {"thresholds": list(THRESHOLDS), "n_reps": N_REPS, "seed": SEED,
                        "bootstrap": "pano-level paired cluster bootstrap, stratified by "
                                     "split (benchmark_power_135.observed_and_se)",
                        "labels": labels},
           "per_split": {}, "pooled": {}}
    for c in SPLITS:
        if not any((lab, c) in data for lab in labels):
            continue
        rep["per_split"][c] = {
            "n_panos": len(next(data[(lab, c)] for lab in labels if (lab, c) in data)),
            "metrics": {lab: _metrics(data[(lab, c)], rsq) for lab in labels if (lab, c) in data},
            "contrasts": contrasts_for(scored, (c,), pl)}
    for name, members in POOLS.items():
        rep["pooled"][name] = {"members": list(members),
                               "contrasts": contrasts_for(scored, members, pl)}
    rep["laurens_paired"] = laurens_rig_effect(data, rsq, labels)
    return rep


def _fmt(d):
    if d is None:
        return "-"
    if d["ci_lo"] is None:
        return f"{d['observed']:+.4f}"
    return f"{d['observed']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"


def markdown(rep):
    L = ["# Issue #82 Step 3: paired fine-tune screen", "",
         "Generated by `scripts/analysis/aug_finetune_82.py`; do not edit by hand.", ""]
    for thr in ("0.30", "0.55"):
        L += [f"## Metrics at {thr} (P / R / F1)", "",
              "| split | " + " | ".join(rep["protocol"]["labels"]) + " |",
              "|---|" + "---|" * len(rep["protocol"]["labels"])]
        for c, ent in rep["per_split"].items():
            cells = []
            for lab in rep["protocol"]["labels"]:
                m = ent["metrics"].get(lab)
                cells.append("-" if m is None else
                             f"{m[thr]['P']:.3f} / {m[thr]['R']:.3f} / {m[thr]['F1']:.3f}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L += ["", f"## Contrasts at {thr}: dR, dF1 (paired pano bootstrap, 95% interval); "
              "the control-seed spread is the first row of every block", "",
              "| split / pool | contrast | dR | dP | dF1 |", "|---|---|---|---|---|"]
        blocks = list(rep["per_split"].items()) + [(f"**{k}**", v)
                                                   for k, v in rep["pooled"].items()]
        for c, ent in blocks:
            for name, cs in ent["contrasts"].items():
                d = cs[thr]
                L.append(f"| {c} | {name} | {_fmt(d['recall'])} | {_fmt(d['precision'])} | "
                         f"{_fmt(d['f1'])} |")
        L.append("")
    lp = rep["laurens_paired"]
    L += [f"## Laurens rig effect on the {lp['n_pairs']} paired corners (GSV minus GoPro)", "",
          "| checkpoint | thr | R gsv | R gopro | dR [95%] | dF1 [95%] |",
          "|---|---|---:|---:|---|---|"]
    for lab, ent in lp["per_checkpoint"].items():
        for thr, row in ent.items():
            r, f = row["recall"], row["f1"]
            L.append(f"| {lab} | {thr} | {r['gsv']:.3f} | {r['gopro']:.3f} | "
                     f"{r['gsv_minus_gopro']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                     f"{f['gsv_minus_gopro']:+.3f} [{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}] |")
    return "\n".join(L) + "\n"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=FT_ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rep = json.loads(json.dumps(build(args.root)))
    md = markdown(rep)
    if args.check:
        with open(RESULTS, encoding="utf-8") as f:
            ok = json.load(f) == rep
        with open(RESULTS_MD, encoding="utf-8", newline="") as f:
            ok &= f.read() == md
        print("finetune results up to date" if ok else "finetune results STALE")
        return 0 if ok else 1
    write_json(RESULTS, rep)
    with open(RESULTS_MD, "w", encoding="utf-8", newline="") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
