"""Read the #263 control arm (Track 3.4) from the committed per-checkpoint caches.

The control arm fine-tunes the RELEASED RampNet weights on the GSV training set only, 2,000
optimizer steps at global batch 16, three seeds (``stage_two/run_finetune_control.slurm``).
Every later #263 arm (Lyon mixes, masked loss) is read against these three checkpoints, so
this script answers two questions before any of that: what do the extra GSV steps alone do
to the released model, and how far apart do three seeds of the same recipe land.

Inputs: ``analysis_out/control_arm_263/caches/<label>/<split>.json``, one
``operating_point_curve.py extract`` cache per checkpoint x split, written on klone by
``scripts/analysis/control263_score.slurm`` (which calls the #82 scorer,
``aug82_score_ckpts.sh``). Labels: ``released`` (the Hub checkpoint, re-scored in the same
pass) and ``control_s1`` .. ``control_s3``.

The statistics are the #82 screen's, imported from ``aug_finetune_82.py`` rather than
re-implemented: P / R / F1 at 0.30 and 0.55, AP, max-F1, and the pano-level paired cluster
bootstrap stratified by split (``benchmark_power_135.observed_and_se``, 2,000 draws, rng seed
82). What is new here:

* three seeds instead of two, so the seed-mean contrast averages three controls, and the
  seed spread is reported as all three pairwise paired contrasts plus the per-seed range;
* the 14 reviewed splits (bayonne and juneau were reviewed after #82);
* the pools #263 reads: the four open-rig splits the plan names, the same without
  budapest_district5 (which ``miss_decomposition.HELD_OUT`` and #258 never pool), the GSV
  tier, the US pooled basis (``miss_decomposition.US_SPLITS``) and manual_gold;
* every split and pool also on the **unbiased subset** (verdicts ``group != "top"``, the
  convention of ``score_validation.py`` and ``open_rig_misses_258.py``), because the five
  hand-picked ``top`` panos per split are excluded from unbiased estimates;
* a determinism check: the released re-score against the committed caches the same call
  wrote earlier (``analysis_out/op_cache`` and the #82 ``finetune/released``);
* the #82 screen's two controls as a replication read. They are the same recipe (released
  init, 2,000 steps, global batch 16, seeds 1 and 2) trained on 2026-10-02 from the earlier
  copy of the Stage 2 dataset at an earlier commit, so ``control_s<k> - aug82_control_s<k>``
  is a same-seed rerun, and the five runs together give a wider seed-spread estimate.
  Their 12-split caches are #82's committed ones; bayonne and juneau were scored for this
  issue into ``caches_aug82_controls/``.

    python scripts/analysis/control_arm_263.py            # -> results.json / results.md
    python scripts/analysis/control_arm_263.py --check    # exit 1 if the committed output is stale
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rampnet.detection_eval import radius_sq_for  # noqa: E402
from operating_point_curve import read_cache  # noqa: E402
import aug_finetune_82 as af  # noqa: E402
import benchmark_power_135 as bp  # noqa: E402
from miss_decomposition import US_SPLITS  # noqa: E402

OUT_DIR = os.path.join(REPO, "analysis_out", "control_arm_263")
CACHE_ROOT = os.path.join(OUT_DIR, "caches")
RESULTS = os.path.join(OUT_DIR, "results.json")
RESULTS_MD = os.path.join(OUT_DIR, "results.md")

SPLITS = ("annapolis", "bayonne", "bend", "budapest_district5", "clovis", "gainesville",
          "juneau", "laurens_gsv", "laurens_mapillary", "manual_gold", "morgantown",
          "paterson", "richmond", "sao_paulo")
SEEDS = (1, 2, 3)
CONTROLS = tuple(f"control_s{s}" for s in SEEDS)
AUG82_CONTROLS = ("aug82_control_s1", "aug82_control_s2")
LABELS = ("released",) + CONTROLS + AUG82_CONTROLS
FIVE_RUNS = CONTROLS + AUG82_CONTROLS
#: Where an #82 control's cache lives: #82's committed directory, else this issue's top-up.
AUG82_DIRS = (os.path.join(REPO, "analysis_out", "aug_transfer_82", "finetune"),
              os.path.join(OUT_DIR, "caches_aug82_controls"))
THRESHOLDS = af.THRESHOLDS
SUBSETS = ("all", "unbiased")
POOLS = {
    "open-rig 4 (laurens_mapillary+bayonne+juneau+budapest_district5)":
        ("laurens_mapillary", "bayonne", "juneau", "budapest_district5"),
    "open-rig 3 (budapest held out, as in #258)": ("laurens_mapillary", "bayonne", "juneau"),
    "GSV tier (bend+paterson+gainesville)": ("bend", "paterson", "gainesville"),
    "US pooled basis (miss_decomposition.US_SPLITS)": tuple(US_SPLITS),
    "manual_gold": ("manual_gold",),
}
#: Committed caches the released re-score is compared against, pano for pano.
PRIOR_RELEASED = {"op_cache": os.path.join(REPO, "analysis_out", "op_cache"),
                  "aug82": os.path.join(REPO, "analysis_out", "aug_transfer_82", "finetune",
                                        "released")}
rnd = af.rnd


def top_panos(city):
    """The hand-picked ``top`` stratum of a split (empty for manual_gold, which has none)."""
    p = os.path.join(REPO, "benchmark", city, "verdicts.json")
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8") as f:
        panos = json.load(f)["panos"]
    return {pid for pid, e in panos.items() if e.get("group") == "top"}


def cache_path(root, lab, c):
    if lab in AUG82_CONTROLS:
        name = lab[len("aug82_"):]
        for d in AUG82_DIRS:
            p = os.path.join(d, name, f"{c}.json")
            if os.path.exists(p):
                return p
        return None
    p = os.path.join(root, lab, f"{c}.json")
    return p if os.path.exists(p) else None


def load(root=CACHE_ROOT):
    data, meta = {}, {}
    for lab in LABELS:
        for c in SPLITS:
            p = cache_path(root, lab, c)
            if p:
                panos, m = read_cache(p)
                data[(lab, c)] = sorted(panos, key=lambda d: d["pano"])
                meta[(lab, c)] = m
    return data, meta


def subset(data, which):
    if which == "all":
        return data
    tops = {c: top_panos(c) for c in SPLITS}
    return {k: [p for p in v if p["pano"] not in tops[k[1]]] for k, v in data.items()}


def contrast_list():
    pl = [(f"{c} - released", c, "released") for c in CONTROLS]
    pl += [(f"spread: control_s{b} - control_s{a}", f"control_s{b}", f"control_s{a}")
           for a, b in ((1, 2), (1, 3), (2, 3))]
    pl += [(f"rerun: control_s{s} - aug82_control_s{s}", f"control_s{s}", f"aug82_control_s{s}")
           for s in (1, 2)]
    return pl


SEED_MEAN = [("control (mean of s1-s3) - released", list(CONTROLS), ["released"]),
             ("control (mean of 5 runs, with #82's two) - released", list(FIVE_RUNS),
              ["released"])]


def spread(scored, members, runs=CONTROLS):
    """Per-run P / R / F1 / max-F1 on the observed sample, and their range and SD."""
    out = {}
    for t in THRESHOLDS:
        per = {}
        for lab in runs:
            if not all((lab, c) in scored for c in members):
                return {}
            s = (bp.stack([scored[(lab, c)] for c in members]) if len(members) > 1
                 else scored[(lab, members[0])])
            p, r, f, mf = (float(x[0]) for x in bp.metrics(s, np.ones((1, len(s.pids))), t))
            per[lab] = {"precision": p, "recall": r, "f1": f, "max_f1": mf}
        ent = {}
        for k in ("precision", "recall", "f1", "max_f1"):
            v = np.array([per[lab][k] for lab in runs])
            ent[k] = {"per_seed": [rnd(x) for x in v], "mean": rnd(v.mean()),
                      "sd": rnd(v.std(ddof=1)), "min": rnd(v.min()), "max": rnd(v.max()),
                      "range": rnd(v.max() - v.min())}
        out[f"{t:.2f}"] = ent
    return out


def point(scored, members, lab="released"):
    """One checkpoint's pooled P / R / F1 / max-F1 at each threshold (micro, as bp.metrics)."""
    if not all((lab, c) in scored for c in members):
        return {}
    s = (bp.stack([scored[(lab, c)] for c in members]) if len(members) > 1
         else scored[(lab, members[0])])
    out = {}
    for t in THRESHOLDS:
        p, r, f, mf = (float(x[0]) for x in bp.metrics(s, np.ones((1, len(s.pids))), t))
        out[f"{t:.2f}"] = {"precision": rnd(p), "recall": rnd(r), "f1": rnd(f), "max_f1": rnd(mf)}
    return out


def determinism(data, root=CACHE_ROOT):
    """Released re-score vs the committed caches: panos whose peak lists are identical."""
    out = {}
    for name, d in PRIOR_RELEASED.items():
        rows = {}
        for c in SPLITS:
            p = os.path.join(d, f"{c}.json")
            if ("released", c) not in data or not os.path.exists(p):
                continue
            prior = {q["pano"]: q["preds"] for q in read_cache(p)[0]}
            now = {q["pano"]: q["preds"] for q in data[("released", c)]}
            same = sum(1 for k in now if k in prior and
                       [list(map(float, x)) for x in prior[k]] ==
                       [list(map(float, x)) for x in now[k]])
            diffs = []
            for k in now:
                if k in prior and len(prior[k]) == len(now[k]) and now[k]:
                    a = np.array(prior[k], dtype=float)
                    b = np.array(now[k], dtype=float)
                    diffs.append(float(np.abs(a - b).max()))
            rows[c] = {"n_now": len(now), "n_prior": len(prior),
                       "panos_identical": same,
                       "panos_same_count_of_peaks": sum(
                           1 for k in now if k in prior and len(prior[k]) == len(now[k])),
                       "max_abs_diff_where_counts_match": rnd(max(diffs), 6) if diffs else 0.0}
        out[name] = rows
    return out


def build(root=CACHE_ROOT):
    rsq = radius_sq_for()
    data_all, meta = load(root)
    missing = [f"{lab}/{c}" for lab in LABELS for c in SPLITS if (lab, c) not in data_all]
    pl = contrast_list()
    rep = {"protocol": {"thresholds": list(THRESHOLDS), "n_reps": af.N_REPS,
                        "bootstrap_seed": af.SEED,
                        "bootstrap": "pano-level paired cluster bootstrap, stratified by "
                                     "split (benchmark_power_135.observed_and_se), via "
                                     "aug_finetune_82",
                        "labels": list(LABELS), "subsets": list(SUBSETS),
                        "unbiased": "verdicts.json group != 'top' (manual_gold has no top "
                                    "stratum, so its two subsets are the same panos)",
                        "missing_caches": missing,
                        "cache_models": {f"{k[0]}/{k[1]}": v.get("model")
                                         for k, v in sorted(meta.items())},
                        "fp16_any": any(bool(v.get("fp16")) for v in meta.values())},
           "per_split": {}, "pooled": {}, "determinism_released": determinism(data_all, root)}
    for sub in SUBSETS:
        data = subset(data_all, sub)
        scored = {k: af._scored(k[1], v, rsq) for k, v in data.items()}
        ps = rep["per_split"].setdefault(sub, {})
        for c in SPLITS:
            if not any((lab, c) in data for lab in LABELS):
                continue
            ps[c] = {
                "n_panos": len(next(data[(lab, c)] for lab in LABELS if (lab, c) in data)),
                "metrics": {lab: af._metrics(data[(lab, c)], rsq)
                            for lab in LABELS if (lab, c) in data},
                "max_f1": {lab: af._max_f1(scored[(lab, c)])
                           for lab in LABELS if (lab, c) in scored},
                "seed_spread": spread(scored, (c,)),
                "five_run_spread": spread(scored, (c,), FIVE_RUNS),
                "contrasts": af.contrasts_for(scored, (c,), pl),
                "seed_mean_contrasts": af.seed_mean_contrasts(scored, (c,), SEED_MEAN)}
        po = rep["pooled"].setdefault(sub, {})
        for name, members in POOLS.items():
            po[name] = {"members": list(members),
                        "n_panos": sum(len(data[("released", c)]) for c in members
                                       if ("released", c) in data),
                        "released": point(scored, members),
                        "seed_spread": spread(scored, members),
                        "five_run_spread": spread(scored, members, FIVE_RUNS),
                        "contrasts": af.contrasts_for(scored, members, pl),
                        "seed_mean_contrasts": af.seed_mean_contrasts(scored, members,
                                                                      SEED_MEAN)}
    # The Laurens rig effect (GSV minus GoPro on the 47 paired corners), per checkpoint and
    # as the three-seed mean against released; aug_finetune_82 reads its seed list from a
    # module constant, so it is pointed at three seeds for this call and restored after.
    saved = af.SEEDS
    af.SEEDS = SEEDS
    try:
        lp = af.laurens_rig_effect(data_all, rsq, ["released"] + list(CONTROLS))
    finally:
        af.SEEDS = saved
    rep["laurens_paired"] = lp
    return rep


def _fmt(d):
    return af._fmt(d)


def _prf(m, thr):
    return f"{m[thr]['P']:.3f} / {m[thr]['R']:.3f} / {m[thr]['F1']:.3f}"


def markdown(rep):
    L = ["# Issue #263 control arm: released weights + 2,000 GSV steps, three seeds", "",
         "Generated by `scripts/analysis/control_arm_263.py`; do not edit by hand.", ""]
    if rep["protocol"]["missing_caches"]:
        L += ["**Missing caches:** " + ", ".join(rep["protocol"]["missing_caches"]), ""]
    labs = rep["protocol"]["labels"][:4]
    for sub in rep["protocol"]["subsets"]:
        ps, po = rep["per_split"][sub], rep["pooled"][sub]
        L += [f"## Subset: {sub}", ""]
        for thr in ("0.30", "0.55"):
            L += [f"### P / R / F1 at {thr}", "",
                  "| split | n | " + " | ".join(labs) + " |", "|---|---:|" + "---|" * len(labs)]
            for c, ent in ps.items():
                L.append(f"| {c} | {ent['n_panos']} | " + " | ".join(
                    _prf(ent["metrics"][lab], thr) if lab in ent["metrics"] else "-"
                    for lab in labs) + " |")
            L += ["", f"### Seed mean minus released at {thr} (paired pano bootstrap, 95%), "
                  "with the seed range beside it", "",
                  "| split / pool | n | dR | dP | dF1 | d max-F1 | F1 per seed (s1, s2, s3) | "
                  "F1 range |", "|---|---:|---|---|---|---|---|---:|"]
            blocks = list(ps.items()) + [(f"**{k}**", v) for k, v in po.items()]
            for c, ent in blocks:
                for name, cs in ent["seed_mean_contrasts"].items():
                    if "5 runs" in name:
                        continue
                    d = cs[thr]
                    sp = ent["seed_spread"].get(thr, {}).get("f1")
                    seeds = ", ".join(f"{x:.3f}" for x in sp["per_seed"]) if sp else "-"
                    rng = f"{sp['range']:.3f}" if sp else "-"
                    L.append(f"| {c} | {ent['n_panos']} | {_fmt(d['recall'])} | "
                             f"{_fmt(d['precision'])} | {_fmt(d['f1'])} | {_fmt(d['max_f1'])} | "
                             f"{seeds} | {rng} |")
            L += ["", f"### Per-seed contrasts at {thr}, pools only (paired pano bootstrap, "
                  "95%); the three `spread` rows are the seed-to-seed noise floor", "",
                  "| pool | contrast | dR | dP | dF1 | d max-F1 |", "|---|---|---|---|---|---|"]
            for name_p, ent in po.items():
                for name, cs in ent["contrasts"].items():
                    d = cs[thr]
                    L.append(f"| {name_p} | {name} | {_fmt(d['recall'])} | "
                             f"{_fmt(d['precision'])} | {_fmt(d['f1'])} | "
                             f"{_fmt(d['max_f1'])} |")
            L += ["", f"### Five runs (three here, two from #82) at {thr}, pools", "",
                  "| pool | F1 per run (s1, s2, s3, #82 s1, #82 s2) | F1 SD | F1 range | "
                  "R range | 5-run mean minus released: dF1 | dR |",
                  "|---|---|---:|---:|---:|---|---|"]
            for name_p, ent in po.items():
                sp = ent["five_run_spread"].get(thr)
                cs = ent["seed_mean_contrasts"].get(
                    "control (mean of 5 runs, with #82's two) - released")
                if not sp or not cs:
                    L.append(f"| {name_p} | - | - | - | - | - | - |")
                    continue
                L.append(f"| {name_p} | " + ", ".join(f"{x:.3f}" for x in sp["f1"]["per_seed"])
                         + f" | {sp['f1']['sd']:.3f} | {sp['f1']['range']:.3f} | "
                         f"{sp['recall']['range']:.3f} | {_fmt(cs[thr]['f1'])} | "
                         f"{_fmt(cs[thr]['recall'])} |")
            L.append("")
        L += [f"### max-F1 (AP) per checkpoint, subset {sub}", "",
              "| split | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for c, ent in ps.items():
            L.append(f"| {c} | " + " | ".join(
                "-" if lab not in ent["max_f1"] else
                f"{ent['max_f1'][lab]:.3f} ({ent['metrics'][lab]['AP']:.3f})"
                for lab in labs) + " |")
        L.append("")
    lp = rep["laurens_paired"]
    L += [f"## Laurens rig effect on the {lp['n_pairs']} paired corners (GSV minus GoPro)", "",
          "| checkpoint | read | R gsv | R gopro | dR [95%] | F1 gsv | F1 gopro | dF1 [95%] |",
          "|---|---|---:|---:|---|---:|---:|---|"]
    for lab, ent in lp["per_checkpoint"].items():
        for thr, row in ent.items():
            f = row if thr == "max_f1" else row["f1"]
            fcell = (f"{f['gsv']:.3f} | {f['gopro']:.3f} | {f['gsv_minus_gopro']:+.3f} "
                     f"[{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}]")
            if thr == "max_f1":
                L.append(f"| {lab} | max-F1 | - | - | - | {fcell} |")
                continue
            r = row["recall"]
            L.append(f"| {lab} | {thr} | {r['gsv']:.3f} | {r['gopro']:.3f} | "
                     f"{r['gsv_minus_gopro']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                     f"{fcell} |")
    L += ["", "Change in the rig effect (difference in differences, same pair resample; "
          "negative = a smaller GSV-minus-GoPro gap):", "",
          "| contrast | read | d(rig effect) R [95%] | d(rig effect) F1 [95%] |",
          "|---|---|---|---|"]
    for key in ("rig_effect_vs_control", "rig_effect_vs_control_seed_mean"):
        for name, ent in lp.get(key, {}).items():
            for thr, row in ent.items():
                rc = _fmt(row["recall"]) if "recall" in row else "-"
                L.append(f"| {name} | {'max-F1' if thr == 'max_f1' else thr} | {rc} | "
                         f"{_fmt(row['f1'])} |")
    L += ["", "## Determinism: the released re-score against committed caches", "",
          "| reference | split | panos | identical peak lists | same peak count | "
          "max abs diff |", "|---|---|---:|---:|---:|---:|"]
    for name, rows in rep["determinism_released"].items():
        for c, r in rows.items():
            L.append(f"| {name} | {c} | {r['n_now']} | {r['panos_identical']} | "
                     f"{r['panos_same_count_of_peaks']} | {r['max_abs_diff_where_counts_match']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=CACHE_ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rep = json.loads(json.dumps(build(args.root)))
    md = markdown(rep)
    if args.check:
        with open(RESULTS, encoding="utf-8") as f:
            ok = json.load(f) == rep
        with open(RESULTS_MD, encoding="utf-8", newline="") as f:
            ok &= f.read() == md
        print("control arm results up to date" if ok else "control arm results STALE")
        return 0 if ok else 1
    af.write_json(RESULTS, rep)
    with open(RESULTS_MD, "w", encoding="utf-8", newline="") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
