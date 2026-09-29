"""One comparison table across every cross-view arm family (#48).

Reads only committed files: the frozen ``pairs.csv``, the shared
``analysis_out/crossview_align_48/predictions/<arm>.jsonl`` and, for the Richmond-only
flat-Mapillary arms (#214), ``analysis_out/flat_mapillary_3d/predictions/<arm>.jsonl``.
Writes ``analysis_out/crossview_align_48/combined_table.json`` and prints the markdown
table in ``docs/crossview_align_48.md`` ("Combined comparison").

Every number uses the harness's own definitions (``crossview_align_48.arm_errors``: a
missing or null prediction falls back to the 2.6 m projection) and the matching family's
``block`` (medians, ramp-cluster bootstrap CIs, paired gain against ``proj_height_auto``).

**Subset arms.** The flat-Mapillary arms answer only on the 60 Richmond pairs and return
null on the 240 GSV pairs. Scored over all 300 they would fall back on 80% and be ranked
as if they had run everywhere, so they are scored on the Richmond stratum ONLY and their
all-pairs and GSV cells are left empty.

**Two screens of the GSV gain over auto.** ``gsv_ci_clear_vs_auto`` is uncorrected (each
arm's 95% CI clears zero); ``multiplicity`` applies a one-sided Bonferroni correction over
every shared arm (20,000 ramp resamples, seed 48) on GSV and on all 300 pairs. Quote a
CI-clear gain only beside the corrected result.

**A common fallback rule.** Each row also carries ``else_auto``: the arm rescored with every
fallback scored as ``proj_height_auto``'s point, and the share of pairs so scored (plus a
hybrid's own auto-prior answers), so plain and ``_hyb`` / ``_else_auto`` arms compare fairly.

Usage::

    python scripts/analysis/crossview_combined_48.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import crossview_align_48 as H  # noqa: E402
from crossview_matching_48 import block  # noqa: E402

OUT_JSON = os.path.join(H.OUT, "combined_table.json")
AUTO = "proj_height_auto"

#: (arm, family, post_hoc, role). ``post_hoc`` is False, or a string saying why the arm is
#: post hoc: it (or a setting of it) was added, or it was picked for this table, after scores
#: on these same 300 pairs (or a subset of them, e.g. a 30-pair pilot) had been seen. Corrected
#: 2026-09-29 after the review of #210 checked each arm against git history: the headline
#: ``mapa_posed_pair`` is post hoc (see ``docs/crossview_align_48.md``, "post hoc" bullet).
#: ``mast3r_poseonly`` was added by the final re-review (N4); its evidence is commit times
#: only, so its reason says "inferred". Every post hoc arm here and in POST_HOC_NOT_IN_TABLE
#: carries the same reason in its meta.json ``provenance_correction`` (tested).
ROWS = [
    ("projection", "baseline", False, "today's 2.6 m flat-ground projection"),
    ("proj_height_auto", "baseline", False, "labeler 'auto' per-rig camera height"),
    ("proj_gsv_depth", "geometry (pilot)", False, "negative"),
    ("lg", "matching (pilot)", "5 deg ground band chosen after the 0.5 deg band's failures",
     "ALIKED+LightGlue homography; 5 deg ground band post hoc"),
    ("sem_chamfer_auto", "semantic", False, "best of family"),
    ("lsd_chamfer", "semantic", False, "negative: raw line segments"),
    ("sem_snap_auto", "semantic", "added after the first semantic arms were scored",
     "negative: snap to curb"),
    ("roma", "matching", False, "RoMa + ground homography"),
    ("roma_local", "matching", "added after `roma` was scored and did not beat auto on GSV",
     "best Mapillary matcher"),
    ("roma_warp", "matching", "added after `roma` was scored and did not beat auto on GSV",
     "RoMa dense warp read at the GT point"),
    ("roma_warp_hyb", "matching", "added after `roma` was scored and did not beat auto on GSV",
     "RoMa dense warp, else auto"),
    ("sp_lg", "matching", False, "negative: SuperPoint+LightGlue"),
    ("loftr", "matching", False, "negative"),
    ("mono_da3_hcal", "depth", "picked for this table from 17 depth arms scored on these pairs",
     "best GSV depth arm (picked from 17)"),
    ("mono_unidepth_point", "depth",
     "picked for this table from 17 depth arms scored on these pairs",
     "best Mapillary depth arm (picked from 17)"),
    ("mono_depthpro_point", "depth", False, "negative"),
    ("mapa_posed_pair", "multi-view 3D",
     "registered in 0428bb7, after mapa_posed_corner's 300-pair result was seen; no pilot",
     "best of family (post hoc: needs the fresh-pair re-test)"),
    ("mapa_posed_corner", "multi-view 3D", False, "with pose priors, up to 12 corner views"),
    ("mapa_k_pair", "multi-view 3D", False, "no pose priors"),
    ("mast3r_pair", "multi-view 3D", False, ""),
    ("dust3r_pair", "multi-view 3D", False, ""),
    ("vggt_pair", "multi-view 3D", False, ""),
    ("mv3d_consensus", "multi-view 3D", "defined after both components were scored",
     "MapAnything+MASt3R midpoint where they agree"),
    ("mv3d_consensus_else_auto", "multi-view 3D", "defined after both components were scored",
     "MapAnything+MASt3R agreement, else auto"),
    ("mapa_posed_poseonly", "multi-view 3D",
     "registered in 0428bb7, after mapa_posed_corner's 300-pair result was seen",
     "MapAnything's relative pose only"),
    ("mast3r_poseonly", "multi-view 3D",
     "registered in bb6cd3c at 18:18, three minutes after 0214816 quoted the 30-pair pilot's "
     "scores (a subset of these pairs); that the pilot was seen first is inferred from commit "
     "times, not stated in a commit message",
     "negative: learned pose only"),
    ("mapa_mono_depthonly", "multi-view 3D",
     "registered in c84dc74, after the posed MapAnything results were seen",
     "negative: source view only"),
    ("sfm_colmap", "multi-view 3D",
     "gravity-level ground lift chosen after the 30-pair pilot (a subset of these pairs)",
     "negative on GSV: per-corner COLMAP"),
    ("flat_sfm", "flat Mapillary (Richmond only)", False, "SfM with flat images, sparse lift"),
    ("noflat_sfm", "flat Mapillary (Richmond only)", False, "control: same SfM, 360 panos only"),
    ("mlypano_sfm", "flat Mapillary (Richmond only)", False, "+ un-thinned Mapillary panos"),
    ("flat_mvs", "flat Mapillary (Richmond only)", False, "negative: MVS depth lift"),
    ("flat_gs", "flat Mapillary (Richmond only)", False, "negative: splat depth lift"),
]
RICHMOND_ONLY = {"flat_sfm", "noflat_sfm", "mlypano_sfm", "flat_mvs", "flat_gs"}

#: Post hoc arms that are not rows of the table, by the same rule as ROWS' ``post_hoc``
#: (final re-review of #210, N4, checked against git history). Their meta.json files carry
#: the same ``provenance_correction``. Not listed: every matching-family arm inherits
#: ``lg``'s 5 deg ground band, which was chosen post hoc in the pilot (matching.md §4).
POST_HOC_NOT_IN_TABLE = {
    "sfm_colmap_prior": "gravity-level ground lift chosen after the 30-pair pilot (0214816; a "
                        "subset of these pairs), as for sfm_colmap",
    "sfm_poseonly": "registered in 0214816, whose commit message quotes the 30-pair pilot's "
                    "scores (a subset of these pairs)",
    "vggt_corner_poseonly": "registered in 6695b4f, whose commit message quotes the pilot's "
                            "mapa_posed_depthonly result (a subset of these pairs)",
    "sem_curb_shift_auto": "added after the 2.6 m snap scores were seen (semantic.md §1)",
    "flat_gsmed": "median-depth splat lift chosen after the _gs lift failed "
                  "(flat_mapillary_3d.md)",
    "noflat_gsmed": "median-depth splat lift chosen after the _gs lift failed "
                    "(flat_mapillary_3d.md)",
    "mlypano_gsmed": "median-depth splat lift chosen after the _gs lift failed "
                     "(flat_mapillary_3d.md)",
}


def meta_path(name):
    """The committed meta.json of an arm (the flat family's live beside its own jsonl)."""
    p = H.prediction_paths(name)[1]
    if os.path.exists(p):
        return p
    return os.path.join(H.OUT_ROOT, "flat_mapillary_3d", "predictions", f"{name}.meta.json")


def load(name):
    if name == "projection":
        return None
    if name in RICHMOND_ONLY:
        import flat_mapillary_48 as F
        return F.read_preds_any(name)[0]
    return H.read_predictions(name)


def _groups(pairs, idx):
    groups = {}
    for i in idx:
        groups.setdefault(pairs[i]["ramp_uid"], []).append(i)
    return [v for _, v in sorted(groups.items())]


def gsv_screen(pairs, idx, auto):
    """Every shared arm (not just the table's rows) whose paired GSV gain over auto has a CI
    lower bound above zero. UNCORRECTED: a 2.5% one-sided test per arm, repeated over every
    arm in predictions/. ``multiplicity_screen`` is the family-wise version."""
    groups = _groups(pairs, idx)
    out = {}
    for arm in H.available_predictions():
        e = H.arm_errors(pairs, H.read_predictions(arm))

        def g(ii, e=e):
            return float(np.median([auto[i][0] - e[i][0] for i in ii]))
        if g(idx) <= 0:
            continue
        ci = H.cluster_bootstrap(groups, g)
        if ci[0] > 0:
            out[arm] = {"gain_vs_auto": g(idx), "ci": ci}
    return out


#: Family-wise screen (review of #210, A1): one-sided Bonferroni over every shared arm.
MT_ALPHA = 0.05
MT_N_BOOT = 20000


def multiplicity_screen(pairs, idx, auto, n_boot=MT_N_BOOT, alpha=MT_ALPHA, seed=H.SEED):
    """Bonferroni-corrected, one-sided ramp-bootstrap screen of the paired gain over auto.

    Every shared arm in ``predictions/`` (83 when this was written, ``proj_height_auto``
    itself included, which can never pass) is tested for gain > 0 at level alpha / n_arms.
    One set of ``n_boot`` ramp resamples (seed fixed) is drawn once and shared by every arm
    (common random numbers). An arm survives when the (100 * alpha / n_arms)th percentile of
    its resampled median gain is above zero.

    Bonferroni ignores the strong correlation between arms, so it is conservative: a
    survivor is robust to the selection; a non-survivor is not thereby shown to be null.

    Returns (config, {arm: {...}}) for the arms whose point gain is positive.

    Example: ``cfg, res = multiplicity_screen(pairs, gsv_idx, auto)``; then
    ``[a for a, v in res.items() if v["survives"]]``.
    """
    arms = H.available_predictions()
    groups = [np.asarray(g) for g in _groups(pairs, idx)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    picks = [np.concatenate([groups[j] for j in rng.integers(0, k, k)]) for _ in range(n_boot)]
    q = 100.0 * alpha / len(arms)
    a = np.array([r[0] for r in auto])
    out = {}
    for arm in arms:
        d = a - np.array([r[0] for r in H.arm_errors(pairs, H.read_predictions(arm))])
        point = float(np.median(d[idx]))
        if point <= 0:
            continue
        vals = np.array([np.median(d[pk]) for pk in picks])
        lo = float(np.percentile(vals, q))
        out[arm] = {"gain_vs_auto": round(point, 4),
                    "share_resamples_le_0": round(float(np.mean(vals <= 0)), 5),
                    "bonferroni_lower": round(lo, 4), "survives": bool(lo > 0)}
    cfg = {"n_arms": len(arms), "alpha": alpha, "sided": "one (gain > 0)",
           "correction": "Bonferroni over n_arms", "percentile": q, "n_boot": n_boot,
           "seed": seed, "resample": "ramps with replacement, one draw shared by all arms"}
    return cfg, out


def else_auto(errs, auto, raw):
    """``errs`` rescored under one common fallback rule: wherever the arm fell back, score
    ``proj_height_auto``'s point instead of the 2.6 m projection (review of #210, B3).

    Also returns, per pair, whether the answer scored is the auto prior: a fallback, or a
    hybrid's own ``via: auto_prior`` / ``used: proj_height_auto`` answer. ``raw`` is
    {pair index: prediction row} or None (the projection)."""
    out, is_auto = [], []
    for i, r in enumerate(errs):
        row = (raw or {}).get(i) or {}
        own_auto = row.get("via") == "auto_prior" or row.get("used") == AUTO
        out.append((auto[i][0], auto[i][1], auto[i][2], r[3]) if r[3] else r)
        is_auto.append(bool(r[3]) or own_auto)
    return out, is_auto


def common_rule_block(pairs, idx, errs, auto, is_auto):
    """Median [CI], gain over auto [CI] and auto-prior share under the common rule."""
    groups = _groups(pairs, idx)

    def m(ii):
        return float(np.median([errs[i][0] for i in ii]))

    def g(ii):
        return float(np.median([auto[i][0] - errs[i][0] for i in ii]))
    c1, c2 = H.cluster_bootstrap(groups, m), H.cluster_bootstrap(groups, g)
    return {"median_deg": round(m(idx), 4), "median_ci": [round(c1[0], 3), round(c1[1], 3)],
            "gain_vs_auto": round(g(idx), 4),
            "gain_vs_auto_ci": [round(c2[0], 3), round(c2[1], 3)],
            "auto_prior_share": round(float(np.mean([is_auto[i] for i in idx])), 4)}


def strata_for(pairs):
    return {"all": list(range(len(pairs))),
            "gsv": [i for i, p in enumerate(pairs) if p["imagery"] != "mapillary"],
            "mapillary": [i for i, p in enumerate(pairs) if p["imagery"] == "mapillary"]}


def build():
    pairs = H.read_frozen_pairs()
    proj = H.arm_errors(pairs, None)
    auto = H.arm_errors(pairs, H.read_predictions(AUTO))
    strata = strata_for(pairs)
    assert len(strata["mapillary"]) == 60 and {pairs[i]["city"] for i in strata["mapillary"]} \
        == {"richmond"}, "Mapillary stratum is expected to be the 60 Richmond pairs"
    pos = {p["pair_id"]: i for i, p in enumerate(pairs)}
    rows = []
    for arm, family, post_hoc, role in ROWS:
        preds = load(arm)
        errs = H.arm_errors(pairs, preds)
        use = ("mapillary",) if arm in RICHMOND_ONLY else ("all", "gsv", "mapillary")
        row = {"arm": arm, "family": family, "post_hoc": bool(post_hoc),
               "post_hoc_why": post_hoc or None, "role": role,
               "richmond_only": arm in RICHMOND_ONLY,
               **{s: block(pairs, strata[s], errs, proj, auto) for s in use}}
        if arm != AUTO:
            raw = None if preds is None else {pos[k]: v for k, v in preds.items()}
            ea, is_auto = else_auto(errs, auto, raw)
            row["else_auto"] = {s: common_rule_block(pairs, strata[s], ea, auto, is_auto)
                                for s in use}
        rows.append(row)
    screen = gsv_screen(pairs, strata["gsv"], auto)
    mt_cfg, mt_gsv = multiplicity_screen(pairs, strata["gsv"], auto)
    _, mt_all = multiplicity_screen(pairs, strata["all"], auto)
    return {"gsv_ci_clear_vs_auto": screen,
            "multiplicity": {"config": mt_cfg, "gsv": mt_gsv, "all": mt_all,
                             "gsv_survivors": sorted(a for a, v in mt_gsv.items()
                                                     if v["survives"]),
                             "all_survivors": sorted(a for a, v in mt_all.items()
                                                     if v["survives"])},
            "config": {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT, "seed": H.SEED,
                       "gain": "median per-pair (proj_height_auto error - arm error), deg, over "
                               "all pairs of the stratum; a fallback is scored as the 2.6 m "
                               "projection",
                       "else_auto": "the same, but a fallback is scored as proj_height_auto's "
                                    "point: one common rule for every arm",
                       "aligned": "the family docs' definition: the pairs the arm did not "
                                  "fall back on only",
                       "ci": "2.5-97.5 percentile, ramps resampled",
                       "richmond_only": sorted(RICHMOND_ONLY)},
            "rows": rows}


def ci(c):
    return "" if not c else f" [{c[0]:.2f}, {c[1]:.2f}]"


def cell_med(b):
    return "–" if b is None else f"{b['median_deg']:.2f}{ci(b['median_ci'])}"


def cell_gain(b):
    return "–" if b is None else f"{b['gain_vs_auto']:+.2f}{ci(b['gain_vs_auto_ci'])}"


def cell_answered(b, auto_row):
    """The arm on the pairs where it did not fall back (all 300, or Richmond for the
    Richmond-only arms): the gain there is not diluted by fallbacks."""
    o = b.get("aligned")
    if auto_row or not o or b["fallback_rate"] == 0:
        return "–" if auto_row or not o else "(never falls back)"
    return (f"{o['n']}: {o['median_deg']:.2f} vs {o['auto_median_deg']:.2f}, "
            f"{o['gain_vs_auto']:+.2f}{ci(o['gain_vs_auto_ci'])}")


def cell_common(r):
    """The arm under the common else-auto rule, on all 300 (Richmond for Richmond-only)."""
    ea = r.get("else_auto")
    if not ea:
        return "–"
    b = ea.get("all") or ea["mapillary"]
    return (f"{b['median_deg']:.2f}, {b['gain_vs_auto']:+.2f}{ci(b['gain_vs_auto_ci'])}"
            + ("" if "all" in ea else " (Richmond)"))


def cell_auto_share(r):
    ea = r.get("else_auto")
    if not ea:
        return "–"
    return f"{(ea.get('all') or ea['mapillary'])['auto_prior_share']:.2f}"


def markdown(res):
    out = ["| arm | family | post hoc | all 300: median ° [CI] | fallback | paired gain vs "
           "auto, all [CI] | GSV (240): median ° [CI] | GSV gain vs auto [CI] | Mapillary "
           "(60): median ° [CI] | Mapillary gain vs auto [CI] | where it answers: n, arm vs auto °, "
           "gain vs auto [CI] | common rule (fallback → auto), all 300: median °, gain vs auto "
           "[CI] | share scored at the auto prior |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        a, g, m = r.get("all"), r.get("gsv"), r["mapillary"]
        fb = a["fallback_rate"] if a else m["fallback_rate"]
        fb_s = f"{fb:.2f}" + ("" if a else " (Richmond)")
        auto_row = r["arm"] == AUTO
        out.append(" | ".join([
            f"| `{r['arm']}`", r["family"], "yes" if r["post_hoc"] else "",
            cell_med(a) if a else "not run (Richmond only)", fb_s,
            "–" if auto_row else (cell_gain(a) if a else "–"),
            cell_med(g) if g else "n/a", "–" if auto_row else (cell_gain(g) if g else "n/a"),
            cell_med(m), "–" if auto_row else cell_gain(m), cell_answered(a or m, auto_row),
            cell_common(r), cell_auto_share(r)]) + " |")
    return "\n".join(out)


def main():
    res = build()
    H.write_json(OUT_JSON, res)
    sys.stdout.reconfigure(encoding="utf-8")
    print(markdown(res))
    print("GSV gain vs auto CI-clear, UNCORRECTED (all shared arms): " + ", ".join(
        f"{a} {v['gain_vs_auto']:+.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
        for a, v in res["gsv_ci_clear_vs_auto"].items()))
    mt = res["multiplicity"]
    print(f"Bonferroni over {mt['config']['n_arms']} arms, one-sided, {mt['config']['n_boot']} "
          f"ramp resamples: GSV survivors {', '.join(mt['gsv_survivors']) or 'none'}; "
          f"all-300 survivors {', '.join(mt['all_survivors']) or 'none'}")
    for s in ("gsv", "all"):
        for a, v in sorted(mt[s].items(), key=lambda kv: -kv[1]["bonferroni_lower"])[:8]:
            print(f"  {s} {a}: gain {v['gain_vs_auto']:+.2f}, share<=0 "
                  f"{v['share_resamples_le_0']:.4f}, Bonferroni lower {v['bonferroni_lower']:+.3f}")
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
