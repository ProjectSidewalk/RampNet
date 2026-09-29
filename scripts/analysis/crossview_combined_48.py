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

#: (arm, family, post_hoc, role). ``post_hoc`` follows each family doc's own record of
#: which arms (or settings) were added after seeing scores on these same 300 pairs.
ROWS = [
    ("projection", "baseline", False, "today's 2.6 m flat-ground projection"),
    ("proj_height_auto", "baseline", False, "labeler 'auto' per-rig camera height"),
    ("proj_gsv_depth", "geometry (pilot)", False, "negative"),
    ("lg", "matching (pilot)", True, "ALIKED+LightGlue homography; 5 deg ground band post hoc"),
    ("sem_chamfer_auto", "semantic", False, "best of family"),
    ("lsd_chamfer", "semantic", False, "negative: raw line segments"),
    ("sem_snap_auto", "semantic", True, "negative: snap to curb"),
    ("roma", "matching", False, "RoMa + ground homography"),
    ("roma_local", "matching", True, "best Mapillary matcher"),
    ("roma_warp", "matching", True, "RoMa dense warp read at the GT point"),
    ("roma_warp_hyb", "matching", True, "RoMa dense warp, else auto"),
    ("sp_lg", "matching", False, "negative: SuperPoint+LightGlue"),
    ("loftr", "matching", False, "negative"),
    ("mono_da3_hcal", "depth", False, "best GSV depth arm"),
    ("mono_unidepth_point", "depth", True, "best Mapillary depth arm (picked from 17)"),
    ("mono_depthpro_point", "depth", False, "negative"),
    ("mapa_posed_pair", "multi-view 3D", False, "best of family"),
    ("mapa_posed_corner", "multi-view 3D", False, "with pose priors, up to 12 corner views"),
    ("mapa_k_pair", "multi-view 3D", False, "no pose priors"),
    ("mast3r_pair", "multi-view 3D", False, ""),
    ("dust3r_pair", "multi-view 3D", False, ""),
    ("vggt_pair", "multi-view 3D", False, ""),
    ("mv3d_consensus", "multi-view 3D", True, "MapAnything+MASt3R midpoint where they agree"),
    ("mv3d_consensus_else_auto", "multi-view 3D", True, "MapAnything+MASt3R agreement, else auto"),
    ("mapa_posed_poseonly", "multi-view 3D", False, "MapAnything's relative pose only"),
    ("mast3r_poseonly", "multi-view 3D", False, "negative: learned pose only"),
    ("mapa_mono_depthonly", "multi-view 3D", False, "negative: source view only"),
    ("sfm_colmap", "multi-view 3D", False, "negative on GSV: per-corner COLMAP"),
    ("flat_sfm", "flat Mapillary (Richmond only)", False, "SfM with flat images, sparse lift"),
    ("noflat_sfm", "flat Mapillary (Richmond only)", False, "control: same SfM, 360 panos only"),
    ("mlypano_sfm", "flat Mapillary (Richmond only)", False, "+ un-thinned Mapillary panos"),
    ("flat_mvs", "flat Mapillary (Richmond only)", False, "negative: MVS depth lift"),
    ("flat_gs", "flat Mapillary (Richmond only)", False, "negative: splat depth lift"),
]
RICHMOND_ONLY = {"flat_sfm", "noflat_sfm", "mlypano_sfm", "flat_mvs", "flat_gs"}


def load(name):
    if name == "projection":
        return None
    if name in RICHMOND_ONLY:
        import flat_mapillary_48 as F
        return F.read_preds_any(name)[0]
    return H.read_predictions(name)


def gsv_screen(pairs, idx, auto):
    """Every shared arm (not just the table's rows) whose paired GSV gain over auto has a CI
    lower bound above zero -- the basis for the doc's "only these beat auto on GSV"."""
    groups = {}
    for i in idx:
        groups.setdefault(pairs[i]["ramp_uid"], []).append(i)
    groups = [v for _, v in sorted(groups.items())]
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


def build():
    pairs = H.read_frozen_pairs()
    proj = H.arm_errors(pairs, None)
    auto = H.arm_errors(pairs, H.read_predictions(AUTO))
    strata = {"all": list(range(len(pairs))),
              "gsv": [i for i, p in enumerate(pairs) if p["imagery"] != "mapillary"],
              "mapillary": [i for i, p in enumerate(pairs) if p["imagery"] == "mapillary"]}
    assert len(strata["mapillary"]) == 60 and {pairs[i]["city"] for i in strata["mapillary"]} \
        == {"richmond"}, "Mapillary stratum is expected to be the 60 Richmond pairs"
    rows = []
    for arm, family, post_hoc, role in ROWS:
        errs = H.arm_errors(pairs, load(arm))
        use = ("mapillary",) if arm in RICHMOND_ONLY else ("all", "gsv", "mapillary")
        rows.append({"arm": arm, "family": family, "post_hoc": post_hoc, "role": role,
                     "richmond_only": arm in RICHMOND_ONLY,
                     **{s: block(pairs, strata[s], errs, proj, auto) for s in use}})
    screen = gsv_screen(pairs, strata["gsv"], auto)
    return {"gsv_ci_clear_vs_auto": screen,
            "config": {"pairs_sha256": H.PAIRS_SHA256, "n_boot": H.N_BOOT, "seed": H.SEED,
                       "gain": "median per-pair (proj_height_auto error - arm error), deg",
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


def markdown(res):
    out = ["| arm | family | post hoc | all 300: median ° [CI] | fallback | paired gain vs "
           "auto, all [CI] | GSV (240): median ° [CI] | GSV gain vs auto [CI] | Mapillary "
           "(60): median ° [CI] | Mapillary gain vs auto [CI] | where it answers: n, arm vs auto °, "
           "gain vs auto [CI] |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
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
            cell_med(m), "–" if auto_row else cell_gain(m), cell_answered(a or m, auto_row)])
                   + " |")
    return "\n".join(out)


def main():
    res = build()
    H.write_json(OUT_JSON, res)
    sys.stdout.reconfigure(encoding="utf-8")
    print(markdown(res))
    print("GSV gain vs auto CI-clear (all shared arms): " + ", ".join(
        f"{a} {v['gain_vs_auto']:+.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
        for a, v in res["gsv_ci_clear_vs_auto"].items()))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
