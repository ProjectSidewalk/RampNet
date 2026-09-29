"""Fresh-pair confirmation of the #48 cross-view placement winners (follow-up 1 of
docs/crossview_align_48.md).

The 300-pair comparison found three MapAnything arms whose paired gain over
``proj_height_auto`` survived a Bonferroni screen over 83 arms, and the best of them
(``mapa_posed_pair``) was registered after its sibling's 300-pair result had been seen. This
script re-tests them on pairs no arm has been scored on, with everything fixed before any
prediction was made:

* **Pairs** (``pairs``): every pair of the committed ``eligible_pairs.csv`` that is not in the
  frozen ``pairs.csv``, drawn by the same rule as the 300 (``crossview_align_48.sample_pairs``:
  per city, ramps in seeded order, at most 2 other views per ramp) with NO per-city cap and
  seed ``FRESH_SEED``. Pinned by ``crossview_align_48.FRESH_PAIRS_SHA256``; ids ``f000``...
* **Arms**: ``FRESH_ARMS``, with their committed settings unchanged, plus the baselines
  ``projection`` and ``proj_height_auto``.
* **Primary metric** (``score``): the paired median gain of each arm over
  ``proj_height_auto`` (a fallback is scored at the 2.6 m projection, the harness rule),
  with a ramp-bootstrap CI, tested one-sided (gain > 0) at alpha 0.05 / 3 (Bonferroni over
  the three arms; ``BONF_N_BOOT`` resamples, seed 48, one draw shared by the arms). Reported
  for all pairs, GSV and Mapillary, each stratum tested separately.
* **Primary analysis**: the pairs whose ramp is NOT one of the 174 ramps in the frozen 300.
  **Secondary**: every fresh pair.

Subcommands:

    # 1. the fresh pair list (desktop CPU, committed inputs only). Frozen once written.
    python scripts/analysis/crossview_fresh_48.py pairs
    # 2. everything else runs through the harness with CROSSVIEW48_PAIR_SET=fresh
    #    (cut-views, _mv3d.py manifest / render / predict-many, predict proj_height_auto)
    # 3. the pre-specified test (CPU, committed predictions only) -> fresh/confirmation.json
    python scripts/analysis/crossview_fresh_48.py score
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import crossview_align_48 as H  # noqa: E402

#: the arms re-tested, settings unchanged from their committed 300-pair runs
FRESH_ARMS = ("mapa_posed_pair", "mapa_posed_corner", "mapa_k_pair")
BASELINE = "proj_height_auto"
FRESH_SEED = "48:fresh"           # sample_pairs seeds each city with f"{seed}:{city}"
ALPHA = 0.05
BONF_N_BOOT = 20000
CONFIRMATION_JSON = os.path.join(H.FRESH_DIR, "confirmation.json")
FRESH_META_JSON = os.path.join(H.FRESH_DIR, "pairs_meta.json")


def _sha(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def draw_fresh(eligible, frozen):
    """Every eligible pair not in ``frozen`` (by ramp and other pano), sampled with
    ``sample_pairs`` at no per-city cap. Returns the fresh rows with ids f000..."""
    used = {(r["ramp_uid"], r["oth_pano"]) for r in frozen}
    pool = [dict(r) for r in eligible if (r["ramp_uid"], r["oth_pano"]) not in used]
    return H.sample_pairs(pool, per_city=None, per_ramp=H.MAX_PAIRS_PER_RAMP, seed=FRESH_SEED,
                          id_prefix="f")


def cmd_pairs(args):
    fresh_csv = os.path.join(H.FRESH_DIR, "pairs.csv")
    if os.path.exists(fresh_csv) and not args.force:
        raise SystemExit(f"{fresh_csv} is frozen (FRESH_PAIRS_SHA256); pass --force to rebuild")
    if _sha(H.ELIGIBLE_CSV) != H.ELIGIBLE_SHA256:
        raise SystemExit(f"{H.ELIGIBLE_CSV} is not the committed eligible list")
    prev = H.use_pair_set("frozen300")
    try:
        frozen = H.read_frozen_pairs()
    finally:
        H.use_pair_set(prev)
    eligible = H.read_rows(H.ELIGIBLE_CSV)
    rows = draw_fresh(eligible, frozen)
    H.write_rows(fresh_csv, rows)
    old = {r["ramp_uid"] for r in frozen}
    per_city = {}
    for c in H.CITIES:
        rc = [r for r in rows if r["city"] == c]
        per_city[c] = {"pairs": len(rc), "ramps": len({r["ramp_uid"] for r in rc}),
                       "pairs_on_new_ramps": sum(r["ramp_uid"] not in old for r in rc),
                       "new_ramps": len({r["ramp_uid"] for r in rc if r["ramp_uid"] not in old}),
                       "eligible_not_in_300": sum(1 for r in eligible if r["city"] == c) -
                       sum(1 for r in frozen if r["city"] == c)}
    H.write_json(FRESH_META_JSON, {
        "eligible_sha256": H.ELIGIBLE_SHA256, "frozen300_sha256": PAIRS_SHA256_FROZEN,
        "seed": FRESH_SEED, "max_pairs_per_ramp": H.MAX_PAIRS_PER_RAMP, "per_city_cap": None,
        "pairs": len(rows), "pairs_on_new_ramps": sum(r["ramp_uid"] not in old for r in rows),
        "per_city": per_city, "pairs_sha256": _sha(fresh_csv)})
    print(json.dumps(per_city, indent=1))
    print(f"{len(rows)} fresh pairs -> {fresh_csv}  sha256 {_sha(fresh_csv)}")


PAIRS_SHA256_FROZEN = H.PAIR_SETS["frozen300"]["pairs_sha256"]


def old_ramps():
    """The ramp uids of the frozen 300 (174 ramps)."""
    prev = H.use_pair_set("frozen300")
    try:
        return {r["ramp_uid"] for r in H.read_frozen_pairs()}
    finally:
        H.use_pair_set(prev)


def _groups(pairs, idx):
    g = defaultdict(list)
    for i in idx:
        g[pairs[i]["ramp_uid"]].append(i)
    return [np.asarray(v) for _, v in sorted(g.items())]


def confirm(pairs, idx, errs, n_boot=BONF_N_BOOT, alpha=ALPHA, seed=H.SEED):
    """The pre-specified test on pairs ``idx``. ``errs``: {arm: arm_errors rows}, including
    ``projection`` and BASELINE. Per arm: median [95% CI], within 2 deg, fallback, paired
    median gain over BASELINE [95% CI], and its one-sided Bonferroni lower bound at
    alpha / len(FRESH_ARMS) (and, as a sensitivity read, at alpha / (3 arms x 3 strata)).
    One set of ramp resamples is shared by every arm and statistic."""
    groups = _groups(pairs, idx)
    rng = np.random.default_rng(seed)
    k = len(groups)
    picks = [np.concatenate([groups[j] for j in rng.integers(0, k, k)]) for _ in range(n_boot)]
    base = np.array([r[0] for r in errs[BASELINE]])
    q3 = 100.0 * alpha / len(FRESH_ARMS)
    q9 = 100.0 * alpha / (len(FRESH_ARMS) * 3)
    out = {"n_pairs": len(idx), "n_ramps": k, "arms": {}}
    ii = np.asarray(idx)
    for arm in ("projection", BASELINE) + FRESH_ARMS:
        e = np.array([r[0] for r in errs[arm]])
        meds = np.array([np.median(e[pk]) for pk in picks])
        row = {"median_deg": float(np.median(e[ii])),
               "median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
               "within_2deg": float(np.mean(e[ii] <= 2.0)),
               "fallback_rate": float(np.mean([errs[arm][i][3] for i in idx]))}
        if arm != BASELINE:
            d = base - e
            gains = np.array([np.median(d[pk]) for pk in picks])
            row.update({
                "gain_vs_auto": float(np.median(d[ii])),
                "gain_ci": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
                "closer_than_auto": float(np.mean(d[ii] > 1e-9)),
                "share_resamples_le_0": float(np.mean(gains <= 0))})
            if arm in FRESH_ARMS:
                lo3, lo9 = float(np.percentile(gains, q3)), float(np.percentile(gains, q9))
                row.update({"bonferroni_lower": lo3, "confirmed": bool(lo3 > 0),
                            "bonferroni9_lower": lo9, "confirmed_at_9": bool(lo9 > 0)})
        out["arms"][arm] = row
    return out


def score_fresh():
    """{"config", "primary", "secondary", "per_city"} from the committed fresh predictions."""
    prev = H.use_pair_set("fresh")
    try:
        pairs = H.read_frozen_pairs()
        errs = {"projection": H.arm_errors(pairs, None)}
        for arm in (BASELINE,) + FRESH_ARMS:
            errs[arm] = H.arm_errors(pairs, H.read_predictions(arm))
    finally:
        H.use_pair_set(prev)
    old = old_ramps()
    new_idx = [i for i, p in enumerate(pairs) if p["ramp_uid"] not in old]
    all_idx = list(range(len(pairs)))
    res = {"config": {
        "pairs_sha256": H.FRESH_PAIRS_SHA256, "frozen300_sha256": PAIRS_SHA256_FROZEN,
        "arms": list(FRESH_ARMS), "baseline": BASELINE, "alpha": ALPHA,
        "test": "one-sided (gain > 0), Bonferroni over the 3 arms within each stratum",
        "sensitivity": "bonferroni9_lower: alpha / 9 (3 arms x 3 strata)",
        "n_boot": BONF_N_BOOT, "seed": H.SEED,
        "resample": "ramps with replacement, one draw per stratum shared by all arms",
        "fallback": "scored at the 2.6 m projection (harness rule)",
        "primary": "pairs whose ramp is not one of the 174 ramps of the frozen 300",
        "secondary": "every fresh pair"}}
    for name, idx in (("primary", new_idx), ("secondary", all_idx)):
        res[name] = {}
        for stratum, keep in (("all", lambda p: True),
                              ("gsv", lambda p: p["imagery"] == "gsv"),
                              ("mapillary", lambda p: p["imagery"] == "mapillary")):
            sub = [i for i in idx if keep(pairs[i])]
            res[name][stratum] = confirm(pairs, sub, errs)
    res["per_city"] = {}
    for c in H.CITIES:
        sub = [i for i in new_idx if pairs[i]["city"] == c]
        res["per_city"][c] = {"n_pairs": len(sub), **{
            a: float(np.median([errs[a][i][0] for i in sub]))
            for a in ("projection", BASELINE) + FRESH_ARMS}}
    return res


def fmt_ci(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def markdown(res, which="primary"):
    lines = ["| arm | stratum | n pairs (ramps) | median ° [CI] | within 2° | fallback | "
             "paired gain vs auto [95% CI] | Bonferroni lower (α/3) | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for stratum in ("all", "gsv", "mapillary"):
        s = res[which][stratum]
        for arm in ("projection", BASELINE) + FRESH_ARMS:
            a = s["arms"][arm]
            gain = (f"{a['gain_vs_auto']:+.2f} {fmt_ci(a['gain_ci'])}"
                    if "gain_vs_auto" in a else "–")
            lo = f"{a['bonferroni_lower']:+.2f}" if "bonferroni_lower" in a else ""
            verdict = ("confirmed" if a["confirmed"] else "not confirmed") \
                if "confirmed" in a else ""
            lines.append(f"| `{arm}` | {stratum} | {s['n_pairs']} ({s['n_ramps']}) | "
                         f"{a['median_deg']:.2f} {fmt_ci(a['median_ci'])} | "
                         f"{a['within_2deg']:.2f} | {a['fallback_rate']:.2f} | {gain} | {lo} | "
                         f"{verdict} |")
    return "\n".join(lines)


def cmd_score(args):
    res = score_fresh()
    H.write_json(args.out, res)
    for which in ("primary", "secondary"):
        print(f"\n### {which}\n")
        print(markdown(res, which))
    print("\nper city (primary), medians:")
    for c, v in res["per_city"].items():
        print(f"  {c:12s} n={v['n_pairs']:3d} " + " ".join(
            f"{a}={v[a]:.2f}" for a in ("projection", BASELINE) + FRESH_ARMS))
    print(f"-> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs", help="draw the fresh pair list (frozen once written)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_pairs)
    p = sub.add_parser("score", help="the pre-specified test -> fresh/confirmation.json")
    p.add_argument("--out", default=CONFIRMATION_JSON)
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
