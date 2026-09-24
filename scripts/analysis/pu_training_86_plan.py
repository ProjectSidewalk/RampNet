"""Numbers behind docs/pu_training_86.md (#86, RampNet 2.0 plan item 5), from committed tables.

The plan for positive-unlabeled (PU) training of the tag head quotes three kinds of derived
number: how much of the ASSETS'24 HF train set survives the test exclusions, the per-tag class
priors, and the PU quantities that follow from a prior and an observed tag rate (labeling
frequency c_t, the positive fraction among untagged labels pi_U, the soft-negative target).
This script derives all of them so the doc can cite one committed output instead of arithmetic
done in a session.

The audit tables (``tags.csv``, ``tiers.csv``) are on ``main`` since PR #183 merged (29fe638),
so the default paths work for them. The two HF tables are on the #178 branch until it merges;
fetch them with ``git show``:

    git show origin/bench/tag-benchmark-86:analysis_out/tag_benchmark_86/hf_curbramp_labels.csv > hf.csv
    git show origin/bench/tag-benchmark-86:analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv > rs.csv
    python scripts/analysis/pu_training_86_plan.py --labels hf.csv --resplit rs.csv \\
        --out analysis_out/pu_training_86/plan_numbers.json

Definitions, per tag t, in the censoring reading of PU data (a label either carries t or it
does not, and a label that does not is unlabeled for t):

- o_t, the observed tag rate: labels carrying t / all labels of the tier (tag era).
- pi_t, the class prior: the fraction of curb ramps that truly carry t.
- c_t = o_t / pi_t, the labeling frequency: P(tagged | truly t).
- pi_U = (pi_t - o_t) / (1 - o_t), the fraction of *untagged* labels that truly carry t.
  Clamped at 0 when o_t >= pi_t (then the prior says every true positive was tagged, and a
  PU loss reduces to treating untagged labels as negatives).
- pi'_t = max(pi_t, o_t), the prior the case-control nnPU risk is given: only with pi' >= o does
  the case-control form equal the clamped censoring form (and reduce to the naive loss when
  pi' == o).

Example: pi = 0.389, o = 0.154 gives c = 0.40 and pi_U = 0.278.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

#: HF column name -> Project Sidewalk tag name, for the eight tags the benchmark scores.
TAGS = {
    "missing-tactile-warning": "missing tactile warning",
    "points-into-traffic": "points into traffic",
    "surface-problem": "surface problem",
    "narrow": "narrow",
    "not-enough-landing-space": "not enough landing space",
    "not-level-with-street": "not level with street",
    "steep": "steep",
    "pooled-water": "debris / pooled water",
}
TIER1 = "tier 1: every human label"
TIER2 = "tier 2: crowd-validated correct (correct == true)"
NEAR_M = 10.0


def haversine_m(lat1, lng1, lat2, lng2):
    """Great-circle distance in metres (same formula as tag_benchmark_86.py)."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def pi_unlabeled(pi, o):
    """Positive fraction among untagged labels, clamped at 0.

    >>> round(pi_unlabeled(0.389, 0.169), 3)
    0.265
    >>> pi_unlabeled(0.01, 0.02)
    0.0
    """
    return max(0.0, (pi - o) / (1.0 - o))


def survivors(lab, rs, near_m=NEAR_M):
    """HF train rows left after removing every panorama of either test set and every label of
    the same city within ``near_m`` of any test label. Keys on (city, label_id) = label_uid."""
    m = lab.merge(rs[["label_uid", "split"]].rename(columns={"split": "resplit"}),
                  on="label_uid", how="left", validate="one_to_one")
    assert m.label_uid.is_unique and m.resplit.notna().all()
    test = m[(m.split == "test") | (m.resplit == "test")]
    test_panos = set(test.pano_id.dropna())
    tr = m[m.split == "train"]
    on_test_pano = tr.pano_id.isin(test_panos)
    rest = tr[~on_test_pano].reset_index(drop=True)
    d = np.full(len(rest), np.inf)
    for i, r in rest.iterrows():
        t = test[(test.city == r.city) & test.lat.notna()]
        if len(t) and not np.isnan(r.lat):
            d[i] = float(np.min(haversine_m(r.lat, r.lng, t.lat.values, t.lng.values)))
    near = d <= near_m
    surv = rest[~near]
    # Rows with no live pano have no coordinates: their proximity to the tests is unknown and
    # they cannot be re-cut, so they are counted but not trainable (the benchmark's leak-free
    # subset likewise drops ``pano_unknown``).
    trainable = surv[surv.pano_id.notna()]
    cross = pd.crosstab(m.split, m.resplit)
    return trainable, {
        "hf_rows": int(len(m)),
        "hf_train": int(len(tr)),
        "hf_test": int((m.split == "test").sum()),
        "resplit_test": int((m.resplit == "test").sum()),
        "hf_split_x_resplit": {f"{a}/{b}": int(cross.loc[a, b]) for a in cross.index for b in cross.columns},
        "test_labels_union": int(len(test)),
        "test_panos_union": len(test_panos),
        "hf_train_on_test_pano": int(on_test_pano.sum()),
        "hf_train_within_10m_of_test": int(near.sum()),
        "survivors": int(len(surv)),
        "survivors_without_live_row": int(surv.pano_id.isna().sum()),
        "survivors_trainable": int(len(trainable)),
        "survivor_panos": int(trainable.pano_id.nunique()),
    }


def rates(df):
    return {TAGS[c]: round(float(df[c].mean()), 4) for c in TAGS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="analysis_out/tag_benchmark_86/hf_curbramp_labels.csv")
    p.add_argument("--resplit", default="analysis_out/tag_benchmark_86/resplit_pano_grouped_seed86.csv")
    p.add_argument("--tags", default="analysis_out/ps_audit/tags.csv")
    p.add_argument("--tiers", default="analysis_out/ps_audit/tiers.csv")
    p.add_argument("--out", default="analysis_out/pu_training_86/plan_numbers.json")
    a = p.parse_args()

    lab = pd.read_csv(a.labels)
    rs = pd.read_csv(a.resplit)
    surv, counts = survivors(lab, rs)

    tags = pd.read_csv(a.tags).set_index("tag")
    tiers = pd.read_csv(a.tiers).set_index("tier")
    n1, n2 = int(tiers.loc[TIER1, "tag_era"]), int(tiers.loc[TIER2, "tag_era"])
    o1 = {t: round(tags.loc[t, "labels"] / n1, 4) for t in TAGS.values()}
    o2 = {t: round(tags.loc[t, "labels_correct"] / n2, 4) for t in TAGS.values()}

    priors = {"hf_all_10857": rates(lab), "hf_train": rates(lab[lab.split == "train"]),
              "survivors": rates(surv)}
    pu = {}
    for src in ("survivors", "hf_all_10857"):
        pu[src] = {}
        for t in TAGS.values():
            pi, o = priors[src][t], o2[t]
            pu[src][t] = {"pi": pi, "o_tier2": o, "c": round(o / pi, 3) if pi else None,
                          "pi_U": round(pi_unlabeled(pi, o), 4),
                          "pi_prime": round(max(pi, o), 4),
                          "soft_target_1_minus_c": round(1 - o / pi, 3) if pi else None}

    out = {"inputs": {k: os.path.basename(getattr(a, k)) for k in ("labels", "resplit", "tags", "tiers")},
           "near_m": NEAR_M, "exclusion": counts,
           "tag_era_labels": {"tier1": n1, "tier2": n2},
           "observed_rate_tier1": o1, "observed_rate_tier2": o2,
           "priors": priors, "pu_tier2": pu}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(counts, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
