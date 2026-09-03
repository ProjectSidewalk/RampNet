#!/usr/bin/env python
"""Apply #135's pre-registered Run B gate to the 8-epoch cosine rung's results.

The gate is written in ``docs/stage2_cosine_rung_135.md`` under "Pre-registered decision
rule", before the rung was launched. This script evaluates it rather than restating it,
so the decision is derived from the committed numbers and re-derivable by anyone:

    primary    Delta = max-F1(cosine, ep8) - max-F1(Run A, ep8), paired.
    secondary  each arm's own ep3 -> ep8 change; is the cosine arm's decline arrested
               (>= 0) or significantly smaller than Run A's?

    PROCEED         primary positive-and-significant, OR primary tie and secondary
                    confirms the decline is arrested.
    DO NOT PROCEED  primary significantly negative AND secondary shows no arrest.
    JUDGMENT CALL   anything else -- explicitly NOT an automatic cancellation.

CPU only, no network: it reads two committed summary CSVs and the committed power JSON.

WHY THE STANDARD ERROR IS A BRACKET AND NOT A NUMBER
A true paired bootstrap between the two arms needs both arms' PER-PANORAMA detections.
Run A's are committed (docs/data/run_a_84_detections/); the cosine arm's are not -- only
its downsampled PR curves are, because the full curves are ~4 MB x 8. So this uses the
paired standard error #138 MEASURED across 28 Run A epoch pairs on the same 1,000
panoramas and the same GT, as a bracket, and reports z at both ends.

That is weaker than a direct bootstrap and it is stated rather than hidden -- but it is
sufficient here, because the primary comparison fails to reach significance at the
FAVOURABLE end of the bracket, so the reading does not depend on which value is picked.

Usage:
    python scripts/analysis/run_b_gate_135.py
    python scripts/analysis/run_b_gate_135.py --check   # fails if the artifact drifted
"""

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN_A = REPO / "docs" / "data" / "run_a_84_manual_gold" / "summary.csv"
COSINE = REPO / "docs" / "data" / "cosine_rung_135_manual_gold" / "summary.csv"
POWER = REPO / "docs" / "data" / "benchmark_power_135.json"
ARTIFACT = REPO / "docs" / "data" / "run_b_gate_135.json"

PRIMARY_EPOCH = 8
SECONDARY_FROM = 3   # Run A's own post-plateau peak, per the pre-registration
SECONDARY_TO = 8


def _max_f1_by_epoch(path):
    with path.open(newline="") as fh:
        return {int(r["epoch"]): float(r["max_f1"]) for r in csv.DictReader(fh)}


def _z_range(delta, se_lo, se_hi):
    """z at both ends of the s.e. bracket. Smaller s.e. => larger |z|, so se_lo is the
    favourable end for declaring significance."""
    return abs(delta) / se_hi, abs(delta) / se_lo


def evaluate():
    run_a = _max_f1_by_epoch(RUN_A)
    cosine = _max_f1_by_epoch(COSINE)
    power = json.loads(POWER.read_text())

    se_lo, se_hi = power["run_a_paired"]["max_f1_se_bracket"]
    z_crit = power["z_ci"]

    # --- primary --------------------------------------------------------------
    primary_delta = cosine[PRIMARY_EPOCH] - run_a[PRIMARY_EPOCH]
    p_z_min, p_z_max = _z_range(primary_delta, se_lo, se_hi)
    primary_significant = p_z_max >= z_crit
    primary_positive = primary_delta > 0

    # --- secondary ------------------------------------------------------------
    run_a_decline = run_a[SECONDARY_TO] - run_a[SECONDARY_FROM]
    cosine_decline = cosine[SECONDARY_TO] - cosine[SECONDARY_FROM]
    arrested = cosine_decline >= 0
    # difference of declines, i.e. is the cosine arm's decline SIGNIFICANTLY smaller
    diff_of_declines = cosine_decline - run_a_decline
    d_z_min, d_z_max = _z_range(diff_of_declines, se_lo, se_hi)
    smaller_significantly = diff_of_declines > 0 and d_z_max >= z_crit
    secondary_confirms = arrested or smaller_significantly

    # --- the gate -------------------------------------------------------------
    if (primary_significant and primary_positive) or \
            (not primary_significant and secondary_confirms):
        verdict = "PROCEED"
    elif primary_significant and not primary_positive and not secondary_confirms:
        verdict = "DO NOT PROCEED"
    else:
        verdict = "JUDGMENT CALL"

    # --- the post-hoc trap, reported so nobody has to rediscover it ------------
    # Epoch 7 is the largest |Delta| anywhere and WOULD clear 1.96 at the favourable
    # end of the bracket. It is not the pre-registered comparison, and picking it after
    # seeing the table is exactly what pre-registration exists to prevent.
    per_epoch = []
    for ep in sorted(set(run_a) & set(cosine)):
        d = cosine[ep] - run_a[ep]
        zl, zh = _z_range(d, se_lo, se_hi)
        per_epoch.append({
            "epoch": ep,
            "delta": round(d, 6),
            "z_at_se_hi": round(zl, 3),
            "z_at_se_lo": round(zh, 3),
            "would_reach_significance_at_favourable_se": bool(zh >= z_crit),
        })
    largest = max(per_epoch, key=lambda r: abs(r["delta"]))

    return {
        "gate_source": "docs/stage2_cosine_rung_135.md, Pre-registered decision rule",
        "se_bracket": [round(se_lo, 6), round(se_hi, 6)],
        "se_source": power["run_a_paired"]["max_f1_se_source"],
        "z_critical": round(z_crit, 6),
        "primary": {
            "epoch": PRIMARY_EPOCH,
            "run_a": round(run_a[PRIMARY_EPOCH], 6),
            "cosine": round(cosine[PRIMARY_EPOCH], 6),
            "delta": round(primary_delta, 6),
            "z_at_se_hi": round(p_z_min, 3),
            "z_at_se_lo": round(p_z_max, 3),
            "significant": bool(primary_significant),
            "reading": "annealing helps" if (primary_significant and primary_positive)
            else ("annealing hurts at this budget"
                  if primary_significant else "no effect at 8 epochs"),
        },
        "secondary": {
            "from_epoch": SECONDARY_FROM,
            "to_epoch": SECONDARY_TO,
            "run_a_change": round(run_a_decline, 6),
            "cosine_change": round(cosine_decline, 6),
            "arrested": bool(arrested),
            "difference_of_declines": round(diff_of_declines, 6),
            "z_at_se_hi": round(d_z_min, 3),
            "z_at_se_lo": round(d_z_max, 3),
            "significantly_smaller": bool(smaller_significantly),
            "confirms_arrest": bool(secondary_confirms),
        },
        "per_epoch": per_epoch,
        "largest_absolute_delta": largest,
        "verdict": verdict,
    }


def _render(r):
    out = []
    out.append("#135 Run B gate, applied to the 8-epoch cosine rung")
    out.append("=" * 62)
    se = r["se_bracket"]
    out.append(f"paired s.e. bracket {se[0]:.6f}-{se[1]:.6f}  ({r['se_source']})")
    out.append(f"significance threshold |z| >= {r['z_critical']:.3f}")
    out.append("")
    p = r["primary"]
    out.append(f"PRIMARY   max-F1(cosine ep{p['epoch']}) - max-F1(Run A ep{p['epoch']})")
    out.append(f"          {p['cosine']:.6f} - {p['run_a']:.6f} = {p['delta']:+.6f}")
    out.append(f"          |z| = {p['z_at_se_hi']:.2f} to {p['z_at_se_lo']:.2f}"
               f"  -> {'SIGNIFICANT' if p['significant'] else 'not significant'}")
    out.append(f"          reading: {p['reading']}")
    out.append("")
    s = r["secondary"]
    out.append(f"SECONDARY ep{s['from_epoch']} -> ep{s['to_epoch']} change per arm")
    out.append(f"          Run A  {s['run_a_change']:+.6f}")
    out.append(f"          cosine {s['cosine_change']:+.6f}"
               f"   arrested (>=0)? {'yes' if s['arrested'] else 'NO'}")
    out.append(f"          difference of declines {s['difference_of_declines']:+.6f}, "
               f"|z| = {s['z_at_se_hi']:.2f} to {s['z_at_se_lo']:.2f}"
               f"  -> {'significantly smaller' if s['significantly_smaller'] else 'not significant'}")
    out.append(f"          confirms arrest? {'yes' if s['confirms_arrest'] else 'NO'}")
    out.append("")
    lg = r["largest_absolute_delta"]
    out.append(f"largest |delta| anywhere: epoch {lg['epoch']}, {lg['delta']:+.6f}, "
               f"|z| up to {lg['z_at_se_lo']:.2f}")
    if lg["would_reach_significance_at_favourable_se"] and lg["epoch"] != PRIMARY_EPOCH:
        out.append(f"  NOTE: epoch {lg['epoch']} would clear the threshold at the favourable end of the")
        out.append("  bracket, but it is NOT the pre-registered comparison. Reading it as the")
        out.append("  result is the post-hoc selection the pre-registration exists to prevent.")
    out.append("")
    out.append(f"GATE VERDICT: {r['verdict']}")
    if r["verdict"] == "JUDGMENT CALL":
        out.append("  The pre-registration says a tie on both is NOT an automatic cancellation:")
        out.append("  it is a judgment call about spending ~1,800 GPU-hours on a mechanism that")
        out.append("  showed nothing at a quarter of the cost, to be recorded as a judgment call")
        out.append("  rather than dressed as a rule. See docs/stage2_cosine_rung_135.md.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed artifact and exit non-zero on drift")
    ap.add_argument("--write", action="store_true", help="rewrite the committed artifact")
    args = ap.parse_args(argv)

    result = evaluate()
    print(_render(result))

    # newline="" so the committed bytes are LF on every platform, and every float is
    # rounded above so a different numpy/py build cannot re-render them differently.
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"

    if args.write:
        with ARTIFACT.open("w", newline="") as fh:
            fh.write(payload)
        print(f"\nwrote {ARTIFACT.relative_to(REPO)}")
        return 0

    if args.check:
        if not ARTIFACT.exists():
            print(f"\nMISSING {ARTIFACT.relative_to(REPO)}", file=sys.stderr)
            return 1
        if ARTIFACT.read_bytes() != payload.encode():
            print(f"\nDRIFT: {ARTIFACT.relative_to(REPO)} does not match a fresh run",
                  file=sys.stderr)
            return 1
        print(f"\nOK: {ARTIFACT.relative_to(REPO)} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
