"""Summarise cluster-review passes and compare two of them (issue #224).

Validates every assignments*.json in a bundle against its snapshot and corners, prints
each pass's summary, and -- given two files -- the pre-registered inter-rater agreement
(docs/cluster_review_protocol.md, "Inter-rater agreement"): pairwise same-ramp agreement,
kappa on not_ramp, uncovered-point counts, and the pilot pass/fail rule.

    python scripts/cluster_review_agreement.py benchmark/vancouver/cluster_review
    python scripts/cluster_review_agreement.py benchmark/vancouver/cluster_review \
        --a assignments.json --b assignments__mikey.json [--pilot] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rampnet import cluster_review as cr  # noqa: E402


def only(a, ids):
    return dict(a, corners={k: v for k, v in (a.get("corners") or {}).items() if k in ids})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--a", default="assignments.json", help="rater A's file (in the bundle)")
    ap.add_argument("--b", default=None, help="rater B's file (in the bundle)")
    ap.add_argument("--pilot", action="store_true", help="restrict to pilot units")
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args(argv)
    snapshot, corners, files = cr.load_bundle(args.bundle)
    if not files:
        print(f"{args.bundle}: no assignments*.json yet -- nothing has been reviewed.")
        return 0
    ids = {c["corner_id"] for c in corners if c.get("pilot")} if args.pilot else \
        {c["corner_id"] for c in corners}
    bad = False
    for name, a in files.items():
        problems = cr.validate(a, corners, snapshot)
        print(f"== {name}: {'valid' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems[:20]:
            print(f"   {p}")
        bad |= bool(problems)
        print(json.dumps(cr.summary(only(a, ids)), indent=1))
    report = None
    if args.b:
        for n in (args.a, args.b):
            if n not in files:
                raise SystemExit(f"{args.bundle / n} does not exist")
        report = cr.agreement(only(files[args.a], ids), only(files[args.b], ids), corners)
        pw, k = report["pairwise"], report["not_ramp_kappa"]
        print(f"== agreement {args.a} vs {args.b}: {report['units']['both']} units both complete")
        print(f"   pairwise same-ramp: {pw['agree']}/{pw['pairs']}"
              + (f" = {pw['rate']:.3f} (95% CI {pw['ci95'][0]:.3f}-{pw['ci95'][1]:.3f})"
                 if pw['pairs'] else ""))
        for tag in ("same_seed", "different_seed"):
            r = report[f"pairwise_{tag}"]
            print(f"     {tag}: {r['agree']}/{r['pairs']}"
                  + (f" = {r['rate']:.3f}" if r['pairs'] else ""))
        print(f"   kappa(not_ramp): {k['kappa']} over {k['labels']} labels "
              f"(not_ramp A {k['not_ramp_a']}, B {k['not_ramp_b']})")
        print(f"   uncovered: A {report['uncovered']['total_a']}, B {report['uncovered']['total_b']}, "
              f"|diff| per unit {report['uncovered']['abs_diff_per_unit']}")
        cj = report["cant_judge"]
        print(f"   can't judge: A {cj['a']}, B {cj['b']}, either {cj['either']}, both {cj['both']}; "
              f"complete in A but can't-judge in B {cj['complete_a_cant_judge_b']}, "
              f"the reverse {cj['cant_judge_a_complete_b']}")
        print(f"   pilot rule ({report['pilot']['rule']}): {report['pilot']['pass']}")
    if args.json:
        args.json.write_text(json.dumps({"files": {n: cr.summary(only(a, ids)) for n, a in files.items()},
                                         "agreement": report}, indent=1) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
