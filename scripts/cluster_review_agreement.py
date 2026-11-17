"""Summarise cluster-review passes and compare two of them (issue #224).

Validates every assignments*.json in a bundle against its snapshot and corners, prints
each pass's summary, and -- given two files -- the pre-registered agreement
(docs/cluster_review_protocol.md, "Inter-rater agreement" as amended by Amendment 5):
same-ramp pair F1 (pooled and per-unit mean, unit-level bootstrap CIs; overall and by
same / different seed), the Rand-style pairwise rate (reported only), kappa on not_ramp,
uncovered-point counts, can't-judge counts, each unit's washout, and the pilot gate.

Which kind of agreement it is goes into the headline and the JSON (`agreement_kind`):
``--kind intra`` for the same rater's delayed re-review (Amendment 2: test-retest
consistency, never "inter-rater reliability"), ``--kind inter`` for two people. It is
inferred as intra when the B file's rater is ``<name>-retest``; otherwise ``--kind`` is
required.

    python scripts/cluster_review_agreement.py benchmark/vancouver/cluster_review
    python scripts/cluster_review_agreement.py benchmark/vancouver/cluster_review \
        --a assignments.json --b assignments__jon-retest.json [--pilot] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rampnet import cluster_review as cr  # noqa: E402

HEADLINE = {"intra": "INTRA-rater agreement (the same rater's delayed retest; not inter-rater "
                     "reliability)",
            "inter": "INTER-rater agreement (two raters)"}


def only(a, ids):
    return dict(a, corners={k: v for k, v in (a.get("corners") or {}).items() if k in ids})


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def ci(c):
    return "" if not c else f" (95% CI {c[0]:.3f}-{c[1]:.3f})"


def f1_line(tag, r):
    return (f"     {tag}: pooled {f3(r['pooled'])}{ci(r['pooled_ci95'])}, per-unit mean "
            f"{f3(r['per_unit_mean'])}{ci(r['per_unit_mean_ci95'])} over {r['units']} units "
            f"({r['units_defined']} with a same-ramp pair)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--a", default="assignments.json", help="rater A's file (in the bundle)")
    ap.add_argument("--b", default=None, help="the other pass (in the bundle)")
    ap.add_argument("--kind", choices=cr.AGREEMENT_KINDS, default=None,
                    help="intra = the same rater's delayed retest, inter = two raters; "
                         "required with --b unless B's rater is <name>-retest")
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
        kind = args.kind or cr.infer_kind(files[args.a], files[args.b])
        if kind is None:
            raise SystemExit(f"--kind intra|inter is required: {args.b}'s rater "
                             f"{files[args.b].get('rater')!r} is not <name>-retest, so this "
                             "script cannot tell a retest from a second rater")
        report = cr.agreement(only(files[args.a], ids), only(files[args.b], ids), corners,
                              kind=kind)
        pw, k = report["pairwise"], report["not_ramp_kappa"]
        print(f"== {HEADLINE[kind]}: {args.a} vs {args.b}: {report['units']['both']} units "
              "both complete")
        print("   same-ramp pair F1:")
        print(f1_line("all", report["pair_f1"]))
        for tag in ("same_seed", "different_seed"):
            print(f1_line(tag, report[f"pair_f1_{tag}"]))
        print(f"   pairwise (Rand-style, reported only, not a gate): {pw['agree']}/{pw['pairs']}"
              + (f" = {pw['rate']:.3f}" if pw['pairs'] else ""))
        for tag in ("same_seed", "different_seed"):
            r = report[f"pairwise_{tag}"]
            print(f"     {tag}: {r['agree']}/{r['pairs']}"
                  + (f" = {r['rate']:.3f}" if r['pairs'] else ""))
        print(f"   kappa(not_ramp): {k['kappa']} over {k['labels']} labels "
              f"(not_ramp A {k['not_ramp_a']}, B {k['not_ramp_b']})"
              + ("; undefined because both passes are constant and identical"
                 if k["undefined_identical"] else ""))
        print(f"   uncovered: A {report['uncovered']['total_a']}, B {report['uncovered']['total_b']}, "
              f"|diff| per unit {report['uncovered']['abs_diff_per_unit']}")
        cj = report["cant_judge"]
        print(f"   can't judge: A {cj['a']}, B {cj['b']}, either {cj['either']}, both {cj['both']}; "
              f"complete in A but can't-judge in B {cj['complete_a_cant_judge_b']}, "
              f"the reverse {cj['cant_judge_a_complete_b']}")
        w = report["washout"]
        print(f"   washout ({w['basis']}): {w['units_with_times']} units timed, shortest "
              f"{w['shortest_days']} d; UNDER {w['min_days']:g} d: {w['under_min'] or 'none'}; "
              f"no time: {len(w['missing'])}")
        if w['under_min']:
            print(f1_line('pair F1 without them', w['pair_f1_without_under_min']))
        p = report["pilot"]
        print(f"   pilot gate ({p['rule']}): {p['pass']}"
              + (f" -- {p['why_none']}" if p.get("why_none") else ""))
        if "pair_f1_different_seed" in p:
            print(f1_line("pilot, different seed", p["pair_f1_different_seed"]))
            wo = p["pair_f1_different_seed_without_under_min"]
            if wo["units"] != p["pair_f1_different_seed"]["units"]:
                print(f1_line(f"pilot, different seed, without units under {w['min_days']:g} d "
                              "(reported, not gated)", wo))
            ref = p["seed_reference"]
            print(f"     seed reference (deployed vs fusion seeds on those units): pooled "
                  f"{f3(ref['pooled'])}, per-unit mean {f3(ref['per_unit_mean'])}")
    if args.json:
        args.json.write_text(json.dumps({"files": {n: cr.summary(only(a, ids)) for n, a in files.items()},
                                         "agreement": report}, indent=1) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
