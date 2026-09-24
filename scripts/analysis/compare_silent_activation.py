"""Compare a re-run of the #46 Phase 1 study against the committed result (#131).

Two result files from ``silent_activation.py --json-out`` -- the committed
``analysis_out/silent_activation.json`` and a replica produced elsewhere -- are compared
at three levels, in the order the issue frames them:

1. **bytes** -- identical files, the strongest statement;
2. **values** -- every row's measured fields equal to the precision the JSON carries
   (``act`` and the nulls are rounded to 5 places, ``null_pct`` to 3, offsets to 1), so
   "same numbers, different bytes" is a distinct outcome from "numbers moved";
3. **what moved** -- per field, how many rows differ, the largest absolute difference,
   and whether any row changed *class* (absent / faint_local / tail) or *bucket*
   (``above_own_null_p95``), which is what the write-up's tables are built from.

Rows are joined on ``(city, pano, x, y)``, never by position. A row present in one file
and not the other is reported, never silently dropped: a replica that scored a different
population is a different study.

    python scripts/analysis/compare_silent_activation.py \\
        analysis_out/silent_activation.json analysis_out/silent_activation_replica.json

Exit status is 0 when the two are byte-identical, 1 when values are identical but bytes
differ, 2 when any value moved, 3 when the populations differ. The printed report is the
result either way; the status is for a script that wants to branch on the outcome.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from silent_activation import class_of  # noqa: E402

#: Measured per-row fields, in the order they are reported. ``group``, ``verdict``,
#: ``field``, ``dist_m``, ``px`` and ``seam`` are read from committed inputs, not the
#: heatmap, so a change there is a change of inputs and is reported under "inputs".
MEASURED = ("act", "null_pct", "null_med", "null_p95", "above_own_null_p95",
            "argmax_off_px", "act_at_site", "nearest_peak_px", "nearest_peak_score")
INPUTS = ("field", "dist_m", "px", "group", "verdict", "seam")
HEADER = ("threshold", "null_trials", "null_seed", "n", "cities", "panos",
          "skipped_no_imagery", "model", "tta")

BYTE_IDENTICAL, VALUES_IDENTICAL, VALUES_MOVED, POPULATIONS_DIFFER = 0, 1, 2, 3


def row_key(r):
    return (r["city"], r["pano"], round(float(r["x"]), 6), round(float(r["y"]), 6))


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), len(raw)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare(a, b):
    """The comparison as data: header diffs, population diffs, per-field movement.

    ``a`` is the reference (the committed file), ``b`` the replica. Returns a dict the
    report prints; ``status`` is one of the four module constants.
    """
    out = {"header": {}, "only_in_a": [], "only_in_b": [], "fields": {}, "inputs": {},
           "class_changes": [], "p95_changes": [], "n_common": 0}
    for k in HEADER:
        va, vb = a.get(k), b.get(k)
        if k == "cities":
            # the scope, not its spelling: the results are sorted by (city, pano) and the
            # null RNG is consumed in that order, so the order --cities was passed in
            # cannot change a number. The committed file lists them alphabetically (the
            # 2026-07-31 run took US_SPLITS from miss_decomposition); the frozen tuple the
            # script carries today starts at richmond.
            va, vb = sorted(va or []), sorted(vb or [])
        if va != vb:
            out["header"][k] = (a.get(k), b.get(k))
    ra = {row_key(r): r for r in a["results"]}
    rb = {row_key(r): r for r in b["results"]}
    out["only_in_a"] = sorted(set(ra) - set(rb))
    out["only_in_b"] = sorted(set(rb) - set(ra))
    common = sorted(set(ra) & set(rb))
    out["n_common"] = len(common)
    for f in MEASURED + INPUTS:
        n_diff, max_abs, worst = 0, 0.0, None
        for k in common:
            va, vb = ra[k].get(f), rb[k].get(f)
            if va == vb:
                continue
            n_diff += 1
            if _num(va) and _num(vb):
                d = abs(va - vb)
                if d > max_abs:
                    max_abs, worst = d, (k, va, vb)
            elif worst is None:
                worst = (k, va, vb)
        if n_diff:
            (out["fields"] if f in MEASURED else out["inputs"])[f] = {
                "n_diff": n_diff, "max_abs": max_abs, "worst": worst}
    for k in common:
        ca, cb = class_of(ra[k]["act"]), class_of(rb[k]["act"])
        if ca != cb:
            out["class_changes"].append((k, ca, cb, ra[k]["act"], rb[k]["act"]))
        if ra[k]["above_own_null_p95"] != rb[k]["above_own_null_p95"]:
            out["p95_changes"].append((k, ra[k]["above_own_null_p95"],
                                       rb[k]["above_own_null_p95"]))
    if out["only_in_a"] or out["only_in_b"] or out["header"]:
        out["status"] = POPULATIONS_DIFFER
    elif out["fields"] or out["inputs"]:
        out["status"] = VALUES_MOVED
    else:
        out["status"] = VALUES_IDENTICAL
    return out


def class_counts(payload):
    counts = {"absent": 0, "faint_local": 0, "tail": 0}
    for r in payload["results"]:
        counts[class_of(r["act"])] += 1
    return counts


def report(a, b, meta_a, meta_b, cmp, byte_identical):
    lines = []
    lines.append(f"reference: {meta_a[0]}  sha256 {meta_a[1][:16]}...  {meta_a[2]:,} bytes")
    lines.append(f"replica:   {meta_b[0]}  sha256 {meta_b[1][:16]}...  {meta_b[2]:,} bytes")
    if byte_identical:
        lines.append("OUTCOME 1: byte-identical.")
        return "\n".join(lines)
    if cmp["header"]:
        lines.append("header differs:")
        for k, (va, vb) in cmp["header"].items():
            lines.append(f"  {k}: {va!r} -> {vb!r}")
    if cmp["only_in_a"] or cmp["only_in_b"]:
        lines.append(f"population differs: {len(cmp['only_in_a'])} rows only in the "
                     f"reference, {len(cmp['only_in_b'])} only in the replica, "
                     f"{cmp['n_common']} in common")
        for k in cmp["only_in_a"][:5]:
            lines.append(f"  only in reference: {k}")
        for k in cmp["only_in_b"][:5]:
            lines.append(f"  only in replica:   {k}")
    if cmp["status"] == POPULATIONS_DIFFER:
        lines.append("OUTCOME: the populations differ -- not the same study; compare inputs first.")
        return "\n".join(lines)
    if cmp["status"] == VALUES_IDENTICAL:
        lines.append(f"OUTCOME 2: every value identical across {cmp['n_common']} rows; "
                     f"bytes differ (whitespace / key order / encoding only).")
        return "\n".join(lines)
    lines.append(f"OUTCOME 3: values moved ({cmp['n_common']} rows compared).")
    lines.append(f"  {'field':>20} {'rows differ':>11} {'max |diff|':>11}  worst row")
    for f in MEASURED:
        d = cmp["fields"].get(f)
        if d is None:
            lines.append(f"  {f:>20} {0:>11} {'-':>11}")
            continue
        k, va, vb = d["worst"]
        lines.append(f"  {f:>20} {d['n_diff']:>11} {d['max_abs']:>11.5g}  "
                     f"{k[0]}/{k[1]} {va!r} -> {vb!r}")
    for f, d in cmp["inputs"].items():
        lines.append(f"  INPUT field {f} differs in {d['n_diff']} rows -- the committed inputs "
                     f"changed between runs, not the heatmap")
    ca, cb = class_counts(a), class_counts(b)
    lines.append(f"  classes (absent / faint_local / tail): reference "
                 f"{ca['absent']} / {ca['faint_local']} / {ca['tail']}, replica "
                 f"{cb['absent']} / {cb['faint_local']} / {cb['tail']}")
    if cmp["class_changes"]:
        for k, x, y, va, vb in cmp["class_changes"]:
            lines.append(f"    class change {k[0]}/{k[1]}: {x} -> {y} (act {va} -> {vb})")
    else:
        lines.append("    no row changed class")
    pa = sum(1 for r in a["results"] if r["above_own_null_p95"])
    pb = sum(1 for r in b["results"] if r["above_own_null_p95"])
    lines.append(f"  above own null p95: reference {pa}, replica {pb}, "
                 f"{len(cmp['p95_changes'])} rows flipped")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("reference", nargs="?",
                   default=os.path.join(REPO, "analysis_out", "silent_activation.json"))
    p.add_argument("replica", nargs="?",
                   default=os.path.join(REPO, "analysis_out", "silent_activation_replica.json"))
    p.add_argument("--json-out", default=None, help="Write the comparison as JSON too")
    args = p.parse_args(argv)
    a, sha_a, n_a = load(args.reference)
    b, sha_b, n_b = load(args.replica)
    byte_identical = sha_a == sha_b
    cmp = compare(a, b)
    if byte_identical:
        cmp["status"] = BYTE_IDENTICAL
    print(report(a, b, (args.reference, sha_a, n_a), (args.replica, sha_b, n_b), cmp,
                 byte_identical))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="") as fh:
            json.dump({"reference": {"path": args.reference, "sha256": sha_a},
                       "replica": {"path": args.replica, "sha256": sha_b},
                       "status": cmp["status"], "comparison": cmp},
                      fh, indent=1, sort_keys=True, default=list)
    return cmp["status"]


if __name__ == "__main__":
    sys.exit(main())
