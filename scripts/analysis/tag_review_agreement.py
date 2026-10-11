"""Per-tag agreement between two raters' tag review exports (#86 item 3).

    python scripts/analysis/tag_review_agreement.py \\
        benchmark/tag_review/jonfroehlich.json benchmark/tag_review/mikey.json \\
        --by tag_state --json analysis_out/tag_review/agreement.json

Prints, per tag: items compared, each rater's positives and rate, pooled prevalence, raw
agreement, positive-specific agreement (2 x both / (pos_a + pos_b), the number that stays
readable when a tag is rare and kappa is unstable), and Cohen's kappa; then severity
agreement as quadratic- and linear-weighted kappa on the 1 / 2 / 3 scale; then the
item-level vote cross-table (agree / disagree / unsure).

Also prints each export's route (``prod_pull`` / ``review_sheet``) and warns when they
differ, the same per-tag table without the items either rater had prior contact with
(``prior_contact_<rater>`` in the list), and how many judgeable items each rater left
without a severity.

Refuses to compare exports made against different review lists, or under different rubric
text (``--allow-rubric-mismatch`` overrides the second and says so in the output). The
comparison rules (which items enter a tag's table) are in ``rampnet.tag_review.agreement``.

**Kappa depends on prevalence, and the list is enriched for rare tags** (see
``tag_review_list.py``). The kappas are agreement on this list, the human ceiling a model
is compared against on the same items; they are not population tag rates.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rampnet import tag_review as tr  # noqa: E402


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def tag_table(rows, a, b):
    out = [f"| tag | n | pos {a} | pos {b} | prevalence | agree | pos-specific agree | kappa | cannot judge {a} / {b} |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['tag']} | {r['n']} | {r['pos_a']} | {r['pos_b']} | {_f(r['prevalence'])} | "
                   f"{_f(r['pct_agree'])} | {_f(r['pos_specific_agree'])} | {_f(r['kappa'])} | "
                   f"{r['cannot_judge_a']} / {r['cannot_judge_b']} |")
    return "\n".join(out)


def render(rep):
    a, b = rep["rater_a"], rep["rater_b"]
    lines = [f"# Tag review agreement: {a} vs {b}", ""]
    rub = rep["rubric"]
    lines.append(f"Rubric: {a} `{rub['a']}`, {b} `{rub['b']}`"
                 + ("" if rub["same_text"] else "  **(DIFFERENT rubric text, compared with --allow-rubric-mismatch)**"))
    lines.append(f"Review list sha256: `{rep['review_list_sha256']}`")
    m = rep.get("method", {})
    lines.append(f"Route: {a} `{m.get('a')}`, {b} `{m.get('b')}`")
    for w in rep.get("warnings", []):
        lines.append(f"**WARNING:** {w}")
    it = rep["items"]
    lines.append(f"Items: {it['common']} in both files, {it['both_reviewed']} reviewed by both, "
                 f"{it['both_judgeable']} judgeable by both (not cannot-judge, not voted 'not a curb ramp', "
                 f"no recorded problem); whole-item cannot-judge {a} {it['cannot_judge_a']}, {b} {it['cannot_judge_b']}.")
    lines += ["", "## Per tag", "", tag_table(rep["per_tag"], a, b), ""]
    pc = rep.get("prior_contact")
    if pc is not None:
        lines += [f"## Per tag, without prior-contact items ({pc['items_without']} judgeable items; "
                  f"{pc['items_with_contact']} excluded because either rater placed, validated or edited the label "
                  "before the list was built)", "", tag_table(rep["per_tag_without_prior_contact"], a, b), ""]
    s = rep["severity"]
    lines += ["## Severity (1 / 2 / 3)", "",
              f"n {s['n']}, quadratic-weighted kappa {_f(s['weighted_kappa_quadratic'])}, "
              f"linear-weighted kappa {_f(s['weighted_kappa_linear'])}, exact agreement {_f(s['exact_agree'])}, "
              f"mean {a} {_f(s['mean_a'], 2)} vs {b} {_f(s['mean_b'], 2)}; "
              f"judgeable items with no severity: {a} {s.get('missing_a', 'n/a')}, {b} {s.get('missing_b', 'n/a')}", "",
              f"| {a} \\ {b} | 1 | 2 | 3 |", "|---|---:|---:|---:|"]
    for u in tr.SEVERITIES:
        lines.append(f"| {u} | " + " | ".join(str(s["confusion"][f"{u}|{v}"]) for v in tr.SEVERITIES) + " |")
    lines += ["", "## Item votes", "", f"| {a} \\| {b} | items |", "|---|---:|"]
    for k, n in rep["verdicts"].items():
        lines.append(f"| {k} | {n} |")
    for key in [k for k in rep if k.startswith("by_")]:
        lines += ["", f"## Per tag, {key[3:]}"]
        for g, rows in rep[key].items():
            lines += ["", f"### {key[3:]} = {g}", "", tag_table(rows, a, b)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rater_a")
    ap.add_argument("rater_b")
    ap.add_argument("--by", choices=["tag_state", "distance_band"], default=None)
    ap.add_argument("--allow-rubric-mismatch", action="store_true")
    ap.add_argument("--json", default=None, help="also write the full report here")
    ap.add_argument("--md", default=None, help="also write the Markdown report here")
    args = ap.parse_args(argv)
    rep = tr.agreement(tr.read_json(args.rater_a), tr.read_json(args.rater_b), group_by=args.by,
                       allow_rubric_mismatch=args.allow_rubric_mismatch)
    text = render(rep)
    print(text)
    for w in rep.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
    for path, write in ((args.json, lambda p: tr.write_json(p, rep)),
                        (args.md, lambda p: open(p, "w", encoding="utf-8", newline="").write(text))):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            write(path)
    return rep


if __name__ == "__main__":
    main()
