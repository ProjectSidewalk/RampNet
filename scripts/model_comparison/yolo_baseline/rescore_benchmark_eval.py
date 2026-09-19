"""Regenerate ``benchmark_eval/`` from the committed YOLO detections, with the current scorer.

The 2026-08-14 pano-trio sweep (``benchmark_eval/run_yolo_pano_eval.sh``, ``driver.log``,
``env.txt``) ran the three YOLO pano arms on a GPU and scored them with ``compare.py``.
The detections it produced are committed in ``benchmark/model_detections/`` (#51), so
every number in ``benchmark_eval/<split>.txt`` is a pure function of those files and of
``rampnet.detection_eval`` -- no checkpoint, no GPU, no network. This script is that
function, made explicit:

    python scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py           # rewrite
    python scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py --check   # CI-style
    python scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py --no-wrap-x \
        --out /tmp/pre140                                                             # audit

Why it exists (#148): the #140 seam-wrap fix changed what ``score_pano`` returns for a
detection that lands within the match radius of a ground-truth ramp *across* the x seam.
Three published YOLO cells moved, and ``benchmark_eval/`` kept the pre-fix values for a
month (#140 merged 2026-08-18, this regeneration is 2026-09-19) because nothing re-derived
it. ``tests/test_benchmark_eval.py`` now calls
:func:`render_split` on every CI run and fails if the committed text drifts from what the
current scorer produces, which is the same guard ``docs/model_comparison.md`` already had.

**What is regenerated and what is not.** ``<split>.txt``, ``pr_<split>/pr_<arm>.json`` and
``SUMMARY_TABLES.md`` are derived and are rewritten. ``driver.log``, ``env.txt`` and
``run_yolo_pano_eval.sh`` are the as-run record of the GPU pass that produced the
detections and are never touched. ``pr_<split>/pr_curves.png`` is rewritten only when
matplotlib is importable (``--no-png`` skips it); the test does not compare PNGs.

**The scorer stamp.** Every regenerated file carries a ``Scorer:`` line naming the
matcher's ``wrap_x`` setting and a sha256 over the four source files every number in the
file is a function of (``SCORER_SOURCES``: the matcher in ``rampnet/geometry.py`` and
``rampnet/metrics.py``, the pano scorer and aggregation in ``rampnet/detection_eval.py``,
and ``rampnet/validation.py`` for the Wilson intervals in the 95% CI columns and the
RampNet verdict cross-check block; LF-normalised). The test recomputes that fingerprint,
so a change to any of those files -- behavioural or not -- requires re-running this
script, which is how a committed number always says which scorer produced it. The repo
HEAD at regeneration is recorded beside it as a pointer, not as the identity: the files
land in the *next* commit.

Output is written LF on every platform and every float is rendered through the same
format strings ``compare.py`` prints with, so a regenerated file is byte-comparable.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

import compare as C  # noqa: E402  (torch-free: detectors.py imports torch lazily)
from export_model_cache import load_detections  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_RADIUS_NORMALIZED, aggregate, prediction_confidence, radius_sq_for, score_pano,
)
from rampnet.validation import collect, format_report  # noqa: E402

BENCHMARK_EVAL = os.path.join(HERE, "benchmark_eval")

#: The three pano-geometry arms the 2026-08-14 sweep scored, in the order the driver
#: passed them to ``--models`` (which is the order the tables print in).
ARMS = ("y11l_pano", "y26_pano", "y11x_pano_h200")

#: The ten splits the sweep covered, in driver order, then manual_gold (fetched and
#: scored separately the same day). The two laurens arms were scored later under #151
#: and are reported in docs/, not here.
SPLITS = ("bend", "richmond", "annapolis", "budapest_district5", "clovis", "gainesville",
          "morgantown", "paterson", "sao_paulo", "manual_gold")

#: Pre-registered headline operating point for the supervised arms (#71).
OP_THRESHOLD = 0.25

#: The YOLO cache floor (``--yolo-conf`` default): no detection below it exists, so no
#: sweep row below it is printed. Equals the lowest sweep threshold, so nothing is cut.
YOLO_FLOOR = 0.05

#: The files every number in a stamped file is a function of: the matcher (geometry,
#: metrics), the pano scorer and aggregation (detection_eval), and validation for the
#: Wilson CIs and the verdict cross-check. Hashed, LF-normalised, into the ``Scorer:``
#: stamp so a committed number names the code that produced it.
SCORER_SOURCES = ("rampnet/geometry.py", "rampnet/metrics.py", "rampnet/detection_eval.py",
                  "rampnet/validation.py")

RELATIVE_SCRIPT = "scripts/model_comparison/yolo_baseline/rescore_benchmark_eval.py"


def scorer_fingerprint(repo=REPO):
    """sha256 (first 12 hex) over the LF-normalised bytes of ``SCORER_SOURCES``.

    Normalised so a core.autocrlf=true checkout and a Linux CI runner agree on it.
    """
    h = hashlib.sha256()
    for rel in SCORER_SOURCES:
        with open(os.path.join(repo, *rel.split("/")), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def repo_head(repo=REPO):
    """The HEAD commit this regeneration ran under, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def scorer_line(wrap_x, fingerprint=None):
    """The provenance stamp, without the (regeneration-specific) HEAD pointer."""
    fp = scorer_fingerprint() if fingerprint is None else fingerprint
    return (f"Scorer: rampnet.detection_eval.score_pano wrap_x={wrap_x}  "
            f"scorer-source sha256 {fp}  ({', '.join(SCORER_SOURCES)})")


def load_split(split):
    """(records, {pano_id: GroundTruth}, verdicts-or-None, gt description) for one bundle."""
    bundle = os.path.join(REPO, "benchmark", split)
    records, verdicts, _ = C.load_bundle(bundle)
    if verdicts is not None:
        C.validate_bundle(records, verdicts)
        gts = C.ground_truths_from_verdicts(records, verdicts)
        return records, gts, verdicts, "reviewer-confirmed ramps + missed marks"
    gts = C.load_manual_ground_truths(bundle)
    C.validate_manual_bundle(records, gts)
    gts = {pid: gts[pid] for pid in records if pid in gts}
    return records, gts, None, "independent manual labels (YOLO box centers)"


def _rescore(scored, radius_sq, min_confidence, wrap_x):
    """``compare.rescore`` with the matcher's ``wrap_x`` exposed, for the pre-#140 audit."""
    return aggregate([
        score_pano([p for p in preds
                    if prediction_confidence(p) is None
                    or prediction_confidence(p) >= min_confidence],
                   gt, radius_sq=radius_sq, wrap_x=wrap_x)
        for preds, gt in scored])


def score_arm(arm, split, gts, radius_sq, wrap_x=True):
    """(operating-point report, sweep rows) for one arm on one split, from the committed
    detections. Returns None when no detections are published for the pair."""
    preds = load_detections(arm, split)
    if preds is None:
        return None
    scored = [(preds.get(pid, []), gt) for pid, gt in gts.items()]
    full = _rescore(scored, radius_sq, 0.0, wrap_x)
    # AP and the PR curve are integrals over the whole confidence range; the operating
    # point only truncates P/R/F1 (compare.operating_report).
    op = _rescore(scored, radius_sq, OP_THRESHOLD, wrap_x)._replace(ap=full.ap,
                                                                    pr_curve=full.pr_curve)
    top = max((prediction_confidence(p) for ps, _ in scored for p in ps
               if prediction_confidence(p) is not None), default=0.0)
    sweep = [(t, _rescore(scored, radius_sq, t, wrap_x)) for t in C.SWEEP_THRESHOLDS
             if t <= top and t >= YOLO_FLOOR]
    return op, sweep


def render_split(split, wrap_x=True, fingerprint=None, head=None):
    """The text of ``benchmark_eval/<split>.txt`` plus ``{arm: ScoreReport}`` for the PR
    curves. Pure: committed bundles and detections in, text out.

    ``head`` is the one line the test does not compare; passing ``None`` omits it so the
    rendered text is a function of the data and the scorer alone.
    """
    records, gts, verdicts, gt_desc = load_split(split)
    radius_sq = radius_sq_for(PANO_RADIUS_NORMALIZED)
    rows, sweeps = [], []
    for arm in ARMS:
        res = score_arm(arm, split, gts, radius_sq, wrap_x)
        if res is None:
            continue
        op, sweep = res
        rows.append((arm, op))
        sweeps.append((arm, sweep))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Bundle: benchmark/{split}  ({len(gts)} scored panos)  "
              f"match radius {PANO_RADIUS_NORMALIZED}  ground truth: {gt_desc}")
        print(f"Detections: benchmark/model_detections/<arm>__{split}.json for "
              f"{', '.join(arm for arm, _ in rows)}  (committed; exported from the "
              f"2026-08-14 run's cache -- see driver.log, env.txt, run_yolo_pano_eval.sh)")
        print(scorer_line(wrap_x, fingerprint))
        print(f"Regenerated by: python {RELATIVE_SCRIPT}"
              + (f"  at repo HEAD {head}" if head else "")
              + "  (tests/test_benchmark_eval.py re-derives this file in CI)")
        print()
        print(f"Operating point: predictions with confidence < {OP_THRESHOLD} dropped "
              "(models without confidences are unaffected).")
        C.print_table(rows)
        print("AP: all-point interpolated, over the recall-confirmed panos, from the "
              "full confidence range (--op-threshold does not truncate it); "
              "'-' = no calibrated per-box score.")
        for arm, sweep in sweeps:
            C.print_sweep(arm, sweep)
        print()
        print(f"PR curves: pr_{split}/pr_<arm>.json (+ pr_curves.png)")
        if verdicts is not None:
            # Cross-check: RampNet's own verdict-based P/R (the published definition),
            # exactly as compare.py prints it after every city bundle.
            confs_by_pid = {pid: [d["confidence"] for d in records[pid]["detections"]]
                            for pid in verdicts}
            pools = collect(verdicts, confs_by_pid)
            print()
            print(format_report("RampNet verdict-based cross-check", pools))
            for w in pools.warnings:
                print(f"  ! {w}")
        else:
            print("\nManual-GT bundle: no verdict-based cross-check (validate the rampnet "
                  "row against the published gold-set numbers instead; see "
                  "docs/model_comparison.md).")
    return buf.getvalue(), dict(rows), dict(sweeps)


def render_summary(per_split, wrap_x=True, fingerprint=None, head=None):
    """``SUMMARY_TABLES.md``: the headline and best-sweep-F1 rollups over every split.

    ``per_split``: ``{split: (rows, sweeps)}`` as returned by :func:`render_split`.
    Same two tables the as-run 2026-08-14 rollup had, with the stamp on top.
    """
    out = ["# YOLO pano trio on the benchmark -- rollup of benchmark_eval/<split>.txt", ""]
    out.append(scorer_line(wrap_x, fingerprint))
    out.append(f"Regenerated by: python {RELATIVE_SCRIPT}"
               + (f"  at repo HEAD {head}" if head else "")
               + "  (tests/test_benchmark_eval.py re-derives this file in CI)")
    out += ["", f"## Headline (conf {OP_THRESHOLD}, pre-registered)", "",
            "| bundle | arm | P | R | F1 | AP | tp/fp/fn |", "|---|---|---|---|---|---|---|"]
    for split in SPLITS:
        rows, _ = per_split[split]
        for arm, r in rows.items():
            out.append(f"| {split} | {arm} | {r.precision:.3f} | {r.recall:.3f} | "
                       f"**{r.f1:.3f}** | {r.ap:.3f} | {r.tp}/{r.fp}/{r.fn} |")
    out += ["", "## Best sweep F1 (exploratory, tuned-on-test caveat)", "",
            "| bundle | arm | thr | P | R | F1 |", "|---|---|---|---|---|---|"]
    for split in SPLITS:
        _, sweeps = per_split[split]
        for arm, sweep in sweeps.items():
            t, r = max(sweep, key=lambda tr: tr[1].f1)
            out.append(f"| {split} | {arm} | {t:.2f} | {r.precision:.3f} | {r.recall:.3f} | "
                       f"**{r.f1:.3f}** |")
    return "\n".join(out) + "\n"


def pr_curve_json(arm, report):
    """The text ``compare.write_pr_curves`` writes for one arm: same keys, same
    ``indent=2``, no trailing newline, so the committed files compare byte-for-byte."""
    recalls, precisions = report.pr_curve
    return json.dumps({"model": arm, "ap": report.ap, "n_gt": report.n_gt_recall,
                       "recalls": recalls, "precisions": precisions}, indent=2)


def write_png(path, rows):
    """The combined PR-curve figure, as ``compare.write_pr_curves`` draws it. Skipped
    (with a note) when matplotlib is not importable; never compared by the test."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {os.path.basename(os.path.dirname(path))}: matplotlib not installed, "
              "pr_curves.png left as is")
        return
    plt.figure(figsize=(7, 6))
    for arm, r in rows.items():
        recalls, precisions = r.pr_curve
        plt.plot(recalls, precisions, marker=".", markersize=3, label=f"{arm} (AP {r.ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curb-ramp detection PR curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_lf(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=BENCHMARK_EVAL,
                    help="Directory to write into (default: the committed benchmark_eval/).")
    ap.add_argument("--no-wrap-x", action="store_true",
                    help="Score with the pre-#140 matcher (x not cyclic). For audits only: "
                         "this is how the committed pre-fix files are reproduced.")
    ap.add_argument("--no-png", action="store_true", help="Do not redraw pr_curves.png.")
    ap.add_argument("--check", action="store_true",
                    help="Compare against the files in --out instead of writing; exit 1 "
                         "on any difference (the HEAD pointer line is ignored).")
    args = ap.parse_args(argv)

    wrap_x = not args.no_wrap_x
    fp = scorer_fingerprint()
    head = None if args.check else repo_head()
    per_split, drift = {}, []
    for split in SPLITS:
        text, rows, sweeps = render_split(split, wrap_x, fp, head)
        per_split[split] = (rows, sweeps)
        targets = {os.path.join(args.out, f"{split}.txt"): text}
        for arm, r in rows.items():
            targets[os.path.join(args.out, f"pr_{split}", f"pr_{arm}.json")] = \
                pr_curve_json(arm, r)
        if args.check:
            drift += _check(targets)
            continue
        # The figure is a function of the JSON curves: redraw it only when one of them
        # actually changed (or it is missing), so a regeneration that moves nothing does
        # not churn ten binary files.
        png = os.path.join(args.out, f"pr_{split}", "pr_curves.png")
        curves_changed = _check({p: b for p, b in targets.items() if p.endswith(".json")})
        for path, body in targets.items():
            write_lf(path, body)
        if not args.no_png and (curves_changed or not os.path.exists(png)):
            write_png(png, rows)
        print(f"wrote {split}: {', '.join(f'{a} F1 {r.f1:.3f}' for a, r in rows.items())}")

    summary = render_summary(per_split, wrap_x, fp, head)
    summary_path = os.path.join(args.out, "SUMMARY_TABLES.md")
    if args.check:
        drift += _check({summary_path: summary})
        if drift:
            print("benchmark_eval/ is stale against the current scorer:")
            for line in drift:
                print("  " + line)
            return 1
        print(f"benchmark_eval/ matches the current scorer (fingerprint {fp}).")
        return 0
    write_lf(summary_path, summary)
    print(f"wrote SUMMARY_TABLES.md; scorer fingerprint {fp}, HEAD {head}")
    return 0


def strip_head_pointer(text):
    """Drop the ``at repo HEAD <sha>`` token so two regenerations compare on content."""
    out = []
    for line in text.split("\n"):
        if line.startswith("Regenerated by:") and "  at repo HEAD " in line:
            before, after = line.split("  at repo HEAD ", 1)
            line = before + after[after.index("  "):] if "  " in after else before
        out.append(line)
    return "\n".join(out)


def _display_path(path):
    """Repo-relative for the drift message when the path is under the repo, else absolute.

    ``os.path.relpath`` raises on Windows when ``--out`` is on a different drive from the
    checkout (``C:`` scratch vs ``D:`` repo), which took the documented ``--no-wrap-x
    --out <dir>`` audit down with it."""
    try:
        rel = os.path.relpath(path, REPO)
    except ValueError:
        return os.path.normpath(path)
    if rel.startswith(os.pardir):
        return os.path.normpath(path)
    return rel.replace(os.sep, "/")


def _check(targets):
    """Which of ``targets`` (path -> expected text) differ from what is on disk."""
    drift = []
    for path, body in targets.items():
        rel = _display_path(path)
        if not os.path.exists(path):
            drift.append(f"{rel}: missing")
            continue
        with open(path, "rb") as fh:
            have = fh.read().replace(b"\r\n", b"\n").decode("utf-8")
        if strip_head_pointer(have) != strip_head_pointer(body):
            drift.append(f"{rel}: differs")
    return drift


if __name__ == "__main__":
    sys.exit(main())
