"""Re-derive the #285 decision numbers from committed files and compare them with the committed outputs (RampNet#285).

Every number in docs/crosswalk_followups_285.md that decided something, and that needs no GPU, imagery or Stage 2
root, regenerates in seconds on CPU from files in the repo. This re-runs those steps into a temporary directory and
compares each output with the committed copy, byte for byte:

* the `crosswalk_s6` primary and guard 6 (`recall_check.py compare`): the s4 calibration and the full-dose,
  half-dose and step-control comparisons, at the matched thresholds the doc reports;
* guards 4 and 5 (`score_274.py retention`): Seattle and the fresh Juneau draw, for the calibration and each arm;
* guard 3 (`score_274.py label-recall`): Columbia labels found, seed 1 of each arm against s4;
* section C, s4 on the missed-crosswalk marks (`recall_check.py score`) and its region rule
  (`recall_region.py score`, from the committed segmenter pairs);
* section B, the car-top rule (`cartop_mask.py score`);
* the A3 precision gates of Sacramento, Kaysville and Brussels (`osm_arm.py city-score`).

Not covered here: guards 1 and 2 (`score_274.py score` and `bprime-score` need the Stage 2 root and the Seattle pano
data, which are not in the repo), the A4 follow-up sheets (they need imagery; `tests/test_pos_sheet_285.py`
re-derives follow-up 2's summary from the committed tags, and `tests/test_fa_sheet_285.py` pins follow-up 1's
spot counts), and every GPU step (inference, placement, the segmenter pairs).

    python scripts/analysis/stage1_crosswalk_sdot_274/check_285.py --check            # exit 1 on any difference
    python scripts/analysis/stage1_crosswalk_sdot_274/check_285.py --check --keep DIR # keep the regenerated files

It is the `crosswalk_followups_285` entry of scripts/check_all.py (in --ci).
"""
import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
S = os.path.join(HERE, "score_274.py")
D = "analysis_out/stage1_crosswalk_detector_274"
N = f"{D}/hardneg"
F = "analysis_out/crosswalk_followups_285"
R = f"{F}/recall"
VERDICTS = f"{R}/ratings/recall285_verdicts_jon_2026-11-11.json"
CITIES = ("laurens-ia", "juneau-ak")

# The thresholds the doc reports (A4 result): s4's seeds at 0.23, 0.30 and 0.13, and each new model at the
# threshold matched to its s4 seed's recall on the val split (s6_score.sh's match-threshold).
S4 = (("crosswalk_s4", 0.23), ("crosswalk_s4_seed2", 0.3), ("crosswalk_s4_seed3", 0.13))
ARMS = {"full": (("crosswalk_s6full", 0.17), ("crosswalk_s6full_seed2", 0.27), ("crosswalk_s6full_seed3", 0.24)),
        "half": (("crosswalk_s6half", 0.23), ("crosswalk_s6half_seed2", 0.23), ("crosswalk_s6half_seed3", 0.25)),
        "steps": (("crosswalk_s4_steps1397", 0.24),)}
A3 = (("sacramento-ca", "ratings_a3_sacramento_jon_2026-11-11.csv"),
      ("kaysville-ut", "ratings_a3_kaysville_jon_2026-11-12.csv"),
      ("brussels-be", "ratings_a3_brussels_jon_2026-11-12.csv"))


def at(label, thr):
    return f"{label}@{thr}"


def recall_base(arm):
    """compare's base side: s4 seed 1 is the klone run (crosswalk_s4_klone), the step control has seed 1 only."""
    seeds = [("crosswalk_s4_klone", 0.23)] + list(S4[1:])
    return seeds[:1] if arm == "steps" else seeds


def score_274_steps():
    """[(argv for score_274.py with {T} for the temp root, [committed outputs it must reproduce])]."""
    out = []
    bp = lambda m: f"{D}/v2_labels/bprime/preds/{m}/bprime.jsonl"                    # noqa: E731
    je = lambda m: (f"{N}/s5_eval/juneau_eval/preds/{m}/juneau_eval.jsonl" if m in dict(S4)  # noqa: E731
                    else f"{N}/s6_eval/juneau_eval/preds/{m}/juneau_eval.jsonl")
    for arm, models in [("calibration", S4[1:])] + list(ARMS.items()):
        for draw, ratings, path in (("seattle", f"{N}/confirm_s3/seattle/s4/ratings.csv", bp),
                                    ("juneau_eval", f"{N}/s5_inputs/juneau_eval/s4/ratings.csv", je)):
            specs = [f"crosswalk_s4={path('crosswalk_s4')}@0.23"] + [f"{m}={path(m)}@{t}" for m, t in models]
            out.append((["retention", "--ratings", ratings, "--models", *specs, "--out", f"{{T}}/{F}/s6/{arm}/{draw}"],
                        [f"{F}/s6/{arm}/{draw}/retention.json"]))
    for arm, models in ARMS.items():
        m, t = models[0]
        out.append((["label-recall", "--labels", f"{N}/s4_inputs/stepA/columbia_gt.csv",
                     "--base", f"crosswalk_s4={N}/s4_eval/columbia/preds/crosswalk_s4/columbia.jsonl@0.23",
                     "--new", f"{m}={N}/s6_eval/columbia/preds/{m}/columbia.jsonl@{t}",
                     "--out", f"{{T}}/{N}/s6_eval/columbia/{arm}"],
                    [f"{N}/s6_eval/columbia/{arm}/label_hits.csv", f"{N}/s6_eval/columbia/{arm}/label_recall.json"]))
    return out


def module_steps():
    """[(module name, {constant: repo-relative dir it is pointed at under {T}}, [files to copy in], argv, outputs)].

    These scripts read and write under a fixed directory (their OUT), so the check points OUT at a copy of the
    inputs in the temporary root and runs them in this process."""
    rv = f"{{T}}/{R}"
    copy_recall = [f"{R}/{c}/intersections.csv" for c in CITIES] + [f"{R}/preds/"]
    out = [("recall_check", {"OUT": R}, copy_recall,
            ["score", "--verdicts", VERDICTS, "--cities", *CITIES, "--label", "crosswalk_s4", "--threshold", "0.23",
             "--preds", *[f"{rv}/preds/crosswalk_s4/{c}.jsonl" for c in CITIES]],
            [f"{R}/score/crosswalk_s4/marks.csv", f"{R}/score/crosswalk_s4/recall.json"])]
    for arm in ("calibration",) + tuple(ARMS):
        new = [] if arm == "calibration" else ["--new", *[at(m, t) for m, t in ARMS[arm]]]
        label = "s4_calibration" if arm == "calibration" else f"s6_{arm}"
        out.append(("recall_check", {"OUT": R}, copy_recall,
                    ["compare", "--verdicts", VERDICTS, "--cities", *CITIES,
                     "--base", *[at(m, t) for m, t in recall_base(arm)], *new, "--out-label", label],
                    [f"{R}/compare/{label}/compare.json"]))
    out.append(("recall_region", {"rc.OUT": R},
                [f"{R}/score/crosswalk_s4/region_marks_raw.csv", f"{R}/score/crosswalk_s4/region_pairs.csv"],
                ["score", "--label", "crosswalk_s4"],
                [f"{R}/score/crosswalk_s4/region_marks.csv", f"{R}/score/crosswalk_s4/region_recall.json"]))
    out.append(("cartop_mask", {"OUT": f"{F}/cartop"}, [f"{F}/cartop/camera_lookup.csv"], ["score"],
                [f"{F}/cartop/{n}" for n in ("arms.csv", "cards.csv", "labels_hidden.csv", "summary.json")]))
    for city, ratings in A3:
        c = f"{F}/osm_arm/cities/{city}"
        out.append(("osm_arm", {"OUT": f"{F}/osm_arm"}, [f"{c}/a3_candidates.csv", f"{c}/placements_kinds.csv"],
                    ["city-score", "--city", city, "--ratings", f"{c}/ratings/{ratings}"],
                    [f"{c}/a3_score.json"]))
    return out


def _copy(rel, tmp):
    src, dst = os.path.join(REPO, rel), os.path.join(tmp, rel)
    if rel.endswith("/"):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def _run_module(name, consts, argv, tmp):
    """Import `name`, point its OUT (or rc.OUT) at the temp copy, run its main() quietly; restore afterwards."""
    import importlib
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    mod = importlib.import_module(name)
    targets = []
    for const, rel in consts.items():
        obj = mod
        if "." in const:
            attr, const = const.split(".")
            obj = getattr(mod, attr)
        targets.append((obj, const, getattr(obj, const)))
        setattr(obj, const, os.path.join(tmp, rel))
    old_argv, cwd = sys.argv, os.getcwd()
    sys.argv = [name + ".py"] + argv
    try:
        os.chdir(REPO)
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        return None
    except SystemExit as e:
        return f"{name} {argv[0]} exited: {e}"
    finally:
        sys.argv = old_argv
        os.chdir(cwd)
        for obj, const, value in targets:
            setattr(obj, const, value)


def _compare(rel, tmp):
    with open(os.path.join(REPO, rel), "rb") as f:
        want = f.read()
    path = os.path.join(tmp, rel)
    if not os.path.exists(path):
        return f"{rel}: not regenerated"
    with open(path, "rb") as f:
        got = f.read()
    return None if got == want else f"{rel}: the regenerated file differs from the committed one"


def run(tmp):
    problems, compared = [], 0
    for argv, outputs in score_274_steps():
        argv = [x.replace("{T}", tmp) for x in argv]
        r = subprocess.run([sys.executable, S] + argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        if r.returncode:
            problems.append(f"score_274.py {argv[0]} failed:\n{r.stdout[-2000:]}")
            continue
        for rel in outputs:
            compared += 1
            problems += [p for p in [_compare(rel, tmp)] if p]
    for name, consts, inputs, argv, outputs in module_steps():
        for rel in inputs:
            _copy(rel, tmp)
        err = _run_module(name, consts, [x.replace("{T}", tmp) for x in argv], tmp)
        if err:
            problems.append(err)
            continue
        for rel in outputs:
            compared += 1
            problems += [p for p in [_compare(rel, tmp)] if p]
    return problems, compared


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare and exit 1 on any difference (what the script always does; named for the index)")
    ap.add_argument("--keep", help="write the regenerated files here instead of a temporary directory")
    a = ap.parse_args()
    tmp = a.keep or tempfile.mkdtemp(prefix="check285_")
    try:
        problems, compared = run(os.path.abspath(tmp))
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    for p in problems:
        print("FAIL", p)
    print(f"{compared} regenerated outputs compared byte for byte; {len(problems)} problem(s)")
    if not compared:
        print("FAIL nothing was compared")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
