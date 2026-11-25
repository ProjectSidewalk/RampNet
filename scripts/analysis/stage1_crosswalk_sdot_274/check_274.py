"""Re-derive the #274 decision numbers from committed files and compare them with the committed outputs (RampNet#274).

Every number the crosswalk-detector study decided on that does not need a GPU, the Stage 2 roots or the pano export
regenerates in seconds, on CPU, from files in the repo. This re-runs those `score_274.py` steps into a temporary
directory and compares each output with the committed copy, byte for byte:

* P1, Columbia brick label recall (`label-recall`, s1 -> s4, with s3 alongside);
* the s4 confirmation on the four draws (Seattle, Juneau, Laurens GSV, Laurens Mapillary): `carry` from Jon's Step B
  export, `audit-summary`, and `audit-compare` against s1 (P2 is the Laurens Mapillary row);
* the s3 confirmation on the same draws (`audit-summary`, `audit-compare` s1 -> s3);
* the s3 deployment audit and its retention guard (`audit-summary` of Step 0 and of s3, `audit-compare`);
* the Juneau finding on the fresh eval draw (`carry` from the s5 Step B export, `audit-summary`, `audit-compare`
  s4 -> s5);
* root_v3's 166 negatives (`export_stage2.confirmed_negatives`): the card ids and the counts in its manifest.

Where pyproj is installed (it is not in requirements-dev.txt, so CI skips this part), also `bprime-score` from the
committed pano-data extract `bprime/seattle_pano_data_bprime.psv`: the (b') `nodes.csv` of v2_labels (s2, s3),
s4_matched (s4 guard 3) and s5_matched (s5 guard 6) byte for byte, and their `scores.json` apart from `scored_utc`.

Not covered, because the Stage 2 roots are not in the repo: `score` (guard 1, the (a) tables, the threshold sweeps).
Nothing here runs a model.

    python scripts/analysis/stage1_crosswalk_sdot_274/check_274.py            # exit 1 on any difference
    python scripts/analysis/stage1_crosswalk_sdot_274/check_274.py --keep DIR # leave the regenerated files in DIR

It is the `stage1_crosswalk_274` entry of scripts/check_all.py (in --ci).
"""
import argparse
import csv
import json
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
C = f"{N}/confirm_s3"
J = f"{N}/s5_inputs/juneau_eval"
DRAWS = ("seattle", "juneau", "laurens_gsv", "laurens_mly")


def steps():
    """[(argv for score_274.py with {T} for the temp root, [committed outputs it must reproduce])]."""
    out = [(["label-recall", "--labels", f"{N}/s4_inputs/stepA/columbia_gt.csv",
             "--base", f"crosswalk_s1={N}/s4_eval/columbia/preds/crosswalk_s1/columbia.jsonl@0.21",
             "--new", f"crosswalk_s4={N}/s4_eval/columbia/preds/crosswalk_s4/columbia.jsonl@0.23",
             "--also", f"crosswalk_s3={N}/s4_eval/columbia/preds/crosswalk_s3/columbia.jsonl@0.27",
             "--out", f"{{T}}/{N}/s4_eval/columbia"],
            [f"{N}/s4_eval/columbia/label_hits.csv", f"{N}/s4_eval/columbia/label_recall.json"])]
    for d in DRAWS:
        preds = (lambda lab: f"{D}/v2_labels/bprime/preds/{lab}/bprime.jsonl") if d == "seattle" else \
                (lambda lab: f"{C}/preds/{lab}/cities.jsonl")
        for m in ("s1", "s3"):
            out.append((["audit-summary", "--panos", f"{C}/{d}/{m}/panos.csv", "--ratings", f"{C}/{d}/{m}/ratings.csv",
                         "--label", f"crosswalk_{m}", "--out", f"{{T}}/{C}/{d}/{m}"],
                        [f"{C}/{d}/{m}/audit_summary.json", f"{C}/{d}/{m}/per_pano.csv"]))
        out.append((["audit-compare", "--base-panos", f"{C}/{d}/s1/panos.csv", "--base-ratings", f"{C}/{d}/s1/ratings.csv",
                     "--new-panos", f"{C}/{d}/s3/panos.csv", "--new-ratings", f"{C}/{d}/s3/ratings.csv",
                     "--new-preds", preds("crosswalk_s3"), "--new-threshold", "0.27",
                     "--base-label", "crosswalk_s1", "--new-label", "crosswalk_s3", "--out", f"{{T}}/{C}/{d}"],
                    [f"{C}/{d}/audit_compare.json"]))
        out.append((["carry", "--ratings", f"{N}/s4_stepB/ratings_jon_2026-11-09_s4_stepB.csv",
                     "--candidates", f"{C}/{d}/s4/candidates.csv", "--out", f"{{T}}/{C}/{d}/s4/ratings.csv"],
                    [f"{C}/{d}/s4/ratings.csv"]))
        out.append((["audit-summary", "--panos", f"{C}/{d}/s4/panos.csv", "--ratings", f"{{T}}/{C}/{d}/s4/ratings.csv",
                     "--label", "crosswalk_s4", "--out", f"{{T}}/{C}/{d}/s4"],
                    [f"{C}/{d}/s4/audit_summary.json", f"{C}/{d}/s4/per_pano.csv"]))
        out.append((["audit-compare", "--base-panos", f"{C}/{d}/s1/panos.csv", "--base-ratings", f"{C}/{d}/s1/ratings.csv",
                     "--new-panos", f"{C}/{d}/s4/panos.csv", "--new-ratings", f"{{T}}/{C}/{d}/s4/ratings.csv",
                     "--new-preds", preds("crosswalk_s4"), "--new-threshold", "0.23",
                     "--base-label", "crosswalk_s1", "--new-label", "crosswalk_s4",
                     "--ci", "0.975" if d == "laurens_mly" else "0.95", "--out", f"{{T}}/{C}/{d}/s4"],
                    [f"{C}/{d}/s4/audit_compare.json"]))
    for sub, rat, lab in (("step0_audit", "ratings_jon_2026-11-06_step0.csv", "crosswalk_s1"),
                          ("s3_audit", "ratings_jon_2026-11-08_s3.csv", "crosswalk_s3")):
        out.append((["audit-summary", "--panos", f"{N}/{sub}/panos.csv", "--ratings", f"{N}/{sub}/{rat}",
                     "--label", lab, "--out", f"{{T}}/{N}/{sub}"],
                    [f"{N}/{sub}/audit_summary.json", f"{N}/{sub}/per_pano.csv"]))
    out.append((["audit-compare", "--base-panos", f"{N}/step0_audit/panos.csv",
                 "--base-ratings", f"{N}/step0_audit/ratings_jon_2026-11-06_step0.csv",
                 "--new-panos", f"{N}/s3_audit/panos.csv", "--new-ratings", f"{N}/s3_audit/ratings_jon_2026-11-08_s3.csv",
                 "--new-preds", f"{D}/v2_labels/bprime/preds/crosswalk_s3/bprime.jsonl", "--new-threshold", "0.29",
                 "--base-label", "crosswalk_s1", "--new-label", "crosswalk_s3", "--out", f"{{T}}/{N}/s3_audit"],
                [f"{N}/s3_audit/audit_compare.json"]))
    for m in ("s4", "s5"):
        out.append((["carry", "--ratings", f"{N}/s5_stepB/ratings_jon_2026-11-10_s5_stepB_fixed.csv",
                     "--candidates", f"{J}/{m}/candidates.csv", "--out", f"{{T}}/{J}/{m}/ratings.csv"],
                    [f"{J}/{m}/ratings.csv"]))
        out.append((["audit-summary", "--panos", f"{J}/{m}/panos.csv", "--ratings", f"{{T}}/{J}/{m}/ratings.csv",
                     "--label", f"crosswalk_{m}", "--out", f"{{T}}/{J}/{m}"],
                    [f"{J}/{m}/audit_summary.json", f"{J}/{m}/per_pano.csv"]))
    out.append((["audit-compare", "--base-panos", f"{J}/s4/panos.csv", "--base-ratings", f"{{T}}/{J}/s4/ratings.csv",
                 "--new-panos", f"{J}/s5/panos.csv", "--new-ratings", f"{{T}}/{J}/s5/ratings.csv",
                 "--new-preds", f"{N}/s5_eval/juneau_eval/preds/crosswalk_s5/juneau_eval.jsonl",
                 "--new-threshold", "0.26", "--base-label", "crosswalk_s4", "--new-label", "crosswalk_s5",
                 "--out", f"{{T}}/{J}/s5"],
                [f"{J}/s5/audit_compare.json"]))
    return out


BPRIME = (("v2_labels", "crosswalk_s3,crosswalk_s2,crosswalk_s1,released", "v2_labels/bprime"),
          ("v2_labels/s4_matched", "crosswalk_s4,crosswalk_s1,crosswalk_s3,released", "v2_labels/s4_matched/bprime"),
          ("v2_labels/s5_matched", "crosswalk_s5,crosswalk_s4,released", "v2_labels/s5_matched/bprime"))


def check_bprime(tmp):
    """(b') from the committed extract, as the doc's `cp` step assembles its inputs. Returns (problems, compared);
    compared is None when pyproj is absent (ground_point needs it)."""
    import importlib.util
    if importlib.util.find_spec("pyproj") is None:
        return [], None
    bp = os.path.join(tmp, "bprime_inputs")
    os.makedirs(bp, exist_ok=True)
    shutil.copy(os.path.join(REPO, D, "bprime", "sample.csv"), bp)
    for src in (f"{D}/bprime/preds", f"{D}/v2_labels/bprime/preds"):
        shutil.copytree(os.path.join(REPO, src), os.path.join(bp, "preds"), dirs_exist_ok=True)
    problems, compared = [], 0
    for scores, labels, ref in BPRIME:
        out = os.path.join(tmp, "bprime_out", ref)
        r = subprocess.run([sys.executable, S, "bprime-score", "--bprime", bp, "--scores", f"{D}/{scores}",
                            "--pano-data", f"{D}/bprime/seattle_pano_data_bprime.psv", "--labels", labels,
                            "--out", out], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            problems.append(f"score_274.py bprime-score for {ref} failed:\n{r.stdout[-2000:]}")
            continue
        for name, strip in (("nodes.csv", False), ("scores.json", True)):
            with open(os.path.join(REPO, D, ref, name), "rb") as f:
                want = f.read()
            with open(os.path.join(out, name), "rb") as f:
                got = f.read()
            if strip:       # the run's own timestamp is the one line that may differ
                want, got = ([ln for ln in b.split(b"\n") if b'"scored_utc"' not in ln] for b in (want, got))
            compared += 1
            if got != want:
                problems.append(f"{D}/{ref}/{name}: the regenerated file differs from the committed one")
    return problems, compared


def check_negatives():
    """root_v3's negatives: confirmed_negatives on the committed verdicts gives negatives.csv's card ids, and the
    counts root_v3/manifest.json records. Returns a list of problems."""
    sys.path.insert(0, HERE)
    import export_stage2
    p = f"{N}/prescreen"
    rows, counts = export_stage2.confirmed_negatives(
        [os.path.join(REPO, p, f"ratings_jon_2026-11-06_{k}.csv") for k in ("confirm", "review9", "confirm2")],
        [os.path.join(REPO, p, "drop_disc_rule.csv")])
    with open(os.path.join(REPO, N, "root_v3", "negatives.csv"), encoding="utf-8", newline="") as f:
        committed = [r["card_id"] for r in csv.DictReader(f)]
    with open(os.path.join(REPO, N, "root_v3", "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    problems = []
    if sorted(r["card_id"] for r in rows) != sorted(committed):
        problems.append("root_v3/negatives.csv: card ids differ from confirmed_negatives")
    for k in ("rated_cards", "verdicts", "dropped_disc_rule", "negatives"):
        if counts[k] != man[k]:
            problems.append(f"root_v3/manifest.json {k}: committed {man[k]}, re-derived {counts[k]}")
    return problems, len(rows)


def run(tmp):
    problems, compared = [], 0
    for argv, outputs in steps():
        argv = [x.replace("{T}", tmp) for x in argv]
        for rel in outputs:     # carry writes a file and does not make its directory
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
        r = subprocess.run([sys.executable, S] + argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        if r.returncode:
            problems.append(f"score_274.py {argv[0]} failed:\n{r.stdout[-2000:]}")
            continue
        for rel in outputs:
            with open(os.path.join(REPO, rel), "rb") as f:
                want = f.read()
            with open(os.path.join(tmp, rel), "rb") as f:
                got = f.read()
            compared += 1
            if got != want:
                problems.append(f"{rel}: the regenerated file differs from the committed one")
    neg_problems, n_neg = check_negatives()
    bp_problems, n_bp = check_bprime(tmp)
    return problems + neg_problems + bp_problems, compared, n_neg, n_bp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", help="write the regenerated files here instead of a temporary directory")
    a = ap.parse_args()
    tmp = a.keep or tempfile.mkdtemp(prefix="check274_")
    try:
        problems, compared, n_neg, n_bp = run(os.path.abspath(tmp))
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    for p in problems:
        print("FAIL", p)
    print("(b') bprime-score: skipped, pyproj is not installed" if n_bp is None else
          f"(b') bprime-score: {n_bp} regenerated outputs compared")
    print(f"{compared} regenerated outputs compared byte for byte; root_v3 negatives {n_neg}; "
          f"{len(problems)} problem(s)")
    if not compared:
        print("FAIL nothing was compared")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
