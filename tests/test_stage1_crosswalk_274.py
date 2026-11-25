"""The #274 crosswalk-detector scripts under CI: their doctests, the decision-number check, and the constants
the scripts copy from the trainer (RampNet#274). CPU only, no network, committed files only."""
import doctest
import importlib
import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "scripts", "analysis", "stage1_crosswalk_sdot_274")
HAS_PYPROJ = importlib.util.find_spec("pyproj") is not None
# Doctests that project through pilot.geod() (pyproj). Everything else runs without it, as on CI.
NEEDS_PYPROJ = {"score_274.ground_point"}


def _load(name):
    """The script as a module, imported by name: the scripts import each other (crowd_arm imports pilot), and
    DocTestFinder only collects objects whose module is the one in sys.modules."""
    if D not in sys.path:
        sys.path.insert(0, D)
    mod = importlib.import_module(name)
    assert os.path.dirname(os.path.abspath(mod.__file__)) == D, f"{name} imported from {mod.__file__}"
    return mod


@pytest.mark.parametrize("name", ["score_274", "brick_positives", "crowd_arm", "pilot"])
def test_doctests(name):
    """`python -m doctest` on each script, which plain pytest does not collect. score_274's doctests pin
    label_hits, paired_recall_diff, pct_interval, audit_rates, recall_matched_threshold and finite."""
    mod = _load(name)
    finder, runner = doctest.DocTestFinder(), doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    ran = 0
    for t in finder.find(mod, name):
        if not t.examples or (t.name in NEEDS_PYPROJ and not HAS_PYPROJ):
            continue
        runner.run(t)
        ran += len(t.examples)
    res = runner.summarize(verbose=False)
    assert ran and res.failed == 0, f"{res.failed} of {ran} doctest examples failed in {name}"


def test_score_274_has_the_doctests_this_file_relies_on():
    """Guard against the doctest test passing vacuously after a refactor drops the examples."""
    mod = _load("score_274")
    have = {t.name.split(".", 1)[1] for t in doctest.DocTestFinder().find(mod, "score_274") if t.examples}
    for fn in ("label_hits", "pct_interval", "finite"):
        assert fn in have, fn


def test_check_274_reproduces_the_decision_numbers():
    """scripts/analysis/stage1_crosswalk_sdot_274/check_274.py: P1 (Columbia), the s4 and s3 confirmation draws, the
    s3 audit, the Juneau eval draw and root_v3's negatives, regenerated and compared byte for byte."""
    r = subprocess.run([sys.executable, os.path.join(D, "check_274.py")], cwd=REPO, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "50 regenerated outputs compared byte for byte; root_v3 negatives 166; 0 problem(s)" in r.stdout


def test_copied_constants_agree_with_the_trainer():
    """export_stage2 and score_274 re-declare the trainer's negative-point rule instead of importing
    stage_two/train.py (a script: importing it builds a model). The copies must agree with it: the minimum
    negative-to-positive distance and its metric (export_stage2.NEGATIVE_MIN_DIST, neg_dist), the scorer's match
    radius it equals (score_274.RADIUS), and a negative's training disc (score_274.NEG_DISC)."""
    import inspect
    import math
    import random
    sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
    from check_lr_schedule_135 import load_from_train_py
    tr = load_from_train_py("NEGATIVE_MIN_DIST", "supervision_negative_points", "_point_list", "add_negative_weight",
                            math=math, np=None, torch=None)
    exp, sc = _load("export_stage2"), _load("score_274")
    assert exp.NEGATIVE_MIN_DIST == tr.NEGATIVE_MIN_DIST == sc.RADIUS
    defaults = inspect.signature(tr.add_negative_weight).parameters
    sigma, (_, width) = defaults["sigma"].default, defaults["heatmap_shape"].default
    assert sc.NEG_DISC == math.ceil(3 * sigma) / width
    rng = random.Random(274)        # the trainer refuses a negative exactly when export_stage2 says it is too near
    for _ in range(2000):
        neg, pos = (rng.random(), rng.uniform(0.4, 0.8)), (rng.random(), rng.uniform(0.4, 0.8))
        if rng.random() < 0.5:      # concentrate on the boundary, seam included
            pos = ((neg[0] + rng.uniform(-0.06, 0.06)) % 1.0, neg[1] + rng.uniform(-0.12, 0.12))
        try:
            tr.supervision_negative_points({"supervision": {"negative_points": [list(neg)]}}, [pos])
            refused = False
        except ValueError:
            refused = True
        assert refused == (exp.neg_dist(neg, pos) < exp.NEGATIVE_MIN_DIST), (neg, pos)


def _section(doc, heading):
    text = open(os.path.join(REPO, "docs", doc), encoding="utf-8").read()
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def _wall(s):
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} m {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} m {s % 60:02d} s"


def test_compute_cost_tables_match_the_ledger():
    """docs/compute_cost.md's #274 section: every training row (job, state, GPUs, wall-clock, GPU-h) and every
    inference range (records, GPU-h) agrees with the ledger rows the committed dump parses to."""
    sys.path.insert(0, REPO)
    from rampnet import ledger
    rows = [r for r in ledger.read_rows(os.path.join(REPO, "analysis_out", "compute_log.jsonl"))
            if r["cluster"] == "klone" and r["job_name"].startswith(("cw274", "crosswalk274"))]
    sec = _section("compute_cost.md", "## klone, 2026-11-04 to 2026-11-09: the #274 crosswalk detector")
    table = [[c.strip() for c in ln.strip("|").split("|")] for ln in sec.splitlines()
             if ln.startswith("| ") and ln[2:3].isdigit()]
    train = [t for t in table if len(t) == 6]
    infer = [t for t in table if len(t) == 4]
    want = [r for r in rows if r["gpus"] == 4]
    assert len(train) == len(want) == 20
    by_job = {}
    for r in want:
        by_job.setdefault(r["job_id"], []).append(r)
    for job, _, state, gpus, wall, gpu_h in train:
        r = by_job[job].pop(0)      # a requeued job's incarnations are listed in start order
        assert (state, gpus, wall, gpu_h) == (r["state"], f"4 {r['gpu_type'].upper()}", _wall(r["elapsed_s"]),
                                              f"{r['gpu_hours']:.2f}"), job
    assert len(infer) == 3
    one = [r for r in rows if r["gpus"] == 1]
    for jobs, _, n, gpu_h in infer:
        lo, hi = jobs.split("-")
        sel = [r for r in one if lo <= r["job_id"] <= hi]
        assert (int(n), gpu_h) == (len(sel), f"{sum(r['gpu_hours'] for r in sel):.2f}"), jobs
    assert sum(int(t[2]) for t in infer) == len(one) == 42
    assert f"{sum(r['gpu_hours'] for r in rows):.2f}" == "84.09" and "84.09 GPU-hours" in sec


def test_compute_cost_tables_match_the_ledger_285():
    """The same for docs/compute_cost.md's #285 `crosswalk_s6` section (jobs named cw285_*): 14 training rows,
    two inference ranges, 65.39 GPU-hours."""
    sys.path.insert(0, REPO)
    from rampnet import ledger
    rows = [r for r in ledger.read_rows(os.path.join(REPO, "analysis_out", "compute_log.jsonl"))
            if r["cluster"] == "klone" and r["job_name"].startswith("cw285_")]
    sec = _section("compute_cost.md", "## klone, 2026-11-13: the #285 `crosswalk_s6` study")
    table = [[c.strip() for c in ln.strip("|").split("|")] for ln in sec.splitlines()
             if ln.startswith("| ") and ln[2:3].isdigit()]
    train = [t for t in table if len(t) == 6]
    infer = [t for t in table if len(t) == 4]
    want = {r["job_id"]: r for r in rows if r["gpus"] == 4}
    assert len(train) == len(want) == 14     # no job was requeued, so one row per job id
    for job, _, state, gpus, wall, gpu_h in train:
        r = want[job]
        assert (state, gpus, wall, gpu_h) == (r["state"], f"4 {r['gpu_type'].upper()}", _wall(r["elapsed_s"]),
                                              f"{r['gpu_hours']:.2f}"), job
    assert len(infer) == 2
    one = [r for r in rows if r["gpus"] == 1]
    for jobs, _, n, gpu_h in infer:
        lo, hi = jobs.split("-")
        sel = [r for r in one if lo <= r["job_id"] <= hi]
        assert (int(n), gpu_h) == (len(sel), f"{sum(r['gpu_hours'] for r in sel):.2f}"), jobs
    assert sum(int(t[2]) for t in infer) == len(one) == 57
    assert f"{sum(r['gpu_hours'] for r in rows):.2f}" == "65.39" and "65.39 GPU-hours" in sec


# score_274.py's `cards --definition` (and DEFINITIONS["paint_brick"]) arrived in f1e2f21b, 2026-11-08 07:52:58 -0800.
PAINT_BRICK_FROM = "2026-11-08T15:52:58Z"


def test_ratings_definitions_sidecar():
    """analysis_out/stage1_crosswalk_detector_274/ratings_definitions.csv: every committed #274 verdict file is
    listed once, every row names files that exist, carries the definition's exact text, and a carried or raw copy
    has its source page's definition. A page cannot have been rated under paint_brick before the code offered it, so
    no file marked paint_brick holds a verdict timed before f1e2f21b."""
    import csv
    import glob
    import re
    base = os.path.join(REPO, "analysis_out", "stage1_crosswalk_detector_274")
    texts = {k: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", v)).strip() for k, v in _load("score_274").DEFINITIONS.items()}
    with open(os.path.join(base, "ratings_definitions.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    listed = {}
    for r in rows:
        assert r["definition_text"] == texts[r["definition"]], r["file"]
        paths = [r["file"].replace("{draw}", d).replace("{model}", m)
                 for d in ("seattle", "juneau", "laurens_gsv", "laurens_mly") for m in ("s4", "s5")]
        for p in sorted(set(paths)):
            assert os.path.isfile(os.path.join(base, p)), p
            assert p not in listed, f"{p} listed twice"
            listed[p] = r["definition"]
    on_disk = {os.path.relpath(p, base).replace(os.sep, "/") for pat in
               ("**/ratings_jon_*.csv", "hardneg/confirm_s3/*/s*/ratings.csv", "hardneg/s5_inputs/juneau_eval/*/ratings.csv",
                "raw_downloads/*_ratings_*.csv") for p in glob.glob(os.path.join(base, pat), recursive=True)}
    assert on_disk == set(listed), sorted(on_disk ^ set(listed))
    h = "hardneg/"
    for d in ("seattle", "juneau", "laurens_gsv", "laurens_mly"):       # carried copies follow their source page
        for m in ("s1", "s3"):
            assert listed[f"{h}confirm_s3/{d}/{m}/ratings.csv"] == listed[f"{h}confirm_s3/page/ratings_jon_2026-11-08_confirm.csv"]
        assert listed[f"{h}confirm_s3/{d}/s4/ratings.csv"] == listed[f"{h}s4_stepB/ratings_jon_2026-11-09_s4_stepB.csv"]
    for m in ("s4", "s5"):
        assert listed[f"{h}s5_inputs/juneau_eval/{m}/ratings.csv"] == \
            listed[f"{h}s5_stepB/ratings_jon_2026-11-10_s5_stepB_fixed.csv"]
    for r in rows:
        if r["file"].startswith("raw_downloads/"):
            page = r["note"].rsplit(" ", 1)[1]
            assert r["definition"] == listed[page], r["file"]
    timed = 0
    for p, d in listed.items():
        with open(os.path.join(base, p), encoding="utf-8", newline="") as f:
            times = [t["rated_at"] for t in csv.DictReader(f) if t.get("rated_at") not in (None, "", "prefill")]
        if times and d == "paint_brick":
            assert min(times) >= PAINT_BRICK_FROM, (p, min(times))
            timed += 1
    assert timed >= 8
