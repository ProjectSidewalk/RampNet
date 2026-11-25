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
