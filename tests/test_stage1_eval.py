"""Guards for the Stage 1 dataset evaluator (stage_one/dataset_evaluation/evaluate.py).

The matching *semantics* — including the issue #18 fix where a redundant
generated point on an already-matched ramp scores as a false positive — live in
``rampnet.metrics`` and are tested in ``tests/test_metrics.py``. What this file
guards is that the Stage 1 evaluator actually **uses** that shared definition
rather than forking its own, which is the failure mode issue #22 exists to
prevent.

The module is loaded by explicit path rather than by putting its directory on
``sys.path``: the repo has several modules named ``evaluate`` (``stage_two``,
``stage_one/dataset_evaluation``), so a bare ``import evaluate`` would resolve to
whichever landed in ``sys.modules`` first in a shared pytest session.
"""
import importlib.util
import os

import rampnet.detection_eval
import rampnet.metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO_ROOT, "stage_one", "dataset_evaluation", "evaluate.py")


def _load():
    spec = importlib.util.spec_from_file_location("stage1_dataset_evaluate", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_heavy_deps():
    # Imports with no torch/matplotlib/PIL, so it stays unit-testable and cheap.
    assert _load() is not None


def test_uses_the_shared_matcher_not_a_fork():
    assert _load().match_points is rampnet.metrics.match_points


def test_uses_the_shared_radius_and_scales():
    module = _load()
    assert module.RADIUS_THRESHOLD_NORMALIZED == rampnet.detection_eval.PANO_RADIUS_NORMALIZED
    assert module.PANO_SCALE_X == rampnet.detection_eval.PANO_SCALE_X
    assert module.PANO_SCALE_Y == rampnet.detection_eval.PANO_SCALE_Y


def test_label_loaders_are_forgiving_about_missing_files():
    # A whole-dataset run must not die on one unreadable label file.
    module = _load()
    assert module.load_manual_label_points(os.path.join(REPO_ROOT, "nope.txt")) == []
    assert module.load_test_split_label_points(os.path.join(REPO_ROOT, "nope.json")) == []
