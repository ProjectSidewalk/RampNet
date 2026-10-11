"""Tests for the #48 cross-view alignment harness (scripts/analysis/crossview_align_48.py)
and its arms (scripts/analysis/crossview_arms/).

CPU only, no network, no labeler checkout, no imagery: the view geometry and the arm
plumbing are checked on toy inputs, and the committed artifacts under
``analysis_out/crossview_align_48/`` are checked for LF-pinned bytes, the frozen pair list's
hash, and re-derived where they can be (the sample from the eligible list, and every arm's
headline median and fallback rate from its committed predictions).
"""
import ast
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import crossview_align_48 as cv  # noqa: E402
from crossview_arms._registry import ANSWER_KEYS, Arm  # noqa: E402

OUT = cv.OUT


# --------------------------------------------------------------------------- #
# view geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.25, 0.6), (0.99, 0.62), (0.01, 0.45)])
def test_view_centre_maps_to_the_view_target(centre):
    x, y = cv.view_to_pano(cv.VIEW_W / 2.0, cv.VIEW_H / 2.0, *centre)
    assert float(x) == pytest.approx(centre[0], abs=1e-9)
    assert float(y) == pytest.approx(centre[1], abs=1e-9)


@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.3, 0.63), (0.995, 0.58)])
def test_view_and_pano_round_trip(centre):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, cv.VIEW_W, 50)
    v = rng.uniform(0, cv.VIEW_H, 50)
    x, y = cv.view_to_pano(u, v, *centre)
    u2, v2, front = cv.pano_to_view(x, y, *centre)
    assert front.all()
    assert np.allclose(u2, u, atol=1e-6) and np.allclose(v2, v, atol=1e-6)


def test_view_axes_right_is_clockwise_and_down_is_down():
    x_r, y_r = cv.view_to_pano(cv.VIEW_W / 2.0 + 100, cv.VIEW_H / 2.0, 0.5, 0.5)
    x_d, y_d = cv.view_to_pano(cv.VIEW_W / 2.0, cv.VIEW_H / 2.0 + 100, 0.5, 0.5)
    assert float(x_r) > 0.5 and float(y_r) == pytest.approx(0.5, abs=1e-12)
    assert float(y_d) > 0.5 and float(x_d) == pytest.approx(0.5, abs=1e-12)


def test_horizontal_fov_edge_is_half_the_fov():
    x, _ = cv.view_to_pano(cv.VIEW_W, cv.VIEW_H / 2.0, 0.5, 0.5)
    assert (float(x) - 0.5) * 360.0 == pytest.approx(cv.HFOV_DEG / 2.0, abs=1e-9)


def test_angular_and_pixel_errors_are_seam_safe():
    assert float(cv.angular_error_deg(0.5, 0.5, 0.5 + 1 / 360.0, 0.5)) == pytest.approx(1.0)
    assert float(cv.angular_error_deg(0.999, 0.5, 0.001, 0.5)) == pytest.approx(0.72, abs=1e-6)
    assert float(cv.angular_error_deg(0.1, 0.99, 0.6, 0.99)) < 3.7   # near the nadir
    assert float(cv.equirect_px_error(0.999, 0.5, 0.001, 0.5)) == pytest.approx(0.002 * 4096)
    assert float(cv.equirect_px_error(0.5, 0.5, 0.5, 0.51)) == pytest.approx(0.01 * 2048)


def test_ground_mask_keeps_below_the_margin_only():
    u = np.array([cv.VIEW_W / 2.0] * 3)
    f = cv.focal_px()
    v = cv.VIEW_H / 2.0 + f * np.tan(np.radians([-1.0, 2.0, 10.0]))
    assert list(cv.ground_mask(u, v, 0.5, 0.5, margin=0.5)) == [False, True, True]
    assert list(cv.ground_mask(u, v, 0.5, 0.5, margin=5.0)) == [False, False, True]


def test_map_point_identity_and_degenerate():
    assert cv.map_point(np.eye(3), 10.0, 20.0) == (10.0, 20.0)
    assert cv.map_point(None, 1.0, 1.0) is None
    assert cv.map_point(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.0]]), 1.0, 1.0) is None


def test_render_view_puts_a_marked_point_where_the_geometry_says():
    cv2 = pytest.importorskip("cv2")
    W = int(round(360.0 / cv.HFOV_DEG * cv.VIEW_W))
    equi = np.zeros((W // 2, W, 3), np.uint8)
    cv2.circle(equi, (int(0.55 * W), int(0.6 * W / 2)), 4, (255, 255, 255), -1)
    view = cv.render_view(equi, 0.5, 0.55)
    vv, uu = np.unravel_index(np.argmax(view[..., 0]), view.shape[:2])
    u, v, _ = cv.pano_to_view(0.55, 0.6, 0.5, 0.55)
    assert abs(uu - float(u)) < 4 and abs(vv - float(v)) < 4


# --------------------------------------------------------------------------- #
# arms: registry, answer hiding, fallback
# --------------------------------------------------------------------------- #
def test_registry_has_the_committed_arms():
    arms = cv.load_arms()
    for name in ("lg", "lg_local", "lg_band0.5", "sift", "ncc", "proj_flat_check",
                 "proj_height_perpano", "proj_height_auto", "proj_mly_gravity",
                 "proj_mly_road", "proj_mly_rawgps", "proj_gsv_depth"):
        assert name in arms, name
        assert arms[name].description


def test_run_arm_hides_the_answer_and_normalizes_fallbacks():
    pairs = [{"pair_id": "a", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7,
              "ref_conf": 0.9, "ref_world_gap_m": 1.0},
             {"pair_id": "b", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7,
              "ref_conf": 0.9, "ref_world_gap_m": 1.0}]
    seen = []

    def fn(pair, ctx):
        seen.append(set(pair))
        return None if pair["pair_id"] == "a" else {"x": 0.4, "y": None, "why": "half"}

    rows, missing = cv.run_arm(Arm("t", fn), pairs, SimpleNamespace())
    assert missing == 0
    assert all(not (s & set(ANSWER_KEYS)) for s in seen)
    assert rows[0] == {"pair_id": "a", "x": None, "y": None}
    assert rows[1] == {"pair_id": "b", "x": None, "y": None, "why": "half"}


def test_a_planted_arm_cannot_read_the_reference_through_ctx_pairs():
    """Review of #210 (A3/C5): run_arm stripped the answer from ``pair`` only, and an arm that
    looked itself up in ``ctx.pairs`` scored 0.00 deg. Context now stores stripped rows."""
    pairs = [{"pair_id": "a", "city": "x", "src_pano": "s", "oth_pano": "o", "proj_x": 0.5,
              "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7, "ref_conf": 0.9,
              "ref_world_gap_m": 1.0}]
    ctx = cv.Context(SimpleNamespace(), pairs)

    def cheat(pair, ctx):
        mine = next(r for r in ctx.pairs if r["pair_id"] == pair["pair_id"])
        return {"x": mine["ref_x"], "y": mine["ref_y"]}

    with pytest.raises(KeyError):
        cv.run_arm(Arm("cheat", cheat), pairs, ctx)
    assert all(not (set(r) & set(ANSWER_KEYS)) for r in ctx.pairs)
    assert "ref_x" in pairs[0], "the caller's rows must not be mutated"


def test_slim_panos_reach_arms_without_detections():
    """The reference is one of the other view's detections, so SlimPanos handed to arms
    (ctx.slim, ctx.at_height, geometry.at_height) carry none."""
    from dataclasses import dataclass

    @dataclass
    class Slim:
        pano_id: str
        detections: list

    p = Slim("o", [(0, 0.1, 0.7, 0.9)])
    q = cv.without_detections(p)
    assert q.detections == [] and q.pano_id == "o" and p.detections, "copy, not in place"
    import inspect
    from crossview_arms import geometry
    assert "ctx.at_height(" in inspect.getsource(geometry.at_height)
    assert "without_detections" in inspect.getsource(cv.Context.at_height)
    assert "without_detections" in inspect.getsource(cv.Context.slim)


# --------------------------------------------------------------------------- #
# answer hiding: planted arms, one per route (final re-review of #210, N1)
# --------------------------------------------------------------------------- #
_PLANT_PAIRS = [{"pair_id": "a", "city": "x", "src_pano": "s", "oth_pano": "o", "proj_x": 0.5,
                 "proj_y": 0.6, "ref_x": 0.1, "ref_y": 0.7, "ref_conf": 0.9,
                 "ref_world_gap_m": 1.0}]


def test_a_planted_arm_cannot_read_the_frozen_pair_list():
    """read_frozen_pairs() returns the answer columns; an arm that calls it and builds the
    key at run time ("ref_" + "x") scored 0.00 deg before. It now raises inside an arm, and
    the flag is cleared again after the arm raises."""
    def cheat(pair, ctx):
        mine = next(r for r in cv.read_frozen_pairs() if r["pair_id"] == pair["pair_id"])
        return {"x": mine["ref_" + "x"], "y": mine["ref_" + "y"]}

    with pytest.raises(RuntimeError, match="inside an arm"):
        cv.run_arm(Arm("cheat", cheat), _PLANT_PAIRS, cv.Context(SimpleNamespace(), _PLANT_PAIRS))
    assert cv._IN_ARM == 0
    assert cv.read_frozen_pairs(), "outside an arm the scorer still reads it"


@pytest.mark.parametrize("name", ["pairs.csv", "eligible_pairs.csv"])
def test_a_planted_arm_cannot_read_a_pair_csv_through_read_rows(name):
    def cheat(pair, ctx):
        rows = cv.read_rows(os.path.join(cv.OUT, name))
        return {"x": rows[0]["ref_" + "x"], "y": rows[0]["ref_" + "y"]}

    with pytest.raises(RuntimeError, match="inside an arm"):
        cv.run_arm(Arm("cheat", cheat), _PLANT_PAIRS, cv.Context(SimpleNamespace(), _PLANT_PAIRS))
    assert cv._IN_ARM == 0


def _fake_labeler_ctx():
    """A Context whose full labeler is a stand-in: its loaders return panos WITH the
    reference as a detection, as the real fuse_sites does."""
    from dataclasses import dataclass

    @dataclass
    class Slim:
        pano_id: str
        detections: list

    def load(*a, **k):
        return [Slim("o", [(0, 0.1, 0.7, 0.9)]), Slim("s", [])]

    fs = SimpleNamespace(HEIGHT_AUTO="auto", pano_pose=lambda pano, mode: ("pose", pano.pano_id),
                         load_results=load,
                         load_at_height=lambda *a, **k: (load(), None, 2.5, {"resolved": 2.5}))
    full = SimpleNamespace(geo=SimpleNamespace(PER_PANO="per-pano"), fs=fs,
                           es=SimpleNamespace(load=load), prov={"git_commit": "fake"})
    ctx = cv.Context(SimpleNamespace(labeler_root="unused", runs_root="unused",
                                     results_root=None), _PLANT_PAIRS)
    ctx._full_labeler = full
    ctx._results_path = lambda city: "unused/results.jsonl"
    return ctx


@pytest.mark.parametrize("route", [
    lambda L: L.fs.load_results("results.jsonl"),
    lambda L: L.fs.load_at_height("results.jsonl", "auto"),
    lambda L: L.es.load(),
])
def test_a_planted_arm_cannot_load_detections_through_ctx_labeler(route):
    """ctx.labeler() used to be the full labeler, so fs.load_results(ctx._results_path(city))
    returned SlimPanos WITH detections. Arms now get only ARM_FS_NAMES of fuse_sites."""
    ctx = _fake_labeler_ctx()

    def cheat(pair, ctx):
        panos = route(ctx.labeler())
        return {"x": panos[0].detections[0][1], "y": panos[0].detections[0][2]}

    with pytest.raises(AttributeError, match="withheld from arms"):
        cv.run_arm(Arm("cheat", cheat), _PLANT_PAIRS, ctx)


def test_the_arm_facing_labeler_holds_only_what_arms_use():
    ctx = _fake_labeler_ctx()
    L = ctx.labeler()
    assert set(vars(L.fs)) - {"_label"} == set(cv.ARM_FS_NAMES)
    assert set(vars(L)) - {"_label"} == {"geo", "fs", "prov"}
    assert L.fs.pano_pose(SimpleNamespace(pano_id="s"), "off") == ("pose", "s")
    assert all(v is not ctx._full_labeler for v in ctx.cache.values())
    slims, height, _ = ctx.at_height("x", "auto")
    assert height == 2.5 and slims["o"].detections == [], "Context loads, then withholds"
    assert ctx.slim("x", "o").detections == []
    assert ctx.labeler_prov() == {"git_commit": "fake"}


#: What an arm module may not name outside the CLI subcommands exempted below. Docstrings
#: and comments are dropped first (ast.unparse), so prose may mention these.
_FORBIDDEN = {
    "answer column": r"\bref_(x|y|conf|world_gap_m)\b",
    "detections": r"\.detections\b",
    "pair-list reader": r"\b(read_frozen_pairs|read_rows|PAIRS_CSV)\b|pairs\.csv|eligible_pairs",
    "labeler loader": (r"\b(load_results|load_at_height|import_labeler|fuse_sites|eval_sites|"
                       r"multiview_evidence_48)\b|results\.jsonl"),
    "private harness state": r"\bctx\._|\b_IN_ARM\b|_full_labeler|__globals__|\bsys\.modules\b",
}

#: Top-level definitions that legitimately read the answer or the full labeler. Each is a
#: CLI step run before or after prediction, never from inside an arm: every other ``cmd_*``
#: function of a module is exempt too, unless it is itself a registered arm.
_EXEMPT = {
    ("_registry.py", "ANSWER_KEYS"): "the list of answer columns itself",
    ("_mv3d.py", "build_manifest"): "the corner manifest step; strips answers before use",
    ("_mv3d.py", "_pairs_without_answers"): "build_manifest's answer-stripped pair list",
    ("_mv3d.py", "main"): "CLI dispatch",
    ("_scenes.py", "main"): "CLI dispatch",
    ("semantic.py", "main"): "CLI dispatch",
}


def _exempt(module, node):
    name = getattr(node, "name", None) or (
        isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None))
    if (module, name) in _EXEMPT:
        return True
    if isinstance(node, ast.FunctionDef) and node.name.startswith("cmd_"):
        return not any("register" in ast.unparse(d) for d in node.decorator_list)
    return False


def _parse_without_docstrings(src):
    tree = ast.parse(src)
    for n in list(ast.walk(tree)):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and \
                isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


def _exempt_names(module, src):
    """Names of the exempt definitions in one module (``main`` aside: nothing calls it)."""
    tree = _parse_without_docstrings(src)
    return {getattr(n, "name", None) for n in tree.body if _exempt(module, n)} - {None, "main"}


def _forbidden_hits(module, src, exempt_elsewhere=()):
    """[(definition, rule, match)] for the non-exempt code of one arm module, plus every
    exempt definition (of this module, or ``exempt_elsewhere``) that non-exempt code names:
    an arm could route through it."""
    import re
    tree = _parse_without_docstrings(src)
    exempt = [n for n in tree.body if _exempt(module, n)]
    exempt_names = _exempt_names(module, src) | set(exempt_elsewhere)
    hits = []
    for node in tree.body:
        if node in exempt:
            continue
        code = ast.unparse(node)
        label = getattr(node, "name", type(node).__name__)
        for rule, pat in _FORBIDDEN.items():
            hits += [(label, rule, m.group()) for m in re.finditer(pat, code)]
        hits += [(label, "calls an exempt definition", e) for e in sorted(exempt_names)
                 if re.search(rf"\b{re.escape(e)}\b", code)]
    return hits


def test_arm_modules_never_reach_the_answer():
    """Grep guard for the routes Context cannot close at run time (an arm opening a file
    itself, importing fuse_sites, poking private state). Final re-review of #210 (N1): the
    guard used to exempt whole files, including _mv3d.py's prediction-path helpers
    (corner_for, select_views, poseonly_transfer); it is now per definition."""
    import glob
    arms_dir = os.path.join(os.path.dirname(cv.__file__), "crossview_arms")
    srcs = {}
    for path in sorted(glob.glob(os.path.join(arms_dir, "*.py"))):
        with open(path, encoding="utf-8") as f:
            srcs[os.path.basename(path)] = f.read()
    everywhere = set().union(*(_exempt_names(m, s) for m, s in srcs.items()))
    assert {"build_manifest", "cmd_agreement", "cmd_compare"} <= everywhere
    hits = [(m,) + h for m, s in srcs.items() for h in _forbidden_hits(m, s, everywhere)]
    assert not hits, hits


@pytest.mark.parametrize("body", [
    'return H.read_frozen_pairs()[0]["ref_" + "x"]',
    'return H.read_rows(os.path.join(H.OUT, "pairs.csv"))',
    'return open(os.path.join(H.OUT, "eligible_pairs.csv")).read()',
    'return ctx.labeler().fs.load_results(ctx._results_path(pair["city"]))',
    'import fuse_sites\n    return fuse_sites.load_at_height',
    'return open(os.path.join(ctx.args.runs_root, "x", "results.jsonl")).read()',
    'return ctx._labeler()',
    'return ctx.slim(pair["city"], pair["oth_pano"]).detections',
    'return pair["ref_x"]',
    'return ctx.labeler().fs.pano_pose.__globals__["load_results"]',
    'return _pairs_without_answers()',
    'return M.build_manifest(ctx.args)',
])
def test_the_grep_guard_catches_each_planted_route(body):
    src = "@register('cheat')\ndef cheat(pair, ctx):\n    " + body + "\n"
    assert _forbidden_hits("ff3d.py", src, {"build_manifest", "_pairs_without_answers"}), body
    # the same exempt helper is fine where it is defined, and caught once an arm calls it
    helper = "def _pairs_without_answers():\n    return H.read_frozen_pairs()\n"
    assert not _forbidden_hits("_mv3d.py", helper)
    assert _forbidden_hits("_mv3d.py", helper + "\n\n" + src, {"build_manifest"}), body


def test_the_grep_guard_passes_a_clean_arm():
    src = ("@register('ok')\ndef ok(pair, ctx):\n"
           "    s, _, _ = ctx.at_height(pair['city'], 'auto')\n"
           "    return {'x': pair['proj_x'], 'y': pair['proj_y']}\n")
    assert not _forbidden_hits("ff3d.py", src, {"build_manifest"})


def test_the_grep_guard_does_not_exempt_a_registered_cmd_function():
    src = "@register('cmd_x')\ndef cmd_x(pair, ctx):\n    return pair['ref_x']\n"
    assert _forbidden_hits("_mv3d.py", src)
    assert not _forbidden_hits("_mv3d.py", "def cmd_x(args):\n    return H.read_frozen_pairs()\n")


def test_rerunning_a_cpu_arm_through_the_hidden_context_reproduces_its_predictions():
    """mv3d_consensus reads only committed predictions, so it re-runs on CPU in seconds. With
    answers now hidden in Context, it must still reproduce its committed rows exactly."""
    pairs = cv.read_frozen_pairs()
    arms = cv.load_arms()
    for name in ("mv3d_consensus", "mv3d_consensus_else_auto"):
        ctx = cv.Context(SimpleNamespace(), pairs)
        rows, missing = cv.run_arm(arms[name], pairs, ctx)
        assert missing == 0
        committed = cv.read_predictions(name)
        for r in rows:
            got = json.loads(json.dumps(cv.rnd(r, 6), sort_keys=True))
            assert got == committed[r["pair_id"]], (name, r["pair_id"])


def test_arm_errors_fall_back_to_the_projection():
    p = [{"pair_id": "a", "proj_x": 0.5, "proj_y": 0.6, "ref_x": 0.5 + 2 / 360.0, "ref_y": 0.6}]
    proj = cv.arm_errors(p, None)
    fb = cv.arm_errors(p, {"a": {"x": None, "y": None}})
    hit = cv.arm_errors(p, {"a": {"x": 0.5 + 2 / 360.0, "y": 0.6}})
    assert proj[0][3] is False and fb[0][3] is True
    assert fb[0][0] == pytest.approx(proj[0][0])
    assert hit[0][0] == pytest.approx(0.0, abs=1e-6) and hit[0][3] is False


def test_homography_recovers_a_known_plane_mapping():
    pytest.importorskip("cv2")
    from crossview_arms import matching
    Hm = np.array([[1.1, 0.05, 20.0], [0.02, 0.9, -15.0], [1e-4, 2e-4, 1.0]])
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1000, (60, 2))
    b = np.array([cv.map_point(Hm, *p) for p in a])
    b[:10] += rng.uniform(50, 100, (10, 2))       # outliers
    Hf, n_in, _ = matching.fit_homography(a, b)
    assert n_in >= 50
    got, want = cv.map_point(Hf, 512, 384), cv.map_point(Hm, 512, 384)
    assert math.hypot(got[0] - want[0], got[1] - want[1]) < 0.5


def test_map_centre_falls_back_below_min_inliers():
    pytest.importorskip("cv2")
    from crossview_arms import matching
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1000, (10, 2))
    assert matching.map_centre(a, a + 5.0, (0.5, 0.6), matching.MIN_INLIERS) is None
    a = np.column_stack([rng.uniform(0, 1024, 40), rng.uniform(0, 768, 40)])
    out = matching.map_centre(a, a, (0.5, 0.6), matching.MIN_INLIERS)
    assert out["inliers"] == 40
    assert out["x"] == pytest.approx(0.5, abs=1e-6) and out["y"] == pytest.approx(0.6, abs=1e-6)


# --------------------------------------------------------------------------- #
# committed artifacts
# --------------------------------------------------------------------------- #
def _committed():
    names = ["eligible_pairs.csv", "pairs.csv", "pairs_meta.json", "reference_noise.json",
             "results.json"]
    for arm in cv.available_predictions():
        names += [f"predictions/{arm}.jsonl", f"predictions/{arm}.meta.json"]
    return names


@pytest.mark.parametrize("name", _committed())
def test_committed_artifacts_are_lf(name):
    with open(os.path.join(OUT, name), "rb") as f:
        assert b"\r\n" not in f.read()


def test_pair_list_is_the_frozen_one():
    assert cv.pairs_sha256() == cv.PAIRS_SHA256
    assert len(cv.read_frozen_pairs()) == cv.PAIRS_PER_CITY * len(cv.CITIES)


def test_committed_sample_rederives_from_the_eligible_list():
    eligible = cv.read_rows(cv.ELIGIBLE_CSV)
    committed = cv.read_rows(cv.PAIRS_CSV)
    sampled = cv.sample_pairs(eligible)
    key = lambda r: (r["pair_id"], r["ramp_uid"], r["src_pano"], r["oth_pano"])  # noqa: E731
    assert [key(r) for r in sampled] == [key(r) for r in committed]
    per_ramp = {}
    for r in committed:
        per_ramp[r["ramp_uid"]] = per_ramp.get(r["ramp_uid"], 0) + 1
    assert max(per_ramp.values()) <= cv.MAX_PAIRS_PER_RAMP
    assert all(r["ref_world_gap_m"] < cv.WORLD_HIT_M and r["oth_range_m"] <= cv.R_OTHER_M
               for r in committed)


def test_every_committed_prediction_covers_the_frozen_pairs():
    ids = {p["pair_id"] for p in cv.read_frozen_pairs()}
    for arm in cv.available_predictions():
        preds = cv.read_predictions(arm)      # refuses a different pairs_sha256
        assert set(preds) == ids, arm


def test_committed_results_rederive_from_committed_predictions():
    pairs = cv.read_frozen_pairs()
    with open(cv.RESULTS_JSON, encoding="utf-8") as f:
        res = json.load(f)
    assert res["config"]["pairs_sha256"] == cv.PAIRS_SHA256
    names = ["projection"] + cv.available_predictions()
    assert sorted(res["arms"]) == sorted(names)
    assert res["config"]["arms"] == names, "score's arm list is the committed roster"
    strata = cv.strata_of(pairs)
    proj = cv.arm_errors(pairs, None)
    for name in names:
        errs = cv.arm_errors(pairs, None if name == "projection" else cv.read_predictions(name))
        assert sorted(res["arms"][name]) == sorted(strata), name
        # every stratum, not only "all" (review of #210, A7): median, fallback, paired gain
        for s, idx in strata.items():
            want = res["arms"][name][s]
            assert want["n_pairs"] == len(idx)
            assert round(float(np.median([errs[i][0] for i in idx])), 4) == \
                pytest.approx(want["median_deg"], abs=1e-4), (name, s)
            assert round(float(np.mean([errs[i][3] for i in idx])), 4) == \
                pytest.approx(want["fallback_rate"], abs=1e-4), (name, s)
            if name == "projection":
                continue
            assert float(np.median([proj[i][0] - errs[i][0] for i in idx])) == \
                pytest.approx(want["median_gain_deg"], abs=1e-4), (name, s)
            used = [i for i in idx if not errs[i][3]]
            assert want["aligned_only"]["n_pairs"] == len(used), (name, s)
            if used:
                assert float(np.median([proj[i][0] - errs[i][0] for i in used])) == \
                    pytest.approx(want["aligned_only"]["median_gain_deg"], abs=1e-4), (name, s)


def test_committed_predictions_are_in_range():
    """read_predictions refuses a non-finite x / y or a y outside [0, 1]; every committed
    file passes (arm_errors would otherwise score y > 1 as a point past the nadir)."""
    for arm in cv.available_predictions():
        for r in cv.read_predictions(arm).values():
            if r["x"] is not None:
                assert 0.0 <= float(r["y"]) <= 1.0, arm


def test_read_predictions_refuses_an_out_of_range_y(tmp_path, monkeypatch):
    d = tmp_path / "crossview_align_48" / "predictions"
    d.mkdir(parents=True)
    (d / "bad.meta.json").write_text(json.dumps({"pairs_sha256": cv.PAIRS_SHA256}))
    (d / "bad.jsonl").write_text(json.dumps({"pair_id": "a", "x": 0.2, "y": 1.3}) + "\n")
    monkeypatch.setattr(cv, "PRED_DIR", str(d))
    with pytest.raises(SystemExit):
        cv.read_predictions("bad")


def test_flat_check_arm_reproduces_the_committed_projection():
    """The geometry arms' plumbing, pinned: recomputing today's projection through
    crossview_arms.geometry lands on the pair list's proj_x / proj_y."""
    pairs = cv.read_frozen_pairs()
    preds = cv.read_predictions("proj_flat_check")
    worst = max(float(cv.angular_error_deg(preds[p["pair_id"]]["x"], preds[p["pair_id"]]["y"],
                                           p["proj_x"], p["proj_y"])) for p in pairs)
    assert worst < 0.01
