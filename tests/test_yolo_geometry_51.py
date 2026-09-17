"""Tests for the #51 geometry-pair read (scripts/analysis/yolo_geometry_51.py).

The thing worth protecting here is not the arithmetic, it is the CONTROL. The whole
comparison is only meaningful because the published `y11x_pano_h200` leg re-scores to
its committed row under current code; if a change to the matcher or the scorer moves
YOLO scoring, a geometry conclusion drawn from these numbers is wrong. So the control
check gets tested in both directions: it passes on the committed data, and it actually
fails when the numbers move.
"""

import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "analysis", "yolo_geometry_51.py")
DATA_DIR = os.path.join(ROOT, "docs", "data", "yolo_geometry_51")
JSON_PATH = os.path.join(ROOT, "docs", "data", "yolo_geometry_51.json")


def _load():
    spec = importlib.util.spec_from_file_location("yolo_geometry_51", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def committed():
    with open(JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_row_regex_parses_a_real_table_line(mod):
    line = ("y11x_pano_h200                        0.986   (0.927-0.998)  "
            "0.248   (0.202-0.301)  0.397  0.662         73/1/221/0")
    m = mod.ROW_RE.match(line)
    assert m, "the operating-point row format changed"
    assert m.group("model") == "y11x_pano_h200"
    assert float(m.group("p")) == 0.986
    assert float(m.group("f1")) == 0.397
    assert (int(m.group("tp")), int(m.group("fp")), int(m.group("fn"))) == (73, 1, 221)


def test_row_regex_ignores_the_header(mod):
    header = ("model                                     P          95% CI      "
              "R          95% CI     F1     AP       tp/fp/fn/ign")
    assert mod.ROW_RE.match(header) is None


def test_the_split_population_is_pinned_not_taken_from_the_live_registry(mod):
    """A frozen study must not follow the benchmark registry.

    ``analysis.low_floor_sweep.US_SPLITS`` grows as splits are added (laurens in #152,
    for one). This study scored ten splits on 2026-08-30 and no others, so a new split
    has no report here: following the registry would make every pooled number ``None``
    and take the committed artifact down with it. Pinned tuples, asserted literally.
    """
    assert mod.POOLED_SPLITS == ("richmond", "bend", "clovis", "morgantown",
                                 "annapolis", "paterson", "gainesville")
    assert mod.ALL_SPLITS_AS_RUN == mod.POOLED_SPLITS + (
        "budapest_district5", "sao_paulo", "manual_gold")
    assert set(mod.HELD_OUT_AS_RUN) == set(mod.ALL_SPLITS_AS_RUN) - set(mod.POOLED_SPLITS)


def test_every_split_has_all_three_legs(mod):
    cells, _ = mod.collect()
    assert set(cells) == set(mod.LEGS), "a leg went missing from the reports"
    for leg, per_split in cells.items():
        missing = set(mod.ALL_SPLITS_AS_RUN) - set(per_split)
        assert not missing, f"{leg} is missing splits: {sorted(missing)}"


def test_control_reproduces_the_published_scoreboard_row(mod):
    """The load-bearing assertion: current code must re-derive the 2026-08-14 numbers."""
    cells, _ = mod.collect()
    ctrl = mod.pooled(cells["y11x_pano_h200"])
    assert ctrl is not None
    assert round(ctrl["macro_p"], 3) == mod.PUBLISHED_CONTROL["p"]
    assert round(ctrl["macro_r"], 3) == mod.PUBLISHED_CONTROL["r"]
    assert round(ctrl["macro_f1"], 3) == mod.PUBLISHED_CONTROL["f1"]


def test_control_check_would_actually_fail_if_scoring_moved(mod):
    """A check that cannot fail is not a check. Perturb one split, expect a mismatch."""
    cells, _ = mod.collect()
    perturbed = {s: dict(v) for s, v in cells["y11x_pano_h200"].items()}
    perturbed["richmond"]["f1"] += 0.05
    ctrl = mod.pooled(perturbed)
    assert round(ctrl["macro_f1"], 3) != mod.PUBLISHED_CONTROL["f1"]


def test_pooled_uses_exactly_the_seven_us_splits(mod):
    cells, _ = mod.collect()
    pool = mod.pooled(cells["y11x_tiles"])
    assert pool["n_splits"] == len(mod.POOLED_SPLITS) == 7
    # held-out splits must not leak into the headline
    for held in ("budapest_district5", "sao_paulo", "manual_gold"):
        assert held not in mod.POOLED_SPLITS


def test_pooled_macro_mean_is_a_plain_mean(mod):
    cells, _ = mod.collect()
    per_split = cells["y11x_tiles"]
    pool = mod.pooled(per_split)
    expect = sum(per_split[s]["f1"] for s in mod.POOLED_SPLITS) / len(mod.POOLED_SPLITS)
    assert pool["macro_f1"] == pytest.approx(expect)


def test_pooled_returns_none_when_a_split_is_absent(mod):
    cells, _ = mod.collect()
    partial = {s: v for s, v in cells["y11x_tiles"].items() if s != "bend"}
    assert mod.pooled(partial) is None


def test_committed_json_matches_the_reports(mod, committed):
    """docs/data/yolo_geometry_51.json must be re-derivable from the committed reports."""
    cells, best = mod.collect()
    pools = {m: mod.pooled(v) for m, v in cells.items()}
    assert mod.rnd(cells) == committed["per_split"]
    assert mod.rnd(pools) == committed["pooled"]
    assert mod.rnd(best) == committed["best_f1_sweep_tune_on_test"]
    assert committed["control_reproduced"] is True


def test_decomposition_sums_to_the_total(committed):
    d = committed["decomposition"]
    assert d["budget"] + d["geometry"] == pytest.approx(d["total"], abs=5e-4)


def test_decomposition_is_smaller_than_the_published_gap(committed):
    """The geometry finding must never be reported as closing more than the gap."""
    d = committed["decomposition"]
    assert 0 < d["total"] < d["published_gap"]
    assert d["residual"] == pytest.approx(d["published_gap"] - d["total"], abs=5e-4)


def test_tiles_beats_both_pano_arms_pooled(committed):
    """The headline claim of the doc, pinned so a data refresh cannot silently invert it."""
    pooled = committed["pooled"]
    assert pooled["y11x_tiles"]["macro_f1"] > pooled["y11x_pano"]["macro_f1"]
    assert pooled["y11x_pano"]["macro_f1"] > pooled["y11x_pano_h200"]["macro_f1"]


def test_geometry_gain_is_recall_not_precision(committed):
    """Stated in the doc as the mechanism; if it flips, the doc is wrong."""
    p = committed["pooled"]
    d_r = p["y11x_tiles"]["macro_r"] - p["y11x_pano"]["macro_r"]
    d_p = p["y11x_tiles"]["macro_p"] - p["y11x_pano"]["macro_p"]
    assert d_r > 0.02
    assert abs(d_p) < d_r


def test_json_is_lf_and_has_no_long_floats():
    """Committed JSON must not churn on line endings or numpy float repr."""
    with open(JSON_PATH, "rb") as fh:
        raw = fh.read()
    assert b"\r\n" not in raw, "JSON must be LF-only"
    text = raw.decode("utf-8")
    for tok in text.replace(",", " ").replace(":", " ").split():
        if tok.replace(".", "", 1).replace("-", "", 1).isdigit() and "." in tok:
            assert len(tok.split(".")[1]) <= 4, f"unrounded float in committed JSON: {tok}"


def test_reports_carry_checkpoint_provenance():
    """A number whose checkpoint hash is not written down is not reproducible."""
    with open(os.path.join(DATA_DIR, "env.txt"), encoding="utf-8") as fh:
        env = fh.read()
    assert "repo HEAD" in env
    assert "ultralytics" in env
    for leg in ("y11x_tiles.pt", "y11x_pano.pt", "y11x_pano_h200.pt"):
        assert leg in env, f"{leg} sha256 missing from env.txt"
    # 64-hex sha256s, one per checkpoint
    hexes = [t for t in env.split() if len(t) == 64 and all(c in "0123456789abcdef" for c in t)]
    assert len(hexes) == 3
