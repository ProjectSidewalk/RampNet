"""Tests for scripts/analysis/horizon_band_263.py (Stages 0b and 0c, #263).

CPU only, no torch, no network: the band, bound and strip-geometry helpers on synthetic
peaks and heatmaps, the 0c rules on synthetic numbers, the held-out guards on a temp dir,
and a drift guard that regenerates the committed simulate.json/.md from the #196 caches.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import horizon_band_263 as hb  # noqa: E402
from rampnet.detection_eval import GroundTruth, radius_sq_for, score_pano  # noqa: E402

RSQ = radius_sq_for()
WIDE, NARROW = hb.CORES["wide"], hb.CORES["narrow"]


# --------------------------------------------------------------------------- #
# band helpers
# --------------------------------------------------------------------------- #
def test_in_core_is_inclusive():
    assert hb.in_core(0.45, WIDE) and hb.in_core(0.60, WIDE)
    assert not hb.in_core(0.4499, WIDE) and not hb.in_core(0.6001, WIDE)


def test_band_merge_adds_only_in_core_band_peaks_and_dedupes():
    base = [(0.10, 0.70, 0.90), (0.30, 0.52, 0.20)]          # near hit; sub-threshold far
    band = [(0.301, 0.52, 0.60),                              # same ramp: replaces nothing,
            (0.50, 0.55, 0.50),                               # new in-core peak
            (0.70, 0.40, 0.90),                               # above the core: dropped
            (0.90, 0.55, 0.25)]                               # below t_band: dropped
    out = hb.band_merge(base, band, WIDE, RSQ)
    assert sorted(out) == sorted([(0.10, 0.70, 0.90), (0.301, 0.52, 0.60), (0.50, 0.55, 0.50)])
    # a band peak within the radius of a stronger 1x peak is suppressed, across the seam too
    out = hb.band_merge([(0.999, 0.55, 0.9)], [(0.001, 0.55, 0.8)], WIDE, RSQ)
    assert out == [(0.999, 0.55, 0.9)]
    # a calibrated band threshold applies to the band only
    out = hb.band_merge(base, band, WIDE, RSQ, t_band=0.55)
    assert (0.50, 0.55, 0.50) not in out and (0.301, 0.52, 0.60) in out


def test_far_for_ignore():
    y25 = 0.5 + math.atan(hb.CAM_H / 25.0) / math.pi
    assert hb.is_far_for_ignore(y25 - 1e-6) and not hb.is_far_for_ignore(y25 + 1e-6)
    assert hb.is_far_for_ignore(0.49)                         # above the horizon
    assert not hb.is_far_for_ignore(0.8)


def test_drop_far_unmatched_keeps_every_tp():
    gt = GroundTruth([(0.20, 0.52), (0.40, 0.75)], [], True)
    preds = [(0.20, 0.52, 0.9),     # far TP: kept
             (0.60, 0.52, 0.8),     # far FP: dropped
             (0.80, 0.48, 0.4),     # above the horizon, FP: dropped
             (0.40, 0.75, 0.7),     # near TP
             (0.90, 0.80, 0.5),     # near FP: kept
             (0.201, 0.52, 0.3)]    # far duplicate of a claimed ramp: dropped
    kept = hb.drop_far_unmatched(preds, gt, RSQ)
    assert sorted(kept) == sorted([(0.20, 0.52, 0.9), (0.40, 0.75, 0.7), (0.90, 0.80, 0.5)])
    a, b = score_pano(preds, gt, radius_sq=RSQ), score_pano(kept, gt, radius_sq=RSQ)
    assert a.tp == b.tp == 2 and a.fp == 4 and b.fp == 1
    panos = [{"pano": "p", "gt": gt, "preds": preds}]
    assert hb.apply_bound(panos, "fp", RSQ) is panos
    assert hb.apply_bound(panos, "ignore", RSQ)[0]["preds"] == kept


def test_loso_core_choice_ties_go_to_wide():
    def c(tp, fp, n):
        return {"tp": tp, "fp": fp, "tp_rec": tp, "n_gt": n}
    counts = {"a": {"wide": c(10, 2, 12), "narrow": c(10, 1, 12)},
              "b": {"wide": c(8, 0, 10), "narrow": c(8, 0, 10)},
              "c": {"wide": c(5, 5, 10), "narrow": c(5, 5, 10)}}
    chosen, tab = hb.loso_core_choice(counts)
    assert chosen["b"] == "narrow" and chosen["c"] == "narrow"   # a's narrow wins when a is in
    assert chosen["a"] == "wide"                                 # b + c tie exactly -> wide
    assert tab["a"]["wide"] == tab["a"]["narrow"]


def test_loso_assembly_uses_each_splits_own_choice():
    """On the real data all seven choices agree, so only a synthetic case can show that each
    split's LOSO variant is built with that split's own core."""
    gt = GroundTruth([(0.5, 0.58)], [], True)
    band = [(0.5, 0.58, 0.6)]                       # in the wide core, not in the narrow one
    rows = {c: [(f"{c}-0", gt, [(0.1, 0.8, 0.9)], {s: band for s in hb.SIM_ARMS})]
            for c in ("a", "b")}
    loso = {s: {"a": "wide", "b": "narrow"} for s in hb.SIM_ARMS}
    var = hb.assemble_variants(rows, RSQ, loso, cities=("a", "b"))
    for s in hb.SIM_ARMS:
        assert var[(f"{s}-wide", "a")] != var[(f"{s}-narrow", "a")]
        assert var[(f"{s}-loso", "a")] == var[(f"{s}-wide", "a")]
        assert var[(f"{s}-loso", "b")] == var[(f"{s}-narrow", "b")]


def test_range_bins():
    assert hb.range_bin(0.49) == "at/above horizon"
    assert hb.range_bin(0.9) == "0-8 m"
    assert hb.range_bin(0.5 + math.atan(hb.CAM_H / 30.0) / math.pi) == ">= 25 m"


# --------------------------------------------------------------------------- #
# strip geometry
# --------------------------------------------------------------------------- #
def test_row_step():
    assert hb.row_step(1.5) == 64 and hb.row_step(2.0) == 16 and hb.row_step(1.0) == 32


@pytest.mark.parametrize("cfg", ["1.5x-wide", "1.5x-narrow", "2x-wide", "2x-narrow"])
def test_strip_geometry_is_aligned_and_keeps_the_context(cfg):
    scale, core = hb.config_parts(cfg)
    g = hb.strip_geometry(hb.CORES[core], hb.SCALES[scale])
    assert g["r0"] / 2048 <= hb.CORES[core][0] - hb.CONTEXT + 1e-9
    assert g["r1"] / 2048 >= hb.CORES[core][1] + hb.CONTEXT - 1e-9
    assert g["R0"] % hb.STRIDE == 0 and g["in_h"] % hb.STRIDE == 0
    assert g["in_w"] % hb.STRIDE == 0 and g["wrap_in"] % hb.STRIDE == 0
    assert (g["hm_h"], g["hm_w"]) == (g["in_h"] // 4, g["in_w"] // 4)
    assert g["full_hm"] == [int(2048 * hb.SCALES[scale]) // 4, int(4096 * hb.SCALES[scale]) // 4]
    assert g["min_distance"] == round(10 * hb.SCALES[scale])


def test_strip_geometry_1p5x_wide_numbers():
    g = hb.strip_geometry(WIDE, 1.5)
    assert (g["r0"], g["r1"], g["in_h"], g["in_w"], g["hm_h"], g["hm_w"]) == \
        (768, 1344, 864, 6528, 216, 1632)
    assert (g["wrap_in"], g["wrap_hm"], g["row_off_hm"]) == (192, 48, 288)


def test_make_strip_wraps_columns():
    g = hb.strip_geometry(WIDE, 1.5, hw=(64, 128), wrap_1x=64, stride=32)
    full = np.arange(1 * 96 * 192, dtype=np.float32).reshape(1, 96, 192)
    s = hb.make_strip(full, g)
    w = g["wrap_in"]
    assert s.shape == (1, g["in_h"], g["in_w"])
    np.testing.assert_array_equal(s[:, :, :w], full[:, g["R0"]:g["R0"] + g["in_h"], -w:])
    np.testing.assert_array_equal(s[:, :, w:-w], full[:, g["R0"]:g["R0"] + g["in_h"], :])
    np.testing.assert_array_equal(s[:, :, -w:], full[:, g["R0"]:g["R0"] + g["in_h"], :w])


def _blob_heatmap(h, w, peaks, sigma=3.0):
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w), dtype=np.float64)
    for r, c, a in peaks:
        dx = np.minimum(np.abs(xx - c), w - np.abs(xx - c))    # wrapped in x
        out = np.maximum(out, a * np.exp(-((yy - r) ** 2 + dx ** 2) / (2 * sigma ** 2)))
    return out


def test_strip_peaks_land_where_the_full_pass_puts_them():
    """A perfect-context strip heatmap (the full heatmap's rows, wrapped) must give exactly
    the full pass's in-core peaks, at the same normalized coordinates, seam included."""
    import threshold_sweep as ts
    g = hb.strip_geometry(WIDE, 1.5)
    fh, fw = g["full_hm"]
    peaks = [(int(0.50 * fh), 100, 0.9), (int(0.58 * fh), fw - 3, 0.7),   # in core, at the seam
             (int(0.47 * fh), 900, 0.5), (int(0.30 * fh), 400, 0.8),      # in core; above it
             (int(0.62 * fh), 1200, 0.6)]                                 # in context, not core
    full = _blob_heatmap(fh, fw, peaks)
    want = [p for p in ts.peaks_to_dets(full, 0.05, g["min_distance"]) if hb.in_core(p[1], WIDE)]
    rows = full[g["row_off_hm"]:g["row_off_hm"] + g["hm_h"]]
    w = g["wrap_hm"]
    strip = np.concatenate([rows[:, -w:], rows, rows[:, :w]], axis=1)
    assert strip.shape == (g["hm_h"], g["hm_w"])
    got = hb.strip_peaks_to_pano(hb.peaks_rc(strip, 0.05, g["min_distance"]), g)
    # Away from the seam the two agree exactly. At the seam the unwrapped full heatmap also
    # reports the blob's tail at col 0 as a border maximum (exclude_border=False); the strip,
    # whose wrap shows peak_local_max both sides, reports the blob once, where it is.
    seam = [p for p in want if p[0] < 0.01]
    assert len(seam) == 1 and seam[0][2] < 0.7
    assert sorted(got) == sorted(p for p in want if p[0] >= 0.01) and len(got) == 3
    assert any(abs(p[0] - (fw - 3) / fw) < 1e-12 for p in got)


# --------------------------------------------------------------------------- #
# 0c rules
# --------------------------------------------------------------------------- #
def test_context_stats():
    full = [(0.1 * i, 0.52, 0.5) for i in range(1, 10)] + [(0.95, 0.52, 0.1)]  # last < 0.30
    same = hb.context_stats([(full, list(full))], RSQ)
    assert same["n_ref"] == 9 and same["reproduced_fraction"] == 1.0
    assert same["median_score_shift"] == 0.0 and same["passes"] and same["passes_strict"]
    lost = hb.context_stats([(full, full[1:])], RSQ)               # 8 of 9
    assert abs(lost["reproduced_fraction"] - 8 / 9) < 1e-12 and not lost["passes"]
    weak = [(x, y, s - 0.25) for x, y, s in full]                  # all present, 0.25 lower
    w = hb.context_stats([(full, weak)], RSQ)
    assert w["reproduced_fraction"] == 1.0 and not w["passes"]     # median shift -0.25
    assert w["reproduced_fraction_strict"] == 0.0
    small = [(x, y, s - 0.04) for x, y, s in full]
    assert hb.context_stats([(full, small)], RSQ)["passes"]


def test_paired_shifts_is_one_to_one_and_needs_one_side_at_threshold():
    base = [(0.20, 0.52, 0.40), (0.60, 0.52, 0.10)]
    strip = [(0.201, 0.52, 0.50), (0.205, 0.52, 0.90), (0.601, 0.52, 0.20)]
    out = hb.paired_shifts(base, strip, RSQ)
    assert out == [pytest.approx(0.10)]       # closest pair only; the 0.10/0.20 pair never acts


def test_calibration_rule_and_threshold():
    assert not hb.calibration_rule(0.03, 0.80, 0.85)
    assert hb.calibration_rule(-0.031, 0.80, 0.80)
    assert hb.calibration_rule(0.0, 0.80, 0.851)
    prec = {t: 0.70 + (t - 0.05) * 0.4 for t in hb.MATCH_GRID}
    assert hb.band_threshold(prec, 0.80) == pytest.approx(0.30)
    assert hb.band_threshold(prec, 1.5) is None


def _c(lo, hi):
    return {"observed": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi}


def test_gate_bound_and_verdict():
    good = {"recall": _c(0.01, 0.05), "f1": _c(-0.01, 0.02)}
    m_good = {"f1": _c(0.001, 0.03)}
    assert hb.gate_bound(good, m_good)["passes"]
    assert not hb.gate_bound(good, None)["passes"]                       # unreachable = not beaten
    assert not hb.gate_bound({"recall": _c(0.0, 0.05), "f1": _c(0, 0.1)}, m_good)["passes"]
    assert not hb.gate_bound({"recall": _c(0.01, 0.05), "f1": _c(-0.03, -0.001)}, m_good)["passes"]
    assert not hb.gate_bound(good, {"f1": _c(-0.001, 0.03)})["passes"]
    ok, bad = hb.gate_bound(good, m_good), hb.gate_bound(good, None)
    assert hb.gate_verdict({"fp": ok, "ignore": ok}) == "PASS"
    assert hb.gate_verdict({"fp": ok, "ignore": bad}) == "FAIL"


def test_open_rig_verdict():
    assert hb.open_rig_verdict(_c(0.001, 0.05)) == "OFFERED"            # observed 0.0255
    assert hb.open_rig_verdict({"observed": 0.019, "ci_lo": 0.005, "ci_hi": 0.03}) == "NOT OFFERED"
    assert hb.open_rig_verdict({"observed": 0.03, "ci_lo": -0.001, "ci_hi": 0.06}) == "NOT OFFERED"


def test_passes_for():
    assert hb.passes_for("bend", hb.CONFIGS_0C) == [
        "r2048", "full@1.5x", "full@2x", "strip@1.5x-wide", "strip@1.5x-narrow", "strip@2x-wide"]
    assert hb.passes_for("manual_gold", ["1.5x-wide"]) == ["r2048", "strip@1.5x-wide"]


# --------------------------------------------------------------------------- #
# held-out guards
# --------------------------------------------------------------------------- #
def _cal(path, passes=True, smoke=False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"smoke": smoke, "context": {hb.PRIMARY_0C: {"passes": passes}}}, f)


def test_guard_extract_refusals(tmp_path):
    root, cal = str(tmp_path / "cache"), str(tmp_path / "calibration.json")
    os.makedirs(root)
    g = hb.guard_extract
    assert g(["bend"], False, root, cal, 0, False) is None
    with pytest.raises(SystemExit, match="not Stage 0c splits"):
        g(["laurens_mapillary"], False, root, cal, 0, False)
    with pytest.raises(SystemExit, match="heldout-read-once"):
        g(["manual_gold"], False, root, cal, 0, False)
    with pytest.raises(SystemExit, match="separately"):
        g(["bend", "juneau"], True, root, cal, 0, False)
    with pytest.raises(SystemExit, match="--limit"):
        g(["juneau"], True, root, cal, 5, False)
    with pytest.raises(SystemExit, match="calibrate"):
        g(["juneau"], True, root, cal, 0, False)
    _cal(cal, smoke=True)
    with pytest.raises(SystemExit, match="smoke"):
        g(["juneau"], True, root, cal, 0, False)
    _cal(cal, passes=False)
    with pytest.raises(SystemExit, match="context rule FAILED"):
        g(["juneau"], True, root, cal, 0, False)
    assert g(["juneau"], True, root, cal, 0, False, despite_failed_context=True)
    _cal(cal)
    sha = g(["juneau", "bayonne"], True, root, cal, 0, False)
    assert sha == hb.sha256_file(cal)
    open(os.path.join(root, "bayonne.json"), "w").close()
    with pytest.raises(SystemExit, match="read once"):
        g(["juneau", "bayonne"], True, root, cal, 0, False)
    with pytest.raises(SystemExit, match="scratch"):
        g(["bend"], False, hb.CACHE_0C, cal, 3, False)


def test_simulate_never_reads_a_held_out_split():
    touched = set(hb.S7 + hb.REPORT_ONLY)
    assert not touched & set(hb.HELD_OUT)
    assert not set(hb.SELECTION) & set(hb.HELD_OUT)


# --------------------------------------------------------------------------- #
# drift guard
# --------------------------------------------------------------------------- #
def test_committed_simulate_reproduces():
    rep = hb.build_sim()
    js = json.dumps(rep, indent=1) + "\n"
    md = hb.markdown_sim(rep) + "\n"
    with open(hb.SIM_JSON, encoding="utf-8", newline="") as f:
        assert f.read() == js
    with open(os.path.splitext(hb.SIM_JSON)[0] + ".md", encoding="utf-8", newline="") as f:
        assert f.read() == md
    s7 = rep["pools"]["S7"]
    assert s7["base"]["fp"]["F1"] == 0.8417                   # the plan's base F1
    plan = {"2x-wide": (100, 174, 357), "2x-narrow": (89, 146, 296),
            "1.5x-wide": (106, 146, 402), "1.5x-narrow": (95, 120, 327)}
    for cfg, (g, f, m) in plan.items():
        e = s7["configs"][cfg]["fp"]
        assert (e["ramps_gained"], e["fps_added"],
                e["matched_recall_threshold"]["extra_fp_vs_base"]) == (g, f, m)
    # what the doc says beside the LOSO sentence: narrow is not separated from wide on F1;
    # it trades recall for precision
    for scale in hb.SIM_ARMS:
        d = rep["narrow_minus_wide_S7"][scale]["fp"]
        assert d["f1"]["ci_lo"] < 0 < d["f1"]["ci_hi"]
        assert d["recall"]["ci_hi"] < 0 < d["precision"]["ci_lo"]


# --------------------------------------------------------------------------- #
# calibrate -> report, end to end on synthetic caches (no GPU, no real held-out data)
# --------------------------------------------------------------------------- #
# Every synthetic pano has the same three ramps and the same peaks, so every bootstrap interval
# collapses to its observed value and each verdict below is exact, not a sampling outcome.
NEAR, MID, FAR = (0.20, 0.75), (0.80, 0.55), (0.50, 0.52)   # MID and FAR are in both cores
FP_1X = (0.35, 0.80)       # near field, outside every core: never dropped by the ignore bound
FP_STRIP = (0.65, 0.50)    # inside every core


def _one_x_default(i):
    """1x finds NEAR and the in-core MID ramp at 0.30; FAR only at 0.08; an FP at 0.15."""
    return [(*NEAR, 0.8), (*MID, 0.7), (*FAR, 0.08), (*FP_1X, 0.15)]


def _strip_default(name, i):
    """Every strip and full pass finds FAR and MID; an in-core FP at 0.40 on every fourth pano."""
    return [(*FAR, 0.6), (*MID, 0.7)] + ([(*FP_STRIP, 0.4)] if i % 4 == 0 else [])


def _write_cache(root, split, n=12, cal_sha=None, one_x=_one_x_default, strip=_strip_default,
                 full=None, s_base=0.5, s_strip=0.25):
    """A synthetic Stage 0c cache. ``full`` defaults to the strip's peaks (perfect context).
    Timings are powers of two so the cost multiple is exact."""
    names = hb.passes_for(split, hb.CONFIGS_0C)
    panos = []
    for i in range(n):
        passes = {"r2048": one_x(i)}
        for name in names[1:]:
            if name.startswith("full@"):
                cfg = next(c for c in hb.CONFIGS_0C if c.startswith(name[len("full@"):] + "-"))
                passes[name] = (full or strip)(f"strip@{cfg}", i)
            else:
                passes[name] = strip(name, i)
        panos.append({"pano": f"{split}-{i:03d}",
                      "gt": {"gt_points": [list(NEAR), list(MID), list(FAR)],
                             "ignore_points": [], "fn_confirmed": True},
                      "passes": {k: [list(t) for t in v] for k, v in passes.items()}})
    timing = {name: {"elapsed_s": (s_base if name == "r2048" else s_strip) * n, "n": n,
                     "s_per_pano": s_base if name == "r2048" else s_strip, "peak_mem_gib": 5.0}
              for name in names}
    meta = {"split": split, "held_out": cal_sha is not None, "calibration_sha256": cal_sha,
            "limit": 0, "configs": list(hb.CONFIGS_0C), "timing": timing}
    with open(hb.cache_file(root, split), "w", encoding="utf-8") as f:
        json.dump({"city": split, "meta": meta, "panos": panos}, f)


def _calibrate(root, edit=None, **cache_kw):
    """Selection caches -> calibration.json (through the CLI, as run0c does). ``edit`` may
    change the parsed calibration before it is rewritten. Returns (path, sha256)."""
    for s in hb.SELECTION:
        _write_cache(root, s, **cache_kw)
    path = os.path.join(root, "calibration.json")
    assert hb.main(["calibrate", "--cache-root", root, "--out", path]) == 0
    if edit:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        edit(cal)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cal, f)
    return path, hb.sha256_file(path)


def _held_out(root, sha, **kw):
    for s in hb.HELD_OUT:
        _write_cache(root, s, n=30, cal_sha=sha, **kw)


def test_calibrate_then_report_end_to_end(tmp_path):
    root = str(tmp_path)
    path, sha = _calibrate(root)
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    assert cal["smoke"] is False
    for cfg in hb.CONFIGS_0C:
        assert cal["context"][cfg]["passes"] and cal["context"][cfg]["median_score_shift"] == 0.0
        c = cal["calibration"][cfg]
        # 1x in-core precision at 0.30 is 1.0 (MID only); the strip's is 24/27, so the
        # threshold is set, at the lowest grid value above the FP's 0.40
        assert c["precision_1x_in_core_at_030"] == 1.0 and c["n_1x_in_core_at_030"] == 12
        assert c["threshold_set"] and c["band_threshold"] == 0.41
    _held_out(root, sha)
    rep = hb.build_gate(root, path)
    for cfg in hb.CONFIGS_0C:
        g = rep["gate"][cfg]
        assert g["verdict"] == "PASS" and g["open_rig"] == "OFFERED"
        assert all(b["passes"] for b in g["bounds"].values())
        e = rep["manual_gold"]["configs"][cfg]["fp"]
        # the calibrated 0.41 keeps the band's 0.40 FP out; at 0.30 it would add 8
        assert e["ramps_gained"] == 30 and e["fps_added"] == 0
    assert rep["decision"] == "band kept for the shipped weights"
    assert rep["cost"]["configs"][hb.PRIMARY_0C]["multiple"] == 1.5
    assert rep["two_x_rule"]["ratio"] == 1.0 and not rep["two_x_rule"]["gains_enough"]
    hb.markdown_gate(rep)
    # a held-out cache read under another calibration is refused
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="calibration"):
        hb.build_gate(root, path)


def test_binding_config_is_the_wide_1p5x_strip(tmp_path):
    """The decision follows 1.5x-wide alone: here it fails (its strip finds nothing new on the
    held-out splits) while the other two pass, and the band is dropped."""
    assert hb.PRIMARY_0C == "1.5x-wide"
    root = str(tmp_path)
    path, sha = _calibrate(root)

    def strip(name, i):
        return [(*MID, 0.7)] if name == "strip@1.5x-wide" else _strip_default(name, i)
    _held_out(root, sha, strip=strip)
    rep = hb.build_gate(root, path)
    assert rep["gate"]["1.5x-wide"]["verdict"] == "FAIL"
    assert rep["gate"]["1.5x-narrow"]["verdict"] == "PASS"
    assert rep["decision"].startswith("band dropped for the shipped weights")


@pytest.mark.parametrize("s_strip,within", [(0.375, True), (0.4, False)])
def test_cost_limit_is_reported_in_the_decision_not_enforced(tmp_path, s_strip, within):
    root = str(tmp_path)
    path, sha = _calibrate(root, s_strip=s_strip)        # cost pools every 0c cache
    _held_out(root, sha, s_strip=s_strip)
    rep = hb.build_gate(root, path)
    c = rep["cost"]["configs"][hb.PRIMARY_0C]
    assert c["multiple"] == (0.5 + s_strip) / 0.5 and c["within_limit"] is within
    assert rep["gate"][hb.PRIMARY_0C]["verdict"] == "PASS"           # the gate ignores cost
    if within:
        assert rep["decision"] == "band kept for the shipped weights"
    else:
        assert rep["decision"].startswith("passes the gate; over the 1.75x cost limit")
        assert "1.800x" in rep["decision"] and "Jon's call" in rep["decision"]


def test_failed_context_override_is_named_in_the_decision(tmp_path):
    root = str(tmp_path)

    def fail_context(cal):
        cal["context"][hb.PRIMARY_0C]["passes"] = False
    path, sha = _calibrate(root, edit=fail_context)
    _held_out(root, sha)
    rep = hb.build_gate(root, path)
    assert rep["gate"][hb.PRIMARY_0C]["verdict"] == "PASS"
    assert "context rule failed" in rep["decision"] and "Jon's call" in rep["decision"]


def test_uncalibratable_config_is_forced_to_fail(tmp_path, monkeypatch):
    root = str(tmp_path)

    def no_threshold(cal):
        cal["calibration"][hb.PRIMARY_0C]["band_threshold"] = None
    path, sha = _calibrate(root, edit=no_threshold)
    _held_out(root, sha)
    monkeypatch.setattr(hb, "gate_bound", lambda d_base, d_matched: {"passes": True})
    rep = hb.build_gate(root, path)
    assert rep["gate"][hb.PRIMARY_0C]["verdict"] == "FAIL"
    assert rep["gate"]["1.5x-narrow"]["verdict"] == "PASS"


def test_calibrate_refuses_without_1x_in_core_detections(tmp_path):
    """No 1x in-core detection at 0.30 on sao_paulo leaves nothing for the band threshold to
    equal: calibrate stops rather than return the grid floor."""
    root = str(tmp_path)
    for s in hb.SELECTION:
        _write_cache(root, s, one_x=lambda i: [(*NEAR, 0.8), (*FAR, 0.08)])
    with pytest.raises(SystemExit, match="no 1x in-core detection"):
        hb.build_calibration(root)


def test_context_bars():
    full = [(0.009 * i, 0.52, 0.5) for i in range(100)]             # 100 reference peaks
    assert hb.context_stats([(full, full[6:])], RSQ)["passes"] is False     # 94%
    assert hb.context_stats([(full, full[5:])], RSQ)["passes"] is True      # 95%
    lower = [(x, y, s - 0.06) for x, y, s in full]
    assert hb.context_stats([(full, lower)], RSQ)["passes"] is False
    lower = [(x, y, s - 0.049) for x, y, s in full]
    assert hb.context_stats([(full, lower)], RSQ)["passes"] is True


def test_calibration_bytes_do_not_depend_on_git_head(tmp_path, monkeypatch):
    """Committing calibration.json moves HEAD; --check must still pass afterwards."""
    root = str(tmp_path)
    path, _ = _calibrate(root)
    monkeypatch.setattr(hb, "git_head", lambda: "0" * 40)
    assert hb.main(["calibrate", "--cache-root", root, "--out", path, "--check"]) == 0


# --------------------------------------------------------------------------- #
# run0c on a fake extract (the real one needs the GPU and the panos)
# --------------------------------------------------------------------------- #
def _fake_extract(calls, die_at=None, **cache_kw):
    """Stands in for cmd_extract: the real guard, then a synthetic cache per split. ``die_at``
    names a split at which the run dies before writing its cache."""
    def fake(args):
        splits = hb.resolve_splits(args.splits)
        calls.append(splits)
        sha = hb.guard_extract(splits, args.heldout_read_once, args.cache_root,
                               args.calibration, args.limit, args.force,
                               args.despite_failed_context)
        for s in splits:
            if os.path.exists(hb.cache_file(args.cache_root, s)):
                continue
            if s == die_at:
                raise SystemExit(f"{s}: died")
            _write_cache(args.cache_root, s, n=30 if sha else 12, cal_sha=sha, **cache_kw)
        return 0
    return fake


def _run0c(root):
    return hb.main(["run0c", "--cache-root", root, "--panos-root", root,
                    "--calibration", os.path.join(root, "calibration.json"),
                    "--out", os.path.join(root, "gate_0c.json"), "--usage-log", "none"])


def test_run0c_stops_before_the_held_out_read_when_the_context_rule_fails(tmp_path, monkeypatch):
    root = str(tmp_path)
    calls = []
    monkeypatch.setattr(hb, "cmd_extract", _fake_extract(
        calls, strip=lambda name, i: [(*MID, 0.7)], full=_strip_default))   # strip loses FAR
    monkeypatch.setattr(hb, "cmd_report", lambda args: pytest.fail("report must not run"))
    assert _run0c(root) == 4
    assert calls == [list(hb.SELECTION)]
    assert not any(os.path.exists(hb.cache_file(root, s)) for s in hb.HELD_OUT)


def test_run0c_resumes_a_partial_held_out_read_and_keeps_its_calibration(tmp_path, monkeypatch):
    root = str(tmp_path)
    cal = os.path.join(root, "calibration.json")
    calls = []
    monkeypatch.setattr(hb, "cmd_extract", _fake_extract(calls, die_at="juneau"))
    with pytest.raises(SystemExit, match="juneau: died"):
        _run0c(root)
    assert os.path.exists(hb.cache_file(root, "manual_gold"))
    with open(cal, "rb") as f:
        frozen = f.read()
    # the second run: another HEAD (a commit happened), and juneau no longer dies
    monkeypatch.setattr(hb, "git_head", lambda: "f" * 40)
    calls.clear()
    monkeypatch.setattr(hb, "cmd_extract", _fake_extract(calls))
    assert _run0c(root) == 0
    assert calls == [list(hb.SELECTION), ["juneau", "bayonne"]]  # manual_gold not read again
    with open(cal, "rb") as f:
        assert f.read() == frozen
    with open(os.path.join(root, "gate_0c.json"), encoding="utf-8") as f:
        assert json.load(f)["decision"] == "band kept for the shipped weights"


def test_run0c_never_rewrites_a_frozen_calibration(tmp_path, monkeypatch):
    """A calibration.json that no longer reproduces from the selection caches stops run0c;
    it is not silently recalibrated (the held-out caches record its sha256)."""
    root = str(tmp_path)

    def tamper(cal):
        cal["calibration"][hb.PRIMARY_0C]["band_threshold"] = 0.5
    path, _ = _calibrate(root, edit=tamper)
    with open(path, "rb") as f:
        frozen = f.read()
    calls = []
    monkeypatch.setattr(hb, "cmd_extract", _fake_extract(calls))
    monkeypatch.setattr(hb, "cmd_report", lambda args: pytest.fail("report must not run"))
    assert _run0c(root) != 0
    assert calls == [list(hb.SELECTION)]
    with open(path, "rb") as f:
        assert f.read() == frozen


def test_run0c_refuses_a_held_out_cache_from_another_calibration(tmp_path, monkeypatch):
    root = str(tmp_path)
    calls = []
    monkeypatch.setattr(hb, "cmd_extract", _fake_extract(calls))
    for s in hb.SELECTION:
        _write_cache(root, s)
    _write_cache(root, "manual_gold", n=30, cal_sha="0" * 64)
    monkeypatch.setattr(hb, "cmd_report", lambda args: pytest.fail("report must not run"))
    assert _run0c(root) != 0
    assert calls == [list(hb.SELECTION)]


# --------------------------------------------------------------------------- #
# extract on a fake model (no checkpoint, no panos: zero tensors of the right shapes)
# --------------------------------------------------------------------------- #
class _ExtractHarness:
    """Patches the model, the image decode and the GT reader so ``cmd_extract`` runs on the
    CPU in seconds. ``fail_at`` = (split, pano index) at which decoding fails."""

    def __init__(self, tmp_path, monkeypatch, fail_at=None):
        torch = pytest.importorskip("torch")
        import operating_point_curve
        import threshold_sweep

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                                                torch.nn.Upsample(size=(512, 1024)))

            def forward(self, x):
                return torch.zeros((x.shape[0], 1) + tuple(self.head[2].size))

        def gts(city):
            gt = GroundTruth([(0.5, 0.52)], [], True)
            return {f"{city}-{i}": gt for i in range(2)}, None

        def prefetch(paths, scales):
            zero = torch.zeros(1, 1, 1)
            for k, p in enumerate(paths):
                if fail_at and os.path.basename(os.path.dirname(os.path.dirname(p))) == \
                        fail_at[0] and k == fail_at[1]:
                    yield p, None, None, OSError("bad jpeg")
                    continue
                t = {"r2048": zero.expand(1, 2048, 4096)}
                for s in scales:
                    t[s] = zero.expand(1, int(2048 * hb.SCALES[s]), int(4096 * hb.SCALES[s]))
                yield p, (4096, 2048), t, None

        monkeypatch.setattr(threshold_sweep, "load_model", lambda: FakeModel())
        monkeypatch.setattr(operating_point_curve, "bundle_ground_truths", gts)
        monkeypatch.setattr(hb, "_prefetch", prefetch)
        self.root = str(tmp_path)
        self.cache = os.path.join(self.root, "cache")
        self.log = os.path.join(self.root, "usage.jsonl")
        for city in hb.SELECTION:
            d = os.path.join(self.root, "benchmark", city, "panos")
            os.makedirs(d)
            for i in range(2):
                open(os.path.join(d, f"{city}-{i}.jpg"), "w").close()

    def extract(self, splits):
        return hb.main(["extract", "--splits", splits, "--configs", "1.5x-narrow",
                        "--cache-root", self.cache, "--panos-root", self.root,
                        "--usage-log", self.log, "--device", "cpu"])

    def rows(self):
        with open(self.log, encoding="utf-8") as f:
            return {r["label"]: r for r in map(json.loads, f)}


def test_extract_ledger_counts_the_split_it_died_in(tmp_path, monkeypatch):
    h = _ExtractHarness(tmp_path, monkeypatch, fail_at=("richmond", 1))
    with pytest.raises(SystemExit, match="bad jpeg"):
        h.extract("sao_paulo,richmond")
    assert os.path.exists(hb.cache_file(h.cache, "sao_paulo"))
    assert not os.path.exists(hb.cache_file(h.cache, "richmond"))
    rows = h.rows()
    assert rows["horizon-band-263:r2048"]["status"] == "failed"
    assert rows["horizon-band-263:r2048"]["panos_scored"] == 3       # 2 sao_paulo + 1 richmond
    assert rows["horizon-band-263:full@1.5x"]["panos_scored"] == 1   # richmond only (S7)


def test_extract_never_leaves_a_partial_cache_under_the_final_name(tmp_path, monkeypatch):
    h = _ExtractHarness(tmp_path, monkeypatch)

    def crash(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(hb.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        h.extract("sao_paulo")
    assert not os.path.exists(hb.cache_file(h.cache, "sao_paulo"))


