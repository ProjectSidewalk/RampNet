"""Tests for scripts/analysis/input_res_sweep_25.py (issue #25, frozen-model input sweep).

CPU only, no network, committed fixtures only: the model is built with
``pretrained_backbone=False`` and the report tests read the committed caches under
analysis_out/input_res_sweep_25/cache/.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import input_res_sweep_25 as irs  # noqa: E402


# (a) arm preprocessing -------------------------------------------------------------
def test_arm_sizes_and_resize_order():
    assert irs.resize_steps("r2048", (11000, 5500)) == [((2048, 4096), "bilinear")]
    assert irs.resize_steps("r3072", (8000, 4000)) == [((3072, 6144), "bilinear")]
    assert irs.resize_steps("r4096", (16384, 8192)) == [((4096, 8192), "bilinear")]
    # the upsample control: bilinear down to the model size FIRST, then bicubic up
    assert irs.resize_steps("u4096", (11000, 5500)) == [((2048, 4096), "bilinear"),
                                                        ((4096, 8192), "bicubic")]
    assert irs.resize_steps("r4096_hm1024", (8000, 4000)) == [((4096, 8192), "bilinear")]


def test_rnative_is_native_floored_and_capped():
    assert irs.arm_input_size("rnative", (11000, 5500)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (5660, 2830)) == (2830, 5660)
    assert irs.arm_input_size("rnative", (16384, 8192)) == (5500, 11000)   # the cap
    assert irs.arm_input_size("rnative", (13312, 6656)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (12288, 6144)) == (5500, 11000)
    assert irs.arm_input_size("rnative", (16384, 8192), (6144, 12288)) == (6144, 12288)
    assert irs.arm_input_size("rnative", (3328, 1664)) == (2048, 4096)     # the floor
    assert irs.arm_input_size("rnative", (4096, 2048)) == (2048, 4096)
    assert irs.arm_input_size("rnative", (16384, 8192), (4096, 8192)) == (4096, 8192)


def test_r2048_is_the_committed_instrument():
    """r2048 must go through threshold_sweep.PRE itself, not a copy of it."""
    torch = pytest.importorskip("torch")
    from PIL import Image
    import threshold_sweep as ts
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (60, 120, 3), dtype=np.uint8))
    a = irs.arm_tensor(img, "r2048")
    assert tuple(a.shape) == (3, 2048, 4096)
    assert torch.equal(a, ts.PRE(img))


def test_u4096_is_bicubic_of_the_2048_derivative():
    torch = pytest.importorskip("torch")
    from PIL import Image
    from torchvision import transforms
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8))
    got = irs.arm_tensor(img, "u4096")
    mid = transforms.Resize((2048, 4096),
                            interpolation=transforms.InterpolationMode.BILINEAR)(img)
    up = transforms.Resize((4096, 8192),
                           interpolation=transforms.InterpolationMode.BICUBIC)(mid)
    want = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(transforms.ToTensor()(up))
    assert tuple(got.shape) == (3, 4096, 8192)
    assert torch.equal(got, want)


# (b) the sensitivity arm's head loads the released weights strictly -----------------
def test_hm1024_head_loads_512_state_dict_strictly():
    torch = pytest.importorskip("torch")
    from rampnet.model import KeypointModel
    base = KeypointModel(pretrained_backbone=False)
    big = KeypointModel(heatmap_size=(1024, 2048), pretrained_backbone=False)
    big.load_state_dict(base.state_dict(), strict=True)
    x = torch.zeros(1, 3, 64, 128)
    with torch.no_grad():
        assert tuple(big.eval()(x).shape) == (1, 1, 1024, 2048)
        assert tuple(base.eval()(x).shape) == (1, 1, 512, 1024)


def test_split_forward_equals_model_call():
    """The extractor runs head(feature_extractor(x)) so r4096 and r4096_hm1024 can share
    one backbone pass; that must be the same numbers as model(x)."""
    torch = pytest.importorskip("torch")
    from rampnet.model import KeypointModel
    torch.manual_seed(0)
    m = KeypointModel(pretrained_backbone=False).eval()
    x = torch.randn(3, 64, 128)
    with torch.no_grad():
        want = m(x.unsqueeze(0)).squeeze().numpy()
    got = irs._forward(m, x, torch.device("cpu"), False)
    assert np.array_equal(got, want)


# (c) headroom classes from the committed records ----------------------------------
def test_headroom_class_from_records():
    rich = irs.native_sizes_from_records("richmond")
    widths = [w for w, _ in rich.values()]
    cls = [irs.headroom_class(w, 8192) for w in widths]
    assert cls.count("none") == 20            # the 4096-wide built-in null
    assert cls.count("full") == 77 + 14       # 11000 and 12288 wide
    assert cls.count("partial") == 13         # 5760 wide
    ann = irs.native_sizes_from_records("annapolis")
    assert {irs.headroom_class(w, 8192) for w, _ in ann.values()} == {"partial"}
    assert {irs.headroom_class(w, 6144) for w, _ in ann.values()} == {"full"}
    morg = irs.native_sizes_from_records("morgantown")
    assert sum(irs.headroom_class(w, 8192) == "none" for w, _ in morg.values()) == 122


# (e) usage rows --------------------------------------------------------------------
def test_usage_rows_are_free_and_timed():
    rows = irs.usage_rows({"r4096": {"elapsed_s": 100.0, "panos_scored": 50, "fp16": False}},
                          wait={"wait_s": 40.0, "cpu_s": 300.0}, wall_s=141.0,
                          host="makelab2.cs.washington.edu",
                          gpus=["NVIDIA A40"], cities=["annapolis"],
                          started="2026-09-26T00:00:00Z")
    assert len(rows) == 2
    for r in rows:
        assert r["paid"] is False
        assert r["est_cost_usd"] == 0.0
        assert r["hardware"] == {"host": "makelab2.cs.washington.edu", "gpus": ["NVIDIA A40"]}
        assert r["elapsed_s"] > 0
        assert r["run_id"].startswith("input-res-sweep-25:")
    assert rows[0]["s_per_pano"] == 2.0
    assert rows[1]["cpu_prep_s"] == 300.0 and rows[1]["run_wall_s"] == 141.0
    assert len({r["run_id"] for r in rows}) == 2


def test_extract_refuses_unrecorded_spend():
    with pytest.raises(SystemExit):
        irs.main(["extract", "--usage-log", "none", "--cities", "richmond"])


# verdict rule ------------------------------------------------------------------------
def _d(p, r, f):
    return {"precision": {"observed": p[0], "ci_lo": p[1], "ci_hi": p[2]},
            "recall": {"observed": r[0], "ci_lo": r[1], "ci_hi": r[2]},
            "f1": {"observed": f[0], "ci_lo": f[1], "ci_hi": f[2]}}


def test_verdict_rule():
    helps = _d((-0.01, -0.02, 0.0), (0.05, 0.02, 0.08), (0.02, -0.01, 0.05))
    beats_up = _d((0, 0, 0), (0.03, 0.01, 0.05), (0, 0, 0))
    ties_up = _d((0, 0, 0), (0.01, -0.01, 0.03), (0, 0, 0))
    assert irs.verdict(helps, beats_up)[0] == "helps"
    assert irs.verdict(helps, ties_up)[0] == "tolerates"     # dF1 CI covers 0
    f1_up = _d((-0.01, -0.02, 0.0), (0.05, 0.02, 0.08), (0.03, 0.01, 0.05))
    assert irs.verdict(f1_up, beats_up)[0] == "helps"
    assert irs.verdict(f1_up, ties_up)[0] == "gains, object scale only"
    prec_led = _d((0.05, 0.02, 0.08), (0.0, -0.01, 0.01), (0.03, 0.01, 0.05))
    assert irs.verdict(prec_led, beats_up)[0] == "F1 up, not recall-led"
    hurts = _d((-0.1, -0.15, -0.05), (0.0, -0.02, 0.02), (-0.05, -0.08, -0.02))
    assert irs.verdict(hurts, beats_up)[0] == "hurts"
    flat = _d((0.0, -0.01, 0.01), (0.0, -0.01, 0.01), (0.0, -0.01, 0.01))
    assert irs.verdict(flat, None)[0] == "tolerates"


def test_check_compares_peak_sets():
    ref = {"panos": [{"pano": "a", "preds": [[0.1, 0.6, 0.9], [0.5, 0.7, 0.4]]}]}
    same = [{"pano": "a", "preds": [(0.5, 0.7, 0.40000004), (0.1, 0.6, 0.9)]}]
    assert irs.compare_to_op_cache(same, ref)["panos_mismatched"] == 0
    off = [{"pano": "a", "preds": [(0.5, 0.7, 0.41), (0.1, 0.6, 0.9)]}]
    assert irs.compare_to_op_cache(off, ref)["panos_mismatched"] == 1
    fewer = [{"pano": "a", "preds": [(0.1, 0.6, 0.9)]}]
    r = irs.compare_to_op_cache(fewer, ref)
    assert r["panos_mismatched"] == 1 and r["missing"] == 1


def test_border_band_is_what_exclude_border_drops():
    """The pre-#132 op_caches were extracted with skimage's exclude_border=True; that is
    exactly the exclude_border=False peak set minus the min_distance border band."""
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = gaussian_filter(rng.random((512, 1024)), 4)
        h = (h - h.min()) / (h.max() - h.min())
        on = peak_local_max(h, min_distance=10, threshold_abs=0.05, exclude_border=True)
        off = peak_local_max(h, min_distance=10, threshold_abs=0.05, exclude_border=False)
        kept = {(int(r), int(c)) for r, c in off
                if not irs.in_border_band(c / 1024, r / 512)}
        assert kept == {(int(r), int(c)) for r, c in on}


def test_border_peak_is_extra_not_missing():
    ref = {"panos": [{"pano": "a", "preds": [[0.5, 0.7, 0.4]]}]}
    mine = [{"pano": "a", "preds": [(0.5, 0.7, 0.4), (0.001953125, 0.6, 0.3)]}]
    strict = irs.compare_to_op_cache(mine, ref)
    assert strict["extra"] == strict["extra_in_border"] == 1
    assert irs.compare_to_op_cache(mine, ref, drop_border=True)["panos_mismatched"] == 0


# (d) the committed results --------------------------------------------------------------
RESULTS = os.path.join(REPO, "analysis_out", "input_res_sweep_25", "results.json")


def test_r2048_richmond_reproduces_op_cache_modulo_the_seam_strip():
    """r2048 = committed op_cache once the pre-f4c71c8 seam strip is set aside; the three
    seam-strip peaks at or above 0.30 move richmond by +1 tp / +3 fp."""
    from rampnet.detection_eval import radius_sq_for
    from operating_point_curve import _score_at, read_cache
    rsq = radius_sq_for()
    mine, _ = irs.read_arm_cache(irs.cache_path(irs.CACHE_ROOT, "r2048", "richmond"))
    ref, _ = read_cache(os.path.join(REPO, "analysis_out", "op_cache", "richmond.json"))
    inner = [{**p, "preds": [t for t in p["preds"] if not irs.in_seam_strip(t[0])]}
             for p in mine]
    for thr, full, op in ((0.30, (258, 31, 52), (257, 28, 53)),
                          (0.55, (239, 11, 71), (238, 9, 72))):
        a, b, c = (_score_at(mine, thr, rsq), _score_at(inner, thr, rsq),
                   _score_at(ref, thr, rsq))
        assert (a.tp, a.fp, a.fn) == full
        assert (b.tp, b.fp, b.fn) == (c.tp, c.fp, c.fn) == op


def test_report_reproduces_committed_headline():
    """Re-derive annapolis r4096 vs r2048 from the committed caches; pinned to 4 dp."""
    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)
    rep = irs._strip_private(irs.build_report(irs.CACHE_ROOT, arms=("r2048", "r4096"),
                                              cities=("annapolis",)))
    got = rep["per_split"]["annapolis"]["vs_r2048"]["r4096"]["0.30"]
    assert got == committed["per_split"]["annapolis"]["vs_r2048"]["r4096"]["0.30"]
    assert got["recall"]["observed"] == -0.0714
    assert got["recall"]["ci_lo"] == -0.1237 and got["recall"]["ci_hi"] == -0.0196
    assert committed["verdicts"]["annapolis"]["r4096"]["verdict"] == "hurts"


# review fixes (PR #196) ---------------------------------------------------------------
def test_rnative_cap_honours_the_height():
    """A pano (or a cap) that is not 2:1 is bounded on whichever side binds."""
    # square native, default cap: the height binds (8000 > 5500), aspect kept
    assert irs.arm_input_size("rnative", (8000, 8000)) == (5500, 5500)
    # a tall user cap: the 5500 native height exceeds 4000, the width does not bind
    assert irs.arm_input_size("rnative", (11000, 5500), (4000, 12000)) == (4000, 8000)
    # width binds (the only case the committed 2:1 bundles hit)
    assert irs.arm_input_size("rnative", (16384, 8192), (6000, 11000)) == (5500, 11000)
    # inside the cap on both sides: native
    assert irs.arm_input_size("rnative", (8000, 4000), (4000, 12000)) == (4000, 8000)


def test_rnative_sizes_match_committed_caches():
    """The cap_h fix changes no committed size: every bundle is 2:1, so the width branch
    is the one taken, and each rnative cache's recorded input is what the fixed
    arm_input_size returns for that pano's native size and the cache's own cap."""
    n = 0
    for city in irs.SPLITS:
        panos, meta = irs.read_arm_cache(irs.cache_path(irs.CACHE_ROOT, "rnative", city))
        cap = tuple(meta["native_cap"])
        assert cap == irs.DEFAULT_NATIVE_CAP
        for p in panos:
            assert irs.arm_input_size("rnative", tuple(p["native"]), cap) == tuple(p["input"])
            n += 1
    assert n == 1289


def test_rows_for_run_logs_a_failure_before_the_first_pano():
    stats = {a: {"elapsed_s": 0.0, "panos_scored": 0, "fp16": False} for a in ("r4096",)}
    wait = {"wait_s": 3.0, "cpu_s": 7.0}
    kw = dict(host="h", gpus=["NVIDIA A40"], cities=["annapolis", "bend"],
              started="2026-09-26T00:00:00Z")
    rows = irs.rows_for_run(stats, wait, 42.0, attempted=["annapolis"], status="failed", **kw)
    assert len(rows) == 1
    r = rows[0]
    assert r["panos_scored"] == 0 and r["elapsed_s"] == 42.0 and r["status"] == "failed"
    assert r["paid"] is False and r["est_cost_usd"] == 0.0
    assert r["bundle"] == "annapolis"
    # a clean run with nothing to do logs nothing
    assert irs.rows_for_run(stats, wait, 5.0, attempted=[], status="ok", **kw) == []


def test_rows_for_run_names_the_city_it_died_in():
    stats = {"r3072": {"elapsed_s": 10.0, "panos_scored": 126, "fp16": False},
             "rnative": {"elapsed_s": 9.0, "panos_scored": 125, "fp16": False}}
    rows = irs.rows_for_run(stats, {"wait_s": 1.0, "cpu_s": 2.0}, 21.0, host="h", gpus=[],
                            attempted=["annapolis", "bend"], cities=list(irs.SPLITS),
                            started="t", status="failed")
    assert {r["bundle"] for r in rows} == {"annapolis,bend"}
    assert [r["panos_scored"] for r in rows] == [126, 125, 126]


def test_limit_refused_on_the_committed_cache_root(tmp_path):
    with pytest.raises(SystemExit):
        irs.refuse_limit_on_committed_caches(3, irs.CACHE_ROOT)
    irs.refuse_limit_on_committed_caches(3, str(tmp_path))     # scratch root: fine
    irs.refuse_limit_on_committed_caches(0, irs.CACHE_ROOT)    # no limit: fine


def test_verdict_decides_on_unrounded_bounds():
    """richmond r4096: dF1 CI upper bound -4.54e-5 rounds to -0.0. Judged on the rounded
    copy it reads 'tolerates'; on the unrounded _raw copy it is 'hurts'."""
    raw = _d((-0.0866, -0.1409, -0.0403), (0.0129, -0.0300, 0.0577),
             (-0.0362, -0.0763, -4.543e-05))
    rounded = _d((-0.0866, -0.1409, -0.0403), (0.0129, -0.0300, 0.0577),
                 (-0.0362, -0.0763, -0.0))
    assert irs.verdict(rounded, None)[0] == "tolerates"
    assert irs.verdict({**rounded, "_raw": raw}, None)[0] == "hurts"


def test_richmond_r4096_recomputes_as_hurts():
    rep = irs.build_report(irs.CACHE_ROOT, arms=("r2048", "r4096", "u4096"),
                           cities=("richmond",), verdicts_only=True)
    assert rep["verdicts"]["richmond"]["r4096"]["verdict"] == "hurts"
    raw = rep["per_split"]["richmond"]["vs_r2048"]["r4096"]["0.30"]["_raw"]["f1"]["ci_hi"]
    assert -1e-4 < raw < 0
    # the rounded copy is the one that would mislead
    assert rep["per_split"]["richmond"]["vs_r2048"]["r4096"]["0.30"]["f1"]["ci_hi"] == 0.0


def test_verdicts_block_matches_a_fresh_build():
    """Drift guard: every committed verdict, per split and pooled, re-derives from the
    committed caches (the 0.30 contrasts only; about half a minute)."""
    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)["verdicts"]
    rep = irs._strip_private(irs.build_report(
        irs.CACHE_ROOT, arms=("r2048", "r3072", "r4096", "rnative", "u4096"),
        verdicts_only=True))
    assert rep["verdicts"] == committed


def test_paired_subset_recall():
    # 3 panos; GT points: pano0 x2, pano1 x1, pano2 x1; the mask drops pano1's point
    pano_of = np.array([0, 0, 1, 2])
    mask = np.array([True, True, False, True])
    hit_a = np.array([0.9, 0.9, 0.9, 0.9])   # arm finds every point
    hit_b = np.array([0.9, 0.1, 0.1, 0.1])   # control finds only the first
    ones = np.ones((1, 3))
    # masked points: 3; a hits 3, b hits 1 -> +2/3
    assert irs.paired_subset_recall(hit_a, hit_b, pano_of, mask, ones, 0.30)[0] == \
        pytest.approx(2 / 3)
    # a replicate that drops pano0 and doubles pano2: a 2/2, b 0/2 -> +1
    w = np.array([[0.0, 1.0, 2.0]])
    assert irs.paired_subset_recall(hit_a, hit_b, pano_of, mask, w, 0.30)[0] == 1.0
    # a replicate with no masked point left is NaN, not a division error
    w0 = np.array([[0.0, 5.0, 0.0]])
    assert np.isnan(irs.paired_subset_recall(hit_a, hit_b, pano_of, mask, w0, 0.30)[0])


def test_band_of_point():
    import math
    assert irs.band_of_point(0.5) == ("above horizon", "above horizon")
    assert irs.band_of_point(0.3) == ("above horizon", "above horizon")
    # 45 deg below the horizon: range = CAM_H = 2.5 m -> 0-8 m; ray 3.54 m -> ~221 px
    assert irs.band_of_point(0.75) == ("0-8 m", "80 px+")
    # a far point, 30 m out on flat ground: 1.2 m at a ~30.1 m ray, 4096 px per 2 pi
    # -> ~26 px
    y = 0.5 + math.atan(irs.CAM_H / 30.0) / math.pi
    assert irs.band_of_point(y) == ("25-40 m", "20-32 px")


def test_non_control_arms_resize_bilinear(monkeypatch):
    """Every arm but u4096's second step resizes BILINEAR, the interpolation PRE uses."""
    torch = pytest.importorskip("torch")
    from PIL import Image
    from torchvision import transforms
    monkeypatch.setitem(irs.ARMS, "tiny", {"size": (32, 64), "via": None,
                                          "heatmap": irs.BASE_HEATMAP, "min_distance": 10})
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def via(mode):
        return norm(transforms.ToTensor()(transforms.Resize((32, 64), interpolation=mode)(img)))
    got = irs.arm_tensor(img, "tiny")
    assert torch.equal(got, via(transforms.InterpolationMode.BILINEAR))
    assert not torch.equal(got, via(transforms.InterpolationMode.BICUBIC))
    for arm in ("r3072", "r4096", "rnative", "r4096_hm1024"):
        assert [m for _, m in irs.resize_steps(arm, (11000, 5500))] == ["bilinear"]


def test_seam_criterion_rejects_a_pole_peak():
    """An extra peak in the top rows is in the border band but not the seam strip, so
    check's criterion (c) -- every extra peak in the seam strip -- does not excuse it."""
    ref = {"panos": [{"pano": "a", "preds": [[0.5, 0.7, 0.4]]}]}
    pole = [{"pano": "a", "preds": [(0.5, 0.7, 0.4), (0.5, 0.005859375, 0.3)]}]
    r = irs.compare_to_op_cache(pole, ref)
    assert r["extra"] == r["extra_in_border"] == 1 and r["extra_at_seam"] == 0
    corner = [{"pano": "a", "preds": [(0.5, 0.7, 0.4), (0.0, 0.005859375, 0.3)]}]
    r = irs.compare_to_op_cache(corner, ref)
    assert r["extra"] == r["extra_in_border"] == r["extra_at_seam"] == 1


def test_committed_instrument_check_extras_are_all_at_the_seam():
    with open(os.path.join(REPO, "analysis_out", "input_res_sweep_25",
                           "instrument_check.json"), encoding="utf-8") as f:
        chk = json.load(f)
    assert chk["pass"] is True
    checked = [c for c in chk["cities"] if c["checked"]]
    assert len(checked) == 10
    assert sum(c["strict"]["extra"] for c in checked) == 173
    for c in checked:
        assert c["strict"]["extra"] == c["strict"]["extra_at_seam"]
        assert c["strict"]["missing"] == 0


def test_sha256sums_cover_and_match_the_committed_outputs():
    """Content hashes (CLAUDE.md): results.json/.md, instrument_check.json and all 66
    caches are listed and match byte for byte."""
    listed = irs.read_sums()
    assert set(irs.HASHED) <= set(listed)
    assert sum(k.startswith("cache/") for k in listed) == 66
    assert irs.verify_sums() == []


def test_sha256sums_catch_a_changed_byte(tmp_path):
    (tmp_path / "results.json").write_bytes(b"{}\n")
    sums = tmp_path / "SHA256SUMS"
    assert irs.main(["sums", "--root", str(tmp_path), "--sums", str(sums), "--write"]) == 0
    assert irs.verify_sums(str(tmp_path), str(sums)) == []
    (tmp_path / "results.json").write_bytes(b"{} \n")
    assert irs.verify_sums(str(tmp_path), str(sums)) != []
    (tmp_path / "results.json").unlink()
    assert irs.verify_sums(str(tmp_path), str(sums), require_all=False) == []
    assert irs.verify_sums(str(tmp_path), str(sums)) != []
