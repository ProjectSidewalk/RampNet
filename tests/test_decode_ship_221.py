"""Shipping the sub-cell decode (#221 items 1-2): CPU only, no checkpoint, no network.

Pins the contract of ``rampnet.subcell.detect_peaks`` -- the one entry point
``stage_two/evaluate.py``, ``stage_two/demo.py`` and the Hugging Face package's
``RampNetModel.detect`` share -- and that the defaults leave every published number
reproducible:

- ``decode="argmax"`` is bit-identical to the bare ``peak_local_max`` call evaluate.py
  always made, so the published evaluation is unchanged by default;
- ``decode="gaussian"`` recovers a known sub-cell position the argmax cannot;
- the exporter ships ``rampnet/subcell.py`` verbatim and the remote-code helper runs;
- evaluate.py's ``--decode`` keeps the argmax cache and result names exactly as they
  were, and gives a refining decode its own cache and result names.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

from rampnet import subcell as sc

skimage_feature = pytest.importorskip("skimage.feature")
peak_local_max = skimage_feature.peak_local_max

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANO = (512, 1024)


def gaussian_coarse(cy, cx, shape=(64, 128), sigma=1.25, amp=0.9):
    """The training target's shape (sigma 10 hi-res px = 1.25 cells), on the coarse grid."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def hires(c):
    """Coarse cell coordinate -> hi-res pixel coordinate (centre of cell i is 8i + 3.5)."""
    return sc.FACTOR * c + (sc.FACTOR - 1) / 2


# --- (a) argmax is on the grid, gaussian recovers the true position ---------------------

#: Tolerance for the gaussian decode on a noiseless sampled Gaussian, in hi-res px. The
#: log-parabola is exact for a sampled Gaussian, so this only absorbs float error in the
#: least-squares recovery of the coarse map.
EXACT_TOL_PX = 1e-6


@pytest.mark.parametrize("oy,ox", [(0.3, -0.2), (-0.45, 0.41), (0.12, 0.37), (-0.25, -0.33)])
def test_argmax_on_grid_and_gaussian_recovers_truth(oy, ox):
    cy, cx = 30 + oy, 70 + ox
    h = sc.upsample(gaussian_coarse(cy, cx))
    assert h.shape == PANO

    am = sc.detect_peaks(h, 0.3, decode="argmax")
    assert len(am) == 1
    r, c, _ = am[0]
    assert int(c) % 8 in (3, 4) and int(r) % 8 in (3, 4)
    # ...so argmax is off by the sub-cell offset, up to half a cell plus half a pixel.
    assert abs(c - hires(cx)) > 0.5

    g = sc.detect_peaks(h, 0.3, decode="gaussian")
    assert len(g) == 1
    assert abs(g[0, 0] - hires(cy)) < EXACT_TOL_PX
    assert abs(g[0, 1] - hires(cx)) < EXACT_TOL_PX
    assert g[0, 2] == am[0, 2]          # the decode never changes the score


def test_gaussian_close_under_noise():
    """With mild noise on the coarse map the decode still lands within 1 hi-res px,
    where argmax is on average ~2 px off on this grid."""
    rng = np.random.default_rng(221)
    errs_g, errs_a = [], []
    for _ in range(40):
        cy, cx = 20 + rng.uniform(-0.5, 0.5), 60 + rng.uniform(-0.5, 0.5)
        c = gaussian_coarse(cy, cx) + rng.normal(0, 0.003, (64, 128))
        h = sc.upsample(c)
        a = sc.detect_peaks(h, 0.5, decode="argmax")
        g = sc.detect_peaks(h, 0.5, decode="gaussian")
        assert len(a) == len(g) == 1
        errs_a.append(np.hypot(a[0, 0] - hires(cy), a[0, 1] - hires(cx)))
        errs_g.append(np.hypot(g[0, 0] - hires(cy), g[0, 1] - hires(cx)))
    assert max(errs_g) < 1.0
    assert np.mean(errs_g) < 0.5 * np.mean(errs_a)


def test_clip_finds_peaks_on_clipped_map_but_decodes_raw():
    """A peak above 1 is a flat plateau after clipping. clip=True must still decode it
    exactly from the raw map, and report the clipped score like evaluate.py does.

    A plateau this wide (~20 px) makes ``peak_local_max`` return several of its pixels.
    That is the extractor's existing behaviour, unchanged by the decode (argmax returns
    the same rows); every one of them decodes to the same, true, position.
    """
    cy, cx = 33.2, 90.4
    h = sc.upsample(gaussian_coarse(cy, cx, amp=1.6))
    g = sc.detect_peaks(h, 0.3, decode="gaussian", clip=True)
    a = sc.detect_peaks(h, 0.3, decode="argmax", clip=True)
    assert len(g) == len(a) >= 1 and np.all(g[:, 2] == 1.0)
    assert np.all(np.abs(g[:, 0] - hires(cy)) < EXACT_TOL_PX)
    assert np.all(np.abs(g[:, 1] - hires(cx)) < EXACT_TOL_PX)


def test_crop_heatmap_shape_is_supported():
    """The crop model's 256x88 heatmap (factor 8, 32x11 coarse) goes through unchanged."""
    cy, cx = 15.3, 5.2
    h = sc.upsample(gaussian_coarse(cy, cx, shape=(32, 11)))
    assert h.shape == (256, 88)
    g = sc.detect_peaks(h, 0.3, decode="gaussian")
    assert len(g) == 1
    assert abs(g[0, 0] - hires(cy)) < EXACT_TOL_PX
    assert abs(g[0, 1] - hires(cx)) < EXACT_TOL_PX


def test_tta_stack_decodes_each_peak_from_its_winning_branch():
    """Flip-TTA max-combines two surfaces. With a coarse stack, each peak is decoded from
    the branch the max took it from: here two ramps, one per branch."""
    a = gaussian_coarse(20.3, 30.2, amp=0.9)
    b = gaussian_coarse(40.1, 100.4, amp=0.8)
    ca, cb = a + 0.1 * gaussian_coarse(40.1, 100.4), b + 0.1 * gaussian_coarse(20.3, 30.2)
    combined = np.maximum(sc.upsample(ca), sc.upsample(cb))
    g = sc.detect_peaks(combined, 0.3, decode="gaussian", coarse=np.stack([ca, cb]))
    g = g[np.argsort(g[:, 1])]
    assert len(g) == 2
    # Each ramp's own branch is a Gaussian plus a far-away bump: exact to well under 0.01 px.
    assert np.allclose(g[0, :2], [hires(20.3), hires(30.2)], atol=1e-2)
    assert np.allclose(g[1, :2], [hires(40.1), hires(100.4)], atol=1e-2)


def test_rejects_unknown_decode_and_mismatched_coarse():
    h = sc.upsample(gaussian_coarse(30, 70))
    with pytest.raises(ValueError):
        sc.detect_peaks(h, 0.3, decode="nope")
    with pytest.raises(ValueError):
        sc.detect_peaks(h, 0.3, decode="gaussian", coarse=np.zeros((32, 64)))


def test_wrap_x_is_passed_through():
    """A peak in coarse column 0 is not refined in x by default (the measured protocol)
    and is refined across the seam with wrap_x=True. (``peak_local_max`` does not wrap,
    so the bump's tail at the right edge is a second peak; with wrap_x it climbs across
    the seam to the same ramp. Only the strongest peak is checked here.)"""
    cx = 0.3
    c = gaussian_coarse(30.2, cx) + gaussian_coarse(30.2, 128 + cx)   # wraps
    h = sc.upsample(c)
    off = sc.detect_peaks(h, 0.3, decode="gaussian")
    on = sc.detect_peaks(h, 0.3, decode="gaussian", wrap_x=True)
    off, on = off[np.argmax(off[:, 2])], on[np.argmax(on[:, 2])]
    assert off[1] == hires(0)
    assert abs(on[1] - hires(cx)) < 1e-3


# --- (b) argmax is bit-identical to the bare peak_local_max call ------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.3, 0.55, 0.9])
def test_argmax_equals_raw_peak_local_max(seed, threshold):
    rng = np.random.default_rng(seed)
    # Smooth-ish random field, float32 like the cached heatmaps, clipped like evaluate.py.
    coarse = rng.uniform(-0.2, 1.1, (64, 128))
    h = np.clip(sc.upsample(coarse) + rng.normal(0, 0.01, PANO), 0, 1).astype(np.float32)
    ref = peak_local_max(h, min_distance=10, threshold_abs=threshold, exclude_border=False)
    got, pix = sc.detect_peaks(h, threshold, decode="argmax", return_pixels=True)
    assert np.array_equal(pix, ref)
    assert np.array_equal(got[:, :2], ref.astype(float))
    assert np.array_equal(got[:, 2], h[ref[:, 0], ref[:, 1]].astype(np.float64))


def _load_evaluate():
    """stage_two/evaluate.py under a unique module name (bare ``evaluate`` collides with
    stage_one's evaluator; see tests/test_stage1_eval.py)."""
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    pytest.importorskip("tqdm")
    spec = importlib.util.spec_from_file_location(
        "stage_two_evaluate_221", os.path.join(REPO_ROOT, "stage_two", "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy_extract(heatmap_np, min_distance, threshold_abs, heatmap_shape):
    """evaluate.extract_peaks_from_heatmap as it was before #221 shipped, verbatim."""
    heatmap_h, heatmap_w = heatmap_shape
    if heatmap_np.ndim > 2:
        heatmap_np = heatmap_np.squeeze()
    heatmap_np_contiguous = np.ascontiguousarray(heatmap_np)
    coordinates = peak_local_max(heatmap_np_contiguous, min_distance=min_distance,
                                 threshold_abs=threshold_abs, exclude_border=False)
    peaks_normalized = []
    for r, c in coordinates:
        confidence = heatmap_np[r, c]
        peaks_normalized.append((c / heatmap_w, r / heatmap_h, confidence))
    return peaks_normalized


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.55])
def test_evaluate_extractor_argmax_is_the_legacy_extractor(threshold):
    """Values *and* types (np.float32 confidences print differently from float64 in the
    committed CSVs), so an argmax run regenerates the committed results byte for byte."""
    ev = _load_evaluate()
    rng = np.random.default_rng(7)
    h = np.clip(sc.upsample(rng.uniform(-0.2, 1.1, (64, 128))), 0, 1).astype(np.float32)
    old = _legacy_extract(h, 10, threshold, PANO)
    new = ev.extract_peaks_from_heatmap(h, 10, threshold, PANO)
    assert len(old) == len(new) > 0
    for o, n in zip(old, new):
        assert o == n
        assert [type(v) for v in o] == [type(v) for v in n]
        assert repr(o[2]) == repr(n[2])


def test_evaluate_extractor_still_reports_exclude_border_false():
    """scripts/analysis/dump_peaks_from_cache.py stamps the extractor's exclude_border by
    reading the literal out of extract_peaks_from_heatmap's source (#132)."""
    import inspect
    ev = _load_evaluate()
    assert "exclude_border=False" in inspect.getsource(ev.extract_peaks_from_heatmap)


# --- (c) the exporter ships subcell.py verbatim ----------------------------------------

def _exporter():
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    pytest.importorskip("transformers")
    import export_hf_model
    return export_hf_model


def test_exporter_ships_subcell_verbatim(tmp_path):
    ex = _exporter()
    assert ex.VERBATIM_COPIES["rampnet_subcell.py"] == os.path.join("rampnet", "subcell.py")
    shipped = ex.copy_code_files(str(tmp_path))
    assert "rampnet_subcell.py" in shipped and "rampnet_model.py" in shipped
    for dst, src in ex.VERBATIM_COPIES.items():
        with open(os.path.join(REPO_ROOT, src), "rb") as f:
            want = f.read()
        with open(tmp_path / dst, "rb") as f:
            assert f.read() == want, f"{dst} is not a verbatim copy of {src}"


def test_shipped_subcell_imports_only_numpy_at_load():
    """The remote-code loader requires every import transformers' get_imports finds to be
    installed (it skips try blocks). subcell.py must not add a load-time dependency."""
    dmu = pytest.importorskip("transformers.dynamic_module_utils")
    found = dmu.get_imports(os.path.join(REPO_ROOT, "rampnet", "subcell.py"))
    assert set(found) - set(sys.stdlib_module_names) == {"numpy"}


# --- (d) the remote-code helper runs on a synthetic heatmap ----------------------------

def test_hf_package_detect_runs_without_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from transformers import AutoModel
    from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE
    ex = _exporter()

    reference = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False).eval()
    pkg = str(tmp_path / "pkg")
    ex.assemble_package(pkg, reference)
    model = AutoModel.from_pretrained(pkg, trust_remote_code=True).eval()

    cy, cx = 30.3, 70.41
    h = sc.upsample(gaussian_coarse(cy, cx)).astype(np.float32)
    want_x, want_y = hires(cx) / PANO[1], hires(cy) / PANO[0]

    (g,) = model.detect(h, threshold=0.3)                       # default decode: gaussian
    assert g.shape == (1, 3)
    assert abs(g[0, 0] - want_x) < 1e-6 and abs(g[0, 1] - want_y) < 1e-6
    (a,) = model.detect(torch.from_numpy(h)[None, None], threshold=0.3, decode="argmax")
    ref = peak_local_max(np.clip(h, 0, 1), min_distance=10, threshold_abs=0.3,
                         exclude_border=False)
    assert np.allclose(a[:, :2], np.column_stack([ref[:, 1] / PANO[1], ref[:, 0] / PANO[0]]))
    assert g[0, 2] == a[0, 2]

    # pixel_values path: runs the model (random weights), returns one array per image.
    out = model.detect(torch.zeros(2, 3, 64, 128), threshold=0.0)
    assert len(out) == 2 and all(o.ndim == 2 and o.shape[1] == 3 for o in out)


# --- (e) evaluate.py's --decode and its cache / result names ---------------------------

def test_evaluate_cli_decode_flag_and_cache_layout():
    ev = _load_evaluate()
    assert ev.parse_args([]).decode == "argmax"
    assert ev.parse_args(["--decode", "gaussian"]).decode == "gaussian"
    with pytest.raises(SystemExit):
        ev.parse_args(["--decode", "dark"])          # CLI exposes only DECODES

    a = ev.cache_dirs("evaluate_cache", "abc123", "manual", True, "argmax")
    g = ev.cache_dirs("evaluate_cache", "abc123", "manual", True, "gaussian")
    # argmax reads exactly the heatmap cache it always has, and nothing else.
    assert a["heatmaps"] == os.path.join("evaluate_cache", "heatmaps", "abc123_manual_tta")
    assert a["coarse"] is None
    # gaussian shares those (decode-independent) heatmaps, and adds a coarse cache in a
    # different location, keyed the same way, that an argmax run never reads.
    assert g["heatmaps"] == a["heatmaps"]
    assert g["coarse"] == os.path.join("evaluate_cache", "coarse", "abc123_manual_tta")
    assert g["coarse"] != a["heatmaps"]
    # ...and that key still separates TTA from single-pass coarse maps.
    assert (ev.cache_dirs("evaluate_cache", "abc123", "manual", False, "gaussian")["coarse"]
            == os.path.join("evaluate_cache", "coarse", "abc123_manual_notta"))

    # Result filenames: argmax keeps the committed name; gaussian cannot overwrite it.
    assert ev.results_params_str(0.0) == "r0.022_pt0.0"
    assert ev.results_params_str(0.0, "gaussian") == "r0.022_pt0.0_dgaussian"


def test_evaluate_refining_decode_requires_coarse_cache():
    ev = _load_evaluate()
    with pytest.raises(ValueError):
        ev.evaluate(None, [], [], True, "unused", decode="gaussian")


def test_evaluate_gaussian_reads_coarse_cache(tmp_path):
    """End to end through evaluate() with a stub model that must never be called: both
    caches pre-filled, so the gaussian decode reads the cached coarse map, and the
    detection lands where the coarse map says (not on the 8-px grid)."""
    ev = _load_evaluate()
    pytest.importorskip("PIL")
    hm_dir, co_dir = tmp_path / "heatmaps", tmp_path / "coarse"
    hm_dir.mkdir()
    co_dir.mkdir()
    cy, cx = 30.3, 70.41
    c = gaussian_coarse(cy, cx)
    np.save(hm_dir / "p1_heatmap.npy", np.clip(sc.upsample(c), 0, 1).astype(np.float32))
    np.save(co_dir / "p1_coarse.npy", c[None].astype(np.float32))
    label = tmp_path / "p1.txt"
    gx, gy = hires(cx) / PANO[1], hires(cy) / PANO[0]
    label.write_text(f"0 {gx} {gy} 0.01 0.01\n")

    captured = {}
    real = ev.match_predictions

    def spy(preds, *args, **kw):
        captured["preds"] = preds
        return real(preds, *args, **kw)

    ev.match_predictions = spy
    try:
        m = ev.evaluate(None, [str(tmp_path / "p1.jpg")], [str(label)], True, str(hm_dir),
                        peak_threshold_abs=0.3, decode="gaussian", coarse_cache_dir=str(co_dir))
    finally:
        ev.match_predictions = real
    assert m["decode"] == "gaussian" and m["total_predictions"] == 1
    (x, y, _), = captured["preds"]
    assert abs(x - gx) < 1e-5 and abs(y - gy) < 1e-5
