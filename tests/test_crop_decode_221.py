"""The crop-model decode measurement (#221): CPU only, no checkpoint, no network.

- (a) the script's own combine / match / residual code places a synthetic 32x11 bump on
  the 8-px grid under argmax and at its true position under the gaussian decode, in
  both the single-pass and the flip-TTA arm;
- (b) its filename GT parser agrees with the committed crop evaluator's inline parser;
- (c) ``--check`` re-derives a committed results file byte for byte from its extract;
- (d) the crop evaluator's ``PEAK_DECODE`` opt-in leaves the argmax extractor exactly
  as it was, and the gaussian path finds the same peaks with the same scores.
"""
import importlib.util
import os

import numpy as np
import pytest

from rampnet import subcell as sc

skimage_feature = pytest.importorskip("skimage.feature")
peak_local_max = skimage_feature.peak_local_max

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP = (256, 88)
COARSE = (32, 11)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cd():
    return _load("crop_decode_221_mod", os.path.join("scripts", "analysis", "crop_decode_221.py"))


def gaussian_coarse(cy, cx, sigma=1.5, amp=0.9):
    """The crop training target's shape (sigma 12 heatmap px = 1.5 cells), coarse grid."""
    yy, xx = np.mgrid[0:COARSE[0], 0:COARSE[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def hires(c):
    return sc.FACTOR * c + (sc.FACTOR - 1) / 2


# --- (a) synthetic bump through the script's pipeline ----------------------------------

@pytest.mark.parametrize("arm", ["single", "tta"])
@pytest.mark.parametrize("cy,cx", [(14.3, 5.2), (9.75, 4.6), (20.1, 6.45)])
def test_argmax_on_grid_gaussian_exact(cd, arm, cy, cx):
    c = gaussian_coarse(cy, cx)
    # A flip-equivariant model's mirrored branch, in the mirrored orientation.
    crop = {"coarse": [c.tolist(), np.fliplr(c).tolist()], "width": 682, "height": 2048,
            "gt": [[hires(cx) / CROP[1], hires(cy) / CROP[0]]]}
    pairs, cache = cd.build_pairs([crop], arm, 0.30)
    assert len(pairs) == 1
    _, k, _ = pairs[0]
    r, col, _ = cache[0]["argmax"][k]
    assert int(col) % 8 in (3, 4) and int(r) % 8 in (3, 4)
    dx, dy = cd.residuals([crop], pairs, cache, "gaussian")
    assert abs(dx[0]) < 1e-6 and abs(dy[0]) < 1e-6
    adx, ady = cd.residuals([crop], pairs, cache, "argmax")
    assert max(abs(adx[0]), abs(ady[0])) > 0.05       # the case the decode exists for


def test_tta_combine_is_evaluate_py(cd):
    """clip each branch, flip the mirrored one back, elementwise max."""
    rng = np.random.default_rng(3)
    c0, c1 = rng.uniform(-0.3, 1.3, COARSE), rng.uniform(-0.3, 1.3, COARSE)
    crop = {"coarse": [c0.tolist(), c1.tolist()]}
    combined, stack, _ = cd.arm_maps(crop, "tta")
    want = np.maximum(np.clip(sc.upsample(c0), 0, 1), np.clip(np.fliplr(sc.upsample(c1)), 0, 1))
    assert np.allclose(combined, want, atol=1e-12)
    assert sc.coarse_mismatch(combined, stack, clip=True) < 1e-12
    assert cd.flip_commutes(crop) < 1e-12


def test_up_matches_upsample_and_is_flip_exact(cd):
    """The report's upsample equals sc.upsample to rounding and is flip-exact by
    construction (symmetric taps, commutative add). Whether it is the same on every CPU
    cannot be shown in-process; the byte-for-byte --check test (test_check_val_extract)
    is the guard for that."""
    c = np.random.default_rng(4).uniform(-0.5, 1.5, COARSE)
    assert np.abs(cd.up(c) - sc.upsample(c)).max() < 1e-14
    assert np.array_equal(cd.up(np.fliplr(c)), np.fliplr(cd.up(c)))


def test_x_fixes(cd):
    """At width 704 the training targets have no x mismatch, so both fixes are identity;
    at 682 the flip-averaged fix has its fixed point near column 44."""
    assert cd.fix_x(30.0, 704, "label_scale") == pytest.approx(30.0)
    assert cd.fix_x(30.0, 704, "flip_average") == pytest.approx(30.0)
    k = 682 / 704
    t0 = 87.75 * (1 - k) / 2 / (1 - k)
    assert cd.fix_x(t0, 682, "flip_average") == pytest.approx(t0)
    assert 43 < t0 < 45


def test_duplicate_images_are_one_cluster(cd):
    """The round-2 splits hold one file per point of a multi-point crop (same bytes,
    points permuted in the name). Dedup keeps the first by name; the bootstrap counts
    the image once."""
    crops = [{"crop": "b_-_1_2.jpg", "sha256": "S"}, {"crop": "a_-_3_4_-_1_2.jpg", "sha256": "S"},
             {"crop": "c_-_5_6.jpg", "sha256": "T"}]
    assert cd.unique_image_indices(crops) == [1, 2]
    boot = cd.Boot(np.array(["S", "S", "T"]), np.random.default_rng(0), 50)
    assert boot.C.shape == (50, 2)
    assert np.all(boot.C.sum(axis=1) == 2)


def test_x_bias_on_gt_x_recovers_a_planted_slope(cd):
    """dx = 0.03 * x_gt - 1.4 with a detection error that does not depend on x: the GT-x
    regression recovers it; the detection-x regression is diluted toward zero."""
    rng = np.random.default_rng(1)
    n = 400
    xg = rng.uniform(5, 83, n)
    e = rng.normal(0, 2, n)                          # detection error, independent of x
    xd = xg - (0.03 * xg - 1.4) + e                  # so dx = GT - det = 0.03 xg - 1.4 - e
    crops = [{"gt": [[xg[i] / CROP[1], 0.5]], "sha256": str(i)} for i in range(n)]
    cache = [{"gaussian": np.array([[128.0, xd[i], 1.0]])} for i in range(n)]
    pairs = [(i, 0, 0) for i in range(n)]
    boot = cd.Boot(np.array([str(i) for i in range(n)]), np.random.default_rng(2), 200)
    xb = cd.x_bias(crops, pairs, cache, "gaussian", boot)
    assert xb["on_gt_x"]["slope"] == pytest.approx(0.03, abs=0.01)
    # planted line 0.03 x - 1.4 vs predicted 0.03125 x - 1.371: the offset is near zero
    assert abs(xb["offset_vs_predicted"]["obs"]) < 0.3
    assert xb["on_det_x"]["slope"] < xb["on_gt_x"]["slope"]
    assert 0 < xb["dilution"] < 1


# --- (b) GT parser ---------------------------------------------------------------------

def _evaluate_py_parse(name, w, h):
    """The inline parser in stage_one/crop_model/ps_and_manual_model/evaluate.py, verbatim
    (minus its warning prints)."""
    base_name_no_ext = os.path.splitext(os.path.basename(name))[0]
    gt_points_normalized = []
    filename_parts = base_name_no_ext.split('_-_')
    if len(filename_parts) > 1:
        for point_str in filename_parts[1:]:
            try:
                x_pixel_str, y_pixel_str = point_str.split('_')
                gt_points_normalized.append((int(x_pixel_str) / w, int(y_pixel_str) / h))
            except ValueError:
                pass
    return gt_points_normalized


@pytest.mark.parametrize("name,w,h", [
    ("ab12cd34_-_341_1500.jpg", 682, 2048),
    ("ab12cd34_-_10_1900_-_600_1210.jpg", 682, 2048),
    ("zz_-_681_0.jpg", 682, 2048),
    ("q9_-_700_1024.jpg", 704, 2048),
    ("nopoints.jpg", 682, 2048),
])
def test_gt_parser_matches_evaluate_py(cd, name, w, h):
    assert cd.parse_gt(name, w, h) == _evaluate_py_parse(name, w, h)


def test_gt_parser_two_points(cd):
    assert cd.parse_gt("u_-_341_1024_-_0_2048.jpg", 682, 2048) == [(0.5, 0.5), (0.0, 1.0)]


# --- (c) committed results re-derive ---------------------------------------------------

def test_check_val_extract(cd, capsys):
    """One of the three committed extracts (val, the smallest); the CLI checks all."""
    assert cd.cmd_check(only={"extract_val.json"}) == 0


def test_committed_extracts_listed(cd):
    for ex_name, stem in cd.COMMITTED:
        for f in (ex_name, stem + ".json", stem + ".md"):
            assert os.path.exists(os.path.join(cd.OUT_DIR, f)), f


# --- (d) evaluate.py's PEAK_DECODE opt-in ---------------------------------------------

def _load_crop_evaluate():
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    pytest.importorskip("tqdm")
    return _load("stage_one_crop_evaluate_221",
                 os.path.join("stage_one", "crop_model", "ps_and_manual_model", "evaluate.py"))


def _legacy_extract(heatmap_np, min_distance, threshold_abs, heatmap_shape):
    """The crop evaluate.py's extract_peaks_from_heatmap before #221, verbatim."""
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


def test_default_decode_is_argmax():
    ev = _load_crop_evaluate()
    assert ev.PEAK_DECODE == "argmax"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.55])
def test_argmax_extractor_unchanged_and_gaussian_same_peaks(seed, threshold):
    ev = _load_crop_evaluate()
    rng = np.random.default_rng(seed)
    c0, c1 = rng.uniform(-0.2, 1.1, COARSE), rng.uniform(-0.2, 1.1, COARSE)
    h0, h1 = sc.upsample(c0), np.fliplr(sc.upsample(c1))
    h = np.maximum(np.clip(h0, 0, 1), np.clip(h1, 0, 1)).astype(np.float32)
    old = _legacy_extract(h, 10, threshold, CROP)
    new = ev.extract_peaks_from_heatmap(h, 10, threshold, CROP)
    assert len(old) == len(new) > 0
    for o, n in zip(old, new):
        assert o == n and [type(v) for v in o] == [type(v) for v in n]
    stack = np.stack([c0, np.fliplr(c1)]).astype(np.float32)
    g = ev.extract_peaks_decoded(h, stack, 10, threshold, CROP, "gaussian")
    assert [p[2] for p in g] == pytest.approx([float(p[2]) for p in old], abs=0)
    # Wiring: the shipped entry point with the crop's settings (no wrap, no border drop).
    want = sc.detect_peaks(h, threshold, min_distance=10, decode="gaussian",
                           exclude_border=False, clip=True, coarse=stack, wrap_x=False)
    assert np.allclose([(p[0], p[1]) for p in g],
                       np.column_stack([want[:, 1] / CROP[1], want[:, 0] / CROP[0]]))


def test_decoded_extractor_refuses_stale_coarse():
    ev = _load_crop_evaluate()
    rng = np.random.default_rng(5)
    c = rng.uniform(0, 1, COARSE)
    h = np.clip(sc.upsample(c), 0, 1).astype(np.float32)
    with pytest.raises(ValueError, match="disagree"):
        ev.extract_peaks_decoded(h, (c + 0.1)[None], 10, 0.3, CROP, "gaussian")


def test_predict_heatmap_and_coarse_matches_inline_tta():
    """The helper the gaussian path uses produces the inline argmax path's combined map
    and a coarse stack that re-builds it."""
    torch = pytest.importorskip("torch")
    from PIL import Image
    ev = _load_crop_evaluate()
    c = np.random.default_rng(9).uniform(-0.2, 1.2, COARSE)
    fixed = torch.tensor(sc.upsample(c), dtype=torch.float32)[None, None]

    def model(_x):                       # input-independent stand-in for the network
        return fixed

    img = Image.new("RGB", (68, 200), (120, 90, 60))
    combined, stack = ev.predict_heatmap_and_coarse(model, img)
    raw = fixed.squeeze().numpy()
    inline = np.maximum(np.clip(raw, 0, 1), np.fliplr(np.clip(raw, 0, 1)))
    assert np.array_equal(combined, inline)
    assert stack.shape == (2,) + COARSE and stack.dtype == np.float32
    assert sc.coarse_mismatch(combined, stack, clip=True) < 1e-5
