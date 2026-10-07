"""Opt-in seam-wrapped NMS in ``rampnet.subcell.detect_peaks`` (#248): CPU only, synthetic
maps, no checkpoint, no network.

A port of sidewalk-auto-labeler PR #138's ``--border wrap`` peak finder, without its 50-peak
cap. Pins that

- the default (``wrap_nms=False``) is still bit-identical to the bare ``peak_local_max``
  call, so every published number is unchanged;
- ``wrap_nms=True`` equals an independent brute-force cylinder exactly, ties included;
- a ramp straddling the seam collapses to its stronger half, and nothing off the seam moves;
- ``stage_two/evaluate.py`` and the Hugging Face package expose the flag, default off.
"""
import importlib.util
import inspect
import os
import sys

import numpy as np
import pytest

from rampnet import subcell as sc

skimage_feature = pytest.importorskip("skimage.feature")
pytest.importorskip("scipy.ndimage")
peak_local_max = skimage_feature.peak_local_max

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANO = (512, 1024)
W = PANO[1]
D = 10                       # PEAK_MIN_DISTANCE
INTERIOR = (30, 60, 0.9)


# --- helpers (copied, not imported across test files) ----------------------------------

def coarse_map(peaks, shape=(64, 128), sigma=1.25):
    """A 64x128 coarse map holding Gaussians at coarse-cell centres (cy, cx, amp)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.zeros(shape)
    for cy, cx, amp in peaks:
        c += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return c


def hm(peaks):
    return sc.upsample(coarse_map(peaks)).astype(np.float32)


def straddle_heatmap(a=0.8, b=0.75):
    """One ramp across the seam (coarse columns 0 and 127, same row) plus an interior peak."""
    return hm([(25, 0, a), (25, 127, b), INTERIOR])


def cylinder_reference(h, d=D, threshold=0.0):
    """An independent brute-force cylinder (no scipy, no skimage): the (2d+1)-square window
    maximum by np.roll in x (cyclic) and edge padding in y, candidates above the threshold
    (none on a constant map), then a greedy pass over (-value, row, col) rejecting wrapped
    Chebyshev distance < d. Ported from the labeler's tests/test_border_wrap.py, minus the
    cap."""
    rows = np.max([np.roll(h, s, axis=1) for s in range(-d, d + 1)], axis=0)
    padded = np.pad(rows, ((d, d), (0, 0)), mode="edge")
    m = np.max([padded[d + s:d + s + h.shape[0]] for s in range(-d, d + 1)], axis=0)
    cand = h == m
    if cand.all():
        return []
    cand &= h > threshold
    kept = []
    for _, r, c in sorted((-float(h[r, c]), int(r), int(c)) for r, c in zip(*np.nonzero(cand))):
        if all(max(abs(r - kr), min(abs(c - kc), h.shape[1] - abs(c - kc))) >= d
               for kr, kc in kept):
            kept.append((r, c))
    return kept


def random_map(rng, clip_ok=True):
    """Coarse Gaussians biased toward the seam (some wrapped across it), plus noise; with
    clip_ok, amplitudes up to 1.5, so plateaus clip at 1.0 and make multi-way exact ties."""
    yy, xx = np.mgrid[0:64, 0:128]
    c = rng.normal(0, 0.02, (64, 128))
    for _ in range(rng.integers(2, 12)):
        cy = rng.uniform(-1, 64)
        cx = [rng.uniform(-1.5, 2.5), rng.uniform(125, 128.5), rng.uniform(0, 128)][
            rng.integers(3)]
        dxx = np.abs(xx - cx)
        dxx = np.minimum(dxx, 128 - dxx)
        s = rng.uniform(0.8, 2.0)
        c += rng.uniform(0.2, 1.5) * np.exp(-((yy - cy) ** 2 + dxx ** 2) / (2 * s * s))
    h = sc.upsample(c)
    if not clip_ok:
        h = h / max(1.0, float(h.max()) / 0.99)
    return h.astype(np.float32)


def pixels(rcs):
    return [(int(r), int(c)) for r, c, _ in rcs]


def wrapped_cheb(a, b, width=W):
    dx = abs(a[1] - b[1])
    return max(abs(a[0] - b[0]), min(dx, width - dx))


def _load_evaluate():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    pytest.importorskip("tqdm")
    spec = importlib.util.spec_from_file_location(
        "stage_two_evaluate_248", os.path.join(REPO_ROOT, "stage_two", "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 1. the default is unchanged ---------------------------------------------------------

def test_wrap_nms_defaults_off():
    assert inspect.signature(sc.detect_peaks).parameters["wrap_nms"].default is False


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.55])
def test_default_is_bare_peak_local_max(seed, threshold):
    rng = np.random.default_rng(seed)
    h = np.clip(sc.upsample(rng.uniform(-0.2, 1.1, (64, 128))), 0, 1).astype(np.float32)
    want = peak_local_max(h, min_distance=D, threshold_abs=threshold, exclude_border=False)
    got, pk = sc.detect_peaks(h, threshold, min_distance=D, return_pixels=True)
    assert np.array_equal(pk, want)
    off, pk_off = sc.detect_peaks(h, threshold, min_distance=D, wrap_nms=False,
                                  return_pixels=True)
    assert np.array_equal(pk_off, want) and np.array_equal(off, got)


# --- 2. wrap_nms equals the brute-force cylinder -----------------------------------------

def test_wrap_nms_equals_the_cylinder_reference_on_random_maps():
    """Exact, including clipped plateaus and multi-way ties, at several thresholds."""
    rng = np.random.default_rng(24800)
    clipped = 0
    for i in range(40):
        h = random_map(rng)
        hc = np.clip(h, 0, 1)
        clipped += int((hc >= 1.0).sum() > 1)
        for t in (0.0, 0.3):
            got = pixels(sc.detect_peaks(h, t, clip=True, wrap_nms=True))
            assert got == cylinder_reference(hc, threshold=t), (i, t)
    assert clipped >= 20                     # the tie-heavy case is exercised


def test_wrap_nms_never_leaves_two_peaks_closer_than_min_distance():
    rng = np.random.default_rng(7)
    for i in range(30):
        p = pixels(sc.detect_peaks(random_map(rng), 0.0, clip=True, wrap_nms=True))
        for a in range(len(p)):
            for b in range(a + 1, len(p)):
                assert wrapped_cheb(p[a], p[b]) >= D, (i, p[a], p[b])


def test_constant_map_has_no_peak():
    h = np.full(PANO, 0.7, np.float32)
    assert len(sc.detect_peaks(h, 0.3, wrap_nms=True)) == 0
    assert len(sc.detect_peaks(h, 0.3)) == 0          # same as peak_local_max


# --- 3. the seam -------------------------------------------------------------------------

def test_straddling_pair_collapses_to_the_stronger_half():
    h = straddle_heatmap(0.8, 0.75)
    off = sc.detect_peaks(h, 0.3)
    on = sc.detect_peaks(h, 0.3, wrap_nms=True)
    assert len(off) == 3 and len(on) == 2
    assert np.array_equal(on[0], off[0])              # the interior peak, untouched
    r, c, s = on[1]
    assert c < D and s == pytest.approx(0.8, abs=0.04)
    assert any(np.array_equal(on[1], o) for o in off)
    # amplitudes swapped: the survivor is the right-edge half
    on2 = sc.detect_peaks(straddle_heatmap(0.75, 0.8), 0.3, wrap_nms=True)
    assert len(on2) == 2 and on2[1, 1] >= W - D


@pytest.mark.parametrize("dx, collapses", [(9, True), (10, True), (11, False)])
def test_the_window_is_exactly_min_distance(dx, collapses):
    """An unequal pair at wrapped |dx| = 10 collapses (the 21-px maximum filter sees the
    stronger pixel from the weaker one), at 11 it does not -- the same window
    peak_local_max applies inside the image, now across the seam."""
    h = np.zeros(PANO, np.float32)
    h[200, 1018], h[200, (1018 + dx) % W] = 0.6, 0.5
    got = pixels(sc.detect_peaks(h, 0.1, wrap_nms=True))
    assert got == cylinder_reference(h, threshold=0.1)
    assert got == ([(200, 1018)] if collapses else [(200, 1018), (200, (1018 + dx) % W)])
    # inside the image, the default behaves the same way at the same distance
    g = np.roll(h, -500, axis=1)
    assert len(sc.detect_peaks(g, 0.1)) == (1 if collapses else 2)


#: Under ``align_corners=False`` the x8 upsample is constant beyond the outermost coarse
#: centres, so hi-res columns 0-3 (and 1020-1023) of every row are an exact 4-px plateau.
EDGE_PLATEAU = 4


def test_interior_peaks_match_the_default_on_unclipped_maps():
    """Off the seam band (columns 10-1013), wrap_nms finds the same peaks with the same
    scores as the default, on unclipped near-seam maps. Every wrap peak is a default peak,
    except where a seam peak suppressed the default's pick on an edge plateau and wrap kept
    another pixel of that same plateau (same row, same score, further from the seam
    partner)."""
    rng = np.random.default_rng(32)
    seam_peaks = 0
    for i in range(30):
        h = random_map(rng, clip_ok=False)
        off = sc.detect_peaks(h, 0.0, clip=True)
        on = sc.detect_peaks(h, 0.0, clip=True, wrap_nms=True)
        band = lambda a: a[(a[:, 1] >= D) & (a[:, 1] < W - D)]
        seam_peaks += len(off) - len(band(off))
        assert {tuple(x) for x in band(on)} == {tuple(x) for x in band(off)}, i
        offs = {tuple(x) for x in off}
        plateau = lambda c: c < EDGE_PLATEAU or c >= W - EDGE_PLATEAU
        for r, c, v in on:
            if (r, c, v) in offs:
                continue
            assert plateau(c), (i, r, c)
            assert any(rr == r and vv == v and plateau(cc) and (cc < W / 2) == (c < W / 2)
                       for rr, cc, vv in offs), (i, r, c)
    assert seam_peaks >= 30                           # the maps do reach the seam


# --- 4. roll equivariance ----------------------------------------------------------------

@pytest.mark.parametrize("shift", [1, 7, 500, 1017])
def test_wrap_nms_is_roll_equivariant(shift):
    """On a cylinder the peaks of a rolled map are the rolled peaks. Exact ties break
    row-major, which a roll reorders, so the map gets a 1e-9 hi-res jitter: the 4-px edge
    plateaus (EDGE_PLATEAU) would otherwise make the pick among equal pixels move."""
    rng = np.random.default_rng(shift)
    for _ in range(5):
        h = np.clip(random_map(rng, clip_ok=False), 0, 1).astype(np.float64)
        h += rng.uniform(0, 1e-9, h.shape)
        a = {(r, (c + shift) % W) for r, c in pixels(sc.detect_peaks(h, 0.0, wrap_nms=True))}
        b = set(pixels(sc.detect_peaks(np.roll(h, shift, axis=1), 0.0, wrap_nms=True)))
        assert a == b


# --- 5. contracts ------------------------------------------------------------------------

def test_wrap_nms_refuses_exclude_border():
    with pytest.raises(ValueError, match="exclude_border"):
        sc.detect_peaks(straddle_heatmap(), 0.3, wrap_nms=True, exclude_border=True)


def test_wrap_nms_with_gaussian_decode_keeps_peaks_and_scores():
    h = straddle_heatmap()
    a = sc.detect_peaks(h, 0.3, wrap_nms=True)
    g = sc.detect_peaks(h, 0.3, decode="gaussian", wrap_nms=True, wrap_x=True)
    assert len(a) == len(g) == 2
    assert np.array_equal(a[:, 2], g[:, 2])


def test_evaluate_wrap_nms_flag_and_result_names():
    ev = _load_evaluate()
    assert ev.parse_args([]).wrap_nms is False
    assert ev.parse_args(["--wrap-nms"]).wrap_nms is True
    assert ev.results_params_str(0.0) == "r0.022_pt0.0"            # unchanged
    assert ev.results_params_str(0.0, "argmax", wrap_nms=True) == "r0.022_pt0.0_wrapnms"
    assert (ev.results_params_str(0.0, "gaussian", wrap_nms=True)
            == "r0.022_pt0.0_dgaussian_wrapnms")
    # dump_peaks_from_cache.py reads this literal out of the extractor's source (#132).
    assert "exclude_border=False" in inspect.getsource(ev.extract_peaks_from_heatmap)
    h = straddle_heatmap()
    assert len(ev.extract_peaks_from_heatmap(h, D, 0.3, PANO)) == 3
    assert len(ev.extract_peaks_from_heatmap(h, D, 0.3, PANO, wrap_nms=True)) == 2


def test_hf_package_detect_wrap_nms_without_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("transformers")
    from transformers import AutoModel
    from rampnet.model import KeypointModel, PANO_HEATMAP_SIZE
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    import export_hf_model as ex

    reference = KeypointModel(heatmap_size=PANO_HEATMAP_SIZE, pretrained_backbone=False).eval()
    pkg = str(tmp_path / "pkg")
    ex.assemble_package(pkg, reference)
    model = AutoModel.from_pretrained(pkg, trust_remote_code=True).eval()

    h = hm([(25, 0, 0.8), (25, 127, 0.75)])         # one straddling ramp, nothing else
    (off,) = model.detect(h, threshold=0.3, decode="argmax")
    (on,) = model.detect(h, threshold=0.3, decode="argmax", wrap_nms=True)
    assert len(off) == 2 and len(on) == 1
    assert on[0, 0] < D / W
    (g,) = model.detect(torch.from_numpy(h)[None, None], threshold=0.3, wrap_nms=True,
                        wrap_x=True)
    assert len(g) == 1
