"""Tests for rampnet.subcell (issue #221): CPU only, no checkpoint, no network."""
import numpy as np
import pytest

from rampnet import subcell as sc


def gaussian_coarse(cy, cx, shape=(16, 32), sigma=1.25, amp=0.9):
    """A Gaussian sampled on the coarse grid, centred at continuous cell (cy, cx)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def test_upsample_matrix_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    c = rng.normal(size=(8, 16))
    ref = torch.nn.functional.interpolate(
        torch.tensor(c)[None, None], size=(64, 128), mode="bilinear",
        align_corners=False)[0, 0].numpy()
    assert np.allclose(sc.upsample(c), ref, atol=1e-12)


def test_coarse_recovered_exactly_from_heatmap():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(16, 32))
    assert np.allclose(sc.coarse_from_heatmap(sc.upsample(c)), c, atol=1e-9)


def test_real_head_output_is_upsampled_coarse_map():
    """The mechanism claim on the actual module: KeypointModel.head(x) equals the
    bilinear upsample of conv1x1(relu(conv3x3(x))) evaluated at feature resolution."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from rampnet.model import KeypointModel
    torch.manual_seed(0)
    m = KeypointModel(heatmap_size=(64, 128)).eval()
    head = m.head
    f = torch.randn(1, head[0].in_channels, 8, 16)
    with torch.no_grad():
        hi = head(f)[0, 0].double().numpy()
        coarse = head[3](head[1](head[0](f)))[0, 0].double().numpy()
    assert np.allclose(hi, sc.upsample(coarse), atol=1e-4)
    assert np.allclose(sc.coarse_from_heatmap(hi), coarse, atol=1e-4)


@pytest.mark.parametrize("off", [-0.45, -0.3, -0.1, 0.0, 0.2, 0.37, 0.49])
def test_argmax_is_quantized_to_flanking_pixels(off):
    c = gaussian_coarse(7 + off, 13 - off)
    h = sc.upsample(c)
    r, col = np.unravel_index(np.argmax(h), h.shape)
    assert r % 8 in (3, 4) and col % 8 in (3, 4)


@pytest.mark.parametrize("oy,ox", [(0.3, -0.2), (-0.45, 0.4), (0.0, 0.1), (0.25, 0.25)])
def test_gaussian_and_dark_recover_subcell_position(oy, ox):
    cy, cx = 7 + oy, 13 + ox
    h = sc.upsample(gaussian_coarse(cy, cx))
    pk = np.array([np.unravel_index(np.argmax(h), h.shape)])
    truth = np.array([(8 * cx + 3.5) / h.shape[1], (8 * cy + 3.5) / h.shape[0]])
    for m in ("gaussian", "dark"):
        assert np.allclose(sc.refine_peaks(h, pk, m)[0], truth, atol=1e-6), m
    err = lambda m: np.abs(sc.refine_peaks(h, pk, m)[0] - truth).max()  # noqa: E731
    for m in ("parabola", "centroid"):
        assert err(m) <= err("argmax") + 1e-12, m
    # quarter moves a fixed 0.25 cell toward the higher neighbour: right direction, but
    # worse than argmax when the true offset is near 0 (that is a property of the rule)
    d = sc.refine_peaks(h, pk, "quarter")[0] - sc.refine_peaks(h, pk, "centre")[0]
    for got, want in ((d[0], ox), (d[1], oy)):
        assert want == 0 or got == 0 or np.sign(got) == np.sign(want)


def test_argmax_method_is_identity():
    h = sc.upsample(gaussian_coarse(7.3, 12.8))
    pk = np.array([[59, 107], [3, 4]])
    out = sc.refine_peaks(h, pk, "argmax")
    assert np.allclose(out, [[107 / 256, 59 / 128], [4 / 256, 3 / 128]])


def test_centre_is_half_pixel_from_argmax():
    h = sc.upsample(gaussian_coarse(7.2, 12.9))
    pk = np.array([np.unravel_index(np.argmax(h), h.shape)])
    d = (sc.refine_peaks(h, pk, "centre") - sc.refine_peaks(h, pk, "argmax"))[0]
    assert np.allclose(np.abs(d * [256, 128]), 0.5)


def test_clipped_plateau_climbs_to_coarse_max():
    c = gaussian_coarse(7.2, 12.9, amp=3.0)      # far above 1: clip makes a plateau
    h = sc.upsample(c)
    # a plateau pixel far from the coarse max, as clipped peak_local_max may return
    r, col = 7 * 8 + 3 - 8, 13 * 8 + 4 + 8
    assert np.clip(h, 0, 1)[r, col] == 1.0
    out = sc.refine_peaks(h, [[r, col]], "gaussian")[0]
    assert np.allclose(out, [(8 * 12.9 + 3.5) / 256, (8 * 7.2 + 3.5) / 128], atol=1e-6)


def test_edges_do_not_refine_without_neighbours():
    c = gaussian_coarse(7.0, 0.3)
    h = sc.upsample(c)
    pk = np.array([np.unravel_index(np.argmax(h), h.shape)])
    x = sc.refine_peaks(h, pk, "gaussian")[0, 0] * 256
    assert np.isclose(x, 3.5)                           # col 0: no left neighbour
    xw = sc.refine_peaks(h, pk, "gaussian", wrap_x=True)[0, 0] * 256
    assert 3.5 < xw < 8 * 0.5 + 3.5                     # wrap: moves right, bounded


def test_offsets_are_clamped():
    n = np.array([[0, 0.1, 0], [0.0, 1.0, 0.999], [0, 0.1, 0]])
    for m in ("parabola", "gaussian", "dark", "centroid", "quarter"):
        dy, dx = sc.refine_offset(n, m)
        assert abs(dy) <= sc.MAX_OFFSET and abs(dx) <= sc.MAX_OFFSET


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        sc.refine_offset(np.ones((3, 3)), "nope")
