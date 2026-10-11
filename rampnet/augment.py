"""Pixel-statistics transforms for issue #82 (rig-transfer augmentation).

One implementation serves two callers, so the probe and the training arms cannot drift
apart:

* ``scripts/analysis/aug_probe_82.py`` applies ONE transform at a FIXED level to every
  benchmark pano and scores the frozen released model on the result (Step 1);
* ``stage_two/train.py --aug ...`` applies RANDOM draws of the same transforms to
  training panos (Step 2/3).

**Where in the pipeline.** Every transform takes and returns an RGB ``PIL.Image`` at the
model's input size, 2048x4096. In training the panos are stored at 2048x4096 already, so
the transform runs after the horizontal flip and before ``transforms.Resize`` (a no-op at
that size) / ``ToTensor`` / ``Normalize``. In the probe the native pano is first resized
to 2048x4096 by the same bilinear ``Resize`` the scorer uses (``threshold_sweep.PRE``),
then transformed, then passed through ``PRE`` again (a no-op resize). Levels are
therefore in model-input pixels: a blur sigma of 1.0 means 1.0 px at 2048x4096 on every
split, whatever the split's native width (5,760 to 16,384).

All transforms are pixel-wise with no geometry change, so curb ramp labels are untouched.

Usage::

    from rampnet import augment as A
    img = A.apply_op(img, "blur", 1.2)                      # fixed level (probe)
    aug = A.Augmenter(A.parse_specs(["blur=0.5:0.5:1.5"]), seed=1)
    img = aug(img, idx=1234, epoch=0)                         # random draw (training)
"""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------- #
# deterministic transforms (level in, image out)
# --------------------------------------------------------------------------- #


def downscale(img, factor):
    """Resize to ``factor`` of each side (bilinear, which PIL antialiases when reducing)
    and back to the original size (bilinear). ``factor`` 0.5 keeps a quarter of the
    pixels' worth of detail. 1.0 is the identity."""
    if factor >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, round(w * factor)), max(1, round(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def blur(img, sigma):
    """Gaussian blur, ``sigma`` in input pixels (PIL's ``radius`` is the standard
    deviation)."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def jpeg(img, quality):
    """Re-encode as JPEG at ``quality`` (PIL defaults: 4:2:0 chroma subsampling)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(round(quality)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def brightness(img, factor):
    """Multiply every channel by ``factor`` (``ImageEnhance.Brightness``)."""
    return img if factor == 1.0 else ImageEnhance.Brightness(img).enhance(float(factor))


def contrast(img, factor):
    """Scale the distance from the mean grey by ``factor`` (``ImageEnhance.Contrast``)."""
    return img if factor == 1.0 else ImageEnhance.Contrast(img).enhance(float(factor))


def saturation(img, factor):
    """Scale chroma by ``factor`` (``ImageEnhance.Color``; 0 = greyscale)."""
    return img if factor == 1.0 else ImageEnhance.Color(img).enhance(float(factor))


def _lut(img, fn):
    x = np.arange(256, dtype=np.float64)
    table = np.clip(np.round(fn(x)), 0, 255).astype(np.uint8).tolist()
    return img.point(table * 3)


def gamma(img, g):
    """``out = 255 * (in / 255) ** g``. g > 1 darkens mid-tones, g < 1 lifts them."""
    if g == 1.0:
        return img
    return _lut(img, lambda x: 255.0 * (x / 255.0) ** float(g))


def _channel_gains(img, gains):
    r, g, b = img.split()
    out = []
    for ch, k in zip((r, g, b), gains):
        if k == 1.0:
            out.append(ch)
        else:
            table = np.clip(np.round(np.arange(256) * k), 0, 255).astype(np.uint8).tolist()
            out.append(ch.point(table))
    return Image.merge("RGB", out)


def white_balance(img, t):
    """Warm/cool shift along the red-blue axis: red gain ``2**(t/2)``, blue gain
    ``2**(-t/2)``, so ``log2(R/B)`` moves by ``t``. t > 0 is warmer."""
    if t == 0:
        return img
    return _channel_gains(img, (2.0 ** (t / 2.0), 1.0, 2.0 ** (-t / 2.0)))


def hue(img, degrees):
    """Rotate hue by ``degrees`` (PIL HSV, 256 steps per turn)."""
    if degrees == 0:
        return img
    h, s, v = img.convert("HSV").split()
    shift = int(round(degrees / 360.0 * 256)) % 256
    h = h.point(lambda x: (x + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def noise(img, sigma, rng):
    """Additive Gaussian noise, ``sigma`` in 8-bit units, i.i.d. per pixel and channel.
    ``rng`` is a ``numpy.random.Generator`` (never the global stream)."""
    if sigma <= 0:
        return img
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0.0, float(sigma), size=a.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))


def unsharp(img, percent, radius=2.0, threshold=0):
    """Unsharp mask (``ImageFilter.UnsharpMask``); a probe-only repair transform."""
    if percent <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(round(percent)),
                                              threshold=threshold))


def clahe(img, clip):
    """CLAHE on the luma channel (YCbCr Y), chroma untouched; a probe-only repair
    transform. ``clip`` is skimage's ``clip_limit``; tiles are 1/8 of each side."""
    if clip <= 0:
        return img
    from skimage.exposure import equalize_adapthist
    y, cb, cr = img.convert("YCbCr").split()
    ya = np.asarray(y, dtype=np.float64) / 255.0
    ks = (max(8, ya.shape[0] // 8), max(8, ya.shape[1] // 8))
    eq = equalize_adapthist(ya, kernel_size=ks, clip_limit=float(clip))
    y2 = Image.fromarray(np.clip(np.round(eq * 255.0), 0, 255).astype(np.uint8))
    return Image.merge("YCbCr", (y2, cb, cr)).convert("RGB")


def colour_match(img, ref_mean, ref_std, alpha=1.0):
    """Per-channel RGB mean/std transfer toward a reference (Reinhard-style, in RGB),
    blended by ``alpha``; a probe-only repair transform. ``ref_mean`` / ``ref_std`` are
    3-vectors in 8-bit units, measured from the GSV splits by the probe's ``stats``."""
    a = np.asarray(img, dtype=np.float64)
    m = a.reshape(-1, 3).mean(axis=0)
    s = a.reshape(-1, 3).std(axis=0) + 1e-6
    t = (a - m) / s * np.asarray(ref_std, dtype=np.float64) + np.asarray(ref_mean,
                                                                           dtype=np.float64)
    out = (1.0 - alpha) * a + alpha * t
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


#: op name -> function(img, level[, rng]). Every op is the identity at its neutral level.
OPS = {
    "downscale": downscale, "blur": blur, "jpeg": jpeg, "brightness": brightness,
    "contrast": contrast, "saturation": saturation, "gamma": gamma,
    "wb": white_balance, "hue": hue, "noise": noise, "unsharp": unsharp, "clahe": clahe,
}
NEUTRAL = {"downscale": 1.0, "blur": 0.0, "jpeg": None, "brightness": 1.0,
           "contrast": 1.0, "saturation": 1.0, "gamma": 1.0, "wb": 0.0, "hue": 0.0,
           "noise": 0.0, "unsharp": 0.0, "clahe": 0.0}
#: Ops ``train.py --aug`` accepts, in the order they are applied: optics/resolution
#: first, then colour, then sensor noise, and compression last -- the order a camera
#: pipeline applies them. unsharp / clahe / colour_match are repair transforms for the
#: probe only.
TRAIN_ORDER = ("downscale", "blur", "brightness", "contrast", "saturation", "gamma",
               "wb", "hue", "noise", "jpeg")


def apply_op(img, op, level, rng=None):
    """Apply one op at a fixed level. ``noise`` needs ``rng``."""
    if op == "noise":
        if rng is None:
            raise ValueError("noise needs an rng")
        return noise(img, level, rng)
    return OPS[op](img, level)


# --------------------------------------------------------------------------- #
# random draws for training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AugSpec:
    """Apply ``op`` with probability ``p`` at a level drawn uniformly from [lo, hi]."""
    op: str
    p: float
    lo: float
    hi: float

    def __str__(self):
        return f"{self.op}={self.p:g}:{self.lo:g}:{self.hi:g}"


def parse_spec(s):
    """``"blur=0.5:0.3:1.5"`` -> AugSpec("blur", 0.5, 0.3, 1.5).

    >>> parse_spec("jpeg=1:40:80")
    AugSpec(op='jpeg', p=1.0, lo=40.0, hi=80.0)
    """
    try:
        op, rest = s.split("=", 1)
        p, lo, hi = (float(x) for x in rest.split(":"))
    except ValueError:
        raise ValueError(f"--aug {s!r}: expected op=p:lo:hi, e.g. blur=0.5:0.3:1.5")
    op = op.strip()
    if op not in TRAIN_ORDER:
        raise ValueError(f"--aug {s!r}: unknown op {op!r}; known: {', '.join(TRAIN_ORDER)}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"--aug {s!r}: p must be in [0, 1]")
    if lo > hi:
        raise ValueError(f"--aug {s!r}: lo > hi")
    return AugSpec(op, p, lo, hi)


def parse_specs(items):
    """Parse a list of ``op=p:lo:hi`` strings; an op may appear once. Returned in
    ``TRAIN_ORDER`` whatever order they were given in."""
    specs = [parse_spec(s) for s in (items or [])]
    ops = [s.op for s in specs]
    dup = sorted({o for o in ops if ops.count(o) > 1})
    if dup:
        raise ValueError(f"--aug given twice for: {', '.join(dup)}")
    return tuple(sorted(specs, key=lambda s: TRAIN_ORDER.index(s.op)))


#: Domain tag mixed into every per-sample seed so this stream cannot coincide with any
#: other seeded stream in train.py.
STREAM_TAG = 82


def sample_rng(seed, epoch, idx):
    """The generator for one sample's draws: a pure function of (seed, epoch, idx).

    Not the global ``random`` / ``numpy`` / ``torch`` stream, on purpose: the horizontal
    flip draws from ``random`` inside the DataLoader worker, so any augmentation that
    consumed that stream would shift every later flip and an augmented arm would stop
    being paired with its same-seed control. Keying on the sample index (not on a
    running counter) also makes a resumed job re-draw exactly what an uninterrupted job
    would have drawn for the same sample."""
    return np.random.default_rng(np.random.SeedSequence([STREAM_TAG, int(seed),
                                                         int(epoch), int(idx)]))


def draw(specs, rng):
    """[(op, level)] for one sample. Two numbers are consumed per spec whether or not it
    fires, so one op's probability never shifts another op's level."""
    out = []
    for s in specs:
        u, v = rng.random(), rng.uniform(s.lo, s.hi)
        if u < s.p:
            out.append((s.op, float(v)))
    return out


class Augmenter:
    """Picklable callable for ``EquiHeatmapDataset``: ``aug(img, idx, epoch) -> img``."""

    def __init__(self, specs, seed):
        self.specs = tuple(specs)
        self.seed = int(seed)

    def __bool__(self):
        return bool(self.specs)

    def params(self, idx, epoch):
        rng = sample_rng(self.seed, epoch, idx)
        return draw(self.specs, rng), rng

    def __call__(self, img, idx, epoch=0):
        ops, rng = self.params(idx, epoch)
        for op, level in ops:
            img = apply_op(img, op, level, rng=rng)
        return img

    def describe(self):
        return " ".join(str(s) for s in self.specs) or "none"
