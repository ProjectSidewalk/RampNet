"""Seed bookkeeping for Stage 2 training.

Stage 2 had two independent sources of run-to-run randomness and only one of them was
ever set:

* ``torch`` / ``numpy`` / ``random`` were seeded to a hardcoded ``42`` -- that governs
  weight initialization for the head, dropout, and the augmentation draws.
* ``DistributedSampler`` carries its **own** ``seed`` (default ``0``) and derives each
  epoch's permutation from ``seed + epoch`` inside ``set_epoch()``. Nothing in
  ``train.py`` touched it, so it stayed at ``0``.

So every published run is the pair ``(42, 0)``, and the recipe's spread across seeds has
never been measured -- it is n=1. That was a footnote while the RampNet-vs-YOLO gap was
0.252 F1; at the matched-operating-point gap of 0.039 it is the binding number. See
``docs/seed_variance_51_135.md``.

The trap this module exists to close: a sweep that varies only the torch seed reuses one
data order across every arm, which understates the true spread, and **does it silently**
-- no log line distinguishes the two. So the two seeds move together, with one exception
that has to be exact: at the historical torch seed the sampler must stay at its
historical ``0``, or the default stops reproducing the published runs.
"""

HISTORICAL_SEED = 42
"""The torch/numpy/random seed every published Stage 2 run used."""

HISTORICAL_SAMPLER_SEED = 0
"""``DistributedSampler``'s default, which every published Stage 2 run inherited."""


def sampler_seed_for(seed: int) -> int:
    """Return the ``DistributedSampler`` seed that pairs with ``seed``.

    At :data:`HISTORICAL_SEED` this is :data:`HISTORICAL_SAMPLER_SEED`, so the default
    reproduces published runs exactly. Every other seed maps to itself, so a sweep gets
    a genuinely different data order as well as different initialization.

    The asymmetry is deliberate and is the whole point of the function: it is the only
    way to add a seed flag without silently changing what the default does.
    """
    if seed == HISTORICAL_SEED:
        return HISTORICAL_SAMPLER_SEED
    return seed
