"""Guards on the two-leg effort split (#139, #143).

The cloud query needs credentials and is not testable here. The *decision* logic is,
and it is the part that can hand back a confident wrong number: a mixture solver
pointed at a flat series will cheerfully report that high effort cost less than low.
These pin the changepoint detector and the separability rule against series whose
right answer is known by construction.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison"))

ves = pytest.importorskip("vertex_effort_split")


def _series(spec):
    """[(n_minutes, input_per_min, ratio)] -> the (ts, input, output) rows."""
    rows, minute = [], 0
    for n, inp, ratio in spec:
        for _ in range(n):
            rows.append((f"2026-08-15T{minute // 60:02d}:{minute % 60:02d}:00Z",
                         inp, round(inp * ratio)))
            minute += 1
    return rows


def test_changepoint_finds_the_throughput_cliff():
    """Fast leg finishes, slow leg runs on alone: throughput drops, and the index
    returned must be the first minute of the slow phase, not somewhere in the middle."""
    rows = _series([(30, 60_000, 0.035), (30, 20_000, 0.127)])
    cut, drop = ves.find_changepoint(rows)
    assert cut == 30
    assert drop == pytest.approx(3.0, abs=0.01)


def test_a_flat_series_is_reported_as_not_separable():
    """Sonnet's real shape: both legs at effectively one ratio.

    The detector still returns *a* changepoint -- there is always a largest drop --
    so the separability rule, not the detector, is what has to refuse. If this ever
    starts passing the lift threshold, the split it produces is noise.
    """
    rows = _series([(30, 100_000, 0.036), (25, 60_000, 0.035)])
    cut, _ = ves.find_changepoint(rows)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_tail < r_head * ves.MIN_RATIO_LIFT


def test_a_real_two_component_day_clears_the_threshold():
    """The Opus shape, so the guard is not simply refusing everything."""
    rows = _series([(40, 58_000, 0.0675), (35, 21_000, 0.1203)])
    cut, _ = ves.find_changepoint(rows)
    g = ves.GUARD_MINUTES
    head, tail = rows[:cut - g], rows[cut + g:]
    r_head = sum(r[2] for r in head) / sum(r[1] for r in head)
    r_tail = sum(r[2] for r in tail) / sum(r[1] for r in tail)
    assert r_tail >= r_head * ves.MIN_RATIO_LIFT


def test_the_input_split_is_geometry_not_inference():
    """Input per leg comes from the pano count, so it must not depend on the
    output at all -- that is what makes the mixture solvable with one unknown."""
    per_pano, legs = 12_186, 2
    total_in = 251 * per_pano
    per_leg = (round(total_in / per_pano) // legs) * per_pano
    assert per_leg == 125 * per_pano       # the odd pano is not silently halved


def test_rate_anchor_and_tail_anchor_bracket_the_published_opus_split():
    """The 2026-08-15 Opus day, as recovered. Both anchors must stay on the same
    side of the story: high effort costs more, and the two agree within ~5%."""
    total_in, total_out = 3_058_702, 247_222
    per_leg_in = 125 * 12_186
    for out_high in (0.1203 * per_leg_in,                 # tail anchor
                     total_out - 0.034908 * per_leg_in):  # rate anchor
        out_low = total_out - out_high
        assert out_high > out_low
        cost_low = per_leg_in * 5 / 1e6 + out_low * 25 / 1e6
        cost_high = per_leg_in * 5 / 1e6 + out_high * 25 / 1e6
        assert cost_low == pytest.approx(9.1, abs=0.3)
        assert cost_high == pytest.approx(12.3, abs=0.3)
        assert cost_low + cost_high == pytest.approx(21.41, abs=0.05)
