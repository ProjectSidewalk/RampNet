"""Unit tests for the paired seam-response experiment (#132, docs/seam.md section 4a).

Pure core only — no torch inference, no imagery, no GPU. What these protect:

* the arm binning. The first committed run of this experiment put ramps the roll is
  not neutral for into the control arm — the centre-band ramp the roll moves ONTO the
  seam, and near-seam ramps whose response window straddles it — and all three of the
  control arm's "lost" rows were exactly those ramps. The binning rules exist to make
  the control mean what the docstring says it means, so they are pinned here.
* the committed artifact's internal consistency: the summary block must be exactly
  re-derivable from the rows in the same file, every control row must actually satisfy
  the binning rules, and the bytes must stay LF with floats at six decimals
  (see json-artifacts rounding note in tests/test_farfield_forensics.py).
"""
import json
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import seam_response as sr  # noqa: E402
from rampnet.detection_eval import PANO_SCALE_X, PANO_SCALE_Y, radius_sq_for  # noqa: E402
from rampnet.geometry import dist_sq  # noqa: E402

R = radius_sq_for() ** 0.5           # 22.53 matcher px
BAND = sr.SEAM_BAND                  # 11.27 matcher px
ARTIFACT = os.path.join(REPO, "analysis_out", "seam_response.json")


def x_at(seam_px):
    """A normalized x whose dist_to_seam is ``seam_px`` (left side of the seam)."""
    return seam_px / PANO_SCALE_X


# --------------------------------------------------------------------------- #
# arm_of — the window-geometry bins
# --------------------------------------------------------------------------- #
def test_seam_band_is_the_measured_arm():
    assert sr.arm_of(BAND - 0.01) == "seam"


def test_window_touching_the_stored_seam_is_excluded():
    assert sr.arm_of(BAND + 0.01) == "excluded"
    assert sr.arm_of(R - 0.01) == "excluded"


def test_window_touching_the_rolled_seam_position_is_excluded():
    # The roll lands the seam on the centre column; a window there reads it.
    assert sr.arm_of(PANO_SCALE_X / 2 - R + 0.01) == "excluded"


def test_clear_of_both_seam_positions_is_control():
    assert sr.arm_of(R + 0.01) == "control"
    assert sr.arm_of(PANO_SCALE_X / 2 - R - 0.01) == "control"


def test_the_first_runs_contaminating_rows_are_now_excluded():
    # The three "lost" control rows of the first committed run, by seam_px:
    # two window-straddles-seam (18.30, 18.43) and the centre ramp (507.52).
    assert sr.arm_of(18.30) == "excluded"
    assert sr.arm_of(18.43) == "excluded"
    assert sr.arm_of(507.52) == "excluded"


# --------------------------------------------------------------------------- #
# bin_pano — the window-overlap exclusions
# --------------------------------------------------------------------------- #
def test_control_overlapping_a_seam_ramps_window_is_excluded():
    seam_g = (x_at(5.0), 0.5)
    near = (x_at(5.0 + 1.5 * R), 0.5)      # control band, but within 2R of the seam GT
    out = dict((tuple(g), (a, why)) for g, a, why in sr.bin_pano([seam_g, near]))
    assert out[tuple(seam_g)][0] == "seam"
    arm, why = out[tuple(near)]
    assert arm == "excluded" and "seam-band" in why


def test_adjacent_controls_keep_exactly_one():
    a = (x_at(200.0), 0.5)
    b = (x_at(200.0 + R), 0.5)             # within 2R of a: windows can overlap
    arms = sorted(arm for _, arm, _ in sr.bin_pano([a, b]))
    assert arms == ["control", "excluded"]


def test_which_control_survives_does_not_depend_on_input_order():
    a = (x_at(200.0), 0.5)
    b = (x_at(200.0 + R), 0.5)
    kept_ab = {tuple(g) for g, arm, _ in sr.bin_pano([a, b]) if arm == "control"}
    kept_ba = {tuple(g) for g, arm, _ in sr.bin_pano([b, a]) if arm == "control"}
    assert kept_ab == kept_ba


def test_independent_controls_all_survive():
    gs = [(x_at(100.0), 0.5), (x_at(200.0), 0.5), (x_at(300.0), 0.5)]
    assert all(arm == "control" for _, arm, _ in sr.bin_pano(gs))


def test_seam_band_ramps_are_never_dropped():
    gs = [(x_at(2.0), 0.5), (x_at(6.0), 0.5)]  # adjacent seam ramps, shared window
    assert [arm for _, arm, _ in sr.bin_pano(gs)] == ["seam", "seam"]
    assert sr.seam_window_overlap_pairs(gs) == 1


# --------------------------------------------------------------------------- #
# summarize — strict thresholds, empty guard
# --------------------------------------------------------------------------- #
def _rows(diffs):
    # stored 0.0 keeps rolled - stored exact in floating point at the boundary
    return [{"stored": 0.0, "rolled": d} if d >= 0 else {"stored": -d, "rolled": 0.0}
            for d in diffs]


def test_moved_means_strictly_more_than_material():
    s = sr.summarize(_rows([sr.MATERIAL, -sr.MATERIAL, 0.0]))
    assert (s["gained"], s["lost"]) == (0, 0)
    s = sr.summarize(_rows([sr.MATERIAL + 1e-4, -(sr.MATERIAL + 1e-4)]))
    assert (s["gained"], s["lost"]) == (1, 1)


def test_empty_group_summarizes_to_empty():
    assert sr.summarize([]) == {}


def test_round_floats_walks_the_whole_object():
    out = sr.round_floats({"a": [0.123456789, {"b": 1.0000004}], "n": 3})
    assert out == {"a": [0.123457, {"b": 1.0}], "n": 3}


# --------------------------------------------------------------------------- #
# the committed artifact — internally consistent, portable bytes
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artifact():
    with open(ARTIFACT, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw.decode("utf-8"))


def test_artifact_is_lf_only(artifact):
    raw, _ = artifact
    assert b"\r" not in raw


def test_artifact_floats_carry_at_most_six_decimals(artifact):
    raw, _ = artifact
    too_long = re.findall(rb"\d\.\d{7,}", raw)
    assert too_long == []


def test_summary_is_rederivable_from_the_rows(artifact):
    _, j = artifact
    for arm, key in (("seam", "seam_rows"), ("control", "control_rows")):
        assert j["summary"][arm] == sr.round_floats(sr.summarize(j[key]))


def test_no_control_row_violates_the_binning_rules(artifact):
    _, j = artifact
    for r in j["control_rows"]:
        assert sr.arm_of(r["seam_px"]) == "control", r
    by_pano = {}
    for arm in ("seam_rows", "control_rows"):
        for r in j[arm]:
            by_pano.setdefault(r["pano"], {"seam_rows": [], "control_rows": []})[
                arm].append((r["x"], r["y"]))
    min_sq = (2 * R) ** 2
    for pano, g in by_pano.items():
        pts = g["control_rows"]
        others = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]]
        others += [(a, b) for a in pts for b in g["seam_rows"]]
        for a, b in others:
            assert dist_sq(a[0], a[1], b[0], b[1], PANO_SCALE_X, PANO_SCALE_Y,
                           wrap_x=True) >= min_sq, (pano, a, b)


def test_every_excluded_row_states_why(artifact):
    _, j = artifact
    assert j["excluded_rows"], "the exclusions this experiment exists for are absent"
    assert all(r.get("reason") for r in j["excluded_rows"])


def test_artifact_carries_its_provenance(artifact):
    _, j = artifact
    for key in ("model_id", "model_revision", "panos_root", "cache_dir",
                "op_cache_sha256", "panos", "skipped"):
        assert key in j, key
    assert j["skipped"] == [], "the committed artifact must come from a complete run"
    assert len(j["seam_rows"]) == j["summary"]["seam"]["n"]
    assert len(j["control_rows"]) == j["summary"]["control"]["n"]
