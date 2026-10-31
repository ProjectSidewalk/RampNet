"""Tests for the image-based tilt estimator (rampnet/pano_tilt.py, scripts/analysis/pano_tilt_estimate.py).

CPU only, no network, no panorama files: the fixtures are synthetic equirects built in the test (a
"street" of vertical facade edges at every bearing, rendered level and then tilted by a known
pitch and roll with the #113 rotation), plus the committed records, id list and per-pano rows.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rampnet import pano_tilt as pt  # noqa: E402
from rampnet import stage1_geometry as sg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_out" / "pano_tilt"


def _load_script():
    spec = importlib.util.spec_from_file_location("pano_tilt_estimate",
                                                  REPO / "scripts/analysis/pano_tilt_estimate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


te = _load_script()

#: Synthetic panos are 1024 wide; the estimator's views scale with the input width.
SYN_W = 1024
SYN_PARAMS = pt.TiltParams(work_width=SYN_W)


def synthetic_street(width=SYN_W, seed=0):
    """A gravity-level grey equirect: facades of random-width blocks all the way round.

    Block edges are image columns, i.e. gravity-vertical lines at every bearing; facades span
    elevation -20..(15..50) degrees, the sky is bright, the road dark with noise, and each block
    has a few horizontal window bands as distractors.
    """
    rng = np.random.default_rng(seed)
    h = width // 2
    img = np.full((h, width), 0.85, np.float32)                        # sky
    el = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    road = el < -20
    img[road] = 0.3 + 0.05 * rng.standard_normal((road.sum(), width)).astype(np.float32)
    x = 0
    while x < width:
        bw = int(rng.integers(width // 64, width // 20))
        top = rng.uniform(15, 50)
        val = rng.uniform(0.1, 0.6)
        rows = (el >= -20) & (el <= top)
        img[rows, x:x + bw] = val
        for wb in rng.uniform(-10, top - 3, size=2):                   # window bands
            band = (el >= wb) & (el <= wb + 1.5)
            img[band, x + 2:max(x + 3, x + bw - 2)] = val + 0.2
        x += bw
    return cv2.GaussianBlur(img, (3, 3), 0.7)


class TestConventions:
    @pytest.mark.parametrize("p, r", [(0.0, 0.0), (1.3, -2.2), (-24.7, -0.5), (11.8, -4.2), (5.0, 10.0)])
    def test_up_round_trip(self, p, r):
        u = pt.up_from_pitch_roll(p, r)
        assert np.linalg.norm(u) == pytest.approx(1.0)
        p2, r2 = pt.pitch_roll_from_up(u)
        assert p2 == pytest.approx(p, abs=1e-9) and r2 == pytest.approx(r, abs=1e-9)

    @pytest.mark.parametrize("p, r", [(3.0, 0.0), (0.0, 3.0), (2.0, -1.5)])
    def test_up_is_normal_to_the_gravity_horizon(self, p, r):
        """u is orthogonal to every gravity-horizon direction, which sits at elevation T(b) (#113)."""
        u = pt.up_from_pitch_roll(p, r)
        for b in (-135.0, -60.0, 0.0, 45.0, 90.0, 170.0):
            rb, rel = sg.gravity_to_rig(b, 0.0, p, r)
            v = sg.direction_rfu(rb, rel)
            assert float(v @ u) == pytest.approx(0.0, abs=1e-12)
            # to first order the horizon sits T(b) above the middle row (pitch > 0 = nose down)
            assert rel == pytest.approx(float(sg.tilt_term_deg(b, p, r)), abs=0.02)

    @pytest.mark.parametrize("p, r", [(5.0, 0.0), (0.0, 5.0), (10.0, -4.0), (-3.0, 2.0)])
    def test_horizon_pixels_follow_the_documented_convention(self, p, r):
        """Read the horizon off the pixels of a tilted pano; it must sit at T(b) = p cos b + r sin b.

        Independent of ``rig_from_gravity`` on the reading side: a level pano with a sharp
        gravity horizon (bright sky, dark ground) is tilted with ``tilt_equirect``, and the
        horizon row is found in pixels at several bearings. T(b) is first order in the tilt; at
        10 degrees its error is under 0.1 degree, half a pixel at 2048 wide is 0.09 degree.
        """
        w, h = 2048, 1024
        el = 90.0 - (np.arange(h) + 0.5) / h * 180.0
        g = np.where(el > 0, 0.9, 0.1).astype(np.float32)[:, None].repeat(w, axis=1)
        t = pt.tilt_equirect(g, p, r)
        for b in (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0):
            x = int(round((b + 180.0) / 360.0 * w - 0.5))
            col = t[:, x].astype(float)
            i = int(np.argmax((col[:-1] >= 0.5) & (col[1:] < 0.5)))       # first sky->ground step
            y = i + 0.5 + (col[i] - 0.5) / (col[i] - col[i + 1])
            b_px = (x + 0.5) / w * 360.0 - 180.0
            assert 90.0 - y / h * 180.0 == pytest.approx(float(sg.tilt_term_deg(b_px, p, r)), abs=0.25)

    def test_level_inverts_tilt(self):
        g = synthetic_street(512)
        back = pt.level_equirect(pt.tilt_equirect(g, 3.0, -2.0), 3.0, -2.0)
        mid = slice(40, 216)                                           # away from the poles
        assert np.median(np.abs(back[mid] - g[mid])) < 0.02

    def test_render_view_is_level_for_a_level_pano(self):
        """In a pitch-0 view of a level pano, a facade edge is an image column."""
        g = synthetic_street()
        segs = pt.detect_segments(pt.render_view(g, pt.make_views(SYN_PARAMS, SYN_W)[0]))
        dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        long_vertical = (np.hypot(dx, dy) > 20) & (np.abs(dx) < np.abs(dy))
        assert long_vertical.sum() >= 5
        assert np.max(np.degrees(np.arctan2(np.abs(dx[long_vertical]), np.abs(dy[long_vertical])))) < 1.0


class TestSyntheticRecovery:
    @pytest.mark.parametrize("p, r", [(0.0, 0.0), (2.0, -1.0), (-3.5, 2.5), (-24.7, -0.5), (6.0, 9.0)])
    def test_recovers_known_tilt(self, p, r):
        g = pt.tilt_equirect(synthetic_street(), p, r)
        est = pt.estimate_tilt(g, SYN_PARAMS)
        assert est.status == "ok", est.reason
        assert est.pitch_deg == pytest.approx(p, abs=0.25)
        assert est.roll_deg == pytest.approx(r, abs=0.25)
        assert est.n_sectors >= 6 and est.ambiguity <= SYN_PARAMS.max_ambiguity

    def test_declines_a_pano_with_no_structure(self):
        rng = np.random.default_rng(1)
        g = cv2.GaussianBlur(rng.random((SYN_W // 2, SYN_W)).astype(np.float32), (9, 9), 3)
        est = pt.estimate_tilt(g, SYN_PARAMS)
        assert est.status == "declined" and est.reason

    def test_rejects_non_2_to_1(self):
        with pytest.raises(ValueError):
            pt.estimate_tilt(np.zeros((100, 300), np.float32), SYN_PARAMS)


def _vertical_constraints(p, r, bearings, rng, noise_deg=0.0):
    """Exact gravity-vertical segments at the given bearings, seen by a rig tilted (p, r)."""
    R = sg.rig_from_gravity(p, r)
    ns, ls, bs = [], [], []
    for b in bearings:
        e0, e1 = sorted(rng.uniform(-10, 40, 2))
        v0 = R @ sg.direction_rfu(b, e0)
        v1 = R @ sg.direction_rfu(b, e1 + 3)
        n = np.cross(v0, v1)
        n /= np.linalg.norm(n)
        if noise_deg:
            n = n + math.radians(noise_deg) * rng.standard_normal(3)
            n /= np.linalg.norm(n)
        ns.append(n)
        ls.append(e1 + 3 - e0)
        bs.append(sg.bearing_elevation(v0)[0])
    z = np.zeros(len(ns))
    return {"normal": np.array(ns), "length_deg": np.array(ls), "bearing_deg": np.array(bs), "lean_deg": z}


class TestFromConstraints:
    def test_exact_lines_all_round(self):
        rng = np.random.default_rng(0)
        cons = _vertical_constraints(1.7, -0.9, rng.uniform(-180, 180, 80), rng, noise_deg=0.05)
        est = pt.estimate_from_constraints(cons)
        assert est.status == "ok", est.reason
        assert est.pitch_deg == pytest.approx(1.7, abs=0.05)
        assert est.roll_deg == pytest.approx(-0.9, abs=0.05)

    def test_lines_in_one_direction_only_are_declined(self):
        """Verticals at a single bearing pin only one tilt component; the gate must say so."""
        rng = np.random.default_rng(0)
        cons = _vertical_constraints(1.7, -0.9, rng.uniform(-10, 10, 60), rng, noise_deg=0.05)
        est = pt.estimate_from_constraints(cons)
        assert est.status == "declined"
        assert "sectors" in est.reason

    def test_two_competing_up_directions_are_declined_as_ambiguous(self):
        """Two equally supported verticals 5 degrees apart: only the ambiguity test can catch it.

        This is the decline path that does the work on real panos (it is among the reasons for
        134 of 137 GSV declines and all 7 Lyon declines): every other test passes here.
        """
        rng = np.random.default_rng(0)
        b = rng.uniform(-180, 180, 60)
        a = _vertical_constraints(0.0, 0.0, b, rng, noise_deg=0.05)
        c = _vertical_constraints(5.0, 0.0, b, rng, noise_deg=0.05)
        est = pt.estimate_from_constraints({k: np.concatenate([a[k], c[k]]) for k in a})
        assert est.status == "declined"
        assert est.reason.startswith("ambiguity") and ";" not in est.reason
        assert est.ambiguity > pt.TiltParams().max_ambiguity

    def test_gate_reasons_example(self):
        """The docstring example of gate_reasons."""
        assert pt.gate_reasons(40, [5, 6, 0, 4, 3, 0, 0, 2], 0.05, 0.06, 0.92) == ["ambiguity 0.92 > 0.8"]
        assert pt.gate_reasons(40, [5, 6, 0, 4, 3, 0, 0, 2], float("nan"), 0.06, 0.5) != []

    def test_too_few_segments(self):
        rng = np.random.default_rng(0)
        est = pt.estimate_from_constraints(_vertical_constraints(0, 0, [10.0], rng))
        assert est.status == "declined" and est.reason == "too few segments"

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        cons = _vertical_constraints(-2.0, 1.0, rng.uniform(-180, 180, 50), rng, noise_deg=0.2)
        a, b = pt.estimate_from_constraints(cons), pt.estimate_from_constraints(cons)
        assert a.to_dict() == b.to_dict()


class TestScript:
    def test_id_list_matches_the_records(self):
        """The committed validation list is exactly the records' posed GSV panos, roles by hash."""
        want = te.ids_bytes(te.validation_ids(str(REPO / "benchmark")))
        assert (OUT / "validation_ids.tsv").read_bytes() == want

    def test_roles(self):
        ids = te.read_ids(OUT / "validation_ids.tsv")
        assert all(role == te.role_of(pid) for _, pid, role in ids)
        assert len(ids) == 571 and sum(r == "dev" for *_, r in ids) == 144

    def test_lyon_groups_match_the_bundle(self):
        """group_of reproduces the audit group the bundle's records carry (benchmark_stratum)."""
        recs = te.load_records(REPO / "benchmark/lyon/records.jsonl")
        gm = te.load_json(REPO / "benchmark/lyon/group_map.json")
        assert all(te.group_of(r["pano"], gm) == r["benchmark_stratum"] for r in recs)

    def test_lyon_metadata_pose_sources(self):
        recs = te.load_records(REPO / "benchmark/lyon/records.jsonl")
        src = [te.metadata_pose(r["pano"])[0] for r in recs]
        assert {s: src.count(s) for s in set(src)} == {"absent": 125, "zero": 63, "panoflat": 11, "other": 1}

    @pytest.mark.parametrize("rows, summary, fn", [
        ("validation_gsv.json", "validation_summary.json", "validation_summary"),
        ("lyon.json", "lyon_summary.json", "lyon_summary"),
    ])
    def test_summaries_rederive_from_committed_rows(self, rows, summary, fn):
        obj = json.loads((OUT / rows).read_text(encoding="utf-8"))
        assert (OUT / summary).read_bytes() == te.dumps(getattr(te, fn)(obj)).encode("utf-8")

    @pytest.mark.parametrize("rows", ["validation_gsv.json", "lyon.json"])
    def test_gate_reproduces_every_committed_status(self, rows):
        """Re-applying the gate (TiltParams defaults) to the stored fields gives every row's
        stored status and the same failed tests, so the gate in the code is the gate that produced
        the rows. The reason's printed values can differ in the last digit (the stored ambiguity is
        rounded to 4 places before it is re-printed to 2), so the tests' names are compared."""
        def keys(reason):
            return [part.split(" ")[0] for part in reason.split("; ")] if reason else []

        obj = json.loads((OUT / rows).read_text(encoding="utf-8"))
        assert obj["params"] == pt.TiltParams().__dict__
        got = [(s, keys(why)) for s, why in map(te.regate_row, obj["rows"])]
        assert got == [(r["status"], keys(r["reason"])) for r in obj["rows"]]

    def test_gate_sensitivity_rederives_and_matches_the_summary(self):
        val = json.loads((OUT / "validation_gsv.json").read_text(encoding="utf-8"))
        lyon = json.loads((OUT / "lyon.json").read_text(encoding="utf-8"))
        sens = te.gate_sensitivity(val, lyon)
        assert (OUT / "gate_sensitivity.json").read_bytes() == te.dumps(sens).encode("utf-8")
        summ = te.validation_summary(val)
        at = sens["by_threshold"][f"{pt.TiltParams().max_ambiguity:.2f}"]
        for role in ("dev", "test"):
            acc = summ["by_role"][role]["accepted"]
            assert at[role]["n_ok"] == summ["by_role"][role]["n_ok"]
            assert at[role]["mae_pitch_deg"] == acc["mae_pitch_deg"]
            assert at[role]["n_worst_above_3deg"] == 0
        assert at["lyon"]["n_ok"] == 193
        assert sens["se_term"]["n_above_max_se_gsv"] == 0 and sens["se_term"]["n_above_max_se_lyon"] == 0

    def test_committed_rows_are_lf_and_carry_the_method(self):
        for name in ("validation_gsv.json", "lyon.json"):
            data = (OUT / name).read_bytes()
            assert b"\r\n" not in data
            obj = json.loads(data)
            assert all(r["method_version"] == obj["method_version"] for r in obj["rows"])
            assert all(len(r["image_sha256"]) == 64 and r["manifest_match"] for r in obj["rows"])

    def test_lyon_rows_join_no_verdicts(self):
        """The Lyon output is a covariate only: no verdict or score field rides along."""
        obj = json.loads((OUT / "lyon.json").read_text(encoding="utf-8"))
        assert len(obj["rows"]) == 200
        keys = set().union(*(r.keys() for r in obj["rows"]))
        assert not any(k.startswith(("verdict", "tp", "fp", "precision", "recall")) for k in keys)
