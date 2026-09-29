"""mined_label_check_158 (#158 step 4): the fixed sampling rule, the instrument draw, the
verdict-file integrity checks and the precision read."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import mined_label_check_158 as M  # noqa: E402


def _lab(site, pano, band, emit=True, bench=False, x=0.5, y=0.6):
    return {"site_id": site, "pano_id": pano, "band": band, "emit": emit,
            "benchmark_pano": bench, "x": x, "y": y, "range_m": 5.0, "peak_conf": 0.3,
            "src_pano": "s" + pano, "src_x": 0.4, "src_y": 0.6, "capture_date": "",
            "src_capture_date": ""}


def test_allocation_is_proportional_and_sums_to_n():
    assert M.allocate({"0-8 m": 50, "8-12 m": 30, "12-15 m": 20}, 10) == \
        {"0-8 m": 5, "8-12 m": 3, "12-15 m": 2}
    a = M.allocate({"0-8 m": 101, "8-12 m": 202, "12-15 m": 303}, 100)
    assert sum(a.values()) == 100
    assert M.allocate({"0-8 m": 3, "8-12 m": 2, "12-15 m": 0}, 100) == \
        {"0-8 m": 3, "8-12 m": 2, "12-15 m": 0}


def test_sample_excludes_benchmark_panos_and_is_seeded():
    labels = [_lab(i, f"p{i}", M.BANDS[i % 3]) for i in range(300)]
    labels += [_lab(1000 + i, f"b{i}", "0-8 m", bench=True) for i in range(50)]
    labels += [_lab(2000 + i, f"n{i}", "0-8 m", emit=False) for i in range(50)]
    s1, alloc, pop = M.draw_sample(labels)
    s2, _, _ = M.draw_sample(list(reversed(labels)))
    assert [r["pano_id"] for r in s1] == [r["pano_id"] for r in s2]   # order-free, seeded
    assert len(s1) == 100 and pop == {b: 100 for b in M.BANDS}
    assert not any(r["benchmark_pano"] or not r["emit"] for r in s1)


def test_instrument_draw_uses_known_answers_and_checks_reproduction():
    labels = [_lab(1, "a", "0-8 m", bench=True), _lab(2, "b", "8-12 m", bench=True),
              _lab(3, "c", "0-8 m", bench=True)]
    step3 = [{"city": "richmond", "site_id": s, "pano_id": p, "emit": True, "x": 0.5, "y": 0.6}
             for s, p in ((1, "a"), (2, "b"), (3, "c"))]
    cands = [{"site_id": "1", "pano_id": "a", "bucket": "tp"},
             {"site_id": "2", "pano_id": "b", "bucket": "fp"},
             {"site_id": "3", "pano_id": "c", "bucket": "unsure"}]
    got, pool = M.draw_instrument(labels, step3, cands)
    assert pool == 2
    assert {(k, a) for k, a, _ in got} == {((1, "a"), "yes"), ((2, "b"), "no")}
    labels[0]["x"] = 0.7                                    # the city-wide pixel moved
    with pytest.raises(SystemExit, match="does not reproduce"):
        M.draw_instrument(labels, step3, cands)


def test_rule_reading_and_precision():
    assert M.rule_reading(0.9, 0.82, 0.96) == "build"
    assert "not decisive" in M.rule_reading(0.7, 0.45, 0.85)
    v = {"a": {"answer": "yes"}, "b": {"answer": "no"}, "c": {"answer": "cant_tell"}}
    p = M.precision(v, ["a", "b", "c", "d"])
    assert (p["yes"], p["no"], p["cant_tell"], p["unanswered"], p["precision"]) == \
        (1, 1, 1, 1, 0.5)


def _ref():
    cards = {"c1": {"instrument": False, "band": "0-8 m", "known_answer": None},
             "c2": {"instrument": True, "band": "8-12 m", "known_answer": "no"},
             "c3": {"instrument": False, "band": "12-15 m", "known_answer": None}}
    return {"manifest_digest": "d" * 16, "items": ["c1", "c2", "c3"], "cards": cards}


def test_verdict_file_integrity_and_rates(tmp_path):
    ref = _ref()
    d = M.empty_verdicts(ref["items"], ref["manifest_digest"], "jonf")
    d["verdicts"] = {"c1": {"answer": "yes"}, "c2": {"answer": "yes"}, "c3": {"answer": "no"}}
    path = tmp_path / "mined_label_check__jonf.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    got = M.rates(M.load_verdicts(str(path), ref), ref)
    assert got["pooled"]["n_items"] == 2 and got["pooled"]["precision"] == 0.5
    assert got["instrument"]["agree_with_earlier_verdicts"] == 0      # said yes, known no
    for bad in ({"manifest_digest": "e" * 16}, {"items": ["c1"]}, {"question": "other"}):
        path.write_text(json.dumps({**d, **bad}), encoding="utf-8")
        with pytest.raises(ValueError):
            M.load_verdicts(str(path), ref)
    wrong_name = tmp_path / "mined_label_check__other.json"
    wrong_name.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="file name"):
        M.load_verdicts(str(wrong_name), ref)


def test_page_has_no_instrument_or_confidence_leak():
    cards = [{"id": "c0000aaaa"}]
    crops = [{"ramp_uid": "c0000aaaa", "city": "richmond", "pano_id": "p1", "x": 0.5,
              "y": 0.6, "ring": True, "capture_date": ""}]
    h = M.render_gallery(cards, crops, "0" * 16)
    for word in ("instrument", "known_answer", "peak_conf", "residual", "item_class"):
        assert word not in h
    assert "mined_label_check__" in h and M.QUESTION in h
