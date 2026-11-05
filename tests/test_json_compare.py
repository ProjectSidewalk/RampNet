"""rampnet.json_compare: exact bytes when they match, a float-only tolerance when not (#251)."""
import json

import pytest

from rampnet.json_compare import assert_json_equal_within, compare_json


def _w(tmp_path, name, obj, newline="\n"):
    p = tmp_path / name
    p.write_bytes((json.dumps(obj, indent=2, sort_keys=True) + "\n").replace("\n", newline)
                  .encode("utf-8"))
    return p


BASE = {"n": 3, "label": "run a", "se": [0.1234567890123, 2.5], "flag": True, "none": None}


def test_equal_files_are_byte_identical(tmp_path):
    a, b = _w(tmp_path, "a.json", BASE), _w(tmp_path, "b.json", BASE)
    c = assert_json_equal_within(a, b)
    assert c.status == "identical" and c.summary() == "byte-identical"
    assert compare_json(a, b, exact=True).ok


def test_last_digit_float_drift_is_within_tolerance(tmp_path):
    drift = json.loads(json.dumps(BASE))
    drift["se"][0] = BASE["se"][0] + 6.66e-16          # the macOS size of difference
    a, b = _w(tmp_path, "a.json", drift), _w(tmp_path, "b.json", BASE)
    assert a.read_bytes() != b.read_bytes()
    c = assert_json_equal_within(a, b, rel=1e-12)
    assert c.status == "within"
    assert c.n_float_diffs == 1 and c.first_key == "$.se[0]"
    assert c.max_abs == pytest.approx(6.66e-16, rel=0.2)
    assert "within rel 1e-12" in c.summary() and "$.se[0]" in c.summary()


def test_float_beyond_tolerance_fails_and_names_the_key(tmp_path):
    moved = json.loads(json.dumps(BASE))
    moved["se"][1] = 2.5 * (1 + 1e-9)
    a, b = _w(tmp_path, "a.json", moved), _w(tmp_path, "b.json", BASE)
    c = compare_json(a, b)
    assert c.status == "differs" and c.n_beyond == 1 and c.first_beyond_key == "$.se[1]"
    with pytest.raises(AssertionError, match=r"\$\.se\[1\]"):
        assert_json_equal_within(a, b)
    assert compare_json(a, b, rel=1e-6).status == "within"


@pytest.mark.parametrize("change, key", [
    (lambda o: o.update(n=4), "$.n"),                    # ints are counts: exact
    (lambda o: o.update(n=3.0), "$.n"),                  # 3 vs 3.0 is a type change
    (lambda o: o.update(label="run b"), "$.label"),
    (lambda o: o.pop("flag"), "$.flag"),
    (lambda o: o["se"].append(1.0), "$.se[2]"),
])
def test_structural_differences_are_never_tolerated(tmp_path, change, key):
    other = json.loads(json.dumps(BASE))
    change(other)
    a, b = _w(tmp_path, "a.json", other), _w(tmp_path, "b.json", BASE)
    c = compare_json(a, b, rel=1.0)                       # even with a huge tolerance
    assert c.status == "differs" and c.first_beyond_key == key
    assert c.n_structural >= 1


def test_crlf_vs_lf_is_caught_when_exactness_is_requested(tmp_path):
    a = _w(tmp_path, "a.json", BASE, newline="\r\n")
    b = _w(tmp_path, "b.json", BASE)
    assert compare_json(a, b, exact=True).status == "differs"
    with pytest.raises(AssertionError):
        assert_json_equal_within(a, b, exact=True)


def test_formatting_differences_fail_even_in_tolerance_mode(tmp_path):
    """Only number spellings may differ: line endings and indentation are still exact."""
    b = _w(tmp_path, "b.json", BASE)
    crlf = _w(tmp_path, "crlf.json", BASE, newline="\r\n")
    c = compare_json(crlf, b)
    assert c.status == "differs" and c.formatting_differs and c.n_structural == 0
    compact = tmp_path / "compact.json"
    compact.write_bytes((json.dumps(BASE, sort_keys=True) + "\n").encode("utf-8"))
    assert compare_json(compact, b).formatting_differs


def test_nan_equals_nan_and_infinities_are_exact():
    assert compare_json(b'{"x": NaN}\n', b'{"x": NaN}\n').status == "identical"
    assert compare_json(b'[NaN, 1.0]\n', b'[NaN, 1.0000000000000002]\n').status == "within"
    assert compare_json(b'[Infinity]\n', b'[1e308]\n', rel=1.0).status == "differs"


def test_abs_tol_defaults_to_zero_so_a_near_zero_drift_fails():
    """Pins the #251 choice: no absolute tolerance by default. 0.0 vs 1e-17 has infinite
    relative difference, so it "differs" at the default and is "within" only when a caller
    opts into abs_tol. docs/data/benchmark_power_135.json holds 45 float leaves that are
    exactly 0.0 (of 2,827 floats; the farfield result holds none of its 44); a platform that
    wrote ~1e-17 for one of them would still fail its check, deliberately."""
    import inspect
    assert inspect.signature(compare_json).parameters["abs_tol"].default == 0.0
    assert inspect.signature(assert_json_equal_within).parameters["abs_tol"].default == 0.0
    assert compare_json(b"[0.0]\n", b"[1e-17]\n").status == "differs"
    assert compare_json(b"[0.0]\n", b"[1e-17]\n", abs_tol=1e-15).status == "within"
