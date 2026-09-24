"""`compare_silent_activation.py`: the three-outcome comparison #131 asks for.

Pure CPU; builds small payloads in the shape `silent_activation.build_payload` writes.
The last test compares the committed file with itself, which must read as outcome 1.
"""
import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "analysis"))

import compare_silent_activation as cs  # noqa: E402

RESULT_JSON = os.path.join(REPO_ROOT, "analysis_out", "silent_activation.json")


def _row(city, pano, x, act, null_p95=0.5, off=10.0):
    return {"city": city, "pano": pano, "x": x, "y": 0.55, "field": "far", "dist_m": 30.0,
            "px": 20.0, "group": "rated", "verdict": "visible", "act": act, "null_pct": 0.8,
            "null_med": 0.003, "null_p95": null_p95, "above_own_null_p95": act > null_p95,
            "argmax_off_px": off, "act_at_site": act / 2, "nearest_peak_px": 40.0,
            "nearest_peak_score": 0.6, "seam": False}


def _payload(rows):
    return {"threshold": 0.3, "null_trials": 200, "null_seed": 20260731, "n": len(rows),
            "cities": ["a", "b"], "panos": len({(r["city"], r["pano"]) for r in rows}),
            "skipped_no_imagery": 0, "model": "projectsidewalk/rampnet-model", "tta": False,
            "results": rows}


def _write(tmp_path, name, payload, indent=2):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=indent)
    return str(p)


BASE = _payload([_row("a", "p1", 0.1, 0.005), _row("a", "p2", 0.2, 0.03),
                 _row("b", "p3", 0.3, 0.4, null_p95=0.3)])


def test_the_same_bytes_are_outcome_1(tmp_path, capsys):
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", BASE)
    assert cs.main([a, b]) == cs.BYTE_IDENTICAL
    assert "OUTCOME 1" in capsys.readouterr().out


def test_the_same_values_in_different_bytes_are_outcome_2(tmp_path, capsys):
    a = _write(tmp_path, "a.json", BASE, indent=2)
    b = _write(tmp_path, "b.json", BASE, indent=1)
    assert cs.main([a, b]) == cs.VALUES_IDENTICAL
    assert "OUTCOME 2" in capsys.readouterr().out


def test_a_moved_value_is_outcome_3_with_the_field_named(tmp_path, capsys):
    moved = copy.deepcopy(BASE)
    moved["results"][1]["act"] = 0.03001          # same class, tiny move
    moved["results"][2]["null_pct"] = 0.9
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", moved)
    assert cs.main([a, b]) == cs.VALUES_MOVED
    out = capsys.readouterr().out
    assert "OUTCOME 3" in out
    assert "no row changed class" in out
    cmp = cs.compare(BASE, moved)
    assert cmp["fields"]["act"]["n_diff"] == 1
    assert abs(cmp["fields"]["act"]["max_abs"] - 1e-5) < 1e-9
    assert cmp["fields"]["null_pct"]["n_diff"] == 1
    assert "null_med" not in cmp["fields"]


def test_a_class_change_and_a_p95_flip_are_called_out(tmp_path, capsys):
    moved = copy.deepcopy(BASE)
    moved["results"][0]["act"] = 0.02             # absent -> faint_local
    moved["results"][2]["above_own_null_p95"] = False
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", moved)
    assert cs.main([a, b]) == cs.VALUES_MOVED
    out = capsys.readouterr().out
    assert "class change a/p1: absent -> faint_local" in out
    assert "1 rows flipped" in out


def test_rows_are_joined_by_key_not_position(tmp_path):
    shuffled = copy.deepcopy(BASE)
    shuffled["results"] = shuffled["results"][::-1]
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", shuffled)
    assert cs.main([a, b]) == cs.VALUES_IDENTICAL


def test_a_different_population_is_not_the_same_study(tmp_path, capsys):
    fewer = copy.deepcopy(BASE)
    fewer["results"] = fewer["results"][:2]
    fewer["n"] = 2
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", fewer)
    assert cs.main([a, b]) == cs.POPULATIONS_DIFFER
    out = capsys.readouterr().out
    assert "1 rows only in the reference" in out
    assert "populations differ" in out


def test_a_changed_input_field_is_reported_as_inputs_not_measurement(tmp_path):
    changed = copy.deepcopy(BASE)
    changed["results"][0]["verdict"] = "not_visible"
    cmp = cs.compare(BASE, changed)
    assert cmp["status"] == cs.VALUES_MOVED
    assert "verdict" in cmp["inputs"] and not cmp["fields"]


def test_the_committed_result_compared_with_itself_is_outcome_1():
    assert cs.main([RESULT_JSON, RESULT_JSON]) == cs.BYTE_IDENTICAL


def test_json_out_records_the_status(tmp_path):
    a = _write(tmp_path, "a.json", BASE, indent=2)
    b = _write(tmp_path, "b.json", BASE, indent=1)
    out = tmp_path / "cmp.json"
    cs.main([a, b, "--json-out", str(out)])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == cs.VALUES_IDENTICAL
    assert rec["reference"]["sha256"] != rec["replica"]["sha256"]
