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
REPLICA_JSON = os.path.join(REPO_ROOT, "analysis_out", "silent_activation_replica.json")
REPLICA_CKPT_JSON = os.path.join(REPO_ROOT, "analysis_out", "silent_activation_replica_ckpt.json")


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


def test_the_order_cities_were_passed_in_is_not_a_difference(tmp_path):
    reordered = copy.deepcopy(BASE)
    reordered["cities"] = list(reversed(BASE["cities"]))
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", reordered)
    assert cs.main([a, b]) == cs.VALUES_IDENTICAL


def test_a_different_city_set_is_a_different_study(tmp_path):
    narrower = copy.deepcopy(BASE)
    narrower["cities"] = ["a"]
    a = _write(tmp_path, "a.json", BASE)
    b = _write(tmp_path, "b.json", narrower)
    assert cs.main([a, b]) == cs.POPULATIONS_DIFFER


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


def test_the_committed_klone_replica_moves_no_number_0c_quotes():
    """#131's outcome, pinned: the klone L40S re-run (2026-09-24) is outcome 3 against the
    RTX 3070 original -- fifth-decimal activation noise -- and nothing the 0c tables are
    built from moved: no class change, no null-p95 flip, null_pct identical to 3 places."""
    with open(RESULT_JSON, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(REPLICA_JSON, encoding="utf-8") as fh:
        b = json.load(fh)
    cmp = cs.compare(a, b)
    assert cmp["status"] == cs.VALUES_MOVED
    assert cmp["n_common"] == 128 and not cmp["only_in_a"] and not cmp["only_in_b"]
    assert not cmp["header"]                      # same scope; only --cities' spelling differs
    assert not cmp["inputs"]
    assert set(cmp["fields"]) <= {"act", "null_med", "null_p95", "act_at_site"}
    assert max(d["max_abs"] for d in cmp["fields"].values()) < 1e-4
    assert cmp["class_changes"] == [] and cmp["p95_changes"] == []
    assert cs.class_counts(a) == cs.class_counts(b) == {"absent": 10, "faint_local": 39, "tail": 79}


def test_the_two_klone_replicas_are_byte_identical():
    """Same GPU class on two nodes (the lab allocation, then a ckpt copy): the run is
    deterministic on that hardware, so the drift above is between machines, not runs."""
    with open(REPLICA_JSON, "rb") as fh, open(REPLICA_CKPT_JSON, "rb") as gh:
        assert fh.read() == gh.read()


def _empty():
    return _payload([])


def test_two_empty_files_are_not_a_match(tmp_path):
    """Nothing compared is not 'every value identical': an empty replica is a failed run."""
    assert cs.compare(_empty(), _empty())["status"] == cs.POPULATIONS_DIFFER
    a = _write(tmp_path, "a.json", _empty())
    b = _write(tmp_path, "b.json", _empty())
    assert cs.main([a, b]) == cs.POPULATIONS_DIFFER       # even byte-identical


def test_a_repeated_row_key_is_not_hidden_by_the_join(tmp_path, capsys):
    doubled = copy.deepcopy(BASE)
    doubled["results"].append(copy.deepcopy(doubled["results"][0]))
    cmp = cs.compare(BASE, doubled)
    assert cmp["status"] == cs.POPULATIONS_DIFFER
    assert cmp["duplicates"]["replica"] == [("a", "p1", 0.1, 0.55)]
    assert not cmp["duplicates"]["reference"]
    a = _write(tmp_path, "a.json", doubled)
    assert cs.main([a, a]) == cs.POPULATIONS_DIFFER
    assert "repeats 1 row keys" in capsys.readouterr().out


def _doc_activation_table():
    """The '| population | n | act q1 / med / q3 | act >= 0.01 |' table in 0c, parsed."""
    import re
    # §0c moved verbatim out of curb_ramp_data_sourcing.md under #145.
    doc = os.path.join(REPO_ROOT, "docs", "data_scaling_59.md")
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index("| population | n | act q1 / med / q3 |")
    rows = {}
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        name, n, q, n01 = [c.strip().strip("*") for c in line.strip("|").split("|")]
        rows[name] = (int(n), tuple(float(x) for x in q.split(" / ")), int(n01))
        assert re.fullmatch(r"[\d.]+ / [\d.]+ / [\d.]+", q)
    return rows


def test_the_activation_quartile_table_0c_prints_reads_from_both_files():
    """0c's per-population act quartiles, at the three decimals the doc prints, from the
    committed original and from the klone replica alike. (Until #131's review the
    near / witnessed q1 cell read 0.033; both files give 0.0325 -> 0.032.)"""
    from farfield_forensics import quartiles
    doc = _doc_activation_table()
    assert len(doc) == 6
    for path in (RESULT_JSON, REPLICA_JSON):
        with open(path, encoding="utf-8") as fh:
            results = json.load(fh)["results"]
        got = {}
        for field in ("near", "far"):
            for grp in ("rated", "below_floor", "witnessed"):
                sel = [r["act"] for r in results if r["field"] == field and r["group"] == grp]
                if sel:
                    got[f"{field} / {grp.replace('_', '-')}"] = sel
        got["all silent misses"] = [r["act"] for r in results]
        assert set(got) == set(doc), (sorted(got), sorted(doc))
        for name, acts in got.items():
            n, q, n01 = doc[name]
            assert len(acts) == n, name
            assert tuple(round(x, 3) for x in quartiles(acts)) == q, (path, name)
            assert sum(a >= 0.01 for a in acts) == n01, name


def test_each_run_record_hashes_the_replica_beside_it():
    """The .run.json's result_sha256 is the replica's own bytes. It holds on every clone only
    because .gitattributes marks the replicas -text (an autocrlf=true checkout is CRLF)."""
    import hashlib
    for path in (REPLICA_JSON, REPLICA_CKPT_JSON):
        with open(path, "rb") as fh:
            raw = fh.read()
        with open(path.replace(".json", ".run.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
        assert b"\r\n" not in raw
        assert rec["result_sha256"] == hashlib.sha256(raw).hexdigest()
