"""scripts/cluster_review_gallery.py (issue #224): unit selection and seeding, the rendered
page (placeholders filled, the script parses under node), the prefill guard, and the
viewer's state bootstrap -- the one JS path that can destroy review work -- run under
node as tests/test_box_gallery.py does.

    pytest tests/test_cluster_review_gallery.py -v
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from cluster_review_gallery import (  # noqa: E402
    STATE_BOOTSTRAP_JS, build_html, load_prefill, main, role_problem, select_units,
    unit_seed_arm, viewer_unit)

SHA = "f" * 64
SNAP = {"schema": "rampnet.cluster_review.snapshot/1", "city": "testville",
        "labels": {"sha256": SHA}, "aerial": {"attribution": "Imagery: test"}}


def corner(cid, pilot=False, double=False, b_seed=None, keys=("1", "2", "3")):
    return {"corner_id": cid, "city": "testville", "type": "residential",
            "centre": {"lat": 45.6, "lng": -122.6}, "window_m": 30.0, "has_labels": bool(keys),
            "pilot": pilot, "double_rate": double, "rater_b_seed": b_seed,
            "aerial": {"file": f"aerial/{cid.replace(':', '_')}.jpg", "px": 512, "zoom": 20,
                       "world_px": {"x0": 0, "y0": 0, "x1": 671, "y1": 671}},
            "inventory": [],
            "labels": [{"key": k, "pano_id": f"P{k}", "user_kind": "ai", "lat": 45.6,
                        "lng": -122.6, "camera": {"lat": 45.6, "lng": -122.6001,
                                                  "heading_deg": 0, "source": "run"},
                        "capture_date": "2024-01", "crop": f"crops/{k}.jpg",
                        "seed_group": {"deployed": "d1" if k != "3" else "d2",
                                       "fusion": "f9" if k != "3" else None}}
                       for k in keys]}


CORNERS = [corner("t:res:000001", pilot=True, double=True, b_seed="fusion"),
           corner("t:res:000002", pilot=True, double=True, b_seed="deployed"),
           corner("t:res:000003"),
           corner("t:res:000004", double=True, b_seed="fusion", keys=())]


def test_seed_arm_resolution_and_selection():
    assert unit_seed_arm(CORNERS[0], "auto", "a") == "deployed"
    assert unit_seed_arm(CORNERS[0], "auto", "b") == "fusion"
    assert unit_seed_arm(CORNERS[1], "auto", "b") == "deployed"
    assert unit_seed_arm(CORNERS[2], "fusion", "a") == "fusion"
    assert {c["corner_id"] for c in select_units(CORNERS, pilot=True)} == \
        {"t:res:000001", "t:res:000002"}
    assert {c["corner_id"] for c in select_units(CORNERS, role="b")} == \
        {"t:res:000001", "t:res:000002", "t:res:000004"}
    assert len(select_units(CORNERS)) == 4


def test_viewer_unit_paths_are_relative_to_the_gallery():
    v = viewer_unit(CORNERS[0], "fusion", "../")
    assert v["aerial"]["file"] == "../aerial/t_res_000001.jpg"
    assert v["labels"][0]["crop"] == "../crops/1.jpg"
    assert [lab["seed"] for lab in v["labels"]] == ["f9", "f9", None]


def _script_of(html):
    return re.search(r"<script>(.*)</script>", html, re.S).group(1)


def test_build_html_fills_every_placeholder_and_parses():
    units = [viewer_unit(c, "deployed", "../") for c in CORNERS]
    html = build_html(units, SNAP, "testville", "mikey", "b", None, "assignments__mikey.json")
    for ph in ("__UNITS__", "__SNAPSHOT__", "__CITY__", "__RATER__", "__ROLE__",
               "__RUBRIC_V__", "__INITIAL__", "__ATTRIBUTION__", "__FILE_NAME__",
               "__STATE_BOOTSTRAP__"):
        assert ph not in html
    assert json.dumps(SHA) in html and "function bootstrapState" in html
    assert "elapsed_s" in html and "visibilityState" in html
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(_script_of(html))
    try:
        proc = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    assert proc.returncode == 0, proc.stderr


def test_prefill_refuses_another_snapshot(tmp_path):
    a = {"schema": "rampnet.cluster_review/1", "snapshot_sha256": "0" * 64, "corners": {}}
    (tmp_path / "assignments.json").write_text(json.dumps(a), encoding="utf-8")
    initial, msg = load_prefill(tmp_path, None, SNAP)
    assert initial is None and "NOT prefilled" in msg
    a["snapshot_sha256"] = SHA
    (tmp_path / "assignments.json").write_text(json.dumps(a), encoding="utf-8")
    initial, msg = load_prefill(tmp_path, None, SNAP)
    assert initial == a
    assert load_prefill(tmp_path, "mikey", SNAP) == (None, None)


def test_main_writes_the_gallery(tmp_path):
    b = tmp_path / "cluster_review"
    b.mkdir()
    (b / "snapshot.json").write_text(json.dumps(SNAP), encoding="utf-8")
    (b / "corners.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CORNERS),
                                     encoding="utf-8")
    assert main([str(b), "--pilot"]) == 0
    html = (b / "gallery" / "index.html").read_text(encoding="utf-8")
    assert '"../crops/1.jpg"' in html and '"assignments.json"' in html


# --- the state bootstrap, under node --------------------------------------------------

def _boot(tmp_path, initial, local, units, snapshot=SHA):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = tmp_path / "boot.js"
    script.write_text(STATE_BOOTSTRAP_JS + "\nconsole.log(JSON.stringify(bootstrapState(%s, %s, %s, %s)));\n"
                      % (json.dumps(initial), json.dumps(local), json.dumps(units),
                         json.dumps(snapshot)), encoding="utf-8")
    proc = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _units(arm="deployed"):
    return [viewer_unit(c, arm, "../") for c in CORNERS[:2]]


def test_bootstrap_seeds_unseen_units_one_group_per_seed_group(tmp_path):
    out = _boot(tmp_path, None, {}, _units())
    s = out["state"]["t:res:000001"]
    assert s["labels"] == {"1": "r1", "2": "r1", "3": "r2"}
    assert set(s["ramps"]) == {"r1", "r2"} and s["complete"] is False
    out = _boot(tmp_path, None, {}, _units("fusion"))
    assert out["state"]["t:res:000001"]["labels"] == {"1": "r1", "2": "r1", "3": None}


def test_bootstrap_local_wins_per_unit_and_prefill_fills_the_rest(tmp_path):
    initial = {"snapshot_sha256": SHA, "corners": {
        "t:res:000001": {"seed_arm": "deployed", "labels": {"1": "r1", "2": "r2", "3": "r2"},
                         "ramps": {"r1": {"lat": 1, "lng": 2, "placed": True},
                                   "r2": {"lat": 0, "lng": 0, "placed": False}},
                         "uncovered": [], "complete": True, "elapsed_s": 40},
        "t:res:000002": {"seed_arm": "deployed", "labels": {"1": "not_ramp", "2": "r1",
                                                            "3": "r1"},
                         "ramps": {"r1": {"lat": 0, "lng": 0}}, "uncovered": [],
                         "complete": True, "elapsed_s": 12},
        "t:res:000099": {"seed_arm": "fusion", "labels": {}, "ramps": {}, "uncovered": [],
                         "complete": True, "elapsed_s": 5}}}
    local = {"t:res:000001": {"seed_arm": "deployed", "labels": {"1": "r1", "2": "r1", "3": "r1"},
                              "ramps": {"r1": {"placed": False}}, "uncovered": [],
                              "complete": False, "elapsed_s": 3, "note": "", "seen": True}}
    out = _boot(tmp_path, initial, local, _units())
    st = out["state"]
    assert st["t:res:000001"]["labels"] == {"1": "r1", "2": "r1", "3": "r1"}   # local kept
    assert st["t:res:000002"]["labels"]["1"] == "not_ramp"                       # prefilled
    assert st["t:res:000002"]["complete"] is True and st["t:res:000002"]["elapsed_s"] == 12
    assert "t:res:000099" in st                           # not rendered: kept for round trip
    assert out["prefilled"] == 2 and out["reopened"] == 0
    assert out["conflicts"] == ["t:res:000001"]           # local work shadows the file: reported


def test_bootstrap_file_beats_merely_seeded_local_state(tmp_path):
    # Review fix: opening the gallery seeds every unit into localStorage (seen: false); an
    # assignments file placed in the bundle afterwards must still prefill those units, not
    # be silently shadowed by the seed grouping.
    seeded = _boot(tmp_path, None, {}, _units())["state"]
    assert seeded["t:res:000002"]["seen"] is False
    initial = {"snapshot_sha256": SHA, "corners": {
        "t:res:000002": {"seed_arm": "deployed", "labels": {"1": "not_ramp", "2": "r1",
                                                            "3": "r1"},
                         "ramps": {"r1": {"lat": 0, "lng": 0}}, "uncovered": [],
                         "complete": True, "elapsed_s": 12, "inventory_seen": True}}}
    out = _boot(tmp_path, initial, seeded, _units())
    s = out["state"]["t:res:000002"]
    assert s["labels"]["1"] == "not_ramp" and s["complete"] is True
    assert s["inventory_seen"] is True
    assert out["prefilled"] == 1 and out["conflicts"] == []


def test_bootstrap_ignores_a_prefill_from_another_snapshot(tmp_path):
    initial = {"snapshot_sha256": "0" * 64, "corners": {
        "t:res:000001": {"labels": {"1": "not_ramp"}, "ramps": {}, "complete": True}}}
    out = _boot(tmp_path, initial, {}, _units())
    assert out["initialIgnored"] is True and out["prefilled"] == 0
    assert out["state"]["t:res:000001"]["labels"]["1"] == "r1"       # seeded, not the file


def test_bootstrap_reopens_a_unit_whose_labels_changed(tmp_path):
    local = {"t:res:000001": {"seed_arm": "deployed",
                              "labels": {"1": "r1", "2": "r1", "3": "r2", "gone": "r2"},
                              "ramps": {"r1": {"placed": False}, "r2": {"placed": False}},
                              "uncovered": [], "complete": True, "elapsed_s": 50, "note": "",
                              "seen": True},
             "t:res:000002": {"seed_arm": "deployed", "labels": {"1": "r1", "2": "r1"},
                              "ramps": {"r1": {"placed": False}}, "uncovered": [],
                              "complete": True, "elapsed_s": 8, "note": "", "seen": True}}
    out = _boot(tmp_path, None, local, _units())
    s1, s2 = out["state"]["t:res:000001"], out["state"]["t:res:000002"]
    assert "gone" not in s1["labels"] and s1["labels"]["3"] == "r2"   # work kept
    assert s1["complete"] is False and s1["elapsed_s"] == 50
    assert s2["labels"]["3"] is None and s2["complete"] is False       # new label: unassigned
    assert out["reopened"] == 2


def test_role_b_needs_a_rater_and_roles_do_not_cross(tmp_path):
    assert "--rater" in role_problem(tmp_path, None, "b")
    assert role_problem(tmp_path, None, "a") is None
    assert role_problem(tmp_path, "mikey", "b") is None
    (tmp_path / "assignments__mikey.json").write_text(json.dumps({"role": "b"}), encoding="utf-8")
    assert role_problem(tmp_path, "mikey", "b") is None
    assert "role b" in role_problem(tmp_path, "mikey", "a")
    (tmp_path / "assignments.json").write_text(json.dumps({"role": "a"}), encoding="utf-8")
    assert role_problem(tmp_path, None, "a") is None


def test_main_refuses_role_b_without_rater(tmp_path):
    b = tmp_path / "cluster_review"
    b.mkdir()
    (b / "snapshot.json").write_text(json.dumps(SNAP), encoding="utf-8")
    (b / "corners.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CORNERS),
                                     encoding="utf-8")
    with pytest.raises(SystemExit, match="--rater"):
        main([str(b), "--role", "b"])
    assert main([str(b), "--role", "b", "--rater", "mikey"]) == 0
    html = (b / "gallery" / "index.html").read_text(encoding="utf-8")
    assert "__CORNERS_SHA__" not in html
    import hashlib
    assert hashlib.sha256((b / "corners.jsonl").read_bytes()).hexdigest() in html


# --- review fixes on PR #280 (Amendment 5) ----------------------------------------------

def _bundle(tmp_path, corners=None):
    b = tmp_path / "cluster_review"
    b.mkdir()
    (b / "snapshot.json").write_text(json.dumps(SNAP), encoding="utf-8")
    (b / "corners.jsonl").write_text("".join(json.dumps(c) + "\n" for c in corners or CORNERS),
                                     encoding="utf-8")
    return b


def _units_in(html):
    return json.loads(re.search(r"const UNITS = (.*?);\n", html).group(1))


def test_inventory_never_reaches_a_retest_and_waits_for_it_on_double_rated_units(tmp_path):
    # S2: the first pass's reveal would teach the retest where the ramps are
    inv = [{"lat": 45.6, "lng": -122.6, "unit_id": "CR1"}]
    cs = [dict(c, inventory=inv) for c in CORNERS]
    from cluster_review_gallery import inventory_revealed, retest_done_units
    assert inventory_revealed(cs[0], "b", set()) is False
    assert inventory_revealed(cs[0], "a", set()) is False                  # double-rated
    assert inventory_revealed(cs[0], "a", {"t:res:000001"}) is True        # retest exported
    assert inventory_revealed(cs[2], "a", set()) is True                   # not double-rated
    b = _bundle(tmp_path, cs)
    assert main([str(b), "--role", "b", "--rater", "jon-retest"]) == 0
    units = _units_in((b / "gallery" / "index.html").read_text(encoding="utf-8"))
    assert all(u["inventory"] == [] for u in units) and all(u["inventory_withheld"] for u in units)
    assert main([str(b)]) == 0
    units = {u["id"]: u for u in _units_in((b / "gallery" / "index.html").read_text(encoding="utf-8"))}
    assert units["t:res:000001"]["inventory"] == [] and units["t:res:000003"]["inventory"] == inv
    retest = {"schema": "rampnet.cluster_review/1", "snapshot_sha256": SHA, "role": "b",
              "rater": "jon-retest",
              "corners": {"t:res:000001": {"labels": {}, "complete": True}}}
    (b / "assignments__jon-retest.json").write_text(json.dumps(retest), encoding="utf-8")
    assert retest_done_units({"x": retest}) == {"t:res:000001"}
    assert main([str(b)]) == 0
    units = {u["id"]: u for u in _units_in((b / "gallery" / "index.html").read_text(encoding="utf-8"))}
    assert units["t:res:000001"]["inventory"] == inv
    assert units["t:res:000002"]["inventory"] == []                        # its retest: not yet


def test_out_on_another_drive_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    # N3: os.path.relpath raises across Windows drives; refuse cleanly, create nothing
    import cluster_review_gallery as crg
    b = _bundle(tmp_path)

    def other_drive(*a, **k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")
    monkeypatch.setattr(crg.os.path, "relpath", other_drive)
    out = tmp_path / "elsewhere"
    with pytest.raises(SystemExit, match="another drive"):
        main([str(b), "--out", str(out)])
    assert not out.exists()


def _node(tmp_path, js):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = tmp_path / "t.js"
    script.write_text(STATE_BOOTSTRAP_JS + "\n" + js, encoding="utf-8")
    proc = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_esc_escapes_every_markup_character(tmp_path):
    # N8: a can't-judge reason goes into a title attribute and the badge text
    out = _node(tmp_path, "console.log(JSON.stringify(esc(%s)));"
                % json.dumps('trees & "deck" <b>\'x\'</b>'))
    assert out == "trees &amp; &quot;deck&quot; &lt;b&gt;&#39;x&#39;&lt;/b&gt;"


def test_stale_stores_find_unexported_work_of_an_earlier_bundle(tmp_path):
    # N5: a re-seed changes the corners.jsonl sha256 and so the storage key
    pre = "clusterreview:testville:jon:" + SHA + ":"
    store = {pre + "old": json.dumps({"u1": {"seen": True}, "u2": {"seen": False}}),
             pre + "old:idx": "3",
             pre + "new": json.dumps({"u1": {"seen": True}}),
             pre + "empty": json.dumps({"u1": {"seen": False}}),
             "clusterreview:testville:A:" + SHA + ":old": json.dumps({"u1": {"complete": True}})}
    js = ("const S = %s; console.log(JSON.stringify(staleStores(Object.keys(S), k => S[k], "
          "'testville', 'jon', %s, 'new')));" % (json.dumps(store), json.dumps(SHA)))
    assert _node(tmp_path, js) == [{"key": pre + "old", "units": 1}]


def test_prefill_keeps_the_unit_times(tmp_path):
    # S3: opened_at / completed_at survive a prefill round trip, and the export writes them
    f = {"labels": {"1": "r1"}, "ramps": {"r1": {}}, "complete": True,
         "opened_at": "2026-11-01T09:00:00.000Z", "completed_at": "2026-11-01T09:05:00.000Z"}
    out = _node(tmp_path, "console.log(JSON.stringify(fromFile(%s)));" % json.dumps(f))
    assert out["opened_at"] == f["opened_at"] and out["completed_at"] == f["completed_at"]
    html = build_html([viewer_unit(c, "deployed", "../") for c in CORNERS], SNAP, "testville",
                      None, "a", None, "assignments.json")
    assert "completed_at: s.completed_at || null" in html and "s.completed_at = new Date()" in html


def test_page_basics_for_assistive_tech():
    # N2: page language, keyboard-reachable crops with alt text, group number off-colour
    html = build_html([viewer_unit(c, "deployed", "../") for c in CORNERS], SNAP, "testville",
                      None, "a", None, "assignments.json")
    assert '<html lang="en">' in html
    assert 'tabindex="0" role="button"' in html and "alt=\"label ' + lab.key" in html
    assert "<title>' + groupName(u, g) + ' ramp point" in html
    assert "<title>label ' + lab.key + ': ' + groupName(u, v)" in html
