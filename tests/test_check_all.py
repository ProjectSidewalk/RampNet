"""scripts/check_all.py: the registry is well-formed and every entry's argv is accepted.

None of the real checks run here (the gate itself does that, in the ``checks`` CI job).
Every entry's argv is parsed by the script's own argparse in ONE subprocess, with
``ArgumentParser.parse_args`` wrapped to stop right after a successful parse, so a renamed
flag or subcommand fails this test instead of failing the gate at merge time.
"""
import contextlib
import importlib.util
import io
import glob
import hashlib
import json
import re
import shutil
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "scripts", "check_all.py")


def _load():
    spec = importlib.util.spec_from_file_location("check_all", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_all"] = mod      # dataclasses resolves the module by name
    spec.loader.exec_module(mod)
    return mod


ca = _load()


def test_registry_well_formed():
    names = [e.name for e in ca.REGISTRY]
    assert len(names) == len(set(names)), "duplicate entry names"
    for e in ca.REGISTRY:
        assert e.steps and all(len(argv) >= 1 for argv in e.steps), e.name
        assert e.verifies.strip(), e.name
        assert set(e.requires) <= set(ca.KNOWN_REQUIREMENTS), (e.name, e.requires)
        if e.requires:
            assert e.note, f"{e.name}: an entry with requirements says what unblocks it"


@pytest.mark.parametrize("entry", ca.REGISTRY, ids=lambda e: e.name)
def test_every_script_exists(entry):
    for script in entry.scripts:
        assert os.path.isfile(os.path.join(REPO, script)), script


# Runs each argv through the script's real argparse, in one interpreter, stopping right
# after parse_args returns. A script that does work before parse_args (imports only, in
# this repo) pays that once; nothing past the parse runs.
_PROBE = r"""
import argparse, json, os, runpy, sys
REPO, cases = sys.argv[1], json.loads(sys.argv[2])
class Parsed(BaseException):
    pass
_orig = argparse.ArgumentParser.parse_args
def _parse(self, args=None, namespace=None):
    _orig(self, args, namespace)
    raise Parsed()
argparse.ArgumentParser.parse_args = _parse
results = []
sys.path.insert(0, REPO)
for argv in cases:
    script = os.path.join(REPO, argv[0])
    sys.argv = [script, *argv[1:]]
    sys.path.insert(0, os.path.dirname(script))
    try:
        runpy.run_path(script, run_name="__main__")
        verdict = "no parse_args call"
    except Parsed:
        verdict = "ok"
    except SystemExit as e:
        verdict = f"argparse rejected it (exit {e.code})"
    except Exception as e:
        verdict = f"{type(e).__name__}: {e}"
    finally:
        sys.path.pop(0)
    results.append([argv, verdict])
print("@@RESULTS@@" + json.dumps(results))
"""


def test_every_argv_is_accepted_by_its_script():
    cases = [ca.expand(argv, "check_all_tmp", "check_all_local_root")
             for e in ca.REGISTRY for argv in e.steps]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    p = subprocess.run([sys.executable, "-c", _PROBE, REPO, json.dumps(cases)], cwd=REPO,
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    out = p.stdout.decode("utf-8", "replace")
    assert "@@RESULTS@@" in out, p.stderr.decode("utf-8", "replace")[-3000:]
    results = json.loads(out.split("@@RESULTS@@", 1)[1])
    bad = [(argv, v) for argv, v in results if v != "ok"]
    assert not bad, "\n".join(f"{' '.join(a)}: {v}" for a, v in bad) + \
        "\n" + p.stderr.decode("utf-8", "replace")[-3000:]
    assert len(results) == len(cases)


def test_ci_excludes_entries_with_requirements_and_slow_ones():
    every = set(ca.KNOWN_REQUIREMENTS)
    for e in ca.REGISTRY:
        r = ca.skip_reason(e, ci=True, run_all=False, allowed=every)
        if e.requires:
            # --ci ignores --allow: nothing it runs may need more than a clean clone
            assert r and "requires" in r, e.name
        elif e.slow:
            assert r and "slow" in r, e.name
        else:
            assert r is None, e.name
        if e.requires:
            assert ca.skip_reason(e, ci=False, run_all=True, allowed=set()) is not None
            # allowed: no longer skipped for the requirement (needs_paths may still skip it)
            r2 = ca.skip_reason(e, ci=False, run_all=True, allowed=set(e.requires))
            assert r2 is None or not r2.startswith("requires"), (e.name, r2)
        if e.slow and not e.requires:
            assert ca.skip_reason(e, ci=True, run_all=True, allowed=set()) is None
    assert any(ca.skip_reason(e, ci=True, run_all=False, allowed=set()) is None
               for e in ca.REGISTRY), "--ci would run nothing"


def test_list_names_every_entry():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert ca.main(["--list"]) == 0
    listed = buf.getvalue()
    for e in ca.REGISTRY:
        assert e.name in listed, e.name
        for argv in e.steps:
            assert " ".join(argv) in listed, argv


def test_unknown_only_is_a_usage_error():
    with pytest.raises(SystemExit) as ex, contextlib.redirect_stderr(io.StringIO()):
        ca.main(["--only", "no_such_entry"])
    assert ex.value.code == 2


# --------------------------------------------------------------------------- #
# pins / --touching (review S4)
# --------------------------------------------------------------------------- #
def glob_repo(pin):
    return glob.glob(os.path.join(REPO, pin.rstrip("/")))


@pytest.mark.parametrize("entry", ca.REGISTRY, ids=lambda e: e.name)
def test_every_pin_exists(entry):
    """A pin that names nothing is a stale registry; --touching would silently miss it."""
    assert entry.pins, f"{entry.name}: every entry lists what it pins"
    for pin in entry.pins:
        assert "\\" not in pin and not pin.startswith("/"), pin
        assert glob_repo(pin), f"{entry.name}: pin {pin!r} matches nothing in the repo"


def _names(path):
    return {e.name for e in ca.touching(path)}


def test_touching_finds_the_cross_pin():
    hit = _names("analysis_out/recall_by_depth_112.json")
    # the owner AND the dependents: the owner alone passes a change the dependent fails
    assert {"recall_by_depth_112", "da3_calibration_101", "laurens_paired_151",
            "manifests_sha256"} <= hit


def test_touching_prefix_and_glob_rules():
    assert "cascade_cost_35" in _names("analysis_out/op_cache/richmond.json")    # under a pinned dir
    assert "cascade_cost_35" in _names("analysis_out/cascade_cost_35")           # the dir itself
    assert "scoreboard" in _names("benchmark/richmond/verdicts.json")             # glob pin
    assert "scoreboard" in _names("benchmark/richmond")                           # dir holding a glob pin
    assert "da3_calibration_101" in _names("analysis_out")                        # dir holding a pin
    assert "da3_calibration_101" in _names(os.path.join(REPO, "analysis_out", "da3_calibration_101",
                                                        "tables.json"))           # absolute path
    assert "two_scale_197" in _names("scripts/analysis/two_scale_197.py")         # its own script
    assert _names("README.md") == set()
    assert not ca.path_matches("analysis_out/op_cache_tta/x.json", "analysis_out/op_cache/")


def test_list_shows_pins_and_touching_selects():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert ca.main(["--touching", "analysis_out/recall_by_depth_112.json", "--list"]) == 0
    out = buf.getvalue()
    assert "da3_calibration_101  [" in out and "pins: " in out
    assert "scoreboard  [" not in out


# --------------------------------------------------------------------------- #
# local-data preconditions (S1)
# --------------------------------------------------------------------------- #
def test_local_cache_entries_have_a_presence_precondition():
    for e in ca.REGISTRY:
        if "local-cache" in e.requires:
            assert e.needs_paths and e.absent_reason, e.name


def test_needs_paths_absent_is_a_skip_naming_what_is_absent(tmp_path):
    e = ca.by_name()["imagery_manifest_verify"]
    r = ca.skip_reason(e, ci=False, run_all=True, allowed={"local-cache"}, local_root=str(tmp_path))
    assert r and r.startswith("panos absent"), r
    (tmp_path / "benchmark" / "richmond" / "panos").mkdir(parents=True)
    assert ca.skip_reason(e, ci=False, run_all=True, allowed={"local-cache"},
                          local_root=str(tmp_path)) is None


ABSENT_OUT = ("               split  panos             digest  status\n"
              "           annapolis      -   0123456789abcdef  imagery absent locally (expected 125 panos)\n"
              "                bend      -   0123456789abcdef  imagery absent locally (expected 110 panos)\n"
              "\nEvery reviewed panorama matches the bytes recorded at review time.\n")


def test_imagery_all_absent_output_is_nothing_verified():
    assert ca.imagery_nothing_verified(ABSENT_OUT)
    one_ok = ABSENT_OUT.replace(
        "                bend      -   0123456789abcdef  imagery absent locally (expected 110 panos)",
        "                bend    110   0123456789abcdef  OK")
    assert ca.imagery_nothing_verified(one_ok) is None


# --------------------------------------------------------------------------- #
# the runner end to end on a scratch git repo with a fake registry (S1, S2, N1)
# --------------------------------------------------------------------------- #
def _git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


WRITER = r'''
import sys
mode, target = sys.argv[1], sys.argv[2]
if mode == "append":
    with open(target, "a", encoding="utf-8") as f:
        f.write("more\n")
elif mode == "absent":
    print("     annapolis      -   0123456789abcdef  imagery absent locally (expected 1 panos)")
'''


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    (repo / "writer.py").write_text(WRITER, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    monkeypatch.setattr(ca, "REPO", str(repo))
    return repo


def _run(monkeypatch, entries, argv):
    monkeypatch.setattr(ca, "REGISTRY", entries)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ca.main(argv)
    return code, buf.getvalue()


def test_rewrite_of_an_already_modified_file_fails(scratch_repo, monkeypatch):
    (scratch_repo / "a.txt").write_text("one\nlocal edit\n", encoding="utf-8")   # tree is dirty
    entries = [ca.E("w_tracked", ("writer.py", "append", "a.txt"), "x", pins=("a.txt",)),
               ca.E("w_tmp", ("writer.py", "append", "{tmp}/scratch.txt"), "x", pins=("a.txt",))]
    code, out = _run(monkeypatch, entries, [])
    assert code == 1, out
    assert "modified the working tree: M a.txt" in out, out
    assert re.search(r"w_tmp\s+PASS", out), out


def test_new_untracked_file_fails(scratch_repo, monkeypatch):
    entries = [ca.E("w_new", ("writer.py", "append", "stray.txt"), "x", pins=("a.txt",))]
    code, out = _run(monkeypatch, entries, [])
    assert code == 1 and "A stray.txt" in out, out


def test_nothing_verified_output_fails(scratch_repo, monkeypatch):
    entries = [ca.E("w_absent", ("writer.py", "absent", "-"), "x", pins=("a.txt",),
                    nothing_verified=ca.imagery_nothing_verified)]
    code, out = _run(monkeypatch, entries, [])
    assert code == 1 and "nothing verified" in out, out


def test_only_a_skipped_entry_exits_nonzero(scratch_repo, monkeypatch):
    entries = [ca.E("w_slow", ("writer.py", "noop", "-"), "x", pins=("a.txt",), slow=True)]
    code, out = _run(monkeypatch, entries, ["--only", "w_slow"])
    assert code == 3 and "slow; runs only with --all" in out, out
    code, out = _run(monkeypatch, entries, ["--only", "w_slow", "--all"])
    assert code == 0, out
    code, out = _run(monkeypatch, entries, [])            # not named: a skip is not an error
    assert code == 0, out


# --------------------------------------------------------------------------- #
# manifests_sha256 (S3)
# --------------------------------------------------------------------------- #
def _load_vs():
    spec = importlib.util.spec_from_file_location(
        "verify_sha256sums", os.path.join(REPO, "scripts", "verify_sha256sums.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = _load_vs()


def test_manifest_discovery_covers_the_unowned_manifests():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    found = set(vs.manifests())
    for m in ("docs/data/rampnet1_stage1_run/SHA256SUMS", "docs/data/rampnet1_stage2_run/SHA256SUMS",
              "stage_two/cosine_rung_135_events/SHA256SUMS", "stage_two/run_a_84_events/SHA256SUMS",
              "docs/data/seed_variance_51_135/SHA256SUMS"):
        assert m in found, m
    pins = ca.by_name()["manifests_sha256"].pins
    for m in found:     # --touching a manifest must find the entry that checks it
        assert any(ca.path_matches(m, p) for p in pins), f"{m} is not covered by manifests_sha256 pins"
    in_ci = {e.name for e in ca.REGISTRY if ca.skip_reason(e, ci=True, run_all=False, allowed=set()) is None}
    assert "manifests_sha256" in in_ci


def test_manifest_mismatch_absent_and_eol(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "good.txt").write_bytes(b"x\n")
    (d / "crlf.txt").write_bytes(b"y\r\n")
    (d / "bad.txt").write_bytes(b"changed\n")

    def h(b):
        return hashlib.sha256(b).hexdigest()

    (d / "SHA256SUMS").write_text(
        f"{h(b'x' + bytes([10]))}  good.txt\n{h(b'y' + bytes([10]))} *crlf.txt\n"
        f"{h(b'orig' + bytes([10]))}  bad.txt\n{h(b'z')}  gone.txt\n", encoding="utf-8")
    rows, problems = vs.verify(str(tmp_path), only=["m/SHA256SUMS"])
    assert rows == [{"manifest": "m/SHA256SUMS", "ok": 1, "eol": 1, "absent": 1, "bad": 1}]
    assert any("bad.txt" in q for q in problems)
    assert any("partly present" in q for q in problems)    # gone.txt, while others exist
    assert len(problems) == 2


def test_manifest_partly_present_fails_but_all_absent_is_a_counted_skip(tmp_path, monkeypatch):
    """Review S5: deleting one committed file listed in a manifest must fail the run."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    src = os.path.join(REPO, "stage_two", "run_a_84_events")
    dst = tmp_path / "stage_two" / "run_a_84_events"
    shutil.copytree(src, dst)
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    (cluster / "only_on_the_cluster.sha256").write_text("0" * 64 + "  /gscratch/x/best.pth\n",
                                                        encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    monkeypatch.setattr(vs, "REPO", str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        assert vs.main([]) == 0                     # complete + all-absent: passes
    listed = [n for _, n in vs.parse(str(dst / "SHA256SUMS"))]
    assert len(listed) >= 2
    os.remove(dst / listed[0])                      # one committed file deleted
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert vs.main([]) == 1
    assert "partly present" in buf.getvalue()
    rows, _ = vs.verify(str(tmp_path), only=["cluster/only_on_the_cluster.sha256"])
    assert rows[0]["absent"] == 1 and rows[0]["ok"] == 0


def test_touching_that_selects_nothing_exits_3():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert ca.main(["--touching", "README.md", "--list"]) == 3
    assert "selects no entry" in buf.getvalue()


def test_recorded_argv_drops_machine_local_paths():
    rec = ca.recorded_argv(["--all", "--json", os.path.join(REPO, "analysis_out", "check_all", "latest.json"),
                            "--local-root", os.path.dirname(REPO), "--json=" + os.path.dirname(REPO) + "/x.json"])
    assert rec == ["--all", "--json", "analysis_out/check_all/latest.json", "--local-root",
                   "<main checkout>", "--json=<outside the repo>"]
