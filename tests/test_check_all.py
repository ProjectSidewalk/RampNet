"""scripts/check_all.py: the registry is well-formed and every entry's argv is accepted.

None of the real checks run here (the gate itself does that, in the ``checks`` CI job).
Every entry's argv is parsed by the script's own argparse in ONE subprocess, with
``ArgumentParser.parse_args`` wrapped to stop right after a successful parse, so a renamed
flag or subcommand fails this test instead of failing the gate at merge time.
"""
import contextlib
import importlib.util
import io
import json
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
            assert ca.skip_reason(e, ci=False, run_all=True, allowed=set(e.requires)) is None
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
