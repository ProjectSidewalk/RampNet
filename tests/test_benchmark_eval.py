"""Guard on the committed YOLO benchmark tables (scripts/model_comparison/yolo_baseline/benchmark_eval/).

Sibling of ``test_scoreboard.py::test_every_number_matches_model_comparison``, for the one
committed artifact that test does not cover. The YOLO pano arms are listed in
``docs/model_comparison.md`` as "published, not in these tables", so when the #140 seam
wrap changed what the scorer returns, that document was regenerated and re-derived in CI
while ``benchmark_eval/`` kept its pre-fix values for a month (#140 merged 2026-08-18;
regenerated 2026-09-19) and nothing failed (#148).

Everything here is CPU-only and reads committed inputs alone -- the benchmark bundles,
``benchmark/model_detections/``, ``manual_labels/`` -- through the same regenerator that
writes the files, so the check is "does the committed text equal what the current scorer
produces", not a transcription of the numbers. The whole module runs in about 5 s: 2.4 s to
re-derive all ten splits once (shared fixture), the rest for the negative control and the
scoreboard cross-check.
"""
import difflib
import inspect
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, os.path.join(REPO, "scripts", "model_comparison", "yolo_baseline"))

import rescore_benchmark_eval as rbe  # noqa: E402
import scoreboard as sb  # noqa: E402
from rampnet.detection_eval import score_pano  # noqa: E402

EVAL_DIR = rbe.BENCHMARK_EVAL
README = os.path.join(REPO, "scripts", "model_comparison", "yolo_baseline", "README.md")

# One committed headline row: model, P, CI, R, CI, F1, AP, tp/fp/fn/ign.
_HEADLINE_ROW = re.compile(
    r"^(\w+)\s+([\d.]+)\s+\([\d.]+-[\d.]+\)\s+([\d.]+)\s+\([\d.]+-[\d.]+\)\s+"
    r"([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)/(\d+)/(\d+)$")


def _read(path):
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n").decode("utf-8")


def _committed(split):
    return _read(os.path.join(EVAL_DIR, f"{split}.txt"))


def _headline_rows(text):
    """``{arm: (P, R, F1, AP, tp, fp, fn)}`` from a benchmark_eval table."""
    out = {}
    for line in text.split("\n"):
        m = _HEADLINE_ROW.match(line)
        if m:
            arm, P, R, F1, AP, tp, fp, fn, _ign = m.groups()
            out[arm] = (float(P), float(R), float(F1), float(AP), int(tp), int(fp), int(fn))
    return out


@pytest.fixture(scope="module")
def rendered():
    """Every split re-derived once with the current scorer; ~2.4 s, shared by the module."""
    fp = rbe.scorer_fingerprint()
    out = {}
    for split in rbe.SPLITS:
        text, rows, sweeps = rbe.render_split(split, wrap_x=True, fingerprint=fp, head=None)
        out[split] = {"text": text, "rows": rows, "sweeps": sweeps}
    out["__summary__"] = rbe.render_summary(
        {s: (out[s]["rows"], out[s]["sweeps"]) for s in rbe.SPLITS},
        wrap_x=True, fingerprint=fp, head=None)
    return out


def _assert_same_text(rel, have, want):
    have, want = rbe.strip_head_pointer(have), rbe.strip_head_pointer(want)
    if have == want:
        return
    diff = [l for l in difflib.unified_diff(have.split("\n"), want.split("\n"),
                                            "committed", "current scorer", lineterm="", n=0)
            if not l.startswith("@@")]
    pytest.fail(f"{rel} is stale against the current scorer. Re-run\n"
                f"    python {rbe.RELATIVE_SCRIPT}\n"
                f"and commit the result. First differences:\n" + "\n".join(diff[:40]))


# --------------------------------------------------------------------------- #
# the committed files equal what the current scorer produces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", rbe.SPLITS)
def test_split_table_matches_the_current_scorer(rendered, split):
    """The whole file, not three cells: headline table, every sweep row, the RampNet
    cross-check. Only the ``at repo HEAD`` pointer is ignored, because it is the one
    token that cannot be known until the regeneration is committed."""
    _assert_same_text(f"benchmark_eval/{split}.txt", _committed(split), rendered[split]["text"])


@pytest.mark.parametrize("split", rbe.SPLITS)
def test_pr_curves_match_the_current_scorer(rendered, split):
    """The JSON curves are the AP's provenance; they have to move with it."""
    for arm, report in rendered[split]["rows"].items():
        path = os.path.join(EVAL_DIR, f"pr_{split}", f"pr_{arm}.json")
        assert os.path.exists(path), f"pr_{split}/pr_{arm}.json is missing"
        assert _read(path) == rbe.pr_curve_json(arm, report), \
            f"pr_{split}/pr_{arm}.json is stale; re-run {rbe.RELATIVE_SCRIPT}"


def test_summary_tables_match_the_current_scorer(rendered):
    _assert_same_text("benchmark_eval/SUMMARY_TABLES.md",
                      _read(os.path.join(EVAL_DIR, "SUMMARY_TABLES.md")),
                      rendered["__summary__"])


def test_every_split_and_arm_was_scored(rendered):
    """A split file with an arm missing would still "match" if the regenerator also
    skipped it. Pin the coverage: three arms on every one of the ten splits."""
    for split in rbe.SPLITS:
        assert tuple(rendered[split]["rows"]) == rbe.ARMS, split
        assert tuple(_headline_rows(_committed(split))) == rbe.ARMS, split


# --------------------------------------------------------------------------- #
# the scorer stamp
# --------------------------------------------------------------------------- #
def test_every_file_names_the_scorer_that_produced_it():
    """The #148 ask: a committed number carries its scorer version. The stamp is the
    sha256 of the four source files every number in the file depends on (matcher, pano
    scorer, validation for the CIs and cross-check) plus the ``wrap_x`` setting; any edit
    to those files, behavioural or not, changes it and forces a regeneration."""
    want = rbe.scorer_line(True, rbe.scorer_fingerprint())
    for name in [f"{s}.txt" for s in rbe.SPLITS] + ["SUMMARY_TABLES.md"]:
        lines = _read(os.path.join(EVAL_DIR, name)).split("\n")
        stamps = [l for l in lines if l.startswith("Scorer:")]
        assert stamps == [want], f"{name}: scorer stamp {stamps} != {want!r}"


def test_the_stamp_records_what_was_passed_and_the_scorer_default_is_true():
    """The stamp echoes the ``wrap_x`` the regenerator passed to ``score_pano`` -- it does
    not consult the function's default. So the committed stamps saying ``wrap_x=True``
    describe the current default only while that default IS True; this pins it, so a
    change to the default fails here and prompts the regenerator (and the stamp wording)
    to follow."""
    assert inspect.signature(score_pano).parameters["wrap_x"].default is True
    assert "wrap_x=True" in rbe.scorer_line(True)
    assert "wrap_x=False" in rbe.scorer_line(False)


def test_fingerprint_is_line_ending_independent(tmp_path):
    """CI runs on Linux and Jon's checkout is autocrlf=input; the stamp must agree."""
    for rel in rbe.SCORER_SOURCES:
        src = os.path.join(REPO, *rel.split("/"))
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as fh:
            dst.write_bytes(fh.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert rbe.scorer_fingerprint(str(tmp_path)) == rbe.scorer_fingerprint()


# --------------------------------------------------------------------------- #
# the negative control: the guard detects the drift #148 describes
# --------------------------------------------------------------------------- #
def test_the_pre_140_scorer_reproduces_the_three_cells_that_moved(rendered):
    """Scored with ``wrap_x=False`` the regenerator returns the pre-#140 values, and with
    the default it returns the corrected ones. This is the test that would have failed
    on 2026-08-19, and it pins the issue's table so the record cannot be quietly
    re-explained later. The manual_gold move is one detection crossing the seam and
    becoming a TP instead of an FP plus a miss; the two AP moves leave counts unchanged."""
    expected = {
        # split, arm, metric index in _headline_rows, pre-#140, current
        ("manual_gold", "y26_pano", 0): (0.739, 0.740),   # precision
        ("clovis", "y26_pano", 3): (0.593, 0.596),        # AP
        ("richmond", "y26_pano", 3): (0.536, 0.537),      # AP
    }
    fp = rbe.scorer_fingerprint()
    for (split, arm, i), (old, new) in expected.items():
        now = _headline_rows(rendered[split]["text"])[arm]
        assert now[i] == new, f"{split}/{arm} metric {i}: current scorer gives {now[i]}"
        before_text, *_ = rbe.render_split(split, wrap_x=False, fingerprint=fp, head=None)
        before = _headline_rows(before_text)[arm]
        assert before[i] == old, f"{split}/{arm} metric {i}: wrap_x=False gives {before[i]}"
        if split == "manual_gold":
            assert before[4:] == (2895, 1021, 1024) and now[4:] == (2896, 1020, 1023)
        else:
            assert before[4:] == now[4:], "an AP-only move leaves tp/fp/fn unchanged"


# --------------------------------------------------------------------------- #
# cross-document consistency
# --------------------------------------------------------------------------- #
def test_benchmark_eval_agrees_with_the_scoreboard(rendered):
    """Two committed documents score the same detections; a difference is a bug, not a
    choice. The scoreboard reads the YOLO arms at the same pre-registered 0.25."""
    board = sb.build(models=list(rbe.ARMS))
    for split in rbe.SPLITS:
        for arm, (P, R, F1, AP, tp, fp, fn) in _headline_rows(_committed(split)).items():
            cell = board["per_split"][arm][split]
            assert (cell["tp"], cell["fp"], cell["fn"]) == (tp, fp, fn), (arm, split)
            for name, want, have in (("P", P, cell["precision"]), ("R", R, cell["recall"]),
                                     ("F1", F1, cell["f1"]), ("AP", AP, cell["ap_bundle"])):
                assert abs(want - have) < 0.0006, f"{arm}/{split} {name}: {want} vs {have:.4f}"


def _readme_table(heading_prefix):
    """Body rows of the first markdown table at/after a README line, as lists of cells.
    ``heading_prefix`` may be a section heading or the table's own header row."""
    lines = _read(README).split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(heading_prefix))
    if heading_prefix.startswith("|"):
        start -= 1  # the header row is the anchor; keep it so it is the one dropped below
    rows, in_table = [], False
    for line in lines[start + 1:]:
        if line.startswith("|"):
            in_table = True
            cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
            if not set(cells[0]) <= set("-: "):
                rows.append(cells)
        elif in_table:
            break
    return rows[1:]  # drop the header row


def test_readme_headline_table_matches(rendered):
    """The README's hand-copied F1 table (y11l / y26 / y11x columns) against the files."""
    rows = _readme_table("### Headline: F1 at the pre-registered conf 0.25")
    assert len(rows) == len(rbe.SPLITS)
    for split, *cells in rows:
        have = rendered[split]["rows"]
        for arm, want in zip(rbe.ARMS, cells[:3]):
            assert f"{have[arm].f1:.3f}" == want, f"README {split}/{arm} F1 {want}"


def test_readme_best_sweep_table_matches(rendered):
    """The README's best-sweep table quotes y11x and y11l at their best threshold."""
    rows = _readme_table("| split | y11x best sweep F1 (thr) |")
    assert len(rows) == len(rbe.SPLITS)
    for split, y11x, delta, y11l in rows:
        sweeps = rendered[split]["sweeps"]
        for arm, want in (("y11x_pano_h200", y11x), ("y11l_pano", y11l)):
            t, r = max(sweeps[arm], key=lambda tr: tr[1].f1)
            assert want == f"{r.f1:.3f} ({t:.2f})", f"README {split}/{arm}: {want}"
        # The README's delta column is the difference of the two PRINTED figures, so
        # round first: richmond is 0.777 - 0.547 = +0.230, while the exact gap is +0.229.
        headline = round(rendered[split]["rows"]["y11x_pano_h200"].f1, 3)
        best = round(max(r.f1 for _, r in sweeps["y11x_pano_h200"]), 3)
        assert delta == f"{best - headline:+.3f}", f"README {split} delta {delta}"


# --------------------------------------------------------------------------- #
# the as-run record is not regenerated, and says what it says
# --------------------------------------------------------------------------- #
def test_the_raw_run_record_is_the_2026_08_14_sweep():
    """driver.log / env.txt / run_yolo_pano_eval.sh are the GPU pass that produced the
    detections; the regenerator never writes them. Pin what they attest."""
    log = _read(os.path.join(EVAL_DIR, "driver.log"))
    assert log.rstrip().endswith("ALL_CITY_BUNDLES_DONE")
    assert set(re.findall(r"^=== (\w+) exit=0 2026-08-14T", log, re.M)) == \
        set(rbe.SPLITS) - {"manual_gold"}
    env = _read(os.path.join(EVAL_DIR, "env.txt")).strip()
    assert env == "torch 2.13.0+cu130 cuda True ultralytics 8.4.120"
