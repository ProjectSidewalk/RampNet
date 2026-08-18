"""Unit tests for the Stage 1 yield measurement (#59, #18).

Two jobs. The parsing helpers are tested on synthetic logs, and the headline numbers in
docs/stage1_generation_cost.md are re-derived from the committed evidence — so if the
rescued logs under docs/data/rampnet1_stage1_run/ are ever truncated, re-encoded, or
silently dropped from the repo (they were: `*.out` in .gitignore had eaten
download_dataset.out), the suite says so instead of the doc quietly ceasing to be
reproducible.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import stage1_yield as sy  # noqa: E402

SCRIPT = os.path.join(REPO, "scripts", "analysis", "stage1_yield.py")
EVIDENCE = os.path.join(REPO, "docs", "data", "rampnet1_stage1_run")


# --------------------------------------------------------------------------- #
# progress.txt — one line per written pano
# --------------------------------------------------------------------------- #
def test_progress_is_a_set_and_counts_repeats(tmp_path):
    p = tmp_path / "progress.txt"
    # Blank lines and non-numeric noise appear in the real log when a job is killed
    # mid-write; a repeat means an index was written twice across restarts.
    p.write_text("0\n1\n1\n\n2\nnot-an-index\n3\n")
    done, dupes = sy.load_progress(p)
    assert done == {0, 1, 2, 3}
    assert dupes == 1


# --------------------------------------------------------------------------- #
# download_dataset.out — the two failure modes it records
# --------------------------------------------------------------------------- #
def test_log_separates_google_refusals_from_the_quota_wall(tmp_path):
    p = tmp_path / "download_dataset.out"
    p.write_text(
        "Failed to fetch panorama for pano_id AAA\n"
        "Failed to fetch panorama for pano_id BBB\n"
        "Failed to fetch panorama for pano_id AAA\n"      # same pano retried
        "Error at line index 7: [Errno 122] Disk quota exceeded\n"
        "Error at line index 9: [Errno 122] Disk quota exceeded\n"
        "[Errno 122] Disk quota exceeded\n"               # no index: unattributable
        "some unrelated line\n"
    )
    failed, quota = sy.load_log(p)
    assert failed == {"AAA", "BBB"}       # unique panos, not occurrences
    assert quota == {7, 9}


def test_log_survives_undecodable_bytes(tmp_path):
    p = tmp_path / "download_dataset.out"
    p.write_bytes(b"Failed to fetch panorama for pano_id AAA\n\xff\xfe garbage\n")
    failed, _ = sy.load_log(p)
    assert failed == {"AAA"}


# --------------------------------------------------------------------------- #
# missing evidence must name itself
# --------------------------------------------------------------------------- #
def test_missing_evidence_names_the_files(tmp_path):
    with pytest.raises(SystemExit) as exc:
        sy.check_evidence(str(tmp_path))
    msg = str(exc.value)
    for name in sy.EVIDENCE:
        assert name in msg
    assert "SHA256SUMS" in msg


def test_cli_refuses_an_empty_evidence_dir(tmp_path):
    r = subprocess.run([sys.executable, SCRIPT, "--evidence-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "missing Stage 1 evidence" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------- #
# the committed evidence still says what the doc says
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(EVIDENCE), reason="Stage 1 evidence not present")
def test_committed_evidence_reproduces_the_published_yield():
    done, dupes = sy.load_progress(os.path.join(EVIDENCE, "progress.txt"))
    failed, quota = sy.load_log(os.path.join(EVIDENCE, "download_dataset.out"))

    assert len(done) == 214_599 and dupes == 0
    # 97.91% of 219,170 intended, and the loss is Google's: 4,570 of 4,571.
    assert len(done) / 219_170 == pytest.approx(0.9791, abs=5e-5)
    assert len(failed) == 4_590
    # The quota wall looked like data loss and was not: every index it hit completed later.
    assert len(quota) == 11_438
    assert not (quota - done)
