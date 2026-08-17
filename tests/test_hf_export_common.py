"""Guards for the shared exporter helpers in scripts/hf_export_common.py.

These stamp *published* artifacts, which is why they were worth consolidating: six copies of
`git_commit` meant a fix landed in five of them and one card on the Hub quietly disagreed.

`clear_build_dir` is the one with teeth. Shards are named by position (`train-00000.parquet`) and
both card rendering and `upload_folder` walk the directory rather than a manifest, so a rebuild
producing fewer shards than last time left orphans that were counted into the card totals and then
published -- duplicating rows in the released split, with every orphan still matching its own
recorded sha256, so no integrity check could see it.
"""
import hashlib
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from hf_export_common import (  # noqa: E402
    clear_build_dir, git_commit, hf_features_metadata, hf_value, sha256_bytes, sha256_file)


def test_sha256_file_matches_a_one_shot_hash_across_chunk_boundaries(tmp_path):
    blob = os.urandom(300_000)
    path = tmp_path / "blob.bin"
    path.write_bytes(blob)
    assert sha256_file(path, chunk=4096) == hashlib.sha256(blob).hexdigest()
    assert sha256_file(path) == sha256_bytes(blob)


def test_git_commit_is_unknown_outside_a_checkout_rather_than_raising(tmp_path):
    # Provenance is best-effort: an sdist or tarball must not fail a 13 GB build.
    assert git_commit(tmp_path) == "unknown"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo)] + list(args), check=True, capture_output=True)


def test_git_commit_marks_a_dirty_working_tree(tmp_path):
    """`git rev-parse HEAD` alone says which commit was checked out, not whether the sources
    matched it -- so a card built from edited files claimed clean provenance."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "first")

    clean = git_commit(repo)
    assert clean != "unknown" and not clean.endswith("-dirty")

    (repo / "a.txt").write_text("two", encoding="utf-8")
    dirty = git_commit(repo)
    assert dirty == clean + "-dirty"


def test_clear_build_dir_removes_a_previous_builds_shards(tmp_path):
    split = tmp_path / "data" / "train"
    split.mkdir(parents=True)
    for i in range(3):
        (split / "train-{:05d}.parquet".format(i)).write_bytes(b"stale")
    clear_build_dir(tmp_path, "data/train")
    assert not split.exists()


def test_clear_build_dir_leaves_the_card_and_sibling_configs_alone(tmp_path):
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    (tmp_path / "data" / "native").mkdir(parents=True)
    (tmp_path / "data" / "native" / "bend.parquet").write_bytes(b"keep")
    (tmp_path / "data" / "records").mkdir(parents=True)
    (tmp_path / "data" / "records" / "bend.parquet").write_bytes(b"drop")

    clear_build_dir(tmp_path, "data/records")

    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "data" / "native" / "bend.parquet").is_file()
    assert not (tmp_path / "data" / "records").exists()


def test_clear_build_dir_is_a_no_op_on_a_first_build(tmp_path):
    clear_build_dir(tmp_path, "data/train")          # must not raise


def test_hf_feature_metadata_is_the_key_datasets_reads():
    meta = hf_features_metadata({"pano_id": hf_value("string"), "image": {"_type": "Image"}})
    assert b"huggingface" in meta
    assert b'"_type": "Image"' in meta[b"huggingface"]
