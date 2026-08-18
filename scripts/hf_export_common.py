"""Provenance helpers shared by every Hugging Face exporter in this repo.

These three functions were copy-pasted into each exporter as it was written -- six copies of
`git_commit`, four of a chunked sha256, two of the `datasets` feature-dict shorthand. That is
worse than ordinary duplication, because every one of them stamps a **published** artifact: a
fix applied to five of six copies leaves one card quietly disagreeing with the others, on the
Hub, where nobody re-reads it.

The dirty-tree marker is the concrete example. `git rev-parse --short HEAD` alone says which
commit was checked out, not whether the working tree matched it -- so a card built from edited
sources claimed a clean provenance. `git_commit()` now appends `-dirty`, and because there is
one definition, every exporter gained that at once.

Consumers: `export_benchmark.py`, `export_crop_dataset.py`, `export_crop_model.py`,
`export_stage1_inputs.py`, `export_hf_model.py`, `build_street_derivative.py`,
`analysis/gov_provenance.py`, `analysis/imagery_manifest.py`.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHUNK_BYTES = 1 << 20


def git_commit(repo_root=REPO_ROOT):
    """Short HEAD sha, with `-dirty` appended when the working tree has uncommitted changes.

    Provenance is best-effort: outside a git checkout (an sdist, a tarball) this returns
    "unknown" rather than failing a 13 GB build over a missing `.git`.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
    except Exception:                                # noqa: BLE001 - provenance is best-effort
        return "unknown"
    try:
        status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                                capture_output=True, text=True, check=True)
        if status.stdout.strip():
            return commit + "-dirty"
    except Exception:                                # noqa: BLE001 - as above
        pass
    return commit


def sha256_file(path, chunk=CHUNK_BYTES):
    """sha256 of a file, read in chunks -- these run over multi-GB Parquet and checkpoints."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hf_value(dtype):
    """The `datasets` feature-dict shorthand for a scalar column."""
    return {"dtype": dtype, "_type": "Value"}


def hf_features_metadata(features):
    """Parquet key-value metadata `datasets` reads to recover column semantics.

    Without it the viewer shows an opaque {bytes, path} struct instead of an image.
    """
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode()}


def clear_build_dir(out, subdir="data"):
    """Remove a previous build's `data/` tree before writing a new one.

    Shards are named by position (`train-00000.parquet`), and both card rendering and
    `upload_folder` walk the directory rather than a manifest. So a rebuild that produces fewer
    shards than last time leaves orphans behind that are counted into the card totals and then
    published -- duplicating rows in the released split, with every orphan still matching its own
    recorded sha256, so no integrity check catches it.

    Only `subdir` is removed: README.md and anything else staged alongside is left in place.
    """
    target = Path(out) / subdir
    if target.exists():
        shutil.rmtree(str(target))
    return target
