"""Guards for the #48 monocular-depth and semantic arms (review of #210, B6, B7, nits).

CPU only, committed files only: the depth model pins agree with what the committed runs
recorded, and the semantic arms refuse label maps that are stale, foreign or carry the
suppressed Curb Cut class.
"""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "analysis"))

import crossview_align_48 as H  # noqa: E402


def test_depth_pins_are_the_revisions_the_committed_runs_recorded():
    import crossview_depth_48 as D
    for name, spec in D.MODELS.items():
        path = os.path.join(H.OUT, "depth", f"{name}.meta.json")
        with open(path, encoding="utf-8") as f:
            prov = json.load(f)["prov"]
        assert spec["commit"] == prov["code_commit"], name
        assert spec["hf_revision"] == prov["hf_revision"], name


def test_seg_dir_ignores_extra_entries_without_equals():
    from crossview_arms import semantic
    ctx = SimpleNamespace(args=SimpleNamespace(extra=["bare", "seg_dir=/x", "k=v"]))
    assert semantic._seg_dir(ctx) == "/x"
    with pytest.raises(SystemExit):
        semantic._seg_dir(SimpleNamespace(args=SimpleNamespace(extra=["bare"])))


def _seg_setup(tmp_path, lab, manifest_sha=None, name="p001_oth", named=True):
    """A seg_dir holding one map and its own manifest.json. ``named``: pass that manifest
    with --extra seg_manifest= (a new segmentation); otherwise the committed one applies."""
    cv2 = pytest.importorskip("cv2")
    from crossview_arms import semantic
    d = tmp_path / "seg"
    d.mkdir()
    p = d / f"{name}.png"
    cv2.imwrite(str(p), lab)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(
        {"maps": {name: {"sha256": manifest_sha or sha, "curb_cut_px_suppressed": 0}}}))
    extra = [f"seg_dir={d}"] + ([f"seg_manifest={d / 'manifest.json'}"] if named else [])
    ctx = SimpleNamespace(args=SimpleNamespace(extra=extra), cache={})
    return semantic, ctx


def test_label_map_reads_a_map_that_matches_the_named_manifest(tmp_path):
    lab = np.full((8, 8), 15, np.uint8)
    semantic, ctx = _seg_setup(tmp_path, lab)
    assert (semantic.label_map({"pair_id": "p001"}, "oth", ctx) == 15).all()


def test_label_map_refuses_a_stale_map(tmp_path):
    semantic, ctx = _seg_setup(tmp_path, np.full((8, 8), 15, np.uint8), manifest_sha="0" * 64)
    with pytest.raises(SystemExit, match="stale or foreign"):
        semantic.label_map({"pair_id": "p001"}, "oth", ctx)


def test_label_map_checks_the_committed_manifest_not_seg_dirs_own(tmp_path):
    """Final re-review of #210 (N3): a foreign map for a real view name, shipped with its own
    seg_dir/manifest.json, was accepted. Without --extra seg_manifest= the committed
    manifest applies, and it does not know this map."""
    semantic, ctx = _seg_setup(tmp_path, np.full((8, 8), 15, np.uint8), name="p000_oth",
                               named=False)
    with pytest.raises(SystemExit, match="semantic_seg_manifest.json"):
        semantic.label_map({"pair_id": "p000"}, "oth", ctx)
    assert semantic._seg_manifest(ctx)[0] == semantic.COMMITTED_SEG_MANIFEST


def test_label_map_names_a_regenerated_map_as_regenerated(tmp_path):
    """A map label_map computed on a miss is not in any manifest; the next run must say
    so, not call it stale or foreign."""
    cv2 = pytest.importorskip("cv2")
    from crossview_arms import semantic
    d = tmp_path / "seg"
    lab = np.full((8, 8), 15, np.uint8)
    ctx = SimpleNamespace(args=SimpleNamespace(extra=[f"seg_dir={d}"]),
                          cache={"segmenter": lambda views: [(lab, 0)]},
                          view=lambda pair, which: None)
    assert (semantic.label_map({"pair_id": "p000"}, "oth", ctx) == 15).all()
    assert ctx.cache["seg_regenerated"] == 1
    assert (d / semantic.REGENERATED).exists()
    later = SimpleNamespace(args=ctx.args, cache={})
    with pytest.raises(SystemExit, match="regenerated this map"):
        semantic.label_map({"pair_id": "p000"}, "oth", later)
    cv2.imwrite(str(d / "p000_oth.png"), lab + 1)             # then overwritten by hand
    with pytest.raises(SystemExit, match="stale or foreign"):
        semantic.label_map({"pair_id": "p000"}, "oth", later)


def test_label_map_refuses_the_curb_cut_class(tmp_path):
    """An explicit raise, not an assert (python -O strips asserts)."""
    from crossview_arms import semantic
    lab = np.full((8, 8), 15, np.uint8)
    lab[2:4, 2:4] = semantic.CURB_CUT
    semantic, ctx = _seg_setup(tmp_path, lab)
    with pytest.raises(SystemExit, match="Curb Cut"):
        semantic.label_map({"pair_id": "p001"}, "oth", ctx)
    import inspect
    assert "assert " not in inspect.getsource(semantic.label_map)


def test_committed_seg_manifest_covers_every_view():
    from crossview_arms import semantic
    with open(semantic.COMMITTED_SEG_MANIFEST, encoding="utf-8") as f:
        maps = json.load(f)["maps"]
    ids = {p["pair_id"] for p in H.read_frozen_pairs()}
    assert set(maps) == {f"{i}_{w}" for i in ids for w in ("src", "oth")}
