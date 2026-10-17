"""Tests for the UChicago showcase frame hashes (scripts/talks/uchicago_2026_showcase.py).

CPU only, no network, no checkpoint, no benchmark panoramas: two synthetic panoramas and
heatmaps stand in for the manifest's. What is pinned is the replication contract: the caption
font is the same file on every OS, a render is deterministic, its frames.json hashes are what
`verify` checks, and one changed pixel fails it.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "uchicago_2026_showcase", REPO / "scripts/talks/uchicago_2026_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load_script()

ITEMS = [
    {"split": "bend", "pano_id": "synthA", "kind": "all_found", "n_ramps": 1},
    {"split": "clovis", "pano_id": "synthB", "kind": "true_negative", "n_ramps": 0},
]


@pytest.fixture
def showcase(tmp_path, monkeypatch):
    """A two-item showcase folder and pano root under tmp_path, wired into the script."""
    show = tmp_path / "showcase"
    show.mkdir()
    (show / "manifest.json").write_text(json.dumps({"items": ITEMS}), encoding="utf-8")
    rng = np.random.default_rng(0)
    for it in ITEMS:
        panos = tmp_path / "bench" / it["split"] / "panos"
        panos.mkdir(parents=True)
        pix = rng.integers(0, 256, (256, 512, 3), dtype=np.uint8)
        Image.fromarray(pix).save(panos / f"{it['pano_id']}.jpg", quality=90)
        heat = np.zeros((512, 1024), dtype=np.uint8)
        if it["n_ramps"]:
            heat[270:280, 500:510] = 230             # one peak inside the kept band
        Image.fromarray(heat).save(show / f"heat_{it['split']}_{it['pano_id']}.png")
    monkeypatch.setattr(sc, "SHOW_DIR", str(show))
    monkeypatch.setattr(sc, "MANIFEST", str(show / "manifest.json"))
    return tmp_path


def _render(root, name):
    out = root / name
    sc.main(["render", "--pano-root", str(root / "bench"), "--frames-dir", str(out)])
    return out


def test_caption_font_is_pinned_dejavu():
    # The old name lookup gave Segoe UI on Windows and Arial on macOS.
    assert sc._font(30).getname()[0] == "DejaVu Sans"


def test_render_is_deterministic_and_recorded(showcase):
    a, b = _render(showcase, "a"), _render(showcase, "b")
    ra = json.loads((a / sc.FRAMES_RECORD).read_text(encoding="utf-8"))
    rb = json.loads((b / sc.FRAMES_RECORD).read_text(encoding="utf-8"))
    assert [f["pixels_sha256"] for f in ra["frames"]] == [f["pixels_sha256"] for f in rb["frames"]]
    assert [(f["file"], f["split"], f["pano_id"]) for f in ra["frames"]] == [
        ("frame_000.png", "bend", "synthA"), ("frame_001.png", "clovis", "synthB")]
    assert ra["cut"] == {"order": "manifest", "map": False, "per_city": None, "caption": "bottom"}
    assert ra["panels_from"] is False
    # The hash is of the pixels, so it re-derives from the PNG on disk.
    with Image.open(a / "frame_000.png") as img:
        assert sc.pixels_sha256(img) == ra["frames"][0]["pixels_sha256"]


def test_verify_passes_on_a_rerender_and_fails_on_one_pixel(showcase, capsys):
    a, b = _render(showcase, "a"), _render(showcase, "b")
    record = str(a / sc.FRAMES_RECORD)
    sc.main(["verify", "--frames-dir", str(b), "--against", record])
    assert "all 2 frames match" in capsys.readouterr().out

    path = b / "frame_001.png"
    img = Image.open(path).convert("RGB")
    img.putpixel((0, 0), tuple((v + 1) % 256 for v in img.getpixel((0, 0))))
    img.save(path)
    with pytest.raises(SystemExit):
        sc.main(["verify", "--frames-dir", str(b), "--against", record])
    assert "frame_001.png (clovis synthB): pixels differ" in capsys.readouterr().out


def test_verify_reports_missing_frames(showcase, capsys):
    a = _render(showcase, "a")
    (a / "frame_000.png").unlink()
    with pytest.raises(SystemExit):
        sc.main(["verify", "--frames-dir", str(a), "--against", str(a / sc.FRAMES_RECORD)])
    assert "frame_000.png: missing" in capsys.readouterr().out
