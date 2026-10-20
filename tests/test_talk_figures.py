"""The comparison-slide captions in scripts/talks/uchicago_2026_figures.py (#261 review).

The talk was delivered 2026-10-16 and its PNGs are committed as delivered; nothing here
renders a figure. What is tested is the caption logic a regeneration would use: it named
the best zero-shot model on the whole board, so once claude-opus-5-5 (0.71) landed the
subtitle named a model the chart does not show. CPU-only, no matplotlib.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "talks"))

import uchicago_2026_figures as talk  # noqa: E402


def _m(model, provider, cls, f1, complete=True, display=None):
    return {"model": model, "provider": provider, "class": cls, "f1": f1,
            "complete": complete, "display": display or model}


def _board(*extra):
    """The delivered slide's eight bars, plus ``extra`` legs that are not bars."""
    shown = [
        _m("rampnet", "rampnet", "purpose-trained", 0.79),
        _m("y11l_pano", "yolo", "supervised", 0.60),
        _m("gemini-3.1-pro-preview", "gemini", "chat-vlm", 0.575, display="Gemini 3.1 Pro"),
        _m("claude-opus-5-effort-low", "claude", "chat-vlm", 0.568,
           display="Claude Opus 5 (low)"),
        _m("gemini-3.7-flash", "gemini", "chat-vlm", 0.54),
        _m("allenai/Molmo2-8B", "molmo", "pointing", 0.42),
        _m("Qwen/Qwen3-VL-8B-Instruct", "qwen", "chat-vlm", 0.32),
        _m("google/owlv2-large-patch14-ensemble", "owlv2", "open-vocab", 0.065),
    ]
    assert [m["model"] for m in shown] == talk.COMPARISON_MODELS
    return {"models": shown + list(extra)}


def test_the_board_as_delivered_keeps_the_delivered_caption():
    """No unshown leg beats its family's bar, so the caption reads as it did on 2026-10-16."""
    best, unshown = talk.comparison_caption_facts(_board(
        _m("gemini-3.6-flash", "gemini", "chat-vlm", 0.53),
        _m("claude-fable-5-effort-low-anthropic", "claude", "chat-vlm", 0.61, complete=False)))
    assert best["model"] == "gemini-3.1-pro-preview"
    assert unshown == [] and talk._unshown_note(unshown) == ""


def test_the_caption_names_only_a_model_the_chart_shows():
    """The 2026-10-19 board: claude-opus-5-5 is the best zero-shot model overall but not a
    bar. The caption must name the best bar and say what is missing, not name the 0.71."""
    opus55 = _m("claude-opus-5-5-effort-low-anthropic", "claude", "chat-vlm", 0.71,
                display="Claude Opus 5.5 (low, anthropic)")
    best, unshown = talk.comparison_caption_facts(_board(opus55))
    assert best["model"] in talk.COMPARISON_MODELS
    assert best["model"] == "gemini-3.1-pro-preview"
    assert unshown == [opus55]
    assert talk._unshown_note(unshown) == " Not shown: Claude Opus 5.5 (low, anthropic) 0.71."


def test_the_committed_board_caption_is_truthful():
    """Against analysis_out/scoreboard.json as committed: whatever it holds, the named model
    is a bar, and every unshown leg flagged really outranks its provider's bars."""
    board = talk.load_board()
    best, unshown = talk.comparison_caption_facts(board)
    assert best["model"] in talk.COMPARISON_MODELS
    assert best["class"] in talk.ZERO_SHOT_CLASSES
    bars = [m for m in board["models"] if m["model"] in talk.COMPARISON_MODELS]
    for m in unshown:
        assert m["model"] not in talk.COMPARISON_MODELS and m["complete"]
        assert all(m["f1"] > b["f1"] for b in bars if b["provider"] == m["provider"])
