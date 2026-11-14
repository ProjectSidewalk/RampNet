"""The #276 definition optimizer: the exemplar picker's vetoes, the review page, the
critic's ledger shape and the trajectory bookkeeping. No model, no network."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "model_comparison"))

import prompt_optimize as P  # noqa: E402
from rampnet.detection_eval import GroundTruth  # noqa: E402

R = 0.022


def _case():
    gts = {
        "A": GroundTruth([(0.50, 0.60), (0.80, 0.70)], [], True),
        "B": GroundTruth([(0.20, 0.65)], [(0.60, 0.60)], True),   # an ignore point
        "C": GroundTruth([(0.10, 0.62)], [], True),
    }
    points = {
        # a TP, a near-miss 1.5R from a GT (localization error), and a far FP at 0.7
        "A": [(0.50, 0.60, 0.9), (0.80 + 1.5 * R, 0.70, 0.95), (0.30, 0.50, 0.7)],
        # a 0.99 box on the ignore point; a weak box that wraps to 312 px from the GT
        "B": [(0.60, 0.60, 0.99), (0.90, 0.55, 0.3)],
        # a 0.97 box a human tagged A/U in incremental_fp_tags (a ramp the GT missed)
        "C": [(0.40, 0.40, 0.97)],
    }
    return points, gts, {"C": [(0.40, 0.40)]}


def test_worst_fp_skips_near_misses_ignores_and_vetoed_points():
    points, gts, veto = _case()
    fp, fn, diag = P.pick_exemplars(points, gts, veto, set())
    # 0.99 sits on an ignore point, 0.97 is vetoed, 0.95 is within 2R of a GT: all unusable.
    assert (fp["pano"], fp["x"], fp["conf"]) == ("A", 0.30, 0.7)
    # The pure miss farthest from any prediction (C: 327 px; B wraps to 312 px; A's
    # second GT has a prediction within 2R so it is a localization error, not a miss).
    assert fn["pano"] == "C" and 320 < fn["nearest_pred_px"] < 335
    assert diag["recall_2R"] > diag["recall_1R"]           # the near-miss counts at 2R
    assert diag["n_gt"] == 4
    assert diag["n_pred"] == 5                             # the ignore-point box is neither TP nor FP


def test_used_panos_are_excluded_and_a_weak_fp_still_qualifies():
    points, gts, veto = _case()
    fp, fn, _ = P.pick_exemplars(points, gts, veto, {"A", "C"})
    assert fp["pano"] == "B" and fp["conf"] == pytest.approx(0.3)
    assert fn["pano"] == "B"
    fp, fn, _ = P.pick_exemplars(points, gts, veto, {"A", "B", "C"})
    assert fp is None and fn is None


def test_unscored_predictions_rank_below_scored_ones():
    gts = {"A": GroundTruth([], [], True)}
    points = {"A": [(0.1, 0.5), (0.5, 0.5, 0.2)]}
    fp, _, _ = P.pick_exemplars(points, gts, {}, set())
    assert fp["x"] == 0.5 and fp["conf"] == 0.2


def test_critic_prompt_names_only_the_confirmed_images():
    both = P.critic_prompt("A curb ramp is X.", True, True)
    assert "Image 1: the RED box" in both and "Image 2: the BLUE box" in both
    assert "excludes things like the red-boxed" in both and "includes cases like the blue" in both
    fn_only = P.critic_prompt("A curb ramp is X.", False, True)
    assert "RED" not in fn_only and "Image 1: the BLUE box" in fn_only
    assert '"A curb ramp is X."' in fn_only
    assert P.alternative_prompt("A curb ramp is X.").startswith("Rewrite the following")


def test_critic_rows_are_recognisable_in_the_ledger():
    pytest.importorskip("google.genai")  # Critic.ask builds its config from the SDK types
    class _UM:
        prompt_token_count = 2000
        candidates_token_count = 80
        thoughts_token_count = 300
        total_token_count = 2380

    class _Cand:
        finish_reason = type("FR", (), {"name": "STOP"})()

    class _Resp:
        text = json.dumps({"definition": "A curb ramp   is a sloped\n cut."})
        usage_metadata = _UM()
        candidates = [_Cand()]
        model_version = "gemini-4-argon-001"

    class _Models:
        kw = None

        def generate_content(self, **kw):
            _Models.kw = kw
            return _Resp()

    class _Client:
        models = _Models()

    critic = P.Critic("gemini-4-argon", _Client(), "prompt text")
    definition, raw = critic.ask([])
    assert definition == "A curb ramp is a sloped cut."          # whitespace-normalized
    assert critic.usage["calls"] == 1 and critic.usage["input_tokens"] == 2000
    assert critic.usage["output_tokens"] == 380 and critic.usage["thoughts_tokens"] == 300
    assert critic.stop_reasons == {"STOP": 1}
    assert critic.model_versions == {"gemini-4-argon-001": 1}
    assert critic.signature()["role"] == "critic" and critic.run_notes()["role"] == "critic"
    assert critic.name == "gemini" and critic.model_id == "gemini-4-argon"   # pricing + reconcile
    cfg = _Models.kw["config"]
    assert cfg.temperature == P.CRITIC_TEMPERATURE
    assert cfg.response_mime_type == "application/json"


def test_trajectory_picks_the_best_accepted_definition(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OPT_DIR", tmp_path)
    m = "gemini-4-argon"
    assert P.current_definition(m) == (None, None) and P.next_iteration(m) == 1
    P.append_trajectory(m, {"kind": "seed", "train_f1": 0.60})
    P.append_trajectory(m, {"kind": "evaluate", "iteration": 1, "definition": "def276-argon-r1",
                            "train_f1": 0.65, "accepted": True, "fp": {"pano": "p1"}, "fn": None})
    P.append_trajectory(m, {"kind": "evaluate", "iteration": 2, "definition": "def276-argon-r2",
                            "train_f1": 0.64, "accepted": False, "fp": None, "fn": {"pano": "p2"}})
    assert P.current_definition(m) == ("def276-argon-r1", 0.65)
    assert P.used_panos(m) == {"p1", "p2"}
    assert P.next_iteration(m) == 3
    assert P.short_name(m) == "argon" and P.short_name("gemini-3.8-flash") == "flash38"


def test_rewrite_acts_on_the_latest_picked_iteration_not_the_trajectory(tmp_path, monkeypatch):
    # Before the first evaluation the trajectory holds only the seed row; the picked
    # iteration is known from its iter_NN/candidates.json, not from the trajectory.
    monkeypatch.setattr(P, "OPT_DIR", tmp_path)
    m = "gemini-4-argon"
    assert P.latest_iteration(m) == 1
    P.append_trajectory(m, {"kind": "seed", "train_f1": 0.60})
    for k in (1, 2):
        d = P.iter_dir(m, k)
        d.mkdir(parents=True)
        (d / "candidates.json").write_text("{}", encoding="utf-8")
    P.iter_dir(m, 3).mkdir()                      # picked nothing yet: no candidates.json
    assert P.latest_iteration(m) == 2


def test_a_candidate_definition_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DEFINITIONS_DIR", tmp_path)
    path = P.write_definition("def276-argon-r1", "  A curb ramp is X.  ")
    assert path.read_text(encoding="utf-8") == "A curb ramp is X.\n"
    with pytest.raises(SystemExit):
        P.write_definition("def276-argon-r1", "something else")


def test_review_page_is_self_contained_and_asks_both_questions(tmp_path):
    img = tmp_path / "fp.jpg"
    img.write_bytes(b"\xff\xd8fake")
    item = {"pano": "p1", "x": 0.3, "y": 0.5, "conf": 0.7, "view_yaw_deg": 300.0,
            "_image": str(img), "_detail": str(img)}
    diag = {"precision_2R": 0.5, "recall_2R": 0.6}
    report = {"f1": 0.55, "precision": 0.5, "recall": 0.6}
    out = tmp_path / "review.html"
    P.write_review_html(str(out), "gemini-4-argon", 1, "A curb ramp is X.", item, None,
                        diag, report)
    html = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html and 'src="fp.jpg"' not in html
    assert "really NOT a curb ramp" in html and "No candidate qualified" in html
    assert "sao_paulo:p1" in html and "verdicts.json" in html


# --------------------------------------------------------------------------- #
# --scope prompt (the whole-prompt arm)
# --------------------------------------------------------------------------- #
class _FakeModels:
    calls = []

    def __init__(self, answer):
        self.answer = answer

    def generate_content(self, **kw):
        _FakeModels.calls.append(kw)
        answer = self.answer

        class _Resp:
            text = json.dumps(answer)
            usage_metadata = None
            candidates = []
            model_version = None
        return _Resp()


def _fake_critic(monkeypatch, answer):
    class _Client:
        models = _FakeModels(answer)
    _FakeModels.calls = []
    monkeypatch.setattr(P, "_client_for", lambda model: _Client())
    monkeypatch.setattr(P.C, "load_dotenv_for_run", lambda: None)
    monkeypatch.setattr(P.C, "report_usage", lambda *a, **k: None)


def _judged_definition_iteration(tmp_path, model, k, fp_verdict, fn_verdict, seed_f1=0.6):
    """A definition-scope iter_NN/ as Step 4 left it: candidates, verdicts, crops."""
    d = tmp_path / "optimize" / model / f"iter_{k:02d}"
    d.mkdir(parents=True)
    from PIL import Image
    for i, f in enumerate(("fp.jpg", "fp_detail.jpg", "fn.jpg", "fn_detail.jpg")):
        Image.new("RGB", (8, 8), (40 * i, 0, 0)).save(d / f, quality=90)
    cand = {"iteration": k, "model": model, "definition": None, "train": {"f1": seed_f1},
            "fp": {"pano": f"fp{k}", "x": 0.3, "y": 0.6, "conf": 0.8,
                   "image": "fp.jpg", "detail": "fp_detail.jpg"},
            "fn": {"pano": f"fn{k}", "x": 0.7, "y": 0.6, "nearest_pred_px": None,
                   "image": "fn.jpg", "detail": "fn_detail.jpg"}}
    (d / "candidates.json").write_text(json.dumps(cand), encoding="utf-8")
    (d / "verdicts.json").write_text(json.dumps(
        {"iteration": k, "model": model, "fp": fp_verdict, "fn": fn_verdict}), encoding="utf-8")
    return d


def test_prompt_scope_names_and_record_location(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OPT_DIR", tmp_path)
    m = "gemini-4-argon"
    assert P.candidate_name("prompt", m, 1) == "prompt276-argon-r1"
    assert P.candidate_name("prompt", "gemini-3.8-flash", "alt") == "prompt276-flash38-alt"
    assert P.candidate_name("definition", m, 2) == "def276-argon-r2"
    assert P.state_dir(m, "prompt") == tmp_path / "optimize" / m / "prompt"
    assert P.iter_dir(m, 3, "prompt") == tmp_path / "optimize" / m / "prompt" / "iter_03"
    assert P.candidate_dir("prompt") == P.PROMPTS_DIR and P.candidate_dir("definition") == P.DEFINITIONS_DIR
    # separate trajectories: a prompt-scope row never lands in the definition record
    P.append_trajectory(m, {"kind": "seed", "train_f1": 0.6})
    P.append_trajectory(m, {"kind": "evaluate", "iteration": 1, "prompt_body": "prompt276-argon-r1",
                            "train_f1": 0.7, "accepted": True, "fp": {"pano": "q"}, "fn": None},
                        "prompt")
    assert P.current_definition(m) == (None, 0.6)
    assert P.current_definition(m, "prompt") == ("prompt276-argon-r1", 0.7)
    assert len(P.read_trajectory(m)) == 1 and P.next_iteration(m, "prompt") == 2
    # a fresh prompt-scope pick also skips panos the definition scope already showed
    P.append_trajectory(m, {"kind": "evaluate", "iteration": 1, "definition": "def276-argon-r1",
                            "train_f1": 0.5, "accepted": False, "fp": {"pano": "d"}, "fn": None})
    assert P.used_panos(m) == {"d"} and P.used_panos(m, "prompt") == {"d", "q"}
    assert P.detector_pins("prompt", "x") == {"definition": None, "prompt_body": "x"}
    assert P.current_text("prompt", None) == P.D.DETECTION_PROMPT


def test_default_scope_is_the_definition_scope_unchanged():
    ap = P.build_parser()
    for cmd in ("pick", "rewrite", "evaluate", "alternative", "status"):
        extra = ["--floor", "0.02"] if cmd == "evaluate" else []
        assert ap.parse_args([cmd, "--model", "gemini-4-argon"] + extra).scope == "definition"
    critic = P.Critic("gemini-4-argon", None, "p")
    assert critic.field == "definition" and critic.signature()["schema"] == json.dumps(
        P.CRITIC_SCHEMA, sort_keys=True)
    assert P.state_dir("gemini-4-argon") == P.OPT_DIR / "optimize" / "gemini-4-argon"


def test_prompt_scope_reuses_judged_crops_and_their_verdicts(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")  # Critic.ask builds its config from the SDK types
    """Iterations 1-2 of the prompt scope take the crops Jon judged in Step 4: no detector
    call, verdicts copied, an 'unsure' crop left out by the same rule as the definition
    scope, and the definition record left byte-for-byte alone."""
    monkeypatch.setattr(P, "OPT_DIR", tmp_path)
    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path / "prompts")
    monkeypatch.setattr(P, "run_detector", lambda *a, **k: pytest.fail("no detector call"))
    m = "gemini-4-argon"
    P.append_trajectory(m, {"kind": "seed", "model": m, "definition": None, "train_f1": 0.6,
                            "train": {"f1": 0.6}})
    src = _judged_definition_iteration(tmp_path, m, 1, "unsure", "ramp")
    def_record = P.trajectory_path(m).read_bytes()

    assert P.main(["pick", "--scope", "prompt", "--model", m, "--usage-log", "none",
                   "--reuse-from", f"optimize/{m}/iter_01"]) == 0
    out = P.iter_dir(m, 1, "prompt")
    assert (out / "verdicts.json").read_bytes() == (src / "verdicts.json").read_bytes()
    assert (out / "fn.jpg").read_bytes() == (src / "fn.jpg").read_bytes()
    cand = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    assert cand["scope"] == "prompt" and cand["prompt_body"] is None and "definition" not in cand
    assert cand["reused_from"] == f"optimize/{m}/iter_01" and cand["fn"]["pano"] == "fn1"
    seed = P.read_trajectory(m, "prompt")[0]
    assert seed["kind"] == "seed" and seed["train_f1"] == 0.6
    assert seed["copied_from"] == f"optimize/{m}/trajectory.jsonl"

    _fake_critic(monkeypatch, {"prompt": "Find   every curb ramp\n and box it."})
    assert P.main(["rewrite", "--scope", "prompt", "--model", m, "--usage-log", "none"]) == 0
    rw = json.loads((out / "rewrite.json").read_text(encoding="utf-8"))
    assert rw["candidate"] == "prompt276-argon-r1" and rw["prompt"].count("BLUE") == 1
    assert rw["used_fp"] is False and rw["used_fn"] is True            # 'unsure' FP left out
    assert "RED" not in rw["prompt"] and P.D.DETECTION_PROMPT in rw["prompt"]
    assert rw["instruction"] == "Find every curb ramp and box it." and rw["scope"] == "prompt"
    assert (tmp_path / "prompts" / "prompt276-argon-r1.txt").read_text(
        encoding="utf-8") == "Find every curb ramp and box it.\n"
    assert len(_FakeModels.calls[0]["contents"]) == 2                 # one image + the prompt
    assert P.trajectory_path(m).read_bytes() == def_record           # definition record untouched

    # the same judged iteration cannot seed a second prompt-scope iteration
    P.append_trajectory(m, {"kind": "evaluate", "iteration": 1, "prompt_body": rw["candidate"],
                            "train_f1": 0.55, "accepted": False, "fp": cand["fp"],
                            "fn": cand["fn"]}, "prompt")
    with pytest.raises(SystemExit, match="reuse each judged iteration once"):
        P.main(["pick", "--scope", "prompt", "--model", m, "--usage-log", "none",
                "--reuse-from", f"optimize/{m}/iter_01"])
    _judged_definition_iteration(tmp_path, m, 2, "not_ramp", "unsure")
    P.main(["pick", "--scope", "prompt", "--model", m, "--usage-log", "none",
            "--reuse-from", f"optimize/{m}/iter_02"])
    assert json.loads((P.iter_dir(m, 2, "prompt") / "candidates.json").read_text(
        encoding="utf-8"))["iteration"] == 2
    with pytest.raises(SystemExit, match="--scope prompt"):
        P.main(["pick", "--model", m, "--usage-log", "none", "--iteration", "9",
                "--reuse-from", f"optimize/{m}/iter_02"])


def test_prompt_scope_alternative_is_an_image_free_paraphrase_of_the_seed(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")  # Critic.ask builds its config from the SDK types
    monkeypatch.setattr(P, "OPT_DIR", tmp_path)
    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path / "prompts")
    _fake_critic(monkeypatch, {"prompt": "Detect all curb ramps; box each."})
    m = "gemini-3.8-flash"
    assert P.main(["alternative", "--scope", "prompt", "--model", m, "--usage-log", "none"]) == 0
    assert (tmp_path / "prompts" / "prompt276-flash38-alt.txt").exists()
    kw = _FakeModels.calls[0]
    assert len(kw["contents"]) == 1 and P.D.DETECTION_PROMPT in kw["contents"][0]
    row = P.read_trajectory(m, "prompt")[0]
    assert row["kind"] == "alternative" and row["from"] is None
    assert row["prompt_body"] == "prompt276-flash38-alt" and row["scope"] == "prompt"
    assert not P.trajectory_path(m).exists()                         # definition record untouched
    assert P.alternative_prompt_whole("X").startswith("Rewrite the following instruction")


def test_best_view_covers_the_whole_ground_band():
    for x in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95):
        picked = P.best_view(x, 0.6)
        assert picked is not None
        view, (u, v) = picked
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
