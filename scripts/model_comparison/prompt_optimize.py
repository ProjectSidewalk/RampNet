"""Exemplar-derived definition optimization for a Gemini detector (#276 Step 4).

The DetPO recipe (arXiv 2603.23455), with a human in the loop: run the detector on
``dev_train``, take its worst false positive and worst false negative, have a reviewer
confirm they really are one ("really not a ramp" / "really a ramp"), show them to the
model with the current definition and ask for a rewrite, keep the rewrite only if
``dev_train`` F1 improves by at least the seed-repeat spread, and select among the
accepted candidates on ``dev_select`` once (that last step is ordinary ``compare.py``
runs with ``--definition``). Everything is written under
``benchmark/prompt_opt_276/optimize/<model>/`` so the trajectory is a committed record.

One iteration is three commands, because the human check sits in the middle::

    python scripts/model_comparison/prompt_optimize.py pick     --model gemini-4-argon
    #   -> optimize/<model>/iter_01/review.html  (open locally; write verdicts.json)
    python scripts/model_comparison/prompt_optimize.py rewrite  --model gemini-4-argon
    python scripts/model_comparison/prompt_optimize.py evaluate --model gemini-4-argon --floor 0.02

``pick`` scores the current definition on ``dev_train`` (confidence on, so the FP is
ranked by the model's own score) and chooses:

- the **worst FP**: the highest-scored unmatched prediction with no ground-truth,
  ignore or A/U-tagged point within 2x the match radius -- so a near-miss of a real
  ramp, or a ramp the GT never recorded, cannot be shown to the model as "not a ramp";
- the **worst FN**: the ground-truth ramp farthest from any prediction, with none
  within 2x the radius -- a pure miss, not a localization error.

Panos already used are excluded from later picks. ``rewrite`` calls the same model as
critic (one call, both crops) and records that spend in the usage log with
``role: critic``. ``evaluate`` scores the candidate on ``dev_train`` and appends one
line to ``trajectory.jsonl``; ``alternative`` asks for an image-free rewording of the
best candidate (DetPO's GenerateAlternative), to be selected against on dev_select.

Nothing here reads a held-out split. Nothing here changes the detector; a candidate
definition is a file under ``definitions/`` and a leg that uses it is
``--definition <name>``.

``--scope prompt`` (the whole-prompt arm) runs the same loop over the WHOLE instruction
(task verb, definition, what to do when unsure) instead of the definition alone. Its
candidates are ``prompts/prompt276-<model>-r<k>.txt``, scored with ``--prompt-body``,
and its record is ``optimize/<model>/prompt/`` with its own ``trajectory.jsonl``, so the
definition-scope record is never touched. An iteration can reuse crops a human already
judged in the definition scope instead of picking (and rating) new ones::

    python scripts/model_comparison/prompt_optimize.py pick --scope prompt \\
        --model gemini-4-argon --reuse-from optimize/gemini-4-argon/iter_01
    python scripts/model_comparison/prompt_optimize.py rewrite  --scope prompt --model gemini-4-argon
    python scripts/model_comparison/prompt_optimize.py evaluate --scope prompt --model gemini-4-argon --floor 0.03
    python scripts/model_comparison/prompt_optimize.py alternative --scope prompt --model gemini-4-argon
"""
import argparse
import base64
import io
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(HERE).resolve().parents[1]
sys.path.insert(0, HERE)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT))

import compare as C  # noqa: E402
import detectors as D  # noqa: E402
from equirect_tiling import default_views, equirect_point_to_perspective, equirect_to_perspective  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    PANO_SCALE_X, PANO_SCALE_Y, aggregate, prediction_confidence, radius_sq_for, score_pano,
)
from rampnet.metrics import greedy_match  # noqa: E402

CITY = "sao_paulo"                       # the dev split (#51 convention)
OPT_DIR = Path(D.PROMPT_OPT_DIR)
SPLIT_FILE = OPT_DIR / "sao_paulo_split.json"
DEFINITIONS_DIR = OPT_DIR / "definitions"
PROMPTS_DIR = OPT_DIR / "prompts"
SCOPES = ("definition", "prompt")
# Per scope: the candidate-name prefix, and the trajectory key that names a candidate.
CANDIDATE_PREFIX = {"definition": "def276", "prompt": "prompt276"}
NAME_KEY = {"definition": "definition", "prompt": "prompt_body"}
SHORT = {"gemini-4-argon": "argon", "gemini-3.8-flash": "flash38"}
VETO_TAGS = ("a", "u")                    # A = a real ramp the GT missed, U = unsure (RUBRICS §2)
BOX_HALF_PX = 80                          # the marker drawn around a point, in 1024-px view pixels
DETAIL_HALF_PX = 150

CRITIC_SCHEMA = {"type": "object", "properties": {"definition": {"type": "string"}},
                 "required": ["definition"]}
PROMPT_CRITIC_SCHEMA = {"type": "object", "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"]}
PROMPT_MAX_WORDS = 150                    # the seed instruction is 66 words
CRITIC_TEMPERATURE = 1.0                  # a rewrite wants variety; recorded in run_options


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def short_name(model):
    try:
        return SHORT[model]
    except KeyError:
        raise SystemExit(f"no short name for {model!r}; add it to prompt_optimize.SHORT")


def state_dir(model, scope="definition"):
    """``optimize/<model>/`` for the definition scope (the Step 4 record, unchanged) and
    ``optimize/<model>/prompt/`` for the whole-prompt scope."""
    base = OPT_DIR / "optimize" / model
    return base if scope == "definition" else base / "prompt"


def trajectory_path(model, scope="definition"):
    return state_dir(model, scope) / "trajectory.jsonl"


def read_trajectory(model, scope="definition"):
    p = trajectory_path(model, scope)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def append_trajectory(model, row, scope="definition"):
    p = trajectory_path(model, scope)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def current_definition(model, scope="definition"):
    """``(name, train_f1)`` of the best accepted candidate, or ``(None, seed_f1)``.
    In the prompt scope the name is a prompt body (``prompts/<name>.txt``)."""
    best, best_f1 = None, None
    for row in read_trajectory(model, scope):
        if row.get("kind") == "seed":
            best_f1 = row["train_f1"]
        elif row.get("kind") == "evaluate" and row.get("accepted"):
            best, best_f1 = row[NAME_KEY[scope]], row["train_f1"]
    return best, best_f1


def used_panos(model, scope="definition"):
    """Panos already shown to the reviewer. The prompt scope also excludes every pano the
    definition scope used, so a fresh prompt-scope pick never re-asks about a judged crop
    (reusing those is what ``pick --reuse-from`` is for)."""
    used = set()
    scopes = ("definition",) if scope == "definition" else ("definition", scope)
    for sc in scopes:
        for row in read_trajectory(model, sc):
            for k in ("fp", "fn"):
                if row.get(k) and row[k].get("pano"):
                    used.add(row[k]["pano"])
    return used


def next_iteration(model, scope="definition"):
    its = [row["iteration"] for row in read_trajectory(model, scope) if "iteration" in row]
    return (max(its) + 1) if its else 1


def iter_dir(model, k, scope="definition"):
    return state_dir(model, scope) / f"iter_{k:02d}"


def latest_iteration(model, scope="definition"):
    """The most recently picked iteration: the highest ``iter_NN/`` holding candidates.json.

    ``rewrite`` and ``evaluate`` act on it. Before the first evaluation the trajectory holds
    only the seed row, so deriving it from the trajectory gave 0 (2026-11-09).
    """
    ks = [int(p.name[5:]) for p in state_dir(model, scope).glob("iter_[0-9][0-9]")
          if (p / "candidates.json").exists()]
    return max(ks) if ks else 1


def current_text(scope, name):
    """The text the critic rewrites: the definition (seed ``CURB_RAMP_DEFINITION``) or, in
    the prompt scope, the whole instruction (seed ``DETECTION_PROMPT``)."""
    if scope == "definition":
        return D.CURB_RAMP_DEFINITION if name is None else D.load_definition(name)
    return D.DETECTION_PROMPT if name is None else D.load_prompt_body(name)


def detector_pins(scope, name):
    """``{definition, prompt_body}`` for ``run_detector``: the candidate goes into its own
    scope's knob; the other stays unset (the two are exclusive)."""
    if scope == "definition":
        return {"definition": name, "prompt_body": None}
    return {"definition": None, "prompt_body": name}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_dev(which):
    with open(SPLIT_FILE, encoding="utf-8") as f:
        return json.load(f)[which]


def load_veto_points():
    """Pano -> [(x, y)] of A/U-tagged incremental FPs: places a human said were, or might
    be, a real ramp the GT missed. Never shown to the model as a false positive."""
    path = REPO_ROOT / "benchmark" / CITY / "incremental_fp_tags.json"
    veto = {}
    if not path.exists():
        return veto
    with open(path, encoding="utf-8") as f:
        tags = json.load(f)
    for key, tag in tags.items():
        if str(tag).lower() not in VETO_TAGS:
            continue
        pano, x, y = key.rsplit("_", 2)
        veto.setdefault(pano, []).append((float(x), float(y)))
    return veto


def compare_args(cache_dir, definition, confidence=True, concurrency=6, debug_dump=None,
                 prompt_body=None):
    from fp_taxonomy import _compare_args
    ns = _compare_args(str(cache_dir))
    ns.definition = definition
    ns.prompt_body = prompt_body
    ns.gemini_confidence = True if confidence else None
    ns.gemini_concurrency = concurrency
    ns.gemini_debug_dump = debug_dump
    return ns


def run_detector(model, definition, pids, cache_dir, usage_log, concurrency=6, tag="",
                 prompt_body=None):
    """Score ``model`` with ``definition`` or ``prompt_body`` (both None = seed) over
    ``pids`` of the dev split, through compare.score_model so the cache, the spend guard
    and the ledger row are the ordinary ones. Returns ``(points_by_pid, gts, report)``."""
    C.load_dotenv_for_run()
    bundle = str(REPO_ROOT / "benchmark" / CITY)
    records, verdicts, panos_dir = C.load_bundle(bundle)
    gts_all = C.ground_truths_from_verdicts(records, verdicts)
    gts = {p: gts_all[p] for p in pids}
    args = compare_args(cache_dir, definition, concurrency=concurrency, prompt_body=prompt_body)
    label, det = D.build_detector("gemini", model, records, args)
    cache = C.DetectionCache(str(cache_dir), enabled=True)
    rsq = radius_sq_for()
    timing = {}
    t0 = perf_counter()
    try:
        run = C.score_model(det, records, gts, panos_dir, rsq, label, CITY, cache,
                            spend_needs_recording=usage_log is None, timing=timing)
    finally:
        timing["elapsed_s"] = round(perf_counter() - t0, 3)
        C.report_usage(det, label, CITY, len(gts), usage_log, timing=timing,
                       pano_list=C.pano_list_note(f"{SPLIT_FILE}@{tag or 'dev_train'}", pids))
    if run.failures:
        raise SystemExit(f"{len(run.failures)} pano(s) failed; fix or re-run before picking "
                         f"exemplars from a partial run: {run.failures[:3]}")
    sig = det.signature()
    points = {p: cache.get(C.cache_key(label, sig, CITY, p)) for p in pids}
    missing = [p for p, v in points.items() if v is None]
    if missing:
        raise SystemExit(f"{len(missing)} pano(s) not in the cache after scoring: {missing[:3]}")
    return points, gts, run.report, panos_dir


# --------------------------------------------------------------------------- #
# picking the exemplars (pure; unit-tested)
# --------------------------------------------------------------------------- #
def _dist_sq(ax, ay, bx, by, wrap_x=True):
    dx = abs(ax - bx) * PANO_SCALE_X
    if wrap_x:
        dx = min(dx, PANO_SCALE_X - dx)
    dy = (ay - by) * PANO_SCALE_Y
    return dx * dx + dy * dy


def _ordered(preds):
    confs = [prediction_confidence(p) for p in preds]
    idx = sorted(range(len(preds)), key=lambda i: -(confs[i] if confs[i] is not None else -1.0))
    return [preds[i] for i in idx]


def pick_exemplars(points, gts, veto, used, radius_sq=None):
    """``(fp, fn, diagnostics)`` over the dev_train panos.

    ``fp`` is ``{pano, x, y, conf}`` for the highest-scored unmatched prediction with no
    GT, ignore or veto point within 2R; ``fn`` is ``{pano, x, y, nearest_pred_px}`` for the
    GT point farthest from any prediction, with none within 2R. Either is ``None`` when no
    candidate qualifies. Panos in ``used`` are skipped. The diagnostics carry the 1R and 2R
    precision/recall (ignore points handled as the scoreboard does) so a reader can see
    how much of the error is localization."""
    rsq = radius_sq_for() if radius_sq is None else radius_sq
    rsq2 = rsq * 4.0                      # (2R)^2
    fp_best, fn_best = None, None
    for pano, gt in gts.items():
        if pano in used:
            continue
        preds = _ordered(points.get(pano) or [])
        xy = [(p[0], p[1]) for p in preds]
        assign = greedy_match(xy, gt.gt_points, rsq, PANO_SCALE_X, PANO_SCALE_Y, wrap_x=True)
        protected = list(gt.gt_points) + list(gt.ignore_points) + list(veto.get(pano, []))
        for p, (g, _) in zip(preds, assign):
            if g >= 0:
                continue
            x, y = p[0], p[1]
            if any(_dist_sq(x, y, qx, qy) < rsq2 for qx, qy in protected):
                continue
            conf = prediction_confidence(p)
            conf = -1.0 if conf is None else conf
            if fp_best is None or conf > fp_best["conf"]:
                fp_best = {"pano": pano, "x": x, "y": y, "conf": conf}
        for gx, gy in gt.gt_points:
            d = min((_dist_sq(gx, gy, px, py) for px, py in xy), default=float("inf"))
            if d < rsq2:
                continue
            dpx = math.sqrt(d) if d != float("inf") else None
            if fn_best is None or d > fn_best["_key"]:
                fn_best = {"pano": pano, "x": gx, "y": gy, "nearest_pred_px": dpx, "_key": d}
    if fn_best is not None:
        fn_best.pop("_key")
    # P/R at 2R over ALL dev_train panos (used ones included): how much of the error is
    # localization. score_pano handles ignore points the same way the scoreboard does.
    one = train_report(points, gts, rsq)
    two = train_report(points, gts, rsq2)
    diag = {"precision_1R": one["precision"], "recall_1R": one["recall"],
            "precision_2R": two["precision"], "recall_2R": two["recall"],
            "n_gt": one["n_gt"], "n_pred": one["tp"] + one["fp"]}
    return fp_best, fn_best, diag


def train_report(points, gts, radius_sq=None):
    rsq = radius_sq_for() if radius_sq is None else radius_sq
    scores = [score_pano(points[p], gts[p], rsq) for p in gts]
    rep = aggregate(scores)
    return {"precision": round(rep.precision, 4), "recall": round(rep.recall, 4),
            "f1": round(rep.f1, 4), "tp": sum(s.tp for s in scores),
            "fp": sum(s.fp for s in scores), "n_gt": sum(s.n_gt for s in scores)}


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #
def best_view(x, y, views=None):
    """The default view that sees pano point (x, y) closest to its centre."""
    best, best_d = None, None
    for v in views or default_views():
        uv = equirect_point_to_perspective(x, y, v)
        if uv is None:
            continue
        d = (uv[0] - 0.5) ** 2 + (uv[1] - 0.5) ** 2
        if best is None or d < best_d:
            best, best_d = (v, uv), d
    return best


def render_marked_view(pano_path, x, y, color, out_stem, source_max_edge=4096):
    """Write ``<out_stem>.jpg`` (the 1024x1024 view the model saw, with a box drawn around
    the point) and ``<out_stem>_detail.jpg`` (a 2x crop around it). Returns metadata."""
    from PIL import ImageDraw
    pano = D.load_pano_image(pano_path, source_max_edge)
    picked = best_view(x, y)
    if picked is None:
        raise ValueError(f"point ({x:.4f}, {y:.4f}) is outside every default view")
    view, (u, v) = picked
    img = equirect_to_perspective(pano, view)
    px, py = u * view.width, v * view.height
    detail = img.crop((int(max(0, px - DETAIL_HALF_PX)), int(max(0, py - DETAIL_HALF_PX)),
                       int(min(view.width, px + DETAIL_HALF_PX)), int(min(view.height, py + DETAIL_HALF_PX))))
    detail = detail.resize((detail.width * 2, detail.height * 2))
    draw = ImageDraw.Draw(img)
    draw.rectangle([px - BOX_HALF_PX, py - BOX_HALF_PX, px + BOX_HALF_PX, py + BOX_HALF_PX],
                   outline=color, width=6)
    img.save(out_stem + ".jpg", quality=90)
    detail.save(out_stem + "_detail.jpg", quality=90)
    return {"view_yaw_deg": view.yaw_deg, "view_pitch_deg": view.pitch_deg,
            "u": round(u, 4), "v": round(v, 4), "image": os.path.basename(out_stem) + ".jpg",
            "detail": os.path.basename(out_stem) + "_detail.jpg"}


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def write_review_html(path, model, k, definition_text, fp, fn, diag, report, what="definition"):
    """A self-contained local page (never an Artifact): the two crops, the question,
    and a form that produces the verdicts.json to save beside it. ``what`` names the
    text shown ("definition", or "instruction" in the prompt scope)."""
    def panel(title, item, question, key):
        if item is None:
            return f"<section><h2>{title}</h2><p>No candidate qualified this iteration.</p></section>"
        return f"""
<section>
  <h2>{title}</h2>
  <p class="meta">pano <code>{CITY}:{item['pano']}</code>, pano-normalized ({item['x']:.4f}, {item['y']:.4f}),
     view yaw {item['view_yaw_deg']:.0f}&deg;{', model score %.2f' % item['conf'] if 'conf' in item else ''}</p>
  <div class="row">
    <figure><img src="data:image/jpeg;base64,{_b64(item['_image'])}" alt="{title}: the 1024x1024 view with the point boxed"><figcaption>as the model saw it (box added)</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{_b64(item['_detail'])}" alt="{title}: 2x detail crop"><figcaption>2x detail</figcaption></figure>
  </div>
  <p><strong>{question}</strong></p>
  <label><input type="radio" name="{key}" value="{'not_ramp' if key == 'fp' else 'ramp'}"> {'Confirmed: NOT a curb ramp' if key == 'fp' else 'Confirmed: a curb ramp the model missed'}</label><br>
  <label><input type="radio" name="{key}" value="{'ramp' if key == 'fp' else 'not_ramp'}"> {'It IS a curb ramp (GT error; do not use)' if key == 'fp' else 'NOT a curb ramp (GT error; do not use)'}</label><br>
  <label><input type="radio" name="{key}" value="unsure"> Unsure (do not use)</label>
</section>"""
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>#276 exemplar check: {model} iteration {k}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#222}}
 .row{{display:flex;gap:16px;flex-wrap:wrap}} figure{{margin:0}} img{{max-width:560px;height:auto;border:1px solid #ccc}}
 figcaption{{font-size:.9em;color:#555}} code{{background:#f3f3f3;padding:1px 4px}} .meta{{color:#555}}
 textarea{{width:100%;height:6em;font-family:monospace}} section{{margin-bottom:32px}}
</style></head><body>
<h1>Exemplar check: <code>{model}</code>, iteration {k}</h1>
<p>Current {what} (dev_train F1 {report['f1']:.3f}, P {report['precision']:.3f}, R {report['recall']:.3f};
at 2x radius P {diag['precision_2R'] if diag['precision_2R'] is None else round(diag['precision_2R'], 3)} /
R {diag['recall_2R'] if diag['recall_2R'] is None else round(diag['recall_2R'], 3)}):</p>
<blockquote>{definition_text}</blockquote>
{panel("Worst false positive (red box)", fp, "Is the boxed region really NOT a curb ramp?", "fp")}
{panel("Worst false negative (blue box)", fn, "Is the boxed region really a curb ramp?", "fn")}
<section>
  <h2>Save your verdict</h2>
  <p>Pick one answer per panel, then copy this into <code>verdicts.json</code> beside this page.</p>
  <textarea id="out" readonly></textarea>
</section>
<script>
 function upd(){{const g=n=>{{const e=document.querySelector('input[name='+n+']:checked');return e?e.value:null}};
  document.getElementById('out').value=JSON.stringify({{iteration:{k},model:"{model}",fp:g('fp'),fn:g('fn'),
   judged_at:new Date().toISOString()}},null,1);}}
 document.querySelectorAll('input').forEach(e=>e.addEventListener('change',upd)); upd();
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)


# --------------------------------------------------------------------------- #
# the critic
# --------------------------------------------------------------------------- #
class Critic:
    """The rewrite call, shaped like a detector for ``compare.report_usage``: usage,
    model_versions, stop_reasons, a signature and run notes carrying ``role: critic``."""
    name = "gemini"

    def __init__(self, model_id, client, prompt, field="definition"):
        """``field`` is the JSON key the critic answers in: ``definition`` (the Step 4
        schema, unchanged) or ``prompt`` (the whole-prompt scope)."""
        self.model_id = model_id
        self._client = client
        self.prompt = prompt
        self.field = field
        self.schema = CRITIC_SCHEMA if field == "definition" else PROMPT_CRITIC_SCHEMA
        self.usage = dict.fromkeys(D._VLMDetector.USAGE_KEYS, 0)
        self.model_versions = None
        self.stop_reasons = Counter()

    def signature(self):
        return {"provider": "gemini", "model_id": self.model_id, "role": "critic",
                "prompt": self.prompt, "schema": json.dumps(self.schema, sort_keys=True),
                "temperature": CRITIC_TEMPERATURE}

    def run_notes(self):
        return {"role": "critic", "temperature": CRITIC_TEMPERATURE, "concurrency": 1}

    def ask(self, images):
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model_id, contents=list(images) + [self.prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=self.schema,
                                               temperature=CRITIC_TEMPERATURE))
        # bookkeeping exactly as the detector does it
        D.GeminiDetector._record_usage(self, resp)
        text = getattr(resp, "text", None) or ""
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            raise SystemExit(f"critic returned non-JSON: {text[:300]!r}")
        definition = " ".join(str(out.get(self.field, "")).split())
        if not definition:
            raise SystemExit(f"critic returned no {self.field}: {text[:300]!r}")
        return definition, text

    # borrowed detector methods (they only touch self.usage / model_versions / stop_reasons)
    _usage_lock = D.threading.Lock()
    _usage_warned = False
    record_model_version = D._VLMDetector.record_model_version
    accumulate_usage = D._VLMDetector.accumulate_usage
    _tally_finish = D.GeminiDetector._tally_finish


def critic_prompt(definition, has_fp, has_fn):
    parts = ["You are improving the written definition of a target class for an object "
             "detector that finds curb ramps in street-level images.",
             f"Current definition: \"{definition}\""]
    n = 0
    if has_fp:
        n += 1
        parts.append(f"Image {n}: the RED box marks a region the detector labeled a curb ramp, "
                     "but a human reviewer confirmed it is NOT a curb ramp.")
    if has_fn:
        n += 1
        parts.append(f"Image {n}: the BLUE box marks a real curb ramp that the detector missed, "
                     "confirmed by the reviewer.")
    goals = ["still covers every ordinary curb ramp"]
    if has_fp:
        goals.append("excludes things like the red-boxed region")
    if has_fn:
        goals.append("includes cases like the blue-boxed one")
    parts.append("Rewrite the definition so that it " + ", ".join(goals) + ". Describe visual "
                 "features a detector can use; do not refer to these specific images or to "
                 "'the reviewer'. One paragraph, at most 90 words, beginning with 'A curb ramp'. "
                 "Return JSON: {\"definition\": \"...\"}.")
    return "\n\n".join(parts)


def alternative_prompt(definition):
    return ("Rewrite the following definition of a curb ramp for an object detector in "
            "different words, keeping exactly the same meaning and the same inclusions and "
            "exclusions. One paragraph, at most 90 words, beginning with 'A curb ramp'. "
            f"Definition: \"{definition}\"\n\nReturn JSON: {{\"definition\": \"...\"}}.")


# What a whole-prompt rewrite must keep, so the detector's output stays parseable and the
# provider suffix (schema / confidence sentence) still reads as a continuation.
_PROMPT_KEEP = ("Keep it an instruction to detect curb ramps and return one tight bounding box "
                "per curb ramp, and an empty list when there are none. Do not specify an "
                "output format, coordinates or scores; those are added separately.")


def critic_prompt_whole(prompt, has_fp, has_fn):
    """The critic call of the prompt scope: the same evidence as ``critic_prompt``, but
    the critic may rewrite the whole instruction (task wording, the description of a
    curb ramp, what to exclude, what to do when unsure)."""
    parts = ["You are improving the instruction given to an object detector, a "
             "vision-language model that finds curb ramps in street-level images. The "
             "instruction is sent with every image.",
             f"Current instruction: \"{prompt}\""]
    n = 0
    if has_fp:
        n += 1
        parts.append(f"Image {n}: the RED box marks a region the detector labeled a curb ramp, "
                     "but a human reviewer confirmed it is NOT a curb ramp.")
    if has_fn:
        n += 1
        parts.append(f"Image {n}: the BLUE box marks a real curb ramp that the detector missed, "
                     "confirmed by the reviewer.")
    goals = ["still finds every ordinary curb ramp"]
    if has_fp:
        goals.append("does not box things like the red-boxed region")
    if has_fn:
        goals.append("finds cases like the blue-boxed one")
    parts.append("Rewrite the whole instruction so that the detector " + ", ".join(goals) +
                 ". You may change any part of it: how the task is worded, how a curb ramp is "
                 "described, what to exclude, and what to do when unsure. " + _PROMPT_KEEP +
                 " Describe visual features a detector can use; do not refer to these specific "
                 f"images or to 'the reviewer'. One paragraph, at most {PROMPT_MAX_WORDS} words. "
                 "Return JSON: {\"prompt\": \"...\"}.")
    return "\n\n".join(parts)


def alternative_prompt_whole(prompt):
    """The image-free paraphrase control of the prompt scope."""
    return ("Rewrite the following instruction for an object detector that finds curb ramps "
            "in street-level images in different words, keeping exactly the same meaning, the "
            "same inclusions and exclusions, and the same requested output. " + _PROMPT_KEEP +
            f" One paragraph, at most {PROMPT_MAX_WORDS} words. "
            f"Instruction: \"{prompt}\"\n\nReturn JSON: {{\"prompt\": \"...\"}}.")


def write_definition(name, text, directory=None):
    """Write a candidate as ``<directory>/<name>.txt`` (default ``definitions/``; the prompt
    scope passes ``PROMPTS_DIR``). A candidate is never overwritten."""
    directory = DEFINITIONS_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.txt"
    if path.exists():
        raise SystemExit(f"{path} exists; a candidate is never overwritten")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.strip() + "\n")
    return path


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _client_for(model):
    det = D.GeminiDetector(model_id=model)
    det.prepare()
    return det._client

def candidate_dir(scope):
    """Where a scope's candidates are written: ``definitions/`` or ``prompts/``."""
    return DEFINITIONS_DIR if scope == "definition" else PROMPTS_DIR


def candidate_name(scope, model, k):
    """``def276-<short>-r<k>`` / ``prompt276-<short>-r<k>``; ``k="alt"`` for the
    image-free alternative."""
    suffix = "alt" if k == "alt" else f"r{k}"
    return f"{CANDIDATE_PREFIX[scope]}-{short_name(model)}-{suffix}"


def _rel(path):
    """``path`` relative to ``OPT_DIR`` when it is under it (for the committed record)."""
    try:
        return Path(path).resolve().relative_to(Path(OPT_DIR).resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_reuse_source(spec):
    """``--reuse-from``: a judged iteration directory, given relative to
    ``benchmark/prompt_opt_276/`` (``optimize/gemini-4-argon/iter_01``), relative to the
    working directory, or absolute. It must hold ``candidates.json`` and ``verdicts.json``."""
    src = Path(spec)
    if not src.is_absolute() and (OPT_DIR / src).is_dir():
        src = OPT_DIR / src
    for f in ("candidates.json", "verdicts.json"):
        if not (src / f).exists():
            raise SystemExit(f"--reuse-from {spec}: no {f} in {src}; it must be a judged "
                             f"iteration (pick, then the saved verdict)")
    return src


def pick_reused(model, k, src, scope="prompt"):
    """Seed iteration ``k`` of ``scope`` from crops a human already judged (``src`` is a
    definition-scope ``iter_NN/``), so no detector call and no new rating is needed.

    Copies the crops and ``verdicts.json`` verbatim; the verdicts are then read by
    ``rewrite`` exactly as in the definition scope (an ``unsure`` answer, or a GT-error
    answer, leaves that crop out; neither confirmed rejects the pick). The new
    ``candidates.json`` names the prompt body being rewritten (the scope's current best)
    and records where the crops came from. If the scope has no seed row yet, the
    definition scope's seed row is copied: the seed whole prompt IS the seed definition
    inside the as-run frame (``DETECTION_PROMPT``), so the two are one signature and
    one set of cached detections.

    Example: ``pick_reused("gemini-4-argon", 1, OPT_DIR / "optimize/gemini-4-argon/iter_01")``.
    """
    import shutil
    with open(src / "candidates.json", encoding="utf-8") as f:
        orig = json.load(f)
    if orig.get("model") != model:
        raise SystemExit(f"{src} was picked for {orig.get('model')!r}, not {model!r}")
    already = {row[key]["pano"] for row in read_trajectory(model, scope)
               for key in ("fp", "fn") if row.get(key) and row[key].get("pano")}
    reused = {orig[key]["pano"] for key in ("fp", "fn") if orig.get(key)}
    if reused & already:
        raise SystemExit(f"{src}: its crops were already used in this scope "
                         f"({sorted(reused & already)}); reuse each judged iteration once")
    if not any(r.get("kind") == "seed" for r in read_trajectory(model, scope)):
        seed = next((r for r in read_trajectory(model, "definition") if r.get("kind") == "seed"),
                    None)
        if seed is None:
            raise SystemExit("no definition-scope seed row to copy; run `pick --scope "
                             f"{scope}` without --reuse-from to score the seed")
        assert D.DETECTION_PROMPT == D.detection_prompt_for(D.CURB_RAMP_DEFINITION)
        append_trajectory(model, {"kind": "seed", "model": model, NAME_KEY[scope]: None,
                                  "train_f1": seed["train_f1"], "train": seed["train"],
                                  "copied_from": _rel(trajectory_path(model)),
                                  "scope": scope, "at": _now()}, scope)
    out = iter_dir(model, k, scope)
    out.mkdir(parents=True, exist_ok=True)
    for item in (orig.get("fp"), orig.get("fn")):
        if item:
            for key in ("image", "detail"):
                shutil.copy2(src / item[key], out / item[key])
    shutil.copy2(src / "verdicts.json", out / "verdicts.json")
    name, _ = current_definition(model, scope)
    cand = dict(orig)
    cand.pop("definition", None)
    cand.update({"iteration": k, "scope": scope, NAME_KEY[scope]: name,
                 "reused_from": _rel(src), "reused_iteration": orig.get("iteration"),
                 "picked_under_definition": orig.get("definition"), "reused_at": _now()})
    with open(out / "candidates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cand, f, indent=1)
    return out, cand


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_pick(args):
    model, scope = args.model, args.scope
    k = args.iteration or next_iteration(model, scope)
    out = iter_dir(model, k, scope)
    if (out / "candidates.json").exists() and not args.force:
        raise SystemExit(f"{out} already has candidates; use --force to redo the pick")
    if args.reuse_from:
        if scope == "definition":
            raise SystemExit("--reuse-from seeds a prompt-scope iteration from judged "
                             "definition-scope crops; use it with --scope prompt")
        out, cand = pick_reused(model, k, resolve_reuse_source(args.reuse_from), scope)
        with open(out / "verdicts.json", encoding="utf-8") as f:
            verdicts = json.load(f)
        print(f"iteration {k} ({scope} scope): reused {cand['reused_from']} "
              f"(verdicts fp={verdicts.get('fp')}, fn={verdicts.get('fn')}); "
              f"rewriting {cand[NAME_KEY[scope]] or 'the seed'} -> run `rewrite --scope {scope}`")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    name, _ = current_definition(model, scope)
    definition_text = current_text(scope, name)
    pins = detector_pins(scope, name)
    pids = load_dev("dev_train")
    points, gts, _, panos_dir = run_detector(model, pins["definition"], pids, args.cache_dir,
                                             args.usage_log, concurrency=args.concurrency,
                                             prompt_body=pins["prompt_body"])
    report = train_report(points, gts)
    if not read_trajectory(model, scope):
        seed = {"kind": "seed", "model": model, NAME_KEY[scope]: None,
                "train_f1": report["f1"], "train": report, "at": _now()}
        if scope != "definition":
            seed["scope"] = scope
        append_trajectory(model, seed, scope)
    used = used_panos(model, scope) | set(args.exclude or [])
    fp, fn, diag = pick_exemplars(points, gts, load_veto_points(), used)
    for item, color, stem in ((fp, (220, 30, 30), "fp"), (fn, (30, 80, 220), "fn")):
        if item is None:
            continue
        meta = render_marked_view(os.path.join(panos_dir, f"{item['pano']}.jpg"),
                                  item["x"], item["y"], color, str(out / stem))
        item.update(meta)
        item["_image"], item["_detail"] = str(out / meta["image"]), str(out / meta["detail"])
    write_review_html(str(out / "review.html"), model, k, definition_text, fp, fn, diag, report,
                      what="definition" if scope == "definition" else "instruction")
    for item in (fp, fn):
        if item:
            item.pop("_image", None)
            item.pop("_detail", None)
    cand = {"iteration": k, "model": model, NAME_KEY[scope]: name, "train": report,
            "diagnostics_2R": diag, "fp": fp, "fn": fn, "picked_at": _now(),
            "excluded_panos": sorted(used)}
    if scope != "definition":
        cand["scope"] = scope
    with open(out / "candidates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cand, f, indent=1)
    print(f"iteration {k}: dev_train F1 {report['f1']:.3f} (P {report['precision']:.3f}, "
          f"R {report['recall']:.3f}); 2R P/R "
          f"{diag['precision_2R'] and round(diag['precision_2R'], 3)}/"
          f"{diag['recall_2R'] and round(diag['recall_2R'], 3)}")
    print(f"  FP: {fp and (fp['pano'], round(fp['conf'], 2))}   FN: {fn and fn['pano']}")
    print(f"  review page: {out / 'review.html'}  -> save verdicts.json beside it")
    return 0


def cmd_rewrite(args):
    model, scope = args.model, args.scope
    k = args.iteration or latest_iteration(model, scope)
    out = iter_dir(model, k, scope)
    with open(out / "candidates.json", encoding="utf-8") as f:
        cand = json.load(f)
    vpath = out / "verdicts.json"
    if not vpath.exists():
        raise SystemExit(f"no {vpath}: open {out / 'review.html'} and save the verdict first")
    with open(vpath, encoding="utf-8") as f:
        verdicts = json.load(f)
    if (out / "rewrite.json").exists() and not args.force:
        raise SystemExit(f"{out / 'rewrite.json'} exists; --force to redo")
    use_fp = cand["fp"] is not None and verdicts.get("fp") == "not_ramp"
    use_fn = cand["fn"] is not None and verdicts.get("fn") == "ramp"
    if not (use_fp or use_fn):
        append_trajectory(model, {"kind": "pick_rejected", "iteration": k, "model": model,
                                  "fp": cand["fp"], "fn": cand["fn"], "verdicts": verdicts,
                                  "at": _now()}, scope)
        raise SystemExit("neither exemplar was confirmed; both panos are now excluded -- run "
                         "`pick` again for this iteration with --force")
    name = cand[NAME_KEY[scope]]
    definition_text = current_text(scope, name)
    if scope == "definition":
        prompt = critic_prompt(definition_text, use_fp, use_fn)
    else:
        prompt = critic_prompt_whole(definition_text, use_fp, use_fn)
    from PIL import Image
    images = []
    if use_fp:
        images.append(Image.open(out / cand["fp"]["image"]).convert("RGB"))
    if use_fn:
        images.append(Image.open(out / cand["fn"]["image"]).convert("RGB"))
    C.load_dotenv_for_run()
    critic = Critic(model, _client_for(model), prompt,
                    field="definition" if scope == "definition" else "prompt")
    t0 = perf_counter()
    try:
        new_def, raw = critic.ask(images)
    finally:
        C.report_usage(critic, model, CITY, None, args.usage_log,
                       timing={"elapsed_s": round(perf_counter() - t0, 3), "detect_s": 0.0,
                               "panos_called": 0})
    cand_name = candidate_name(scope, model, k)
    path = write_definition(cand_name, new_def, candidate_dir(scope))
    # The new text is "definition" in the definition scope (Step 4's key) and "instruction"
    # in the prompt scope ("prompt" is already the critic prompt).
    record = {"iteration": k, "model": model, "from": name, "candidate": cand_name,
              "used_fp": use_fp, "used_fn": use_fn, "verdicts": verdicts,
              "prompt": prompt, "raw": raw,
              "definition" if scope == "definition" else "instruction": new_def, "at": _now()}
    if scope != "definition":
        record["scope"] = scope
    with open(out / "rewrite.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
    print(f"candidate {cand_name} -> {path}\n{new_def}")
    return 0


def cmd_evaluate(args):
    model, scope = args.model, args.scope
    k = args.iteration or latest_iteration(model, scope)
    out = iter_dir(model, k, scope)
    with open(out / "rewrite.json", encoding="utf-8") as f:
        rw = json.load(f)
    cand_name = rw["candidate"]
    best_name, best_f1 = current_definition(model, scope)
    pins = detector_pins(scope, cand_name)
    pids = load_dev("dev_train")
    points, gts, _, _ = run_detector(model, pins["definition"], pids, args.cache_dir,
                                     args.usage_log, concurrency=args.concurrency,
                                     prompt_body=pins["prompt_body"])
    report = train_report(points, gts)
    gain = report["f1"] - best_f1
    accepted = gain >= args.floor
    rows = read_trajectory(model, scope)
    prev_rejections = 0
    for row in reversed(rows):
        if row.get("kind") == "evaluate":
            if row["accepted"]:
                break
            prev_rejections += 1
    with open(out / "candidates.json", encoding="utf-8") as f:
        cand = json.load(f)
    row = {"kind": "evaluate", "iteration": k, "model": model, NAME_KEY[scope]: cand_name,
           "from": best_name, "train": report, "train_f1": report["f1"], "gain": round(gain, 4),
           "floor": args.floor, "accepted": accepted, "fp": cand["fp"], "fn": cand["fn"],
           "verdicts": rw["verdicts"], "at": _now()}
    if scope != "definition":
        row["scope"] = scope
    append_trajectory(model, row, scope)
    verdict = "ACCEPTED" if accepted else "rejected"
    print(f"{cand_name}: dev_train F1 {report['f1']:.3f} vs {best_f1:.3f} "
          f"({gain:+.3f}, floor {args.floor}) -> {verdict}")
    if not accepted and prev_rejections + 1 >= 2:
        print("two consecutive rejections: the pre-registered early stop. Run `alternative`, "
              "then select on dev_select.")
    return 0


def cmd_alternative(args):
    """DetPO's GenerateAlternative. Definition scope (unchanged): an image-free rewording of
    the best accepted definition. Prompt scope: the paraphrase CONTROL, an image-free
    rewording of the seed whole prompt (``--from-best`` rewords the best accepted body
    instead), so a candidate counts as exemplar learning only if it beats this."""
    model, scope = args.model, args.scope
    best_name, _ = current_definition(model, scope)
    source = best_name if (scope == "definition" or args.from_best) else None
    text = current_text(scope, source)
    C.load_dotenv_for_run()
    if scope == "definition":
        critic = Critic(model, _client_for(model), alternative_prompt(text))
    else:
        critic = Critic(model, _client_for(model), alternative_prompt_whole(text), field="prompt")
    t0 = perf_counter()
    try:
        new_def, raw = critic.ask([])
    finally:
        C.report_usage(critic, model, CITY, None, args.usage_log,
                       timing={"elapsed_s": round(perf_counter() - t0, 3), "detect_s": 0.0,
                               "panos_called": 0})
    cand_name = candidate_name(scope, model, "alt")
    path = write_definition(cand_name, new_def, candidate_dir(scope))
    row = {"kind": "alternative", "model": model, "from": source,
           NAME_KEY[scope]: cand_name, "raw": raw, "at": _now()}
    if scope != "definition":
        row["scope"] = scope
    append_trajectory(model, row, scope)
    print(f"alternative {cand_name} -> {path}\n{new_def}")
    return 0


def cmd_status(args):
    scope = args.scope
    rows = read_trajectory(args.model, scope)
    if not rows:
        print("no trajectory yet")
        return 0
    key = NAME_KEY[scope]
    for r in rows:
        if r["kind"] == "seed":
            print(f"seed            dev_train F1 {r['train_f1']:.3f}")
        elif r["kind"] == "evaluate":
            print(f"iter {r['iteration']:02d} {r[key]:<22} F1 {r['train_f1']:.3f} "
                  f"({r['gain']:+.3f}) {'ACCEPTED' if r['accepted'] else 'rejected'}")
        else:
            print(f"{r['kind']:<15} {r.get(key) or ''}")
    best, f1 = current_definition(args.model, scope)
    print(f"current: {best or 'seed'} (dev_train F1 {f1:.3f})")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pick", cmd_pick), ("rewrite", cmd_rewrite), ("evaluate", cmd_evaluate),
                     ("alternative", cmd_alternative), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--model", required=True, choices=sorted(SHORT))
        p.add_argument("--scope", choices=SCOPES, default="definition",
                       help="What the critic rewrites: the class definition (Step 4, the "
                            "default) or the whole instruction (prompt276-* candidates under "
                            "prompts/, scored with --prompt-body, record under "
                            "optimize/<model>/prompt/).")
        p.add_argument("--iteration", type=int, help="Default: the next (pick) or the latest.")
        p.add_argument("--cache-dir", default=str(REPO_ROOT / ".model_cache"))
        p.add_argument("--usage-log", default=C.default_usage_log(),
                       help="The ledger (default: the main checkout's), as compare.py.")
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--force", action="store_true")
        if name == "pick":
            p.add_argument("--exclude", nargs="*", metavar="PANO",
                           help="Panos to skip this pick, beyond the already-used ones.")
            p.add_argument("--reuse-from", metavar="ITER_DIR",
                           help="With --scope prompt: seed this iteration from a judged "
                                "definition-scope iteration (e.g. optimize/<model>/iter_01, "
                                "relative to benchmark/prompt_opt_276/) instead of running "
                                "the detector and asking for new verdicts.")
        if name == "evaluate":
            p.add_argument("--floor", type=float, required=True,
                           help="Minimum dev_train F1 gain to accept: the seed-repeat spread "
                                "measured in Step 3, pre-registered before the first evaluate.")
        if name == "alternative":
            p.add_argument("--from-best", action="store_true",
                           help="Prompt scope only: reword the best accepted prompt body "
                                "instead of the seed (the default paraphrase control).")
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.usage_log == "none":
        args.usage_log = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
