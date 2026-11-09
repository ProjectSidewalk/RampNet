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
SHORT = {"gemini-4-argon": "argon", "gemini-3.8-flash": "flash38"}
VETO_TAGS = ("a", "u")                    # A = a real ramp the GT missed, U = unsure (RUBRICS §2)
BOX_HALF_PX = 80                          # the marker drawn around a point, in 1024-px view pixels
DETAIL_HALF_PX = 150

CRITIC_SCHEMA = {"type": "object", "properties": {"definition": {"type": "string"}},
                 "required": ["definition"]}
CRITIC_TEMPERATURE = 1.0                  # a rewrite wants variety; recorded in run_options


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def short_name(model):
    try:
        return SHORT[model]
    except KeyError:
        raise SystemExit(f"no short name for {model!r}; add it to prompt_optimize.SHORT")


def state_dir(model):
    return OPT_DIR / "optimize" / model


def trajectory_path(model):
    return state_dir(model) / "trajectory.jsonl"


def read_trajectory(model):
    p = trajectory_path(model)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def append_trajectory(model, row):
    p = trajectory_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def current_definition(model):
    """``(name, train_f1)`` of the best accepted candidate, or ``(None, seed_f1)``."""
    best, best_f1 = None, None
    for row in read_trajectory(model):
        if row.get("kind") == "seed":
            best_f1 = row["train_f1"]
        elif row.get("kind") == "evaluate" and row.get("accepted"):
            best, best_f1 = row["definition"], row["train_f1"]
    return best, best_f1


def used_panos(model):
    used = set()
    for row in read_trajectory(model):
        for k in ("fp", "fn"):
            if row.get(k) and row[k].get("pano"):
                used.add(row[k]["pano"])
    return used


def next_iteration(model):
    its = [row["iteration"] for row in read_trajectory(model) if "iteration" in row]
    return (max(its) + 1) if its else 1


def iter_dir(model, k):
    return state_dir(model) / f"iter_{k:02d}"


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


def compare_args(cache_dir, definition, confidence=True, concurrency=6, debug_dump=None):
    from fp_taxonomy import _compare_args
    ns = _compare_args(str(cache_dir))
    ns.definition = definition
    ns.gemini_confidence = True if confidence else None
    ns.gemini_concurrency = concurrency
    ns.gemini_debug_dump = debug_dump
    return ns


def run_detector(model, definition, pids, cache_dir, usage_log, concurrency=6, tag=""):
    """Score ``model`` with ``definition`` (None = seed) over ``pids`` of the dev split,
    through compare.score_model so the cache, the spend guard and the ledger row are
    the ordinary ones. Returns ``(points_by_pid, gts, report)``."""
    C.load_dotenv_for_run()
    bundle = str(REPO_ROOT / "benchmark" / CITY)
    records, verdicts, panos_dir = C.load_bundle(bundle)
    gts_all = C.ground_truths_from_verdicts(records, verdicts)
    gts = {p: gts_all[p] for p in pids}
    args = compare_args(cache_dir, definition, concurrency=concurrency)
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


def write_review_html(path, model, k, definition_text, fp, fn, diag, report):
    """A self-contained local page (never an Artifact): the two crops, the question,
    and a form that produces the verdicts.json to save beside it."""
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
<p>Current definition (dev_train F1 {report['f1']:.3f}, P {report['precision']:.3f}, R {report['recall']:.3f};
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

    def __init__(self, model_id, client, prompt):
        self.model_id = model_id
        self._client = client
        self.prompt = prompt
        self.usage = dict.fromkeys(D._VLMDetector.USAGE_KEYS, 0)
        self.model_versions = None
        self.stop_reasons = Counter()

    def signature(self):
        return {"provider": "gemini", "model_id": self.model_id, "role": "critic",
                "prompt": self.prompt, "schema": json.dumps(CRITIC_SCHEMA, sort_keys=True),
                "temperature": CRITIC_TEMPERATURE}

    def run_notes(self):
        return {"role": "critic", "temperature": CRITIC_TEMPERATURE, "concurrency": 1}

    def ask(self, images):
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model_id, contents=list(images) + [self.prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=CRITIC_SCHEMA,
                                               temperature=CRITIC_TEMPERATURE))
        # bookkeeping exactly as the detector does it
        D.GeminiDetector._record_usage(self, resp)
        text = getattr(resp, "text", None) or ""
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            raise SystemExit(f"critic returned non-JSON: {text[:300]!r}")
        definition = " ".join(str(out.get("definition", "")).split())
        if not definition:
            raise SystemExit(f"critic returned no definition: {text[:300]!r}")
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


def write_definition(name, text):
    DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFINITIONS_DIR / f"{name}.txt"
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


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_pick(args):
    model = args.model
    k = args.iteration or next_iteration(model)
    out = iter_dir(model, k)
    if (out / "candidates.json").exists() and not args.force:
        raise SystemExit(f"{out} already has candidates; use --force to redo the pick")
    out.mkdir(parents=True, exist_ok=True)
    name, _ = current_definition(model)
    definition_text = D.CURB_RAMP_DEFINITION if name is None else D.load_definition(name)
    pids = load_dev("dev_train")
    points, gts, _, panos_dir = run_detector(model, name, pids, args.cache_dir, args.usage_log,
                                             concurrency=args.concurrency)
    report = train_report(points, gts)
    if not read_trajectory(model):
        append_trajectory(model, {"kind": "seed", "model": model, "definition": None,
                                  "train_f1": report["f1"], "train": report, "at": _now()})
    used = used_panos(model) | set(args.exclude or [])
    fp, fn, diag = pick_exemplars(points, gts, load_veto_points(), used)
    for item, color, stem in ((fp, (220, 30, 30), "fp"), (fn, (30, 80, 220), "fn")):
        if item is None:
            continue
        meta = render_marked_view(os.path.join(panos_dir, f"{item['pano']}.jpg"),
                                  item["x"], item["y"], color, str(out / stem))
        item.update(meta)
        item["_image"], item["_detail"] = str(out / meta["image"]), str(out / meta["detail"])
    write_review_html(str(out / "review.html"), model, k, definition_text, fp, fn, diag, report)
    for item in (fp, fn):
        if item:
            item.pop("_image", None)
            item.pop("_detail", None)
    cand = {"iteration": k, "model": model, "definition": name, "train": report,
            "diagnostics_2R": diag, "fp": fp, "fn": fn, "picked_at": _now(),
            "excluded_panos": sorted(used)}
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
    model = args.model
    k = args.iteration or (next_iteration(model) - 1 if read_trajectory(model) else 1)
    out = iter_dir(model, k)
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
                                  "at": _now()})
        raise SystemExit("neither exemplar was confirmed; both panos are now excluded -- run "
                         "`pick` again for this iteration with --force")
    name = cand["definition"]
    definition_text = D.CURB_RAMP_DEFINITION if name is None else D.load_definition(name)
    prompt = critic_prompt(definition_text, use_fp, use_fn)
    from PIL import Image
    images = []
    if use_fp:
        images.append(Image.open(out / cand["fp"]["image"]).convert("RGB"))
    if use_fn:
        images.append(Image.open(out / cand["fn"]["image"]).convert("RGB"))
    C.load_dotenv_for_run()
    critic = Critic(model, _client_for(model), prompt)
    t0 = perf_counter()
    try:
        new_def, raw = critic.ask(images)
    finally:
        C.report_usage(critic, model, CITY, None, args.usage_log,
                       timing={"elapsed_s": round(perf_counter() - t0, 3), "detect_s": 0.0,
                               "panos_called": 0})
    cand_name = f"def276-{short_name(model)}-r{k}"
    path = write_definition(cand_name, new_def)
    with open(out / "rewrite.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"iteration": k, "model": model, "from": name, "candidate": cand_name,
                   "used_fp": use_fp, "used_fn": use_fn, "verdicts": verdicts,
                   "prompt": prompt, "raw": raw, "definition": new_def, "at": _now()},
                  f, indent=1)
    print(f"candidate {cand_name} -> {path}\n{new_def}")
    return 0


def cmd_evaluate(args):
    model = args.model
    k = args.iteration or (next_iteration(model) - 1 if read_trajectory(model) else 1)
    out = iter_dir(model, k)
    with open(out / "rewrite.json", encoding="utf-8") as f:
        rw = json.load(f)
    cand_name = rw["candidate"]
    best_name, best_f1 = current_definition(model)
    pids = load_dev("dev_train")
    points, gts, _, _ = run_detector(model, cand_name, pids, args.cache_dir, args.usage_log,
                                     concurrency=args.concurrency)
    report = train_report(points, gts)
    gain = report["f1"] - best_f1
    accepted = gain >= args.floor
    rows = read_trajectory(model)
    prev_rejections = 0
    for row in reversed(rows):
        if row.get("kind") == "evaluate":
            if row["accepted"]:
                break
            prev_rejections += 1
    with open(out / "candidates.json", encoding="utf-8") as f:
        cand = json.load(f)
    append_trajectory(model, {
        "kind": "evaluate", "iteration": k, "model": model, "definition": cand_name,
        "from": best_name, "train": report, "train_f1": report["f1"], "gain": round(gain, 4),
        "floor": args.floor, "accepted": accepted, "fp": cand["fp"], "fn": cand["fn"],
        "verdicts": rw["verdicts"], "at": _now()})
    verdict = "ACCEPTED" if accepted else "rejected"
    print(f"{cand_name}: dev_train F1 {report['f1']:.3f} vs {best_f1:.3f} "
          f"({gain:+.3f}, floor {args.floor}) -> {verdict}")
    if not accepted and prev_rejections + 1 >= 2:
        print("two consecutive rejections: the pre-registered early stop. Run `alternative`, "
              "then select on dev_select.")
    return 0


def cmd_alternative(args):
    model = args.model
    best_name, _ = current_definition(model)
    text = D.CURB_RAMP_DEFINITION if best_name is None else D.load_definition(best_name)
    C.load_dotenv_for_run()
    critic = Critic(model, _client_for(model), alternative_prompt(text))
    t0 = perf_counter()
    try:
        new_def, raw = critic.ask([])
    finally:
        C.report_usage(critic, model, CITY, None, args.usage_log,
                       timing={"elapsed_s": round(perf_counter() - t0, 3), "detect_s": 0.0,
                               "panos_called": 0})
    cand_name = f"def276-{short_name(model)}-alt"
    path = write_definition(cand_name, new_def)
    append_trajectory(model, {"kind": "alternative", "model": model, "from": best_name,
                              "definition": cand_name, "raw": raw, "at": _now()})
    print(f"alternative {cand_name} -> {path}\n{new_def}")
    return 0


def cmd_status(args):
    rows = read_trajectory(args.model)
    if not rows:
        print("no trajectory yet")
        return 0
    for r in rows:
        if r["kind"] == "seed":
            print(f"seed            dev_train F1 {r['train_f1']:.3f}")
        elif r["kind"] == "evaluate":
            print(f"iter {r['iteration']:02d} {r['definition']:<22} F1 {r['train_f1']:.3f} "
                  f"({r['gain']:+.3f}) {'ACCEPTED' if r['accepted'] else 'rejected'}")
        else:
            print(f"{r['kind']:<15} {r.get('definition') or ''}")
    best, f1 = current_definition(args.model)
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
        p.add_argument("--iteration", type=int, help="Default: the next (pick) or the latest.")
        p.add_argument("--cache-dir", default=str(REPO_ROOT / ".model_cache"))
        p.add_argument("--usage-log", default=C.default_usage_log(),
                       help="The ledger (default: the main checkout's), as compare.py.")
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--force", action="store_true")
        if name == "pick":
            p.add_argument("--exclude", nargs="*", metavar="PANO",
                           help="Panos to skip this pick, beyond the already-used ones.")
        if name == "evaluate":
            p.add_argument("--floor", type=float, required=True,
                           help="Minimum dev_train F1 gain to accept: the seed-repeat spread "
                                "measured in Step 3, pre-registered before the first evaluate.")
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
