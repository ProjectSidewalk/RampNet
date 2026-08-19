"""Compare curb-ramp detectors on a benchmark bundle.

Scores each selected model against the same model-agnostic ground truth (see
rampnet/detection_eval.py), so RampNet and the VLMs are compared on equal
footing. City bundles derive that GT from the human review of RampNet's
detections (verdicts.json; RampNet's verdict-based numbers are re-printed as a
cross-check); a manual-GT bundle (benchmark/manual_gold, issue #58) loads it
from independently-labeled YOLO points instead — bigger, and free of RampNet
anchoring.

    python scripts/model_comparison/compare.py benchmark/richmond --models rampnet
    python scripts/model_comparison/compare.py benchmark/richmond \
        --models rampnet,gemini:gemini-2.5-flash,gemini:gemini-3.6-flash
    python scripts/model_comparison/compare.py benchmark/manual_gold --models rampnet,gemini

Each --models token is a provider (rampnet/gemini/claude/qwen/owlv2/gdino/molmo/
vistas/yolo — the roster is detectors.PROVIDERS) or provider:model_id to pin a variant,
so several models from the same provider compare side by side. Detectors that emit calibrated scores (RampNet, OWLv2,
Grounding DINO, Vistas, YOLO) additionally get AP, a PR curve (--pr-out) and a
threshold sweep (--sweep); chat VLMs have no score to rank by, so they get one
operating point. See docs/model_comparison.md.

The supervised YOLO baseline (--models yolo:<best.pt>) is evaluated under the
pre-registered checkpoint-selection & eval protocol in
scripts/model_comparison/yolo_baseline/README.md (issue #71): checkpoint and
config chosen on val only, headline F1 at conf 0.25, test bundles touched once.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))          # rampnet.* (editable install fallback)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # local detectors.py

from rampnet import ledger, roster  # noqa: E402
from rampnet.detection_eval import (  # noqa: E402
    build_ground_truth, load_yolo_ground_truths, score_pano, aggregate,
    prediction_confidence, radius_sq_for, PANO_RADIUS_NORMALIZED,
)
from rampnet.validation import collect, format_report  # noqa: E402
from detectors import (  # noqa: E402
    GDINO_QUERY, OWLV2_QUERY, PROVIDERS, PanoSample, build_detector, parse_model_spec,
)
from pricing import estimate_cost, price_for  # noqa: E402

# Where the spend record goes by default. NOT inside --cache-dir: .model_cache/ is
# gitignored, and a cost ledger that lands there is lost to every other clone —
# the exact problem export_model_cache.py exists to undo for the detections.
# analysis_out/ is the established home for committed derived artifacts
# (op_cache/, fp_taxonomy.json, silent_witness.json), and .gitignore re-includes
# this file by name.
DEFAULT_USAGE_LOG_REL = os.path.join("analysis_out", "usage_log.jsonl")


def canonical_repo_root(start=None):
    """The MAIN checkout, even when this is running from a linked worktree.

    ``REPO_ROOT`` is this file's own checkout, so from a worktree the default
    ledger lands *inside that worktree* -- and scratch worktrees get deleted,
    taking the ledger with them. That is not hypothetical: the #139
    claude-opus-5 leg spent **$70.41 and left no row**, recovered only because
    Cloud Monitoring still had it inside its ~6-week window (#143).

    The #119 guard cannot see this failure. It proves a log path was *accepted*,
    not that the file it wrote still exists -- the operator followed the rule and
    lost the record anyway. See rampnet/ledger.py, which both ledgers share."""
    return ledger.canonical_repo_root(start or REPO_ROOT)


def default_usage_log():
    return str(canonical_repo_root() / DEFAULT_USAGE_LOG_REL)


# Generated from detectors.PROVIDERS so the roster cannot drift out of the help
# text again (`claude` shipped working but undocumented in three places).
MODELS_HELP = (
    "Comma-separated detectors. Each is a provider ("
    + "/".join(PROVIDERS) +
    ", using its default model) or provider:model_id to pin a variant, e.g. "
    "'rampnet,gemini:gemini-2.5-flash,owlv2'. yolo needs trained weights: "
    "'yolo:<path.pt>' or --yolo-model."
)

def load_dotenv(root):
    """Load KEY=VALUE lines from a repo-root .env into os.environ (without
    overriding already-set vars), so a Gemini key can live in a git-ignored file
    instead of the shell/transcript. Minimal parser — no python-dotenv dependency."""
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def cache_key(label, signature, city, pano_id):
    """Stable hash over everything that determines a detector's output for one
    pano, so re-runs reuse cached detections and don't re-pay the API."""
    blob = json.dumps({"label": label, "sig": signature, "city": city, "pid": pano_id},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class DetectionCache:
    """On-disk cache of per-pano detection points (a paid VLM call is expensive;
    scoring/radius changes are free, so we cache the detector output, not the
    score). Sharded by key prefix. A no-op when disabled."""

    def __init__(self, root, enabled=True):
        self.root = root
        self.enabled = enabled

    def _path(self, key):
        return os.path.join(self.root, key[:2], f"{key}.json")

    def get(self, key):
        if not self.enabled:
            return None
        path = self._path(key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["points"]
        return None

    def put(self, key, points):
        if not self.enabled:
            return
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"points": [list(p) for p in points]}, f)


def load_bundle(bundle_dir):
    """Return (records_by_pid, verdicts_panos, panos_dir) for a benchmark bundle.

    ``verdicts_panos`` is None for a manual-GT bundle (``gt_source.json`` instead
    of ``verdicts.json`` — see ``load_manual_ground_truths``); the city bundles
    always carry a verdict review. A directory with neither is rejected here so a
    mistyped path fails with one clear message instead of a downstream KeyError.
    """
    records = {}
    with open(os.path.join(bundle_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["pano"]["panorama_id"]] = r
    vpath = os.path.join(bundle_dir, "verdicts.json")
    verdicts = None
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            verdicts = json.load(f)["panos"]
    elif not os.path.exists(os.path.join(bundle_dir, "gt_source.json")):
        raise SystemExit(f"{bundle_dir}: neither verdicts.json nor gt_source.json — "
                         "not a benchmark bundle")
    return records, verdicts, os.path.join(bundle_dir, "panos")


def ground_truths_from_verdicts(records, verdicts):
    """{pid: GroundTruth} derived from a bundle's human review (the city path)."""
    return {pid: build_ground_truth(records[pid]["detections"], entry["dets"],
                                    entry["missed"], entry["no_missed"])
            for pid, entry in verdicts.items()}


def load_manual_ground_truths(bundle_dir):
    """{pid: GroundTruth} for a manual-GT bundle (``benchmark/manual_gold``).

    The bundle's ``gt_source.json`` points at a directory of YOLO-format label
    files that were produced by independent manual labeling — no RampNet review
    to derive from, hence no verdicts and no RampNet anchoring. Box centers
    become GT points, there are no ignore points, and every pano is
    recall-confirmed (see ``rampnet.detection_eval.yolo_ground_truth``).
    """
    with open(os.path.join(bundle_dir, "gt_source.json"), encoding="utf-8") as f:
        src = json.load(f)
    if src.get("format") != "yolo_points":
        raise SystemExit(f"{bundle_dir}/gt_source.json: unsupported format "
                         f"{src.get('format')!r} (expected 'yolo_points')")
    labels_dir = os.path.normpath(os.path.join(bundle_dir, src["labels_dir"]))
    gts = load_yolo_ground_truths(labels_dir)
    if not gts:
        raise SystemExit(f"{labels_dir}: no .txt label files found")
    return gts


def validate_bundle(records, verdicts):
    """Fail fast on a structurally broken bundle, *before* any (paid) detector call.

    ``score_model`` builds each pano's ground truth from ``records[pid]`` + the
    verdict entry outside its per-pano failure guard (that guard is for transient
    detect() errors, not data integrity). Without this pre-flight a reviewed pano
    missing from records.jsonl, a missing verdict field, or detections/verdicts
    that don't line up would surface as a raw KeyError/ValueError partway through a
    long VLM run — after spend, and aborting models already scored. Catch it here
    with a clear message instead. Raises SystemExit listing every offending pano.

    (Legacy verdicts.json without ``no_missed`` are intentionally rejected here
    rather than silently defaulted — the current/planned bundles are new-schema;
    see docs/model_comparison.md.)"""
    problems = []
    for pid, entry in verdicts.items():
        rec = records.get(pid)
        if rec is None:
            problems.append(f"{pid}: reviewed in verdicts.json but absent from records.jsonl")
            continue
        missing = [k for k in ("dets", "missed", "no_missed") if k not in entry]
        if missing:
            problems.append(f"{pid}: verdict entry missing field(s) {missing}")
            continue
        n_det, n_ver = len(rec.get("detections", [])), len(entry["dets"])
        if n_det != n_ver:
            problems.append(f"{pid}: {n_det} detections vs {n_ver} verdicts (misaligned)")
    if problems:
        _fail_validation(problems)


def validate_manual_bundle(records, gts, need_detections=False):
    """Pre-flight for a manual-GT bundle, mirroring ``validate_bundle``'s job.

    The label files and ``records.jsonl`` are built by different tools
    (``manual_labels/`` is hand-curated; records come from ``fetch_manual_gold`` +
    ``export_gold_records``), so catch any drift between them before a paid
    detector call. ``need_detections`` is set when the rampnet baseline was
    requested: its detections live in the records, and a bundle whose exporter
    hasn't run yet must say so instead of scoring RampNet as all-misses.
    """
    problems = []
    for pid in gts:
        rec = records.get(pid)
        if rec is None:
            problems.append(f"{pid}: labeled but absent from records.jsonl "
                            "(re-run scripts/fetch_manual_gold.py)")
        elif need_detections and "detections" not in rec:
            problems.append(f"{pid}: no RampNet detections in records.jsonl "
                            "(run scripts/export_gold_records.py first)")
    for pid in records:
        if pid not in gts:
            problems.append(f"{pid}: in records.jsonl but has no label file")
    if problems:
        _fail_validation(problems)


def _fail_validation(problems):
    shown = "\n  ".join(problems[:10])
    more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
    raise SystemExit(f"Bundle validation failed ({len(problems)} pano(s)):\n  {shown}{more}")


# scored: [(pred_points, GroundTruth)] for every pano that was successfully
# scored, kept so the threshold sweep and PR curves can re-score from memory
# instead of re-running the detector.
ModelRun = namedtuple("ModelRun", ["report", "failures", "scored"])


class UnrecordedSpend(Exception):
    """A paid leg was about to make its first real call with no usage log."""


def score_model(detector, records, gts, panos_dir, radius_sq, label, city, cache,
                max_consecutive_failures=10, spend_needs_recording=False,
                timing=None):
    """Run one detector over every scored pano and aggregate the score.

    ``gts`` maps pano id -> GroundTruth (verdict-derived for city bundles,
    label-derived for a manual bundle — see ``ground_truths_from_verdicts`` /
    ``load_manual_ground_truths``); it decides which panos are scored.

    Returns a ``ModelRun``. ``detector.prepare()`` runs before the pano loop
    (outside the per-pano guard) so credential / dependency / not-wired errors
    propagate to the caller and skip the whole model — but it is skipped entirely
    when every pano is already cached, so a ``.model_cache`` copied back from a GPU
    cluster scores on a machine that can't load the model at all. Each pano's
    detections are cached, so re-runs don't re-pay the API. A transient per-pano
    failure is recorded and skipped rather than crashing the run;
    ``max_consecutive_failures`` aborts the model early during an outage instead of
    burning budget.

    ``timing``, when given, is filled in as the run proceeds: model-load seconds,
    inference seconds, and how many panos actually reached the model. It is a
    caller-owned dict rather than a return value on purpose -- a leg that dies
    partway has still spent the time, and the caller reads it from a ``finally``
    where there is no ModelRun to read (#143)."""
    if timing is None:
        timing = {}
    # panos_called counts ATTEMPTS, not successes: a call that raised still spent
    # its wall-clock and, on a paid provider, still billed. It is the honest
    # denominator for both seconds-per-pano and dollars-per-pano.
    timing.setdefault("load_s", 0.0)
    timing.setdefault("detect_s", 0.0)
    timing.setdefault("panos_called", 0)
    sig = detector.signature() if hasattr(detector, "signature") else None
    keys = {pid: (cache_key(label, sig, city, pid) if sig is not None else None)
            for pid in gts}
    cached = {pid: (cache.get(k) if k else None) for pid, k in keys.items()}
    if not cached or any(p is None for p in cached.values()):
        # This leg WILL call the API. If it is a paid one and nothing is recording
        # the spend, stop here: token counts are the one artifact that cannot be
        # back-filled -- a later re-run reads the cache, makes zero calls, and can
        # never reproduce them, which is how the four Claude legs (#122) cost
        # $28.82 with no committed record. Checked here rather than at parse time
        # so a fully cached re-score, which cannot spend anything, still runs.
        if spend_needs_recording and getattr(detector, "name", None) in roster.PAID_PROVIDERS:
            raise UnrecordedSpend(
                f"{label} would call a paid API for "
                f"{sum(1 for p in cached.values() if p is None)} uncached pano(s) with "
                f"--usage-log none. Token counts cannot be back-filled, so this spend "
                f"would go unrecorded. Drop --usage-log none, or pass "
                f"--allow-unrecorded-spend deliberately.")
        t_load = perf_counter()
        detector.prepare()
        timing["load_s"] = round(perf_counter() - t_load, 3)
    else:
        print(f"[{label}] all {len(cached)} panos already cached; model load skipped")
    pano_scores, failures, consecutive, scored = [], [], 0, []
    for pid, gt in gts.items():
        rec = records[pid]
        key, preds = keys[pid], cached[pid]
        if preds is None:
            sample = PanoSample(
                pano_id=pid,
                image_path=os.path.join(panos_dir, f"{pid}.jpg"),
                width=rec["pano"].get("width"),
                height=rec["pano"].get("height"),
                meta=rec["pano"],
            )
            t_pano = perf_counter()
            try:
                preds = detector.detect(sample)
            except Exception as e:  # transient API/network failure: isolate this pano
                failures.append((pid, f"{type(e).__name__}: {str(e)[:120]}"))
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    failures.append(("<abort>", f"{consecutive} consecutive failures; stopped early"))
                    break
                continue
            finally:
                # In a finally so the `continue` and `break` paths above are
                # counted too: a failed call is spent time and spent money.
                timing["detect_s"] += perf_counter() - t_pano
                timing["panos_called"] += 1
            consecutive = 0
            if key:
                cache.put(key, preds)
        scored.append((preds, gt))
        pano_scores.append(score_pano(preds, gt, radius_sq=radius_sq))
    timing["detect_s"] = round(timing["detect_s"], 3)
    return ModelRun(aggregate(pano_scores), failures, scored)


def hardware_note():
    """What this run had to compute with -- best effort, and cheap.

    Reports GPUs only when torch is ALREADY imported. A hosted-API leg must not
    pay a multi-second torch import for bookkeeping, and "torch was never
    loaded" is itself the honest answer: this harness ran no local model.

    The GPU list is what the MACHINE had, not proof this leg used it --
    ``device_map="auto"`` and the CPU fallbacks in detectors.py both exist. Read
    it as the run's environment, and the .slurm job record (compute_log.jsonl)
    as what was actually allocated."""
    note = {"host": socket.gethostname()}
    torch = sys.modules.get("torch")
    if torch is None:
        return note
    try:
        if torch.cuda.is_available():
            note["gpus"] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
    except Exception:
        pass  # a torch that cannot answer must not take down the cost record
    return note


def report_usage(detector, label, city, panos_scored, usage_log_path, timing=None):
    """Print and (append-)log what one leg cost: time always, tokens and dollars
    when the provider bills for them.

    Standing rule (#143): every experiment on a non-free model or non-free
    compute records BOTH time and money, at the time it runs, so a paper can
    report cost alongside accuracy. Neither half is back-fillable -- a re-run
    reads the detection cache, makes no calls, and returns in seconds, so it can
    reproduce neither the token counts nor the runtime.

    Free local models bill no tokens but cost real GPU-hours, so they get a row
    too. Before #143 they returned early here and the whole GPU half of the
    roster -- OWLv2, Grounding DINO, Qwen, Molmo, YOLO, RampNet itself -- left no
    record at all; their runtimes survive only as prose in docs/model_comparison.md.

    A leg that spent nothing still writes nothing: a fully cached re-score (model
    load skipped, zero calls) and a leg that never became runnable both have no
    cost to record, and a zero row per re-run would bury the rows that carry a
    measurement.

    Called from a ``finally``, so it must survive being handed a half-finished
    run: a leg that dies or is interrupted after paying for 400 calls is exactly
    the one whose spend must not vanish. It never raises for that reason --
    bookkeeping that can kill the run it is bookkeeping is worse than no
    bookkeeping."""
    timing = dict(timing or {})
    usage = getattr(detector, "usage", None) or {}
    calls = usage.get("calls", 0)
    panos_called = timing.get("panos_called", 0)
    if not calls and not panos_called:
        return
    if not calls and getattr(detector, "replays_committed_detections", False):
        # Reading detections back out of the bundle is not an experiment that cost
        # anything. Logging it would put a zero row in a committed, append-only
        # file every time anyone re-scores the rampnet arm, which is most runs.
        return
    model_id = getattr(detector, "model_id", None)
    cost = pricing = None
    if calls:
        cost = estimate_cost(model_id, usage["input_tokens"], usage["output_tokens"],
                             cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                             cache_write_tokens=usage.get("cache_write_input_tokens", 0))
        pricing = price_for(model_id)
        cost_txt = (f"  ~=${cost:.2f} (pricing as of {pricing['as_of']})" if cost is not None
                    else "  (no verified price in pricing.py — record one)")
        # .get() on thinking: most providers don't report it, and a KeyError here
        # would take down a run that has already been paid for.
        print(f"[{label}] API usage this run: {usage['calls']} calls, "
              f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out tokens "
              f"({usage.get('thoughts_tokens', 0):,} thinking){cost_txt}")
    # Time, for every leg. The per-pano figure divides by panos actually put
    # through the model, never by the bundle: on a resumed run most panos come
    # from the cache and cost nothing, so len(gts) would understate it severalfold.
    detect_s, load_s = timing.get("detect_s") or 0.0, timing.get("load_s") or 0.0
    s_per_pano = (detect_s / panos_called) if panos_called else None
    elapsed = timing.get("elapsed_s")
    if elapsed is not None:
        per_txt = f", {s_per_pano:.2f} s/pano" if s_per_pano is not None else ""
        print(f"[{label}] time this run: {elapsed:,.1f}s wall  "
              f"({load_s:,.1f}s model load, {detect_s:,.1f}s inference over "
              f"{panos_called:,} pano(s){per_txt})")
    # Which BUILD answered, as distinct from the alias we asked for (#121).
    versions = getattr(detector, "model_versions", None)
    if versions:
        served = ", ".join(f"{v} ({n:,} calls)" for v, n in sorted(versions.items()))
        print(f"[{label}] served by: {served}")
    # How the calls ENDED, when the provider reports it. A leg with refusals or
    # truncations is a different measurement from one without, and this is the
    # one place a reader checking the numbers will see it.
    stop_reasons = dict(getattr(detector, "stop_reasons", None) or {})
    abnormal = {k: v for k, v in stop_reasons.items()
                if k not in ("end_turn", "tool_use", "stop_sequence")}
    if abnormal:
        print(f"[{label}] ABNORMAL stop reasons: "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(abnormal.items())))
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": city,
        "label": label,
        "provider": getattr(detector, "name", None),
        "model_id": model_id,
        # Whether this leg cost API money. Free legs cost GPU time instead and
        # carry no token keys at all, so this is the unambiguous filter rather
        # than testing for a key's absence.
        "paid": bool(calls),
        # The build(s) that actually served this run, against the alias above.
        # Cannot be reconstructed later -- .model_cache stores points only -- so
        # a run that didn't record it never can (#121).
        "model_versions": dict(versions) if versions else None,
        # Panos this run actually scored, NOT the size of the bundle: on a
        # resumed run most panos come from the cache and cost nothing, so
        # len(gts) would understate cost-per-pano several-fold.
        "panos_scored": panos_scored,
        # ...and panos this run actually put through the model. The two differ by
        # the cache hits, which is exactly the difference between "what this leg
        # measured" and "what this leg reported on".
        "panos_called": panos_called,
        "elapsed_s": elapsed,
        "load_s": round(load_s, 3),
        "detect_s": round(detect_s, 3),
        "s_per_pano": round(s_per_pano, 4) if s_per_pano is not None else None,
        # The machine, so a runtime can be read against the hardware that produced
        # it. A GPU figure is meaningless without it.
        "hardware": hardware_note(),
        # The rig is part of the price: the same model on the same bundle costs
        # ~6x more tiled than whole-pano, and without this two such runs log
        # identically. Same reasoning as the detections export (9e87290).
        "signature": detector.signature() if hasattr(detector, "signature") else None,
        # How each call terminated. Empty for providers that report nothing.
        "stop_reasons": stop_reasons or None,
        **usage,
        "est_cost_usd": round(cost, 4) if cost is not None else None,
        "pricing": pricing,
    }
    if not usage_log_path:
        # The measurement is the one artifact that cannot be back-filled: a
        # re-run reads the detection cache, makes zero calls, and so can never
        # reproduce it. For a paid leg that means the spend is gone for good,
        # which is exactly what happened to #123's four Claude legs ($28.82, no
        # record); for a free one it means the runtime is.
        what = "spent money and its usage was" if calls else "spent GPU time and its runtime was"
        print(f"[{label}] WARNING: this leg {what} NOT recorded — --usage-log is "
              f"disabled. It cannot be back-filled from a later run (a cached "
              f"re-run makes no calls), so this measurement is unrecoverable. The "
              f"standing rule is that every non-free experiment records what it "
              f"cost, in time and money, at the time it runs.\n"
              f"[{label}] usage record: {json.dumps(rec)}")
        return
    try:
        os.makedirs(os.path.dirname(usage_log_path) or ".", exist_ok=True)
        # newline="" so a Windows run appends LF, not CRLF. This ledger is
        # append-only and byte-compared in review; a CRLF line silently breaks that
        # (the same defect imagery_manifest.py was fixed for).
        with open(usage_log_path, "a", encoding="utf-8", newline="") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        # Print the record so the numbers survive in the run log even when the
        # file can't be written; never let this abort the comparison.
        print(f"[{label}] WARNING: could not write {usage_log_path}: "
              f"{type(e).__name__}: {e}\n[{label}] usage record: {json.dumps(rec)}")
        return
    # Absolute, and with the running total: a leg that logged somewhere
    # unexpected has to be visible now, not six weeks later when the provider's
    # usage telemetry has aged out and the number is gone at any price (#143).
    print(f"[{label}] usage logged to {os.path.abspath(usage_log_path)}")
    totals = ledger.ledger_totals(usage_log_path)
    if totals:
        rows, usd, hours = totals
        print(f"[{label}] ledger now: {rows:,} rows, ${usd:,.2f}, {hours:,.1f} h")
def rescore(scored, radius_sq, min_confidence=0.0):
    """Re-aggregate a finished run with predictions below ``min_confidence`` dropped.

    Detections are cached with their scores, so every operating point of a
    confidence-carrying detector is a free re-score — no second model run. A
    prediction with no confidence (chat VLMs) is never dropped: there is nothing to
    threshold on."""
    return aggregate([
        score_pano([p for p in preds
                    if prediction_confidence(p) is None
                    or prediction_confidence(p) >= min_confidence],
                   gt, radius_sq=radius_sq)
        for preds, gt in scored])


def operating_report(report, scored, radius_sq, op_threshold):
    """The table row for one model: P/R/F1/counts at the operating threshold, but
    AP and the PR curve kept from the full-range ``report``. Those two are
    integrals over the whole confidence range — rescore()'s filtered aggregate
    would silently truncate them at the operating point, exactly the caveat the
    manual_gold bundle's 0.05 export floor exists to avoid."""
    if op_threshold <= 0:
        return report
    return rescore(scored, radius_sq, op_threshold)._replace(
        ap=report.ap, pr_curve=report.pr_curve)


def has_confidences(scored):
    """True when every prediction in the run carries a score (so AP / a sweep mean
    something). An empty run counts as no confidences."""
    preds = [p for ps, _ in scored for p in ps]
    return bool(preds) and all(prediction_confidence(p) is not None for p in preds)


SWEEP_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def sweep_rows(scored, radius_sq, thresholds=SWEEP_THRESHOLDS, floor=None):
    """(threshold, ScoreReport) for each threshold that still keeps a prediction.

    ``floor`` is the detector's cache floor (--score-threshold): the cache holds
    no detections below it, so a sweep row under the floor would silently repeat
    the floor row while reading as a real measurement. Those rows are dropped
    (with the default floor, 0.05, nothing is — it equals the lowest threshold)."""
    top = max((prediction_confidence(p) for ps, _ in scored for p in ps
               if prediction_confidence(p) is not None),
              default=0.0)
    return [(t, rescore(scored, radius_sq, t)) for t in thresholds
            if t <= top and (floor is None or t >= floor)]


def _pct(x):
    return f"{x:.3f}"


def _ci(lo_hi):
    return f"({lo_hi[0]:.3f}-{lo_hi[1]:.3f})"


def _ap(r):
    # Blank for chat VLMs: no calibrated per-box score, so no curve to integrate.
    return f"{r.ap:.3f}" if r.ap is not None else "  -  "


def print_table(rows):
    # Wide enough for the longest HF id in play (google/owlv2-large-patch14-ensemble).
    header = (f"{'model':<36} {'P':>6} {'95% CI':>15} {'R':>6} {'95% CI':>15} "
              f"{'F1':>6} {'AP':>6}   {'tp/fp/fn/ign':>16}")
    print(header)
    print("-" * len(header))
    for name, r in rows:
        counts = f"{r.tp}/{r.fp}/{r.fn}/{r.ignored}"
        print(f"{name:<36} {_pct(r.precision):>6} {_ci(r.precision_ci):>15} "
              f"{_pct(r.recall):>6} {_ci(r.recall_ci):>15} {_pct(r.f1):>6} {_ap(r):>6}   "
              f"{counts:>16}")


def print_sweep(label, rows):
    """Threshold sweep for one model: what tuning the score cutoff buys.

    This is the point of a real detector over a chat VLM — the recall-first
    direction needs a knob, and a model pinned at one operating point has none.
    The best-F1 row is flagged; it is chosen *on the benchmark itself*, so it is an
    optimistic, tune-on-test number and must be quoted as such."""
    if not rows:
        # has_confidences was True, so the model did detect — every score just
        # sits below the sweepable range. Say so instead of printing nothing.
        print(f"\n[{label}] threshold sweep: no detections at or above the lowest "
              "sweepable threshold; nothing to sweep")
        return
    best = max(range(len(rows)), key=lambda i: rows[i][1].f1)
    print(f"\n[{label}] threshold sweep (re-scored from cached detections)")
    print(f"  {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}   {'tp/fp/fn':>14}")
    for i, (t, r) in enumerate(rows):
        mark = " <- best F1" if i == best else ""
        print(f"  {t:>5.2f} {_pct(r.precision):>6} {_pct(r.recall):>6} {_pct(r.f1):>6}   "
              f"{f'{r.tp}/{r.fp}/{r.fn}':>14}{mark}")


def write_pr_curves(out_dir, curves):
    """Write each model's PR curve to JSON, and a combined PNG if matplotlib is
    around (it is not a harness dependency). ``curves``: [(label, ScoreReport)]."""
    os.makedirs(out_dir, exist_ok=True)
    for label, r in curves:
        recalls, precisions = r.pr_curve
        safe = label.replace("/", "_")
        with open(os.path.join(out_dir, f"pr_{safe}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": label, "ap": r.ap, "n_gt": r.n_gt_recall,
                       "recalls": recalls, "precisions": precisions}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"PR curves written to {out_dir} (JSON only; matplotlib not installed)")
        return
    plt.figure(figsize=(7, 6))
    for label, r in curves:
        recalls, precisions = r.pr_curve
        plt.plot(recalls, precisions, marker=".", markersize=3,
                 label=f"{label} (AP {r.ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curb-ramp detection PR curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    path = os.path.join(out_dir, "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curves written to {out_dir} (JSON + {os.path.basename(path)})")


def build_parser():
    """The CLI, as one object a test can inspect.

    Extracted from main() so that the coupling between PROVIDER_DEFAULTS and the
    flags that feed build_detector is checkable rather than asserted in prose:
    a drift there does not crash, it changes the detection cache key and silently
    misses every already-paid detection. test_roster.py reads the defaults off
    this parser.
    """
    ap = argparse.ArgumentParser(description="Compare curb-ramp detectors on a benchmark bundle.")
    ap.add_argument("bundle", help="Bundle dir (e.g. benchmark/richmond) with records.jsonl + verdicts.json.")
    ap.add_argument("--models", default="rampnet",
                    help=MODELS_HELP)
    # Every provider default below comes from rampnet.roster.PROVIDER_DEFAULTS. They
    # feed build_detector and so the detection signature and cache key, and used to
    # be copied into four separate parsers; a copy that drifts does not crash, it
    # changes the cache key and silently misses every already-paid detection.
    _D = roster.PROVIDER_DEFAULTS
    ap.add_argument("--gemini-model", default=_D["gemini_model"])
    ap.add_argument("--claude-model", default=_D["claude_model"],
                    help="Claude model id, served via Vertex + ADC (same credentials "
                         "as the Gemini legs). Each model must be enabled separately "
                         "in Vertex Model Garden.")
    ap.add_argument("--claude-effort", default=_D["claude_effort"],
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="How much Claude thinks. Thinking bills as OUTPUT and is the "
                         "dominant cost term, so this is the main cost lever; it is "
                         "part of the detection cache key. Default 'low' — reading a "
                         "view and emitting a box list is not intelligence-sensitive.")
    ap.add_argument("--claude-tool-choice", default=_D["claude_tool_choice"],
                    choices=["auto", "forced"],
                    help="'forced' guarantees the answer arrives as a tool call, but "
                         "SUPPRESSES THINKING, which makes --claude-effort inert. "
                         "'auto' (default) lets effort actually do something. Also "
                         "part of the cache key.")
    ap.add_argument("--claude-image-format", default=_D["claude_image_format"], choices=["jpeg", "png"],
                    help="How each view is encoded before it is sent. Default 'jpeg' "
                         "(q90) is what the published annapolis legs ran; 'png' is "
                         "lossless and matches what the Gemini leg receives, so it "
                         "removes an input asymmetry between the two paid legs. "
                         "Costs no extra tokens, but it IS a cache-key change: "
                         "switching means re-paying for the detections.")
    ap.add_argument("--claude-temperature", type=float, default=_D["claude_temperature"],
                    help="Sampling temperature. Default: send none and take the "
                         "provider default, which is what the published legs did. "
                         "Pass 0.0 to match GeminiDetector's greedy decoding. Also "
                         "a cache-key change.")
    ap.add_argument("--qwen-model", default=_D["qwen_model"])
    ap.add_argument("--qwen-coord-space", choices=["auto", "norm1000", "pixels"],
                    default=_D["qwen_coord_space"],
                    help="Box convention the Qwen checkpoint emits: 'norm1000' (Qwen3-VL, "
                         "0-1000) or 'pixels' (Qwen2/2.5-VL, absolute). 'auto' infers it "
                         "from the model id.")
    ap.add_argument("--owlv2-model", default=_D["owlv2_model"])
    ap.add_argument("--gdino-model", default=_D["gdino_model"])
    ap.add_argument("--molmo-model", default=_D["molmo_model"])
    ap.add_argument("--owlv2-query", help=f"OWLv2 text query (default {OWLV2_QUERY!r}).")
    ap.add_argument("--gdino-query", help=f"Grounding DINO category text (default {GDINO_QUERY!r}); "
                                          "lowercase, period-terminated.")
    ap.add_argument("--gdino-text-threshold", type=float,
                    help="Grounding DINO token-alignment threshold (default 0.2).")
    ap.add_argument("--score-threshold", type=float,
                    help="Score floor for the open-vocabulary detectors (owlv2/gdino), default "
                         "0.05. This is a CACHE floor, not the operating point: it is part of "
                         "the detector signature, so lowering it re-runs the model, while every "
                         "higher operating point is a free re-score (--op-threshold, --sweep).")
    ap.add_argument("--molmo-coord-scale", choices=["auto", "100", "1000"],
                    default=_D["molmo_coord_scale"],
                    help="Divisor for Molmo point coordinates: Molmo 1 emits percentages "
                         "(100), Molmo 2 emits 0-1000. 'auto' infers it from the tag syntax.")
    ap.add_argument("--yolo-model",
                    help="Trained YOLO weights (.pt) for the 'yolo' provider — the supervised "
                         "baseline (issue #51). Required for --models yolo; e.g. "
                         "runs/detect/train/weights/best.pt. Also settable as 'yolo:<path>'.")
    ap.add_argument("--yolo-conf", type=float, default=_D["yolo_conf"],
                    help="Score floor for YOLO boxes (default 0.05). Like --score-threshold for "
                         "the open-vocab detectors, this is a CACHE floor in the signature, not "
                         "the operating point; higher points are free re-scores (--op-threshold, "
                         "--sweep).")
    ap.add_argument("--yolo-iou", type=float, default=_D["yolo_iou"],
                    help="YOLO NMS IoU threshold (default 0.5).")
    ap.add_argument("--yolo-imgsz", type=int, default=_D["yolo_imgsz"],
                    help="YOLO inference image size (default 1024, matching the perspective view "
                         "size). For --tiling none, set this to the pano-geometry training size.")
    ap.add_argument("--vistas-class-set", default=_D["vistas_class_set"],
                    help="Which Vistas classes are read out as curb ramps, when the "
                         "--models spec does not say ('vistas' rather than "
                         "'vistas:curb-cut'). Part of the detection signature: the "
                         "arm IS its class set.")
    ap.add_argument("--vistas-input-size", type=int, nargs=2, metavar=("H", "W"),
                    default=None,
                    help="Override what the model actually sees. The checkpoint's own "
                         "processor resizes every view to 384x384 — about 1/7 the "
                         "pixel area of the 1024x1024 views every other tiled leg "
                         "gets — so the published richmond numbers carry an "
                         "uncontrolled resolution handicap. Default: leave the "
                         "processor alone, which is what was published. Setting this "
                         "IS a cache-key change.")
    ap.add_argument("--vistas-revision", default=None,
                    help="Pin the checkpoint revision. Default: unpinned, which is "
                         "what the published run used — recorded in the signature "
                         "only when set, so pinning does not orphan those detections.")
    ap.add_argument("--vistas-model", default=_D["vistas_model"],
                    help="Vistas-supervised segmentation checkpoint (#126). The arm "
                         "itself is chosen by the --models spec — 'vistas:curb-cut' "
                         "or 'vistas:curb-cut+curb' — because it varies by which "
                         "Vistas classes are read out, not by checkpoint.")
    ap.add_argument("--vistas-min-area-px", type=int, default=_D["vistas_min_area_px"],
                    help="Drop mask components smaller than this (default 16 px). "
                         "Like --score-threshold, a CACHE floor in the signature, not "
                         "the operating point; higher points are free re-scores.")
    ap.add_argument("--vistas-dtype", choices=["float16", "float32"],
                    default=_D["vistas_dtype"],
                    help="Inference precision. In the signature, because fp16 and "
                         "fp32 do not produce identical masks — a desktop run and a "
                         "cluster run must not silently share a cache entry.")
    ap.add_argument("--tiling", choices=["perspective", "none"], default="perspective",
                    help="VLM input: 'perspective' reprojects the pano into rectilinear "
                         "views (fair); 'none' uses one whole-pano call (lower bound). "
                         "No effect on rampnet.")
    ap.add_argument("--radius", type=float, default=PANO_RADIUS_NORMALIZED,
                    help=f"Normalized match radius (default {PANO_RADIUS_NORMALIZED}).")
    ap.add_argument("--op-threshold", type=float, default=0.0,
                    help="Drop predictions scoring below this before the main table, so the "
                         "reported operating point is comparable across models. Free (re-scores "
                         "the cache); models without confidences are unaffected.")
    ap.add_argument("--sweep", action="store_true",
                    help="Also print a threshold sweep for every model whose detections carry "
                         "confidences (RampNet, owlv2, gdino, vistas, yolo) — the tunable operating range.")
    ap.add_argument("--pr-out", help="Directory to write PR curves to (JSON per model, plus a "
                                     "combined PNG when matplotlib is installed).")
    ap.add_argument("--limit", type=int,
                    help="Score at most N panos (smoke test / cost control for VLM runs).")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / ".model_cache"),
                    help="Where to cache per-pano detections (keyed by model + rig + pano). "
                         "Re-runs reuse hits and don't re-pay the API.")
    ap.add_argument("--no-cache", action="store_true", help="Disable the detection cache.")
    ap.add_argument("--usage-log", default=default_usage_log(),
                    help="Append one JSONL record per model run — wall-clock always, "
                         "token counts and estimated cost when the provider bills for "
                         f"them (see pricing.py). Default: {DEFAULT_USAGE_LOG_REL} in the "
                         "MAIN checkout, not this worktree (tracked in git — the cost "
                         "record is a durable research fact, not scratch). Pass 'none' "
                         "to disable, which a paid model refuses to run under unless "
                         "--allow-unrecorded-spend is given too.")
    ap.add_argument("--allow-unrecorded-spend", action="store_true",
                    help="Permit a paid model to run with --usage-log none. Needed only "
                         "to spend money without recording what it cost, which is not "
                         "recoverable afterwards; say why in the PR if you use it.")
    return ap


def main():
    # Windows consoles are cp1252; avoid UnicodeEncodeError on stray bytes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = build_parser()
    args = ap.parse_args()

    load_dotenv(str(REPO_ROOT))
    specs = [parse_model_spec(t) for t in args.models.split(",") if t.strip()]
    records, verdicts, panos_dir = load_bundle(args.bundle)
    # Fail fast on a broken bundle before any (paid) detector call, then reduce
    # both GT sources to the same {pid: GroundTruth} shape.
    if verdicts is not None:
        if args.limit:
            verdicts = dict(list(verdicts.items())[:args.limit])
        validate_bundle(records, verdicts)
        gts = ground_truths_from_verdicts(records, verdicts)
        gt_desc = "reviewer-confirmed ramps + missed marks"
    else:
        gts = load_manual_ground_truths(args.bundle)
        # Validate the whole bundle (all labels <-> all records) before slicing to
        # --limit: a manual bundle is only valid if every label has a record and
        # vice versa, so a smoke run should still catch a partial/misbuilt bundle.
        # (The city branch above validates post-slice because its verdicts and
        # records are built together and can't drift independently.)
        validate_manual_bundle(records, gts,
                               need_detections=any(p == "rampnet" for p, _ in specs))
        if args.limit:
            gts = dict(list(gts.items())[:args.limit])
        gt_desc = "independent manual labels (YOLO box centers)"
    radius_sq = radius_sq_for(args.radius)
    city = os.path.basename(os.path.normpath(args.bundle))
    cache = DetectionCache(args.cache_dir, enabled=not args.no_cache)
    usage_log = None if args.usage_log == "none" else args.usage_log
    if usage_log:
        # The #139 failure mode: the ledger written inside a scratch worktree that
        # is deleted afterwards, so the run leaves no record and no guard notices
        # ($70.41, recovered only from Cloud Monitoring — #143). The default now
        # resolves to the main checkout; an explicit --usage-log can still land in
        # a worktree, so say so while someone is watching.
        canonical = canonical_repo_root()
        if canonical != REPO_ROOT and Path(usage_log).resolve().is_relative_to(REPO_ROOT):
            print(f"WARNING: --usage-log points inside this worktree "
                  f"({REPO_ROOT}), not the main checkout ({canonical}). A worktree "
                  f"is deleted when its session ends and takes the ledger with it; "
                  f"commit the log or point it at the main checkout.\n")
    # Whether a paid leg may run without recording what it spends. The check itself
    # happens in score_model, at the first pano that is NOT already cached -- still
    # before any money moves, but without refusing a re-score that provably cannot
    # spend anything (a fully cached leg skips the model load and makes no calls).
    spend_needs_recording = usage_log is None and not args.allow_unrecorded_spend

    print(f"Bundle: {args.bundle}  ({len(gts)} scored panos)  "
          f"match radius {args.radius}  ground truth: {gt_desc}")
    print(f"Detection cache: {'off' if args.no_cache else args.cache_dir}\n")

    rows, runs = [], []
    seen = {}
    for provider, model_id in specs:
        label, detector = build_detector(provider, model_id, records, args)
        # Disambiguate if the same label appears twice (e.g. same model, two configs).
        if label in seen:
            seen[label] += 1
            label = f"{label}#{seen[label]}"
        else:
            seen[label] = 1
        run, timing = None, {}
        t_leg = perf_counter()
        try:
            run = score_model(
                detector, records, gts, panos_dir, radius_sq, label, city, cache,
                spend_needs_recording=spend_needs_recording, timing=timing)
        except UnrecordedSpend:
            # Not a "this model is not runnable here" condition: it is a deliberate
            # refusal, and swallowing it would let the next paid leg in the list
            # spend unrecorded too.
            raise
        except Exception as e:
            # Missing client lib, missing credentials, a checkpoint whose remote
            # code won't load on this transformers version: skip the whole model
            # with a clear note rather than crashing a multi-model cluster run that
            # has already paid for the models before it. Per-pano faults are
            # isolated inside score_model; data-integrity problems are caught by
            # validate_bundle before any of this. The type is printed so a genuine
            # bug here is still diagnosable rather than silently "not runnable".
            print(f"[{label}] not runnable: {type(e).__name__}: {e}\n")
        finally:
            # In a finally, not after the try: the run that dies or is Ctrl-C'd
            # partway through has ALREADY paid for the calls it made, and that is
            # precisely the spend worth recording. KeyboardInterrupt isn't an
            # Exception, so on the success path alone it would unwind past this.
            # panos_scored is None when score_model didn't return — the token
            # counts are still exact, only the denominator is unknown.
            # The wall clock is stopped here rather than inside score_model so it
            # still covers a leg that raised partway -- the case this finally
            # exists for. It brackets the model load and the pano loop both.
            timing["elapsed_s"] = round(perf_counter() - t_leg, 3)
            report_usage(detector, label, city,
                         len(run.scored) if run is not None else None, usage_log,
                         timing=timing)
        if run is None:
            continue
        report = operating_report(run.report, run.scored, radius_sq, args.op_threshold)
        rows.append((label, report))
        # The detector's cache floor travels with the run so the sweep can drop
        # thresholds the cache has no detections for (phantom rows otherwise).
        runs.append((label, run, getattr(detector, "score_threshold", None)))
        if run.failures:
            print(f"[{label}] {len(run.failures)} pano failure(s) isolated "
                  "(excluded from the score):")
            for pid, msg in run.failures[:5]:
                print(f"    {pid}: {msg}")
            if len(run.failures) > 5:
                print(f"    ... and {len(run.failures) - 5} more")
            print()

    if rows:
        if args.op_threshold > 0:
            print(f"Operating point: predictions with confidence < {args.op_threshold} dropped "
                  "(models without confidences are unaffected).")
        print_table(rows)
        # AP is over the recall-confirmed panos only (one consistent GT denominator);
        # the P/R columns count every pano. See rampnet/detection_eval.aggregate.
        if any(r.ap is not None for _, r in rows):
            print("AP: all-point interpolated, over the recall-confirmed panos, from the "
                  "full confidence range (--op-threshold does not truncate it); "
                  "'-' = no calibrated per-box score.")

    if args.sweep:
        for label, run, floor in runs:
            if has_confidences(run.scored):
                print_sweep(label, sweep_rows(run.scored, radius_sq, floor=floor))
        print()

    if args.pr_out:
        curves = [(label, r) for label, r in rows if r.pr_curve]
        if curves:
            write_pr_curves(args.pr_out, curves)
        else:
            print("No model produced a PR curve (needs per-detection confidences).")

    if verdicts is not None:
        # Cross-check: RampNet's own verdict-based P/R (the published definition).
        confs_by_pid = {pid: [d["confidence"] for d in records[pid]["detections"]]
                        for pid in verdicts}
        pools = collect(verdicts, confs_by_pid)
        print()
        print(format_report("RampNet verdict-based cross-check", pools))
        # collect() records verdict/results mismatches as notes and leaves surfacing to
        # the caller; print them so a silently-skipped pano is visible, not swallowed.
        for w in pools.warnings:
            print(f"  ! {w}")
    else:
        # No verdicts to cross-check against; the manual bundle's equivalent is
        # reproducing the published gold-set numbers (printed by the exporter, and
        # documented in docs/model_comparison.md).
        print("\nManual-GT bundle: no verdict-based cross-check (validate the rampnet row "
              "against the published gold-set numbers instead; see docs/model_comparison.md).")


if __name__ == "__main__":
    main()
