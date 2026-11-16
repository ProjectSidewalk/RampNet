"""Lyon audit top-up for the "other cameras" group (#278): draw, pre-registered plan, stop rule.

The first Lyon label audit (#159, ``benchmark/lyon/``, ``lyon_review_159.py``) read the
``other_cameras`` group at P@0.55 = 40/46 = 0.870, eligible but "not decisive": both 95%
intervals reached below the 0.80 bar. This script draws a top-up of the same group from the
same run with the same sampler, pins how the pooled precision is read **before any pixel
is fetched**, and scores the review at three pre-registered looks.

The design, all of it in ``benchmark/lyon_topup278/audit_plan.json``:

* 80 random ``other_cameras`` panos (70 + a reserve of 10), seed 278, spaced 30 m from
  each other and from every first-draw pano (the pool drops ``benchmark/lyon/exclude_ids.txt``);
* a detections-only review (no missed-ramp pass) in ascending ``sha1(panorama_id)`` order;
* pooled precision = first-draw units (30 panos, 40/46) + the counted top-up panos;
* looks at exactly the first 25, 45 and 70 staged panos; at each look a Wilson interval
  at z = 2.35 and a pano-bootstrap interval at the matching level must both clear 0.80
  (SETTLED eligible) or both sit below it (SETTLED not eligible); otherwise CONTINUE, and
  look 3 is final whatever it reads;
* interim output is blind: CONTINUE or SETTLED only, no estimate anywhere until the final
  reading.

**Nothing here produces a verdict.** The review is Jon's; ``score`` reads his exports.
``lyon_review_159.py`` is imported read-only and never edited (``score`` there refuses a
changed plan or plan-building code).

Commands, in order (``docs/lyon_review_159.md`` section 11 has the full replication):

    # 1. the pool: the run's other_cameras records outside the first draw's 30 m buffer
    python scripts/analysis/lyon_topup_278.py pool --results <runs/lyon/results.jsonl> \\
        --out <scratch>/pool_other_cameras.jsonl

    # 2. the draw: sidewalk-auto-labeler scripts/export_benchmark.py at 01b9422 on that pool
    #    (command in bundle_provenance.json -> export), then copy it in and derive the lists
    python scripts/analysis/lyon_topup_278.py stage --from <scratch>/bundle \\
        --results <runs/lyon/results.jsonl>

    # 3. operating characteristics (numpy, slow, no check mode) and the plan
    python scripts/analysis/lyon_topup_278.py oc --write
    python scripts/analysis/lyon_topup_278.py plan --write      # without --write: byte check
    python scripts/analysis/lyon_topup_278.py verify --records-only

    # 4. pixels (after the plan is committed and pushed): copy verified archive files in
    python scripts/analysis/lyon_topup_278.py stage-pixels --from <dir of <id>.jpg> \\
        --index <archive index.csv>
    python scripts/analysis/imagery_manifest.py --write --cities lyon_topup278 --panos-root .
    python scripts/analysis/lyon_topup_278.py band --write
    python scripts/analysis/lyon_topup_278.py verify
    python scripts/gt_gallery.py benchmark/lyon_topup278 --hide-credit --detections-only

    # 5. the review: after pano 25, 45, 70 save the export as verdicts_backups/look<k>.json
    python scripts/analysis/lyon_topup_278.py score --write

``tests/test_lyon_topup_278.py`` covers the logic on committed inputs and synthetic looks.
"""
import argparse
import collections
import csv
import datetime
import fnmatch
import hashlib
import json
import math
import os
import random
import shutil
import sys
from statistics import NormalDist

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lyon_review_159 as L  # noqa: E402  (read-only: never edit the posted instrument)

GROUP = "other_cameras"
SEED = 278               # the issue number, as the first draw used 159
N_MAX = 70               # panos the review reads at most
RESERVE = 10             # replaces only a pano whose pixels cannot be staged
LOOKS = (25, 45, 70)     # the stop rule reads exactly these prefixes of the staged list
Z_SEQ = 2.35             # Wilson z and the bootstrap's matching level at every look
BOOT_REPS = 10000
BAR = L.PASS_PRECISION   # 0.80, the #158 build line
DEPLOYED = L.DEPLOYED    # 0.55
FIRST = L.BUNDLE         # benchmark/lyon, frozen
BUNDLE = os.path.join(REPO, "benchmark", "lyon_topup278")
OUT_DIR = os.path.join(REPO, "analysis_out", "lyon_topup_278")
DESIGN_OC = os.path.join(OUT_DIR, "design_oc.json")
LOOKS_JSON = os.path.join(OUT_DIR, "looks.json")
SCORE = os.path.join(OUT_DIR, "score.json")
PLAN_NAME = "audit_plan.json"
PROVENANCE = "bundle_provenance.json"
LOOK_DIR = "verdicts_backups"
EXPORTER_COMMIT = "01b9422"
POOL_EXPECT = {"group_with_det055": 1513, "group_without_det055": 7883,
               "pool_records": 8997, "pool_with_det055": 1342}
#: First-draw files the top-up leans on; the plan pins their sha256 and score re-checks.
FIRST_PINS = ("records.jsonl", "verdicts.json", "audit_plan.json", "exclude_ids.txt")
TEXT_FILES = ("records.jsonl", "sample.json", "ids.txt", "never_train_ids.txt",
              "exclude_ids.txt", PLAN_NAME, PROVENANCE, L.NADIR_BAND_NAME)

_r = L._r
_dumps = L._dumps
_sha256 = L._sha256


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rel(path):
    try:
        return os.path.relpath(path, REPO).replace(os.sep, "/")
    except ValueError:          # another drive (a test's tmp dir on Windows)
        return path


def _write_lines(path, lines):
    L._write_lines(path, lines)


def _read_lines(path):
    return L._read_lines(path)


def load_group_map(path=os.path.join(FIRST, "group_map.json")):
    with open(path, encoding="utf-8") as f:
        return [tuple(p) for p in json.load(f)]


def group_of(pano, group_map):
    """The audit group of a pano block, exactly as the labeler's exporter assigns it.

    The raw label is ``camera_model|copyright`` with a missing or null value read as the
    empty string; the first ``fnmatch.fnmatchcase`` pattern of the map names the group, and
    an unmatched label keeps its raw value (labeler ``make_stratum_of``).

    Example:
        >>> gm = [("Max|ecartip", "ecartip_gopro_max"), ("Max|*", "other_gopro_max"),
        ...       ("*", "other_cameras")]
        >>> group_of({"camera_model": "QooCam 3", "copyright": "x"}, gm)
        'other_cameras'
        >>> group_of({"camera_model": "Max", "copyright": None}, gm)
        'other_gopro_max'
    """
    label = "|".join("" if pano.get(k) is None else str(pano.get(k))
                     for k in ("camera_model", "copyright"))
    for pattern, name in group_map:
        if fnmatch.fnmatchcase(label, pattern):
            return name
    return label


def has_det055(rec):
    return any(d["confidence"] >= DEPLOYED for d in rec.get("detections") or [])


def review_order(ids):
    """Ascending ``sha1(panorama_id)``: the gallery's own order (``gt_gallery.py``), which
    does not depend on content. Example: ``review_order(['b', 'a'])`` -> ``['a', 'b']``
    (sha1('a') = 86f7..., sha1('b') = e9d7...)."""
    return sorted(ids, key=lambda p: hashlib.sha1(p.encode()).hexdigest())


def boot_level(z):
    """The (lower, upper) percentile levels that match a two-sided Wilson z."""
    a = NormalDist().cdf(-z)
    return a, 1 - a


def bootstrap_ci_level(units, z, reps=BOOT_REPS, seed=SEED):
    """Percentile interval of sum(k)/sum(n) over pano units resampled with replacement, at
    the levels ``NormalDist().cdf(-z)`` and its complement.

    The resampling is ``lyon_review_159.bootstrap_ci``'s, draw for draw, so at z = 1.96 and
    the same seed it returns the same 95% interval (a test). Pure. (None, None) without any n.

    Example:
        >>> bootstrap_ci_level([(1, 1), (0, 1)], 1.96, reps=200, seed=1)
        (0.0, 1.0)
    """
    if not units or not sum(n for _, n in units):
        return None, None
    rng = random.Random(seed)
    m = len(units)
    vals = []
    for _ in range(reps):
        k = n = 0
        for _ in range(m):
            a, b = units[rng.randrange(m)]
            k += a
            n += b
        if n:
            vals.append(k / n)
    vals.sort()
    lo, hi = boot_level(z)
    return vals[int(lo * (len(vals) - 1))], vals[int(hi * (len(vals) - 1))]


def pano_unit(dets):
    """``(correct, decided)`` for one pano's detection verdicts: ``unsure`` dropped,
    ``duplicate`` and ``False`` count as false (``score_groups`` at lenient off)."""
    ok = [d is True for d in dets if d != "unsure"]
    return sum(ok), len(ok)


def first_draw(bundle=FIRST):
    """The first draw's ``other_cameras`` random panos: units in sorted-id order (the order
    ``score_groups`` bootstraps in), plus the shown / unsure / duplicate counts."""
    recs = L.load_records(bundle)
    with open(os.path.join(bundle, "verdicts.json"), encoding="utf-8") as f:
        v = json.load(f)["panos"]
    pids = sorted(r["pano"]["panorama_id"] for r in recs
                  if r["benchmark_stratum"] == GROUP and r["benchmark_group"] == "random")
    units, shown, unsure, dup = [], 0, 0, 0
    for pid in pids:
        dets = v[pid]["dets"]
        units.append(pano_unit(dets))
        shown += len(dets)
        unsure += sum(d == "unsure" for d in dets)
        dup += sum(d == "duplicate" for d in dets)
    return {"panos": len(units), "units": units, "correct": sum(k for k, _ in units),
            "decided": sum(n for _, n in units), "shown": shown, "unsure": unsure,
            "duplicate": dup}


# --------------------------------------------------------------------------- #
# pool
# --------------------------------------------------------------------------- #
def build_pool(results, out, group_map, excluded):
    """Stream ``results`` and write every raw line whose pano is in ``GROUP`` and not in
    ``excluded``. Returns the counts. The bytes of each kept line are unchanged."""
    c = collections.Counter()
    with open(results, "rb") as src, open(out, "wb") as dst:
        for raw in src:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            if group_of(rec["pano"], group_map) != GROUP:
                continue
            det = has_det055(rec)
            c["group_with_det055" if det else "group_without_det055"] += 1
            if rec["pano"]["panorama_id"] in excluded:
                continue
            c["pool_records"] += 1
            c["pool_with_det055"] += det
            dst.write(raw if raw.endswith(b"\n") else raw + b"\n")
    return dict(c)


def check_mapping(group_map, bundle=FIRST):
    """Problems where ``group_of`` disagrees with the committed ``benchmark_stratum``."""
    return [r["pano"]["panorama_id"] for r in L.load_records(bundle)
            if group_of(r["pano"], group_map) != r["benchmark_stratum"]]


def cmd_pool(args):
    gm = load_group_map()
    bad = check_mapping(gm)
    if bad:
        print(f"STOP: the group mapping disagrees with benchmark_stratum on {len(bad)} "
              f"first-draw records, e.g. {bad[:3]}")
        return 1
    excluded = set(_read_lines(os.path.join(FIRST, "exclude_ids.txt")))
    counts = build_pool(args.results, args.out, gm, excluded)
    counts["pool_sha256"] = _sha256(args.out)
    counts["results_sha256"] = _sha256(args.results)
    print(json.dumps(counts, sort_keys=True))
    off = {k: (counts.get(k), v) for k, v in POOL_EXPECT.items() if counts.get(k) != v}
    if off:
        print(f"STOP: counts differ from the expected {off} (got, expected)")
        return 1
    print("pool: OK (the mapping reproduces benchmark_stratum on all 200 first-draw records)")
    return 0


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def stage(src, results, dst=BUNDLE, first=FIRST, buffer_m=L.EXCLUDE_BUFFER_M):
    """Copy the exporter's ``records.jsonl`` and ``sample.json`` into ``dst`` (email-masked)
    and write ``ids.txt`` (the draw, sorted), ``never_train_ids.txt`` (first draw + top-up)
    and ``exclude_ids.txt`` (cumulative: the first list + every run pano within ``buffer_m``
    of a top-up pano). Asserts the draw's spacing and strata. Returns a summary."""
    os.makedirs(dst, exist_ok=True)
    for name in ("records.jsonl", "sample.json"):
        with open(os.path.join(src, name), encoding="utf-8", newline="") as f:
            text = f.read()
        with open(os.path.join(dst, name), "wb") as f:
            f.write(L.mask_email(text).encode("utf-8"))
    recs = L.load_records(dst)
    new = {r["pano"]["panorama_id"]: (r["pano"]["lat"], r["pano"]["lng"]) for r in recs}
    problems = draw_problems(recs, first, buffer_m)
    if problems:
        raise ValueError("draw fails its checks: " + "; ".join(problems[:5]))
    first_ids = _read_lines(os.path.join(first, "ids.txt"))
    first_excl = set(_read_lines(os.path.join(first, "exclude_ids.txt")))
    points = [(r["pano"]["panorama_id"], r["pano"].get("lat"), r["pano"].get("lng"))
              for r in L._iter_results(results)]
    excl = sorted(first_excl | L.exclusion_ids(points, new, buffer_m))
    _write_lines(os.path.join(dst, "ids.txt"), sorted(new))
    _write_lines(os.path.join(dst, "never_train_ids.txt"), sorted(set(first_ids) | set(new)))
    _write_lines(os.path.join(dst, "exclude_ids.txt"), excl)
    return {"ids": len(new), "never_train_ids": len(set(first_ids) | set(new)),
            "exclude_ids": len(excl), "exclude_ids_added": len(excl) - len(first_excl),
            "run_panos": len(points), "results_jsonl_sha256": _sha256(results),
            "records_jsonl_sha256": _sha256(os.path.join(dst, "records.jsonl"))}


def draw_problems(recs, first=FIRST, buffer_m=L.EXCLUDE_BUFFER_M):
    """Why a top-up draw is not what the plan says (empty list = it is). Brute force."""
    problems = []
    first_recs = L.load_records(first)
    first_excl = set(_read_lines(os.path.join(first, "exclude_ids.txt")))
    pts = [(r["pano"]["panorama_id"], r["pano"]["lat"], r["pano"]["lng"]) for r in recs]
    if len({p for p, _, _ in pts}) != len(pts):
        problems.append("duplicate panorama_id")
    for r in recs:
        pid = r["pano"]["panorama_id"]
        if r.get("benchmark_group") != "random" or r.get("benchmark_stratum") != GROUP:
            problems.append(f"{pid}: {r.get('benchmark_group')}/{r.get('benchmark_stratum')}")
        if not has_det055(r):
            problems.append(f"{pid}: no detection >= {DEPLOYED}")
        if pid in first_excl:
            problems.append(f"{pid}: inside the first draw's exclusion list")
    for pid, lat, lng in pts:
        for fr in first_recs:
            d = L.haversine_m(lat, lng, fr["pano"]["lat"], fr["pano"]["lng"])
            if d <= buffer_m:
                problems.append(f"{pid}: {d:.1f} m from first-draw {fr['pano']['panorama_id']}")
    for i, (a, la, ga) in enumerate(pts):
        for b, lb, gb in pts[i + 1:]:
            d = L.haversine_m(la, ga, lb, gb)
            if d < buffer_m:
                problems.append(f"{a} and {b}: {d:.1f} m apart")
    return problems


def cmd_stage(args):
    print(json.dumps(stage(args.src, args.results), sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# oc: operating characteristics of the stop rule (simulation; no check mode)
# --------------------------------------------------------------------------- #
OC_PS = (0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.87, 0.90, 0.93)
OC_DESIGNS = (
    {"design": "looks 25/45/70, z 2.35 (pre-registered)", "looks": list(LOOKS), "z": Z_SEQ},
    {"design": "looks 25/45/70, z 2.2 (option A2)", "looks": list(LOOKS), "z": 2.2},
    {"design": "one look at +69, z 1.96 (the issue's proposal, two-sided review)",
     "looks": [69], "z": 1.96},
)
OC_RHOS = (0.0, 0.30)
OC_SIMS = 4000
OC_BOOT = 1000
PRIOR = (41, 7)          # flat prior updated by the first draw's 40/46


def simulate(first_units, looks, z, p, rho=0.0, sims=OC_SIMS, boot=OC_BOOT, seed=SEED,
             prior=None):
    """Monte Carlo of the stop rule, conditional on the observed first draw.

    Each top-up pano draws its decided-detection count from the first draw's per-pano
    counts; each detection is correct with probability ``p`` (``rho`` > 0: a beta-binomial
    pano effect with that intra-pano correlation; ``prior`` = (a, b): ``p`` drawn per run
    from Beta(a, b) instead). At each look: Wilson(z) and a pano bootstrap at the matching
    level on the pooled units. Returns the rates of each final reading and the mean panos.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    u = np.array(first_units)
    k1, n1 = u[:, 0], u[:, 1]
    counts = n1
    lo_q, hi_q = boot_level(z)
    out = collections.Counter()
    stop_at = [0] * len(looks)
    panos = 0
    for _ in range(sims):
        pr = rng.beta(*prior) if prior else p
        nt = rng.choice(counts, size=looks[-1])
        if rho > 0:
            a = pr * (1 - rho) / rho
            b = (1 - pr) * (1 - rho) / rho
            pp = rng.beta(a, b, size=looks[-1])
        else:
            pp = np.full(looks[-1], pr)
        kt = rng.binomial(nt, pp)
        for i, m in enumerate(looks):
            k = np.concatenate([k1, kt[:m]])
            n = np.concatenate([n1, nt[:m]])
            K, N = int(k.sum()), int(n.sum())
            est = K / N
            wlo, whi = L.wilson(K, N, z)
            idx = rng.integers(0, len(k), size=(boot, len(k)))
            kk, nn = k[idx].sum(1), n[idx].sum(1)
            vals = np.sort(kk[nn > 0] / nn[nn > 0])
            blo = vals[int(lo_q * (len(vals) - 1))]
            bhi = vals[int(hi_q * (len(vals) - 1))]
            last = i == len(looks) - 1
            if est >= BAR and wlo >= BAR and blo >= BAR:
                out["settled_eligible"] += 1
            elif est < BAR and whi < BAR and bhi < BAR:
                out["settled_not_eligible"] += 1
            elif last:
                out["final_ge_080_unsettled" if est >= BAR else "final_lt_080_unsettled"] += 1
            else:
                continue
            stop_at[i] += 1
            panos += m
            break
    row = {k: _r(out[k] / sims) for k in ("settled_eligible", "settled_not_eligible",
                                          "final_ge_080_unsettled", "final_lt_080_unsettled")}
    row["p_final_estimate_ge_080"] = _r((out["settled_eligible"]
                                        + out["final_ge_080_unsettled"]) / sims)
    row["mean_panos"] = round(panos / sims, 1)
    row["stop_at"] = [_r(s / sims) for s in stop_at]
    return row


def build_oc(first_units=None, sims=OC_SIMS, boot=OC_BOOT):
    first_units = first_units or first_draw()["units"]
    rows, j = [], 0
    for d in OC_DESIGNS:
        for rho in OC_RHOS:
            for p in OC_PS:
                j += 1
                rows.append({**d, "rho": rho, "p": p,
                             **simulate(first_units, d["looks"], d["z"], p, rho, sims, boot,
                                        SEED * 1000 + j)})
        j += 1
        rows.append({**d, "rho": 0.0, "p": f"Beta{PRIOR}",
                     **simulate(first_units, d["looks"], d["z"], None, 0.0, sims, boot,
                                SEED * 1000 + j, prior=PRIOR)})
    import numpy
    return {
        "what": "operating characteristics of the #278 top-up stop rule, simulated "
                "conditional on the observed first draw of other_cameras (30 random panos, "
                "40/46). Rates are shares of simulated reviews ending in each reading.",
        "model": "each top-up pano draws its decided-detection count from the first draw's "
                 "per-pano counts; each detection is correct with probability p (rho > 0: "
                 "beta-binomial pano effect with that intra-pano correlation; the group's "
                 "ANOVA ICC in the first draw is about 0.33, estimated from 27 panos). "
                 f"p = Beta{PRIOR}: p drawn per review from the flat prior updated by 40/46 "
                 "(a predictive average over what the first draw supports).",
        "readings": {"settled_eligible": "estimate >= 0.80 and both lower bounds >= 0.80 at "
                                         "some look (the false-settle rate where p < 0.80)",
                     "settled_not_eligible": "estimate < 0.80 and both upper bounds < 0.80",
                     "final_ge_080_unsettled": "last look, estimate >= 0.80, not settled",
                     "final_lt_080_unsettled": "last look, estimate < 0.80, not settled"},
        "sims_per_cell": sims, "bootstrap_resamples": boot, "seed_base": SEED * 1000,
        "monte_carlo_error": f"about +-{round(1.96 * math.sqrt(0.25 / sims), 3)} on a rate "
                             f"near 0.5, less near 0 or 1",
        "numpy": numpy.__version__,
        "not_a_byte_contract": "numpy random streams may change across versions; plan reads "
                               "this file, it never recomputes it",
        "first_draw_units": [list(u) for u in first_units],
        "rows": rows,
    }


def cmd_oc(args):
    if not args.write:
        print("oc has no check mode (numpy streams are not a byte contract); "
              "use --write to regenerate")
        return 1
    oc = build_oc(sims=args.sims, boot=args.boot)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DESIGN_OC, "wb") as f:
        f.write(_dumps(oc).encode("utf-8"))
    print(f"wrote {_rel(DESIGN_OC)}")
    for r in oc["rows"]:
        print(f"  {r['design'][:24]:<24} rho {r['rho']:<4} p {str(r['p']):<12} "
              f"settle {r['settled_eligible']:.3f}  not {r['settled_not_eligible']:.3f}  "
              f"P(est>=0.80) {r['p_final_estimate_ge_080']:.3f}  panos {r['mean_panos']}")
    return 0


FIGURE = os.path.join(REPO, "docs", "figures", "lyon_topup_278", "settle_by_p.png")


def cmd_figure(args):
    """Two panels from the stored ``design_oc.json`` (never recomputed): the chance of
    "SETTLED eligible" and of a final estimate >= 0.80, by true precision, per design;
    solid = independent detections, dashed = intra-pano correlation 0.30."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(DESIGN_OC, encoding="utf-8") as f:
        rows = [r for r in json.load(f)["rows"] if isinstance(r["p"], float)]
    colors = ("#1b6ca8", "#d1495b", "#6c757d")       # distinct in grey-scale too
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, key, title in ((axes[0], "settled_eligible", "P(SETTLED eligible)"),
                           (axes[1], "p_final_estimate_ge_080",
                            "P(final pooled estimate >= 0.80)")):
        for d, c, m in zip(OC_DESIGNS, colors, markers):
            for rho, ls in ((0.0, "-"), (0.30, "--")):
                sel = sorted((r["p"], r[key]) for r in rows
                             if r["design"] == d["design"] and r["rho"] == rho)
                ax.plot([s[0] for s in sel], [s[1] for s in sel], ls, color=c, marker=m,
                        ms=4, label=f"{d['design'].split(' (')[0]}, rho {rho:g}")
        ax.axvline(BAR, color="#999", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("true precision at 0.55 (other cameras)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Lyon top-up stop rule (#278), conditional on the first draw's 40/46",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {_rel(args.out)}")
    return 0


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
CONSEQUENCES = [
    {"reading": "SETTLED eligible (any look)",
     "meaning": "eligible for a Lyon training pool; the 'not decisive' flag is cleared (the "
                "posted 'eligible (decisive)')"},
    {"reading": "look 3, estimate >= 0.80, not settled",
     "meaning": "eligible under the posted rule; stays 'not decisive'"},
    {"reading": "look 3, estimate < 0.80, not settled",
     "meaning": "not eligible: stays out of any Lyon training pool unless a later, separately "
                "pre-registered audit says otherwise"},
    {"reading": "SETTLED not eligible (any look)",
     "meaning": "the posted 'decisive fail': excluded from any Lyon training pool"},
]

DEVIATIONS = [
    "detections-only review: each 0.55 detection is judged (rubric unchanged), but there is "
    "no missed-ramp pass, the page exports no_missed: false, and recall is not read from this "
    "review (benchmark/RUBRICS.md section 1, 'Detections-only variant (#278)')",
    "sequential reading: three looks at z = 2.35 instead of the issue's single look at +69 "
    "panos at z = 1.96 (section 'looks'; design_oc compares both)",
    "pooled with the first draw: the first draw's 30 other_cameras random panos are the "
    "first units of the pool, so the pooled reading is conditional on a first draw that "
    "already read 40/46",
]


def first_pins(first=FIRST):
    return {f"benchmark/lyon/{n}": _sha256(os.path.join(first, n)) for n in FIRST_PINS}


def build_plan(bundle=BUNDLE, first=FIRST, oc_path=DESIGN_OC):
    recs = L.load_records(bundle)
    with open(os.path.join(bundle, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    with open(oc_path, encoding="utf-8") as f:
        oc = json.load(f)
    fd = first_draw(first)
    ids = [r["pano"]["panorama_id"] for r in recs]
    lo_q, hi_q = boot_level(Z_SEQ)
    return {
        "what": "Lyon audit top-up for other_cameras (#278), pre-registered before any pixel "
                "of the draw was fetched. Lyon is not a benchmark split; these panos never "
                "train. score refuses a plan whose sha256 differs from "
                "bundle_provenance.json -> pre_registered, or code that no longer rebuilds it.",
        "issue": "https://github.com/ProjectSidewalk/RampNet/issues/278",
        "question": "Does other_cameras clear the 0.80 precision bar at 0.55 decisively, or "
                    "does its eligibility flip, once the first draw is pooled with a top-up?",
        "group": GROUP,
        "pool": {"run_results_jsonl_sha256": "60e688b28b62f85e71f51c603dc1b6f4ff020348220c6cc7955891dcfef061ef",
                 "rule": "records of the 5 m run whose group, by benchmark/lyon/group_map.json "
                         "on camera_model|copyright (null read as empty), is other_cameras, "
                         "and whose id is not in benchmark/lyon/exclude_ids.txt",
                 "counts": POOL_EXPECT},
        "sampler": {"exporter": f"sidewalk-auto-labeler scripts/export_benchmark.py at "
                                f"{EXPORTER_COMMIT}",
                    "flags": "--sample 80 --empty-sample 0 --top 0 --seed 278 --min-spacing 30 "
                             "--group-key camera_model --group-key copyright --group-map "
                             "benchmark/lyon/group_map.json --group-min 0 --empty-group-min 0 "
                             "--records-only",
                    **{k: sample[k] for k in ("seed", "sample", "empty_sample", "min_spacing_m",
                                              "min_confidence", "selected", "groups",
                                              "source_record_count")},
                    "strata": sample["stratified"]["strata"]},
        "n_max": N_MAX, "reserve": RESERVE,
        "review_order": review_order(ids),
        "review_order_rule": "ascending sha1(panorama_id) over the drawn panos (the gallery's "
                             "order, content-blind)",
        "staged_list": f"the first {N_MAX} ids of review_order whose pixels were staged; "
                       f"written to bundle_provenance.json -> imagery.staged_ids before the "
                       f"page is built and frozen once any look export exists. Every rule "
                       f"below runs over the staged list, so an unstaged pano never blocks "
                       f"the count.",
        "reserve_rule": "the reserve only replaces a pano whose pixels cannot be staged (named, "
                        "with the reason, in bundle_provenance.json); an unsure verdict never "
                        "needs it",
        "judged": "each detection >= 0.55 of a staged pano: correct / false positive / "
                  "duplicate / unsure, rubric unchanged (benchmark/RUBRICS.md section 1); "
                  "duplicates false; unsure dropped; no missed-ramp pass; producer hidden "
                  "(gt_gallery.py --hide-credit --detections-only)",
        "pooled_precision": {
            "units": "per pano (correct, decided); decided excludes unsure",
            "pool": "the first draw's 30 other_cameras random panos (sorted by id) followed by "
                    "the counted top-up panos (in staged order)",
            "estimate": "sum correct / sum decided",
            "first_draw": {k: fd[k] for k in ("panos", "correct", "decided", "shown",
                                               "unsure", "duplicate")},
        },
        "looks": list(LOOKS),
        "look_rule": [
            "look k can be read once the first looks[k] staged panos are all fully judged; it "
            "reads those panos only, and ratings beyond them are ignored at that look",
            "looks are evaluated in order; the first settlement stands and ends the review",
            "each look has its own export file, verdicts_backups/look<k>.json, kept and "
            "committed; score refuses a later export that changes any verdict of a pano "
            "counted at an earlier look",
            "SETTLED eligible: estimate >= 0.80 and both lower bounds >= 0.80; SETTLED not "
            "eligible: estimate < 0.80 and both upper bounds < 0.80; otherwise CONTINUE, and "
            "at look 3 the reading is final whatever it is",
            "no discretionary stop: no futility or may-stop output; the review runs until "
            "SETTLED or look 3",
            "interim output is blind: before the final reading score prints only CONTINUE "
            "(with the next look's size) or SETTLED, and writes no estimate or interval "
            "anywhere; the numbers are printed and written only with the final reading",
        ],
        "z": Z_SEQ,
        "intervals": {"wilson": f"Wilson score interval at z = {Z_SEQ} over decided detections",
                      "bootstrap": {"what": "percentile interval of a pano bootstrap of the "
                                            "pooled units",
                                    "levels": [_r(lo_q, 6), _r(hi_q, 6)],
                                    "reps": BOOT_REPS, "seed": SEED,
                                    "resampling": "lyon_review_159.bootstrap_ci's, at the "
                                                  "levels NormalDist().cdf(-z) and its "
                                                  "complement"}},
        "bar": BAR,
        "consequences": CONSEQUENCES,
        "consequences_note": "the pooled reading replaces the first-draw reading for "
                             "other_cameras, in both directions; the 95% intervals printed "
                             "with the final reading are descriptive and cannot change a row",
        "reported_with_final_reading": [
            "the unsure share and the duplicate share of the counted top-up panos beside the "
            f"first draw's ({fd['unsure']} unsure, {fd['duplicate']} duplicate of "
            f"{fd['shown']} shown)",
            "any ratings beyond the look in force (descriptive, never counted)",
            "descriptive 95% intervals (Wilson z = 1.96 and the bootstrap at 2.5/97.5%)",
        ],
        "not_updated": ["recall at 0.55", "precision and recall at 0.30",
                        "the elevation split"],
        "never_train": "All drawn panos (ids.txt), staged or not, rated or not, and every run "
                       "pano within 30 m of one are held out of every training pool: pass "
                       "never_train_ids.txt and exclude_ids.txt of this directory (cumulative, "
                       "the first draw included) to the training-label builder, which refuses "
                       "the first draw's lists alone once this directory exists.",
        "deviations": DEVIATIONS,
        "first_draw_pins": first_pins(first),
        "rig_breakdown": L.rig_breakdown(recs),
        "detections_055": sum(d["confidence"] >= DEPLOYED for r in recs
                              for d in r["detections"]),
        "design_oc": oc,
    }


def cmd_plan(args):
    return L._check_or_write(build_plan(args.bundle), os.path.join(args.bundle, PLAN_NAME),
                             args.write)


# --------------------------------------------------------------------------- #
# stage-pixels
# --------------------------------------------------------------------------- #
def look_files(bundle=BUNDLE):
    d = os.path.join(bundle, LOOK_DIR)
    return sorted(n for n in os.listdir(d) if n.startswith("look") and n.endswith(".json")) \
        if os.path.isdir(d) else []


def load_index(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["panorama_id"]: row for row in csv.DictReader(f)}


def choose_staged(order, index, src, unverified=None, n_max=N_MAX):
    """(staged [(id, path, sha256, source)], dropped [(id, reason)], unused [ids]).

    Walks ``order``; an id is staged when ``src/<id>.jpg`` hashes to its archive index row,
    or, only for an id absent from the index, when ``unverified/<id>.jpg`` exists (marked
    not verified). Stops after ``n_max`` staged; ids after that are the unused reserve."""
    staged, dropped, unused = [], [], []
    for pid in order:
        if len(staged) >= n_max:
            unused.append(pid)
            continue
        row = index.get(pid)
        path = os.path.join(src, f"{pid}.jpg") if src else None
        if row is not None:
            if not path or not os.path.exists(path):
                dropped.append((pid, "in the archive index, but not fetched"))
                continue
            sha = _sha256(path)
            if sha != row["sha256"]:
                dropped.append((pid, f"bytes {sha[:12]} differ from the archive index "
                                     f"{row['sha256'][:12]}"))
                continue
            staged.append((pid, path, sha, "makelab2 archive, verified against index.csv"))
            continue
        upath = os.path.join(unverified, f"{pid}.jpg") if unverified else None
        if upath and os.path.exists(upath):
            staged.append((pid, upath, _sha256(upath),
                           f"panoramax {datetime.date.today().isoformat()}, not verified "
                           f"against the run's bytes"))
        else:
            dropped.append((pid, "absent from the archive index" if index else
                                 "no archive index and no fetched file"))
    return staged, dropped, unused


def cmd_stage_pixels(args):
    if look_files(args.bundle):
        print("refusing: a look export exists, so the staged list is frozen")
        return 1
    with open(os.path.join(args.bundle, PLAN_NAME), encoding="utf-8") as f:
        order = json.load(f)["review_order"]
    index = load_index(args.index) if args.index else {}
    staged, dropped, unused = choose_staged(order, index, args.src, args.unverified)
    panos = os.path.join(args.bundle, "panos")
    os.makedirs(panos, exist_ok=True)
    keep = {f"{pid}.jpg" for pid, _, _, _ in staged}
    extra = sorted(set(os.listdir(panos)) - keep)
    if extra:
        print(f"refusing: panos/ holds {len(extra)} file(s) outside the staged list, "
              f"e.g. {extra[:3]}")
        return 1
    rows = []
    for pid, path, sha, source in staged:
        dst = os.path.join(panos, f"{pid}.jpg")
        if not (os.path.exists(dst) and _sha256(dst) == sha):
            shutil.copyfile(path, dst)
        rows.append([pid, f"{pid}.jpg", os.path.getsize(dst), sha, source])
    with open(os.path.join(args.bundle, "index.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["panorama_id", "filename", "bytes", "sha256", "source"])
        w.writerows(rows)
    block = {"staged_ids": [pid for pid, _, _, _ in staged],
             "dropped": [{"id": pid, "reason": why} for pid, why in dropped],
             "unused_reserve": unused,
             "sources": dict(collections.Counter(s for _, _, _, s in staged)),
             "archive_index_sha256": _sha256(args.index) if args.index else None,
             "staged_on": datetime.date.today().isoformat()}
    prov_path = os.path.join(args.bundle, PROVENANCE)
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    prov.setdefault("imagery", {}).update(block)
    with open(prov_path, "wb") as f:
        f.write(_dumps(prov).encode("utf-8"))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in block.items()},
                     sort_keys=True))
    for pid, why in dropped:
        print(f"  dropped {pid}: {why}")
    if len(staged) < N_MAX:
        print(f"WARNING: only {len(staged)} of {N_MAX} staged; looks beyond that cannot be read")
    return 0


# --------------------------------------------------------------------------- #
# band: the nadir overlays of the staged panos (geometry only)
# --------------------------------------------------------------------------- #
#: Burned-in logo bands the black/white rule cannot see, listed after a contact sheet of
#: the bottom 31% of the staged panos (geometry only; filled in by the implementer after
#: the plan was committed). Empty means none was found.
LOGO_BANDS = {}


def build_band(bundle=BUNDLE):
    recs = {r["pano"]["panorama_id"]: r for r in L.load_records(bundle)}
    staged = load_provenance(bundle)["imagery"]["staged_ids"]
    panos = {}
    for pid in staged:
        path = os.path.join(bundle, "panos", f"{pid}.jpg")
        kind, y = L.measure_band(path)
        entry = {"band_top_y": _r(y), "kind": kind, "label": L.LABELS.get(kind),
                 "group": GROUP, "rig": L.rig_of(recs[pid]["pano"]), "method": "auto"}
        if pid in LOGO_BANDS:
            entry.update(band_top_y=_r(L.measure_logo_band(path)), kind="logo",
                         label="nadir logo band", overlay=LOGO_BANDS[pid],
                         method="found by eye on a contact sheet; edge by logo_band_top")
        panos[pid] = {k: v for k, v in entry.items() if v is not None}
    return {
        "what": "top edge of an opaque region at the bottom of the equirect, normalized y "
                "(0 = top), per staged pano; absent band_top_y = none found. gt_gallery.py "
                "hatches and labels it.",
        "found_by_eye": "the bottom 31% of every staged pano was looked at on a contact sheet "
                        "for burned-in logo bands (LOGO_BANDS in lyon_topup_278.py); geometry "
                        "only, nothing about the detections",
        "method": "lyon_review_159.measure_band / measure_logo_band (same constants)",
        "n_with_band": sum("band_top_y" in e for e in panos.values()),
        "n_without_band": sum("band_top_y" not in e for e in panos.values()),
        "panos": panos,
    }


def cmd_band(args):
    return L._check_or_write(build_band(args.bundle),
                             os.path.join(args.bundle, L.NADIR_BAND_NAME), args.write)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def load_provenance(bundle=BUNDLE):
    with open(os.path.join(bundle, PROVENANCE), encoding="utf-8") as f:
        return json.load(f)


def verify_bundle(bundle=BUNDLE, first=FIRST, require_imagery=True):
    """(problems, notes) for the top-up bundle; empty problems = OK."""
    problems, notes = [], []
    recs = L.load_records(bundle)
    ids = sorted(r["pano"]["panorama_id"] for r in recs)
    problems += draw_problems(recs, first)
    if _read_lines(os.path.join(bundle, "ids.txt")) != ids:
        problems.append("ids.txt is not the sorted record ids")
    first_ids = set(_read_lines(os.path.join(first, "ids.txt")))
    if set(_read_lines(os.path.join(bundle, "never_train_ids.txt"))) != first_ids | set(ids):
        problems.append("never_train_ids.txt is not first-draw ids + top-up ids")
    excl = set(_read_lines(os.path.join(bundle, "exclude_ids.txt")))
    if not (set(_read_lines(os.path.join(first, "exclude_ids.txt"))) | set(ids)) <= excl:
        problems.append("exclude_ids.txt does not contain the first list and every top-up id")
    plan_path = os.path.join(bundle, PLAN_NAME)
    if not os.path.exists(plan_path):
        problems.append("audit_plan.json absent")
    else:
        with open(plan_path, "rb") as f:
            if f.read() != _dumps(build_plan(bundle, first)).encode("utf-8"):
                problems.append("the current code does not rebuild audit_plan.json")
    for name in TEXT_FILES:
        path = os.path.join(bundle, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                if L.EMAIL_RE.search(f.read()):
                    problems.append(f"{name} carries an email-shaped token")
    if not require_imagery:
        notes.append("imagery not checked (--records-only)")
        return problems, notes
    prov = load_provenance(bundle)
    staged = (prov.get("imagery") or {}).get("staged_ids")
    if not staged:
        problems.append("bundle_provenance.json has no imagery.staged_ids")
        return problems, notes
    with open(plan_path, encoding="utf-8") as f:
        order = json.load(f)["review_order"]
    problems += staged_order_problems(order, staged, prov["imagery"].get("dropped", []))
    on_disk = sorted(os.path.splitext(n)[0] for n in os.listdir(os.path.join(bundle, "panos"))) \
        if os.path.isdir(os.path.join(bundle, "panos")) else []
    man = os.path.join(bundle, "imagery_manifest.json")
    manifest = json.load(open(man, encoding="utf-8"))["panos"] if os.path.exists(man) else {}
    index = load_index(os.path.join(bundle, "index.csv")) \
        if os.path.exists(os.path.join(bundle, "index.csv")) else {}
    for what, got in (("panos/", on_disk), ("imagery_manifest.json", sorted(manifest)),
                      ("index.csv", sorted(index))):
        if got != sorted(staged):
            problems.append(f"{what} lists {len(got)} panos, not the {len(staged)} staged ids")
    for pid in staged:
        if pid in manifest and pid in index and manifest[pid]["sha256"] != index[pid]["sha256"]:
            problems.append(f"{pid}: imagery_manifest.json and index.csv disagree")
        path = os.path.join(bundle, "panos", f"{pid}.jpg")
        if pid in manifest and os.path.exists(path) and _sha256(path) != manifest[pid]["sha256"]:
            problems.append(f"{pid}: bytes differ from imagery_manifest.json")
    if not os.path.exists(os.path.join(bundle, L.NADIR_BAND_NAME)):
        problems.append("nadir_band.json absent")
    notes.append(f"{len(staged)} staged panos checked against panos/, the manifest and index.csv")
    return problems, notes


def staged_order_problems(order, staged, dropped):
    """The staged list must be the review order with only named drops removed, cut at its
    last staged id."""
    dropped_ids = {d["id"] for d in dropped}
    if not staged:
        return ["no staged ids"]
    if any(p not in order for p in staged):
        return ["a staged id is not in review_order"]
    upto = order[:order.index(staged[-1]) + 1]
    expect = [p for p in upto if p not in dropped_ids]
    return [] if expect == list(staged) else \
        ["staged_ids is not review_order minus the named dropped ids"]


def cmd_verify(args):
    problems, notes = verify_bundle(args.bundle, require_imagery=not args.records_only)
    for n in notes:
        print("  " + n)
    for p in problems:
        print("  PROBLEM: " + p)
    print("verify: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# score: the stop rule
# --------------------------------------------------------------------------- #
def reading_at(units, z=Z_SEQ, reps=BOOT_REPS, seed=SEED):
    """(outcome, numbers) for pooled ``units``: outcome is 'eligible', 'not eligible' (both
    settled) or None (not settled). Pure."""
    k, n = sum(a for a, _ in units), sum(b for _, b in units)
    est = k / n
    wlo, whi = L.wilson(k, n, z)
    blo, bhi = bootstrap_ci_level(units, z, reps, seed)
    if est >= BAR and wlo >= BAR and blo >= BAR:
        out = "eligible"
    elif est < BAR and whi < BAR and bhi < BAR:
        out = "not eligible"
    else:
        out = None
    return out, {"correct": k, "decided": n, "estimate": est, "wilson": (wlo, whi),
                 "bootstrap": (blo, bhi)}


def fully_judged(entry, n_dets):
    return (entry is not None and len(entry.get("dets", [])) == n_dets
            and all(d is not None for d in entry["dets"]))


def evaluate_looks(looks, staged, n_dets, first_units, look_sizes=LOOKS, run_key=None):
    """The stop rule over the look exports. Pure.

    ``looks``: the ``panos`` maps of look1.json, look2.json, ... in order (as far as they
    exist); ``staged``: the staged id list; ``n_dets``: {pano: detections >= 0.55}. Returns
    a dict with ``status`` one of ``refuse`` (``problems``), ``waiting`` (the newest file
    does not cover its look), ``continue`` (``next``: the size of the next look), or
    ``final``, plus ``looks``: one record per look read (no estimate unless final)."""
    read, counted_at = [], {}
    for k, panos in enumerate(looks, start=1):
        if k > len(look_sizes):
            return {"status": "refuse", "problems": [f"look{k}.json: only "
                                                     f"{len(look_sizes)} looks exist"]}
        if read and read[-1]["outcome"] != "CONTINUE":
            return {"status": "refuse",
                    "problems": [f"look{k}.json exists, but look {k - 1} read "
                                 f"{read[-1]['outcome']}"]}
        size = look_sizes[k - 1]
        if len(staged) < size:
            return {"status": "refuse", "problems": [f"look {k} needs {size} staged panos, "
                                                     f"{len(staged)} are staged"]}
        counted = staged[:size]
        changed = sorted(pid for pid, j in counted_at.items()
                         if (panos.get(pid) or {}).get("dets") != looks[j - 1][pid]["dets"])
        if changed:
            return {"status": "refuse",
                    "problems": [f"look{k}.json changes the verdicts of {len(changed)} pano(s) "
                                 f"counted at an earlier look: {', '.join(changed)}"]}
        done = sum(fully_judged(panos.get(pid), n_dets[pid]) for pid in counted)
        if done < size:
            if k < len(looks):
                return {"status": "refuse",
                        "problems": [f"look{k}.json does not cover look {k} ({done} of the "
                                     f"first {size} judged), yet look{k + 1}.json exists"]}
            return {"status": "waiting", "look": k, "need": size, "done": done, "looks": read}
        units = list(first_units) + [pano_unit(panos[pid]["dets"]) for pid in counted]
        outcome, nums = reading_at(units)
        last = k == len(look_sizes)
        rec = {"look": k, "panos_counted": size,
               "outcome": ("SETTLED " + outcome) if outcome else ("FINAL" if last
                                                                  else "CONTINUE")}
        for pid in counted:
            counted_at.setdefault(pid, k)
        read.append(rec)
        if outcome or last:
            if k < len(looks):
                return {"status": "refuse",
                        "problems": [f"look{k + 1}.json exists, but look {k} read "
                                     f"{rec['outcome']}"]}
            return {"status": "final", "looks": read, "look": k, "outcome": outcome,
                    "numbers": nums, "units": units, "counted": counted, "deciding": panos}
    return {"status": "continue", "looks": read, "next": look_sizes[len(read)]}


def consequence_row(outcome, estimate):
    if outcome == "eligible":
        return CONSEQUENCES[0]
    if outcome == "not eligible":
        return CONSEQUENCES[3]
    return CONSEQUENCES[1] if estimate >= BAR else CONSEQUENCES[2]


def final_report(res, recs_by_id, first, staged, plan_sha, prov):
    nums, counted, panos = res["numbers"], res["counted"], res["deciding"]
    shown = sum(len(panos[p]["dets"]) for p in counted)
    unsure = sum(d == "unsure" for p in counted for d in panos[p]["dets"])
    dup = sum(d == "duplicate" for p in counted for d in panos[p]["dets"])
    beyond = [p for p in staged[len(counted):] if p in panos
              and any(d is not None for d in panos[p].get("dets", []))]
    bunits = [pano_unit([d for d in panos[p]["dets"] if d is not None]) for p in beyond]
    w95 = L.wilson(nums["correct"], nums["decided"])
    b95 = bootstrap_ci_level(res["units"], 1.96)
    row = consequence_row(res["outcome"], nums["estimate"])
    pre = prov.get("pre_registered") or {}
    return {
        "group": GROUP, "reading": row["reading"], "meaning": row["meaning"],
        "look": res["look"], "panos_counted": len(counted),
        "settled": res["outcome"] is not None,
        "pooled": {"panos": len(res["units"]), "correct": nums["correct"],
                   "decided": nums["decided"], "estimate": _r(nums["estimate"])},
        f"wilson_z{Z_SEQ}": [_r(x) for x in nums["wilson"]],
        f"bootstrap_z{Z_SEQ}": [_r(x) for x in nums["bootstrap"]],
        "descriptive_95": {"wilson": [_r(x) for x in w95], "bootstrap": [_r(x) for x in b95],
                           "note": "descriptive; cannot change the reading"},
        "top_up_counted": {"shown": shown, "unsure": unsure, "duplicate": dup,
                           "unsure_share": _r(unsure / shown) if shown else None,
                           "duplicate_share": _r(dup / shown) if shown else None},
        "first_draw": {k: first[k] for k in ("shown", "unsure", "duplicate")},
        "beyond_the_look": {"panos": len(beyond), "correct": sum(k for k, _ in bunits),
                            "decided": sum(n for _, n in bunits),
                            "note": "rated past the look in force; never counted"},
        "looks": res["looks"],
        "audit_plan_sha256": plan_sha,
        "plan_comment_posted": bool(pre.get("comment")),
    }


def preflight(bundle=BUNDLE, first=FIRST):
    """Refusals before any look is read: first-draw pins, the pre-registered plan hash,
    and the plan-rebuild check."""
    problems = []
    plan_path = os.path.join(bundle, PLAN_NAME)
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    if plan["first_draw_pins"] != first_pins(first):
        problems.append("a first-draw file changed since the plan pinned it")
    pre = load_provenance(bundle).get("pre_registered") or {}
    if _sha256(plan_path) != pre.get("audit_plan_sha256"):
        problems.append("audit_plan.json does not hash to bundle_provenance.json -> "
                        "pre_registered.audit_plan_sha256")
    if _sha256(os.path.join(bundle, "records.jsonl")) != pre.get("records_jsonl_sha256"):
        problems.append("records.jsonl does not hash to the pre-registered value")
    with open(plan_path, "rb") as f:
        if f.read() != _dumps(build_plan(bundle, first)).encode("utf-8"):
            problems.append("the current code does not rebuild audit_plan.json")
    return problems


def cmd_score(args):
    bundle = args.bundle
    problems = preflight(bundle)
    if problems:
        for p in problems:
            print("  PROBLEM: " + p)
        print("refusing to score")
        return 1
    prov = load_provenance(bundle)
    staged = (prov.get("imagery") or {}).get("staged_ids") or []
    names = look_files(bundle)
    expect = [f"look{k}.json" for k in range(1, len(names) + 1)]
    if names != expect:
        print(f"refusing: look files {names} are not look1.json, look2.json, ... in sequence")
        return 1
    if not names:
        print(f"no {LOOK_DIR}/look1.json yet: rate the first {LOOKS[0]} panos, Export, save it "
              f"there, and run score again")
        return 1
    looks, shas = [], []
    for n in names:
        path = os.path.join(bundle, LOOK_DIR, n)
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("run_key") not in (None, os.path.basename(bundle)):
            print(f"refusing: {n} is an export of {v.get('run_key')!r}, not this bundle")
            return 1
        looks.append(v.get("panos", {}))
        shas.append(_sha256(path))
    recs = {r["pano"]["panorama_id"]: r for r in L.load_records(bundle)}
    n_dets = {pid: len(r["detections"]) for pid, r in recs.items()}
    fd = first_draw()
    res = evaluate_looks(looks, staged, n_dets, fd["units"])
    if res["status"] == "refuse":
        for p in res["problems"]:
            print("  PROBLEM: " + p)
        print("refusing to score")
        return 1
    if res["status"] == "waiting":
        print(f"CONTINUE: look {res['look']} needs the first {res['need']} panos judged, "
              f"{res['done']} so far")
        return 0
    record = {"what": "the look exports read by score and their outcome; no estimate before "
                      "the final reading (blind interim)",
              "audit_plan_sha256": _sha256(os.path.join(bundle, PLAN_NAME)),
              "looks": [{**rec, "file": f"{LOOK_DIR}/{names[i]}", "sha256": shas[i]}
                        for i, rec in enumerate(res["looks"])]}
    rc = L._check_or_write(record, LOOKS_JSON, args.write)
    if res["status"] == "continue":
        print(f"CONTINUE: rate on to pano {res['next']}, Export, save it as "
              f"{LOOK_DIR}/look{len(names) + 1}.json, and run score again")
        return rc
    rep = final_report(res, recs, fd, staged, record["audit_plan_sha256"], prov)
    head = (f"SETTLED {res['outcome']} at look {res['look']}" if res["outcome"]
            else f"FINAL at look {res['look']}, not settled")
    print(head)
    print(f"  reading: {rep['reading']} -> {rep['meaning']}")
    p = rep["pooled"]
    print(f"  pooled P@0.55 {p['estimate']} ({p['correct']}/{p['decided']}, {p['panos']} panos)"
          f"  Wilson z{Z_SEQ} {rep[f'wilson_z{Z_SEQ}']}  bootstrap z{Z_SEQ} "
          f"{rep[f'bootstrap_z{Z_SEQ}']}  (95%: {rep['descriptive_95']['wilson']}, "
          f"{rep['descriptive_95']['bootstrap']})")
    t, f0 = rep["top_up_counted"], rep["first_draw"]
    print(f"  unsure {t['unsure']}/{t['shown']}, duplicate {t['duplicate']}/{t['shown']} "
          f"(first draw {f0['unsure']}/{f0['shown']}, {f0['duplicate']}/{f0['shown']})")
    if not rep["plan_comment_posted"]:
        print("  WARNING: pre_registered.comment is null: the plan was not posted on #159")
    rc |= L._check_or_write(rep, SCORE, args.write)
    final_path = os.path.join(bundle, "verdicts.json")
    src = os.path.join(bundle, LOOK_DIR, names[res["look"] - 1])
    if args.write:
        shutil.copyfile(src, final_path)
        print(f"copied {_rel(src)} -> {_rel(final_path)}")
    elif not os.path.exists(final_path) or _sha256(final_path) != _sha256(src):
        print(f"DIFFERS: {_rel(final_path)} is not the deciding look file (run with --write)")
        rc = 1
    return rc


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=BUNDLE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pool")
    s.add_argument("--results", required=True, help="the run's results.jsonl")
    s.add_argument("--out", required=True, help="the pool file to write (outside the repo)")
    s.set_defaults(fn=cmd_pool)
    s = sub.add_parser("stage")
    s.add_argument("--from", dest="src", required=True, help="the exporter's bundle dir")
    s.add_argument("--results", required=True, help="the run's results.jsonl")
    s.set_defaults(fn=cmd_stage)
    s = sub.add_parser("oc")
    s.add_argument("--write", action="store_true")
    s.add_argument("--sims", type=int, default=OC_SIMS)
    s.add_argument("--boot", type=int, default=OC_BOOT)
    s.set_defaults(fn=cmd_oc)
    s = sub.add_parser("figure")
    s.add_argument("--out", default=FIGURE)
    s.set_defaults(fn=cmd_figure)
    for name, fn in (("plan", cmd_plan), ("band", cmd_band), ("score", cmd_score)):
        s = sub.add_parser(name)
        s.add_argument("--write", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("stage-pixels")
    s.add_argument("--from", dest="src", help="directory of verified-candidate <id>.jpg")
    s.add_argument("--index", help="the makelab2 archive's index.csv")
    s.add_argument("--unverified", help="directory of Panoramax-fetched <id>.jpg, used only "
                                        "for ids absent from the archive index")
    s.set_defaults(fn=cmd_stage_pixels)
    s = sub.add_parser("verify")
    s.add_argument("--records-only", action="store_true")
    s.set_defaults(fn=cmd_verify)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
