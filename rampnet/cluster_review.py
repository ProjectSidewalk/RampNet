"""Corner-level cluster review (issue #224): load, validate, summarise and compare passes.

A review unit is one intersection or mid-block window; its labels open grouped the way a
seed clustering arm grouped them and the reviewer corrects the grouping, producing a
label -> ramp assignment on a frozen label snapshot (``benchmark/RUBRICS.md`` §6; the
pre-registered protocol is ``docs/cluster_review_protocol.md``). The auto-labeler's
exporter writes the bundle (``snapshot.json``, ``corners.jsonl``, crops, aerials);
``scripts/cluster_review_gallery.py`` renders the tool whose Export writes
``assignments.json`` (rater A) or ``assignments__<rater>.json``. This module is the
RampNet side of reading those files: standard library plus ``tag_review.cohen_kappa`` and
``validation.wilson_interval``, CPU only, no network.

The auto-labeler's scorer reads the same files as data and never imports this module, so
the schema here and the protocol's must stay in step.

Example::

    from rampnet import cluster_review as cr
    snap, corners, files = cr.load_bundle("benchmark/vancouver/cluster_review")
    a, b = files["assignments.json"], files["assignments__jon-retest.json"]
    print(cr.summary(a))
    rep = cr.agreement(a, b, corners, kind="intra")      # the same rater's delayed retest
    print(rep["pair_f1"], rep["pilot"])
"""
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

from rampnet.tag_review import cohen_kappa
from rampnet.validation import wilson_interval

SCHEMA = "rampnet.cluster_review/1"
SNAPSHOT_SCHEMA = "rampnet.cluster_review.snapshot/1"
RUBRIC_VERSION = 1
NOT_RAMP = "not_ramp"
UNSURE = "unsure"
RAMP_KEY = re.compile(r"^r\d+$")
#: an uncovered point this close to an assigned ramp is that ramp, not a second one
UNCOVERED_MIN_SEP_M = 1.0
#: the pilot rule as first pre-registered (protocol, "Inter-rater agreement") used this
#: Rand-style pairwise rate; Amendment 5 (B1) keeps the rate as a reported number, not a gate
PILOT_MIN_PAIRWISE = 0.90
#: pilot gate since Amendment 5 (B1): same-ramp pair F1 on the pilot units whose two passes
#: opened with DIFFERENT seeds, pooled AND per-unit mean, each >= this; plus the kappa leg
PILOT_MIN_PAIR_F1 = 0.90
PILOT_MIN_KAPPA = 0.6
#: Amendment 2: a unit's re-review starts >= this many days after its first-pass completion
WASHOUT_MIN_DAYS = 7.0
#: unit-level bootstrap of the pair F1 (Amendment 5): resamples and RNG seed
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 224
AGREEMENT_KINDS = ("intra", "inter")
EARTH_RADIUS_M = 6371008.8


def haversine_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------- loading

def load_corners(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bundle(bundle_dir):
    """(snapshot, corners, {file name: assignments}) for a cluster_review bundle dir.
    Every ``assignments*.json`` beside ``corners.jsonl`` is loaded; none may exist yet."""
    d = Path(bundle_dir)
    snapshot = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"{d / 'snapshot.json'}: schema {snapshot.get('schema')!r}, "
                         f"expected {SNAPSHOT_SCHEMA!r}")
    corners = load_corners(d / "corners.jsonl")
    files = {p.name: json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(d.glob("assignments*.json"))}
    return snapshot, corners, files


def rater_file_name(rater=None):
    """``assignments.json`` for rater A (no name), ``assignments__<rater>.json`` otherwise."""
    if not rater:
        return "assignments.json"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", rater):
        raise ValueError(f"rater name {rater!r}: letters, digits, '_', '.', '-' only")
    return f"assignments__{rater}.json"


# --------------------------------------------------------------------------- validation

def validate(assignments, corners, snapshot):
    """Every reason the file cannot be scored, as a list of strings (empty = valid).

    Rules (protocol, "Schemas"): the schema id; the snapshot sha256; an integer
    rubric_version; every corner id known; every label key in its unit; every value a
    ramp key, ``not_ramp`` or ``unsure``; every label of a COMPLETE unit assigned; every
    ramp key used by a label present in ``ramps`` and every ramp holding a label; every
    uncovered point inside the window and, in a COMPLETE unit, >= 1 m from an assigned ramp
    (the gallery refuses Export only there: an unplaced ramp sits at the mean of its labels,
    so regrouping an unfinished unit can move it onto a mark -- review S9); elapsed_s >= 0;
    ``opened_at`` / ``completed_at`` ISO 8601 times when present (Amendment 5, S3);
    ``cant_judge`` a boolean, never together with ``complete``, and only with a non-empty
    ``cant_judge_reason`` (protocol, Amendment 3).
    """
    problems = []
    if assignments.get("schema") != SCHEMA:
        problems.append(f"schema {assignments.get('schema')!r}, expected {SCHEMA!r}")
    want = (snapshot.get("labels") or {}).get("sha256")
    if assignments.get("snapshot_sha256") != want:
        problems.append(f"snapshot_sha256 {str(assignments.get('snapshot_sha256'))[:12]} is not "
                        f"the bundle's {str(want)[:12]}")
    if not isinstance(assignments.get("rubric_version"), int):
        problems.append(f"rubric_version {assignments.get('rubric_version')!r} is not an integer")
    by_id = {c["corner_id"]: c for c in corners}
    for cid, u in sorted((assignments.get("corners") or {}).items()):
        c = by_id.get(cid)
        if c is None:
            problems.append(f"{cid}: not a unit of this bundle")
            continue
        keys = {lab["key"] for lab in c["labels"]}
        labels = u.get("labels") or {}
        ramps = u.get("ramps") or {}
        for key, v in labels.items():
            if key not in keys:
                problems.append(f"{cid}: label {key} is not in the unit")
            if v not in (NOT_RAMP, UNSURE) and not (isinstance(v, str) and RAMP_KEY.match(v)):
                problems.append(f"{cid}: label {key} has value {v!r}")
            elif RAMP_KEY.match(v) and v not in ramps:
                problems.append(f"{cid}: label {key} names ramp {v}, which is not in ramps")
        used = {v for v in labels.values() if isinstance(v, str) and RAMP_KEY.match(v)}
        for r in sorted(set(ramps) - used):
            problems.append(f"{cid}: ramp {r} holds no label")
        if u.get("complete"):
            for key in sorted(keys - set(labels)):
                problems.append(f"{cid}: complete, but label {key} is unassigned")
        clat, clng, win = c["centre"]["lat"], c["centre"]["lng"], c.get("window_m", 30.0)
        for k, p in enumerate(u.get("uncovered") or []):
            d = haversine_m(clat, clng, p["lat"], p["lng"])
            if d > win:
                problems.append(f"{cid}: uncovered point {k} is {d:.1f} m from the centre "
                                f"(window {win:g} m)")
            for r in sorted(used & set(ramps)) if u.get("complete") else ():
                q = ramps[r]
                if haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]) < UNCOVERED_MIN_SEP_M:
                    problems.append(f"{cid}: uncovered point {k} sits on ramp {r}")
        for flag in ("inventory_seen", "edited_after_inventory"):
            if flag in u and not isinstance(u[flag], bool):
                problems.append(f"{cid}: {flag} {u[flag]!r} is not a boolean")
        if u.get("edited_after_inventory") and not u.get("inventory_seen"):
            problems.append(f"{cid}: edited_after_inventory without inventory_seen")
        e = u.get("elapsed_s", 0)
        if not isinstance(e, (int, float)) or e < 0:
            problems.append(f"{cid}: elapsed_s {e!r}")
        for f in ("opened_at", "completed_at"):
            if u.get(f) is not None and parse_time(u[f]) is None:
                problems.append(f"{cid}: {f} {u[f]!r} is not an ISO 8601 time with a zone")
        cj = u.get("cant_judge", False)
        reason = u.get("cant_judge_reason", "")
        if not isinstance(cj, bool):
            problems.append(f"{cid}: cant_judge {cj!r} is not a boolean")
        elif cj:
            if u.get("complete"):
                problems.append(f"{cid}: cant_judge and complete together")
            if not isinstance(reason, str) or not reason.strip():
                problems.append(f"{cid}: cant_judge without a cant_judge_reason")
        if "cant_judge_reason" in u and not isinstance(reason, str):
            problems.append(f"{cid}: cant_judge_reason {reason!r} is not a string")
    return problems


def require_valid(assignments, corners, snapshot):
    problems = validate(assignments, corners, snapshot)
    if problems:
        raise ValueError(f"{len(problems)} problem(s): " + "; ".join(problems[:10]))


# ----------------------------------------------------------------------------- reading

def complete_units(assignments):
    """Units that count: attested complete (a can't-judge unit never is -- validate())."""
    return {cid: u for cid, u in (assignments.get("corners") or {}).items()
            if u.get("complete") and not u.get("cant_judge")}


def cant_judge_units(assignments):
    """{corner_id: reason} for units the reviewer marked can't judge (Amendment 3)."""
    return {cid: (u.get("cant_judge_reason") or "").strip()
            for cid, u in (assignments.get("corners") or {}).items() if u.get("cant_judge") is True}


def is_ramp(v):
    return isinstance(v, str) and bool(RAMP_KEY.match(v))


def parse_time(t):
    """A timezone-aware datetime from the gallery's ISO 8601 (``...Z``) string, else None."""
    if not isinstance(t, str):
        return None
    try:
        d = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo is not None else None


def pairs(unit):
    """{(key_i, key_j): same_ramp} over label pairs (i < j) that both have a ramp key."""
    ramp = sorted((k, v) for k, v in (unit.get("labels") or {}).items() if is_ramp(v))
    return {(a, b): va == vb for i, (a, va) in enumerate(ramp) for b, vb in ramp[i + 1:]}


def summary(assignments):
    """Counts over one pass: units, labels by class, ramps, uncovered, elapsed seconds."""
    units = assignments.get("corners") or {}
    done = complete_units(assignments)
    cls = {"ramp": 0, NOT_RAMP: 0, UNSURE: 0}
    ramps = unc_sure = unc_unsure = 0
    by_type = {}
    for u in done.values():
        for v in (u.get("labels") or {}).values():
            cls["ramp" if is_ramp(v) else v] += 1
        ramps += len(u.get("ramps") or {})
        unc = u.get("uncovered") or []
        unc_sure += sum(1 for p in unc if not p.get("unsure"))
        unc_unsure += sum(1 for p in unc if p.get("unsure"))
        t = (u.get("stratum") or {}).get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    el = sorted(float(u.get("elapsed_s", 0)) for u in done.values())
    med = None
    if el:
        m = len(el) // 2
        med = el[m] if len(el) % 2 else (el[m - 1] + el[m]) / 2
    cj = cant_judge_units(assignments)
    cj_by_type = {}
    for cid in cj:
        t = ((units[cid].get("stratum") or {}).get("type")) or "?"
        cj_by_type[t] = cj_by_type.get(t, 0) + 1
    return {"units": len(units), "complete": len(done), "complete_by_type": by_type,
            "cant_judge": len(cj), "cant_judge_by_type": cj_by_type,
            "labels": cls, "ramps": ramps, "uncovered_sure": unc_sure,
            "uncovered_unsure": unc_unsure, "elapsed_s_median": med,
            "elapsed_s_total": sum(el) if el else 0.0,
            "rater": assignments.get("rater"), "seed_arm": assignments.get("seed_arm")}


# --------------------------------------------------------------------------- agreement

def check_comparable(a, b):
    """Raise unless two passes are the same schema, rubric version and snapshot."""
    for name, x in (("a", a), ("b", b)):
        if x.get("schema") != SCHEMA:
            raise ValueError(f"file {name}: schema {x.get('schema')!r}, expected {SCHEMA!r}")
    if a.get("rubric_version") != b.get("rubric_version"):
        raise ValueError(f"rubric_version differs ({a.get('rubric_version')} vs "
                         f"{b.get('rubric_version')}): passes under different rubrics are "
                         "not comparable")
    if a.get("snapshot_sha256") != b.get("snapshot_sha256"):
        raise ValueError("the two passes were made on different label snapshots")


def infer_kind(a, b):
    """'intra' when B is the same rater's delayed retest (Amendment 2: rater
    ``<name>-retest``, A's rater ``<name>`` or unnamed), else None -- say which then."""
    rb = str(b.get("rater") or "")
    if rb.endswith("-retest") and a.get("rater") in (None, "", rb[:-len("-retest")]):
        return "intra"
    return None


def pair_counts(ua, ub):
    """(same_a, same_b, same_both, rand_agree, rand_n) over label pairs of one unit where
    BOTH passes gave both labels a ramp key: each pass's same-ramp pairs, the pairs both put
    together, and the Rand-style agreement (both same or both different) with its count."""
    pa, pb = pairs(ua), pairs(ub)
    both = set(pa) & set(pb)
    return (sum(1 for p in both if pa[p]), sum(1 for p in both if pb[p]),
            sum(1 for p in both if pa[p] and pb[p]),
            sum(1 for p in both if pa[p] == pb[p]), len(both))


def pair_f1(rows):
    """(pooled, per-unit mean, units with a defined F1) of the same-ramp pair F1 over
    per-unit rows (same_a, same_b, same_both, ...). Pooled = 2 * sum(same_both) /
    (sum(same_a) + sum(same_b)); a unit where neither pass puts any two labels together has
    no defined F1 and is left out of the mean (and counted apart)."""
    num = sum(2 * r[2] for r in rows)
    den = sum(r[0] + r[1] for r in rows)
    per = [2 * r[2] / (r[0] + r[1]) for r in rows if r[0] + r[1]]
    return (num / den if den else None), (sum(per) / len(per) if per else None), len(per)


def bootstrap_pair_f1(rows, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Unit-level percentile bootstrap 95% CIs of (pooled, per-unit mean) pair F1: units are
    resampled with replacement, never pairs, because pairs within a unit are not
    independent. Deterministic for a given (rows, n, seed)."""
    if not rows:
        return None, None
    rng = random.Random(seed)
    pooled, mean = [], []
    for _ in range(n):
        p, m, _k = pair_f1([rows[rng.randrange(len(rows))] for _ in rows])
        if p is not None:
            pooled.append(p)
        if m is not None:
            mean.append(m)

    def pct(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(round(0.975 * (len(xs) - 1)))]]
    return pct(pooled), pct(mean)


def seed_pairs(corner, arm):
    """{(key_i, key_j): same seed group} over label pairs that both have a seed in `arm`."""
    g = {lab["key"]: (lab.get("seed_group") or {}).get(arm) for lab in corner["labels"]}
    ks = sorted(k for k, v in g.items() if v)
    return {(x, y): g[x] == g[y] for i, x in enumerate(ks) for y in ks[i + 1:]}


def seed_reference(corners, ids):
    """Pair F1 of the two seed arms against each other (deployed vs fusion) over the units
    ``ids``: what a rater who accepted every seed would score on different-seed units."""
    by_id = {c["corner_id"]: c for c in corners}
    rows = []
    for cid in ids:
        if cid in by_id:
            pa, pb = seed_pairs(by_id[cid], "deployed"), seed_pairs(by_id[cid], "fusion")
            both = set(pa) & set(pb)
            rows.append((sum(1 for p in both if pa[p]), sum(1 for p in both if pb[p]),
                         sum(1 for p in both if pa[p] and pb[p])))
    pooled, mean, k = pair_f1(rows)
    return {"units": len(rows), "units_defined": k, "pooled": pooled, "per_unit_mean": mean}


def kappa_leg(kx, ky):
    """(kappa, leg passes, undefined_identical). Kappa is undefined when both passes are
    constant and identical (or there is no label); that is complete agreement, so the leg
    passes (Amendment 5, S5). Otherwise it passes at kappa >= PILOT_MIN_KAPPA."""
    kappa = cohen_kappa(kx, ky) if kx else None
    undefined_identical = kappa is None and list(kx) == list(ky)
    return kappa, undefined_identical or (kappa is not None and kappa >= PILOT_MIN_KAPPA), \
        undefined_identical


def washout(ua, ub):
    """Days from the first pass's completion of a unit to the retest's first opening of it
    (``opened_at``; the retest's ``completed_at`` when it has no opening time, which
    overstates the washout), or None when either time is missing."""
    t0 = parse_time(ua.get("completed_at"))
    t1 = parse_time(ub.get("opened_at")) or parse_time(ub.get("completed_at"))
    if t0 is None or t1 is None:
        return None
    return (t1 - t0).total_seconds() / 86400.0


def agreement(a, b, corners=None, kind=None):
    """Agreement between two passes over the units both completed, per the pre-registration.

    ``kind`` is ``"intra"`` (the same rater's delayed retest, Amendment 2: test-retest
    consistency, NOT inter-rater reliability) or ``"inter"`` (two people). None infers
    ``intra`` from a ``<name>-retest`` rater, else ``agreement_kind`` stays None.

    Reported (Amendment 5): same-ramp pair F1, pooled and per-unit mean with unit-level
    bootstrap CIs, overall and split by whether the passes opened the unit with the same
    seed arm; the Rand-style pairwise rate (reported only, no longer a gate: B1); Cohen's
    kappa on not_ramp vs ramp over labels neither pass marked unsure; uncovered-point and
    can't-judge counts; and each unit's washout in days, listed when under 7 (S3).

    ``pilot`` applies the Amendment 5 gate to the PILOT units, read from ``corners``: on
    pilot units whose passes opened with different seeds, pair F1 pooled AND per-unit mean
    each >= PILOT_MIN_PAIR_F1; and the kappa leg (``kappa_leg``) over every pilot unit both
    completed. ``pass`` is None unless every pilot unit is complete or can't judge in both
    passes (S5: a partial pilot, any other selection, or no ``corners`` decides nothing).
    ``seed_reference`` is the seed arms' own pair F1 on those units: what accepting every
    seed would score."""
    check_comparable(a, b)
    if kind is None:
        kind = infer_kind(a, b)
    if kind is not None and kind not in AGREEMENT_KINDS:
        raise ValueError(f"kind {kind!r}: one of {AGREEMENT_KINDS}")
    ca, cb = complete_units(a), complete_units(b)
    ja, jb = cant_judge_units(a), cant_judge_units(b)
    common = sorted(set(ca) & set(cb))
    rows, split_of, kx, ky, wash = {}, {}, {}, {}, {}
    unc_a = unc_b = 0
    diffs = {}
    per_unit = []
    for cid in common:
        ua, ub = ca[cid], cb[cid]
        rows[cid] = sa, sb, sab, ra, rn = pair_counts(ua, ub)
        split_of[cid] = ("same_seed" if ua.get("seed_arm") == ub.get("seed_arm")
                         else "different_seed")
        la, lb = ua.get("labels") or {}, ub.get("labels") or {}
        keys = [k for k in sorted(set(la) & set(lb)) if UNSURE not in (la[k], lb[k])]
        kx[cid] = [la[k] == NOT_RAMP for k in keys]
        ky[cid] = [lb[k] == NOT_RAMP for k in keys]
        na = sum(1 for p in ua.get("uncovered") or [] if not p.get("unsure"))
        nb = sum(1 for p in ub.get("uncovered") or [] if not p.get("unsure"))
        unc_a += na
        unc_b += nb
        diffs[abs(na - nb)] = diffs.get(abs(na - nb), 0) + 1
        wash[cid] = washout(ua, ub)
        per_unit.append({"corner_id": cid, "pairs": rn, "agree": ra, "same_a": sa,
                         "same_b": sb, "same_both": sab,
                         "pair_f1": 2 * sab / (sa + sb) if sa + sb else None,
                         "seed_a": ua.get("seed_arm"), "seed_b": ub.get("seed_arm"),
                         "uncovered_a": na, "uncovered_b": nb,
                         "washout_days": None if wash[cid] is None else round(wash[cid], 2)})

    def rand_rate(ids):
        agree, n = sum(rows[c][3] for c in ids), sum(rows[c][4] for c in ids)
        lo, hi = wilson_interval(agree, n)
        return {"pairs": n, "agree": agree, "rate": agree / n if n else None,
                "ci95": [lo, hi] if n else None}

    def f1_block(ids):
        r = [rows[c] for c in ids]
        pooled, mean, k = pair_f1(r)
        ci_p, ci_m = bootstrap_pair_f1(r)
        return {"units": len(r), "units_defined": k, "pooled": pooled, "per_unit_mean": mean,
                "pooled_ci95": ci_p, "per_unit_mean_ci95": ci_m,
                "same_pairs_a": sum(x[0] for x in r), "same_pairs_b": sum(x[1] for x in r),
                "same_pairs_both": sum(x[2] for x in r)}

    def flat(vec, ids):
        return [v for c in ids for v in vec[c]]

    split = {k: [c for c in common if split_of[c] == k] for k in ("same_seed", "different_seed")}
    known = [d for d in wash.values() if d is not None]
    short = sorted(c for c, d in wash.items() if d is not None and d < WASHOUT_MIN_DAYS)
    kappa, _ok, k_undef = kappa_leg(flat(kx, common), flat(ky, common))
    pilot = {"rule": (f"pilot units opened with different seeds: pair F1 pooled >= "
                      f"{PILOT_MIN_PAIR_F1} and per-unit mean >= {PILOT_MIN_PAIR_F1}; kappa"
                      f"(not_ramp) >= {PILOT_MIN_KAPPA} over the pilot units, or undefined "
                      "because both passes are constant and identical (Amendment 5)"),
             "pass": None}
    if corners is None:
        pilot["why_none"] = "no corners given: the pilot units are unknown"
    else:
        pilot_ids = {c["corner_id"] for c in corners if c.get("pilot")}
        full = (bool(pilot_ids) and pilot_ids <= set(ca) | set(ja)
                and pilot_ids <= set(cb) | set(jb))
        p_common = sorted(pilot_ids & set(common))
        p_diff = [c for c in p_common if split_of[c] == "different_seed"]
        f1d = f1_block(p_diff)
        pk, pk_ok, pk_undef = kappa_leg(flat(kx, p_common), flat(ky, p_common))
        f1_ok = (f1d["pooled"] is not None and f1d["per_unit_mean"] is not None
                 and f1d["pooled"] >= PILOT_MIN_PAIR_F1
                 and f1d["per_unit_mean"] >= PILOT_MIN_PAIR_F1)
        pilot.update({
            "pilot_units": len(pilot_ids), "both_complete": len(p_common), "full_pilot": full,
            "pair_f1_different_seed": f1d, "seed_reference": seed_reference(corners, p_diff),
            # the gate reads every unit, retested under 7 days or not; this is the same pair
            # F1 without those units, reported beside it, never gated (Amendment 5a, N4)
            "pair_f1_different_seed_without_under_min": f1_block([c for c in p_diff
                                                                  if c not in short]),
            "kappa": pk, "kappa_undefined_identical": pk_undef,
            "pair_f1_leg": f1_ok, "kappa_leg": pk_ok,
            "pass": (f1_ok and pk_ok) if full else None,
            "why_none": None if full else ("not every pilot unit is complete or can't judge "
                                           "in both passes: a partial pilot decides nothing")})

    return {"agreement_kind": kind,
            "rater_a": a.get("rater"), "rater_b": b.get("rater"),
            "rubric_version": a.get("rubric_version"),
            "units": {"complete_a": len(ca), "complete_b": len(cb), "both": len(common)},
            "cant_judge": {"a": len(ja), "b": len(jb), "either": len(set(ja) | set(jb)),
                           "both": len(set(ja) & set(jb)),
                           # one pass completed the unit, the other said it could not be judged
                           "complete_a_cant_judge_b": sorted(set(ca) & set(jb)),
                           "cant_judge_a_complete_b": sorted(set(ja) & set(cb))},
            "pair_f1": f1_block(common), "pair_f1_same_seed": f1_block(split["same_seed"]),
            "pair_f1_different_seed": f1_block(split["different_seed"]),
            # Rand-style: reported only (Amendment 5, B1); its Wilson CI treats pairs as
            # independent, which they are not, so it is narrower than the truth
            "pairwise": rand_rate(common), "pairwise_same_seed": rand_rate(split["same_seed"]),
            "pairwise_different_seed": rand_rate(split["different_seed"]),
            "not_ramp_kappa": {"labels": len(flat(kx, common)),
                               "not_ramp_a": sum(flat(kx, common)),
                               "not_ramp_b": sum(flat(ky, common)),
                               "kappa": kappa, "undefined_identical": k_undef},
            "uncovered": {"total_a": unc_a, "total_b": unc_b,
                          "abs_diff_per_unit": dict(sorted(diffs.items()))},
            "washout": {"basis": "retest opened_at (else completed_at) minus first-pass "
                                 "completed_at, days",
                        "min_days": WASHOUT_MIN_DAYS, "units_with_times": len(known),
                        "shortest_days": round(min(known), 2) if known else None,
                        "under_min": short,
                        "missing": sorted(c for c, d in wash.items() if d is None),
                        # the agreement again without the units re-reviewed too soon (S3)
                        "pair_f1_without_under_min": f1_block([c for c in common
                                                               if c not in short])},
            "pilot": pilot,
            "per_unit": per_unit}


# ---------------------------------------------------------------- verdicts.json check

def verdict_consistency(assignments, corners, verdicts_panos, records_by_pid):
    """Labels that map pixel-exactly to a judged detection of a RampNet bundle, and whether
    the reviewer's class agrees with the verdict (True <-> a ramp key, False <-> not_ramp).
    ``verdicts_panos`` is verdicts.json's ``panos``; ``records_by_pid`` the bundle's
    records keyed by pano id. Unsure on either side is skipped."""
    # a label key is a per-city label_id: look it up in the unit's OWN corner, never in a
    # pool over every corner (review N4)
    labels = {c["corner_id"]: {lab["key"]: lab for lab in c["labels"]} for c in corners}
    checked, disagree = 0, []
    for cid, u in sorted(complete_units(assignments).items()):
        for key, v in sorted((u.get("labels") or {}).items()):
            lab = labels.get(cid, {}).get(key)
            if lab is None or v == UNSURE:
                continue
            rec, ver = records_by_pid.get(lab["pano_id"]), verdicts_panos.get(lab["pano_id"])
            if rec is None or ver is None:
                continue
            w, h = rec["pano"]["width"], rec["pano"]["height"]
            for i, d in enumerate(rec.get("detections", [])):
                if (round(d["x_normalized"] * w), round(d["y_normalized"] * h)) != \
                        (lab["pano_x"], lab["pano_y"]):
                    continue
                dets = ver.get("dets", [])
                verdict = dets[i] if i < len(dets) else None
                if verdict not in (True, False):
                    break
                checked += 1
                if verdict != is_ramp(v):
                    disagree.append({"corner_id": cid, "key": key, "pano_id": lab["pano_id"],
                                     "verdict": verdict, "review": v})
                break
    return {"checked": checked, "disagree": disagree}
