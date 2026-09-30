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
    a, b = files["assignments.json"], files["assignments__mikey.json"]
    print(cr.summary(a))
    print(cr.agreement(a, b, corners)["pairwise"])
"""
import json
import math
import re
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
#: pre-registered pilot rule (protocol, "Inter-rater agreement"); revisable once as rubric v2
PILOT_MIN_PAIRWISE = 0.90
PILOT_MIN_KAPPA = 0.6
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
    uncovered point inside the window and >= 1 m from an assigned ramp; elapsed_s >= 0;
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
            for r in sorted(used & set(ramps)):
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


def agreement(a, b, corners=None):
    """Inter-rater agreement between two passes, per the pre-registration.

    Over units both raters completed: pairwise same-ramp agreement on label pairs both
    raters put in ramps (overall, and split by whether the two raters had the same seed
    arm on the unit), Cohen's kappa on not_ramp vs ramp over labels neither marked
    unsure, and the uncovered-point counts. ``pilot`` applies the pre-registered rule."""
    check_comparable(a, b)
    ca, cb = complete_units(a), complete_units(b)
    ja, jb = cant_judge_units(a), cant_judge_units(b)
    common = sorted(set(ca) & set(cb))
    tot = {"all": [0, 0], "same_seed": [0, 0], "different_seed": [0, 0]}
    kx, ky = [], []
    unc_a = unc_b = 0
    diffs = {}
    per_unit = []
    for cid in common:
        ua, ub = ca[cid], cb[cid]
        pa, pb = pairs(ua), pairs(ub)
        both = sorted(set(pa) & set(pb))
        agree = sum(1 for p in both if pa[p] == pb[p])
        split = "same_seed" if ua.get("seed_arm") == ub.get("seed_arm") else "different_seed"
        for k in ("all", split):
            tot[k][0] += agree
            tot[k][1] += len(both)
        la, lb = ua.get("labels") or {}, ub.get("labels") or {}
        for key in sorted(set(la) & set(lb)):
            if UNSURE in (la[key], lb[key]):
                continue
            kx.append(la[key] == NOT_RAMP)
            ky.append(lb[key] == NOT_RAMP)
        na = sum(1 for p in ua.get("uncovered") or [] if not p.get("unsure"))
        nb = sum(1 for p in ub.get("uncovered") or [] if not p.get("unsure"))
        unc_a += na
        unc_b += nb
        diffs[abs(na - nb)] = diffs.get(abs(na - nb), 0) + 1
        per_unit.append({"corner_id": cid, "pairs": len(both), "agree": agree,
                         "seed_a": ua.get("seed_arm"), "seed_b": ub.get("seed_arm"),
                         "uncovered_a": na, "uncovered_b": nb})

    def rate(k):
        agree, n = tot[k]
        lo, hi = wilson_interval(agree, n)
        return {"pairs": n, "agree": agree, "rate": agree / n if n else None,
                "ci95": [lo, hi] if n else None}
    kappa = cohen_kappa(kx, ky) if kx else None
    pw = rate("all")
    passed = (pw["rate"] is not None and kappa is not None
              and pw["rate"] >= PILOT_MIN_PAIRWISE and kappa >= PILOT_MIN_KAPPA)
    return {"rater_a": a.get("rater"), "rater_b": b.get("rater"),
            "rubric_version": a.get("rubric_version"),
            "units": {"complete_a": len(ca), "complete_b": len(cb), "both": len(common)},
            "cant_judge": {"a": len(ja), "b": len(jb), "either": len(set(ja) | set(jb)),
                           "both": len(set(ja) & set(jb)),
                           # one pass completed the unit, the other said it could not be judged
                           "complete_a_cant_judge_b": sorted(set(ca) & set(jb)),
                           "cant_judge_a_complete_b": sorted(set(ja) & set(cb))},
            "pairwise": pw, "pairwise_same_seed": rate("same_seed"),
            "pairwise_different_seed": rate("different_seed"),
            "not_ramp_kappa": {"labels": len(kx), "not_ramp_a": sum(kx),
                               "not_ramp_b": sum(ky), "kappa": kappa},
            "uncovered": {"total_a": unc_a, "total_b": unc_b,
                          "abs_diff_per_unit": dict(sorted(diffs.items()))},
            "pilot": {"rule": f"pairwise >= {PILOT_MIN_PAIRWISE} and kappa >= {PILOT_MIN_KAPPA}",
                      "pass": passed if common else None},
            "per_unit": per_unit}


# ---------------------------------------------------------------- verdicts.json check

def verdict_consistency(assignments, corners, verdicts_panos, records_by_pid):
    """Labels that map pixel-exactly to a judged detection of a RampNet bundle, and whether
    the reviewer's class agrees with the verdict (True <-> a ramp key, False <-> not_ramp).
    ``verdicts_panos`` is verdicts.json's ``panos``; ``records_by_pid`` the bundle's
    records keyed by pano id. Unsure on either side is skipped."""
    labels = {lab["key"]: lab for c in corners for lab in c["labels"]}
    checked, disagree = 0, []
    for cid, u in sorted(complete_units(assignments).items()):
        for key, v in sorted((u.get("labels") or {}).items()):
            lab = labels.get(key)
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
