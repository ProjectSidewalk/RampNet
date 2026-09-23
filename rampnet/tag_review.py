"""Shared pieces of the curb-ramp tag review pass (RampNet 2.0 plan item 3, issue #86).

Three scripts use this module:

- ``scripts/analysis/tag_review_list.py`` builds the stratified review list
  (``benchmark/tag_review/review_list.csv``);
- ``scripts/analysis/tag_review_pull.py`` turns one rater's pass into a per-rater export
  (``benchmark/tag_review/<rater>.json``), either from production (``/v3/api/labelEdits``
  + ``/v3/api/validations``) or from a filled-in review sheet;
- ``scripts/analysis/tag_review_agreement.py`` compares two exports.

Everything here is standard library plus numpy, so the test suite exercises it on CPU
with synthetic rater files and no network.

**The export is self-describing.** Every export carries the rubric version, the sha256 of
the rubric text and the rubric text itself (``docs/tag_rubric_draft.md``, the part between
the ``rubric:begin`` / ``rubric:end`` markers), plus the sha256 of the review list it was
made against. ``agreement`` refuses to compare two exports made under different rubric
text or different lists, because a rubric that changed between passes silently averages two
definitions into one rate (``benchmark/RUBRICS.md`` §3 is the precedent).

**Absence is never inferred.** An item a rater did not review is ``reviewed: false`` and
drops out of every rate; a tag is a negative only on an item the rater reviewed, for a tag
the item's city offers, that the rater did not mark as "cannot judge".

Example::

    from rampnet import tag_review as tr
    a = tr.read_json("benchmark/tag_review/jonfroehlich.json")
    b = tr.read_json("benchmark/tag_review/mikey.json")
    report = tr.agreement(a, b)
    for row in report["per_tag"]:
        print(row["tag"], row["kappa"], row["n"])
"""
import hashlib
import json
import math
import re

import numpy as np

#: Format tag written into every export. Bump the suffix if a field changes meaning.
SCHEMA = "rampnet.tag_review/1"

#: The CurbRamp tags every deployment in the modal vocabulary offers (audit PR #175, §4).
CORE_TAGS = (
    "debris / pooled water",
    "missing tactile warning",
    "narrow",
    "not enough landing space",
    "not level with street",
    "points into traffic",
    "steep",
    "surface problem",
)
#: Offered by a few deployments only. In the rubric; rare in any list.
CITY_SPECIFIC_TAGS = ("not aligned with crosswalk", "parallel lines", "not visible")
#: Retired from the review vocabulary (plan §3). Never an applicable tag in the review.
RETIRED_TAGS = ("tactile warning",)

SEVERITIES = (1, 2, 3)
#: Put this in ``cannot_judge_tags`` to abstain on severity while still judging tags.
SEVERITY_TOKEN = "severity"
VERDICTS = ("agree", "disagree", "unsure")

RUBRIC_BEGIN = "<!-- rubric:begin -->"
RUBRIC_END = "<!-- rubric:end -->"
_VERSION_RE = re.compile(r"^\*\*Rubric version:\*\*\s*`([^`]+)`", re.M)


# ----------------------------------------------------------------------------- I/O

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def _round_floats(obj, ndigits=6):
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _round_floats(float(obj), ndigits)
    return obj


def write_json(path, obj):
    """Write ``obj`` as LF-terminated, key-sorted JSON with rounded floats.

    ``newline=""`` so a Windows run writes the same bytes as a Linux one, and rounded
    floats because numpy's repr differs by build; both are needed for a committed file
    whose hash means something."""
    text = json.dumps(_round_floats(obj), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_tag_list(value):
    """Tags from a list, a JSON list string (the API form) or a ``;``-joined string."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return sorted({str(t).strip() for t in value if str(t).strip()})
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        return sorted({str(t).strip() for t in json.loads(s) if str(t).strip()})
    return sorted({t.strip() for t in s.split(";") if t.strip()})


def join_tags(tags):
    return ";".join(sorted(tags))


# ----------------------------------------------------------------------------- rubric

def load_rubric(path):
    """``{version, sha256, text}`` for the rubric section of the rubric doc.

    Only the text between the markers is embedded and hashed, so editing the doc's
    status notes or the list hash outside them does not change the rubric identity."""
    with open(path, encoding="utf-8") as fh:
        doc = fh.read().replace("\r\n", "\n")
    try:
        start = doc.index(RUBRIC_BEGIN) + len(RUBRIC_BEGIN)
        end = doc.index(RUBRIC_END, start)
    except ValueError as e:
        raise ValueError(f"{path}: rubric markers {RUBRIC_BEGIN} / {RUBRIC_END} not found") from e
    text = doc[start:end].strip("\n") + "\n"
    m = _VERSION_RE.search(text)
    if not m:
        raise ValueError(f"{path}: no '**Rubric version:** `...`' line inside the rubric markers")
    return {"version": m.group(1), "sha256": sha256_bytes(text.encode("utf-8")), "text": text}


# ----------------------------------------------------------------------------- export

def _norm_verdict(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s == "nan":
        return None
    if s not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {v!r}")
    return s


def _norm_severity(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    sev = int(float(s))
    if sev not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {v!r}")
    return sev


def make_item(row, *, reviewed, verdict=None, tags_affirmed=None, severity=None,
              cannot_judge=False, cannot_judge_tags=(), note="", judged_at=None, evidence=None):
    """One export item from a review-list row plus the rater's judgment.

    ``row`` is a review-list row (dict of strings). ``tags_affirmed`` is the full tag set
    the rater left on the label; ``tags_added`` / ``tags_removed`` are derived against the
    tags the label carried when the list was built."""
    applicable = parse_tag_list(row["applicable_tags"])
    before = parse_tag_list(row.get("tags_at_list"))
    after = parse_tag_list(tags_affirmed) if reviewed else []
    unknown = sorted(set(after) - set(applicable))
    if unknown:
        raise ValueError(f"item {row['item_id']}: tags {unknown} are not offered in {row['city']}")
    cj_tags = parse_tag_list(list(cannot_judge_tags))
    verdict = _norm_verdict(verdict)
    return {
        "item_id": row["item_id"],
        "city": row["city"],
        "label_id": int(row["label_id"]),
        "label_uid": row["label_uid"],
        "pano_id": row["pano_id"],
        "strata": {"tag_state": row.get("tag_state"), "distance_band": row.get("distance_band")},
        "applicable_tags": applicable,
        "tags_at_list": before,
        "severity_at_list": _norm_severity(row.get("severity_at_list")),
        "reviewed": bool(reviewed),
        "verdict": verdict if reviewed else None,
        "tags_affirmed": after,
        "tags_added": sorted(set(after) - set(before)) if reviewed else [],
        "tags_removed": sorted(set(before) - set(after)) if reviewed else [],
        "severity": _norm_severity(severity) if reviewed else None,
        # "unsure" on prod is the item-level cannot-judge convention (protocol step 5).
        "cannot_judge": bool(reviewed and (cannot_judge or verdict == "unsure")),
        "cannot_judge_tags": cj_tags if reviewed else [],
        "note": (note or "").strip() if reviewed else "",
        "judged_at": judged_at if reviewed else None,
        "evidence": evidence or {},
    }


def make_export(*, rater, items, rubric, list_path_rel, list_sha256, method, exported_at,
                rater_user_id=None, window=None, extra=None):
    reviewed = sum(1 for it in items if it["reviewed"])
    out = {
        "schema": SCHEMA,
        "rater": rater,
        "rater_user_id": rater_user_id,
        "exported_at": exported_at,
        "method": method,
        "window": window or {},
        "review_list": {"path": list_path_rel, "sha256": list_sha256},
        "rubric": {"version": rubric["version"], "sha256": rubric["sha256"], "text": rubric["text"]},
        "counts": {"items": len(items), "reviewed": reviewed, "not_reviewed": len(items) - reviewed},
        "items": sorted(items, key=lambda it: it["item_id"]),
    }
    if extra:
        out.update(extra)
    return out


def validate_export(obj):
    """Raise ``ValueError`` if ``obj`` is not a well-formed export."""
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"schema {obj.get('schema')!r} is not {SCHEMA!r}")
    for key in ("rater", "rubric", "review_list", "items"):
        if key not in obj:
            raise ValueError(f"export is missing {key!r}")
    for key in ("version", "sha256", "text"):
        if not obj["rubric"].get(key):
            raise ValueError(f"export rubric is missing {key!r}")
    if sha256_bytes(obj["rubric"]["text"].encode("utf-8")) != obj["rubric"]["sha256"]:
        raise ValueError("embedded rubric text does not match its sha256")
    seen = set()
    for it in obj["items"]:
        if it["item_id"] in seen:
            raise ValueError(f"duplicate item {it['item_id']}")
        seen.add(it["item_id"])
        if it["label_uid"] != f"{it['city']}:{it['label_id']}":
            raise ValueError(f"item {it['item_id']}: label_uid must be city:label_id")
        bad = set(it["tags_affirmed"]) - set(it["applicable_tags"])
        if bad:
            raise ValueError(f"item {it['item_id']}: non-applicable tags {sorted(bad)}")
    return obj


# ----------------------------------------------------------------------------- pull

def _ts(s):
    """ISO timestamps from the API compare correctly as strings only after normalising;
    numpy's datetime64 handles the offsets the API emits."""
    return np.datetime64(_strip_tz(s)) if s else None


def _strip_tz(s):
    """'2026-09-19T03:30:28.851762-07:00' -> naive UTC ISO string."""
    s = str(s).strip().replace("Z", "+00:00")
    m = re.match(r"^(.*?)([+-]\d\d:\d\d)$", s)
    if not m:
        return s
    base, off = m.groups()
    sign = 1 if off[0] == "+" else -1
    hh, mm = int(off[1:3]), int(off[4:6])
    t = np.datetime64(base) - sign * (np.timedelta64(hh, "h") + np.timedelta64(mm, "m"))
    return str(t)


def items_from_prod(rows, edits, validations, *, since=None, until=None, sidecar=None):
    """Reconstruct a rater's pass from their production edits and validations.

    ``rows``: review-list rows. ``edits`` / ``validations``: dicts shaped like the
    ``/v3/api/labelEdits`` and ``/v3/api/validations`` CSV rows, already filtered to the
    rater's ``user_id`` and carrying a ``city`` key. ``since`` / ``until`` bound the pass
    (ISO strings); anything outside is ignored, so an earlier expert-validate of the same
    label does not count as this pass.

    Reviewed = the rater left an edit or a validation on the label inside the window.
    The affirmed tag set is the ``new_tags`` of the rater's last edit, else the tags the
    label had when the list was built (an Agree with no edit = "correct as is"; the gallery
    editor writes nothing when unchanged, which is why the protocol requires the Agree).
    Known limit: if someone else edited the label between list build and review, an
    unchanged-Agree is recorded against the list-time tags, not what the rater saw.
    """
    lo = _ts(since) if since else None
    hi = _ts(until) if until else None

    def inside(t):
        t = _ts(t)
        return t is not None and (lo is None or t >= lo) and (hi is None or t <= hi)

    by_key_e, by_key_v = {}, {}
    for e in edits:
        if inside(e.get("edit_time")):
            by_key_e.setdefault((e["city"], int(e["label_id"])), []).append(e)
    for v in validations:
        t = v.get("end_timestamp") or v.get("start_timestamp")
        if inside(t):
            by_key_v.setdefault((v["city"], int(v["label_id"])), []).append(v)
    sidecar = sidecar or {}
    items = []
    for row in rows:
        key = (row["city"], int(row["label_id"]))
        es = sorted(by_key_e.get(key, []), key=lambda e: _ts(e["edit_time"]))
        vs = sorted(by_key_v.get(key, []), key=lambda v: _ts(v.get("end_timestamp") or v.get("start_timestamp")))
        sc = sidecar.get(row["item_id"], {})
        if not es and not vs:
            items.append(make_item(row, reviewed=False))
            continue
        tags = parse_tag_list(es[-1]["new_tags"]) if es else parse_tag_list(row.get("tags_at_list"))
        sev = row.get("severity_at_list")
        if es and _norm_severity(es[-1].get("new_severity")) is not None:
            sev = es[-1]["new_severity"]
        verdict = vs[-1]["validation_result"] if vs else None
        times = [e["edit_time"] for e in es] + [v.get("end_timestamp") or v.get("start_timestamp") for v in vs]
        items.append(make_item(
            row, reviewed=True, verdict=verdict, tags_affirmed=tags, severity=sev,
            cannot_judge=_truthy(sc.get("cannot_judge")),
            cannot_judge_tags=parse_tag_list(sc.get("cannot_judge_tags")),
            note=sc.get("note", ""), judged_at=_strip_tz(max(times, key=_ts)),
            evidence={"edit_ids": [int(e["label_edit_id"]) for e in es if e.get("label_edit_id")],
                      "validation_ids": [int(v["label_validation_id"]) for v in vs
                                         if v.get("label_validation_id")],
                      "sources": sorted({e.get("source") for e in es if e.get("source")}
                                        | {v.get("source") for v in vs if v.get("source")})}))
    return items


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "x")


def items_from_sheet(rows, sheet_rows):
    """Items from a filled review sheet (``tag_review_pull.py sheet-template`` output).

    A sheet row counts as reviewed when its ``verdict`` is filled in. ``tags`` is the full
    tag set the rater leaves on the label (``;``-joined); the template pre-fills the
    list-time tags, the same anchor the production editor shows."""
    by_id = {r["item_id"]: r for r in sheet_rows}
    items = []
    for row in rows:
        s = by_id.get(row["item_id"])
        verdict = _norm_verdict(s.get("verdict")) if s else None
        if not verdict:
            items.append(make_item(row, reviewed=False))
            continue
        items.append(make_item(
            row, reviewed=True, verdict=verdict, tags_affirmed=parse_tag_list(s.get("tags")),
            severity=s.get("severity"), cannot_judge=_truthy(s.get("cannot_judge")),
            cannot_judge_tags=parse_tag_list(s.get("cannot_judge_tags")), note=s.get("note", ""),
            judged_at=(s.get("judged_at") or None)))
    return items


# ----------------------------------------------------------------------------- agreement

def cohen_kappa(x, y):
    """Cohen's kappa for two binary vectors. ``None`` when chance agreement is 1
    (both raters constant and identical: kappa is undefined, not 1)."""
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    n = len(x)
    if n == 0:
        return None
    po = float(np.mean(x == y))
    p1, p2 = float(x.mean()), float(y.mean())
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def weighted_kappa(x, y, categories=SEVERITIES, weights="quadratic"):
    """Cohen's weighted kappa on ordinal ratings (quadratic or linear disagreement weights)."""
    cats = list(categories)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    if len(x) == 0:
        return None
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[idx[a], idx[b]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) / (k - 1)) ** 2 if weights == "quadratic" else np.abs(i - j) / (k - 1)
    denom = float((w * exp).sum())
    if denom == 0:
        return None
    return 1.0 - float((w * obs).sum()) / denom


def kappa_ci_halfwidth(n, prevalence, kappa, sims=4000, seed=0):
    """Simulated 95% CI half-width of Cohen's kappa for ``n`` items.

    Both raters have marginal rate ``prevalence`` and true agreement ``kappa``; the 2x2
    cell probabilities follow from those two numbers. Used to size the review list: the
    half-width is governed by the expected number of positives, n * prevalence."""
    rng = np.random.default_rng(seed)
    p = prevalence
    q = p * (1 - p)
    probs = np.array([p * p + kappa * q, q * (1 - kappa), q * (1 - kappa), (1 - p) ** 2 + kappa * q])
    counts = rng.multinomial(n, probs, size=sims).astype(float)
    a, b, c, d = counts.T
    po = (a + d) / n
    p1, p2 = (a + b) / n, (a + c) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = (po - pe) / (1 - pe)
    kap = kap[np.isfinite(kap)]
    lo, hi = np.percentile(kap, [2.5, 97.5])
    return float((hi - lo) / 2)


def check_comparable(a, b, allow_rubric_mismatch=False):
    """Raise unless two exports were made under the same rubric text and the same list."""
    validate_export(a)
    validate_export(b)
    if a["review_list"]["sha256"] != b["review_list"]["sha256"]:
        raise ValueError("the two exports were made against different review lists "
                         f"({a['review_list']['sha256'][:12]} vs {b['review_list']['sha256'][:12]})")
    if a["rubric"]["sha256"] != b["rubric"]["sha256"] and not allow_rubric_mismatch:
        raise ValueError(f"rubric mismatch: {a['rater']} rated under {a['rubric']['version']} "
                         f"({a['rubric']['sha256'][:12]}), {b['rater']} under {b['rubric']['version']} "
                         f"({b['rubric']['sha256'][:12]})")


def _judgeable(it):
    return it["reviewed"] and not it["cannot_judge"] and it["verdict"] != "disagree"


def agreement(a, b, tags=None, group_by=None, allow_rubric_mismatch=False):
    """Per-tag Cohen's kappa, prevalence and severity weighted kappa between two exports.

    An item enters a tag's comparison only if both raters reviewed it, neither marked the
    whole item cannot-judge or voted it not a curb ramp (``disagree``), the tag is offered
    in the item's city, and neither rater marked that tag cannot-judge.

    ``group_by`` (``"tag_state"`` / ``"distance_band"``) adds the same per-tag table per
    stratum value."""
    check_comparable(a, b, allow_rubric_mismatch)
    ia = {it["item_id"]: it for it in a["items"]}
    ib = {it["item_id"]: it for it in b["items"]}
    common = sorted(set(ia) & set(ib))
    both_rev = [i for i in common if ia[i]["reviewed"] and ib[i]["reviewed"]]
    judged = [i for i in both_rev if _judgeable(ia[i]) and _judgeable(ib[i])]
    if tags is None:
        seen = set()
        for i in judged:
            seen |= set(ia[i]["applicable_tags"])
        tags = [t for t in CORE_TAGS if t in seen] + sorted(seen - set(CORE_TAGS))

    def table(ids):
        out = []
        for tag in tags:
            use = [i for i in ids if tag in ia[i]["applicable_tags"]
                   and tag not in ia[i]["cannot_judge_tags"] and tag not in ib[i]["cannot_judge_tags"]]
            x = [tag in ia[i]["tags_affirmed"] for i in use]
            y = [tag in ib[i]["tags_affirmed"] for i in use]
            n = len(use)
            pa, pb = sum(x), sum(y)
            both = sum(1 for u, v in zip(x, y) if u and v)
            out.append({
                "tag": tag, "n": n, "pos_a": pa, "pos_b": pb, "both": both,
                "prevalence_a": pa / n if n else None, "prevalence_b": pb / n if n else None,
                "prevalence": (pa + pb) / (2 * n) if n else None,
                "pct_agree": sum(1 for u, v in zip(x, y) if u == v) / n if n else None,
                "pos_specific_agree": 2 * both / (pa + pb) if (pa + pb) else None,
                "kappa": cohen_kappa(x, y) if n else None,
                "cannot_judge_a": sum(1 for i in ids if tag in ia[i]["cannot_judge_tags"]),
                "cannot_judge_b": sum(1 for i in ids if tag in ib[i]["cannot_judge_tags"]),
            })
        return out

    # "severity" in cannot_judge_tags is the per-item way to abstain on severity alone.
    sev = [i for i in judged if ia[i]["severity"] is not None and ib[i]["severity"] is not None
           and SEVERITY_TOKEN not in ia[i]["cannot_judge_tags"] and SEVERITY_TOKEN not in ib[i]["cannot_judge_tags"]]
    sx = [ia[i]["severity"] for i in sev]
    sy = [ib[i]["severity"] for i in sev]
    verdict_matrix = {}
    for i in both_rev:
        k = f"{ia[i]['verdict']}|{ib[i]['verdict']}"
        verdict_matrix[k] = verdict_matrix.get(k, 0) + 1
    report = {
        "rater_a": a["rater"], "rater_b": b["rater"],
        "rubric": {"a": a["rubric"]["version"], "b": b["rubric"]["version"],
                   "same_text": a["rubric"]["sha256"] == b["rubric"]["sha256"]},
        "review_list_sha256": a["review_list"]["sha256"],
        "items": {"common": len(common), "both_reviewed": len(both_rev), "both_judgeable": len(judged),
                  "cannot_judge_a": sum(1 for i in both_rev if ia[i]["cannot_judge"]),
                  "cannot_judge_b": sum(1 for i in both_rev if ib[i]["cannot_judge"])},
        "verdicts": dict(sorted(verdict_matrix.items())),
        "per_tag": table(judged),
        "severity": {
            "n": len(sev),
            "weighted_kappa_quadratic": weighted_kappa(sx, sy, weights="quadratic") if sev else None,
            "weighted_kappa_linear": weighted_kappa(sx, sy, weights="linear") if sev else None,
            "exact_agree": sum(1 for u, v in zip(sx, sy) if u == v) / len(sev) if sev else None,
            "mean_a": float(np.mean(sx)) if sev else None,
            "mean_b": float(np.mean(sy)) if sev else None,
            "confusion": {f"{u}|{v}": sum(1 for p, q in zip(sx, sy) if (p, q) == (u, v))
                          for u in SEVERITIES for v in SEVERITIES},
        },
    }
    if group_by:
        groups = {}
        for i in judged:
            groups.setdefault(ia[i]["strata"].get(group_by), []).append(i)
        report["by_" + group_by] = {str(g): table(ids) for g, ids in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    return report
