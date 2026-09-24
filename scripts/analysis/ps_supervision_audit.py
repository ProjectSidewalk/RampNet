"""Audit of the Project Sidewalk supervision available for RampNet 2.0 (issue #86).

RampNet 2.0 wants to find curb ramps *and* tag, rate and measure them. The only large
supervised source for tags and severity is Project Sidewalk's own label store, which is
spread over ~60 per-city deployments, has changed vocabulary over time, and mixes human,
crowd-validated and AI-placed labels. The 2026-09-21/22 census on #86 sized it by hand
with scratch scripts; this script is the committed, re-runnable version, and
`docs/ps_supervision_audit.md` is its output.

Two steps:

    # 1. pull every deployment's labels, validations, label edits and tag vocabulary
    #    (~450 MB, cached under analysis_out/ps_audit/raw, gitignored; re-runs skip
    #    files already present unless --refresh)
    python scripts/analysis/ps_supervision_audit.py fetch

    # 2. reduce the cache to the tables + the doc
    python scripts/analysis/ps_supervision_audit.py report --cutoff 2018-04-29

    # optional: index the HF sidewalk-tagger-ai-validated CurbRamp set without
    # downloading its 30 GB zip (reads the zip's central directory by HTTP range)
    python scripts/analysis/ps_supervision_audit.py hf-index

Everything comes from the public `/v3/api/*` endpoints; no database access. The
deployments the cities API lists as *private* have no URL there, so they are reached by
hostname from the committed `PRIVATE_HOSTS` list (probed 2026-09-21; `--private-hosts`
overrides it, `--no-private` skips them).

The three supervision tiers sized here are the ones on #86: tier 1 = every human label,
tier 2 = labels the crowd validated as correct (`correct == true`, a *position* tier — the
validate UI has no tag control), tier 3 = labels placed by trusted raters. Tier 3 defaults
to the two Owner accounts; a role list (`role|username|user_id`, one per line, as exported
from `sidewalk_login.user_role`) widens it via `--trusted-users`. That list is not committed
because it names MTurk worker accounts.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time

import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
DEFAULT_CACHE = os.path.join(ANALYSIS_OUT, "ps_audit", "raw")
DEFAULT_TABLES = os.path.join(ANALYSIS_OUT, "ps_audit")
DEFAULT_DOC = os.path.join(REPO, "docs", "ps_supervision_audit.md")

CITIES_API = "https://sidewalk-sea.cs.washington.edu/v3/api/cities?filetype=csv"
UA = {"User-Agent": "Mozilla/5.0 (RampNet research; jonf@cs.uw.edu)"}

#: Deployments the cities API marks private and lists without a URL. Found by probing
#: `sidewalk-<name>.cs.washington.edu` for every private city id and a few known aliases
#: (2026-09-21). A host that stops answering is reported in the fetch manifest, not hidden.
PRIVATE_HOSTS = (
    "dc", "validation-study", "zurich", "zurich-infra3d", "taipei", "new-taipei", "keelung",
    "auckland", "cuenca", "burnaby", "walla-walla", "la", "kaohsiung", "taichung", "columbia",
    "west-chester", "tainan", "la-piedad-old", "winterthur-infra3d",
)

#: The SidewalkAI account: its labels are model output, not supervision. Excluded from
#: every human count by user id (its `high_quality_user` flag is true, so that flag cannot
#: be used to find it).
SIDEWALK_AI_USER = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"

#: Owner accounts (the default trusted tier). Ids are the same on every deployment because
#: `sidewalk_login` is shared.
OWNERS = {
    "549187e0-82c9-4014-a48d-31f18083d575": "jonfroehlich",
    "18b26a38-24ab-402d-a64e-158fc0bb8a8a": "mikey",
}

#: Tags entered the schema on 2018-04-29 (SidewalkWebpage evolution 14). A label placed
#: before that is pre-tag, not tag-negative, so the default cutoff for every tag statistic
#: is the schema date, by label placement date.
DEFAULT_CUTOFF = "2018-04-29"
#: Production has stored a browser crop of every label placed since this date.
DEFAULT_CROP_DATE = "2023-10-12"

LABEL_TYPES = ("CurbRamp", "NoCurbRamp")
#: Validation `source` values that carry a tag review, as opposed to a position vote.
TAG_REVIEW_SOURCES = ("ExpertValidate", "ExternalTagValidationASSETS2024")


# ----------------------------------------------------------------------------- fetch

def list_hosts(private_hosts, include_private=True):
    """(city_id, base_url, visibility) for every deployment to pull."""
    cities = pd.read_csv(io.StringIO(requests.get(CITIES_API, headers=UA, timeout=60).text))
    pub = cities[(cities.visibility == "public") & cities.url.notna()]
    hosts = [(r.city_id, r.url.rstrip("/"), "public") for r in pub.itertuples()]
    if include_private:
        hosts += [(h, f"https://sidewalk-{h}.cs.washington.edu", "private") for h in private_hosts]
    return hosts


def _endpoints(label_types):
    eps = [(f"rawLabels__{lt}", f"/v3/api/rawLabels?filetype=csv&labelType={lt}") for lt in label_types]
    eps += [(f"validations__{lt}", f"/v3/api/validations?filetype=csv&labelType={lt}") for lt in label_types]
    eps += [("labelEdits", "/v3/api/labelEdits?filetype=csv"),
            ("labelTags", "/v3/api/labelTags")]
    return eps


def _fetch_one(city, base, name, path, cache, refresh):
    ext = "json" if name == "labelTags" else "csv"
    dest = os.path.join(cache, f"{city}__{name}.{ext}")
    if os.path.exists(dest) and not refresh:
        return dict(city=city, endpoint=name, status="cached", bytes=os.path.getsize(dest))
    t0 = time.time()
    try:
        r = requests.get(base + path, headers=UA, timeout=900)
        r.raise_for_status()
        data = r.content
    except Exception as e:  # noqa: BLE001 - a dead host is a finding, not a crash
        return dict(city=city, endpoint=name, status="error", error=str(e)[:200], secs=round(time.time() - t0, 1))
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    return dict(city=city, endpoint=name, status="fetched", bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), secs=round(time.time() - t0, 1),
                fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


def cmd_fetch(args):
    os.makedirs(args.cache, exist_ok=True)
    hosts = list_hosts(args.private_hosts, include_private=not args.no_private)
    if args.only:
        hosts = [h for h in hosts if h[0] in set(args.only)]
    print(f"{len(hosts)} deployments, {len(_endpoints(args.label_types))} endpoints each -> {args.cache}")
    jobs = [(city, base, name, path) for city, base, vis in hosts for name, path in _endpoints(args.label_types)]
    manifest_path = os.path.join(args.cache, "fetch_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
    manifest.setdefault("hosts", {}).update({c: dict(url=b, visibility=v) for c, b, v in hosts})
    files = manifest.setdefault("files", {})
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_fetch_one, c, b, n, p, args.cache, args.refresh): (c, n) for c, b, n, p in jobs}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            key = f"{rec['city']}__{rec['endpoint']}"
            if rec["status"] != "cached":
                files[key] = rec
            print(f"  {key:45s} {rec['status']:8s} {rec.get('bytes', ''):>12} {rec.get('secs', '')}s {rec.get('error', '')}",
                  flush=True)
            with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
                json.dump(manifest, fh, indent=1, sort_keys=True)
    errs = [k for k, v in files.items() if v.get("status") == "error"]
    print(f"done; {len(errs)} errors" + (": " + ", ".join(sorted(errs)) if errs else ""))


# ----------------------------------------------------------------------------- loading

def _read_csv(path, usecols=None):
    """Read one cached CSV. Chicago's region names contain commas that the API does not
    quote (SidewalkWebpage#3756), so a strict parse can fail; then re-parse skipping bad
    lines and count them, so the loss is reported rather than silent."""
    kw = dict(low_memory=False, usecols=usecols)
    try:
        return pd.read_csv(path, **kw), 0
    except pd.errors.ParserError:
        bad = []
        df = pd.read_csv(path, engine="python", on_bad_lines=lambda row: bad.append(1) or None, usecols=usecols)
        return df, len(bad)


def _json_list(s):
    if isinstance(s, str) and s.startswith("["):
        try:
            return json.loads(s)
        except ValueError:
            return []
    return []


def _parse_times(values, what):
    """Parse an API timestamp column as ISO 8601 (UTC), refusing to drop any row silently.

    The API mixes `2019-06-01T00:00:00.123Z` and `2022-03-05T14:58:59Z` in one column.
    Without `format="ISO8601"`, pandas infers one format from the first value and coerces
    the other shape to NaT, and a NaT row then falls out of both the tag-era and pre-tag
    counts while staying in the total (499 labels in #175). Any NaT left after this parse
    is a new timestamp form or a missing value, so stop and name it rather than count it.

    Example: `_parse_times(df.time_created, "rawLabels time_created")`.
    """
    out = pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")
    bad = out.isna()
    if bad.any():
        sample = values[bad].iloc[0]
        raise SystemExit(f"{int(bad.sum())} of {len(out)} {what} values did not parse as ISO 8601 "
                         f"(first: {sample!r}); fix the parse before counting, or rows drop out of every era")
    return out


def load_labels(cache, label_type, exclude_users):
    """All cached rawLabels rows of one label type, across deployments, with derived columns."""
    frames, skipped = [], {}
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(f"__rawLabels__{label_type}.csv"):
            continue
        city = fn.split("__")[0]
        df, nbad = _read_csv(os.path.join(cache, fn))
        if nbad:
            skipped[city] = nbad
        df["city"] = city
        frames.append(df)
    if not frames:
        raise SystemExit(f"no rawLabels__{label_type} files in {cache}; run `fetch` first")
    df = pd.concat(frames, ignore_index=True)
    df["is_ai"] = df.user_id.isin(exclude_users)
    df["time_created"] = _parse_times(df.time_created, f"rawLabels__{label_type} time_created")
    df["year"] = df.time_created.dt.year
    df["taglist"] = df.tags.map(_json_list)
    df["n_tags"] = df.taglist.map(len)
    df["tagged"] = df.n_tags > 0
    df["sev"] = pd.to_numeric(df.severity, errors="coerce")
    df["rated"] = df.sev.notna()
    df["correct_true"] = df.correct.astype(str).str.lower().eq("true")
    df["correct_false"] = df.correct.astype(str).str.lower().eq("false")
    vals = df.validations.map(_json_list)
    df["human_agree"] = vals.map(lambda v: sum(1 for x in v if x.get("validator_type") == "Human" and x.get("validation") == "Agree"))
    df["human_disagree"] = vals.map(lambda v: sum(1 for x in v if x.get("validator_type") == "Human" and x.get("validation") == "Disagree"))
    df["human_validated"] = (df.human_agree + df.human_disagree) > 0
    df["human_net_ge2"] = (df.human_agree - df.human_disagree) >= 2
    df["agree_users"] = vals.map(lambda v: frozenset(x.get("user_id") for x in v
                                                     if x.get("validator_type") == "Human" and x.get("validation") == "Agree"))
    df["has_description"] = df.description.fillna("").astype(str).str.strip().ne("")
    return df, skipped


def load_validations(cache, label_type):
    frames = []
    for fn in sorted(os.listdir(cache)):
        if fn.endswith(f"__validations__{label_type}.csv"):
            df, _ = _read_csv(os.path.join(cache, fn),
                              usecols=["label_validation_id", "label_id", "validation_result", "user_id",
                                       "validator_type", "source", "end_timestamp"])
            df["city"] = fn.split("__")[0]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_edits(cache):
    frames = []
    for fn in sorted(os.listdir(cache)):
        if fn.endswith("__labelEdits.csv"):
            df, _ = _read_csv(os.path.join(cache, fn))
            df["city"] = fn.split("__")[0]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["old"] = df.old_tags.map(lambda s: set(_json_list(s)))
    df["new"] = df.new_tags.map(lambda s: set(_json_list(s)))
    df["edit_time"] = _parse_times(df.edit_time, "labelEdits edit_time")
    return df


def load_label_tags(cache):
    """{city: {label_type: [tag, ...]}} from each deployment's labelTags endpoint."""
    out = {}
    for fn in sorted(os.listdir(cache)):
        if fn.endswith("__labelTags.json"):
            try:
                d = json.load(open(os.path.join(cache, fn), encoding="utf-8"))
            except ValueError:
                continue
            by = {}
            for t in d.get("label_tags", []):
                by.setdefault(t["label_type"], []).append(t["tag"])
            out[fn.split("__")[0]] = by
    return out


def load_trusted(path):
    """role|username|user_id lines -> {user_id: (role, username)}; the Owners are always in."""
    trusted = {uid: ("Owner", name) for uid, name in OWNERS.items()}
    if path:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) == 3 and parts[2]:
                    trusted[parts[2]] = (parts[0], parts[1])
    return trusted


# ----------------------------------------------------------------------------- tables

def tier_row(name, d, cutoff, crop_date):
    era = d.time_created >= cutoff
    return dict(tier=name, labels=len(d), users=d.user_id.nunique(), cities=d.city.nunique(),
                rated=int(d.rated.sum()), tagged=int(d.tagged.sum()),
                correct=int(d.correct_true.sum()), tagged_and_correct=int((d.tagged & d.correct_true).sum()),
                human_net_agree_ge2=int(d.human_net_ge2.sum()),
                tag_era=int(era.sum()), tagged_tag_era=int((d.tagged & era).sum()),
                with_server_crop=int((d.time_created >= crop_date).sum()))


def tiers_table(h, trusted, cutoff, crop_date):
    owners = {u for u, (r, _) in trusted.items() if r == "Owner"}
    admins = owners | {u for u, (r, _) in trusted.items() if r == "Administrator"}
    everyone = set(trusted)
    rows = [tier_row("tier 1: every human label", h, cutoff, crop_date),
            tier_row("tier 2: crowd-validated correct (correct == true)", h[h.correct_true], cutoff, crop_date),
            tier_row("tier 2b: human net Agree >= 2", h[h.human_net_ge2], cutoff, crop_date),
            tier_row("tier 3: placed by an Owner", h[h.user_id.isin(owners)], cutoff, crop_date)]
    for uid in sorted(owners, key=lambda u: trusted[u][1]):
        rows.append(tier_row(f"    {trusted[uid][1]}", h[h.user_id == uid], cutoff, crop_date))
    rows.append(tier_row("tier 3v: Agree-validated by an Owner", h[h.agree_users.map(lambda s: bool(s & owners))], cutoff, crop_date))
    if len(admins) > len(owners):
        rows.append(tier_row("tier 3+: placed by Owner or Administrator", h[h.user_id.isin(admins)], cutoff, crop_date))
        rows.append(tier_row("tier 3v+: Agree-validated by Owner or Administrator",
                             h[h.agree_users.map(lambda s: bool(s & admins))], cutoff, crop_date))
    if len(everyone) > len(admins):
        rows.append(tier_row("tier 3++: placed by Owner, Administrator or Researcher", h[h.user_id.isin(everyone)], cutoff, crop_date))
    return pd.DataFrame(rows)


def by_year_table(h):
    g = h.groupby("year").agg(labels=("label_id", "size"), cities=("city", "nunique"),
                              private=("visibility", lambda s: int((s == "private").sum())),
                              rated=("rated", "sum"), tagged=("tagged", "sum"), correct=("correct_true", "sum"),
                              human_validated=("human_validated", "sum"))
    g["tagged_and_correct"] = h.groupby("year").apply(lambda d: int((d.tagged & d.correct_true).sum()), include_groups=False)
    g["pct_tagged"] = (100 * g.tagged / g.labels).round(1)
    g["pct_rated"] = (100 * g.rated / g.labels).round(1)
    return g.reset_index()


def by_city_table(h, cutoff):
    era = h[h.time_created >= cutoff]
    g = h.groupby(["city", "visibility"]).agg(labels=("label_id", "size"), users=("user_id", "nunique"),
                                              rated=("rated", "sum"), tagged=("tagged", "sum"),
                                              correct=("correct_true", "sum"), human_validated=("human_validated", "sum"),
                                              first=("time_created", "min"), last=("time_created", "max"))
    g["tag_era"] = era.groupby(["city", "visibility"]).size()
    g["tag_era"] = g.tag_era.fillna(0).astype(int)
    g["pct_tagged_tag_era"] = (100 * era.groupby(["city", "visibility"]).tagged.mean()).round(1)
    g["first"] = g["first"].dt.strftime("%Y-%m-%d")
    g["last"] = g["last"].dt.strftime("%Y-%m-%d")
    return g.reset_index().sort_values("labels", ascending=False)


def tag_table(h, cutoff, label_tags, label_type="CurbRamp"):
    """Per tag: how many tag-era labels carry it, in how many deployments, how many
    deployments currently list it in their vocabulary, and how many labels carry it in a
    deployment that no longer lists it (a tag can be retired or excluded per city while the
    labels keep it)."""
    era = h[h.time_created >= cutoff]
    counts = pd.Series([t for L in era.taglist for t in L]).value_counts()
    corr = pd.Series([t for L in era[era.correct_true].taglist for t in L]).value_counts()
    df = pd.DataFrame({"labels": counts, "labels_correct": corr}).fillna(0).astype(int)
    df["pct_of_tag_era"] = (100 * df.labels / len(era)).round(2)
    listed_by = {t: {c for c, by in label_tags.items() if t in by.get(label_type, [])} for t in df.index}
    rows = []
    for t in df.index:
        has = era[era.taglist.map(lambda L, t=t: t in L)]
        unlisted = has[~has.city.isin(listed_by[t])]
        rows.append(dict(cities_with_labels=has.city.nunique(), deployments_listing=len(listed_by[t]),
                         labels_where_unlisted=len(unlisted),
                         unlisted_in=", ".join(sorted(unlisted.city.unique())[:6]) + (" ..." if unlisted.city.nunique() > 6 else "")))
    df = pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
    return df.rename_axis("tag").reset_index()


def excluded_tags_table(label_tags, label_type):
    """Which of the union vocabulary each deployment does NOT list for this label type."""
    union = sorted({t for by in label_tags.values() for t in by.get(label_type, [])})
    rows = []
    for city, by in sorted(label_tags.items()):
        have = set(by.get(label_type, []))
        rows.append(dict(city=city, n_tags=len(have), excluded=", ".join(t for t in union if t not in have)))
    return pd.DataFrame(rows), union


def severity_tables(h, cutoff):
    d = h[h.sev.isin([1, 2, 3]) & (h.time_created >= cutoff)]
    y = d.sev.astype(int).values
    dist = pd.Series(y).value_counts().sort_index()
    rows = [dict(state="no tags", n=int((d.n_tags == 0).sum()), p_sev_ge2=round(float((y[d.n_tags.values == 0] >= 2).mean()), 3),
                 p_sev3=round(float((y[d.n_tags.values == 0] == 3).mean()), 3)),
            dict(state="any tag", n=int((d.n_tags > 0).sum()), p_sev_ge2=round(float((y[d.n_tags.values > 0] >= 2).mean()), 3),
                 p_sev3=round(float((y[d.n_tags.values > 0] == 3).mean()), 3))]
    for k in range(1, 5):
        m = d.n_tags.values == k
        if m.sum():
            rows.append(dict(state=f"{k} tag(s)", n=int(m.sum()), p_sev_ge2=round(float((y[m] >= 2).mean()), 3),
                             p_sev3=round(float((y[m] == 3).mean()), 3)))
    tags = sorted({t for L in d.taglist for t in L})
    X = pd.DataFrame({t: d.taglist.map(lambda L, t=t: t in L).astype(int).values for t in tags})
    for t in tags:
        m = X[t].values == 1
        if m.sum() >= 300:
            rows.append(dict(state=f"tag: {t}", n=int(m.sum()), p_sev_ge2=round(float((y[m] >= 2).mean()), 3),
                             p_sev3=round(float((y[m] == 3).mean()), 3)))
    kappa = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        m = X.sum(axis=1).values > 0
        if m.sum() < 50 or pd.Series(y[m]).value_counts().min() < 5:
            raise ImportError("too few tagged, rated labels for a 5-fold fit")
        pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X[m].values, y[m],
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
        kappa = dict(n=int(m.sum()), weighted_kappa=round(float(cohen_kappa_score(y[m], pred, weights="quadratic")), 3),
                     balanced_acc=round(float(balanced_accuracy_score(y[m], pred)), 3))
    except ImportError:
        pass
    return dist, pd.DataFrame(rows), kappa


def review_corpus_table(v):
    """Tag-reviewing validations (expert validate + the ASSETS'24 external pass) per city."""
    if v.empty:
        return pd.DataFrame()
    hv = v[v.validator_type == "Human"]
    rows = []
    for city, d in hv.groupby("city"):
        ev = d[d.source == "ExpertValidate"]
        ax = d[d.source == "ExternalTagValidationASSETS2024"]
        rows.append(dict(city=city, human_validations=len(d), expert_validations=len(ev), expert_labels=ev.label_id.nunique(),
                         expert_users=ev.user_id.nunique(), expert_agree=int((ev.validation_result == "Agree").sum()),
                         assets2024_validations=len(ax), assets2024_labels=ax.label_id.nunique()))
    df = pd.DataFrame(rows).sort_values("expert_validations", ascending=False)
    return df[(df.expert_validations > 0) | (df.assets2024_validations > 0)]


def edits_tables(e, trusted):
    if e.empty:
        return pd.DataFrame(), pd.DataFrame()
    e = e.copy()
    e["tags_changed"] = e.old != e.new
    e["from_empty"] = (e.old.map(len) == 0) & (e.new.map(len) > 0)
    e["added"] = (e.new - e.old).map(len)
    e["removed"] = (e.old - e.new).map(len)
    e["sev_changed"] = e.old_severity.fillna(-1) != e.new_severity.fillna(-1)
    e["type_changed"] = e.old_label_type != e.new_label_type
    by_src = e.groupby(["label_type", "source"]).agg(edits=("label_edit_id", "size"), labels=("label_id", "nunique"),
                                                     cities=("city", "nunique"), users=("user_id", "nunique"),
                                                     tags_changed=("tags_changed", "sum"), from_empty=("from_empty", "sum"),
                                                     tags_added=("added", "sum"), tags_removed=("removed", "sum"),
                                                     severity_changed=("sev_changed", "sum"), type_changed=("type_changed", "sum"),
                                                     first=("edit_time", "min"), last=("edit_time", "max")).reset_index()
    by_src["first"] = by_src["first"].dt.strftime("%Y-%m-%d")
    by_src["last"] = by_src["last"].dt.strftime("%Y-%m-%d")
    cr = e[e.label_type == "CurbRamp"]
    by_user = cr.groupby("user_id").agg(edits=("label_edit_id", "size"), labels=("label_id", "nunique"), cities=("city", "nunique"),
                                        sources=("source", lambda s: ", ".join(sorted(set(s)))),
                                        tags_changed=("tags_changed", "sum"), from_empty=("from_empty", "sum")).reset_index()
    # names only for Owner / Administrator accounts (the PS team); Researcher accounts stay anonymous
    by_user["who"] = by_user.user_id.map(lambda u: trusted[u][1] if trusted.get(u, ("",))[0] in ("Owner", "Administrator")
                                         else ("SidewalkAI" if u == SIDEWALK_AI_USER else ""))
    by_user["role"] = by_user.user_id.map(lambda u: trusted.get(u, ("", ""))[0])
    return by_src, by_user.sort_values("edits", ascending=False)


def rater_drift_table(h, trusted, cutoff, focus_tags):
    rows = []
    owners = [(u, n) for u, (r, n) in trusted.items() if r == "Owner"]
    for uid, name in sorted(owners, key=lambda x: x[1]):
        j = h[(h.user_id == uid) & (h.time_created >= cutoff)]
        for year, d in j.groupby("year"):
            row = dict(rater=name, year=int(year), labels=len(d), cities=d.city.nunique(),
                       pct_any_tag=round(100 * d.tagged.mean(), 1), pct_rated=round(100 * d.rated.mean(), 1))
            for t in focus_tags:
                row[t] = round(100 * d.taglist.map(lambda L, t=t: t in L).mean(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def hf_join_table(h, hf_index):
    """Join the HF sidewalk-tagger-ai-validated CurbRamp index to today's live labels."""
    hf = pd.read_csv(hf_index)
    tagcols = [c for c in hf.columns if c not in ("filename", "normalized_x", "normalized_y", "split", "city", "label_id", "census_city")]
    live = h[["city", "label_id", "taglist", "human_validated"]].copy()
    cmap = {}
    for hc in hf.city.dropna().unique():
        cands = [c for c in live.city.unique() if c == hc or c.startswith(hc + "-") or c.replace("-", "") == hc.replace("-", "")]
        cmap[hc] = cands[0] if len(cands) == 1 else None
    hf["city_id"] = hf.city.map(cmap)
    unmapped = {c: int(n) for c, n in hf[hf.city_id.isna()].city.value_counts().items()}
    hf = hf[hf.label_id.notna() & hf.city_id.notna()].copy()
    hf["label_id"] = hf.label_id.astype(int)
    j = hf.merge(live.rename(columns={"city": "city_id"}), on=["city_id", "label_id"], how="left", validate="one_to_one")
    ren = {"pooled-water": "debris / pooled water"}
    j["hf_tags"] = j.apply(lambda r: {ren.get(c, c.replace("-", " ")) for c in tagcols if r[c] in (1, True, 1.0)}, axis=1)
    m = j.taglist.notna()
    jj = j[m].copy()
    jj["live_tags"] = jj.taglist.map(set)
    same = int((jj.hf_tags == jj.live_tags).sum())
    added = jj.apply(lambda r: r.live_tags - r.hf_tags, axis=1)
    removed = jj.apply(lambda r: r.hf_tags - r.live_tags, axis=1)
    # leak check: does a pano appear in both splits?
    pano = h.set_index(["city", "label_id"]).pano_id
    jj["pano_id"] = [pano.get((c, l)) for c, l in zip(jj.city_id, jj.label_id)]
    split_panos = jj.groupby("pano_id").split.nunique()
    return dict(rows=int(len(hf) + sum(unmapped.values())), by_split=hf.split.value_counts().to_dict(), cities=hf.city.value_counts().to_dict(),
                city_map=cmap, unmapped_cities=unmapped, matched_live=int(m.sum()), identical_tags=same,
                pct_identical=round(100 * same / max(1, m.sum()), 1),
                labels_with_tags_added=int((added.map(len) > 0).sum()), labels_with_tags_removed=int((removed.map(len) > 0).sum()),
                top_added=pd.Series([t for s in added for t in s]).value_counts().head(6).to_dict(),
                top_removed=pd.Series([t for s in removed for t in s]).value_counts().head(6).to_dict(),
                matched_with_human_validation=int(jj.human_validated.sum()),
                hf_rows_with_no_tag=int((hf[tagcols].sum(axis=1) == 0).sum()),
                panos_in_both_splits=int((split_panos > 1).sum()), panos_total=int(split_panos.shape[0]))


# ----------------------------------------------------------------------------- report

def _md(df, index=False, floatfmt=None):
    """Markdown table without a tabulate dependency."""
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else (f"{v:,.0f}" if v.is_integer() and abs(v) >= 1000 and c != "year" else f"{v:g}"))
            elif isinstance(v, (int,)) and not isinstance(v, bool):
                cells.append(str(v) if c == "year" else f"{v:,}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def cmd_report(args):
    cache = args.cache
    manifest_path = os.path.join(cache, "fetch_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {"hosts": {}, "files": {}}
    vis = {c: v.get("visibility", "?") for c, v in manifest.get("hosts", {}).items()}
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    crop_date = pd.Timestamp(args.crop_date, tz="UTC")
    trusted = load_trusted(args.trusted_users)
    exclude = set(args.exclude_user)

    cr, skipped = load_labels(cache, "CurbRamp", exclude)
    cr["visibility"] = cr.city.map(vis).fillna("?")
    h = cr[~cr.is_ai]
    ncr, skipped_n = load_labels(cache, "NoCurbRamp", exclude)
    ncr["visibility"] = ncr.city.map(vis).fillna("?")
    hn = ncr[~ncr.is_ai]
    v = load_validations(cache, "CurbRamp")
    e = load_edits(cache)
    label_tags = load_label_tags(cache)

    fetched = [f for f in manifest.get("files", {}).values() if f.get("status") == "fetched"]
    fetch_dates = sorted({f["fetched_at"][:10] for f in fetched if "fetched_at" in f})
    errors = {k: f.get("error") for k, f in manifest.get("files", {}).items() if f.get("status") == "error"}

    T = {}
    T["tiers"] = tiers_table(h, trusted, cutoff, crop_date)
    T["by_year"] = by_year_table(h)
    T["by_city"] = by_city_table(h, cutoff)
    T["tags"] = tag_table(h, cutoff, label_tags)
    T["excluded_tags"], vocab = excluded_tags_table(label_tags, "CurbRamp")
    sev_dist, T["severity_by_tag_state"], kappa = severity_tables(h, cutoff)
    T["review_corpus"] = review_corpus_table(v)
    T["edits_by_source"], T["edits_by_user"] = edits_tables(e, trusted)
    focus = ["missing tactile warning", "points into traffic", "not level with street", "narrow", "steep",
             "not enough landing space", "surface problem"]
    T["rater_drift"] = rater_drift_table(h, trusted, cutoff, focus)
    hf = hf_join_table(h, args.hf_index) if args.hf_index and os.path.exists(args.hf_index) else None
    for name, df in T.items():
        _write_csv(df, os.path.join(args.tables, f"{name}.csv"))
    if hf:
        with open(os.path.join(args.tables, "hf_validated_join.json"), "w", encoding="utf-8", newline="") as fh:
            json.dump(hf, fh, indent=1, sort_keys=True)

    # ---- numbers the prose quotes
    n_ai = int(cr.is_ai.sum())
    era = h[h.time_created >= cutoff]
    pre = h[h.time_created < cutoff]
    excl_cities = T["excluded_tags"][T["excluded_tags"].excluded != ""]
    sev_pct = {int(k): round(100 * n / sev_dist.sum(), 1) for k, n in sev_dist.items()}
    ev = T["review_corpus"]
    ncr_tagged = int(hn.tagged.sum())
    hosts_pub = sum(1 for x in vis.values() if x == "public")
    hosts_priv = sum(1 for x in vis.values() if x == "private")

    L = []
    w = L.append
    w("# Project Sidewalk supervision audit for RampNet 2.0 (#86)\n")
    w("Generated by `scripts/analysis/ps_supervision_audit.py report` from the `/v3/api` pulls cached under "
      f"`analysis_out/ps_audit/raw/` (fetched {', '.join(fetch_dates) or 'unknown'}; the raw pulls are gitignored, "
      "the reduced tables beside this file under `analysis_out/ps_audit/*.csv` are committed). "
      "Regenerate with:\n")
    w("```bash\npython scripts/analysis/ps_supervision_audit.py fetch\n"
      f"python scripts/analysis/ps_supervision_audit.py report --cutoff {args.cutoff}"
      + (" --trusted-users <role list>" if args.trusted_users else "") + "\n```\n")
    w("**What this answers.** RampNet 2.0 needs supervision for tags, severity and (later) measurement, and the "
      "only large source is Project Sidewalk's label store. This audit sizes it, by supervision tier, by era, by "
      "deployment and by tag, and records the three things that make the raw counts misleading: tags are "
      "positive-unlabeled and rater-dependent (§4, §7), severity is a 3-level quality scale mostly carried by "
      "the tags (§5), and only a small, identifiable subset of labels has ever had its *tags* reviewed (§6). "
      "The numbers here replace the hand-run census on "
      "[#86](https://github.com/ProjectSidewalk/RampNet/issues/86).\n")

    w("## 1. Scope and provenance\n")
    w(f"- **Deployments:** {len(vis)} ({hosts_pub} public from the cities API, {hosts_priv} private by hostname from "
      "the committed `PRIVATE_HOSTS` list). The cities API lists private deployments without a URL, so a private "
      "host not in that list is invisible to this audit.\n"
      f"- **Endpoints per deployment:** `rawLabels` (CurbRamp, NoCurbRamp), `validations` (CurbRamp, NoCurbRamp), "
      "`labelEdits`, `labelTags`.\n"
      f"- **Excluded account:** `{SIDEWALK_AI_USER[:8]}…` (SidewalkAI), {n_ai:,} CurbRamp labels and "
      f"{int(ncr.is_ai.sum()):,} NoCurbRamp labels, dropped by user id from every human count below. Its "
      "`high_quality_user` flag is true, so that flag cannot be used to find it.\n"
      f"- **Tag era cutoff:** `{args.cutoff}` by label placement date (tags entered the schema on 2018-04-29, "
      "SidewalkWebpage evolution 14). A label placed before that is pre-tag, not tag-negative; every tag rate "
      f"below is over the {len(era):,} human CurbRamp labels on or after the cutoff, and the {len(pre):,} before "
      f"it ({pre.tagged.sum():,} of which carry tags added later) are position-only supervision.\n"
      f"- **Server crops:** production stores a browser crop for every label placed since `{args.crop_date}`; "
      f"{int((h.time_created >= crop_date).sum()):,} human CurbRamp labels have one. Anything older needs a re-cut "
      "from the pano store (plan item 2b).\n"
      f"- **Trusted raters:** {', '.join(sorted(n for r, n in trusted.values() if r == 'Owner'))} (Owners, committed)"
      + (f" plus {sum(1 for r, _ in trusted.values() if r == 'Administrator')} Administrators and "
         f"{sum(1 for r, _ in trusted.values() if r == 'Researcher')} Researchers from `--trusted-users` (not committed; "
         "it names MTurk worker accounts)." if args.trusted_users else
         ". No `--trusted-users` list was given, so the Administrator and Researcher rows are absent.") + "\n")
    if skipped or skipped_n:
        w(f"- **Parser losses:** rows skipped because the API does not quote commas in region names "
          f"(SidewalkWebpage#3756): CurbRamp {skipped or 'none'}, NoCurbRamp {skipped_n or 'none'}.\n")
    if errors:
        w(f"- **Fetch errors ({len(errors)}):** " + "; ".join(f"`{k}`: {e}" for k, e in sorted(errors.items())) + "\n")
    w("")

    w("## 2. Supervision tiers (CurbRamp, human labels)\n")
    w("Tier 2 is a **position** tier: the validate and mobile UIs have no tag control, so an Agree says the ramp is "
      "there, not that its tags are right. Tier 3 is by *placer*; the `3v` rows are by *validator*, which is the "
      "set an Owner has looked at (still without a tag control unless it came through expert validate, §6).\n")
    w(_md(T["tiers"]))
    w(f"NoCurbRamp, for the record: {len(hn):,} human labels, {ncr_tagged:,} tagged "
      f"({100 * hn.tagged.mean():.0f}%), {int(hn.correct_true.sum()):,} validated correct. Absence detection is "
      "out of scope for 2.0 (plan §5) but the supervision exists.\n")

    w("## 3. By year\n")
    w(_md(T["by_year"]))
    w(f"Labels on or after the cutoff: {len(era):,} ({int(era.tagged.sum()):,} tagged, "
      f"{int((era.tagged & era.correct_true).sum()):,} tagged and validated correct). "
      f"The private deployments contribute {int((h.visibility == 'private').sum()):,} of the {len(h):,} human labels; "
      f"`dc` alone is {int((h.city == 'dc').sum()):,}, almost all pre-cutoff.\n")

    w("## 4. Tags\n")
    modal = T["excluded_tags"].excluded.mode().iloc[0] if len(T["excluded_tags"]) else ""
    odd = T["excluded_tags"][T["excluded_tags"].excluded != modal]
    w(f"Union vocabulary across the {len(label_tags)} deployments' `labelTags` endpoints: {len(vocab)} CurbRamp tags. "
      "Each deployment lists its own subset (`config.excluded_tags`), so `deployments_listing` says how many "
      "currently offer the tag and `labels_where_unlisted` how many tag-era labels carry it in a deployment that "
      "does not, which is how a retired tag (`tactile warning`) or a city-specific one (`not aligned with "
      "crosswalk`, `parallel lines`) shows up. A zero count in a city can mean hidden, not absent. "
      "Counts are over human labels in the tag era; `labels_correct` is the subset validated correct (a position "
      "vote, see §2).\n")
    w(_md(T["tags"]))
    w(f"The modal vocabulary omits: {modal or 'nothing'}. Deployments whose list differs from that "
      f"({len(odd)}; the full per-deployment table is `excluded_tags.csv`):\n")
    w(_md(odd))

    w("## 5. Severity\n")
    w(f"CurbRamp severity is a 3-level quality scale, not 1–5: over the {int(sev_dist.sum()):,} rated human labels "
      f"in the tag era the split is " + ", ".join(f"{k} = {p}%" for k, p in sev_pct.items()) + ". "
      "It is largely carried by the tags:\n")
    w(_md(T["severity_by_tag_state"]))
    if kappa:
        w(f"A logistic regression from tag indicators to recorded severity, 5-fold cross-validated on the "
          f"{kappa['n']:,} tagged labels, reaches quadratic-weighted κ {kappa['weighted_kappa']} (balanced accuracy "
          f"{kappa['balanced_acc']}, chance 0.333). That is the S1 baseline in plan §Phase 1b; a severity head has "
          "to beat it on consensus severity, not on this recorded scale, whose two-rater κ was only 0.21 on the "
          "validation-study deployment (#86).\n")

    w("## 6. Tag-reviewed corpora\n")
    w("Only two validation sources carry a tag review: `ExpertValidate` (the `/expertValidate` UI, the only "
      "place a validator can edit tags) and `ExternalTagValidationASSETS2024` (the sidewalk-tagger-ai pass, "
      "frozen 2024-10-29). Everything else is a position vote.\n")
    if len(ev):
        w(_md(ev))
        w(f"Totals: expert validate {int(ev.expert_validations.sum()):,} validations on {int(ev.expert_labels.sum()):,} "
          f"labels across {len(ev[ev.expert_validations > 0])} deployments; ASSETS'24 "
          f"{int(ev.assets2024_validations.sum()):,} validations on {int(ev.assets2024_labels.sum()):,} labels across "
          f"{len(ev[ev.assets2024_validations > 0])} deployments.\n")
    w("**Label edits** (`/v3/api/labelEdits`, every tag, severity or type change with who made it and from which "
      "UI). `from_empty` counts edits that added tags to an untagged label: an untagged label that an editor "
      "*looked at and left empty* leaves no row at all, so absence is affirmed only where the source is a review "
      "pass that touched every label (the ASSETS'24 pass did; the gallery editor does not).\n")
    w(_md(T["edits_by_source"]))
    w("By editor (CurbRamp; names only for Owner and Administrator accounts):\n")
    w(_md(T["edits_by_user"].head(15).assign(user_id=lambda d: d.user_id.str[:8] + "…")))

    w("## 7. Rater drift on the Owners' own labels\n")
    w("Per-tag rate on labels the two Owners placed, by year. The reporting threshold moved, which is why a "
      "tag-gold anchor is 'trusted raters, recent years', not 'trusted raters'.\n")
    w(_md(T["rater_drift"]))

    if hf:
        w("## 8. The HF `sidewalk-tagger-ai-validated` CurbRamp set against today's labels\n")
        w(f"- {hf['rows']:,} crops ({hf['by_split']}), cities {hf['cities']}"
          + (f"; unmapped to a deployment: {hf['unmapped_cities']}" if hf['unmapped_cities'] else "") + ".\n"
          f"- {hf['matched_live']:,} match a live label by (city, label_id); {hf['identical_tags']:,} "
          f"({hf['pct_identical']}%) carry the same tag set today. Tags added since on "
          f"{hf['labels_with_tags_added']:,} labels (top: {hf['top_added']}), removed on "
          f"{hf['labels_with_tags_removed']:,} (top: {hf['top_removed']}).\n"
          f"- {hf['matched_with_human_validation']:,} of the matched labels have a human validation today.\n"
          f"- {hf['hf_rows_with_no_tag']:,} rows carry no tag; these are affirmed negatives (the ASSETS'24 pass "
          "edited every label it showed, §6).\n"
          f"- **Leak check:** {hf['panos_in_both_splits']:,} of {hf['panos_total']:,} panoramas appear in both "
          "train and test. The split is by label, not by panorama.\n")
    w("## 9. Tables\n")
    w("| file | contents |\n|---|---|\n" + "\n".join(f"| `analysis_out/ps_audit/{n}.csv` | {n.replace('_', ' ')} |" for n in T)
      + ("\n| `analysis_out/ps_audit/hf_validated_join.json` | section 8 summary |" if hf else "") + "\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(L))
    print(f"wrote {args.out} and {len(T)} tables to {args.tables}")


# ----------------------------------------------------------------------------- hf-index

class _RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, so zipfile can read a remote
    zip's central directory and a few small members without fetching the whole archive."""

    def __init__(self, url):
        self.url = requests.head(url, headers=UA, allow_redirects=True).url
        self.size = int(requests.head(self.url, headers=UA).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n <= 0 or self.pos >= self.size:
            return 0
        r = requests.get(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        r.raise_for_status()
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


HF_VALIDATED_ZIP = "https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/resolve/main/Validated/CurbRamp.zip"


def cmd_hf_index(args):
    import re
    import zipfile
    z = zipfile.ZipFile(io.BufferedReader(_RangeFile(args.url), buffer_size=1 << 20))
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    print(f"{len(z.namelist())} members, csv: {csvs}")
    frames = []
    for n in csvs:
        df = pd.read_csv(io.BytesIO(z.read(n)))
        df["split"] = os.path.splitext(os.path.basename(n))[0]
        frames.append(df)
    hf = pd.concat(frames, ignore_index=True)
    # e.g. gsv-pittsburgh-9291-CurbRamp.png, gsv-walla_walla-124-CurbRamp.png
    m = hf.filename.astype(str).str.extract(r"^[a-z]+-([a-z_\-]+?)-(\d+)-CurbRamp")
    hf["city"], hf["label_id"] = m[0].str.replace("_", "-"), pd.to_numeric(m[1]).astype("Int64")
    _write_csv(hf, args.out)
    print(f"wrote {args.out}: {len(hf)} rows, splits {hf.split.value_counts().to_dict()}")


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="pull every deployment's labels, validations, edits and tag list")
    f.add_argument("--cache", default=DEFAULT_CACHE)
    f.add_argument("--private-hosts", nargs="*", default=list(PRIVATE_HOSTS), help="hostname stems, sidewalk-<stem>.cs.washington.edu")
    f.add_argument("--no-private", action="store_true")
    f.add_argument("--only", nargs="*", help="city ids to fetch (default all)")
    f.add_argument("--label-types", nargs="*", default=list(LABEL_TYPES))
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--refresh", action="store_true", help="re-fetch files already in the cache")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("report", help="reduce the cache to tables and the doc")
    r.add_argument("--cache", default=DEFAULT_CACHE)
    r.add_argument("--tables", default=DEFAULT_TABLES)
    r.add_argument("--out", default=DEFAULT_DOC)
    r.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="tag-era start, by label placement date")
    r.add_argument("--crop-date", default=DEFAULT_CROP_DATE)
    r.add_argument("--exclude-user", nargs="*", default=[SIDEWALK_AI_USER], help="user ids to drop from human counts")
    r.add_argument("--trusted-users", help="role|username|user_id lines; Owners are always included")
    r.add_argument("--hf-index", default=os.path.join(DEFAULT_TABLES, "hf_validated_curbramp_index.csv"))
    r.set_defaults(func=cmd_report)

    x = sub.add_parser("hf-index", help="index the HF validated CurbRamp zip over HTTP range requests")
    x.add_argument("--url", default=HF_VALIDATED_ZIP)
    x.add_argument("--out", default=os.path.join(DEFAULT_TABLES, "hf_validated_curbramp_index.csv"))
    x.set_defaults(func=cmd_hf_index)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
