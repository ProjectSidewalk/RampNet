"""Turn one rater's tag review pass into ``benchmark/tag_review/<rater>.json`` (#86 item 3).

The export format and the rubric embedding live in ``rampnet/tag_review.py``; the rater's
steps are ``docs/tag_review_protocol.md``. Three subcommands:

    # 1. the production route (network): reconstruct the pass from the rater's own
    #    /v3/api/labelEdits and /v3/api/validations rows on every deployment in the list
    python scripts/analysis/tag_review_pull.py prod --rater jonfroehlich \\
        --since 2026-09-23T00:00:00Z --sidecar benchmark/tag_review/jonfroehlich__sidecar.csv

    # 2. a blank review sheet for the offline route (no production writes)
    python scripts/analysis/tag_review_pull.py sheet-template --out mikey__sheet.csv

    # 3. the offline route: a filled sheet -> export
    python scripts/analysis/tag_review_pull.py sheet --rater mikey \\
        --sheet benchmark/tag_review/mikey__sheet.csv

**What production can and cannot give back.** Tags, severity and the Agree / Disagree /
Unsure vote are all retrievable per label and per user. Two things are not: a per-tag
"cannot judge" and a free-text note (no API returns validation or gallery comments). Those
go in a small per-rater sidecar CSV (``item_id,cannot_judge,cannot_judge_tags,note``), which
this script merges. An item with no edit and no vote from the rater inside ``--since`` /
``--until`` is exported as ``reviewed: false`` and drops out of every rate.

The pulled API files' sha256 values are recorded in the export (``pulls``), so a later
re-pull can be checked against the one the committed export came from.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import os
import sys
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rampnet import tag_review as tr  # noqa: E402

DEFAULT_LIST = os.path.join(REPO, "benchmark", "tag_review", "review_list.csv")
DEFAULT_RUBRIC = os.path.join(REPO, "docs", "tag_rubric_draft.md")
LIST_REL = "benchmark/tag_review/review_list.csv"
UA = {"User-Agent": "Mozilla/5.0 (RampNet research; tag review pull)"}

#: Owner account ids by username (shared across deployments). ``--user-id`` for anyone else.
RATER_IDS = {
    "jonfroehlich": "549187e0-82c9-4014-a48d-31f18083d575",
    "mikey": "18b26a38-24ab-402d-a64e-158fc0bb8a8a",
}
#: No production URL in the sheet on purpose: the production editor shows the first rater's
#: edits, and the sheet route exists so the second rater does not see them.
SHEET_COLUMNS = ("item_id", "city", "label_uid", "pano_id", "gsv_url", "applicable_tags",
                 "tags", "verdict", "severity", "cannot_judge", "cannot_judge_tags", "note", "judged_at")


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _host(url):
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"


def fetch_rater_rows(rows, user_id, timeout=300):
    """(edits, validations, pulls) for ``user_id`` on every deployment the list touches."""
    import requests  # network only on this path

    hosts = {}
    for r in rows:
        hosts.setdefault(r["city"], _host(r["editor_url"]))
    edits, vals, pulls = [], [], {}
    for city, host in sorted(hosts.items()):
        for name, path in (("labelEdits", f"/v3/api/labelEdits?userId={user_id}&filetype=csv"),
                           ("validations", f"/v3/api/validations?userId={user_id}&labelType=CurbRamp&filetype=csv")):
            resp = requests.get(host + path, headers=UA, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
            pulls[f"{city}__{name}"] = {"url": host + path, "sha256": hashlib.sha256(data).hexdigest(),
                                        "bytes": len(data), "fetched_at": _now()}
            for rec in csv.DictReader(io.StringIO(data.decode("utf-8"))):
                if rec.get("user_id") and rec["user_id"] != user_id:
                    continue
                rec["city"] = city
                (edits if name == "labelEdits" else vals).append(rec)
    return edits, vals, pulls


def read_sidecar(path):
    if not path:
        return {}
    return {r["item_id"]: r for r in read_csv_rows(path)}


def _export(args, rows, items, method, window=None, extra=None, user_id=None):
    rubric = tr.load_rubric(args.rubric)
    obj = tr.make_export(rater=args.rater, rater_user_id=user_id, items=items, rubric=rubric,
                         list_path_rel=LIST_REL, list_sha256=tr.sha256_file(args.list), method=method,
                         exported_at=_now(), window=window, extra=extra)
    tr.validate_export(obj)
    out = args.out or os.path.join(REPO, "benchmark", "tag_review", f"{args.rater}.json")
    tr.write_json(out, obj)
    c = obj["counts"]
    print(f"wrote {out}: {c['reviewed']} of {c['items']} items reviewed, rubric {rubric['version']}")


def cmd_prod(args):
    rows = read_csv_rows(args.list)
    user_id = args.user_id or RATER_IDS.get(args.rater)
    if not user_id:
        raise SystemExit(f"no user id for {args.rater!r}; pass --user-id")
    edits, vals, pulls = fetch_rater_rows(rows, user_id)
    items = tr.items_from_prod(rows, edits, vals, since=args.since, until=args.until,
                               sidecar=read_sidecar(args.sidecar))
    _export(args, rows, items, "prod_pull", window={"since": args.since, "until": args.until},
            extra={"pulls": pulls}, user_id=user_id)


def cmd_sheet(args):
    rows = read_csv_rows(args.list)
    items = tr.items_from_sheet(rows, read_csv_rows(args.sheet))
    _export(args, rows, items, "review_sheet", extra={"sheet_sha256": tr.sha256_file(args.sheet)},
            user_id=args.user_id or RATER_IDS.get(args.rater))


def cmd_sheet_template(args):
    rows = read_csv_rows(args.list)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({"item_id": r["item_id"], "city": r["city"], "label_uid": r["label_uid"],
                    "pano_id": r["pano_id"], "gsv_url": r["gsv_url"],
                    "applicable_tags": r["applicable_tags"], "tags": r["tags_at_list"]})
    with open(args.out, "wb") as fh:
        fh.write(buf.getvalue().encode("utf-8"))
    print(f"wrote {args.out} ({len(rows)} rows; 'tags' pre-filled with the list-time tags)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rater=True):
        p.add_argument("--list", default=DEFAULT_LIST)
        if rater:
            p.add_argument("--rater", required=True, help="rater id, used as the file name")
            p.add_argument("--user-id", default=None, help="Project Sidewalk user id (Owners known)")
            p.add_argument("--rubric", default=DEFAULT_RUBRIC)
        p.add_argument("--out", default=None)

    p = sub.add_parser("prod", help="network: rebuild the pass from production")
    common(p)
    p.add_argument("--since", required=True, help="pass start, ISO time (UTC if no offset)")
    p.add_argument("--until", default=None)
    p.add_argument("--sidecar", default=None, help="item_id,cannot_judge,cannot_judge_tags,note CSV")
    p.set_defaults(func=cmd_prod)
    s = sub.add_parser("sheet", help="offline: a filled review sheet -> export")
    common(s)
    s.add_argument("--sheet", required=True)
    s.set_defaults(func=cmd_sheet)
    t = sub.add_parser("sheet-template", help="write a blank review sheet for the list")
    common(t, rater=False)
    t.set_defaults(func=cmd_sheet_template)
    args = ap.parse_args(argv)
    if args.cmd == "sheet-template" and not args.out:
        ap.error("sheet-template needs --out")
    args.func(args)


if __name__ == "__main__":
    main()
