#!/usr/bin/env python3
"""`crosswalk_s8` pre-registration prep (RampNet#298): input snapshot, the held-out Juneau slice, candidate
training-point counts, card-count and power estimates. CPU only, no network, no rating, no training.

Plan and rules: `docs/crosswalk_s8.md`. Everything here is a draft input to that pre-registration; nothing in this
script trains, infers or builds a page.

Subcommands:

``snapshot``
    Copies the inputs that live outside this repo into ``analysis_out/crosswalk_s8/inputs/``: a compact pano table
    from the labeler's two Juneau `results.jsonl` files (position, sequence, camera, detection counts; the files
    themselves are 12 MB and gitignored in the labeler), the production Crosswalk validator's scores for all 3,579
    AI labels sent to juneau-test (a scratchpad file today), Jon's juneau-test votes and the seeded random sample.
    Records every source's sha256 in ``inputs_sha256.json``.

``slice``
    The held-out slice: K whole areas grown around seeded centres until each holds its share of the AI labels,
    then a buffer of every pano within ``--buffer-m`` of a slice pano. Writes ``slice/panos.csv`` (every pano's
    role), ``slice/areas.json`` and ``slice/summary.json``.

``pools``
    Candidate Juneau training points outside the slice and its buffer, by source and rule, before any audit or
    disc screen. Writes ``pools/candidates.csv`` and ``pools/summary.json``.

``cards``
    How many rating cards a three-seed-per-model comparison makes, measured on the 80 Juneau missed-crosswalk panos
    where six models' predictions already exist (s4 seeds 1-3 and s6 half-dose seeds 1-3, the closest stand-in for
    "s4 against a retrained model").

``power``
    Simulated chance that the primary clears, for several pano-sample sizes and effect sizes, on the slice's own
    s4 labels with Jon's votes where he voted and the validator's verdict as a stand-in elsewhere.

Usage::

    S=scripts/analysis/stage1_crosswalk_sdot_274
    python $S/s8_prep.py snapshot --records <labeler>/runs/juneau_crosswalk/results.jsonl \\
        <labeler>/runs/juneau_expansion_crosswalk/results.jsonl --validator <validator_xwalk>/results/all_labels_scored.csv \\
        --validator-docs <validator_xwalk>/REPORT.md <validator_xwalk>/results/tables.md \\
        --votes <labeler>/runs/juneau_crosswalk/validation/juneau_test_votes_2026-11-17.csv \\
        --random-sample <labeler>/runs/juneau_crosswalk/validation/random_sample_2026-11-17.json
    python $S/s8_prep.py slice
    python $S/s8_prep.py pools
    python $S/s8_prep.py cards
    python $S/s8_prep.py power
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "analysis_out", "crosswalk_s8")
INP = os.path.join(OUT, "inputs")
HARDNEG = os.path.join(REPO, "analysis_out", "stage1_crosswalk_detector_274", "hardneg")
RECALL = os.path.join(REPO, "analysis_out", "crosswalk_followups_285", "recall")
CITY = "juneau-ak"

SLICE_SEED = 298          # RampNet#298
Y_GATE = 0.5 + 5.625 / 180    # the labeler's Crosswalk range gate at send time (5.625 deg below the horizon)
Y_RIG = 0.5 + 49.0 / 180      # the labeler's 49 deg camera-rig mask (NADIR_MASK_DEG)
Y_25M = 0.532                 # recall_check's 25 m line
MERGE_R = 0.01                # score_274.PREFILL_R: one card per paint across models
NEAR_R = 0.05                 # the scorer's matching radius; s5's merge and disc-screen radius
NEG_CUT = 0.9999              # validator p_incorrect above which an AI label is a candidate negative (to be audited)
NEG_MIN_DIP = 8.4375          # ... and at least this far below the horizon (deg): a detector row-cell edge, ~17 m and nearer
POS_CUT = 0.9999              # validator p_correct above which an AI label is a candidate positive (to be audited)
SEAM_PER_SEQ = 10             # rear-seam negatives (x < 0.04 or > 0.96) kept per capture sequence
NEG_PER_PANO = 3              # negatives kept per pano
POS_PER_PANO = 2              # validator positives kept per pano
CAP_SEED = 298


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wrap_dx(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, 1 - d)


def enu(lat, lng, lat0, lng0):
    """Local metres east/north of (lat0, lng0); equirectangular, fine over a few km."""
    return ((np.asarray(lng) - lng0) * 111320.0 * math.cos(math.radians(lat0)),
            (np.asarray(lat) - lat0) * 110540.0)


def min_dist(src_xy, dst_xy):
    """For every row of dst_xy, the distance (m) to the nearest row of src_xy (brute force in blocks)."""
    out = np.full(len(dst_xy), np.inf)
    if len(src_xy) == 0:
        return out
    for i in range(0, len(dst_xy), 2000):
        d = dst_xy[i:i + 2000, None, :] - src_xy[None, :, :]
        out[i:i + 2000] = np.sqrt((d ** 2).sum(-1)).min(1)
    return out


def read_ids(path):
    return [x.strip().replace(".jpg", "") for x in open(path, encoding="utf-8").read().split() if x.strip()]


# ---------------------------------------------------------------------------------------------------- snapshot
def cmd_snapshot(a):
    os.makedirs(INP, exist_ok=True)
    rows = []
    for path in a.records:
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = r["pano"]
                conf = [d["confidence"] for d in r["detections"]]
                ys = [d["y_normalized"] for d in r["detections"]]
                rows.append({"pano_id": str(p["panorama_id"]), "run": run, "lat": p["lat"], "lng": p["lng"],
                             "sequence_id": p.get("sequence_id"), "camera_model": p.get("camera_model"),
                             "capture_date": p.get("capture_date"), "width": p.get("width"), "height": p.get("height"),
                             "n_det_005": len(conf),
                             "n_det_023_gated": sum(1 for c, y in zip(conf, ys) if c >= 0.23 and y >= Y_GATE)})
    pt = pd.DataFrame(rows)
    assert pt.pano_id.is_unique, "a pano appears in both runs"
    pt.to_csv(os.path.join(INP, "juneau_panos.csv"), index=False, lineterminator="\n")
    copies = {"validator_all_labels_scored.csv": a.validator, "juneau_test_votes_2026-11-17.csv": a.votes,
              "random_sample_2026-11-17.json": a.random_sample}
    for d in a.validator_docs:
        copies["validator_" + os.path.basename(d)] = d
    for name, src in copies.items():
        with open(src, "rb") as f:   # text inputs, stored with LF endings so the committed bytes match their hash
            data = f.read().replace(b"\r\n", b"\n")
        with open(os.path.join(INP, name), "wb") as f:
            f.write(data)
    meta = {"sources": {os.path.abspath(p): sha256(p) for p in list(a.records) + list(copies.values())},
            "written": {n: sha256(os.path.join(INP, n)) for n in sorted(list(copies) + ["juneau_panos.csv"])},
            "panos": len(pt), "panos_by_run": pt.run.value_counts().to_dict()}
    json.dump(meta, open(os.path.join(INP, "inputs_sha256.json"), "w", newline="\n"), indent=1)
    print(json.dumps(meta["panos_by_run"]), "panos;", len(copies), "files copied")


def load_inputs():
    pt = pd.read_csv(os.path.join(INP, "juneau_panos.csv"), dtype={"pano_id": str, "sequence_id": str})
    lab = pd.read_csv(os.path.join(INP, "validator_all_labels_scored.csv"), dtype={"pano_id": str})
    lab = lab.drop(columns=["validation_result", "comment", "dip_deg"], errors="ignore")
    votes = pd.read_csv(os.path.join(INP, "juneau_test_votes_2026-11-17.csv"), dtype={"pano_id": str})
    n_raw = len(votes)
    votes = votes.sort_values("validated_at").groupby("label_uid").tail(1)   # a label's latest vote, as the report
    lab = lab.merge(votes[["label_uid", "validation_result", "comment"]].rename(columns={"validation_result": "vote"}),
                    on="label_uid", how="left", validate="one_to_one")
    assert lab.label_uid.is_unique and lab.label_uid.str.startswith(CITY + ":").all()
    lab["x"] = lab.pano_x / lab.width
    lab["y"] = lab.pano_y / lab.height
    lab["dip_deg"] = (lab.y - 0.5) * 180
    rs = json.load(open(os.path.join(INP, "random_sample_2026-11-17.json")))
    rs_ids = set(rs["sample_kept"]) | set(rs["sample_dropped"])
    lab["random_sample"] = lab.label_id.isin(rs_ids)
    assert lab.pano_id.isin(pt.pano_id).all(), "an AI label on a pano not in the records"
    lat0, lng0 = pt.lat.mean(), pt.lng.mean()
    pt["e"], pt["n"] = enu(pt.lat, pt.lng, lat0, lng0)
    lab["e"], lab["n"] = enu(lab.lat, lab.lng, lat0, lng0)   # PS's stored label position (approximate)
    return pt, lab, n_raw


def legacy_sets():
    """Juneau panos already used by earlier rounds, by role."""
    h = HARDNEG
    return {
        "recall285": read_ids(os.path.join(RECALL, CITY, "draw.txt")),
        "confirm_draw": read_ids(os.path.join(h, "confirm_s3", "juneau", "draw.txt")),
        "fresh_eval_draw": read_ids(os.path.join(h, "s5_inputs", "juneau_eval_draw.txt")),
        "s5_mine_draw": read_ids(os.path.join(h, "s5_inputs", "juneau_mine_draw.txt")),
        "ps_labelled": pd.read_csv(os.path.join(h, "s5_inputs", "ps_labelled_panos.csv"), dtype=str).pano_id.tolist(),
    }


def recall285_marks():
    d = json.load(open(os.path.join(RECALL, "ratings", "recall285_verdicts_jon_2026-11-11.json"), encoding="utf-8"))
    rows = []
    for pid, e in d["panos"].items():
        if e.get("group") != CITY:
            continue
        for m in e.get("missed", []):
            rows.append({"pano_id": pid, "x": m["x"], "y": m["y"], "unsure": bool(m.get("unsure"))})
        if not e.get("missed") and e.get("no_missed"):
            rows.append({"pano_id": pid, "x": np.nan, "y": np.nan, "unsure": False, "empty": True})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------- slice
def grow_slice(pt, lab_per_pano, seed, areas, target, min_sep_m, cell_m):
    """K areas. Strata come from cell_m cells sorted by AI-label count: 'dense' = the fewest cells holding half
    the labels, 'other' = every other cell with a pano. Area k's centre is a pano drawn uniformly (seeded) from
    its stratum (area 0 dense, every later area other), at least min_sep_m from every pano already in the slice. The
    area takes the nearest not-yet-sliced panos in order of distance until it holds target / K of all AI labels,
    then every pano within that final radius."""
    rng = np.random.default_rng(seed)
    xy = pt[["e", "n"]].to_numpy()
    nl = lab_per_pano.reindex(pt.pano_id).fillna(0).to_numpy()
    cell = list(zip((pt.e // cell_m).astype(int), (pt.n // cell_m).astype(int)))
    cs = pd.Series(nl).groupby(pd.Series(cell)).sum().sort_values(ascending=False)
    cum = cs.cumsum() / cs.sum()
    dense_cells = set(cs.index[cum.shift(fill_value=0) < 0.5])
    stratum = np.array(["dense" if c in dense_cells else "other" for c in cell])
    in_slice = np.full(len(pt), -1)
    per_area = target * nl.sum() / areas
    out = []
    for k in range(areas):
        want = "dense" if k == 0 else "other"
        sl = in_slice >= 0
        dmin = min_dist(xy[sl], xy) if sl.any() else np.full(len(pt), np.inf)
        cand = np.flatnonzero((stratum == want) & ~sl & (dmin >= min_sep_m))
        if len(cand) == 0:
            raise SystemExit(f"area {k}: no '{want}' pano at least {min_sep_m} m from the slice; lower --min-sep-m")
        c = int(rng.choice(cand))
        d = np.sqrt(((xy - xy[c]) ** 2).sum(1))
        d[sl] = np.inf
        order = np.argsort(d, kind="stable")
        acc, r = 0.0, 0.0
        for i in order:
            if not np.isfinite(d[i]):
                break
            acc += nl[i]
            r = d[i]
            if acc >= per_area:
                break
        members = np.flatnonzero(d <= r)
        in_slice[members] = k
        out.append({"area": k, "stratum": want, "centre_pano": pt.pano_id.iloc[c], "centre_lat": pt.lat.iloc[c],
                    "centre_lng": pt.lng.iloc[c], "radius_m": round(float(r), 1), "panos": int(len(members)),
                    "ai_labels": int(nl[members].sum())})
    return in_slice, out, stratum


def cmd_slice(a):
    pt, lab, _ = load_inputs()
    lpp = lab.groupby("pano_id").size()
    in_slice, areas, stratum = grow_slice(pt, lpp, a.seed, a.areas, a.target, a.min_sep_m, a.cell_m)
    xy = pt[["e", "n"]].to_numpy()
    S = in_slice >= 0
    dS = min_dist(xy[S], xy)
    role = np.where(S, "slice", np.where(dS <= a.buffer_m, "buffer", "train"))
    pt["role"], pt["area"], pt["stratum"], pt["dist_to_slice_m"] = role, in_slice, stratum, np.round(dS, 1)
    # A training label whose stored position is near a slice pano could be the same crosswalk a slice pano sees.
    lab = lab.merge(pt[["pano_id", "role"]], on="pano_id", how="left", validate="many_to_one")
    lab["pos_dist_to_slice_m"] = np.round(min_dist(xy[S], lab[["e", "n"]].to_numpy()), 1)
    lab["near_slice_by_position"] = (lab.role == "train") & (lab.pos_dist_to_slice_m <= a.label_margin_m)
    leg = legacy_sets()
    for k, ids in leg.items():
        pt[k] = pt.pano_id.isin(ids)
    os.makedirs(os.path.join(OUT, "slice"), exist_ok=True)
    pt[["pano_id", "run", "camera_model", "sequence_id", "role", "area", "stratum", "dist_to_slice_m"]
       + list(leg)].to_csv(os.path.join(OUT, "slice", "panos.csv"), index=False, lineterminator="\n")
    lab[["label_uid", "pano_id", "role", "pos_dist_to_slice_m", "near_slice_by_position"]].to_csv(
        os.path.join(OUT, "slice", "labels.csv"), index=False, lineterminator="\n")

    marks = recall285_marks().merge(pt[["pano_id", "role"]], on="pano_id", how="left", validate="many_to_one")
    sure_near = marks[~marks.unsure & (marks.y >= Y_25M)]
    seq_slice = set(pt.sequence_id[S])
    summ = {"params": {"seed": a.seed, "areas": a.areas, "target_label_share": a.target, "min_sep_m": a.min_sep_m,
                       "cell_m": a.cell_m, "buffer_m": a.buffer_m, "label_margin_m": a.label_margin_m},
            "areas": areas, "by_role": {}}
    for r in ("slice", "buffer", "train"):
        p = pt[pt.role == r]
        l = lab[lab.role == r]
        summ["by_role"][r] = {
            "panos": len(p), "panos_with_ai_label": int(p.pano_id.isin(lab.pano_id).sum()),
            "gopro_panos": int((p.camera_model == "GoPro Max").sum()), "sequences": int(p.sequence_id.nunique()),
            "ai_labels": len(l), "ai_labels_gopro": int((l.camera_model == "GoPro Max").sum()),
            "validator_agree": int((l.verdict == "Agree").sum()), "validator_disagree": int((l.verdict == "Disagree").sum()),
            "jon_agree": int((l.vote == "Agree").sum()), "jon_disagree": int((l.vote == "Disagree").sum()),
            "jon_unsure": int((l.vote == "Unsure").sum()), "random_sample_labels": int(l.random_sample.sum()),
            "random_sample_voted": int((l.random_sample & l.vote.notna()).sum()),
            "recall285_panos": int(p.recall285.sum()),
            "recall285_sure_marks_within_25m": int((sure_near.role == r).sum()),
            "recall285_empty_panos": int(marks[marks.get("empty", False).fillna(False).astype(bool) & (marks.role == r)].pano_id.nunique()),
            **{f"{k}_panos": int(p[k].sum()) for k in leg if k != "recall285"},
        }
    summ["train_labels_near_slice_by_position"] = int(lab.near_slice_by_position.sum())
    summ["train_panos_sharing_a_sequence_with_the_slice"] = int(((pt.role == "train") & pt.sequence_id.isin(seq_slice)).sum())
    summ["sequences_total"] = int(pt.sequence_id.nunique())
    sl = lab[lab.role == "slice"]
    summ["slice_expected"] = {
        "s4_labels": len(sl),
        "real_crosswalk_labels_if_city_precision_0.72": round(0.72 * len(sl)),
        "look_alike_labels_if_city_precision_0.72": round(0.28 * len(sl)),
        "validator_agree_share": round(float((sl.verdict == "Agree").mean()), 3),
        "visible_crosswalks_within_25m_if_s4_recall_0.57": round(0.72 * len(sl) / 0.57),
        "note": "0.72 = stratified precision as sent (random sample); 0.57 = s4 recall on the 110 Juneau marks. "
                "Rough sizes, not measurements of the slice.",
    }
    # The primary's pano sample: a seeded simple random draw from the slice, plus a seeded reserve for panos that
    # turn out unusable (missing native, corrupt JPEG). Drawn now, before any s8 checkpoint exists.
    rng = np.random.default_rng(a.seed + 1)
    order = rng.permutation(pt.pano_id[S].to_numpy())
    sample, reserve = order[:a.sample], order[a.sample:a.sample + a.reserve]
    with open(os.path.join(OUT, "slice", "primary_sample.txt"), "w", newline="\n") as f:
        f.write("\n".join(sample) + "\n")
    with open(os.path.join(OUT, "slice", "primary_reserve.txt"), "w", newline="\n") as f:
        f.write("\n".join(reserve) + "\n")
    ls = lab[lab.pano_id.isin(sample)]
    n4 = len(ls)
    summ["primary_sample"] = {
        "panos": len(sample), "reserve": len(reserve), "draw_seed": a.seed + 1,
        "panos_with_s4_label": int(ls.pano_id.nunique()), "s4_labels": n4,
        "s4_labels_gopro": int((ls.camera_model == "GoPro Max").sum()),
        "jon_votes_inherited": int(ls.vote.notna().sum()),
        "recall285_sure_marks_within_25m": int(sure_near.pano_id.isin(sample).sum()),
        "est_cards_seed1_pair": round(n4 * a.cards_seed1),
        "est_cards_three_seeds_each": round(n4 * a.cards_three),
        "card_factors_from": "cards/summary.json (s4 vs s6 half dose on the 80 Juneau recall285 panos)",
    }
    json.dump(summ, open(os.path.join(OUT, "slice", "summary.json"), "w", newline="\n"), indent=1)
    json.dump(areas, open(os.path.join(OUT, "slice", "areas.json"), "w", newline="\n"), indent=1)
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------------------------------------------- pools
def cmd_pools(a):
    pt, lab, _ = load_inputs()
    sp = pd.read_csv(os.path.join(OUT, "slice", "panos.csv"), dtype={"pano_id": str, "sequence_id": str})
    sl = pd.read_csv(os.path.join(OUT, "slice", "labels.csv"), dtype={"pano_id": str})
    lab = lab.merge(sl[["label_uid", "role", "near_slice_by_position"]], on="label_uid", validate="one_to_one")
    lab = lab.merge(sp[["pano_id", "sequence_id"]].rename(columns={"sequence_id": "seq"}), on="pano_id",
                    validate="many_to_one")
    # Jon's verdicts on earlier pages (s4 cards on the confirmation and fresh draws) carry to an AI label within 0.01.
    pages = [os.path.join(HARDNEG, "confirm_s3", "juneau", "s4", "ratings.csv"),
             os.path.join(HARDNEG, "s5_inputs", "juneau_eval", "s4", "ratings.csv")]
    pv = pd.concat([pd.read_csv(p, dtype={"pano_id": str}) for p in pages], ignore_index=True)
    lab["page_verdict"] = None
    for pid, g in pv.groupby("pano_id"):
        idx = lab.index[lab.pano_id == pid]
        for i in idx:
            d = np.hypot(wrap_dx(g.x, lab.at[i, "x"]), g.y - lab.at[i, "y"])
            if (d <= MERGE_R).any():
                lab.at[i, "page_verdict"] = g.verdict.iloc[int(np.argmin(d))]
    # Jon's missed-crosswalk marks are never training data (approved 2026-11-15 for measuring only), so their panos
    # carry no training point; nor do the panos with live PS Crosswalk labels (held out since s4).
    leg = legacy_sets()
    lab["on_recall285_or_ps_pano"] = lab.pano_id.isin(set(leg["recall285"]) | set(leg["ps_labelled"]))
    eligible = (lab.role == "train") & ~lab.near_slice_by_position & ~lab.on_recall285_or_ps_pano
    src = np.full(len(lab), "none", dtype=object)
    jon_pos = lab.vote.eq("Agree") | (lab.vote.isna() & lab.page_verdict.eq("crosswalk"))
    jon_neg = lab.vote.eq("Disagree") | (lab.vote.isna() & lab.page_verdict.eq("not_crosswalk"))
    jon_uns = lab.vote.eq("Unsure") | (lab.vote.isna() & lab.page_verdict.eq("unsure"))
    unrated = lab.vote.isna() & lab.page_verdict.isna()
    src[jon_pos] = "jon_positive"
    src[jon_neg] = "jon_negative"
    src[jon_uns] = "jon_unsure"
    src[unrated & (lab.p_correct > POS_CUT)] = "validator_positive_candidate"
    src[unrated & (lab.p_incorrect > NEG_CUT) & (lab.dip_deg >= NEG_MIN_DIP)] = "validator_negative_candidate"
    lab["source"] = src
    # s5's mining panos keep their own Jon-rated, disc-rule, reviewed points (root_v5); PS-derived points there drop.
    r5 = {k: pd.read_csv(os.path.join(HARDNEG, "s5_inputs", "root_v5_inputs", f"{k}.csv"), dtype={"pano_id": str})
          for k in ("positives", "negatives", "ignore")}
    v5_panos = set().union(*[set(v.pano_id) for v in r5.values()])
    lab["on_root_v5_pano"] = lab.pano_id.isin(v5_panos)
    # Automatic disc screen for every negative candidate: none within NEAR_R of a label that is, or may be, a crosswalk.
    maybe_xw = jon_pos | jon_uns | (unrated & (lab.p_correct > 0.5))
    lab["neg_near_crosswalk"] = False
    for pid, g in lab.groupby("pano_id"):
        negs = g.index[g.source.isin(["jon_negative", "validator_negative_candidate"])]
        xs = g[maybe_xw.loc[g.index]]
        for i in negs:
            d = np.hypot(wrap_dx(xs.x, lab.at[i, "x"]), xs.y - lab.at[i, "y"])
            lab.at[i, "neg_near_crosswalk"] = bool((d <= NEAR_R).any())
    lab["rear_seam"] = (lab.x < 0.04) | (lab.x > 0.96)
    lab["eligible"] = eligible
    # ---- the planned recipe's counts (before the audits and the disc review, which can only remove points) ----
    rng = np.random.default_rng(CAP_SEED)
    lab["planned"] = ""
    e_idx = lab.index[eligible & ~lab.on_root_v5_pano]
    neg = lab.loc[e_idx][lab.loc[e_idx].source.isin(["jon_negative", "validator_negative_candidate"])
                         & ~lab.loc[e_idx].neg_near_crosswalk]
    neg = neg.assign(jon=neg.source.eq("jon_negative"), r=rng.random(len(neg)))
    neg = neg.sort_values(["jon", "r"], ascending=[False, True])           # Jon's first, then a seeded order
    seam_rank = neg[neg.rear_seam].groupby("seq").cumcount()
    neg = neg.drop(seam_rank.index[seam_rank >= SEAM_PER_SEQ])
    neg = neg[neg.groupby("pano_id").cumcount() < NEG_PER_PANO]
    lab.loc[neg.index, "planned"] = "negative"
    n_neg_juneau = len(neg) + int(len(r5["negatives"][r5["negatives"].pano_id.map(sp.set_index("pano_id").role) == "train"]))
    pos_j = lab.loc[e_idx][lab.loc[e_idx].source.eq("jon_positive")]
    lab.loc[pos_j.index, "planned"] = "positive"
    n_pos_v5 = int(len(r5["positives"][r5["positives"].pano_id.map(sp.set_index("pano_id").role) == "train"]))
    room = max(0, n_neg_juneau - len(pos_j) - n_pos_v5)                    # Juneau positives <= Juneau negatives
    vp = lab.loc[e_idx][lab.loc[e_idx].source.eq("validator_positive_candidate")]
    vp = vp.assign(r=rng.random(len(vp))).sort_values("r")
    vp = vp[vp.groupby("pano_id").cumcount() < POS_PER_PANO].head(room)
    lab.loc[vp.index, "planned"] = "positive"
    cols = ["label_uid", "pano_id", "seq", "camera_model", "x", "y", "dip_deg", "confidence", "p_correct",
            "p_incorrect", "vote", "page_verdict", "source", "role", "near_slice_by_position", "eligible",
            "on_root_v5_pano", "neg_near_crosswalk", "rear_seam", "planned"]
    os.makedirs(os.path.join(OUT, "pools"), exist_ok=True)
    lab[cols].to_csv(os.path.join(OUT, "pools", "candidates.csv"), index=False, lineterminator="\n")
    e = lab[eligible & ~lab.on_root_v5_pano]
    tab = e.groupby("source").agg(labels=("label_uid", "size"), panos=("pano_id", "nunique"),
                                  gopro=("camera_model", lambda s: int((s == "GoPro Max").sum())),
                                  rear_seam=("rear_seam", "sum"), near_crosswalk=("neg_near_crosswalk", "sum"))
    role = sp.set_index("pano_id").role
    r5c = {k: v.pano_id.map(role).value_counts().to_dict() for k, v in r5.items()}
    r5p = {k: int(v[v.pano_id.map(role) == "train"].pano_id.nunique()) for k, v in r5.items()}
    # In-sample precision of the validator against every Jon verdict (PS votes and carried page verdicts).
    jv = lab.vote.map({"Agree": "crosswalk", "Disagree": "not_crosswalk", "Unsure": "unsure"}).fillna(lab.page_verdict)
    sweep = []
    for kind, col, cuts in (("negative", "p_incorrect", (0.99, 0.999, 0.9999)), ("positive", "p_correct", (0.999, 0.9999, 0.99999))):
        for near in (False, True):
            for cut in cuts:
                m = (lab[col] > cut) & ((lab.dip_deg >= NEG_MIN_DIP) if near else True)
                v = jv[m].value_counts()
                good = v.get("not_crosswalk" if kind == "negative" else "crosswalk", 0)
                sweep.append({"kind": kind, "cut": cut, "near_field_only": near, "labels": int(m.sum()),
                              "jon_rated": int(v.sum()), "jon_agrees_with_validator": int(good),
                              "unrated_eligible": int((m & eligible & jv.isna()).sum())})
    planned = lab[lab.planned != ""]
    # Every other AI label on a planned pano becomes an ignore point, unless it sits within NEAR_R of a planned
    # positive (s5's merge rule: an ignore point that close would zero the positive's own loss).
    rest = lab[lab.pano_id.isin(planned.pano_id) & (lab.planned == "")]
    pos = planned[planned.planned == "positive"]
    near_pos = np.zeros(len(rest), bool)
    for k, (i, r) in enumerate(rest.iterrows()):
        g = pos[pos.pano_id == r.pano_id]
        near_pos[k] = bool(len(g)) and bool((np.hypot(wrap_dx(g.x, r.x), g.y - r.y) <= NEAR_R).any())
    summ = {
            "planned_ignore_points": {"other_ai_labels_on_planned_panos": int(len(rest)),
                                      "of_which_within_0.05_of_a_positive_(merged,_no_ignore_point)": int(near_pos.sum())},"cuts": {"validator_negative_p_incorrect_gt": NEG_CUT, "validator_negative_min_dip_deg": NEG_MIN_DIP,
                     "validator_positive_p_correct_gt": POS_CUT, "rear_seam_negatives_per_sequence": SEAM_PER_SEQ,
                     "negatives_per_pano": NEG_PER_PANO, "validator_positives_per_pano": POS_PER_PANO,
                     "juneau_positives_le_juneau_negatives": True, "cap_seed": CAP_SEED},
            "excluded": {"slice_labels": int((lab.role == "slice").sum()), "buffer_labels": int((lab.role == "buffer").sum()),
                         "train_labels_near_slice_by_position": int((lab.role.eq("train") & lab.near_slice_by_position).sum()),
                         "train_labels_on_recall285_or_ps_labelled_panos": int((lab.role.eq("train") & lab.on_recall285_or_ps_pano).sum()),
                         "train_labels_on_root_v5_panos": int((eligible & lab.on_root_v5_pano).sum())},
            "eligible_by_source": tab.reset_index().to_dict(orient="records"),
            "validator_cut_sweep": sweep,
            "root_v5_juneau_points_by_role": r5c, "root_v5_train_panos": r5p,
            "page_verdicts_carried": int(lab.page_verdict.notna().sum()),
            "planned_from_ps_labels": {
                "negatives_jon": int(((planned.planned == "negative") & (planned.source == "jon_negative")).sum()),
                "negatives_validator": int(((planned.planned == "negative") & (planned.source == "validator_negative_candidate")).sum()),
                "negatives_rear_seam": int(((planned.planned == "negative") & planned.rear_seam).sum()),
                "positives_jon": int(((planned.planned == "positive") & (planned.source == "jon_positive")).sum()),
                "positives_validator": int(((planned.planned == "positive") & (planned.source == "validator_positive_candidate")).sum()),
                "panos": int(planned.pano_id.nunique())},
            "planned_from_root_v5": {"positives": n_pos_v5, "negatives": n_neg_juneau - len(neg),
                                     "panos": int(len((set(r5["positives"].pano_id) | set(r5["negatives"].pano_id))
                                                      & set(role.index[role == "train"])))}}
    json.dump(summ, open(os.path.join(OUT, "pools", "summary.json"), "w", newline="\n"), indent=1, default=int)
    print(json.dumps(summ, indent=1, default=int))


# ------------------------------------------------------------------------------------------------------- cards
def load_preds(label):
    rows = []
    for line in open(os.path.join(RECALL, "preds", label, f"{CITY}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        preds = r["preds"] if isinstance(r["preds"], list) else json.loads(r["preds"])
        for x, y, c in preds:
            rows.append((str(r["pano_id"]), x, y, c))
    return pd.DataFrame(rows, columns=["pano_id", "x", "y", "conf"])


def merged_cards(models):
    """models: list of (name, family, df at threshold). Greedy union within MERGE_R per pano (score_274's rule)."""
    cards = []
    for name, fam, df in models:
        for r in df.itertuples():
            near = [c for c in cards if c["pano_id"] == r.pano_id
                    and math.hypot(min(abs(c["x"] - r.x), 1 - abs(c["x"] - r.x)), c["y"] - r.y) <= MERGE_R]
            if near:
                near[0]["by"].add(name)
            else:
                cards.append({"pano_id": r.pano_id, "x": r.x, "y": r.y, "by": {name}})
    return cards


def cmd_cards(a):
    base = [("crosswalk_s4_klone", 0.23), ("crosswalk_s4_seed2", 0.30), ("crosswalk_s4_seed3", 0.13)]
    new = [("crosswalk_s6half", 0.23), ("crosswalk_s6half_seed2", 0.23), ("crosswalk_s6half_seed3", 0.25)]
    ms = []
    for fam, lst in (("base", base), ("new", new)):
        for lab, t in lst:
            d = load_preds(lab)
            d = d[(d.conf >= t) & (d.y >= Y_GATE) & (d.y <= Y_RIG)]
            ms.append((lab, fam, d))
    panos = len(read_ids(os.path.join(RECALL, CITY, "draw.txt")))
    res = {"panos": panos, "filters": "threshold (recall-matched, as A4), range gate y >= 0.53125, rig mask y <= 0.772"}
    for scope, sub in (("seed1_pair", [ms[0], ms[3]]), ("three_seeds_each", ms)):
        cards = merged_cards(sub)
        names_b = {m[0] for m in sub if m[1] == "base"}
        names_n = {m[0] for m in sub if m[1] == "new"}
        k = np.array([len(c["by"] & names_n) for c in cards])
        j = np.array([len(c["by"] & names_b) for c in cards])
        res[scope] = {"base_seed1_detections": int(len(ms[0][2])), "union_cards": len(cards),
                      "cards_per_base_seed1_detection": round(len(cards) / len(ms[0][2]), 2),
                      "cards_with_unequal_counts": int((k != j).sum())}
    os.makedirs(os.path.join(OUT, "cards"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "cards", "summary.json"), "w", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------- power
def cmd_power(a):
    pt, lab, _ = load_inputs()
    sp = pd.read_csv(os.path.join(OUT, "slice", "panos.csv"), dtype={"pano_id": str})
    sl_panos = sp.pano_id[sp.role == "slice"].to_numpy()
    lab = lab[lab.pano_id.isin(sl_panos)]
    # Look-alike stand-in: Jon's vote where he voted, else the validator's verdict (raw Disagree precision ~0.81).
    la = np.where(lab.vote.notna(), lab.vote.eq("Disagree"), lab.verdict.eq("Disagree")).astype(int)
    per = pd.Series(la, index=lab.pano_id.to_numpy()).groupby(level=0).sum().reindex(sl_panos).fillna(0).to_numpy()
    rng = np.random.default_rng(a.seed)
    rows = []
    for n in a.sizes:
        n = min(n, len(per))
        for f in a.removed:
            for add in a.added:
                hits = 0
                for _ in range(a.sims):
                    s4 = per[rng.choice(len(per), n, replace=False)]
                    kept = rng.binomial(s4.astype(int), 1 - f)
                    lam = add * s4.sum() / n
                    s8 = kept + rng.poisson(lam, n)
                    d = (s8 - s4).astype(float)
                    bs = d[rng.integers(0, n, (a.boot, n))].mean(1) * 100
                    hits += np.quantile(bs, 0.975) < 0
                rows.append({"panos": n, "share_removed": f, "new_as_share_of_s4": add,
                             "s4_look_alikes_per_100_panos": round(per.mean() * 100, 1),
                             "p_primary_met": round(hits / a.sims, 3)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "power"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "power", "power.csv"), index=False, lineterminator="\n")
    json.dump({"seed": a.seed, "sims": a.sims, "boot": a.boot, "slice_panos": int(len(per)),
               "slice_look_alike_standin_per_100_panos": round(per.mean() * 100, 1),
               "note": "Seed-to-seed noise is not modelled; the three-seed mean primary should be tighter than this."},
              open(os.path.join(OUT, "power", "summary.json"), "w", newline="\n"), indent=1)
    print(df.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--records", nargs="+", required=True)
    s.add_argument("--validator", required=True)
    s.add_argument("--validator-docs", nargs="*", default=[])
    s.add_argument("--votes", required=True)
    s.add_argument("--random-sample", required=True)
    s = sub.add_parser("slice")
    s.add_argument("--seed", type=int, default=SLICE_SEED)
    s.add_argument("--areas", type=int, default=3)
    s.add_argument("--target", type=float, default=0.12, help="share of all AI labels in the slice")
    s.add_argument("--min-sep-m", type=float, default=400.0)
    s.add_argument("--cell-m", type=float, default=150.0, help="cell size for the dense/other strata")
    s.add_argument("--buffer-m", type=float, default=50.0)
    s.add_argument("--label-margin-m", type=float, default=35.0,
                   help="a training label whose stored position is this close to a slice pano is excluded")
    s.add_argument("--sample", type=int, default=200, help="slice panos in the primary's rated sample")
    s.add_argument("--reserve", type=int, default=50)
    s.add_argument("--cards-seed1", type=float, default=1.83, help="cards per s4 seed-1 detection (cards/summary.json)")
    s.add_argument("--cards-three", type=float, default=3.42, help="the same, three seeds per model")
    sub.add_parser("pools")
    sub.add_parser("cards")
    s = sub.add_parser("power")
    s.add_argument("--seed", type=int, default=SLICE_SEED)
    s.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200, 300])
    s.add_argument("--removed", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    s.add_argument("--added", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    s.add_argument("--sims", type=int, default=300)
    s.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    {"snapshot": cmd_snapshot, "slice": cmd_slice, "pools": cmd_pools, "cards": cmd_cards, "power": cmd_power}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
