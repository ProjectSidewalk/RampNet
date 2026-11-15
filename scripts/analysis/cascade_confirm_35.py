"""Stage 0 of the open-weight cascade confirmation plan (#35, option B1): a CPU screen.

The plan (``open-weight-cascade-confirm-2026-11-15.md``, section 3) compares three policies
for curb-ramp labels on open imagery, each scored on the four GoPro Max splits
(``laurens_mapillary``, ``bayonne``, ``budapest_district5``, ``morgantown``):

* **N** -- RampNet's floor peaks at 0.30 or above (the labeler's operational tier).
* **T** -- one threshold ``t*`` in 0.10..0.55 for the rig, chosen on other splits.
* **C** -- N plus RampNet peaks in ``[T_lo, 0.30)`` that lie within ``r_gate`` of an
  open-weight YOLO pano box scoring ``c_min`` or more (``cascade_cost_35``'s rule).

Every policy has the labeler's 49 degree nadir rig mask applied to RampNet's peaks first
(``labeler: detectors.on_camera_rig``: a peak with normalized y above 0.5 + 49/180 is the
camera vehicle and is never submitted).

Steps, as the plan numbers them:

1. **Reproduce.** ``cascade_cost_35.py --check`` and ``cascade_argon_35.py --check`` (run
   separately; this script does not call them).
2. **Grid.** Arms ``y11l_pano``, ``y11x_pano_h200``, ``y26_pano``; T_lo in {0.10, 0.15,
   0.20, 0.25}; r_gate in {R/2, R, 2R}; c_min (absolute box score) in {0.05, 0.15, 0.25,
   0.40}: 144 candidates (``CANDIDATES``).
3. **Leave-one-split-out.** For each held-out split, pick ``t*`` and one C candidate on the
   other three by summed-count F1, then score N, T and C on the held-out split. The held-out
   counts are summed over the four folds into one pooled F1 per policy. The interval is a
   pano bootstrap stratified by split (2,000 resamples, seed 0) with **both selections
   repeated inside every resample**.
4. **Rig key test** (no ground truth). Per contributor, the share of RampNet peaks at 0.30
   or above that the nadir mask removes, and whether one cut separates ``laurens_mapillary``
   from ``morgantown`` and ``budapest_district5``.
5. **Concentration.** On ``laurens_mapillary``, net promoted ramps (promoted TP minus
   promoted FP, C against N) per panorama, and the Laurens gain recomputed leaving out each
   sequence in turn.

Rules (plan section 3), numbered here in the order the plan lists them:

* Rule 1, **drop T** if the rig key test fails, or the pooled held-out T minus N has a 95%
  lower bound at or below 0.
* Rule 2, **drop C** if pooled held-out C minus max(N, T) is below +0.03, or its lower
  bound is at or below 0, or C beats max(N, T) on fewer than two of the four splits.
* Rule 3, **flag "one instance"** if leaving out one sequence takes the Laurens gain below
  +0.02, or five panoramas carry over half the net promoted ramps.
* Rule 4, both dropped means **N** and the work stops.

**Lyon and Kaysville are never read.** They are the plan's Stage 1 and Stage 2
confirmation sets. The plan's step 4 also names Lyon's ecartip account; that half of the
test is left unevaluated here (see the doc's Amendments). The split list is hard-coded,
and nothing here globs ``benchmark/``.

Inputs, all committed: ``benchmark/<split>/{records.jsonl,verdicts.json}``,
``analysis_out/op_cache/<split>.json``, ``benchmark/model_detections/<arm>__<split>.json``.
CPU only; no network.

    python scripts/analysis/cascade_confirm_35.py            # write stage0.json + inputs.json, print the gates
    python scripts/analysis/cascade_confirm_35.py --check    # regenerate + byte-compare
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cascade_cost_35 as cc  # noqa: E402
from rampnet import roster  # noqa: E402
from rampnet.detection_eval import radius_sq_for  # noqa: E402
from complementarity import load_floor_peaks  # noqa: E402

SPLITS = ("laurens_mapillary", "bayonne", "budapest_district5", "morgantown")
LAURENS = "laurens_mapillary"
#: The splits the rig key test must put on the other side of the cut from Laurens.
OTHER_SIDE = ("morgantown", "budapest_district5")
ARMS = ("y11l_pano", "y11x_pano_h200", "y26_pano")
T_HI = 0.30
T_LO = (0.10, 0.15, 0.20, 0.25)
R = 0.022
R_GATE = (R / 2, R, 2 * R)
C_MIN = (0.05, 0.15, 0.25, 0.40)
CANDIDATES = tuple((a, tl, rg, cm) for a in ARMS for tl in T_LO for rg in R_GATE
                   for cm in C_MIN)
#: ``t*`` search grid: 0.10 to 0.55 in steps of 0.01 (46 values).
T_GRID = tuple(round(0.10 + 0.01 * i, 2) for i in range(46))
#: ``labeler: detectors.NADIR_MASK_Y`` (0.5 + 49/180); a peak strictly below it in the
#: image (larger y) is on the camera rig. Copied, not imported: the labeler is another repo.
NADIR_MASK_DEG = 49.0
NADIR_MASK_Y = 0.5 + NADIR_MASK_DEG / 180.0
BOOTSTRAP = 2000
SEED = 0
#: Rule thresholds (plan section 3).
C_MIN_GAIN = 0.03
C_MIN_WINS = 2
ONE_INSTANCE_GAIN = 0.02
TOP_PANOS = 5

OUT_DIR = os.path.join(cc.OUT, "cascade_confirm_35")
RESULT = "stage0.json"
#: sha256 of every input, written beside the result but NOT byte-compared by ``--check``:
#: a checkout with ``core.autocrlf=true`` rewrites ``records.jsonl`` line endings, which
#: changes its hash and nothing else.
INPUTS = "inputs.json"


# --------------------------------------------------------------------------- #
# pure helpers (tested on synthetic inputs)
# --------------------------------------------------------------------------- #
def on_camera_rig(y):
    """``labeler: detectors.on_camera_rig``: is a peak at normalized ``y`` on the rig?"""
    return y > NADIR_MASK_Y


def mask_rig(peaks):
    """Peaks with the rig ones dropped, order kept."""
    return [p for p in peaks if not on_camera_rig(p[1])]


def f1_of(c):
    """F1 of summed ``(tp, fp, recall tp, recall n_gt)`` along the last axis, as
    ``aggregate`` computes it (precision over every pano, recall over recall panos)."""
    return cc._prf(np.asarray(c))[2]


def select(train_counts):
    """Index of the best row by summed-count F1. ``train_counts`` is ``[..., K, 4]``.

    Ties go to the first row (lowest ``t`` for the threshold grid, grid order for C),
    which is ``cascade_cost_35``'s ``max(F1, -t)`` convention for thresholds.
    """
    f = f1_of(train_counts)
    return np.argmax(f, axis=-1)


def loso(N, T, C, weights=None):
    """Leave-one-split-out under pano weights.

    ``N[s]`` is ``[m_s, 4]``, ``T[s]`` ``[K_t, m_s, 4]``, ``C[s]`` ``[K_c, m_s, 4]`` per-pano
    counts; ``weights[s]`` is ``[B, m_s]`` (all ones when ``None``, one row). For each
    held-out split ``h`` the threshold and the C candidate are chosen on the summed counts
    of the other splits, then N, T and C are summed on ``h``. Returns ``[B]``/``[B, S]``
    arrays: per-split held-out counts, the chosen indices, and the pooled counts (summed
    over the held-out folds).
    """
    S = len(N)
    if weights is None:
        weights = [np.ones((1, N[s].shape[0]), dtype=np.int64) for s in range(S)]
    B = weights[0].shape[0]
    sn = [weights[s] @ N[s] for s in range(S)]                          # [B, 4]
    st = [np.einsum("bm,kmj->bkj", weights[s], T[s]) for s in range(S)]  # [B, Kt, 4]
    sc = [np.einsum("bm,kmj->bkj", weights[s], C[s]) for s in range(S)]  # [B, Kc, 4]
    rows = np.arange(B)
    out = {"n": np.zeros((B, S, 4), np.int64), "t": np.zeros((B, S, 4), np.int64),
           "c": np.zeros((B, S, 4), np.int64), "t_idx": np.zeros((B, S), np.int64),
           "c_idx": np.zeros((B, S), np.int64)}
    for h in range(S):
        tr = [s for s in range(S) if s != h]
        ti = select(sum(st[s] for s in tr))
        ci = select(sum(sc[s] for s in tr))
        out["t_idx"][:, h], out["c_idx"][:, h] = ti, ci
        out["n"][:, h] = sn[h]
        out["t"][:, h] = st[h][rows, ti]
        out["c"][:, h] = sc[h][rows, ci]
    for k in ("n", "t", "c"):
        out[k + "_pool"] = out[k].sum(axis=1)
    return out


def deltas(o):
    """Pooled and per-split F1 deltas from a ``loso`` result (``[B]`` / ``[B, S]``)."""
    fn, ft, fc = f1_of(o["n_pool"]), f1_of(o["t_pool"]), f1_of(o["c_pool"])
    sn, st, sc = f1_of(o["n"]), f1_of(o["t"]), f1_of(o["c"])
    return {"F1_N": fn, "F1_T": ft, "F1_C": fc,
            "T_minus_N": ft - fn, "C_minus_maxNT": fc - np.maximum(fn, ft),
            "C_minus_N": fc - fn,
            "split_F1_N": sn, "split_F1_T": st, "split_F1_C": sc,
            "split_C_minus_maxNT": sc - np.maximum(sn, st),
            "split_T_minus_N": st - sn}


def ci95(v):
    """Percentile 95% interval, ``cascade_cost_35._ci95``'s indexing."""
    return cc._ci95(v)


def cut_separates(side_a, side_b):
    """Does one threshold put every value of ``side_a`` strictly on one side and every
    value of ``side_b`` strictly on the other? Either orientation. Empty sides fail."""
    if not side_a or not side_b:
        return False
    return min(side_a) > max(side_b) or max(side_a) < min(side_b)


def top_share(net, k=TOP_PANOS):
    """Share of the total net promoted ramps carried by the ``k`` largest panoramas.

    ``None`` when the total is not positive (there is nothing to concentrate).
    Example: nets [5, 1, 1, 1, 1, 1, 0] -> total 10, top 5 = 9, share 0.9.
    """
    total = sum(net)
    if total <= 0:
        return None
    return sum(sorted(net, reverse=True)[:k]) / total


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes():
    """``{repo-relative path: sha256}`` for every file Stage 0 reads."""
    return {p.replace(os.sep, "/"): _sha(os.path.join(cc.REPO, p))
            for s in SPLITS for p in input_paths(s)}


def input_paths(split):
    paths = [os.path.join("benchmark", split, "records.jsonl"),
             os.path.join("benchmark", split, "verdicts.json"),
             os.path.join("analysis_out", "op_cache", f"{split}.json")]
    paths += [os.path.join("benchmark", "model_detections",
                           roster.published_filename(roster.BY_PUBLISHED[a], split))
              for a in ARMS]
    return paths


def contributor(pano):
    """Mapillary ``creator.username``; Panoramax ``geovisio:producer`` (the copyright)."""
    sm = pano.get("source_metadata") or {}
    cr = sm.get("creator")
    if isinstance(cr, dict) and cr.get("username"):
        return f"mapillary:{cr['username']}"
    if sm.get("geovisio:producer"):
        return f"panoramax:{sm['geovisio:producer']}"
    return f"{pano.get('source')}:{pano.get('copyright')}"


def load_split(split):
    gts = cc.load_gts(split)
    raw = load_floor_peaks(split)
    recs = {}
    with open(os.path.join(cc.REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pano"]["panorama_id"]] = r["pano"]
    cands = {}
    for a in ARMS:
        cm, _ = cc.load_challenger(split, a)
        cands[a] = cm
    return {"split": split, "gts": gts, "raw": raw,
            "peaks": {pid: mask_rig(raw.get(pid, [])) for pid in gts},
            "cands": cands, "recs": recs}


def split_counts(d):
    """Per-pano counts ``[m, 4]`` for N, ``[46, m, 4]`` for T and ``[144, m, 4]`` for C,
    all on the rig-masked peaks, plus the scorer (for the concentration step)."""
    pids = list(d["gts"])
    scorer = cc._Scorer(d["gts"], d["peaks"], radius_sq_for(R))

    def thr_idx(pid, t):
        return tuple(i for i, p in enumerate(d["peaks"][pid]) if p[2] >= t)

    N = cc._counts([scorer.pano(pid, thr_idx(pid, T_HI)) for pid in pids])
    T = np.stack([cc._counts([scorer.pano(pid, thr_idx(pid, t)) for pid in pids])
                  for t in T_GRID])
    filt = {(a, cm): {pid: cc.filter_cands(d["cands"][a].get(pid, []), cm) for pid in pids}
            for a in ARMS for cm in C_MIN}
    C = np.stack([cc._counts([scorer.pano(pid, cc._gate_indices(
        d["peaks"][pid], filt[(a, cm)][pid], T_HI, tl, radius_sq_for(rg))) for pid in pids])
        for a, tl, rg, cm in CANDIDATES])
    return N, T, C, scorer


def _rep(c):
    c = [int(x) for x in c]
    p = c[0] / (c[0] + c[1]) if c[0] + c[1] else 0.0
    f, r = cc._f1(*c)
    return {"tp": c[0], "fp": c[1], "fn": c[3] - c[2], "n_gt": c[3], "P": p, "R": r, "F1": f}


def _cand(i):
    a, tl, rg, cm = CANDIDATES[int(i)]
    return {"arm": a, "t_lo": tl, "r_gate": rg, "c_min": cm}


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def rig_key(data):
    """Step 4: per contributor, the share of RampNet peaks >= 0.30 the nadir mask removes."""
    by = {}
    for d in data:
        for pid in d["gts"]:
            who = contributor(d["recs"][pid])
            e = by.setdefault((d["split"], who), {"split": d["split"], "contributor": who,
                                                  "panos": 0, "peaks_ge_0.30": 0, "masked": 0,
                                                  "exploratory_peaks_ge_0.10": 0,
                                                  "exploratory_masked_ge_0.10": 0})
            e["panos"] += 1
            for p in d["raw"].get(pid, []):
                if p[2] >= T_HI:
                    e["peaks_ge_0.30"] += 1
                    e["masked"] += on_camera_rig(p[1])
                if p[2] >= 0.10:
                    e["exploratory_peaks_ge_0.10"] += 1
                    e["exploratory_masked_ge_0.10"] += on_camera_rig(p[1])
    rows = sorted(by.values(), key=lambda e: (e["split"], e["contributor"]))
    for e in rows:
        e["share"] = e["masked"] / e["peaks_ge_0.30"] if e["peaks_ge_0.30"] else None
        n10 = e["exploratory_peaks_ge_0.10"]
        e["exploratory_share_ge_0.10"] = e["exploratory_masked_ge_0.10"] / n10 if n10 else None
    splits = {}
    for s in SPLITS:
        n = sum(e["peaks_ge_0.30"] for e in rows if e["split"] == s)
        k = sum(e["masked"] for e in rows if e["split"] == s)
        splits[s] = {"peaks_ge_0.30": n, "masked": k, "share": k / n if n else None}

    def side(level, names):
        if level == "contributor":
            return [e["share"] for e in rows if e["split"] in names and e["share"] is not None]
        return [splits[s]["share"] for s in names if splits[s]["share"] is not None]

    readings = {}
    for level in ("contributor", "split"):
        a, b = side(level, (LAURENS,)), side(level, OTHER_SIDE)
        sep = cut_separates(a, b)
        readings[level] = {
            "laurens_side": a, "other_side": b, "non_lyon_cut_exists": sep,
            "ecartip": "not evaluated: Lyon is withheld from Stage 0",
            "passes": False,
            "why": ("no cut separates laurens_mapillary from morgantown and budapest_district5"
                    if not sep else
                    "a cut exists without ecartip, but the test also needs ecartip on "
                    "laurens's side, and Lyon is withheld, so it cannot pass at Stage 0")}
    return {"mask_y": NADIR_MASK_Y, "contributors": rows, "splits": splits,
            "readings": readings,
            "primary_reading": "contributor",
            "exploratory_note": ("exploratory_* columns count peaks >= 0.10 (the labeler's "
                                 "storage floor). Read after the data, used by no rule."),
            "excluded_no_peaks": [e["contributor"] for e in rows if e["share"] is None]}


def concentration(d, N, T, C, t_idx, c_idx, scorer):
    """Step 5 on ``laurens_mapillary`` with the settings its fold selected (chosen on the
    other three splits, so leaving out a Laurens sequence cannot change them)."""
    pids = list(d["gts"])
    seq = [d["recs"][pid].get("sequence_id") for pid in pids]
    t, c = T[t_idx], C[c_idx]
    net = [int((c[i, 0] - N[i, 0]) - (c[i, 1] - N[i, 1])) for i in range(len(pids))]

    def gain(mask):
        sn, st, sc = N[mask].sum(0), t[mask].sum(0), c[mask].sum(0)
        fn, ft, fc = (float(f1_of(x)) for x in (sn, st, sc))
        return fc - max(fn, ft), fc - fn

    allm = np.ones(len(pids), bool)
    g_full, gn_full = gain(allm)
    rows = []
    for s in sorted(set(seq)):
        m = np.array([q != s for q in seq])
        g, gn = gain(m)
        rows.append({"sequence": s, "panos": int((~m).sum()),
                     "net_promoted": int(sum(n for n, q in zip(net, seq) if q == s)),
                     "gain_without": g, "gain_vs_N_without": gn})
    worst = min(rows, key=lambda r: r["gain_without"])
    order = sorted(range(len(pids)), key=lambda i: (-net[i], pids[i]))
    share = top_share(net)
    flag_seq = worst["gain_without"] < ONE_INSTANCE_GAIN
    flag_top = share is not None and share > 0.5
    return {"split": LAURENS, "t_star": T_GRID[t_idx], "candidate": _cand(c_idx),
            "gain": g_full, "gain_vs_N": gn_full,
            "gain_definition": "held-out F1(C) - max(F1(N), F1(T)) on laurens_mapillary",
            "net_promoted_total": int(sum(net)),
            "net_promoted_positive_panos": int(sum(1 for n in net if n > 0)),
            "net_promoted_negative_panos": int(sum(1 for n in net if n < 0)),
            "top5": [{"pano": pids[i], "sequence": seq[i], "net": net[i]} for i in order[:5]],
            "top5_share": share, "sequences": len(rows), "by_sequence": rows,
            "worst_leave_one_out": {"sequence": worst["sequence"],
                                    "gain_without": worst["gain_without"]},
            "full_gain_already_below_0.02": g_full < ONE_INSTANCE_GAIN,
            "flag_sequence": bool(flag_seq), "flag_top5": bool(flag_top),
            "flag_one_instance": bool(flag_seq or flag_top)}


def in_sample(N, T, C):
    """Context only: per-split in-sample best threshold and best C on its own panos."""
    ft, fc = f1_of(T.sum(1)), f1_of(C.sum(1))
    ti, ci = int(np.argmax(ft)), int(np.argmax(fc))
    return {"N": _rep(N.sum(0)), "best_t": T_GRID[ti], "best_T": _rep(T[ti].sum(0)),
            "best_C": {**_cand(ci), **_rep(C[ci].sum(0))}}


def run(n_boot=BOOTSTRAP, seed=SEED):
    data = [load_split(s) for s in SPLITS]
    counts = [split_counts(d) for d in data]
    N = [c[0] for c in counts]
    T = [c[1] for c in counts]
    C = [c[2] for c in counts]

    point = loso(N, T, C)
    dp = deltas(point)
    rng = np.random.default_rng(seed)
    W = []
    for s in range(len(SPLITS)):
        m = N[s].shape[0]
        idx = rng.integers(0, m, size=(n_boot, m))
        W.append(np.stack([np.bincount(r, minlength=m) for r in idx]).astype(np.int64))
    boot = loso(N, T, C, W)
    db = deltas(boot)

    folds = []
    for h, s in enumerate(SPLITS):
        ti, ci = int(point["t_idx"][0, h]), int(point["c_idx"][0, h])
        folds.append({
            "held_out": s, "trained_on": [x for x in SPLITS if x != s],
            "t_star": T_GRID[ti], "candidate": _cand(ci),
            "N": _rep(point["n"][0, h]), "T": _rep(point["t"][0, h]),
            "C": _rep(point["c"][0, h]),
            "T_minus_N": float(dp["split_T_minus_N"][0, h]),
            "C_minus_maxNT": float(dp["split_C_minus_maxNT"][0, h]),
            "C_beats_maxNT": bool(dp["split_C_minus_maxNT"][0, h] > 0),
            "boot_T_minus_N_ci95": ci95(db["split_T_minus_N"][:, h]),
            "boot_C_minus_maxNT_ci95": ci95(db["split_C_minus_maxNT"][:, h]),
            "boot_same_t_star_frac": float((boot["t_idx"][:, h] == ti).mean()),
            "boot_same_candidate_frac": float((boot["c_idx"][:, h] == ci).mean()),
            "boot_same_arm_frac": float(np.mean(
                [CANDIDATES[int(k)][0] == CANDIDATES[ci][0] for k in boot["c_idx"][:, h]])),
        })
    pooled = {"N": _rep(point["n_pool"][0]), "T": _rep(point["t_pool"][0]),
              "C": _rep(point["c_pool"][0])}
    for k in ("T_minus_N", "C_minus_maxNT", "C_minus_N"):
        pooled[k] = float(dp[k][0])
        pooled[k + "_ci95"] = ci95(db[k])
    wins = sum(f["C_beats_maxNT"] for f in folds)

    rk = rig_key(data)
    li = SPLITS.index(LAURENS)
    conc = concentration(data[li], N[li], T[li], C[li], int(point["t_idx"][0, li]),
                         int(point["c_idx"][0, li]), counts[li][3])

    t_lb = pooled["T_minus_N_ci95"][0]
    rule1 = {"rig_key_passes": rk["readings"][rk["primary_reading"]]["passes"],
             "T_minus_N": pooled["T_minus_N"], "T_minus_N_lower": t_lb,
             "drop_T": (not rk["readings"][rk["primary_reading"]]["passes"]) or t_lb <= 0,
             "drop_T_on_interval_alone": t_lb <= 0}
    c_pt, c_lb = pooled["C_minus_maxNT"], pooled["C_minus_maxNT_ci95"][0]
    rule2 = {"C_minus_maxNT": c_pt, "C_minus_maxNT_lower": c_lb, "splits_won": wins,
             "below_0.03": c_pt < C_MIN_GAIN, "lower_le_0": c_lb <= 0,
             "fewer_than_2_wins": wins < C_MIN_WINS,
             "drop_C": c_pt < C_MIN_GAIN or c_lb <= 0 or wins < C_MIN_WINS}
    # Second reading (Amendments): if T is dropped as not implementable, compare C with N.
    cn_pt, cn_lb = pooled["C_minus_N"], pooled["C_minus_N_ci95"][0]
    wins_n = sum(f["C"]["F1"] > f["N"]["F1"] for f in folds)
    rule2_vs_N = {"C_minus_N": cn_pt, "C_minus_N_lower": cn_lb, "splits_won": wins_n,
                  "drop_C": cn_pt < C_MIN_GAIN or cn_lb <= 0 or wins_n < C_MIN_WINS}
    rule3 = {"flag_one_instance": conc["flag_one_instance"],
             "flag_sequence": conc["flag_sequence"], "flag_top5": conc["flag_top5"]}
    outcome = ("N" if rule1["drop_T"] and rule2["drop_C"] else
               "T only" if rule2["drop_C"] else "C only" if rule1["drop_T"] else "T and C")

    return {
        "plan": "open-weight-cascade-confirm-2026-11-15.md, section 3 (Stage 0)",
        "splits": list(SPLITS), "arms": list(ARMS), "t_hi": T_HI,
        "grid": {"t_lo": list(T_LO), "r_gate": list(R_GATE), "c_min": list(C_MIN),
                 "n_candidates": len(CANDIDATES), "t_star_grid": [T_GRID[0], T_GRID[-1], 0.01]},
        "rig_mask": {"deg": NADIR_MASK_DEG, "y": NADIR_MASK_Y,
                     "applied_to": "RampNet peaks of every policy (N, T, C)",
                     "peaks_removed_ge_0.10": {
                         d["split"]: sum(1 for pid in d["gts"] for p in d["raw"].get(pid, [])
                                         if p[2] >= 0.10 and on_camera_rig(p[1]))
                         for d in data}},
        "bootstrap": {"resamples": n_boot, "seed": seed, "rng": "numpy.default_rng",
                      "unit": "pano, stratified by split",
                      "selection": "t* and the C candidate re-chosen in every resample"},
        "panos": {d["split"]: len(d["gts"]) for d in data},
        "panos_missing_op_cache": {d["split"]: sorted(p for p in d["gts"] if p not in d["raw"])
                                   for d in data},
        "panos_missing_arm": {d["split"]: {a: sorted(p for p in d["gts"]
                                                     if p not in d["cands"][a]) for a in ARMS}
                              for d in data},
        "in_sample": {d["split"]: in_sample(N[i], T[i], C[i]) for i, d in enumerate(data)},
        "folds": folds, "pooled": pooled,
        "rig_key": rk, "concentration": conc,
        "rules": {"1_drop_T": rule1, "2_drop_C": rule2, "2_alt_C_vs_N": rule2_vs_N,
                  "3_one_instance": rule3, "4_outcome": outcome},
        "lyon_kaysville": "not read",
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def report(p):
    po, r = p["pooled"], p["rules"]
    print("held-out, leave-one-split-out (rig mask applied):")
    print(f"{'split':20s} {'t*':>5s} {'C candidate':34s} {'N':>7s} {'T':>7s} {'C':>7s} "
          f"{'C-max':>7s}")
    for f in p["folds"]:
        c = f["candidate"]
        cs = f"{c['arm']} {c['t_lo']:g}/{c['r_gate']:g}/{c['c_min']:g}"
        print(f"{f['held_out']:20s} {f['t_star']:5.2f} {cs:34s} {f['N']['F1']:7.4f} "
              f"{f['T']['F1']:7.4f} {f['C']['F1']:7.4f} {f['C_minus_maxNT']:+7.4f}")
    print(f"{'pooled':20s} {'':5s} {'':34s} {po['N']['F1']:7.4f} {po['T']['F1']:7.4f} "
          f"{po['C']['F1']:7.4f} {po['C_minus_maxNT']:+7.4f}")
    for k in ("T_minus_N", "C_minus_maxNT", "C_minus_N"):
        lo, hi = po[k + "_ci95"]
        print(f"  {k:14s} {po[k]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("rig key (share of peaks >= 0.30 the mask removes):")
    for e in p["rig_key"]["contributors"]:
        sh = "-" if e["share"] is None else f"{e['share']:.4f}"
        print(f"  {e['split']:20s} {e['contributor']:28s} {e['masked']:3d}/"
              f"{e['peaks_ge_0.30']:<4d} {sh}   (exploratory >= 0.10: "
              f"{e['exploratory_masked_ge_0.10']}/{e['exploratory_peaks_ge_0.10']})")
    c = p["concentration"]
    print(f"concentration ({c['split']}): gain {c['gain']:+.4f}, net promoted "
          f"{c['net_promoted_total']}, top-5 share {c['top5_share']}, worst leave-one-out "
          f"{c['worst_leave_one_out']['gain_without']:+.4f}")
    print(f"Rule 1 drop T: {r['1_drop_T']['drop_T']}  | Rule 2 drop C: {r['2_drop_C']['drop_C']}"
          f"  | Rule 3 one instance: {r['3_one_instance']['flag_one_instance']}"
          f"  | Rule 4 outcome: {r['4_outcome']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="Regenerate stage0.json and byte-compare; exit 1 on a difference.")
    args = ap.parse_args()
    path = os.path.join(args.out, RESULT)
    p = run()
    text = cc.dumps(p)
    if args.check:
        with open(path, encoding="utf-8", newline="") as f:
            same = f.read() == text
        print(("same  " if same else "DIFF  ") + path)
        sys.exit(0 if same else 1)
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    with open(os.path.join(args.out, INPUTS), "w", encoding="utf-8", newline="") as f:
        f.write(cc.dumps(input_hashes()))
    report(p)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
