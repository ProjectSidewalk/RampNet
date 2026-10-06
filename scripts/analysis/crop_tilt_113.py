"""How far rig tilt displaces the Stage 1 crop model's training targets, and whether the model follows (#113).

``stage_one/crop_model/ps_model/data/download_data.py`` builds the crop model's round-1 training set by
projecting Project Sidewalk's stored ``Panorama X/Y`` into a fixed perspective strip rendered from the
GSV tiles. sidewalk-panorama-tools' 2026-09-26 tilt study (its PR #158) showed that the tiles are in the
camera rig's frame while the stored ``pano_y`` is gravity-levelled, so the labelled feature sits at the
rig pixel ``pano_y - T(b) * h / 180`` (to first order), not at ``pano_y``. Every round-1 target is
therefore displaced from its object by an amount that depends on the pano's pitch and roll and on the
label's bearing ``b``. This script puts that displacement in the crop model's own units.

Subcommands::

    # analytic half: CPU, seconds, reads two committed sidewalk-panorama-tools tables
    python scripts/analysis/crop_tilt_113.py predict --pano-tools-root ../sidewalk-panorama-tools \
        --out analysis_out/crop_tilt_113
    # the stratified sample for the empirical half
    python scripts/analysis/crop_tilt_113.py sample --out analysis_out/crop_tilt_113/sample.csv
    # empirical half: makelab2 (pano store + GPU); writes response.csv / response.json
    python scripts/analysis/crop_tilt_113.py respond --sample analysis_out/crop_tilt_113/sample.csv \
        --store /m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas --out analysis_out/crop_tilt_113
    # re-derive summary.json, sample.csv and response.json from the committed CSVs, byte for byte
    python scripts/analysis/crop_tilt_113.py --check

Units (the chain ``docs/crop_tilt_113.md`` section 3 writes out): a displacement is measured in
**render px** of the 2048 x 2048 strip render; ``train.py`` halves the render coordinates into the
1024 x 352 input (x0.5 on both axes, as its dataset class does), the head is 4x down (x0.25), and the
target Gaussian has sigma = 12 heatmap px. So 1 sigma = 12 / 0.25 / 0.5 = 96 render px.

Sign: ``d = corrected - stored`` (render px), where ``corrected`` is the projected rig pixel scaled by
beta (beta = 1 is the full first-order leak; beta = 0.90 the RampNet-detection measurement in
sidewalk-auto-labeler #113). In ``respond``, ``peak - stored`` regressed on ``d`` has slope +1 if the
model's peak sits on the object and 0 if it reproduces the displaced target.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rampnet import stage1_geometry as sg  # noqa: E402

# --- pinned inputs ----------------------------------------------------------------------------------

PANO_TOOLS_COMMIT = "21d10aa3767167e67a098557f58f327def2396a5"
POOL_REL = "reports/data/2026-09-29-tilt-jm-pool.csv.gz"
POSE_REL = "reports/data/2026-09-29-tilt-pose-jm.csv.gz"
INPUT_SHA256 = {
    POOL_REL: "c048471769b3835b422b5df5c130561503fe0a6969fa3ff469bf402c8b774dcd",
    POSE_REL: "2c4c2ab0c75ac7b05ea2d960223eafb282fbcecbcf08bf155499921ab5f38362",
}
CROP_MODEL_REPO = "projectsidewalk/rampnet-crop-model"
CROP_MODEL_REVISION = "7aa79b8edb10b384ed69c2e99f74945e9c527fd3"
CROP_CHECKPOINTS = {
    "round1": ("round1_ps_best_model.safetensors",
               "23e40b5926bf377d1b1065fe4adaca08e23284926fd8d751c685b64ac92509b7"),
    "round2": ("round2_ps_and_manual_best_model.safetensors",
               "d129c0c6beffbb565633042f41598e44874fd27012f1c98ea5eb326851a75239"),
}

#: The 12 deployments ``download_data.py`` reads, by their pool city names.
CITIES = {
    "blackhawk-hills-il": "blackhawk-hills", "chicago-il": "chicago",
    "cliffside-park-nj": "cliffside-park", "columbus-oh": "columbus", "knox-oh": "knox",
    "mendota-il": "mendota", "newberg-or": "newberg", "oradell-nj": "oradell",
    "pittsburgh-pa": "pittsburgh", "seattle-wa": "sea", "st-louis-mo": "st-louis",
    "teaneck-nj": "teaneck",
}

# --- the crop model's unit chain (train.py / evaluate.py constants) ---------------------------------

RENDER_TO_INPUT = 0.5        # train.py: adj = orig * 0.5, both axes
INPUT_TO_HEATMAP = 0.25      # (1024, 352) input -> (256, 88) heatmap
SIGMA_HM = 12.0              # train.py generate_heatmap sigma, heatmap px
SIGMA_RENDER = SIGMA_HM / INPUT_TO_HEATMAP / RENDER_TO_INPUT   # 96 render px
#: evaluate.py: RADIUS_THRESHOLD_NORMALIZED 0.132 x (341 / 4) heatmap px.
EVAL_RADIUS_HM = 0.132 * 341 / 4
#: The paper's working panorama: download_data.py resizes every pano to 8192 x 4096.
PAPER_PANO_H = 4096
#: pano-tools S3 (reports/data/2026-09-26-tilt-error-study.json, s3_miscentering) took RampNet stage
#: one's sigma to be 12 px on a 4096-high pano (24 px at 8192 high).
S3_ASSUMED_SIGMA_PX_4096 = 12.0

BETAS = {"b100": 1.0, "b090": 0.90}
ABS_T_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
SAMPLE_STRATA = [("lt1", 0.0, 1.0), ("1to3", 1.0, 3.0), ("ge3", 3.0, 1e9)]

LABEL_FIELDS = [
    "label_uid", "city", "label_id", "pano_id", "era", "pose_source", "agree_count", "disagree_count",
    "crowd_ok", "pano_x", "pano_y", "pano_width", "pano_height", "pitch_deg", "roll_deg",
    "jpg_present", "jpg_width", "jpg_height", "b_deg", "el_deg", "nearest_theta", "T_deg",
    "strip_x", "strip_y", "in_strip", "px_per_deg_y", "d_x_b100", "d_y_b100", "d_x_b090", "d_y_b090",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rnd(v, nd=4):
    """Round for committed text; None/NaN stay None so the JSON never holds NaN."""
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    out = round(v, nd)
    return 0.0 if out == 0 else out


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def json_bytes(obj):
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def csv_bytes(fields, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- geometry per label -----------------------------------------------------------------------------

def strip_point(x_px, y_px, w, h, theta):
    """Stored-pano pixel -> float strip coords (render px) in the strip at ``theta``."""
    return sg.equirect_point_to_strip(x_px / w, y_px / h, theta)


def label_geometry(pano_x, pano_y, w, h, pitch, roll):
    """Everything ``predict`` records for one label; see LABEL_FIELDS."""
    x_norm = pano_x / w
    b = x_norm * 360.0 - 180.0                       # download_data.py's theta
    theta = sg.nearest_strip_theta(x_norm)
    el = 90.0 - pano_y / h * 180.0
    T = float(sg.tilt_term_deg(b, pitch, roll))
    sx, sy = strip_point(pano_x, pano_y, w, h, theta)
    # local vertical scale of the strip at this label: render px per degree of elevation
    dy_px = h / 180.0 * 0.05
    _, sy2 = strip_point(pano_x, pano_y - dy_px, w, h, theta)
    px_per_deg = (sy - sy2) / 0.05
    rx, ry = sg.rig_pixel_from_gravity_pixel(pano_x, pano_y, w, h, pitch, roll)
    rx, ry = float(rx), float(ry)
    dxp = (rx - pano_x + w / 2) % w - w / 2          # wrap the x move into (-w/2, w/2]
    dyp = ry - pano_y
    out = {"b_deg": b, "el_deg": el, "nearest_theta": theta, "T_deg": T, "strip_x": sx,
           "strip_y": sy, "px_per_deg_y": px_per_deg,
           "in_strip": int(0 <= sx < sg.STRIP_SLICE[1] - sg.STRIP_SLICE[0] and 0 <= sy < 2048)}
    for key, beta in BETAS.items():
        cx, cy = strip_point(pano_x + beta * dxp, pano_y + beta * dyp, w, h, theta)
        out["d_x_" + key] = cx - sx
        out["d_y_" + key] = cy - sy
    return out


# --- predict ----------------------------------------------------------------------------------------

def load_inputs(root):
    import pandas as pd
    root = Path(root)
    for rel, want in INPUT_SHA256.items():
        got = sha256_file(root / rel)
        if got != want:
            sys.exit("error: {} sha256 {} != pinned {}".format(rel, got, want))
    pool = pd.read_csv(root / POOL_REL, low_memory=False)
    pose = pd.read_csv(root / POSE_REL, low_memory=False)
    return pool, pose


def build_labels(pool, pose):
    """Filter, join poses and compute geometry. Returns (rows, funnel)."""
    import pandas as pd
    funnel = {"pool_rows": int(len(pool))}
    df = pool[pool["label_type"] == "CurbRamp"]
    funnel["curbramp"] = int(len(df))
    df = df[df["pano_source"] == "gsv"]
    funnel["curbramp_gsv"] = int(len(df))
    df = df[df["city"].isin(CITIES)]
    funnel["curbramp_gsv_12_cities"] = int(len(df))
    assert df["label_uid"].is_unique, "label_uid not unique in the pool"
    pose = pose[pose["city"].isin(CITIES)]
    assert not pose.duplicated(["city", "pano_id"]).any(), "pose table has duplicate (city, pano_id)"
    cols = ["city", "pano_id", "pitch_deg", "roll_deg", "xml_pano_yaw_deg", "xml_tilt_yaw_deg",
            "xml_tilt_pitch_deg", "jpg_present", "jpg_width", "jpg_height"]
    df = df.merge(pose[cols], on=["city", "pano_id"], how="left", validate="many_to_one")
    npz = df["pitch_deg"].notna() & df["roll_deg"].notna()
    xml = ~npz & df["xml_tilt_pitch_deg"].notna() & df["xml_pano_yaw_deg"].notna() \
        & df["xml_tilt_yaw_deg"].notna()
    funnel["posed_npz"] = int(npz.sum())
    funnel["posed_xml"] = int(xml.sum())
    funnel["unposed"] = int((~npz & ~xml).sum())
    xp, xr = sg.xml_tilt_to_pitch_roll(df["xml_pano_yaw_deg"].to_numpy(float),
                                       df["xml_tilt_yaw_deg"].to_numpy(float),
                                       df["xml_tilt_pitch_deg"].to_numpy(float))
    pitch = np.where(npz, sg.wrap_deg(df["pitch_deg"].to_numpy(float)), sg.wrap_deg(np.nan_to_num(xp)))
    roll = np.where(npz, sg.wrap_deg(df["roll_deg"].to_numpy(float)), sg.wrap_deg(np.nan_to_num(xr)))
    df = df.assign(pitch_use=pitch, roll_use=roll,
                   pose_source=np.where(npz, "npz", np.where(xml, "xml", "")))
    df = df[npz | xml].sort_values(["city", "label_id"]).reset_index(drop=True)
    funnel["posed"] = int(len(df))
    rows = []
    for r in df.itertuples(index=False):
        g = label_geometry(float(r.pano_x), float(r.pano_y), float(r.pano_width), float(r.pano_height),
                           float(r.pitch_use), float(r.roll_use))
        row = {
            "label_uid": r.label_uid, "city": r.city, "label_id": int(r.label_id),
            "pano_id": r.pano_id, "era": r.era, "pose_source": r.pose_source,
            "agree_count": int(r.agree_count), "disagree_count": int(r.disagree_count),
            "crowd_ok": int(r.agree_count - r.disagree_count >= 2),
            "pano_x": rnd(r.pano_x, 2), "pano_y": rnd(r.pano_y, 2),
            "pano_width": int(r.pano_width), "pano_height": int(r.pano_height),
            "pitch_deg": rnd(r.pitch_use), "roll_deg": rnd(r.roll_use),
            "jpg_present": int(r.jpg_present) if pd.notna(r.jpg_present) else 0,
            "jpg_width": int(r.jpg_width) if pd.notna(r.jpg_width) else "",
            "jpg_height": int(r.jpg_height) if pd.notna(r.jpg_height) else "",
        }
        for k, v in g.items():
            row[k] = v if k in ("nearest_theta", "in_strip") else rnd(v)
        rows.append(row)
    funnel["posed_crowd_ok"] = sum(r["crowd_ok"] for r in rows)
    return rows, funnel


# --- summary (from labels.csv alone, so --check needs no sibling checkout) --------------------------

def _dist(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": rnd(v.mean()), "sd": rnd(v.std(ddof=1)) if len(v) > 1 else None,
            "p50": rnd(np.percentile(v, 50)), "p90": rnd(np.percentile(v, 90)),
            "p99": rnd(np.percentile(v, 99))}


def _abs_block(dy_sigma, dx_sigma):
    ay, ax = np.abs(dy_sigma), np.abs(dx_sigma)
    return {
        "signed_d_y_sigma": _dist(dy_sigma), "abs_d_y_sigma": _dist(ay),
        "signed_d_x_sigma": _dist(dx_sigma), "abs_d_x_sigma": _dist(ax),
        "abs_d_sigma": _dist(np.hypot(ay, ax)),
        "share_abs_d_y_gt": {t: rnd((ay > float(t)).mean()) for t in ("0.25", "0.5", "1.0")},
    }


def _group_table(rows, key, dy_sigma, dx_sigma, T):
    out = {}
    keys = sorted({r[key] for r in rows})
    for k in keys:
        m = np.array([r[key] == k for r in rows])
        ay = np.abs(dy_sigma[m])
        out[str(k)] = {"n": int(m.sum()), "mean_abs_T_deg": rnd(np.abs(T[m]).mean()),
                       "mean_abs_d_y_sigma": rnd(ay.mean()), "p90_abs_d_y_sigma": rnd(np.percentile(ay, 90)),
                       "mean_abs_d_x_sigma": rnd(np.abs(dx_sigma[m]).mean()),
                       "share_abs_d_y_gt_0.5": rnd((ay > 0.5).mean())}
    return out


def _ols(X, y, clusters=None):
    """OLS with CR1 cluster-robust SE (clusters=None -> each row its own cluster, i.e. HC1)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    if clusters is None:
        clusters = np.arange(n)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    meat = np.zeros((k, k))
    for c in uniq:
        s = X[clusters == c].T @ resid[clusters == c]
        meat += np.outer(s, s)
    G = len(uniq)
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V)), resid


def heading_table(rows, dy, T, pitch, roll):
    """30-degree bins of the strip heading (nearest_theta; -180 folded into 180)."""
    theta = np.array([int(r["nearest_theta"]) for r in rows])
    theta = np.where(theta == -180, 180, theta)
    clusters = np.array([r["pano_id"] for r in rows])
    out = []
    for t in sorted(set(theta.tolist())):
        m = theta == t
        y = dy[m]
        # SE of the bin mean, clustered by pano
        _, se, _ = _ols(np.ones((m.sum(), 1)), y, clusters[m]) if m.sum() > 1 else (None, [None], None)
        out.append({"theta_deg": int(t), "n": int(m.sum()), "mean_d_y_render_px": rnd(y.mean()),
                    "se_d_y_render_px": rnd(se[0]), "mean_d_y_sigma": rnd(y.mean() / SIGMA_RENDER),
                    "rms_d_y_sigma": rnd(np.sqrt(np.mean((y / SIGMA_RENDER) ** 2))),
                    "mean_T_deg": rnd(T[m].mean()), "mean_pitch_deg": rnd(pitch[m].mean()),
                    "mean_roll_deg": rnd(roll[m].mean())})
    return out


def sinusoid_fit(rows, dy, T, pitch, roll):
    """d_y = c + a cos b + s sin b over labels (render px, clustered by pano), against the amplitude the
    fleet-mean pose predicts, and d_y regressed on T (render px per degree of T)."""
    b = np.radians(np.array([float(r["b_deg"]) for r in rows]))
    ppd = np.array([float(r["px_per_deg_y"]) for r in rows])
    clusters = np.array([r["pano_id"] for r in rows])
    X = np.column_stack([np.ones_like(b), np.cos(b), np.sin(b)])
    coef, se, _ = _ols(X, dy, clusters)
    k = float(np.median(ppd))
    slope, sse, _ = _ols(np.column_stack([np.ones_like(T), T]), dy, clusters)
    return {
        "model": "d_y_render_px = c + a cos(b) + s sin(b), SE clustered by pano",
        "c": rnd(coef[0]), "c_se": rnd(se[0]), "a": rnd(coef[1]), "a_se": rnd(se[1]),
        "s": rnd(coef[2]), "s_se": rnd(se[2]),
        "amplitude_render_px": rnd(math.hypot(coef[1], coef[2])),
        "predicted_from_mean_pose": {
            "note": "a = -k * mean(pitch), s = -k * mean(roll), k = median render px per degree",
            "k_render_px_per_deg": rnd(k), "a": rnd(-k * pitch.mean()), "s": rnd(-k * roll.mean())},
        "d_y_on_T": {"slope_render_px_per_deg": rnd(slope[1]), "slope_se": rnd(sse[1]),
                     "intercept": rnd(slope[0]), "intercept_se": rnd(sse[0]),
                     "expected_slope": rnd(-k)},
    }


def unit_chain(rows):
    ppd = np.array([float(r["px_per_deg_y"]) for r in rows])
    ppd_centre = sg.STRIP_RENDER_SIZE / 2 / math.tan(math.radians(sg.STRIP_FOV_DEG / 2)) * math.pi / 180
    sigma_deg_centre = SIGMA_RENDER / ppd_centre
    pano_px_per_deg_4096 = PAPER_PANO_H / 180.0
    sigma_pano_px_4096_centre = sigma_deg_centre * pano_px_per_deg_4096
    sigma_deg_labels = SIGMA_RENDER / ppd
    return {
        "render_to_input": RENDER_TO_INPUT, "input_to_heatmap": INPUT_TO_HEATMAP,
        "sigma_heatmap_px": SIGMA_HM, "sigma_input_px": SIGMA_HM / INPUT_TO_HEATMAP,
        "sigma_render_px": SIGMA_RENDER,
        "eval_radius_heatmap_px": rnd(EVAL_RADIUS_HM), "eval_radius_render_px": rnd(EVAL_RADIUS_HM * 8),
        "render_px_per_deg_at_strip_centre": rnd(ppd_centre),
        "render_px_per_deg_labels": _dist(ppd),
        "sigma_deg_at_strip_centre": rnd(sigma_deg_centre),
        "sigma_deg_labels": _dist(sigma_deg_labels),
        "paper_pano_px_per_deg_4096_high": rnd(pano_px_per_deg_4096),
        "sigma_paper_pano_px_4096_high_at_centre": rnd(sigma_pano_px_4096_centre),
        "s3_assumed_sigma_px_4096_high": S3_ASSUMED_SIGMA_PX_4096,
        "s3_sigma_understated_by": rnd(sigma_pano_px_4096_centre / S3_ASSUMED_SIGMA_PX_4096),
        "issue_claim_1_to_3_deg": {
            "note": "T of 1 and 3 degrees in sigma, at the strip centre and at the label median px/deg",
            "centre": [rnd(1 / sigma_deg_centre), rnd(3 / sigma_deg_centre)],
            "label_median": [rnd(1 * np.median(ppd) / SIGMA_RENDER), rnd(3 * np.median(ppd) / SIGMA_RENDER)],
        },
    }


def summarize(rows):
    """summary.json from labels.csv rows (strings, as read back)."""
    def col(name, rr):
        return np.array([float(r[name]) for r in rr])

    out = {"pano_tools_commit": PANO_TOOLS_COMMIT, "inputs_sha256": INPUT_SHA256,
           "sigma_render_px": SIGMA_RENDER, "unit_chain": unit_chain(rows)}
    pops = {"crowd_ok": [r for r in rows if r["crowd_ok"] == "1"], "all_posed": rows}
    out["counts"] = {k: len(v) for k, v in pops.items()}
    out["counts"]["crowd_ok_in_strip"] = sum(r["in_strip"] == "1" for r in pops["crowd_ok"])
    for pname, rr in pops.items():
        T = col("T_deg", rr)
        pitch, roll = col("pitch_deg", rr), col("roll_deg", rr)
        block = {"T_deg": _dist(T), "abs_T_deg": _dist(np.abs(T)),
                 "abs_pitch_deg": _dist(np.abs(pitch)), "abs_roll_deg": _dist(np.abs(roll))}
        for key in BETAS:
            dy, dx = col("d_y_" + key, rr) / SIGMA_RENDER, col("d_x_" + key, rr) / SIGMA_RENDER
            block["beta_" + key[1:]] = _abs_block(dy, dx)
        dy, dx = col("d_y_b100", rr) / SIGMA_RENDER, col("d_x_b100", rr) / SIGMA_RENDER
        block["by_city_beta_100"] = _group_table(rr, "city", dy, dx, T)
        block["by_era_beta_100"] = _group_table(rr, "era", dy, dx, T)
        block["by_pose_source_beta_100"] = _group_table(rr, "pose_source", dy, dx, T)
        tb = []
        for lo, hi in ABS_T_BINS:
            m = (np.abs(T) >= lo) & (np.abs(T) < hi)
            ay = np.abs(dy[m])
            tb.append({"abs_T_lo": lo, "abs_T_hi": hi if hi < 1e8 else None, "n": int(m.sum()),
                       "mean_abs_d_y_sigma": rnd(ay.mean()) if m.any() else None,
                       "p90_abs_d_y_sigma": rnd(np.percentile(ay, 90)) if m.any() else None,
                       "mean_abs_d_x_sigma": rnd(np.abs(dx[m]).mean()) if m.any() else None})
        block["by_abs_T_beta_100"] = tb
        dyr = col("d_y_b100", rr)
        block["heading_table_beta_100"] = heading_table(rr, dyr, T, pitch, roll)
        block["sinusoid_beta_100"] = sinusoid_fit(rr, dyr, T, pitch, roll)
        out[pname] = block
    return out


def cmd_predict(args):
    pool, pose = load_inputs(args.pano_tools_root)
    rows, funnel = build_labels(pool, pose)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "labels.csv", LABEL_FIELDS, rows)
    summary = summarize(read_csv(out / "labels.csv"))
    summary["funnel"] = funnel
    write_json(out / "summary.json", summary)
    print(json.dumps(funnel, indent=1))
    print("wrote {} labels".format(len(rows)))


# --- sample -----------------------------------------------------------------------------------------

SAMPLE_FIELDS = ["stratum", "label_uid", "city", "pano_id", "pano_x", "pano_y", "pano_width",
                 "pano_height", "jpg_width", "jpg_height", "nearest_theta", "T_deg", "strip_x",
                 "strip_y", "d_x_b100", "d_y_b100"]


def draw_sample(rows, per_stratum, seed):
    """One label per pano, crowd_ok and in the strip with a store JPEG; ``per_stratum`` per |T| bin."""
    rng = np.random.default_rng(seed)
    elig = [r for r in rows if r["crowd_ok"] == "1" and r["in_strip"] == "1" and r["jpg_present"] == "1"]
    elig.sort(key=lambda r: r["label_uid"])
    # one label per pano: pick one at random per pano first, deterministically
    by_pano = {}
    for r in elig:
        by_pano.setdefault((r["city"], r["pano_id"]), []).append(r)
    one = []
    for key in sorted(by_pano):
        grp = by_pano[key]
        one.append(grp[int(rng.integers(len(grp)))])
    out = []
    for name, lo, hi in SAMPLE_STRATA:
        cand = [r for r in one if lo <= abs(float(r["T_deg"])) < hi]
        idx = rng.permutation(len(cand))[:per_stratum]
        for i in sorted(idx.tolist()):
            r = cand[i]
            out.append(dict({k: r[k] for k in SAMPLE_FIELDS if k != "stratum"}, stratum=name))
    return out


def cmd_sample(args):
    rows = read_csv(Path(args.labels))
    out = draw_sample(rows, args.per_stratum, args.seed)
    write_csv(args.out, SAMPLE_FIELDS, out)
    print("wrote {} sample rows".format(len(out)))


# --- respond (makelab2) -----------------------------------------------------------------------------

RESPONSE_FIELDS = ["stratum", "label_uid", "city", "pano_id", "checkpoint", "T_deg", "d_x_b100",
                   "d_y_b100", "stored_hm_x", "stored_hm_y", "peak_hm_x", "peak_hm_y", "peak_value",
                   "peak2_hm_x", "peak2_hm_y", "peak2_value", "peak_minus_stored_x", "peak_minus_stored_y",
                   "peak2_minus_stored_x", "peak2_minus_stored_y"]


def _subpixel(hm, iy, ix):
    """Parabolic refinement of an integer argmax, per axis (clamped to +/- 0.5)."""
    H, W = hm.shape

    def off(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    dx = off(hm[iy, ix - 1], hm[iy, ix], hm[iy, ix + 1]) if 0 < ix < W - 1 else 0.0
    dy = off(hm[iy - 1, ix], hm[iy, ix], hm[iy + 1, ix]) if 0 < iy < H - 1 else 0.0
    return ix + dx, iy + dy


def peak_near(hm, cx, cy, radius):
    """Sub-pixel argmax of ``hm`` within ``radius`` (heatmap px) of (cx, cy). -> (x, y, value)."""
    H, W = hm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    if not mask.any():
        return None
    masked = np.where(mask, hm, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), hm.shape)
    x, y = _subpixel(hm, iy, ix)
    return x, y, float(hm[iy, ix])


def load_crop_models(hf_cache):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from rampnet.model import CROP_HEATMAP_SIZE, KeypointModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, shas = {}, {}
    for name, (fname, want) in CROP_CHECKPOINTS.items():
        path = hf_hub_download(CROP_MODEL_REPO, fname, revision=CROP_MODEL_REVISION, cache_dir=hf_cache)
        got = sha256_file(path)
        if got != want:
            sys.exit("error: {} sha256 {} != pinned {}".format(fname, got, want))
        m = KeypointModel(heatmap_size=CROP_HEATMAP_SIZE, pretrained_backbone=False)
        m.load_state_dict(load_file(path))
        models[name] = m.to(dev).eval()
        shas[name] = got
    return models, shas, dev


def cmd_respond(args):
    import torch
    from PIL import Image
    from torchvision import transforms
    from concurrent.futures import ThreadPoolExecutor
    from rampnet.model import CROP_INPUT_SIZE

    sample = read_csv(args.sample)
    if args.limit:
        sample = sample[:args.limit]
    models, shas, dev = load_crop_models(args.hf_cache)
    pre = transforms.Compose([
        transforms.Resize(CROP_INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    hm_scale = RENDER_TO_INPUT * INPUT_TO_HEATMAP
    t0 = time.time()
    gpu_s = 0.0
    rows, missing, shapes = [], [], {}

    # rendering uses the GPU too; keep decode in threads (I/O bound) and do one render at a time
    def decode(r):
        import cv2
        p = Path(args.store) / r["city"] / r["pano_id"][:2] / (r["pano_id"] + ".jpg")
        if not p.exists():
            return r, None
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return r, bgr

    from rampnet.gsv import equirectangular_to_perspective
    import cv2
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, bgr) in enumerate(ex.map(decode, sample)):
            if bgr is None:
                missing.append(r["label_uid"])
                continue
            shapes["{}x{}".format(bgr.shape[1], bgr.shape[0])] = shapes.get(
                "{}x{}".format(bgr.shape[1], bgr.shape[0]), 0) + 1
            rgb = cv2.cvtColor(cv2.resize(bgr, (8192, 4096), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            del bgr
            g0 = time.time()
            persp = equirectangular_to_perspective(rgb, 90, int(r["nearest_theta"]), -30, 2048, 2048)
            strip = persp[0:2048, sg.STRIP_SLICE[0]:sg.STRIP_SLICE[1]]
            x = pre(Image.fromarray(strip)).unsqueeze(0).to(dev)
            sx, sy = float(r["strip_x"]) * hm_scale, float(r["strip_y"]) * hm_scale
            for name, m in models.items():
                with torch.no_grad():
                    hm = np.clip(m(x).squeeze().float().cpu().numpy(), 0, 1)
                p1 = peak_near(hm, sx, sy, EVAL_RADIUS_HM)
                p2 = peak_near(hm, sx, sy, 2 * EVAL_RADIUS_HM)
                rows.append({
                    "stratum": r["stratum"], "label_uid": r["label_uid"], "city": r["city"],
                    "pano_id": r["pano_id"], "checkpoint": name, "T_deg": r["T_deg"],
                    "d_x_b100": r["d_x_b100"], "d_y_b100": r["d_y_b100"],
                    "stored_hm_x": rnd(sx), "stored_hm_y": rnd(sy),
                    "peak_hm_x": rnd(p1[0]), "peak_hm_y": rnd(p1[1]), "peak_value": rnd(p1[2]),
                    "peak2_hm_x": rnd(p2[0]), "peak2_hm_y": rnd(p2[1]), "peak2_value": rnd(p2[2]),
                    "peak_minus_stored_x": rnd((p1[0] - sx) / hm_scale),
                    "peak_minus_stored_y": rnd((p1[1] - sy) / hm_scale),
                    "peak2_minus_stored_x": rnd((p2[0] - sx) / hm_scale),
                    "peak2_minus_stored_y": rnd((p2[1] - sy) / hm_scale),
                })
            gpu_s += time.time() - g0
            if (i + 1) % 25 == 0:
                print("{}/{} panos, {:.0f} s".format(i + 1, len(sample), time.time() - t0), flush=True)
    elapsed = time.time() - t0
    out = Path(args.out)
    write_csv(out / "response.csv", RESPONSE_FIELDS, rows)
    resp = fit_response(read_csv(out / "response.csv"))
    write_json(out / "response.json", resp)
    usage = {"elapsed_s": round(elapsed, 1), "render_and_forward_s": round(gpu_s, 1),
             "panos": len(sample) - len(missing), "missing": missing, "jpeg_shapes": shapes,
             "checkpoint_sha256": shas, "device": str(dev),
             "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             "ts_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(args.usage_out), usage)
    print(json.dumps(usage, indent=1))


def fit_response(rows):
    """Regress peak - stored on d (beta = 1), per checkpoint and axis, SE clustered by pano."""
    out = {"model": "peak_minus_stored = c + slope * d_b100 (render px); slope +1 = peak on the object, "
                    "0 = peak on the displaced target; SE clustered by pano; 95% CI = slope +/- 1.96 SE",
           "eval_radius_render_px": rnd(EVAL_RADIUS_HM / (RENDER_TO_INPUT * INPUT_TO_HEATMAP)),
           "sigma_render_px": SIGMA_RENDER}
    for ck in sorted({r["checkpoint"] for r in rows}):
        rr = [r for r in rows if r["checkpoint"] == ck]
        block = {"n": len(rr), "strata": {s: sum(r["stratum"] == s for r in rr) for s, _, _ in SAMPLE_STRATA}}
        for window, pre in (("radius", "peak"), ("radius_x2", "peak2")):
            for subset, keep in (("all", lambda r: True),
                                 ("peak_ge_0.3", lambda r, p=pre: float(r[p + "_value"]) >= 0.3)):
                sub = [r for r in rr if keep(r)]
                res = {"n": len(sub)}
                if len(sub) > 10:
                    clusters = np.array([r["pano_id"] for r in sub])
                    for ax in ("x", "y"):
                        d = np.array([float(r["d_{}_b100".format(ax)]) for r in sub])
                        y = np.array([float(r["{}_minus_stored_{}".format(pre, ax)]) for r in sub])
                        coef, se, resid = _ols(np.column_stack([np.ones_like(d), d]), y, clusters)
                        res[ax] = {"slope": rnd(coef[1]), "slope_se": rnd(se[1]),
                                   "slope_ci95": [rnd(coef[1] - 1.96 * se[1]), rnd(coef[1] + 1.96 * se[1])],
                                   "intercept_render_px": rnd(coef[0]), "intercept_se": rnd(se[0]),
                                   "resid_sd_render_px": rnd(resid.std(ddof=2)),
                                   "sd_d_render_px": rnd(d.std(ddof=1))}
                    res["median_peak_value"] = rnd(np.median([float(r[pre + "_value"]) for r in sub]))
                block["{}:{}".format(window, subset)] = res
        out[ck] = block
    # paired: the same strips under both checkpoints, so regress the per-label difference on d
    by = {}
    for r in rows:
        by.setdefault(r["label_uid"], {})[r["checkpoint"]] = r
    pairs = [v for _, v in sorted(by.items()) if "round1" in v and "round2" in v]
    if len(pairs) > 10:
        diff = {"n": len(pairs), "note": "(peak_round2 - peak_round1) = c + slope * d_b100; slope is the "
                                         "round-2 minus round-1 slope, paired on the same strips"}
        clusters = np.array([v["round1"]["pano_id"] for v in pairs])
        for window, pre in (("radius", "peak"), ("radius_x2", "peak2")):
            for ax in ("x", "y"):
                d = np.array([float(v["round1"]["d_{}_b100".format(ax)]) for v in pairs])
                y = np.array([float(v["round2"]["{}_minus_stored_{}".format(pre, ax)])
                              - float(v["round1"]["{}_minus_stored_{}".format(pre, ax)]) for v in pairs])
                coef, se, _ = _ols(np.column_stack([np.ones_like(d), d]), y, clusters)
                diff["{}:{}".format(window, ax)] = {
                    "slope": rnd(coef[1]), "slope_se": rnd(se[1]),
                    "slope_ci95": [rnd(coef[1] - 1.96 * se[1]), rnd(coef[1] + 1.96 * se[1])],
                    "intercept_render_px": rnd(coef[0]), "intercept_se": rnd(se[0])}
        out["round2_minus_round1"] = diff
    return out


# --- check ------------------------------------------------------------------------------------------

def cmd_check(args):
    out = Path(args.out)
    ok = True

    def cmp(name, want_bytes):
        nonlocal ok
        have = (out / name).read_bytes()
        same = have == want_bytes
        ok &= same
        print("{:<16} {}".format(name, "ok" if same else "DIFFERS"))

    rows = read_csv(out / "labels.csv")
    if args.pano_tools_root:
        pool, pose = load_inputs(args.pano_tools_root)
        fresh, funnel = build_labels(pool, pose)
        cmp("labels.csv", csv_bytes(LABEL_FIELDS, fresh))
    else:
        funnel = json.loads((out / "summary.json").read_text(encoding="utf-8"))["funnel"]
    summary = summarize(rows)
    summary["funnel"] = funnel
    cmp("summary.json", json_bytes(summary))
    if (out / "sample.csv").exists():
        cmp("sample.csv", csv_bytes(SAMPLE_FIELDS, draw_sample(rows, args.per_stratum, args.seed)))
    if (out / "response.csv").exists():
        cmp("response.json", json_bytes(fit_response(read_csv(out / "response.csv"))))
    if not ok:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="re-derive summary.json, sample.csv and response.json from the committed CSVs")
    ap.add_argument("--out", default=str(REPO / "analysis_out" / "crop_tilt_113"))
    ap.add_argument("--pano-tools-root", default=None,
                    help="with --check: also re-derive labels.csv from the pinned pano-tools inputs")
    ap.add_argument("--per-stratum", type=int, default=200)
    ap.add_argument("--seed", type=int, default=113)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("predict")
    p.add_argument("--pano-tools-root", required=True)
    p.add_argument("--out", default=str(REPO / "analysis_out" / "crop_tilt_113"))
    s = sub.add_parser("sample")
    s.add_argument("--labels", default=str(REPO / "analysis_out" / "crop_tilt_113" / "labels.csv"))
    s.add_argument("--out", default=str(REPO / "analysis_out" / "crop_tilt_113" / "sample.csv"))
    s.add_argument("--per-stratum", type=int, default=200)
    s.add_argument("--seed", type=int, default=113)
    r = sub.add_parser("respond")
    r.add_argument("--sample", required=True)
    r.add_argument("--store", required=True)
    r.add_argument("--hf-cache", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--usage-out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0, help="first N sample rows only (smoke test)")
    args = ap.parse_args(argv)
    if args.check:
        return cmd_check(args)
    {"predict": cmd_predict, "sample": cmd_sample, "respond": cmd_respond}.get(
        args.cmd, lambda a: ap.print_help())(args)


if __name__ == "__main__":
    main()
