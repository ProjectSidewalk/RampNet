"""How far rig tilt displaces the Stage 1 crop model's training targets, and whether the model follows (#113).

``stage_one/crop_model/ps_model/data/download_data.py`` builds the crop model's round-1 training set by
projecting Project Sidewalk's stored ``Panorama X/Y`` into a fixed perspective strip rendered from the
GSV tiles. sidewalk-panorama-tools' 2026-09-26 tilt study (its PR #158) showed that the tiles are in the
camera rig's frame while the stored ``pano_y`` is gravity-levelled, so the labelled feature sits at the
rig pixel ``pano_y - T(b) * h / 180`` (to first order), not at ``pano_y``. Every round-1 target is
therefore displaced from its object by an amount that depends on the pano's pitch and roll and on the
label's bearing ``b``. This script puts that displacement in the crop model's own units.

Subcommands::

    # analytic half: CPU, about a minute, reads two committed sidewalk-panorama-tools tables; writes
    # labels.csv / summary.json (the 12 round-1 deployments) and labels_heldout.csv /
    # summary_heldout.json (every other city in the pool)
    python scripts/analysis/crop_tilt_113.py predict --pano-tools-root ../sidewalk-panorama-tools
    # the stratified samples for the empirical half (in-population and held-out)
    python scripts/analysis/crop_tilt_113.py sample
    python scripts/analysis/crop_tilt_113.py sample --labels analysis_out/crop_tilt_113/labels_heldout.csv \
        --out analysis_out/crop_tilt_113/sample_heldout.csv
    # the round-1 training keypoints, read from the Hub by column (no images), and the overlap test
    python scripts/analysis/crop_tilt_113.py fetch-keypoints
    python scripts/analysis/crop_tilt_113.py overlap
    # empirical half: makelab2 (pano store + GPU); respond writes response<suffix>.csv and the usage
    # record, and `fit --name` writes the matching .json (CPU)
    python scripts/analysis/crop_tilt_113.py respond --sample analysis_out/crop_tilt_113/sample.csv \
        --store /m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas --out analysis_out/crop_tilt_113 \
        --usage-out analysis_out/crop_tilt_113/usage_respond.json
    python scripts/analysis/crop_tilt_113.py fit --name response      # or response_heldout, response_v1
    # re-derive every summary from the committed CSVs, byte for byte
    python scripts/analysis/crop_tilt_113.py --check

Units (the chain ``docs/crop_tilt_113.md`` section 3 writes out): a displacement is measured in
**render px** of the 2048 x 2048 strip render; ``train.py`` halves the render coordinates into the
1024 x 352 input (x0.5 on both axes, as its dataset class does), the head is 4x down (x0.25), and the
target Gaussian has sigma = 12 heatmap px. So 1 sigma = 12 / 0.25 / 0.5 = 96 render px.

Sign: ``d = corrected - stored`` (render px), where ``corrected`` is the projected rig pixel scaled by
beta (beta = 1 is the full first-order leak; beta = 0.90 the RampNet-detection measurement in
sidewalk-auto-labeler #113). In ``respond``, ``peak - stored`` regressed on ``d`` has slope +1 if the
model's peak sits on the object (at beta = 1) and 0 if it reproduces the displaced target.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rampnet import stage1_geometry as sg  # noqa: E402

OUT_DEFAULT = REPO / "analysis_out" / "crop_tilt_113"

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
ROUND1_DATASET = "projectsidewalk/rampnet-crop-model-dataset-round1"
ROUND1_DATASET_REVISION = "521f74ff752d57824400c8f7d5ca4717efa7bf16"
ROUND1_FILES = (["data/train/train-0000{}.parquet".format(i) for i in range(7)]
                + ["data/val/val-0000{}.parquet".format(i) for i in range(2)]
                + ["data/test/test-0000{}.parquet".format(i) for i in range(2)])

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
HM_SCALE = RENDER_TO_INPUT * INPUT_TO_HEATMAP
SIGMA_HM = 12.0              # train.py generate_heatmap sigma, heatmap px
SIGMA_RENDER = SIGMA_HM / HM_SCALE   # 96 render px
#: evaluate.py: RADIUS_THRESHOLD_NORMALIZED 0.132 x (341 / 4) heatmap px.
EVAL_RADIUS_HM = 0.132 * 341 / 4
#: The paper's working panorama: download_data.py resizes every pano to 8192 x 4096.
PAPER_PANO_H = 4096
#: pano-tools S3 (reports/data/2026-09-26-tilt-error-study.json, s3_miscentering) took RampNet stage
#: one's sigma to be 12 px on a 4096-high pano (24 px at 8192 high).
S3_ASSUMED_SIGMA_PX_4096 = 12.0
#: S3's four depression bands at pano-tools 21d10aa: (band, depression_deg, ceiling_shift_deg = T p90).
S3_BANDS = [("<5", 3.746337890625, 4.8432671319998555), ("5-15", 10.83251953125, 4.375441916820636),
            ("15-30", 19.53369140625, 3.981300696555321), (">30", 33.99169921875, 3.814573777523714)]

BETAS = {"b100": 1.0, "b090": 0.90}
ABS_T_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
SAMPLE_STRATA = [("lt1", 0.0, 1.0), ("1to3", 1.0, 3.0), ("ge3", 3.0, 1e9)]
#: Peak-search windows in respond, heatmap px: (name, radius, centre). "stored" is the plan's
#: window (evaluate.py's radius around the stored target); "stored_x2" doubles it; "mid" centres
#: evaluate.py's radius half way between the stored target and the corrected point, so neither
#: hypothesis is favoured by the window edge.
WINDOWS = [("stored", EVAL_RADIUS_HM, "stored"), ("stored_x2", 2 * EVAL_RADIUS_HM, "stored"),
           ("mid", EVAL_RADIUS_HM, "mid")]
#: A peak within this many heatmap px of its window's boundary counts as an edge hit.
EDGE_TOL_HM = 1.0

LABEL_FIELDS = [
    "label_uid", "city", "label_id", "pano_id", "era", "pose_source", "agree_count", "disagree_count",
    "crowd_ok", "pano_x", "pano_y", "pano_width", "pano_height", "pitch_deg", "roll_deg",
    "jpg_present", "jpg_width", "jpg_height", "b_deg", "el_deg", "nearest_theta", "T_deg",
    "strip_x", "strip_y", "paper_x", "paper_y", "in_strip", "px_per_deg_y",
    "d_x_b100", "d_y_b100", "d_x_b090", "d_y_b090", "d_y_pitch_b100", "d_y_roll_b100",
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


def json_bytes(obj):
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(json_bytes(obj))


def csv_bytes(fields, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(csv_bytes(fields, rows))


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- geometry per label -----------------------------------------------------------------------------

def strip_point(x_px, y_px, w, h, theta):
    """Stored-pano pixel -> float strip coords (render px) in the strip at ``theta``.

    Never None as used here: a label is always within 15 degrees of its strip's heading, so it is in
    front of the camera. The assert keeps a future caller from unpacking a None silently.
    """
    res = sg.equirect_point_to_strip(x_px / w, y_px / h, theta)
    assert res is not None, "point behind the strip camera"
    return res


def paper_point(x_px, y_px, w, h, theta):
    """The integer point download_data.py writes into a round-1 filename for this label."""
    res = sg.equirect_point_to_perspective_float(x_px / w * 8192, y_px / h * 4096, 8192, 4096,
                                                 sg.STRIP_FOV_DEG, theta, sg.STRIP_PHI_DEG,
                                                 sg.STRIP_RENDER_SIZE, sg.STRIP_RENDER_SIZE)
    assert res is not None, "point behind the strip camera"
    return int(int(res[0]) - sg.STRIP_RENDER_SIZE / 3), int(res[1])


def _displacement(pano_x, pano_y, w, h, pitch, roll, beta, theta, sx, sy):
    rx, ry = sg.rig_pixel_from_gravity_pixel(pano_x, pano_y, w, h, pitch, roll)
    dxp = (float(rx) - pano_x + w / 2) % w - w / 2          # wrap the x move into (-w/2, w/2]
    dyp = float(ry) - pano_y
    cx, cy = strip_point(pano_x + beta * dxp, pano_y + beta * dyp, w, h, theta)
    return cx - sx, cy - sy


def label_geometry(pano_x, pano_y, w, h, pitch, roll):
    """Everything ``predict`` records for one label; see LABEL_FIELDS."""
    x_norm = pano_x / w
    b = x_norm * 360.0 - 180.0                       # download_data.py's theta
    theta = sg.nearest_strip_theta(x_norm)
    el = 90.0 - pano_y / h * 180.0
    T = float(sg.tilt_term_deg(b, pitch, roll))
    sx, sy = strip_point(pano_x, pano_y, w, h, theta)
    px, py = paper_point(pano_x, pano_y, w, h, theta)
    # local vertical scale of the strip at this label: render px per degree of elevation
    _, sy2 = strip_point(pano_x, pano_y - h / 180.0 * 0.05, w, h, theta)
    out = {"b_deg": b, "el_deg": el, "nearest_theta": theta, "T_deg": T, "strip_x": sx,
           "strip_y": sy, "paper_x": px, "paper_y": py, "px_per_deg_y": (sy - sy2) / 0.05,
           "in_strip": int(0 <= sx < sg.STRIP_SLICE[1] - sg.STRIP_SLICE[0] and 0 <= sy < 2048)}
    for key, beta in BETAS.items():
        out["d_x_" + key], out["d_y_" + key] = _displacement(pano_x, pano_y, w, h, pitch, roll, beta,
                                                             theta, sx, sy)
    # the pitch and roll parts of d_y separately (exact geometry with the other angle zeroed)
    out["d_y_pitch_b100"] = _displacement(pano_x, pano_y, w, h, pitch, 0.0, 1.0, theta, sx, sy)[1]
    out["d_y_roll_b100"] = _displacement(pano_x, pano_y, w, h, 0.0, roll, 1.0, theta, sx, sy)[1]
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


def pose_agreement(pose):
    """npz vs XML pose on panos that carry both (all cities): the pose noise that attenuates slopes."""
    both = pose[pose["pitch_deg"].notna() & pose["roll_deg"].notna() & pose["xml_tilt_pitch_deg"].notna()
                & pose["xml_pano_yaw_deg"].notna() & pose["xml_tilt_yaw_deg"].notna()]
    xp, xr = sg.xml_tilt_to_pitch_roll(both["xml_pano_yaw_deg"].to_numpy(float),
                                       both["xml_tilt_yaw_deg"].to_numpy(float),
                                       both["xml_tilt_pitch_deg"].to_numpy(float))
    dp = sg.wrap_deg(both["pitch_deg"].to_numpy(float) - xp)
    dr = sg.wrap_deg(both["roll_deg"].to_numpy(float) - xr)
    return {"n_panos": int(len(both)), "sd_pitch_diff_deg": rnd(np.std(dp, ddof=1)),
            "sd_roll_diff_deg": rnd(np.std(dr, ddof=1)),
            "median_abs_pitch_diff_deg": rnd(np.median(np.abs(dp))),
            "median_abs_roll_diff_deg": rnd(np.median(np.abs(dr)))}


def build_labels(pool, pose, heldout=False):
    """Filter, join poses and compute geometry. Returns (rows, funnel).

    ``heldout=False``: the 12 deployments download_data.py reads. ``heldout=True``: every other city
    in the pool, i.e. labels the round-1 crop model cannot have trained on.
    """
    import pandas as pd
    funnel = {"pool_rows": int(len(pool))}
    if not heldout:
        funnel["pose_npz_vs_xml_all_cities"] = pose_agreement(pose)
    df = pool[pool["label_type"] == "CurbRamp"]
    funnel["curbramp"] = int(len(df))
    df = df[df["pano_source"] == "gsv"]
    funnel["curbramp_gsv"] = int(len(df))
    in_deploy = df["city"].isin(CITIES)
    df = df[~in_deploy] if heldout else df[in_deploy]
    funnel["curbramp_gsv_heldout_cities" if heldout else "curbramp_gsv_12_cities"] = int(len(df))
    assert df["label_uid"].is_unique, "label_uid not unique in the pool"
    pose = pose[pose["city"].isin(set(df["city"]))]
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
            row[k] = v if k in ("nearest_theta", "in_strip", "paper_x", "paper_y") else rnd(v)
        rows.append(row)
    funnel["posed_crowd_ok"] = sum(r["crowd_ok"] for r in rows)
    return rows, funnel


# --- small stats ------------------------------------------------------------------------------------

def _dist(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": rnd(v.mean()), "sd": rnd(v.std(ddof=1)) if len(v) > 1 else None,
            "p50": rnd(np.percentile(v, 50)), "p90": rnd(np.percentile(v, 90)),
            "p99": rnd(np.percentile(v, 99))}


def _ols(X, y, clusters=None, weights=None):
    """(Weighted) least squares with CR1 cluster-robust SE.

    ``clusters=None`` makes each row its own cluster (HC1). Returns (beta, se, resid), or None when
    there are not more rows than coefficients.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    if n <= k:
        return None
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    Xw = X * w[:, None]
    bread = np.linalg.inv(X.T @ Xw)
    beta = bread @ (Xw.T @ y)
    resid = y - X @ beta
    _, cid = np.unique(np.arange(n) if clusters is None else np.asarray(clusters), return_inverse=True)
    G = int(cid.max()) + 1
    scores = np.zeros((G, k))
    np.add.at(scores, cid, Xw * resid[:, None])
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = adj * bread @ (scores.T @ scores) @ bread
    return beta, np.sqrt(np.diag(V)), resid


def _slope_block(d, y, clusters, weights=None):
    fit = _ols(np.column_stack([np.ones_like(d), d]), y, clusters, weights)
    if fit is None:
        return {"n": int(len(d))}
    coef, se, resid = fit
    return {"n": int(len(d)), "slope": rnd(coef[1]), "slope_se": rnd(se[1]),
            "slope_ci95": [rnd(coef[1] - 1.96 * se[1]), rnd(coef[1] + 1.96 * se[1])],
            "intercept_render_px": rnd(coef[0]), "intercept_se": rnd(se[0]),
            "resid_sd_render_px": rnd(resid.std(ddof=2)) if len(d) > 2 else None,
            "sd_d_render_px": rnd(np.std(d, ddof=1))}


# --- summary (from labels.csv alone, so --check needs no sibling checkout) --------------------------

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
    for k in sorted({r[key] for r in rows}):
        m = np.array([r[key] == k for r in rows])
        ay = np.abs(dy_sigma[m])
        out[str(k)] = {"n": int(m.sum()), "mean_abs_T_deg": rnd(np.abs(T[m]).mean()),
                       "mean_abs_d_y_sigma": rnd(ay.mean()), "p90_abs_d_y_sigma": rnd(np.percentile(ay, 90)),
                       "mean_abs_d_x_sigma": rnd(np.abs(dx_sigma[m]).mean()),
                       "share_abs_d_y_gt_0.5": rnd((ay > 0.5).mean())}
    return out


def heading_table(rows, dy, T, pitch, roll):
    """30-degree bins of the strip heading (nearest_theta; -180 folded into 180)."""
    theta = np.array([int(r["nearest_theta"]) for r in rows])
    theta = np.where(theta == -180, 180, theta)
    clusters = np.array([r["pano_id"] for r in rows])
    out = []
    for t in sorted(set(theta.tolist())):
        m = theta == t
        y = dy[m]
        fit = _ols(np.ones((int(m.sum()), 1)), y, clusters[m])
        out.append({"theta_deg": int(t), "n": int(m.sum()), "mean_d_y_render_px": rnd(y.mean()),
                    "se_d_y_render_px": rnd(fit[1][0]) if fit else None,
                    "mean_d_y_sigma": rnd(y.mean() / SIGMA_RENDER),
                    "rms_d_y_sigma": rnd(np.sqrt(np.mean((y / SIGMA_RENDER) ** 2))),
                    "mean_T_deg": rnd(T[m].mean()), "mean_pitch_deg": rnd(pitch[m].mean()),
                    "mean_roll_deg": rnd(roll[m].mean())})
    return out


def sinusoid_fit(rows, dy, T, pitch, roll):
    """d_y = c + a cos b + s sin b over labels (render px, clustered by pano), against the amplitude the
    fleet-mean pose predicts, and d_y regressed on T. Both are internal-consistency checks: d_y is a
    deterministic function of (pitch, roll, b), so they cannot test the sign; respond does that."""
    if len(rows) < 4:
        return {"n": len(rows)}
    b = np.radians(np.array([float(r["b_deg"]) for r in rows]))
    ppd = np.array([float(r["px_per_deg_y"]) for r in rows])
    clusters = np.array([r["pano_id"] for r in rows])
    coef, se, _ = _ols(np.column_stack([np.ones_like(b), np.cos(b), np.sin(b)]), dy, clusters)
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


def _ppd_at_depression(dep_deg):
    """Render px per degree of elevation on the strip's centre column at a given depression."""
    f = sg.STRIP_RENDER_SIZE / 2 / math.tan(math.radians(sg.STRIP_FOV_DEG / 2))
    off = math.radians(-sg.STRIP_PHI_DEG - dep_deg)           # angle from the strip's optical axis
    return f / math.cos(off) ** 2 * math.pi / 180


def unit_chain(rows):
    ppd = np.array([float(r["px_per_deg_y"]) for r in rows])
    sy = np.array([float(r["strip_y"]) for r in rows])
    ppd_centre = _ppd_at_depression(-sg.STRIP_PHI_DEG)
    sigma_deg_centre = SIGMA_RENDER / ppd_centre
    pano_px_per_deg_4096 = PAPER_PANO_H / 180.0
    sigma_pano_px_4096_centre = sigma_deg_centre * pano_px_per_deg_4096
    s3 = []
    for band, dep, shift in S3_BANDS:
        sd = SIGMA_RENDER / _ppd_at_depression(dep)
        s3.append({"band": band, "depression_deg": rnd(dep), "sigma_deg": rnd(sd),
                   "s3_sigma_understated_by": rnd(sd * pano_px_per_deg_4096 / S3_ASSUMED_SIGMA_PX_4096),
                   "p90_shift_deg": rnd(shift), "p90_shift_sigma": rnd(shift / sd)})
    return {
        "render_to_input": RENDER_TO_INPUT, "input_to_heatmap": INPUT_TO_HEATMAP,
        "sigma_heatmap_px": SIGMA_HM, "sigma_input_px": SIGMA_HM / INPUT_TO_HEATMAP,
        "sigma_render_px": SIGMA_RENDER,
        "eval_radius_heatmap_px": rnd(EVAL_RADIUS_HM), "eval_radius_render_px": rnd(EVAL_RADIUS_HM / HM_SCALE),
        "render_px_per_deg_at_strip_centre": rnd(ppd_centre),
        "render_px_per_deg_labels": _dist(ppd),
        "strip_y_labels": _dist(sy), "share_above_strip_centre": rnd((sy < 1024).mean()),
        "sigma_deg_at_strip_centre": rnd(sigma_deg_centre),
        "sigma_deg_labels": _dist(SIGMA_RENDER / ppd),
        "paper_pano_px_per_deg_4096_high": rnd(pano_px_per_deg_4096),
        "sigma_paper_pano_px_4096_high_at_centre": rnd(sigma_pano_px_4096_centre),
        "s3_assumed_sigma_px_4096_high": S3_ASSUMED_SIGMA_PX_4096,
        "s3_sigma_understated_by_at_centre": rnd(sigma_pano_px_4096_centre / S3_ASSUMED_SIGMA_PX_4096),
        "s3_by_depression_band": s3,
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
                 "abs_pitch_deg": _dist(np.abs(pitch)), "abs_roll_deg": _dist(np.abs(roll)),
                 "sample_strata_share": {s: rnd(((np.abs(T) >= lo) & (np.abs(T) < hi)).mean())
                                         for s, lo, hi in SAMPLE_STRATA}}
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
    out = Path(args.out)
    for heldout, stem in ((False, ""), (True, "_heldout")):
        rows, funnel = build_labels(pool, pose, heldout=heldout)
        write_csv(out / "labels{}.csv".format(stem), LABEL_FIELDS, rows)
        summary = summarize(read_csv(out / "labels{}.csv".format(stem)))
        summary["funnel"] = funnel
        write_json(out / "summary{}.json".format(stem), summary)
        print(stem or "deployments", json.dumps(funnel))


# --- sample -----------------------------------------------------------------------------------------

SAMPLE_FIELDS = ["stratum", "label_uid", "city", "pano_id", "pano_x", "pano_y", "pano_width",
                 "pano_height", "jpg_width", "jpg_height", "nearest_theta", "T_deg", "strip_x",
                 "strip_y", "d_x_b100", "d_y_b100", "d_y_pitch_b100", "d_y_roll_b100"]


def draw_sample(rows, per_stratum, seed):
    """One label per pano, crowd_ok and in the strip with a store JPEG; ``per_stratum`` per |T| bin."""
    rng = np.random.default_rng(seed)
    elig = [r for r in rows if r["crowd_ok"] == "1" and r["in_strip"] == "1" and r["jpg_present"] == "1"]
    elig.sort(key=lambda r: r["label_uid"])
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
    out = draw_sample(read_csv(Path(args.labels)), args.per_stratum, args.seed)
    write_csv(args.out, SAMPLE_FIELDS, out)
    print("wrote {} sample rows".format(len(out)))


# --- overlap with the round-1 training set (no GPU, no images) ---------------------------------------

KEYPOINT_FIELDS = ["split", "crop_id"]
#: Null offsets (render px) for the chance-match rate: shift each label's point and match again.
NULL_OFFSETS = [(dx, dy) for dx in (-23, -11, 11, 23) for dy in (-29, -13, 13, 29)]


def cmd_fetch_keypoints(args):
    """Read crop_id (which carries every keypoint) for all 27,704 round-1 crops by column, no images."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    rows = []
    for rel in ROUND1_FILES:
        path = "datasets/{}@{}/{}".format(ROUND1_DATASET, ROUND1_DATASET_REVISION, rel)
        with fs.open(path, "rb", block_size=1 << 20) as f:
            tab = pq.ParquetFile(f).read(columns=["crop_id"])
        split = rel.split("/")[1]
        rows += [{"split": split, "crop_id": c} for c in tab.column("crop_id").to_pylist()]
        print(rel, tab.num_rows, flush=True)
    rows.sort(key=lambda r: (r["split"], r["crop_id"]))
    write_csv(Path(args.out) / "round1_keypoints.csv", KEYPOINT_FIELDS, rows)
    print("wrote {} crops".format(len(rows)))


def _parse_crop_id(cid):
    parts = cid.split("_-_")
    return [tuple(int(v) for v in p.split("_")) for p in parts[1:]]


def overlap_report(labels, labels_heldout, sample, sample_heldout, crops):
    """Exact-match each label's download_data.py integer point against round-1 crop keypoints."""
    main = {}
    for c in crops:
        kps = _parse_crop_id(c["crop_id"])
        if kps:
            main.setdefault(kps[0], []).append(c["split"])

    def rate(rr):
        if not rr:
            return {"n": 0}
        pts = [(int(r["paper_x"]), int(r["paper_y"])) for r in rr]
        hits = [p for p in pts if p in main]
        by_split = {}
        for p in hits:
            for s in sorted(set(main[p])):
                by_split[s] = by_split.get(s, 0) + 1
        null = [np.mean([(x + dx, y + dy) in main for x, y in pts]) for dx, dy in NULL_OFFSETS]
        return {"n": len(pts), "exact_main_keypoint_match": len(hits),
                "match_rate": rnd(len(hits) / len(pts)), "matched_crop_split": by_split,
                "null_rate_mean": rnd(np.mean(null)), "null_rate_max": rnd(np.max(null)),
                "excess_over_null": rnd(len(hits) / len(pts) - np.mean(null)),
                "overlap_fraction_est": rnd((len(hits) / len(pts) - np.mean(null)) / (1 - np.mean(null))),
                "expected_chance_matches": rnd(len(pts) * np.mean(null) * (1 - (len(hits) / len(pts) - np.mean(null))
                                                                           / (1 - np.mean(null))))}

    def by_uid(rr):
        return {r["label_uid"]: r for r in rr}

    lab = by_uid(labels)
    labh = by_uid(labels_heldout)
    out = {
        "note": ("A label 'matches' when download_data.py's integer point for it equals the first keypoint "
                 "of some round-1 crop (the crop's own label). The null shifts every point by 16 offsets "
                 "of 11-29 render px and matches again: that is the chance rate of hitting some other "
                 "crop's point. If a share f of labels truly have a round-1 crop, match = f + (1 - f) * null, so "
                 "overlap_fraction_est = (match - null) / (1 - null) estimates f; expected_chance_matches is "
                 "(1 - f) * null * n, the matched labels that are matches by chance. Sensitivity (a true "
                 "round-1 label whose integer point differs, e.g. a label edited since 2025) is NOT measured, "
                 "so the unmatched set can still contain round-1 labels."),
        "round1_dataset": "{}@{}".format(ROUND1_DATASET, ROUND1_DATASET_REVISION),
        "round1_crops": len(crops), "distinct_main_keypoints": len(main),
        "labels_crowd_ok": rate([r for r in labels if r["crowd_ok"] == "1"]),
        "labels_heldout_crowd_ok": rate([r for r in labels_heldout if r["crowd_ok"] == "1"]),
        "sample": rate([lab[s["label_uid"]] for s in sample]),
        "sample_by_stratum": {name: rate([lab[s["label_uid"]] for s in sample if s["stratum"] == name])
                              for name, _, _ in SAMPLE_STRATA},
    }
    if sample_heldout:
        out["sample_heldout"] = rate([labh[s["label_uid"]] for s in sample_heldout])
    # are the unmatched sample labels hiding as a SECONDARY keypoint of some other label's crop?
    sec = set()
    for c in crops:
        sec.update(_parse_crop_id(c["crop_id"])[1:])
    pts = [(int(lab[s["label_uid"]]["paper_x"]), int(lab[s["label_uid"]]["paper_y"])) for s in sample]
    um = [p for p in pts if p not in main]
    out["sample_unmatched_as_secondary_keypoint"] = {
        "n_unmatched": len(um), "hits": sum(p in sec for p in um),
        "null_hits_mean": rnd(np.mean([sum((x + dx, y + dy) in sec for x, y in um) for dx, dy in NULL_OFFSETS]))}
    return out


def _overlap_inputs(out):
    sh = out / "sample_heldout.csv"
    return (read_csv(out / "labels.csv"), read_csv(out / "labels_heldout.csv"), read_csv(out / "sample.csv"),
            read_csv(sh) if sh.exists() else [], read_csv(out / "round1_keypoints.csv"))


def cmd_overlap(args):
    out = Path(args.out)
    rep = overlap_report(*_overlap_inputs(out))
    write_json(out / "overlap.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "note"}, indent=1))


# --- respond (makelab2) -----------------------------------------------------------------------------

RESPONSE_FIELDS = (["stratum", "label_uid", "city", "pano_id", "checkpoint", "T_deg", "strip_x", "strip_y",
                    "d_x_b100", "d_y_b100", "d_y_pitch_b100", "d_y_roll_b100"]
                   + ["{}_{}".format(w, f) for w, _, _ in WINDOWS
                      for f in ("dx", "dy", "value", "raw_value", "edge")])


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
    """Sub-pixel argmax of the RAW (unclipped) heatmap within ``radius`` heatmap px of (cx, cy).

    Unclipped because clipping at 1.0 turns the top of a strong peak into a plateau, and argmax on a
    plateau returns its top-left pixel (review N1 on PR #244). -> (x, y, raw value, edge hit) or None.
    """
    H, W = hm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist2 <= radius ** 2
    if not mask.any():
        return None
    iy, ix = np.unravel_index(int(np.argmax(np.where(mask, hm, -np.inf))), hm.shape)
    x, y = _subpixel(hm, iy, ix)
    edge = math.sqrt(dist2[iy, ix]) >= radius - EDGE_TOL_HM
    return x, y, float(hm[iy, ix]), int(edge)


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
    import cv2
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from torchvision import transforms
    from rampnet.gsv import equirectangular_to_perspective
    from rampnet.model import CROP_INPUT_SIZE

    sample = read_csv(args.sample)
    if args.limit:
        sample = sample[:args.limit]
    models, shas, dev = load_crop_models(args.hf_cache)
    pre = transforms.Compose([
        transforms.Resize(CROP_INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t0 = time.time()
    gpu_s = 0.0
    rows, missing, shapes = [], [], {}

    def decode(r):
        p = Path(args.store) / r["city"] / r["pano_id"][:2] / (r["pano_id"] + ".jpg")
        return r, (cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, bgr) in enumerate(ex.map(decode, sample)):
            if bgr is None:
                missing.append(r["label_uid"])
                continue
            key = "{}x{}".format(bgr.shape[1], bgr.shape[0])
            shapes[key] = shapes.get(key, 0) + 1
            rgb = cv2.cvtColor(cv2.resize(bgr, (8192, 4096), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            del bgr
            g0 = time.time()
            persp = equirectangular_to_perspective(rgb, 90, int(r["nearest_theta"]), -30, 2048, 2048)
            strip = persp[0:2048, sg.STRIP_SLICE[0]:sg.STRIP_SLICE[1]]
            x = pre(Image.fromarray(strip)).unsqueeze(0).to(dev)
            sx, sy = float(r["strip_x"]) * HM_SCALE, float(r["strip_y"]) * HM_SCALE
            mx = sx + 0.5 * float(r["d_x_b100"]) * HM_SCALE
            my = sy + 0.5 * float(r["d_y_b100"]) * HM_SCALE
            for name, m in models.items():
                with torch.no_grad():
                    hm = m(x).squeeze().float().cpu().numpy()
                row = {k: r[k] for k in ("stratum", "label_uid", "city", "pano_id", "T_deg", "strip_x",
                                         "strip_y", "d_x_b100", "d_y_b100", "d_y_pitch_b100", "d_y_roll_b100")}
                row["checkpoint"] = name
                for wname, radius, centre in WINDOWS:
                    cx, cy = (sx, sy) if centre == "stored" else (mx, my)
                    px, py, val, edge = peak_near(hm, cx, cy, radius)
                    row.update({wname + "_dx": rnd((px - sx) / HM_SCALE), wname + "_dy": rnd((py - sy) / HM_SCALE),
                                wname + "_value": rnd(min(max(val, 0.0), 1.0)), wname + "_raw_value": rnd(val),
                                wname + "_edge": edge})
                rows.append(row)
            gpu_s += time.time() - g0
            if (i + 1) % 50 == 0:
                print("{}/{} panos, {:.0f} s".format(i + 1, len(sample), time.time() - t0), flush=True)
    elapsed = time.time() - t0
    out = Path(args.out)
    write_csv(out / "response{}.csv".format(args.suffix), RESPONSE_FIELDS, rows)
    usage = {"elapsed_s": round(elapsed, 1), "render_and_forward_s": round(gpu_s, 1),
             "panos": len(sample) - len(missing), "missing": missing, "jpeg_shapes": shapes,
             "checkpoint_sha256": shas, "device": str(dev), "sample": str(args.sample),
             "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             "ts_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(args.usage_out), usage)
    print(json.dumps(usage, indent=1))


# --- response fit (from response*.csv, so --check re-derives it) ------------------------------------

def response_from_v1(rows, sample):
    """Map the first respond run's response.csv (2026-10-06, PR #244 as first opened) to the current
    column names, so the review's per-stratum, edge and decomposition fits run on it.

    That run localized peaks on the CLIPPED heatmap (review N1) and searched two windows centred on the
    stored target, so it has no "mid" window and its raw value is the clipped one. Edge hits are
    recomputed from the refined peak's distance to the window centre.
    """
    smp = {r["label_uid"]: r for r in sample}
    out = []
    for r in rows:
        s = smp[r["label_uid"]]
        row = {k: r[k] for k in ("stratum", "label_uid", "city", "pano_id", "checkpoint", "T_deg",
                                 "d_x_b100", "d_y_b100")}
        row.update({k: s[k] for k in ("strip_x", "strip_y", "d_y_pitch_b100", "d_y_roll_b100")})
        for wname, pre, radius in (("stored", "peak", EVAL_RADIUS_HM), ("stored_x2", "peak2", 2 * EVAL_RADIUS_HM)):
            dx, dy = float(r[pre + "_minus_stored_x"]), float(r[pre + "_minus_stored_y"])
            dist_hm = math.hypot(dx, dy) * HM_SCALE
            row.update({wname + "_dx": r[pre + "_minus_stored_x"], wname + "_dy": r[pre + "_minus_stored_y"],
                        wname + "_value": r[pre + "_value"], wname + "_raw_value": r[pre + "_value"],
                        wname + "_edge": "1" if dist_hm >= radius - EDGE_TOL_HM else "0"})
        out.append(row)
    return out


def round1_matches(labels, crops):
    """label_uid -> sorted round-1 splits whose crop's own keypoint equals the label's paper point."""
    main = {}
    for c in crops:
        kps = _parse_crop_id(c["crop_id"])
        if kps:
            main.setdefault(kps[0], set()).add(c["split"])
    return {r["label_uid"]: sorted(main.get((int(r["paper_x"]), int(r["paper_y"])), ())) for r in labels}


def fit_response(rows, strata_share, r1_match=None):
    """Regress peak - stored on d (beta = 1), per checkpoint, window, axis and stratum; SE clustered by pano.

    ``strata_share``: the population share of each |T| stratum (summary.json's sample_strata_share for
    crowd_ok labels). The population-weighted fit reweights each row by share / sample share, so its
    slope answers "what is the slope over the labels round 1 was built from", not over this
    deliberately tail-heavy sample.
    """
    out = {"model": "peak_minus_stored = c + slope * d_b100 (render px); slope +1 = peak on the object "
                    "(at beta = 1), 0 = peak on the displaced target; SE clustered by pano; 95% CI = "
                    "slope +/- 1.96 SE. Peaks localized on the unclipped heatmap.",
           "windows": {w: {"radius_render_px": rnd(r / HM_SCALE), "centre": c} for w, r, c in WINDOWS
                       if rows and w + "_dy" in rows[0]},
           "edge_tolerance_render_px": rnd(EDGE_TOL_HM / HM_SCALE),
           "sigma_render_px": SIGMA_RENDER, "population_strata_share": strata_share}

    def f(r, k):
        return float(r[k])

    for ck in sorted({r["checkpoint"] for r in rows}):
        rr = [r for r in rows if r["checkpoint"] == ck]
        n_s = {s: sum(r["stratum"] == s for r in rr) for s, _, _ in SAMPLE_STRATA}
        block = {"n": len(rr), "strata": n_s}
        for wname, _, _ in WINDOWS:
            if wname + "_dy" not in rr[0]:
                continue
            wb = {"edge_hits": {s: sum(r[wname + "_edge"] == "1" for r in rr if r["stratum"] == s)
                                for s, _, _ in SAMPLE_STRATA},
                  "median_peak_value": rnd(np.median([f(r, wname + "_value") for r in rr])),
                  "share_raw_peak_ge_1": rnd(np.mean([f(r, wname + "_raw_value") >= 1.0 for r in rr]))}
            for ax in ("x", "y"):
                d = np.array([f(r, "d_{}_b100".format(ax)) for r in rr])
                y = np.array([f(r, "{}_d{}".format(wname, ax)) for r in rr])
                cl = np.array([r["pano_id"] for r in rr])
                st = np.array([r["stratum"] for r in rr])
                ab = {"all": _slope_block(d, y, cl)}
                for s, _, _ in SAMPLE_STRATA:
                    m = st == s
                    ab["stratum_" + s] = _slope_block(d[m], y[m], cl[m])
                noedge = np.array([r[wname + "_edge"] == "0" for r in rr])
                ab["no_edge_hits"] = _slope_block(d[noedge], y[noedge], cl[noedge])
                small = np.abs(d) < 45
                ab["abs_d_lt_45px"] = _slope_block(d[small], y[small], cl[small])
                if strata_share and all(n_s[s] for s in n_s):
                    wts = np.array([strata_share[s] / (n_s[s] / len(rr)) for s in st])
                    ab["population_weighted"] = _slope_block(d, y, cl, wts)
                if r1_match is not None:
                    mt = np.array([bool(r1_match.get(r["label_uid"])) for r in rr])
                    tr = np.array(["train" in r1_match.get(r["label_uid"], []) for r in rr])
                    ab["round1_unmatched"] = _slope_block(d[~mt], y[~mt], cl[~mt])
                    ab["round1_matched"] = _slope_block(d[mt], y[mt], cl[mt])
                    ab["round1_matched_train"] = _slope_block(d[tr], y[tr], cl[tr])
                    # independent label sets, so SE of the difference = sqrt(se1^2 + se2^2)
                    for tag, keep in (("all", np.ones(len(rr), bool)), ("stratum_ge3", st == "ge3")):
                        a = _slope_block(d[~mt & keep], y[~mt & keep], cl[~mt & keep])
                        b = _slope_block(d[tr & keep], y[tr & keep], cl[tr & keep])
                        if "slope" in a and "slope" in b:
                            dse = math.hypot(a["slope_se"], b["slope_se"])
                            ab["unmatched_minus_matched_train_" + tag] = {
                                "diff": rnd(a["slope"] - b["slope"]), "se": rnd(dse),
                                "z": rnd((a["slope"] - b["slope"]) / dse), "unmatched": a["slope"],
                                "unmatched_se": a["slope_se"], "matched_train": b["slope"],
                                "matched_train_se": b["slope_se"]}
                wb[ax] = ab
            # S2: the pitch and roll parts of d_y as separate regressors
            dp = np.array([f(r, "d_y_pitch_b100") for r in rr])
            dr = np.array([f(r, "d_y_roll_b100") for r in rr])
            y = np.array([f(r, wname + "_dy") for r in rr])
            fit = _ols(np.column_stack([np.ones_like(dp), dp, dr]), y, np.array([r["pano_id"] for r in rr]))
            if fit is not None:
                coef, se, _ = fit
                wb["y_pitch_roll_parts"] = {"slope_pitch": rnd(coef[1]), "slope_pitch_se": rnd(se[1]),
                                            "slope_roll": rnd(coef[2]), "slope_roll_se": rnd(se[2])}
            # S4: x error = (peak - stored) - d_x, in render px of the stored-target frame (peaks are
            # converted at x0.5 as train.py's targets are). A peak on the flip-averaged training target
            # predicts a constant +10 px, slope 0; a peak exactly on the object predicts +0.031 * strip_x
            # (the image's true x scale is 352/683, not 0.5). Review of PR #244, second round.
            sxs = np.array([f(r, "strip_x") for r in rr])
            ex = np.array([f(r, wname + "_dx") - f(r, "d_x_b100") for r in rr])
            fit = _ols(np.column_stack([np.ones_like(sxs), sxs]), ex, np.array([r["pano_id"] for r in rr]))
            if fit is not None:
                coef, se, _ = fit
                wb["x_error_on_strip_x"] = {"intercept_render_px": rnd(coef[0]), "intercept_se": rnd(se[0]),
                                            "slope": rnd(coef[1]), "slope_se": rnd(se[1]),
                                            "predicted_if_on_flip_averaged_target": {"intercept_render_px": 10.0,
                                                                                     "slope": 0.0},
                                            "predicted_if_on_object": {"intercept_render_px": 0.0, "slope": 0.031}}
            block[wname] = wb
        out[ck] = block
    # paired: the same strips under both checkpoints
    by = {}
    for r in rows:
        by.setdefault(r["label_uid"], {})[r["checkpoint"]] = r
    pairs = [v for _, v in sorted(by.items()) if "round1" in v and "round2" in v]
    if len(pairs) > 10:
        diff = {"n": len(pairs), "note": "(peak_round2 - peak_round1) = c + slope * d_b100, paired on strips"}
        cl = np.array([v["round1"]["pano_id"] for v in pairs])
        for wname, _, _ in WINDOWS:
            if wname + "_dy" not in pairs[0]["round1"]:
                continue
            for ax in ("x", "y"):
                d = np.array([f(v["round1"], "d_{}_b100".format(ax)) for v in pairs])
                y = np.array([f(v["round2"], "{}_d{}".format(wname, ax)) - f(v["round1"], "{}_d{}".format(wname, ax))
                              for v in pairs])
                diff["{}:{}".format(wname, ax)] = _slope_block(d, y, cl)
        out["round2_minus_round1"] = diff
    return out


# --- check ------------------------------------------------------------------------------------------

#: (response file stem, labels stem): response_v1.csv is the first run (2026-10-06, clipped peaks, old
#: column layout); response.csv and response_heldout.csv are where a respond run in the current layout
#: lands, so retrieving the second run never overwrites the first.
RESPONSE_SETS = [("response_v1", ""), ("response", ""), ("response_heldout", "_heldout")]


def load_response(out, name, labels_stem, labels):
    """<name>.csv in the current format (the first run's layout is adapted) plus round-1 match flags."""
    rows = read_csv(out / "{}.csv".format(name))
    if rows and "stored_dy" not in rows[0]:
        rows = response_from_v1(rows, read_csv(out / "sample{}.csv".format(labels_stem)))
    kp = out / "round1_keypoints.csv"
    match = round1_matches(labels, read_csv(kp)) if kp.exists() else None
    return rows, match


def cmd_check(args):
    out = Path(args.out)
    ok = True

    def cmp(name, want_bytes):
        nonlocal ok
        same = (out / name).read_bytes() == want_bytes
        ok &= same
        print("{:<24} {}".format(name, "ok" if same else "DIFFERS"))

    if args.pano_tools_root:
        pool, pose = load_inputs(args.pano_tools_root)
    strata_share = None
    for stem, heldout in (("", False), ("_heldout", True)):
        rows = read_csv(out / "labels{}.csv".format(stem))
        if args.pano_tools_root:
            fresh, funnel = build_labels(pool, pose, heldout=heldout)
            cmp("labels{}.csv".format(stem), csv_bytes(LABEL_FIELDS, fresh))
        else:
            # without the sibling checkout the funnel is copied from the committed summary, and
            # labels*.csv is taken as given (CI never re-derives it)
            funnel = json.loads((out / "summary{}.json".format(stem)).read_text(encoding="utf-8"))["funnel"]
        summary = summarize(rows)
        summary["funnel"] = funnel
        cmp("summary{}.json".format(stem), json_bytes(summary))
        if not heldout:
            strata_share = summary["crowd_ok"]["sample_strata_share"]
        if (out / "sample{}.csv".format(stem)).exists():
            cmp("sample{}.csv".format(stem), csv_bytes(SAMPLE_FIELDS, draw_sample(rows, args.per_stratum, args.seed)))
        for name, lstem in RESPONSE_SETS:
            if lstem == stem and (out / "{}.csv".format(name)).exists():
                resp, match = load_response(out, name, lstem, rows)
                cmp("{}.json".format(name), json_bytes(fit_response(resp, strata_share, match)))
    if (out / "round1_keypoints.csv").exists():
        cmp("overlap.json", json_bytes(overlap_report(*_overlap_inputs(out))))
    if not ok:
        sys.exit(1)


def cmd_fit(args):
    """Write <name>.json from <name>.csv (CPU; the respond run only writes the CSV)."""
    out = Path(args.out)
    share = json.loads((out / "summary.json").read_text(encoding="utf-8"))["crowd_ok"]["sample_strata_share"]
    lstem = dict(RESPONSE_SETS)[args.name]
    resp, match = load_response(out, args.name, lstem, read_csv(out / "labels{}.csv".format(lstem)))
    write_json(out / "{}.json".format(args.name), fit_response(resp, share, match))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="re-derive every summary from the committed CSVs, byte for byte")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--pano-tools-root", default=None,
                    help="with --check: also re-derive labels*.csv from the pinned pano-tools inputs")
    ap.add_argument("--per-stratum", type=int, default=200)
    ap.add_argument("--seed", type=int, default=113)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("predict")
    p.add_argument("--pano-tools-root", required=True)
    p.add_argument("--out", default=str(OUT_DEFAULT))
    s = sub.add_parser("sample")
    s.add_argument("--labels", default=str(OUT_DEFAULT / "labels.csv"))
    s.add_argument("--out", default=str(OUT_DEFAULT / "sample.csv"))
    s.add_argument("--per-stratum", type=int, default=200)
    s.add_argument("--seed", type=int, default=113)
    k = sub.add_parser("fetch-keypoints")
    k.add_argument("--out", default=str(OUT_DEFAULT))
    o = sub.add_parser("overlap")
    o.add_argument("--out", default=str(OUT_DEFAULT))
    r = sub.add_parser("respond")
    r.add_argument("--sample", required=True)
    r.add_argument("--store", required=True)
    r.add_argument("--hf-cache", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--suffix", default="", help="response<suffix>.csv, e.g. _heldout")
    r.add_argument("--usage-out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0, help="first N sample rows only (smoke test)")
    fi = sub.add_parser("fit")
    fi.add_argument("--out", default=str(OUT_DEFAULT))
    fi.add_argument("--name", default="response", choices=[n for n, _ in RESPONSE_SETS])
    args = ap.parse_args(argv)
    if args.check:
        return cmd_check(args)
    handlers = {"predict": cmd_predict, "sample": cmd_sample, "fetch-keypoints": cmd_fetch_keypoints,
                "overlap": cmd_overlap, "respond": cmd_respond, "fit": cmd_fit}
    if args.cmd not in handlers:
        ap.print_help()
        return None
    return handlers[args.cmd](args)


if __name__ == "__main__":
    main()
