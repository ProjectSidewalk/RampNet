"""Phase 0 item 3 (RampNet#274): do Mapillary Vistas' crosswalk classes land on crowd Crosswalk labels?

Runs `facebook/mask2former-swin-large-mapillary-vistas-semantic` (the #126 checkpoint) over RampNet
benchmark panos on the 1024 parity rig (equirect_tiling.default_views(): 6 yaws, 90x90 deg, pitch -30,
1024x1024 views, source capped at 4096; processor size forced to 1024x1024 as `--vistas-input-size 1024
1024` does), unions classes 8 (`Crosswalk - Plain`) and 23 (`Lane Marking - Crosswalk`), takes 8-connected
components per view (min area 16 px, the #126 cache floor), maps each centroid to the pano, and merges
components whose pano centroids are within 0.03 of the pano width (RampNet's matcher metric: x wrapped at
the seam and scaled by 1024, y by 512). One point per merged component.

Denominator (pinned on the issue): benchmark panos carrying >= 1 crowd Crosswalk label (server rawLabels,
pano_x/pano_width, pano_y/pano_height). Reading rule: merged-component recall >= 0.8 of those labels at
radius 0.05 of the pano width.

Per-pano components are cached as JSON under <out>/components/<split>/<pano>.json, so scoring is offline
and a run resumes. The one departure from #126's VistasDetector._raw_detect: the class/mask einsum is done
at the logits' own 256x256 and the argmax map upsampled (nearest) to the view, instead of upsampling every
query mask to 1024 first -- the memory-light order #126's note says to use next time; it differs only at
boundary pixels, which this probe does not read.

Usage (RampNet's .venv):
  python vistas_crosswalk_probe.py --rampnet D:/Git/RampNet --rawlabels <census>/out/rawlabels
      --split paterson=paterson-nj --split gainesville=gainesville-fl --out ./out [--all] [--score-only]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

MERGE_R = 0.03
MATCH_R = 0.05
CLASS_IDS = (8, 23)
CLASS_NAMES = {8: "Crosswalk - Plain", 23: "Lane Marking - Crosswalk"}
CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
SOURCE_MAX_EDGE = 4096
INPUT_SIZE = (1024, 1024)
MIN_AREA_PX = 16
SCALE_X, SCALE_Y = 1024.0, 512.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pano_dist(ax, ay, bx, by):
    """RampNet matcher distance in pano-width units (x wraps at the seam; y counts half)."""
    dx = abs(ax - bx)
    dx = min(dx, 1.0 - dx) * SCALE_X
    dy = (ay - by) * SCALE_Y
    return math.hypot(dx, dy) / SCALE_X


def verdict(validations_json):
    """Human-only margin rule (same as the census): net >= 2 Agree -> correct, net >= 2 Disagree -> incorrect."""
    try:
        votes = json.loads(validations_json) if validations_json else []
    except json.JSONDecodeError:
        return "undecided"
    agree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Agree")
    disagree = sum(1 for v in votes if v.get("validator_type") != "AI" and v.get("validation") == "Disagree")
    if agree - disagree >= 2:
        return "correct"
    if disagree - agree >= 2:
        return "incorrect"
    return "undecided"


def load_labels(rawlabels_dir, city, pano_ids):
    path = os.path.join(rawlabels_dir, f"{city}_Crosswalk.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r.get("label_type") != "Crosswalk" or r["pano_id"] not in pano_ids:
                continue
            try:
                x = float(r["pano_x"]) / float(r["pano_width"])
                y = float(r["pano_y"]) / float(r["pano_height"])
            except (ValueError, ZeroDivisionError, KeyError):
                continue
            out.setdefault(r["pano_id"], []).append({
                "label_uid": f"{city}:{r['label_id']}", "x": x, "y": y,
                "verdict": verdict(r.get("validations", "")), "server_correct": r.get("correct", "")})
    return out


class Segmenter:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
        self.processor.size = {"height": INPUT_SIZE[0], "width": INPUT_SIZE[1]}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(CHECKPOINT).eval()
        id2label = self.model.config.id2label
        for cid, name in CLASS_NAMES.items():
            actual = id2label.get(cid, id2label.get(str(cid)))
            if actual != name:
                raise RuntimeError(f"class {cid} is {actual!r}, expected {name!r}")
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.backend = None
        self._place()

    def _place(self, force_cpu=False):
        torch = self.torch
        if not force_cpu and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free >= 1.5 * 2 ** 30:
                self.model = self.model.half().to("cuda")
                self.device, self.dtype, self.backend = "cuda", "float16", "cuda-fp16"
                log(f"backend cuda fp16 ({free / 2**30:.1f} GB free)")
                return
            log(f"cuda has only {free / 2**30:.1f} GB free; using cpu")
        self.model = self.model.float().to("cpu")
        self.device, self.dtype, self.backend = "cpu", "float32", "cpu-fp32"
        log("backend cpu fp32")

    def seg_prob(self, image):
        """(seg, prob) at the view resolution: argmax class id per pixel and its normalized score."""
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == "float16":
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
        try:
            with torch.inference_mode():
                out = self.model(**inputs)
        except torch.cuda.OutOfMemoryError:
            log("cuda OOM; moving to cpu fp32 for the rest of the run")
            del inputs
            torch.cuda.empty_cache()
            self._place(force_cpu=True)
            return self.seg_prob(image)
        mask_probs = out.masks_queries_logits.float().sigmoid()                  # (1,Q,h,w)
        class_probs = out.class_queries_logits.float().softmax(-1)[..., :-1]      # (1,Q,C)
        semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)[0]     # (C,h,w)
        normalized = semantic / semantic.sum(0, keepdim=True).clamp_min(1e-6)
        prob, seg = normalized.max(0)
        H, W = image.height, image.width
        seg = torch.nn.functional.interpolate(seg[None, None].float(), size=(H, W), mode="nearest")[0, 0]
        prob = torch.nn.functional.interpolate(prob[None, None], size=(H, W), mode="bilinear",
                                               align_corners=False)[0, 0]
        return seg.to(torch.int64).cpu().numpy(), prob.cpu().numpy()


def view_components(seg, prob, view, perspective_point_to_equirect):
    import numpy as np
    from skimage.measure import label as cc_label, regionprops
    selected = np.isin(seg, list(CLASS_IDS))
    comps = []
    if not selected.any():
        return comps
    h, w = seg.shape
    for region in regionprops(cc_label(selected, connectivity=2), intensity_image=prob):
        if region.area < MIN_AREA_PX:
            continue
        cy, cx = region.centroid
        u, v = (cx + 0.5) / w, (cy + 0.5) / h
        X, Y = perspective_point_to_equirect(u, v, view)
        coords = region.coords
        left = coords[coords[:, 1].argmin()]
        right = coords[coords[:, 1].argmax()]
        XL, YL = perspective_point_to_equirect((left[1] + 0.5) / w, (left[0] + 0.5) / h, view)
        XR, YR = perspective_point_to_equirect((right[1] + 0.5) / w, (right[0] + 0.5) / h, view)
        cls = seg[coords[:, 0], coords[:, 1]]
        comps.append({"view_yaw": view.yaw_deg, "area_px": int(region.area),
                      "mean_prob": float(region.mean_intensity),
                      "share_plain": float((cls == 8).mean()),
                      "u": u, "v": v, "x": X, "y": Y,
                      "x_left": XL, "y_left": YL, "x_right": XR, "y_right": YR})
    return comps


def merge_components(comps, merge_r=MERGE_R):
    """Union-find over pano centroids within merge_r (matcher metric); area-weighted merged point."""
    n = len(comps)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if pano_dist(comps[i]["x"], comps[i]["y"], comps[j]["x"], comps[j]["y"]) <= merge_r:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(comps[i])
    merged = []
    for members in groups.values():
        members.sort(key=lambda c: -c["area_px"])
        ref = members[0]["x"]
        tot = sum(c["area_px"] for c in members)
        xs = 0.0
        for c in members:
            x = c["x"]
            if x - ref > 0.5:
                x -= 1.0
            elif ref - x > 0.5:
                x += 1.0
            xs += x * c["area_px"]
        merged.append({"x": (xs / tot) % 1.0, "y": sum(c["y"] * c["area_px"] for c in members) / tot,
                       "area_px": tot, "n_views": len({c["view_yaw"] for c in members}),
                       "n_components": len(members),
                       "mean_prob": sum(c["mean_prob"] * c["area_px"] for c in members) / tot,
                       "share_plain": sum(c["share_plain"] * c["area_px"] for c in members) / tot})
    return merged


def run_pano(seg_model, image_path, comp_path, rig):
    default_views, equirect_to_perspective, perspective_point_to_equirect, load_pano_image = rig
    pano = load_pano_image(image_path, SOURCE_MAX_EDGE)
    comps = []
    t0 = time.time()
    for view in default_views():
        view_img = equirect_to_perspective(pano, view)
        seg, prob = seg_model.seg_prob(view_img)
        comps += view_components(seg, prob, view, perspective_point_to_equirect)
    rec = {"pano_id": os.path.splitext(os.path.basename(image_path))[0],
           "source_size": [pano.width, pano.height], "backend": seg_model.backend,
           "seconds": round(time.time() - t0, 1), "components": comps}
    tmp = comp_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, comp_path)
    return rec


def score(out, split_city, labels_by_split, comp_dir, radii=(0.022, 0.05, 0.1)):
    rows, per_pano = [], []
    for split, city in split_city:
        d = os.path.join(comp_dir, split)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(d, fn)))
            merged = merge_components(rec["components"])
            labs = labels_by_split.get(split, {}).get(rec["pano_id"], [])
            per_pano.append({"split": split, "city": city, "pano_id": rec["pano_id"], "n_labels": len(labs),
                             "n_view_components": len(rec["components"]), "n_merged": len(merged),
                             "n_merged_near_label": sum(1 for m in merged if any(
                                 pano_dist(m["x"], m["y"], l["x"], l["y"]) <= MATCH_R for l in labs)),
                             "backend": rec["backend"], "seconds": rec["seconds"]})
            for l in labs:
                best = min(((pano_dist(m["x"], m["y"], l["x"], l["y"]), m) for m in merged),
                           key=lambda t: t[0], default=(None, None))
                rows.append({"split": split, "city": city, "pano_id": rec["pano_id"], **l,
                             "nearest_m_dist": None if best[0] is None else round(best[0], 4),
                             "nearest_area_px": None if best[1] is None else best[1]["area_px"],
                             "nearest_share_plain": None if best[1] is None else round(best[1]["share_plain"], 3),
                             "n_merged_on_pano": len(merged)})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "labels_scored.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "panos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_pano[0].keys()) if per_pano else ["split"])
        w.writeheader()
        w.writerows(per_pano)

    def recall(sub, r):
        return None if not sub else round(sum(1 for x in sub if x["nearest_m_dist"] is not None
                                              and x["nearest_m_dist"] <= r) / len(sub), 3)

    summary = {"merge_r": MERGE_R, "match_r": MATCH_R, "classes": CLASS_IDS, "cuts": {}}
    cuts = {"all": rows, "human_confirmed": [x for x in rows if x["verdict"] == "correct"]}
    for split, _ in split_city:
        cuts[split] = [x for x in rows if x["split"] == split]
        cuts[split + "_human_confirmed"] = [x for x in rows if x["split"] == split and x["verdict"] == "correct"]
    for name, sub in cuts.items():
        summary["cuts"][name] = {"n_labels": len(sub), "n_panos": len({x["pano_id"] for x in sub}),
                                 **{f"recall@{r}": recall(sub, r) for r in radii}}
    labeled = [p for p in per_pano if p["n_labels"]]
    unlabeled = [p for p in per_pano if not p["n_labels"]]
    summary["panos"] = {
        "labeled": {"n": len(labeled),
                    "merged_per_pano": round(sum(p["n_merged"] for p in labeled) / max(1, len(labeled)), 2),
                    "merged_within_match_r_of_a_label": sum(p["n_merged_near_label"] for p in labeled),
                    "merged_total": sum(p["n_merged"] for p in labeled)},
        "unlabeled": {"n": len(unlabeled),
                      "merged_per_pano": round(sum(p["n_merged"] for p in unlabeled) / max(1, len(unlabeled)), 2),
                      "panos_with_any_component": sum(1 for p in unlabeled if p["n_merged"])},
        "backends": sorted({p["backend"] for p in per_pano})}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rampnet", required=True)
    ap.add_argument("--rawlabels", required=True)
    ap.add_argument("--split", action="append", default=[], help="<benchmark split>=<PS city key>")
    ap.add_argument("--out", default="out")
    ap.add_argument("--all", action="store_true", help="also run panos with no crowd label (exploratory)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.rampnet)
    sys.path.insert(0, os.path.join(a.rampnet, "scripts", "model_comparison"))
    from equirect_tiling import default_views, equirect_to_perspective, perspective_point_to_equirect  # noqa: E402
    from detectors import load_pano_image  # noqa: E402
    rig = (default_views, equirect_to_perspective, perspective_point_to_equirect, load_pano_image)

    split_city = [s.split("=", 1) for s in a.split]
    comp_dir = os.path.join(a.out, "components")
    labels_by_split, todo = {}, []
    for split, city in split_city:
        pdir = os.path.join(a.rampnet, "benchmark", split, "panos")
        files = {os.path.splitext(f)[0]: os.path.join(pdir, f) for f in os.listdir(pdir)}
        labels_by_split[split] = load_labels(a.rawlabels, city, set(files))
        n_lab = sum(len(v) for v in labels_by_split[split].values())
        log(f"{split} ({city}): {len(files)} benchmark panos, {len(labels_by_split[split])} carry "
            f"{n_lab} crowd Crosswalk labels")
        os.makedirs(os.path.join(comp_dir, split), exist_ok=True)
        order = sorted(files, key=lambda p: (p not in labels_by_split[split], p))
        for pid in order:
            if pid not in labels_by_split[split] and not a.all:
                continue
            cp = os.path.join(comp_dir, split, pid + ".json")
            if not os.path.exists(cp):
                todo.append((split, files[pid], cp))
    if a.limit:
        todo = todo[:a.limit]
    log(f"{len(todo)} panos to run")
    if todo and not a.score_only:
        seg_model = Segmenter()
        meta = {"checkpoint": CHECKPOINT, "revision": seg_model.revision, "input_size": INPUT_SIZE,
                "source_max_edge": SOURCE_MAX_EDGE, "min_area_px": MIN_AREA_PX, "classes": CLASS_IDS,
                "views": [list(v) for v in default_views()], "merge_r": MERGE_R, "match_r": MATCH_R}
        os.makedirs(a.out, exist_ok=True)
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        for i, (split, path, cp) in enumerate(todo, 1):
            rec = run_pano(seg_model, path, cp, rig)
            log(f"[{i}/{len(todo)}] {split} {rec['pano_id']}: {len(rec['components'])} view components, "
                f"{rec['seconds']} s, {rec['backend']}")
    summary = score(a.out, split_city, labels_by_split, comp_dir)
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
