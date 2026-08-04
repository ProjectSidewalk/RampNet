import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
from pyproj import Geod
from search_panos import search_panoramas, get_pano_heading
from inference_isolator import infer_image
# The GSV fetch and both projections live in the rampnet package so that
# analysis tooling (#103) renders with the exact production path. One
# definition each — edit them there, not here.
from rampnet.gsv import (
    heading_to_azimuth,
    fetch_panorama,
    equirectangular_to_perspective,
    perspective_to_equirectangular,
)
import string
import random
from tqdm import tqdm
import os
import threading
import time
from skimage.feature import peak_local_max

progress_lock = threading.Lock()

geod = Geod(ellps="WGS84")

def mark_done(idx):
    with progress_lock:
        with open("progress.txt", "a") as pf:
            pf.write(f"{idx}\n")

def process_line(idx, line):
    try:
        data = json.loads(line.rstrip())
        equi = fetch_panorama(data["pano_id"])
        if equi is None:
            print(f"Failed to fetch panorama for pano_id {data['pano_id']}")
            return
        pano_coords = data["pano_coords"]
        curb_ramp_coords = data["curb_ramps_coords"]
        pano_angle = heading_to_azimuth(get_pano_heading(data["pano_id"]))
        combined_heatmap = np.zeros((2048, 4096), dtype=np.uint8)
        for curb_ramp_coord in curb_ramp_coords:
            azimuth, _, _ = geod.inv(pano_coords[1], pano_coords[0], curb_ramp_coord[1], curb_ramp_coord[0])
            azimuth = azimuth - pano_angle
            persp = equirectangular_to_perspective(equi, 90, azimuth, -30, 1024, 1024)
            persp = persp[0:1024, 341:341+341]
            heatmap = infer_image(persp)
            heatmap = cv2.resize(heatmap, (341, 1024), interpolation=cv2.INTER_CUBIC)
            left_padding = 341
            right_padding = 342
            heatmap = cv2.copyMakeBorder(heatmap, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)
            heatmap = np.clip(heatmap, 0, 1)
            heatmap = np.stack((heatmap * 255,)*3, axis=-1)
            heatmap = perspective_to_equirectangular(heatmap, 90, azimuth, -30, 2048, 4096)
            heatmap = heatmap[..., 0].astype(np.uint8)
            combined_heatmap = np.maximum(combined_heatmap, heatmap)
        
        min_peak_distance = 40
        threshold_abs_value = 0.4 * 255

        peak_coords_raw = peak_local_max(
            combined_heatmap,
            min_distance=min_peak_distance,
            threshold_abs=threshold_abs_value, 
            exclude_border=False,
        )

        centers = []
        heatmap_h, heatmap_w = combined_heatmap.shape[:2]
        for r, c in peak_coords_raw:
            cx_normalized = c / heatmap_w
            cy_normalized = r / heatmap_h
            centers.append((cx_normalized, cy_normalized))

        name = data["pano_id"]
        
        
        equi_out = f"../../dataset/{name}.jpg"
        json_out = f"../../dataset/{name}.json"

        json_result = {
            "record_creation_time": int(time.time()),
            "pano_id": data["pano_id"],
            "curb_ramp_points_normalized": centers,
            "pano_coord": pano_coords,
            "curb_ramp_coords": curb_ramp_coords,
            "pano_azimuth": pano_angle
        }

        cv2.imwrite(equi_out, equi)
        with open(json_out, 'w') as f:
            json.dump(json_result, f, indent=4)


        mark_done(idx)
    except Exception as e:
        print(f"Error processing line index {idx}: {e}")

if __name__ == "__main__":
    if os.path.exists("progress.txt"):
        with open("progress.txt") as pf:
            done_indices = set(int(line.strip()) for line in pf if line.strip().isdigit())
    else:
        done_indices = set()
    with open("finaldataset.jsonl") as file:
        all_lines = file.readlines()
    lines_to_process = [(idx, line) for idx, line in enumerate(all_lines) if idx not in done_indices]
    with ThreadPoolExecutor(max_workers=26) as executor:
        list(tqdm(executor.map(lambda p: process_line(*p), lines_to_process), total=len(lines_to_process)))
