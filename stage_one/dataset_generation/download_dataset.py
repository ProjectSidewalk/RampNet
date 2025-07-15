import json
import requests
from requests.adapters import HTTPAdapter
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import io
import torch
import torch.nn.functional as F
from pyproj import Geod
from search_panos import search_panoramas, get_pano_heading
from inference_isolator import infer_image
import string
import random
from tqdm import tqdm
import os
import threading
import time
from skimage.feature import peak_local_max

progress_lock = threading.Lock()

def heading_to_azimuth(heading_degrees):
    heading_degrees %= 360
    azimuth = (heading_degrees + 180) % 360 - 180
    return azimuth

def fetch_panorama(pano_id):
    def _fetch_tile(x, y, zoom=3):
        url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
        try:
            s = requests.Session()
            s.mount("https://", HTTPAdapter(max_retries=1))
            response = s.get(url, timeout=20)
            if response.status_code == 200:
                return x, y, Image.open(io.BytesIO(response.content))
            return x, y, None
        except Exception as e:
            print(f"Error fetching tile for pano {pano_id}, x={x}, y={y}: {e}")
            return x, y, None

    def _is_black_tile(tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _find_panorama_dimensions():
        tiles_cache = {}
        x, y = 4, 1
        is_first = True
        while True:
            tile_info = _fetch_tile(x, y)
            if tile_info is None:
                return None
            tile = tile_info[2]
            if tile is None:
                return None
            if is_first:
                is_first = False
                if _is_black_tile(tile):
                    return None
            tiles_cache[(x, y)] = tile
            if _is_black_tile(tile):
                y = y - 1
                while True:
                    tile_info = _fetch_tile(x, y)
                    if tile_info is None:
                        return None
                    tile = tile_info[2]
                    tiles_cache[(x, y)] = tile
                    if _is_black_tile(tile):
                        return x - 1, y, tiles_cache
                    x += 1
            x += 1
            y += 1

    def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(_fetch_tile, x, y))
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    x, y, tile = result
                    if tile is not None:
                        tiles_cache[(x, y)] = tile
        return tiles_cache

    def _assemble_panorama(tiles, max_x, max_y):
        if not tiles:
            return None
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))
        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))
        return panorama

    def _crop(image):
        img_array = np.array(image)
        y_nonzero, x_nonzero, _ = np.nonzero(img_array)
        if y_nonzero.size > 0 and x_nonzero.size > 0:
            return img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        return img_array

    dimension_result = _find_panorama_dimensions()
    if dimension_result is None:
        return None
    max_x, max_y, initial_tiles = dimension_result
    full_tiles = _fetch_remaining_tiles(max_x, max_y, initial_tiles)
    assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
    if assembled_panorama is None:
        return None
    cropped_panorama = _crop(assembled_panorama)
    height, width = cropped_panorama.shape[:2]

    max_width = height * 2
    cropped_panorama = cropped_panorama[:, :max_width]
    
    resized = cv2.resize(cropped_panorama, (4096, 2048), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def equirectangular_to_perspective(equi_img, fov, theta, phi, height, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.tensor(equi_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    h, w = equi_img.shape[:2]
    hFOV = float(height) / width * fov
    w_len = torch.tan(torch.deg2rad(torch.tensor(fov / 2.0, device=device)))
    h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=device)))
    x_map = torch.ones((height, width), dtype=torch.float32, device=device)
    y_map = torch.linspace(-w_len, w_len, width, device=device).repeat(height, 1)
    z_map = -torch.linspace(-h_len, h_len, height, device=device).unsqueeze(1).repeat(1, width)
    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1) / D.unsqueeze(-1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))
    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)
    xyz = xyz.view(-1, 3).T
    xyz = torch.matmul(R1, xyz)
    xyz = torch.matmul(R2, xyz).T
    xyz = xyz.view(height, width, 3)
    lat = torch.asin(xyz[:, :, 2])
    lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])
    lon = lon / np.pi * (w - 1) / 2.0 + (w - 1) / 2.0
    lat = lat / (np.pi / 2.0) * (h - 1) / 2.0 + (h - 1) / 2.0
    lat = h - lat
    lon = (lon / ((w - 1) / 2.0)) - 1
    lat = (lat / ((h - 1) / 2.0)) - 1
    grid = torch.stack((lon, lat), dim=-1).unsqueeze(0)
    persp = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return (persp[0].permute(1, 2, 0) * 255).byte().cpu().numpy()

def perspective_to_equirectangular(persp_img, fov, theta, phi, equi_height, equi_width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.tensor(persp_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    persp_h, persp_w = persp_img.shape[:2]
    hFOV = (persp_h / persp_w) * fov
    tan_fov = torch.tan(torch.deg2rad(torch.tensor(fov/2, device=device)))
    tan_hfov = torch.tan(torch.deg2rad(torch.tensor(hFOV/2, device=device)))
    u = torch.linspace(0, equi_width - 1, equi_width, device=device)
    v = torch.linspace(0, equi_height - 1, equi_height, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    lon = (u_grid / (equi_width - 1)) * 2 * np.pi - np.pi
    lat = (np.pi / 2) - (v_grid / (equi_height - 1)) * np.pi
    x_world = torch.cos(lat) * torch.cos(lon)
    y_world = torch.cos(lat) * torch.sin(lon)
    z_world = torch.sin(lat)
    v_world = torch.stack((x_world, y_world, z_world), dim=-1).view(-1, 3).T
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))
    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)
    R = R2 @ R1
    R_inv = R.t()
    v_camera = R_inv @ v_world
    v_camera = v_camera.T.view(equi_height, equi_width, 3)
    x_cam = v_camera[..., 0]
    y_cam = v_camera[..., 1]
    z_cam = v_camera[..., 2]
    eps = 1e-6
    valid_mask = x_cam > eps
    y_proj = torch.zeros_like(y_cam)
    z_proj = torch.zeros_like(z_cam)
    y_proj[valid_mask] = y_cam[valid_mask] / x_cam[valid_mask]
    z_proj[valid_mask] = z_cam[valid_mask] / x_cam[valid_mask]
    in_fov_mask = (y_proj >= -tan_fov) & (y_proj <= tan_fov) & (z_proj >= -tan_hfov) & (z_proj <= tan_hfov) & valid_mask
    u_persp = ((y_proj + tan_fov) / (2 * tan_fov)) * (persp_w - 1)
    v_persp = (((-z_proj) + tan_hfov) / (2 * tan_hfov)) * (persp_h - 1)
    norm_u = (u_persp / ((persp_w - 1) / 2)) - 1
    norm_v = (v_persp / ((persp_h - 1) / 2)) - 1
    grid = torch.stack((norm_u, norm_v), dim=-1).unsqueeze(0)
    equi = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    in_fov_mask = in_fov_mask.unsqueeze(0).unsqueeze(0).float()
    equi = equi * in_fov_mask
    equi = equi[0].permute(1, 2, 0) * 255.0
    equi = equi.byte().cpu().numpy()
    return equi

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
