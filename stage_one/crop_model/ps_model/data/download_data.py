import pandas as pd
import io
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import requests
import torch
import torch.nn.functional as F
import os
import random
import time
from datetime import timedelta
import string

label_type = "CurbRamp"

def fetch_panorama(pano_id):
    def _fetch_tile(x, y, zoom=4):
        url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
        try:
            response = requests.get(url, timeout=10)
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
        x, y = 5, 2

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

        with ThreadPoolExecutor(max_workers=100) as executor:
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
    return cv2.cvtColor(cv2.resize(cropped_panorama, (8192, 4096), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)

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

def equirectangular_point_to_perspective(label_x, label_y, equi_width, equi_height, fov, theta, phi, height, width):
    lon_deg = (label_x / equi_width) * 360.0 - 180.0
    lat_deg = 90.0 - (label_y / equi_height) * 180.0

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    point = np.array([
        np.cos(lat) * np.cos(lon),
        np.sin(lat),
        np.cos(lat) * np.sin(lon)
    ])

    theta_rad = np.deg2rad(theta)
    phi_rad   = np.deg2rad(phi)
    forward = np.array([
        np.cos(phi_rad) * np.cos(theta_rad),
        np.sin(phi_rad),
        np.cos(phi_rad) * np.sin(theta_rad)
    ])
    forward /= np.linalg.norm(forward)
    
    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    
    p_cam_x = np.dot(point, right)
    p_cam_y = np.dot(point, up)
    p_cam_z = np.dot(point, forward)
    
    if p_cam_z <= 0:
        return None

    fov_rad = np.deg2rad(fov)
    f = (width / 2) / np.tan(fov_rad / 2)

    x_img = (p_cam_x * f / p_cam_z) + (width / 2)
    y_img = -(p_cam_y * f / p_cam_z) + (height / 2)
    
    return int(x_img), int(y_img)

blackhawk_hills = pd.read_csv('https://sidewalk-blackhawk-hills.cs.washington.edu/v2/rawLabels?filetype=csv')

chicago = pd.read_csv(io.StringIO(requests.get('https://sidewalk-chicago.cs.washington.edu/v2/rawLabels?filetype=csv').text.replace("Little Italy, UIC", "").replace("Sauganash,Forest Glen", "")))
cliffside_park = pd.read_csv('https://sidewalk-cliffside-park.cs.washington.edu/v2/rawLabels?filetype=csv')
columbus = pd.read_csv('https://sidewalk-columbus.cs.washington.edu/v2/rawLabels?filetype=csv')
knox = pd.read_csv('https://sidewalk-knox.cs.washington.edu/v2/rawLabels?filetype=csv')
mendota = pd.read_csv('https://sidewalk-mendota.cs.washington.edu/v2/rawLabels?filetype=csv')
newberg = pd.read_csv('https://sidewalk-newberg.cs.washington.edu/v2/rawLabels?filetype=csv')
oradell = pd.read_csv('https://sidewalk-oradell.cs.washington.edu/v2/rawLabels?filetype=csv')
pittsburgh = pd.read_csv('https://sidewalk-pittsburgh.cs.washington.edu/v2/rawLabels?filetype=csv')
sea = pd.read_csv('https://sidewalk-sea.cs.washington.edu/v2/rawLabels?filetype=csv')
st_louis = pd.read_csv('https://sidewalk-st-louis.cs.washington.edu/v2/rawLabels?filetype=csv')
teaneck = pd.read_csv('https://sidewalk-teaneck.cs.washington.edu/v2/rawLabels?filetype=csv')

all_cities = pd.concat([blackhawk_hills, chicago,
                        cliffside_park, columbus, knox, mendota, newberg, oradell,
                        pittsburgh, sea, st_louis, teaneck])

all_cities = all_cities[all_cities['Label Type'].isin([label_type])]

all_cities = all_cities[
    all_cities['Agree Count'] - all_cities['Disagree Count'] >= 2
]

total_images = len(all_cities)

all_cities = [x for _, x in all_cities.sample(frac=1).reset_index(drop=True).groupby('Panorama ID')]


start_time = time.time()
processed_count = 0

os.makedirs(f"dataset_1", exist_ok=False)

alphabet = string.ascii_lowercase + string.digits

for group in all_cities:
    for index, row in group.iterrows():
        try:
            pano_id = row['Panorama ID']
            label_x = row['Panorama X']
            label_y = row['Panorama Y']
            pano_width = row['Panorama Width']
            pano_height = row['Panorama Height']

            equi_img_bgr = fetch_panorama(pano_id)

            if equi_img_bgr is not None:
                equi_img_rgb = cv2.cvtColor(equi_img_bgr, cv2.COLOR_BGR2RGB)
                equi_h, equi_w = equi_img_rgb.shape[:2]
                label_x_norm = label_x / pano_width
                label_y_norm = label_y / pano_height
                theta = label_x_norm * 360 - 180
                nearest_theta = round(theta / 30) * 30
                cropped_perspective = equirectangular_to_perspective(equi_img_rgb, 90, nearest_theta, -30, 2048, 2048)
                point_x, point_y = equirectangular_point_to_perspective(label_x_norm * equi_w, label_y_norm * equi_h, equi_w, equi_h, 90, nearest_theta, -30, 2048, 2048)
                cropped_perspective = cropped_perspective[0:2048, int(2048 / 3):int(2048 / 3 * 2)]
                point_x = int(point_x - (2048 / 3))
                filename = f"dataset_1/{''.join(random.choices(alphabet, k=8))}_-_{point_x}_{point_y}"
                for index2, row2 in group.iterrows():
                    if index2 == index:
                        continue
                    label_x2 = row2['Panorama X']
                    label_y2 = row2['Panorama Y']
                    label_x_norm2 = label_x2 / pano_width
                    label_y_norm2 = label_y2 / pano_height
                    result2 = equirectangular_point_to_perspective(label_x_norm2 * equi_w, label_y_norm2 * equi_h, equi_w, equi_h, 90, nearest_theta, -30, 2048, 2048)
                    if result2 is not None:
                        point_x2, point_y2 = result2
                        point_x2 = int(point_x2 - (2048 / 3))
                        if point_x2 <= 683 and point_x2 >= 0:
                            filename += f"_-_{point_x2}_{point_y2}"
                filename += ".jpg"
                cv2.imwrite(filename, cv2.cvtColor(cropped_perspective, cv2.COLOR_RGB2BGR))
                processed_count += 1
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = total_images - processed_count
                estimated_remaining_time = avg_time_per_image * remaining_images
                print(f"Processed {processed_count}/{total_images} images. Saved image to {filename}. "
                        f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, "
                        f"Time Left: {estimated_remaining_time}")
            else:
                processed_count += 1
                print(f"Processed {processed_count}/{total_images} images. Could not fetch panorama for label ID {row['Label ID']}")
        except Exception as e:
            processed_count += 1
            print(f"Processed {processed_count}/{total_images} images. Error processing label ID {row['Label ID']}: {e}")

print("Processing complete.")