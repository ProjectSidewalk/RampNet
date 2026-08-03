"""The production Google Street View path: pano fetch + the two projections.

Lifted **verbatim** from ``stage_one/dataset_generation/download_dataset.py``
for #103 (the street-level review instrument), which must render *exactly* the
crop Stage 1 cuts. Before this move the production functions were unimportable:
``download_dataset.py`` imports ``inference_isolator``, which loads the round-2
crop-model checkpoint from a relative path at module import time — a path that
only resolves with ``cwd == stage_one/dataset_generation`` and a checkpoint
that is not in the repo. ``download_dataset.py`` now imports these functions
from here, so there is still exactly one definition of each.

The only edits in the move are import wiring: ``cv2``, ``requests``, and
``torch`` are imported lazily inside the functions that need them, because
``requirements-dev.txt`` deliberately excludes ``cv2``/``requests`` and the
test suite imports this module for its pure geometry helpers. Everything else
— tile endpoint, dimension probing, the 4096x2048 resize, **the BGR return**,
the grid_sample projection — is byte-for-byte the production behaviour.

Conventions callers must know (they have bitten before):

- ``fetch_panorama`` returns a ``(2048, 4096, 3)`` uint8 ndarray in **BGR**
  channel order (production writes it straight to ``cv2.imwrite``). Convert
  with ``cv2.cvtColor(..., COLOR_BGR2RGB)`` before handing it to PIL.
- ``equirectangular_to_perspective(equi, fov, theta, phi, height, width)``:
  ``theta`` is yaw in degrees **relative to the panorama heading**, increasing
  clockwise; ``phi`` is pitch, negative = down; output size is ``(height,
  width)`` in that order. Production always calls it as
  ``(equi, 90, azimuth, -30, 1024, 1024)`` and slices ``[0:1024, 341:341+341]``
  for the Stage 1 strip.
- Azimuth increases clockwise and maps to *rightward* in the rendered image —
  the same convention as ``scripts/analysis/stage1_bearing_residual.py`` (§5j),
  whose tests pin it.
"""
import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image


def heading_to_azimuth(heading_degrees):
    heading_degrees %= 360
    azimuth = (heading_degrees + 180) % 360 - 180
    return azimuth


def fetch_panorama(pano_id):
    import cv2
    import requests
    from requests.adapters import HTTPAdapter

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
    import cv2
    import torch
    import torch.nn.functional as F

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
    import cv2
    import torch
    import torch.nn.functional as F

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


# --- Pure helpers, new for #103 (not part of the verbatim move) -------------
#
# The click-to-angle map for the production perspective render. A pinhole
# projection is NOT linear in angle, so a pixel column converts through
# atan, exactly as scripts/analysis/stage1_offset_tolerance.py's
# crop_half_angle_deg() does — these share its formula, and tests/test_gsv.py
# asserts agreement with that function rather than with a literal.

PERSP_WIDTH = 1024
PERSP_FOV_DEG = 90.0


def perspective_col_to_azimuth_deg(col, width=PERSP_WIDTH, fov=PERSP_FOV_DEG):
    """Continuous pixel column in the production render -> signed azimuth
    offset from the view centre, degrees. Positive = right of centre =
    clockwise (the §5j residual sign convention)."""
    f = (width / 2.0) / math.tan(math.radians(fov / 2.0))
    return math.degrees(math.atan((col - width / 2.0) / f))


def azimuth_deg_to_perspective_col(deg, width=PERSP_WIDTH, fov=PERSP_FOV_DEG):
    """Inverse of :func:`perspective_col_to_azimuth_deg`. Defined for
    ``|deg| < 90``; an azimuth beyond ``fov/2`` maps to a column outside
    ``[0, width]`` (the caller decides whether that is drawable)."""
    if not -90.0 < deg < 90.0:
        raise ValueError(f"azimuth {deg} deg is not renderable in a {fov} deg pinhole view")
    f = (width / 2.0) / math.tan(math.radians(fov / 2.0))
    return width / 2.0 + f * math.tan(math.radians(deg))
