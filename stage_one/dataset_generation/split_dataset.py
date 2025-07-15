import os
import random
import shutil
from tqdm import tqdm
import json
import numpy as np
from sklearn.neighbors import BallTree
from math import radians
from collections import deque

CONSIDER_MANUAL = False
MANUAL_LABELS_DIR = "../../manual_labels"

DATASET_DIR = "../../dataset"
BASE_OUTPUT_DIR = "../../dataset_split"

OUTPUT_DIRS = {
    'train': os.path.join(BASE_OUTPUT_DIR, "train"),
    'val': os.path.join(BASE_OUTPUT_DIR, "val"),
    'test': os.path.join(BASE_OUTPUT_DIR, "test")
}
DISTANCE_THRESHOLD_METERS = 60
EARTH_RADIUS_METERS = 6371000

for split_dir in OUTPUT_DIRS.values():
    os.makedirs(split_dir, exist_ok=True)

manual_labels_basenames = set()
if CONSIDER_MANUAL:
    print("Step 0: Reading manual labels...")
    if os.path.exists(MANUAL_LABELS_DIR):
        manual_files = [f for f in os.listdir(MANUAL_LABELS_DIR) if f.endswith('.txt')]
        for f in manual_files:
            manual_labels_basenames.add(f.rsplit('.', 1)[0])
        print(f"Found {len(manual_labels_basenames)} basenames in {MANUAL_LABELS_DIR} to be forced into the test set.")
    else:
        print(f"Warning: Manual labels directory not found at {MANUAL_LABELS_DIR}. No files will be forced to test.")

print("\nStep 1: Collecting pano data and coordinates...")
pano_metadata_list = []

all_files_in_dataset_dir = os.listdir(DATASET_DIR)

base_names_from_jpgs = sorted(set(f.rsplit('.', 1)[0] for f in all_files_in_dataset_dir if f.endswith('.jpg')))
print(f"Found {len(base_names_from_jpgs)} unique .jpg base names.")

json_filenames_set = set(f for f in all_files_in_dataset_dir if f.endswith('.json'))

for base_name in tqdm(base_names_from_jpgs, desc="Loading JSON metadata"):
    json_filename = base_name + ".json"
    if json_filename not in json_filenames_set:
        print(f"Warning: JSON file {json_filename} not found for {base_name}.jpg. Skipping this pano.")
        continue

    json_path = os.path.join(DATASET_DIR, json_filename)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        coords = data.get("pano_coord")
        if isinstance(coords, list) and len(coords) == 2:
            lat, lon = float(coords[0]), float(coords[1])
            pano_metadata_list.append({
                'id': base_name,
                'coord_deg': (lat, lon),
                'coord_rad': (radians(lat), radians(lon))
            })
        else:
            print(f"Warning: 'pano_coord' key missing, not a list, or not length 2 in {json_filename}. Skipping.")
            continue
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_path}. Skipping.")
    except (TypeError, ValueError) as e:
        print(f"Warning: Invalid coordinate data in {json_path} (Lat/Lon: {coords}): {e}. Skipping.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred processing {json_path}: {e}. Skipping.")

if not pano_metadata_list:
    print("No valid pano metadata (with coordinates) loaded. Exiting.")
    exit()

print(f"Successfully loaded coordinate data for {len(pano_metadata_list)} panos.")

print("\nStep 2: Building BallTree for efficient spatial queries...")
coords_rad_array = np.array([p['coord_rad'] for p in pano_metadata_list])
ball_tree = BallTree(coords_rad_array, metric='haversine')
radius_for_query_rad = DISTANCE_THRESHOLD_METERS / EARTH_RADIUS_METERS

print("\nStep 3: Finding connected components (groups of nearby panos)...")
num_panos_with_coords = len(pano_metadata_list)
visited_indices = [False] * num_panos_with_coords
pano_groups = []

for i in tqdm(range(num_panos_with_coords), desc="Grouping panos"):
    if visited_indices[i]:
        continue

    current_group_pano_ids = []
    queue = deque()

    queue.append(i)
    visited_indices[i] = True

    while queue:
        current_pano_idx = queue.popleft()
        current_group_pano_ids.append(pano_metadata_list[current_pano_idx]['id'])

        indices_within_radius = ball_tree.query_radius(
            coords_rad_array[current_pano_idx].reshape(1, -1),
            r=radius_for_query_rad
        )[0]

        for neighbor_idx in indices_within_radius:
            if not visited_indices[neighbor_idx]:
                visited_indices[neighbor_idx] = True
                queue.append(neighbor_idx)

    pano_groups.append(current_group_pano_ids)

print(f"Found {len(pano_groups)} groups of panos.")
total_panos_in_groups = sum(len(g) for g in pano_groups)
print(f"Total {total_panos_in_groups} panos distributed into these groups.")

if total_panos_in_groups != num_panos_with_coords:
    print(f"Warning: Mismatch in pano count! Pano metadata: {num_panos_with_coords}, In groups: {total_panos_in_groups}")

print("\nStep 4: Shuffling and splitting groups into train/val/test...")

split_assignments = {'train': [], 'val': [], 'test': []}
current_pano_counts_in_splits = {'train': 0, 'val': 0, 'test': 0}

manual_groups = []
remaining_groups = []
if CONSIDER_MANUAL and manual_labels_basenames:
    for group in pano_groups:
        if not set(group).isdisjoint(manual_labels_basenames):
            manual_groups.append(group)
        else:
            remaining_groups.append(group)
    print(f"Identified {len(manual_groups)} groups containing manually labeled panos.")
else:
    remaining_groups = pano_groups

if manual_groups:
    print("Forcing manual groups into the test split...")
    for group in tqdm(manual_groups, desc="Assigning manual groups to test"):
        split_assignments['test'].extend(group)
        current_pano_counts_in_splits['test'] += len(group)

random.shuffle(remaining_groups)

train_target_pano_count = int(0.7 * total_panos_in_groups)
val_target_pano_count = int(0.2 * total_panos_in_groups)

for group in tqdm(remaining_groups, desc="Assigning remaining groups to splits"):
    group_pano_count = len(group)

    if current_pano_counts_in_splits['train'] < train_target_pano_count:
        split_assignments['train'].extend(group)
        current_pano_counts_in_splits['train'] += group_pano_count
    elif current_pano_counts_in_splits['val'] < val_target_pano_count:
        split_assignments['val'].extend(group)
        current_pano_counts_in_splits['val'] += group_pano_count
    else:
        split_assignments['test'].extend(group)
        current_pano_counts_in_splits['test'] += group_pano_count


print("\nSplit distribution (number of panos):")
for split_name, pano_ids_in_split in split_assignments.items():
    percentage = (len(pano_ids_in_split) / total_panos_in_groups * 100) if total_panos_in_groups > 0 else 0
    print(f"  {split_name}: {len(pano_ids_in_split)} panos ({percentage:.2f}%)")

print("\nStep 5: Moving files with progress bar...")
for split_name, pano_ids_in_split in split_assignments.items():
    output_split_dir = OUTPUT_DIRS[split_name]
    for pano_id in tqdm(pano_ids_in_split, desc=f"Moving {split_name}", unit="pano_pair"):
        for ext in ('.jpg', '.json'):
            src_filename = pano_id + ext
            src_path = os.path.join(DATASET_DIR, src_filename)
            dst_path = os.path.join(output_split_dir, src_filename)

            if os.path.exists(src_path):
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.move(src_path, dst_path)
            else:
                print(f"Warning: Source file {src_path} not found during move operation for pano {pano_id}. It might have been skipped earlier or already moved.")

print("\nDataset splitting complete.")
print(f"Files moved to subdirectories within: {BASE_OUTPUT_DIR}")
print("Please check the new directories to ensure files are moved as expected.")
print("The original DATASET_DIR will now contain only panos that couldn't be processed (e.g., missing JSON, bad coords) or were not part of the initial .jpg scan.")