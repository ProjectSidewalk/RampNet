import json
import random
import pandas as pd
from shapely.geometry import shape, Point, LineString, MultiLineString, GeometryCollection
from scipy.spatial import cKDTree
from pyproj import Transformer
from search_panos import search_panoramas
from tqdm import tqdm
import concurrent.futures
import bisect
import os
import threading

MINIMUM_DISTANCE_TO_CURB_RAMP = 60
cities = ["New York", "Bend", "Portland"]
TOTAL_POINTS = 95000

def load_city_boundaries(geojson_path="cityboundaries.geojson"):
    with open(geojson_path, "r") as f:
        data = json.load(f)
    city_geometries = {}
    for feature in data["features"]:
        city_name = feature["properties"]["NAME"]
        geom = shape(feature["geometry"])
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)
        city_geometries[city_name] = geom
    return city_geometries

def load_city_streets(city_boundaries):
    city_street_data = {}
    for city in cities:
        file_path = f"street_data/{city} - Streets.geojson"
        with open(file_path, "r") as f:
            data = json.load(f)
        lines = []
        boundary = city_boundaries.get(city)
        for feature in data["features"]:
            properties = feature["properties"]
            if "FULL_NAME" in properties:
                if properties["FULL_NAME"] == "":
                    continue
            if "FULLNAME" in properties:
                if properties["FULLNAME"] == "":
                    continue
            if "Street" in properties:
                if properties["Street"] == "":
                    continue
            
            geom = shape(feature["geometry"])
            if boundary:
                geom = geom.intersection(boundary)
            if geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                if geom.length > 0:
                    lines.append(geom)
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    if part.length > 0:
                        lines.append(part)
            elif geom.geom_type == "GeometryCollection":
                for part in geom.geoms:
                    if part.geom_type == "LineString" and part.length > 0:
                        lines.append(part)
        if not lines:
            raise ValueError(f"No street lines found for {city}.")
        cumulative_lengths = []
        total_length = 0.0
        for line in lines:
            total_length += line.length
            cumulative_lengths.append(total_length)
        city_street_data[city] = {
            "lines": lines,
            "cumulative_lengths": cumulative_lengths,
            "total_length": total_length
        }
    return city_street_data

def load_curb_ramps(csv_path="all_locations.csv"):
    df = pd.read_csv(csv_path)
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("CSV file must contain 'latitude' and 'longitude' columns.")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(df["longitude"].values, df["latitude"].values)
    points = list(zip(xs, ys))
    return points, transformer

def get_random_point_on_street(street_info):
    lines = street_info["lines"]
    cumulative_lengths = street_info["cumulative_lengths"]
    total_length = street_info["total_length"]
    random_distance = random.uniform(0, total_length)
    index = bisect.bisect_left(cumulative_lengths, random_distance)
    prev_cum = 0.0 if index == 0 else cumulative_lengths[index - 1]
    distance_along_line = random_distance - prev_cum
    return lines[index].interpolate(distance_along_line)

def load_seen_panos(filepath):
    seen = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        seen.add(data["pano_id"])
                    except Exception:
                        pass
    return seen

def generate_valid_point(city_street_data, curb_data, seen_panos, seen_lock):
    curb_points, transformer = curb_data
    kd_tree = cKDTree(curb_points)
    while True:
        city = random.choice(list(city_street_data.keys()))
        street_info = city_street_data[city]
        candidate_point = get_random_point_on_street(street_info)
        candidate_lon, candidate_lat = candidate_point.x, candidate_point.y
        panos = search_panoramas(candidate_lat, candidate_lon)
        if not panos:
            continue
        for pano in panos:
            with seen_lock:
                if pano.pano_id in seen_panos:
                    continue
            pano_x, pano_y = transformer.transform(pano.lon, pano.lat)
            distance, _ = kd_tree.query([pano_x, pano_y])
            if distance >= MINIMUM_DISTANCE_TO_CURB_RAMP:
                with seen_lock:
                    if pano.pano_id in seen_panos:
                        continue
                    seen_panos.add(pano.pano_id)
                return {
                    "pano_id": pano.pano_id,
                    "pano_coords": [pano.lat, pano.lon],
                    "curb_ramps_coords": []
                }

def main():
    output_file = "negativepanos.jsonl"
    city_boundaries = load_city_boundaries()
    city_street_data = load_city_streets(city_boundaries)
    curb_data = load_curb_ramps()
    seen_panos = load_seen_panos(output_file)
    seen_lock = threading.Lock()
    existing_count = len(seen_panos)
    remaining_points = TOTAL_POINTS - existing_count
    if remaining_points <= 0:
        print("Already reached target number of points.")
        return
    with open(output_file, "a") as outfile, concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_valid_point, city_street_data, curb_data, seen_panos, seen_lock)
            for _ in range(remaining_points)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=remaining_points):
            try:
                result = future.result()
                outfile.write(json.dumps(result) + "\n")
                outfile.flush()
            except Exception:
                pass

if __name__ == "__main__":
    main()
