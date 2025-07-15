from search_panos import search_panoramas
import pandas as pd
import geopy.distance
from scipy.spatial import cKDTree
from pyproj import Transformer
import json
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor

DISCOVERY_DISTANCE_THRESHOLD = 10
INCLUSION_DISTANCE_THRESHOLD = 35

curb_ramp_locations = pd.read_csv("all_locations.csv")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def project_point(lon, lat):
    return transformer.transform(lon, lat)

curb_ramp_locations[['x', 'y']] = curb_ramp_locations.apply(
    lambda row: pd.Series(project_point(row['longitude'], row['latitude'])),
    axis=1
)
tree = cKDTree(curb_ramp_locations[['x', 'y']].values)
panos_already_processed = set()
output_file = None
processed_lock = threading.Lock()
output_lock = threading.Lock()

def process_pano(pano):
    pano_x, pano_y = transformer.transform(pano.lon, pano.lat)
    indices = tree.query_ball_point([pano_x, pano_y], INCLUSION_DISTANCE_THRESHOLD)
    curb_ramps_to_include = []
    for idx in indices:
        row = curb_ramp_locations.iloc[idx]
        distance = geopy.distance.geodesic((row["latitude"], row["longitude"]), (pano.lat, pano.lon)).m
        if distance <= INCLUSION_DISTANCE_THRESHOLD:
            curb_ramps_to_include.append(row)
            curb_year, curb_month = int(row["date"].split("-")[0]), int(row["date"].split("-")[1])
            pano_year, pano_month = int(pano.date.split("-")[0]), int(pano.date.split("-")[1])
            if curb_year > pano_year:
                return
            if curb_month >= pano_month:
                return
    if curb_ramps_to_include:
        curb_coords = [[row["latitude"], row["longitude"]] for row in curb_ramps_to_include]
        data = {"pano_id": pano.pano_id, "pano_coords": [pano.lat, pano.lon], "curb_ramps_coords": curb_coords}
        with output_lock:
            output_file.write(json.dumps(data) + "\n")
            output_file.flush()

def process_row(row):
    try:
        location_lat = row["latitude"]
        location_lng = row["longitude"]
        panos = search_panoramas(location_lat, location_lng)
        for pano in panos:
            with processed_lock:
                if pano.pano_id in panos_already_processed:
                    continue
            distance = geopy.distance.geodesic((location_lat, location_lng), (pano.lat, pano.lon)).m
            if distance <= DISCOVERY_DISTANCE_THRESHOLD:
                process_pano(pano)
            with processed_lock:
                panos_already_processed.add(pano.pano_id)
    except Exception as e:
        try:
            print(f"Error processing - {pano.pano_id}")
        except Exception:
            print("Error processing row")

def main():
    global output_file
    output_file = open("dataset.jsonl", "w")
    rows = [row for _, row in curb_ramp_locations.iterrows()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_row, rows), total=len(rows), desc="Processing curb ramps"))
    output_file.close()

if __name__ == '__main__':
    main()
