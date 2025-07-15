import json
import csv
import datetime
import re
import random

def convert_date(value):
    if value is None or value == "":
        return "2000-01-01"
    if isinstance(value, str):
        if value.lower() == "none" or value.strip() == "":
            return "2000-01-01"
    if isinstance(value, (int, float)):
        try:
            
            dt = datetime.datetime.utcfromtimestamp(value / 1000)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "2000-01-01"
    elif isinstance(value, str):
        try:
            dt = datetime.datetime.strptime(value, "%m/%d/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return "2000-01-01"
    else:
        return "2000-01-01"

def parse_geojson(file_path, date_field):
    results = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        raw_date = props.get(date_field)
        date_str = convert_date(raw_date)
        
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            results.append({
                "latitude": lat,
                "longitude": lon,
                "date": date_str
            })
    return results

def parse_csv(file_path, date_field):
    results = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_date = row.get(date_field, "")
            date_str = convert_date(raw_date)
            
            geom = row.get("the_geom", "")
            
            match = re.search(r'POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)', geom)
            if match:
                lon = float(match.group(1))
                lat = float(match.group(2))
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": date_str
                })
    return results

def main():
    all_data = []
    
    
    all_data.extend(parse_geojson("location_data/bend.geojson", "InstallDate"))
    
    
    all_data.extend(parse_geojson("location_data/portland.geojson", "InstallDate"))
    
    
    all_data.extend(parse_csv("location_data/nyc.csv", "GeoCyclora"))
    
    
    random.shuffle(all_data)
    
    
    with open("all_locations.csv", "w", newline="") as csvfile:
        fieldnames = ["latitude", "longitude", "date"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_data:
            writer.writerow(entry)

if __name__ == "__main__":
    main()
