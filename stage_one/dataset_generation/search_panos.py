import json
import re
from typing import List, Optional, Dict, Tuple
import requests
from functools import lru_cache
from pydantic import BaseModel
from requests.models import Response


class Panorama(BaseModel):
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float]
    roll: Optional[float]
    date: Optional[str]
    elevation: Optional[float]


def make_search_url(lat: float, lon: float) -> str:
    url = (
        "https://maps.googleapis.com/maps/api/js/"
        "GeoPhotoService.SingleImageSearch"
        "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(lat, lon)


@lru_cache(maxsize=None)  
def search_request(lat: float, lon: float) -> Response:
    url = make_search_url(lat, lon)
    return requests.get(url)


@lru_cache(maxsize=None)
def get_date_of_panorama(pano_id: str) -> str:
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json+protobuf",
    }

    data = [
        ["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
        ["en", "US"],
        [[[2, pano_id]]],
        [[1, 2, 3, 4, 8, 6]]
    ]

    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    response = requests.post(url, headers=headers, json=data)

    year = response.json()[1][0][6][7][0]
    month = response.json()[1][0][6][7][1]

    return f"{year}-{month}"

@lru_cache(maxsize=None)
def get_pano_heading(pano_id: str) -> str:
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json+protobuf",
    }

    data = [
        ["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
        ["en", "US"],
        [[[2, pano_id]]],
        [[1, 2, 3, 4, 8, 6]]
    ]

    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    response = requests.post(url, headers=headers, json=data)

    return float(response.json()[1][0][5][0][1][2][0])

def extract_panoramas(text: str) -> List[Panorama]:
    blob = re.findall(r"callbackfunc\( (.*) \)$", text)[0]
    data = json.loads(blob)

    if data == [[5, "generic", "Search returned no images."]]:
        return []

    subset = data[1][5][0]
    raw_panos = subset[3][0][::-1]

    raw_dates = subset[8] if len(subset) >= 9 and subset[8] else []

    return [
        Panorama(
            pano_id=pano[0][1],
            lat=pano[2][0][2],
            lon=pano[2][0][3],
            heading=pano[2][2][0],
            pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None,
            roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None,
            date=get_date_of_panorama(pano[0][1]),
            elevation=pano[3][0] if len(pano) >= 4 else None,
        )
        for pano in raw_panos
    ]


def search_panoramas(lat: float, lon: float) -> List[Panorama]:
    resp = search_request(lat, lon)
    return extract_panoramas(resp.text)


def parse_url(url: str) -> Tuple[str, str, str]:
    matches = re.search(
        r"\/@([-+]?[0-9]+[.]?[0-9]*),([-+]?[0-9]+[.]?[0-9]*).+!1s(.+)!2e", url
    )
    return matches.groups() if matches else ("", "", "")


def search_panoramas_url(url: str) -> List[Panorama]:
    lat, lon, _ = parse_url(url)
    return search_panoramas(float(lat), float(lon))


def search_panoramas_url_exact(url: str) -> Optional[Panorama]:
    _, _, pano_id = parse_url(url)
    panos = search_panoramas_url(url)
    return next((p for p in panos if p.pano_id == pano_id), None)
