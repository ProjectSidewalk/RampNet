"""Could the CurbRamp validator have seen any benchmark panorama in training? (#258)

#258 proposes the production CurbRamp validator (sidewalk-validator-ai, HF
``projectsidewalk/sidewalk-validator-ai-curbramp``) as a second-pass judge on RampNet's
candidates. If a benchmark panorama was in the validator's training data, its scores there
are contaminated. This script bounds that risk.

**What the training set is, and is not, known to be.** The validator's training images
were never published: sidewalk-validator-ai's 2026-10-14 audit, finding 13, shows the HF
CurbRamp dataset repo is a copy of the Crosswalk one (sidewalk-validator-ai#7). What is
known is the selection code. At sidewalk-validator-ai commit ``eb9bc65`` (the code before
the audit), ``data_generation.py`` pulls ``rawLabels`` from the 19 deployments in
``TRAINING_CITIES`` below and keeps labels by their agree/disagree counts. That the
2025-08-16 model run used exactly this city list is not recorded anywhere; it is the
best available reading.

**The check, conservative by construction.** For every training city this fetches the
current public ``rawLabels`` CSV restricted to ``labelType=CurbRamp`` and treats *every*
CurbRamp label as possibly trained on, which is a superset of any agree/disagree
selection. Per benchmark split it reports:

* exact ``pano_id`` overlap with any training-city CurbRamp label;
* the distance from each benchmark pano to the nearest training-city CurbRamp label, as
  counts within 50 m and 500 m and the minimum. A nearby label means a different capture
  of the same scene could have been in training, which is weaker than pano leakage but
  worth knowing.

**Limits, stated beside the numbers.** The live API omits labels deleted since the
2025-08 data pull, so a deleted label that was trained on is invisible here. The API's
counts have moved since then (AI votes, new validations), which is why no selection rule
is applied.

    python scripts/analysis/validator_leakage_258.py             # fetch missing CSVs, write
    python scripts/analysis/validator_leakage_258.py --offline   # cached CSVs only

Network: one GET per training city, sequential, cached under ``--cache-dir``
(gitignored). The output records each CSV's sha256, so a later run on the same cache is
byte-identical and a re-fetch that drifted shows up in the diff. Writes
``analysis_out/verifier_258/leakage.json``.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("RAMPNET_ANALYSIS_OUT", os.path.join(REPO, "analysis_out"))
sys.path.insert(0, REPO)

from rampnet.eval import PINNED_SPLITS  # noqa: E402

# sidewalk-validator-ai @ eb9bc65, data_generation.py: the deployments its rawLabels pull
# reads, as the subdomain of https://sidewalk-<name>.cs.washington.edu.
VALIDATOR_COMMIT = "eb9bc65"
TRAINING_CITIES = ("cdmx", "la-piedad", "spgg", "amsterdam", "blackhawk-hills", "chicago",
                   "cliffside-park", "columbus", "knox", "mendota", "newberg", "oradell",
                   "pittsburgh", "sea", "st-louis", "teaneck", "keelung", "new-taipei",
                   "taipei")
URL = "https://sidewalk-{city}.cs.washington.edu/v3/api/rawLabels?filetype=csv&labelType=CurbRamp"
NEAR_M = (50, 500)
EARTH_R_M = 6_371_000.0
OUT_PATH = os.path.join(OUT, "verifier_258", "leakage.json")
DEFAULT_CACHE = os.path.join(OUT, "verifier_258", "rawlabels_cache")


def fetch(city, cache_dir, offline):
    """Path of the cached CSV for ``city``; download it first unless cached or offline."""
    path = os.path.join(cache_dir, f"{city}.csv")
    meta_path = path + ".meta.json"
    if os.path.exists(path) and os.path.exists(meta_path):
        return path
    if offline:
        raise SystemExit(f"{city}: no cached CSV in {cache_dir} and --offline given")
    os.makedirs(cache_dir, exist_ok=True)
    url = URL.format(city=city)
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=900) as r, open(path + ".tmp", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".tmp", path)
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        json.dump({"url": url, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seconds": round(time.time() - t0, 1)}, f, indent=1)
    print(f"  fetched {city} in {time.time() - t0:.0f} s")
    return path


def read_labels(path):
    """``(pano_ids, lat_lng array)`` for every CurbRamp row of one rawLabels CSV."""
    pano_ids, coords = set(), []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["label_type"] != "CurbRamp":
                raise ValueError(f"{path}: non-CurbRamp row; the labelType filter failed")
            pano_ids.add(row["pano_id"])
            if row["latitude"] and row["longitude"]:
                coords.append((float(row["latitude"]), float(row["longitude"])))
    return pano_ids, np.array(coords, dtype=float).reshape(-1, 2)


def benchmark_panos(split):
    """``[(pano_id, lat, lng), ...]`` from ``benchmark/<split>/records.jsonl``."""
    out = []
    with open(os.path.join(REPO, "benchmark", split, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["pano"]
            lat, lng = (p["lat"], p["lng"]) if "lat" in p else p["pano_coord"]
            out.append((p["panorama_id"], float(lat), float(lng)))
    return out


def nearest_m(lat, lng, coords):
    """Haversine distance in metres from one point to the nearest row of ``coords``."""
    if len(coords) == 0:
        return math.inf
    la1, lo1 = math.radians(lat), math.radians(lng)
    la2, lo2 = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    a = (np.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return float(2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1))).min())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--offline", action="store_true", help="use cached CSVs only")
    args = ap.parse_args(argv)

    cities, all_ids, all_coords = {}, set(), []
    for city in TRAINING_CITIES:
        path = fetch(city, args.cache_dir, args.offline)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        with open(path + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        ids, coords = read_labels(path)
        cities[city] = {"fetched_at": meta["fetched_at"], "sha256": sha,
                        "curbramp_labels": int(len(coords)), "panos": len(ids)}
        all_ids |= ids
        all_coords.append(coords)
    coords = np.vstack(all_coords)

    splits = {}
    for split in PINNED_SPLITS:
        panos = benchmark_panos(split)
        overlap = sorted(pid for pid, _, _ in panos if pid in all_ids)
        dists = [nearest_m(lat, lng, coords) for _, lat, lng in panos]
        splits[split] = {
            "n_panos": len(panos),
            "pano_id_overlap": overlap,
            **{f"within_{m}m": sum(d <= m for d in dists) for m in NEAR_M},
            "min_distance_m": round(min(dists), 1),
        }

    payload = {
        "validator": {"repo": "ProjectSidewalk/sidewalk-validator-ai",
                      "selection_code_commit": VALIDATOR_COMMIT,
                      "model": "projectsidewalk/sidewalk-validator-ai-curbramp"},
        "method": ("every current public rawLabels CurbRamp label in the validator's "
                   "training cities is treated as possibly trained on (superset of any "
                   "agree/disagree selection); exact pano_id membership plus haversine "
                   "distance to the nearest such label"),
        "limits": ["labels deleted since the 2025-08 data pull are not in the live API",
                   "the 2025-08-16 run's city list is not recorded; this is the code's list"],
        "training_cities": cities,
        "splits": splits,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for split, s in splits.items():
        print(f"{split:20s} panos {s['n_panos']:4d}  id overlap {len(s['pano_id_overlap']):3d}  "
              f"<=50m {s['within_50m']:4d}  <=500m {s['within_500m']:4d}  "
              f"min {s['min_distance_m']:.0f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
