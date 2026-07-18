"""
Raw data acquisition — NASA POWER Daily Point API.
Master §11.1 / Phase 0 §3.

One request per district (district = one lat/lon point).
Idempotent: skips districts already present in data/raw/ with a manifest
entry, unless force=True. Resumable: safe to re-run after partial failure.

-999 fill value is mapped to NaN HERE, at the raw layer, per Phase 0 §4 —
this must not be deferred to clean_and_engineer.py.
"""

import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import yaml

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
CONFIG_PATH = "configs/data.yaml"
RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

REQUEST_DELAY_SECONDS = 1.0  # polite fixed delay between requests; no fixed
                              # NASA rate limit exists, but this avoids
                              # hammering the same-relative-location pattern
                              # their docs warn can trigger a block.


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def fetch_district(district: dict, date_range: dict, nasa_cfg: dict) -> pd.DataFrame:
    """Fetch one district's daily time series, with retry + backoff."""
    params = {
        "parameters": ",".join(nasa_cfg["parameters"]),
        "community": nasa_cfg["community"],
        "longitude": district["lon"],
        "latitude": district["lat"],
        "start": date_range["start"].replace("-", ""),
        "end": date_range["end"].replace("-", ""),
        "format": "JSON",
    }

    max_retries = nasa_cfg["max_retries"]
    backoff_base = nasa_cfg["backoff_base_seconds"]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            return parse_response(payload, district, nasa_cfg)
        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = e
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"  [{district['district_id']}] attempt {attempt}/{max_retries} "
                  f"failed ({e}); retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch {district['district_id']} after {max_retries} attempts: {last_error}"
    )


def parse_response(payload: dict, district: dict, nasa_cfg: dict) -> pd.DataFrame:
    """Convert NASA POWER JSON response into a per-day dataframe.
    Maps the documented fill value (-999) to NaN here, at raw ingestion —
    per Phase 0 §4, this is the single most important transformation at
    this layer and must not be silently deferred.
    """
    param_data = payload["properties"]["parameter"]
    fill_value = nasa_cfg["fill_value"]

    # All parameters share the same date keys; use the first parameter's
    # keys as the canonical date index.
    first_param = next(iter(param_data))
    dates = list(param_data[first_param].keys())

    rows = []
    for date_str in dates:
        row = {
            "district_id": district["district_id"],
            "date": datetime.strptime(date_str, "%Y%m%d").date().isoformat(),
            "lat": district["lat"],
            "lon": district["lon"],
        }
        for param_name, values_by_date in param_data.items():
            val = values_by_date.get(date_str)
            row[param_name] = None if val == fill_value else val
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def already_fetched(district_id: str, manifest: dict) -> bool:
    entry = manifest.get(district_id)
    out_path = RAW_DIR / f"{district_id}.parquet"
    return entry is not None and out_path.exists()


def run(force: bool = False) -> None:
    config = load_config()
    manifest = load_manifest()
    date_range = config["date_range"]
    nasa_cfg = config["nasa_power"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    districts = config["districts"]
    print(f"Fetching {len(districts)} districts, date range "
          f"{date_range['start']} to {date_range['end']}")

    for i, district in enumerate(districts, 1):
        did = district["district_id"]

        if not force and already_fetched(did, manifest):
            print(f"[{i}/{len(districts)}] {did}: already fetched, skipping")
            continue

        print(f"[{i}/{len(districts)}] {did}: fetching...")
        df = fetch_district(district, date_range, nasa_cfg)

        out_path = RAW_DIR / f"{did}.parquet"
        df.to_parquet(out_path, index=False)

        checksum = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

        manifest[did] = {
            "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "date_range_returned": [df["date"].min(), df["date"].max()],
            "row_count": len(df),
            "checksum_sha256": checksum,
        }
        save_manifest(manifest)  # save after each district, not just at the end

        print(f"  -> {len(df)} rows written to {out_path}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nDone. Manifest summary:")
    for did, entry in manifest.items():
        print(f"  {did}: {entry['row_count']} rows, "
              f"{entry['date_range_returned'][0]} to {entry['date_range_returned'][1]}")


if __name__ == "__main__":
    run(force=False)