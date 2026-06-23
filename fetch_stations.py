"""Fetch DS100 station coordinates from OpenStreetMap and elevation from Open-Meteo."""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("Data/stations")
CACHE_FILE = OUTPUT_DIR / "ds100_stations.csv"


def fetch_osm_stations() -> list[dict]:
    """Query Overpass API for all German railway stations/halts with DS100 codes."""
    query = """
    [out:json][timeout:120];
    area["ISO3166-1"="DE"]->.a;
    (
      node["railway"="station"]["railway:ref"](area.a);
      node["railway"="halt"]["railway:ref"](area.a);
    );
    out;
    """
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "Hack4Rail/1.0 (educational hackathon project)"},
    )
    print("Querying Overpass API for German railway stations...")
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read().decode())

    stations = []
    for el in result.get("elements", []):
        ref = el.get("tags", {}).get("railway:ref", "")
        name = el.get("tags", {}).get("name", "")
        lat = el.get("lat")
        lon = el.get("lon")
        if ref and lat and lon:
            stations.append({"ds100": ref, "name": name, "lat": lat, "lon": lon})

    print(f"  Found {len(stations)} stations with DS100 codes")
    return stations


def fetch_elevations(stations: list[dict], batch_size: int = 100) -> list[dict]:
    """Fetch elevation for stations using Open-Meteo Elevation API (free, no key needed).
    
    API docs: https://open-meteo.com/en/docs/elevation-api
    Supports up to 100 coordinates per request.
    """
    print(f"Fetching elevation for {len(stations)} stations (batch size {batch_size})...")
    
    for i in range(0, len(stations), batch_size):
        batch = stations[i : i + batch_size]
        lats = ",".join(str(s["lat"]) for s in batch)
        lons = ",".join(str(s["lon"]) for s in batch)
        
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
        req = urllib.request.Request(url, headers={"User-Agent": "Hack4Rail/1.0"})
        
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode())
            elevations = data.get("elevation", [])
            
            for j, elev in enumerate(elevations):
                stations[i + j]["elevation_m"] = elev
        except Exception as e:
            print(f"  Warning: batch {i//batch_size + 1} failed: {e}")
            for j in range(len(batch)):
                stations[i + j]["elevation_m"] = None
        
        # Be polite: ~200ms between requests
        if i + batch_size < len(stations):
            time.sleep(0.2)
        
        done = min(i + batch_size, len(stations))
        if done % 500 == 0 or done == len(stations):
            print(f"  {done}/{len(stations)} elevations fetched")
    
    return stations


def save_cache(stations: list[dict]) -> None:
    """Save station data to CSV cache."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CACHE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds100", "name", "lat", "lon", "elevation_m"])
        writer.writeheader()
        writer.writerows(stations)
    
    print(f"Saved {len(stations)} stations to {CACHE_FILE}")


def load_cache() -> list[dict] | None:
    """Load cached station data if available."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    return None


def main():
    # Check if we already have cached data
    cached = load_cache()
    if cached:
        print(f"Cache exists with {len(cached)} stations at {CACHE_FILE}")
        print("Delete the file to re-fetch. Skipping.")
        return

    # Step 1: Get coordinates from OSM
    stations = fetch_osm_stations()
    
    if not stations:
        print("ERROR: No stations returned from Overpass API")
        return
    
    # Step 2: Get elevation
    stations = fetch_elevations(stations)
    
    # Step 3: Cache
    save_cache(stations)
    
    # Stats
    with_elev = sum(1 for s in stations if s.get("elevation_m") is not None)
    print(f"\nDone! {with_elev}/{len(stations)} stations have elevation data.")


if __name__ == "__main__":
    main()
