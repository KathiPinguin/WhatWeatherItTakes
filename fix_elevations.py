"""Fetch missing elevations using the srtm Python package (local SRTM tile lookup)."""

from __future__ import annotations

import csv
from pathlib import Path

import srtm

CACHE_FILE = Path("Data/stations/ds100_stations.csv")


def main():
    # Read existing data
    with open(CACHE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        stations = list(reader)

    print(f"Total stations: {len(stations)}")
    
    # Find stations missing elevation
    missing = [s for s in stations if not s.get("elevation_m") or s["elevation_m"] == ""]
    print(f"Missing elevation: {len(missing)}")
    
    if not missing:
        print("All elevations present!")
        return

    # Use srtm package for local elevation lookup
    elevation_data = srtm.get_data()
    fetched = 0
    failed = 0
    
    for s in missing:
        try:
            lat = float(s["lat"])
            lon = float(s["lon"])
            elev = elevation_data.get_elevation(lat, lon)
            if elev is not None:
                s["elevation_m"] = str(elev)
                fetched += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    
    print(f"Fetched: {fetched}, Failed: {failed}")
    
    # Write back
    with open(CACHE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds100", "name", "lat", "lon", "elevation_m"])
        writer.writeheader()
        writer.writerows(stations)
    
    print(f"Saved to {CACHE_FILE}")
    total_with_elev = sum(1 for s in stations if s.get("elevation_m") and s["elevation_m"] != "")
    print(f"Total with elevation: {total_with_elev}/{len(stations)}")


if __name__ == "__main__":
    main()
