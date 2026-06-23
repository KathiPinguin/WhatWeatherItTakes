"""Filtering pipeline: apply filters to ingested parquet data."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

INPUT_PATH = Path("Output/ingest/combined.parquet")
OUTPUT_PATH = Path("Output/filtered/filtered.parquet")


def filter_data(df: pl.DataFrame) -> pl.DataFrame:
    before = df.height
    print(f"[filter] Starting with {before:,} rows")

    # Normalize column names: strip whitespace, collapse multiple spaces
    import re
    df = df.rename({c: re.sub(r"\s+", " ", c.strip()) for c in df.columns})

    # Cast numeric columns to Float64
    # Replace comma decimal separators (German format) with dots before casting
    all_cast_cols = [
        "Energiebezug [kWh]",
        "Rückspeisung [kWh]",
        "Ersatzwert [kWh]",
        "Energieverbrauch [kWh]",
        "sEV (Monat)",
        "Energiebzug Spitz [kWh]",
        "Rückspeisung Spitz [kWh]",
        "Ersatzwert Spitz [kWh]",
        "Energieverbrauch Spitz [kWh]",
        "sEV (Spitz)",
        "Zugeinheiten Km",
        "TFZ-km",
        "TLKM",
        "Temperatur Ab [°C]",
        "Temperatur An [°C]",
        "Schneehöhe Ab [cm]",
        "Schneehöhe An [cm]",
        "Niederschlagsmenge Ab [l/m²]",
        "Niederschlagsmenge An [l/m²]",
        "Neuschneemenge Ab [cm/m²]",
        "Neuschneemenge An [cm/m²]",
        "Luftfeuchtigkeit Ab [%]",
        "Luftfeuchtigkeit An [%]",
        "Wind Mittel Ab [m/s]",
        "Wind Mittel An [m/s]",
        "Windspitze Ab [m/s]",
        "Windspitze An [m/s]",
    ]
    existing_cast = [c for c in all_cast_cols if c in df.columns]

    df = df.with_columns([
        pl.when(pl.col(c).cast(pl.String, strict=False).str.contains(","))
        .then(pl.col(c).cast(pl.String, strict=False).str.replace(",", "."))
        .otherwise(pl.col(c))
        .cast(pl.Float64, strict=False)
        .alias(c)
        for c in existing_cast
    ])

    # Cast date columns
    date_cols = [
        "Abfahrtsdatum",
        "LB-Ab-Datum (Plan)",
        "LB-An-Datum (Plan)",
    ]
    existing_dates = [c for c in date_cols if c in df.columns]
    # Cast date columns (German format dd.mm.yyyy)
    date_cols = [
        "Abfahrtsdatum",
        "LB-Ab-Datum (Plan)",
        "LB-An-Datum (Plan)",
    ]
    existing_dates = [c for c in date_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).str.to_date("%d.%m.%Y", strict=False).alias(c)
        for c in existing_dates
    ])

    # Cast time columns (HH:MM)
    time_cols = ["Ab-Zeit (Plan)", "An-Zeit (Plan)"]
    existing_times = [c for c in time_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).str.to_time("%H:%M", strict=False).alias(c)
        for c in existing_times
    ])

    # Cast timestamp strings (yyyy-mm-dd HH:MM:SS.sss)
    timestamp_str_cols = ["Zeitstempel_ab_ist", "Zeitstempel_an_ist"]
    existing_ts_str = [c for c in timestamp_str_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).str.to_datetime("%Y-%m-%d %H:%M:%S%.3f", strict=False).alias(c)
        for c in existing_ts_str
    ])

    # Cast duration/relative columns to Int64 (they are minutes/offsets, not timestamps)
    int_cols = ["Dauer_plan", "Dauer_ist", "Abfahrtzeitrel", "Ankunftzeitrel"]
    existing_ints = [c for c in int_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.Int64, strict=False).alias(c)
        for c in existing_ints
    ])

    # Zeitstempel_ab_plan, Zeitstempel_an_plan, Zeitstempel_ab_kombiniert,
    # Zeitstempel_an_kombiniert, Dauer_kombiniert are already correct types from ingest

    df = df.filter(
        pl.col("Energiebezug [kWh]").is_not_null()
        & pl.col("Rückspeisung [kWh]").is_not_null()
    )
    print(f"[filter] After energy not-null: {df.height:,}")

    df = df.filter(
        ~(
            (pl.col("Energiebezug [kWh]") == 0)
            & (pl.col("Rückspeisung [kWh]") == 0)
        )
    )
    print(f"[filter] After energy not both 0: {df.height:,}")

    df = df.filter(
        pl.col("Temperatur Ab [°C]").is_not_null()
        & pl.col("Temperatur An [°C]").is_not_null()
    )
    print(f"[filter] After temperature not-null: {df.height:,}")

    df = df.filter(
        pl.col("Leist Typ") != "S"
    )
    print(f"[filter] After Leist Typ != S: {df.height:,}")

    df = df.filter(pl.col("TFZ-km") > 5)
    print(f"[filter] After TFZ-km > 5: {df.height:,}")

    df = df.filter(
        pl.col("sEV (Spitz)").is_not_null() | pl.col("sEV (Monat)").is_not_null()
    )
    print(f"[filter] After sEV not-null: {df.height:,}")

    # Create sEV: prefer sEV (Spitz), fall back to sEV (Monat)
    df = df.with_columns(
        pl.coalesce("sEV (Spitz)", "sEV (Monat)").alias("sEV")
    )

    # Only keep sEV between 6 and 50
    df = df.filter((pl.col("sEV") >= 6) & (pl.col("sEV") <= 50))
    print(f"[filter] After sEV in [6, 50]: {df.height:,}")

    # Derive delay_diff_min = arrival delay - departure delay
    if "Abfahrtzeitrel" in df.columns and "Ankunftzeitrel" in df.columns:
        df = df.with_columns(
            (pl.col("Ankunftzeitrel") - pl.col("Abfahrtzeitrel")).alias("delay_diff_min")
        )

    # Extract year, month, day from Abfahrtsdatum
    if "Abfahrtsdatum" in df.columns:
        df = df.with_columns(
            pl.col("Abfahrtsdatum").dt.year().alias("year"),
            pl.col("Abfahrtsdatum").dt.month().alias("month"),
            pl.col("Abfahrtsdatum").dt.day().alias("day"),
        )

    # Join elevation from station cache (DS100 → elevation_m)
    station_cache = Path("Data/stations/ds100_stations.csv")
    if station_cache.exists() and "Ab-Ort" in df.columns and "An-Ort" in df.columns:
        stations = pl.read_csv(station_cache).select("ds100", "elevation_m")
        stations = stations.filter(pl.col("elevation_m").is_not_null())
        stations = stations.unique(subset=["ds100"], keep="first")

        # Create base code column (part before space) for fallback matching
        df = df.with_columns(
            pl.col("Ab-Ort").str.split(" ").list.first().alias("_ab_base"),
            pl.col("An-Ort").str.split(" ").list.first().alias("_an_base"),
        )

        # Join departure elevation (try exact match first, fallback to base)
        ab_elev = stations.rename({"ds100": "_ab_base", "elevation_m": "elevation_ab_m"})
        df = df.join(ab_elev, on="_ab_base", how="left")

        # Join arrival elevation
        an_elev = stations.rename({"ds100": "_an_base", "elevation_m": "elevation_an_m"})
        df = df.join(an_elev, on="_an_base", how="left")

        # Compute elevation difference (arrival - departure)
        df = df.with_columns(
            (pl.col("elevation_an_m") - pl.col("elevation_ab_m")).alias("elevation_diff_m")
        )

        # Drop helper columns
        df = df.drop("_ab_base", "_an_base")

        matched = df.filter(pl.col("elevation_diff_m").is_not_null()).height
        print(f"[filter] Elevation joined: {matched:,}/{df.height:,} rows ({matched/df.height*100:.1f}%)")

    # Normalize whitespace in string columns (strip + collapse multiple spaces)
    str_normalize_cols = ["Ab-Ort", "An-Ort", "Baureihe Zug", "Leist Typ"]
    existing_str = [c for c in str_normalize_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).str.strip_chars().str.replace_all(r"\s+", " ").alias(c)
        for c in existing_str
    ])

    after = df.height
    print(f"[filter] Final: {before:,} → {after:,} rows (removed {before - after:,})")
    return df


def process(input_path: Path = INPUT_PATH, output: Path = OUTPUT_PATH, sample: int | None = None) -> pl.DataFrame:
    df = pl.read_parquet(input_path)
    df = filter_data(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    print(f"Saved filtered data to: {output}")

    if sample:
        n = min(sample, df.height)
        sample_path = output.with_stem(f"{output.stem}_sample{n}")
        df.sample(n).write_csv(sample_path.with_suffix(".csv"))
        print(f"Saved sample ({n:,} rows) to: {sample_path.with_suffix('.csv')}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ingested data")
    parser.add_argument("--sample", type=int, default=None, help="Output a CSV sample of N rows (e.g. 10000)")
    args = parser.parse_args()
    process(sample=args.sample)
