"""Ingest pipeline: read raw Excel files from Data/ and produce a combined parquet."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

DATA_DIR = Path("Data")
OUTPUT_PATH = Path("Output/ingest/combined.parquet")


def load_excel_files(data_dir: Path) -> pl.DataFrame:
    excel_files = [
        path
        for path in sorted(data_dir.glob("*.xlsx"))
        if not path.name.startswith("~$")
    ]
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    print(f"[ingest] Found {len(excel_files)} Excel files in {data_dir}")

    frames: list[pl.DataFrame] = []
    total_start = time.perf_counter()
    for i, file_path in enumerate(excel_files, 1):
        t0 = time.perf_counter()
        frame = pl.read_excel(file_path)
        elapsed = time.perf_counter() - t0
        frame = frame.with_columns(
            pl.lit(file_path.name).alias("source_file"),
            pl.lit(file_path.stem).alias("source_period"),
        )
        frames.append(frame)
        print(f"[ingest] ({i}/{len(excel_files)}) {file_path.name}: {frame.height:,} rows in {elapsed:.1f}s")

    print(f"[ingest] All files read in {time.perf_counter() - total_start:.1f}s, concatenating...")
    combined = pl.concat(frames, how="diagonal_relaxed")
    print(f"[ingest] Combined: {combined.height:,} rows x {combined.width} cols")
    return combined


def ingest(data_dir: Path = DATA_DIR, output: Path = OUTPUT_PATH) -> pl.DataFrame:
    print(f"[ingest] Starting ingest from {data_dir} -> {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Output directory ready: {output.parent}")

    combined = load_excel_files(data_dir)

    t0 = time.perf_counter()
    combined.write_parquet(output)
    print(f"[ingest] Wrote parquet in {time.perf_counter() - t0:.1f}s ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    return combined


if __name__ == "__main__":
    ingest()
