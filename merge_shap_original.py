"""Merge SHAP weather attribution back onto original data with human-readable columns."""

from pathlib import Path
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis_common import load_data, prepare_Xy, split, TARGET

OUTPUT = Path("Output/analysis/shap/weather_attribution_full.csv")

# --- Reproduce exact same row selection ---
df, num_feats, cat_feats = load_data()
all_feats = num_feats + cat_feats
X, y, cmap = prepare_Xy(df, all_feats, cat_feats)
X_train, X_test, y_train, y_test = split(X, y)

sample_n = min(50_000, len(X_test))
X_sample = X_test.sample(n=sample_n, random_state=42)

# These are the original integer indices into the polars→pandas dataframe
sample_indices = X_sample.index

# --- Load original parquet with ALL columns ---
df_full = pl.read_parquet("Output/filtered/filtered.parquet").to_pandas()

# Pick the same rows from the original data
original_rows = df_full.iloc[sample_indices].reset_index(drop=True)

# --- Load SHAP results ---
shap_results = pd.read_csv("Output/analysis/shap/weather_attribution.csv")

# --- Select key original columns to prepend ---
id_cols = [
    "Ab-Ort", "An-Ort", "Abfahrtsdatum",
    "Baureihe Zug", "Marketing-Linie", "Zug-Nr.",
    "Ab-Zeit (Plan)", "An-Zeit (Plan)",
    "Energieverbrauch [kWh]", "sEV",
    "Temperatur Ab [°C]", "Temperatur An [°C]",
    "Schneehöhe Ab [cm]", "Schneehöhe An [cm]",
    "Niederschlagsmenge Ab [l/m²]", "Niederschlagsmenge An [l/m²]",
    "Luftfeuchtigkeit Ab [%]", "Luftfeuchtigkeit An [%]",
    "Wind Mittel Ab [m/s]", "Wind Mittel An [m/s]",
    "Windspitze Ab [m/s]", "Windspitze An [m/s]",
    "TLKM",
]
id_cols = [c for c in id_cols if c in original_rows.columns]

# Grab SHAP columns (individual + totals)
shap_cols = [c for c in shap_results.columns if c.startswith("shap_")]
shap_cols += ["weather_shap_total", "non_weather_shap_total"]

merged = pd.concat([
    original_rows[id_cols].reset_index(drop=True),
    shap_results[shap_cols].reset_index(drop=True),
], axis=1)

# Compute corrected sEV (weather effect removed)
merged["corrected_sEV"] = merged["sEV"] - merged["weather_shap_total"]

merged.to_csv(OUTPUT, index=False)
print(f"Saved {len(merged):,} rows to {OUTPUT}")
print(f"Columns: {list(merged.columns)}")
print(f"\nSample (first 5 rows):")
print(merged.head().to_string())
