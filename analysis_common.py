"""Shared data loading and feature preparation for all analysis scripts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("Output/filtered/filtered.parquet")
TARGET = "sEV"

NUMERIC_FEATURES = [
    "Dauer_plan",
    "Abfahrtzeitrel",
    "Ankunftzeitrel",
    "Dauer_ist",
    "Dauer_kombiniert",
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

DERIVED_FEATURES = ["delay_diff_min", "year", "month", "day",
                    "elevation_ab_m", "elevation_an_m", "elevation_diff_m"]

CATEGORICAL_FEATURES = ["Baureihe Zug"]

WEATHER_FEATURES_ORIG = [
    "Temperatur Ab [°C]", "Temperatur An [°C]",
    "Schneehöhe Ab [cm]", "Schneehöhe An [cm]",
    "Niederschlagsmenge Ab [l/m²]", "Niederschlagsmenge An [l/m²]",
    "Neuschneemenge Ab [cm/m²]", "Neuschneemenge An [cm/m²]",
    "Luftfeuchtigkeit Ab [%]", "Luftfeuchtigkeit An [%]",
    "Wind Mittel Ab [m/s]", "Wind Mittel An [m/s]",
    "Windspitze Ab [m/s]", "Windspitze An [m/s]",
]

ROUTE_COLS = ["Ab-Ort", "An-Ort"]


def _clean_name(c: str) -> str:
    return c.replace("[", "").replace("]", "").replace("°", "").replace("²", "2").replace("/", "_")


def clean_names_map(columns: list[str]) -> dict[str, str]:
    return {c: _clean_name(c) for c in columns}


WEATHER_FEATURES_CLEAN = [_clean_name(c) for c in WEATHER_FEATURES_ORIG]


def load_data(input_path: Path = INPUT_PATH, extra_cols: list[str] | None = None):
    """Load filtered parquet and return (polars DataFrame, feature lists).

    extra_cols: additional columns to keep (e.g. Ab-Ort, An-Ort for route analysis).
    """
    df = pl.read_parquet(input_path)
    df = df.with_columns(pl.col(TARGET).cast(pl.Float64, strict=False))

    existing_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    existing_derived = [c for c in DERIVED_FEATURES if c in df.columns]
    existing_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    all_features = existing_num + existing_derived + existing_cat
    keep_cols = list(dict.fromkeys(all_features + [TARGET] + (extra_cols or [])))

    df = df.filter(pl.col(TARGET).is_not_null())
    df = df.filter(pl.col(TARGET) >= 5)
    df = df.select([c for c in keep_cols if c in df.columns])

    return df, existing_num + existing_derived, existing_cat


def prepare_Xy(df: pl.DataFrame, features: list[str], cat_cols: list[str]):
    """Convert to pandas X, y with clean names and categorical encoding."""
    pdf = df.to_pandas()
    X = pdf[features].copy()
    y = pdf[TARGET]

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    cmap = clean_names_map(X.columns.tolist())
    X = X.rename(columns=cmap)
    return X, y, cmap


def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_lgbm(X_train, y_train, X_test, y_test, n_estimators=10000, verbose=False):
    """Train LightGBM with standard hyperparameters. Returns fitted model."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(500, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(500))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
    return model
