"""Train a LightGBM model to predict sEV and report feature importances."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import polars as pl
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("Output/filtered/filtered.parquet")
OUTPUT_DIR = Path("Output/model_tim")

NUMERIC_CAST_COLS = [
    #"Dauer_plan",
    #"Abfahrtzeitrel",
    #"Ankunftzeitrel",
    #"Dauer_ist",
    "Dauer_kombiniert",
    "Zugeinheiten Km",
    #"TFZ-km",
    #"TLKM",
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

TARGET = "sEV"
BAUREIHE = None  # Set to e.g. "BR 146" to train on one, or None for all

CATEGORICAL_COLS = ["Baureihe Zug"]


def train(input_path: Path = INPUT_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    df = pl.read_parquet(input_path)

    # Optionally filter to a specific Baureihe
    if BAUREIHE:
        df = df.filter(pl.col("Baureihe Zug") == BAUREIHE)
        print(f"Rows after Baureihe Zug = '{BAUREIHE}' filter: {df.height:,}")
    else:
        print(f"Training on all Baureihen: {df.height:,} rows")

    # Cast feature columns to float
    existing_features = [c for c in NUMERIC_CAST_COLS if c in df.columns]
    existing_cat = [c for c in CATEGORICAL_COLS if c in df.columns]

    df = df.with_columns(pl.col(TARGET).cast(pl.Float64, strict=False))

    # Derived features computed in process.py
    #if "delay_diff_min" in df.columns:
     #   existing_features.append("delay_diff_min")
    #for col in ("year", "month", "day"):
     #   if col in df.columns:
      #      existing_features.append(col)
    for col in ("elevation_diff_m",):
        if col in df.columns:
            existing_features.append(col)

    all_features = existing_features + existing_cat

    # Drop rows with null target or all-null features
    df = df.filter(pl.col(TARGET).is_not_null())
    df = df.filter(pl.col(TARGET) >= 5)
    df = df.select(all_features + [TARGET])

    # Export training data as CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_dir / "training_data.csv")
    print(f"Exported training data: {df.height:,} rows → {output_dir / 'training_data.csv'}")

    # Convert to pandas for sklearn/lightgbm
    pdf = df.to_pandas()
    X = pdf[all_features]
    y = pdf[TARGET]

    # Encode categorical columns
    for col in existing_cat:
        X[col] = X[col].astype("category")

    # Sanitize feature names for LightGBM (no special JSON chars)
    clean_names = {c: c.replace("[", "").replace("]", "").replace("°", "").replace("²", "2").replace("/", "_") for c in X.columns}
    X = X.rename(columns=clean_names)

    print(f"Training set shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(500, verbose=True), lgb.log_evaluation(500)],
    )

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"\nTest R²: {score:.4f}")

    # Feature importance (map back to original names, as percentage)
    reverse_names = {v: k for k, v in clean_names.items()}
    total_importance = model.feature_importances_.sum()
    importance = pl.DataFrame({
        "feature": [reverse_names[c] for c in X.columns],
        "importance_pct": (model.feature_importances_ / total_importance * 100).round(2),
    }).sort("importance_pct", descending=True)

    print("\nFeature Importances:")
    print(importance)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_dir / "model.txt"))
    print(f"Model saved to: {output_dir / 'model.txt'}")
    importance.write_csv(output_dir / "feature_importance.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    imp_pd = importance.to_pandas()
    ax.barh(imp_pd["feature"], imp_pd["importance_pct"])
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"LightGBM Feature Importance for sEV (R²={score:.4f})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {output_dir}/")
    plt.close()


if __name__ == "__main__":
    train()
