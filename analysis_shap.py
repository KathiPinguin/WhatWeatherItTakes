"""Approach 1: SHAP Decomposition — quantify weather effect via SHAP values."""

from __future__ import annotations

from pathlib import Path

import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from analysis_common import (
    WEATHER_FEATURES_CLEAN,
    load_data,
    prepare_Xy,
    split,
)

OUTPUT_DIR = Path("Output/model_tim/analysis/shap")
MODEL_PATH = Path("Output/model_tim/model.txt")

# Set to a number (e.g. 10_000) for quick testing, or None for all rows
SAMPLE_N: int | None = None
SAMPLE_N = 10_000

def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def elapsed() -> str:
        return f"{time.time() - t0:.1f}s"

    # --- Load pre-trained model ---
    print(f"[shap] [{elapsed()}] 1/8 Loading model from {MODEL_PATH}...")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model_features = booster.feature_name()
    print(f"[shap] Model features ({len(model_features)}): {model_features}")

    # --- Load data and select model features ---
    print(f"[shap] [{elapsed()}] 2/8 Loading and preparing data...")
    df, num_feats, cat_feats = load_data()
    all_feats = num_feats + cat_feats
    X, y, cmap = prepare_Xy(df, all_feats, cat_feats)

    # LightGBM replaces spaces with underscores when saving to .txt
    X.columns = X.columns.str.replace(" ", "_")

    # Keep only the features the model was trained on
    X = X[model_features]
    X_train, X_test, y_train, y_test = split(X, y)

    # Evaluate R² on test set
    print(f"[shap] [{elapsed()}] 3/8 Evaluating model...")
    y_pred_test = booster.predict(X_test)
    ss_res = ((y_test - y_pred_test) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"[shap] [{elapsed()}] Test R² = {r2:.4f}")

    # --- SHAP on ALL rows ---
    # Concatenate train+test to get full dataset
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)

    if SAMPLE_N is not None:
        X_all = X_all.sample(n=min(SAMPLE_N, len(X_all)), random_state=42)
        y_all = y_all.loc[X_all.index]

    sample_n = len(X_all)

    print(f"[shap] [{elapsed()}] 4/8 Computing SHAP values on ALL {sample_n:,} rows (this may take a while)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_all)
    print(f"[shap] [{elapsed()}] SHAP computation done.")

    # Alias for downstream code
    X_sample = X_all
    y_sample = y_all

    # --- Weather attribution ---
    weather_clean = [w.replace(" ", "_") for w in WEATHER_FEATURES_CLEAN]
    weather_idx = [i for i, c in enumerate(X_sample.columns) if c in weather_clean]
    weather_names = [X_sample.columns[i] for i in weather_idx]
    non_weather_idx = [i for i in range(len(X_sample.columns)) if i not in weather_idx]

    weather_shap = shap_values[:, weather_idx].sum(axis=1)
    non_weather_shap = shap_values[:, non_weather_idx].sum(axis=1)

    result_df = X_sample.copy()
    result_df["sEV_actual"] = y_sample.values
    result_df["predicted_sEV"] = booster.predict(X_sample)
    result_df["corrected_sEV"] = y_sample.values - weather_shap
    # Individual weather SHAP values
    for i, name in zip(weather_idx, weather_names):
        result_df[f"shap_{name}"] = shap_values[:, i]
    result_df["weather_shap_total"] = weather_shap
    result_df["non_weather_shap_total"] = non_weather_shap
    result_df["month"] = result_df.get("month", pd.Series(dtype=float))

    print(f"[shap] [{elapsed()}] 5/8 Saving attribution CSV...")
    result_df.to_csv(OUTPUT_DIR / "weather_attribution.csv", index=False)
    print(f"[shap] [{elapsed()}] Saved per-trip attribution → {OUTPUT_DIR / 'weather_attribution.csv'}")

    # --- Aggregate by month ---
    print(f"[shap] [{elapsed()}] 6/8 Aggregating by month...")
    if "month" in result_df.columns:
        monthly = result_df.groupby("month")["weather_shap_total"].agg(["mean", "median", "std", "count"])
        monthly.columns = ["mean_weather_shap", "median_weather_shap", "std_weather_shap", "n_trips"]
        monthly = monthly.sort_index()
        monthly.to_csv(OUTPUT_DIR / "monthly_weather_impact.csv")

        fig, ax = plt.subplots(figsize=(10, 5))
        months = monthly.index.astype(int)
        ax.bar(months, monthly["mean_weather_shap"], color="steelblue")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Weather SHAP (kWh/km)")
        ax.set_title("Average Weather Contribution to sEV by Month")
        ax.set_xticks(range(1, 13))
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "monthly_weather_impact.png", dpi=150)
        plt.close()

    # --- Summary stats ---
    print(f"[shap] [{elapsed()}] 7/8 Computing summary stats...")
    abs_weather = np.abs(weather_shap)
    abs_total = np.abs(shap_values).sum(axis=1)
    weather_pct = (abs_weather.sum() / abs_total.sum()) * 100

    stats_lines = [
        f"SHAP Weather Analysis (n={sample_n:,})",
        f"Model R²: {r2:.4f}",
        f"",
        f"Weather SHAP contribution:",
        f"  Mean:   {weather_shap.mean():+.3f} kWh/km",
        f"  Median: {np.median(weather_shap):+.3f} kWh/km",
        f"  Std:    {weather_shap.std():.3f} kWh/km",
        f"  Min:    {weather_shap.min():+.3f} kWh/km",
        f"  Max:    {weather_shap.max():+.3f} kWh/km",
        f"",
        f"Weather share of total |SHAP|: {weather_pct:.1f}%",
        f"",
        f"Per weather feature mean |SHAP|:",
    ]
    for i, name in zip(weather_idx, weather_names):
        mean_abs = np.abs(shap_values[:, i]).mean()
        stats_lines.append(f"  {name}: {mean_abs:.3f} kWh/km")

    stats_text = "\n".join(stats_lines)
    (OUTPUT_DIR / "stats.txt").write_text(stats_text)
    print(stats_text)

    # --- SHAP summary plot (subsample for visualization) ---
    print(f"[shap] [{elapsed()}] 8/8 Generating plots...")
    plot_n = min(50_000, sample_n)
    plot_idx = np.random.default_rng(42).choice(sample_n, size=plot_n, replace=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[plot_idx], X_sample.iloc[plot_idx], show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Weather vs non-weather grouped bar ---
    weather_mean_abs = np.abs(shap_values[:, weather_idx]).mean(axis=0)
    non_weather_mean_abs = np.abs(shap_values[:, non_weather_idx]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(["Weather features", "Non-weather features"],
           [weather_mean_abs.sum(), non_weather_mean_abs.sum()],
           color=["#e74c3c", "#3498db"])
    ax.set_ylabel("Sum of mean |SHAP| (kWh/km)")
    ax.set_title("Weather vs Non-Weather Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weather_vs_nonweather.png", dpi=150)
    plt.close()

    # --- Dependence plots for top weather features ---
    top_weather = sorted(zip(weather_idx, weather_names,
                             np.abs(shap_values[:, weather_idx]).mean(axis=0)),
                         key=lambda x: x[2], reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (idx, name, _) in zip(axes.flat, top_weather):
        ax.scatter(X_sample.iloc[plot_idx, idx], shap_values[plot_idx, idx], alpha=0.1, s=1, rasterized=True)
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value (kWh/km)")
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("SHAP Dependence: Top 6 Weather Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_weather_dependence.png", dpi=150)
    plt.close()

    # --- Per-feature SHAP table ---
    feat_table_rows = []
    for i, name in zip(weather_idx, weather_names):
        vals = shap_values[:, i]
        feat_table_rows.append({
            "feature": name,
            "mean_abs_shap": round(float(np.abs(vals).mean()), 4),
            "mean_shap": round(float(vals.mean()), 4),
            "std_shap": round(float(vals.std()), 4),
            "min_shap": round(float(vals.min()), 4),
            "max_shap": round(float(vals.max()), 4),
        })
    feat_table_rows.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    # --- Monthly table for report ---
    monthly_rows = []
    if "month" in result_df.columns:
        for m in sorted(result_df["month"].dropna().unique()):
            mask = result_df["month"] == m
            vals = result_df.loc[mask, "weather_shap_total"]
            monthly_rows.append({
                "month": int(m),
                "mean": round(float(vals.mean()), 3),
                "median": round(float(vals.median()), 3),
                "std": round(float(vals.std()), 3),
                "n": int(mask.sum()),
            })

    # --- Generate markdown report ---
    report_lines = [
        "# SHAP Weather Attribution Report",
        "",
        "## Overview",
        "",
        f"- **Model:** LightGBM (10k estimators, lr=0.01, depth=10, leaves=127)",
        f"- **Test R²:** {r2:.4f}",
        f"- **SHAP sample size:** {sample_n:,} trips",
        f"- **Features:** {len(X_sample.columns)} total ({len(weather_idx)} weather, {len(non_weather_idx)} non-weather)",
        "",
        "## Key Finding",
        "",
        f"Weather features account for **{weather_pct:.1f}%** of total feature attribution (sum of |SHAP|).",
        "",
        f"On average, weather contributes **{weather_shap.mean():+.3f} kWh/km** per trip "
        f"(std: {weather_shap.std():.3f}, range: [{weather_shap.min():+.2f}, {weather_shap.max():+.2f}]).",
        "",
        "## Per-Feature Weather Attribution",
        "",
        "| Feature | Mean |SHAP| (kWh/km) | Mean SHAP | Std | Min | Max |",
        "|---------|----------------------|-----------|-----|-----|-----|",
    ]
    for row in feat_table_rows:
        report_lines.append(
            f"| {row['feature']} | {row['mean_abs_shap']:.4f} | {row['mean_shap']:+.4f} | {row['std_shap']:.4f} | {row['min_shap']:+.4f} | {row['max_shap']:+.4f} |"
        )

    report_lines += [
        "",
        "## Monthly Weather Impact",
        "",
        "| Month | Mean Weather SHAP (kWh/km) | Median | Std | N trips |",
        "|-------|---------------------------|--------|-----|---------|",
    ]
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    for row in monthly_rows:
        mname = month_names.get(row["month"], str(row["month"]))
        report_lines.append(
            f"| {row['month']} ({mname}) | {row['mean']:+.3f} | {row['median']:+.3f} | {row['std']:.3f} | {row['n']:,} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
        "- **Positive SHAP** = weather condition *increases* energy consumption relative to average",
        "- **Negative SHAP** = weather condition *decreases* energy consumption relative to average",
        "- The sum of all SHAP values for a trip equals the difference between that trip's predicted sEV and the global average sEV",
        "",
        "## Plots",
        "",
        "- `shap_summary.png` — Beeswarm plot of all features ranked by importance",
        "- `shap_weather_dependence.png` — Scatter plots showing how each weather variable drives sEV",
        "- `weather_vs_nonweather.png` — Bar chart comparing weather vs non-weather total attribution",
        "- `monthly_weather_impact.png` — Monthly bar chart of mean weather SHAP",
        "",
        "## Data Source",
        "",
        "- Training data: `Output/filtered/filtered.parquet` (1,380,790 rows, 30 features)",
        "- Weather data: station-level temperature, wind, precipitation, snow, humidity at departure and arrival",
        "- Elevation: SRTM 90m via srtm.py (NASA public domain data)",
    ]

    report_text = "\n".join(report_lines)
    (OUTPUT_DIR / "report.md").write_text(report_text)
    print(f"\n[shap] [{elapsed()}] Report saved to {OUTPUT_DIR / 'report.md'}")
    print(f"[shap] [{elapsed()}] All done! Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
