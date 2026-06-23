"""Approach 1: SHAP Decomposition — quantify weather effect via SHAP values."""

from __future__ import annotations

from pathlib import Path

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
    train_lgbm,
)

OUTPUT_DIR = Path("Output/analysis/shap")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load & train ---
    print("[shap] Loading data...")
    df, num_feats, cat_feats = load_data()
    all_feats = num_feats + cat_feats
    X, y, cmap = prepare_Xy(df, all_feats, cat_feats)
    X_train, X_test, y_train, y_test = split(X, y)

    print(f"[shap] Training LightGBM ({X_train.shape[0]:,} train, {X_test.shape[0]:,} test)...")
    model = train_lgbm(X_train, y_train, X_test, y_test, verbose=True)
    r2 = model.score(X_test, y_test)
    print(f"[shap] Test R² = {r2:.4f}")

    # --- SHAP on sample ---
    sample_n = min(50_000, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    print(f"[shap] Computing SHAP values on {sample_n:,} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- Weather attribution ---
    weather_idx = [i for i, c in enumerate(X_sample.columns) if c in WEATHER_FEATURES_CLEAN]
    weather_names = [X_sample.columns[i] for i in weather_idx]
    non_weather_idx = [i for i in range(len(X_sample.columns)) if i not in weather_idx]

    weather_shap = shap_values[:, weather_idx].sum(axis=1)
    non_weather_shap = shap_values[:, non_weather_idx].sum(axis=1)

    result_df = X_sample.copy()
    result_df["sEV_actual"] = y_sample.values
    result_df["corrected_sEV"] = y_sample.values - weather_shap
    # Individual weather SHAP values
    for i, name in zip(weather_idx, weather_names):
        result_df[f"shap_{name}"] = shap_values[:, i]
    result_df["weather_shap_total"] = weather_shap
    result_df["non_weather_shap_total"] = non_weather_shap
    result_df["month"] = result_df.get("month", pd.Series(dtype=float))

    result_df.to_csv(OUTPUT_DIR / "weather_attribution.csv", index=False)
    print(f"[shap] Saved per-trip attribution → {OUTPUT_DIR / 'weather_attribution.csv'}")

    # --- Aggregate by month ---
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

    # --- SHAP summary plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=30)
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
        ax.scatter(X_sample.iloc[:, idx], shap_values[:, idx], alpha=0.1, s=1, rasterized=True)
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
    print(f"\n[shap] Report saved to {OUTPUT_DIR / 'report.md'}")
    print(f"[shap] All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
