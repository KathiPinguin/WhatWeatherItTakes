# Hack4Rail – Quantifying the Weather Effect on Train Energy Consumption

## What We Do

We measure how much weather (temperature, wind, snow, rain, humidity) affects the energy consumption of Deutsche Bahn regional trains. Starting from ~5 million trip records, we train a machine learning model that predicts energy use per kilometre, then use SHAP (a game-theory method) to decompose each trip's prediction into individual contributions — isolating the exact kWh/km attributable to weather vs. operational factors.

**Key result:** Weather accounts for ~2.7% of total energy attribution. Cold months add up to +0.6 kWh/km per trip; warm months save up to -0.35 kWh/km. Temperature is the dominant weather driver.

---

## How It Works

### Simple Explanation

1. We collect trip data: energy used, distance, train type, weather at departure/arrival, route elevation
2. We train a model that learns the relationship between all these factors and energy per km
3. For each trip, we ask: "How much did temperature contribute? Wind? Snow?" — using a fair mathematical split (SHAP values)
4. We subtract the weather effect to get a **corrected sEV** — what the trip *would have* consumed under neutral weather

### Technical Explanation

We use **LightGBM** (gradient-boosted decision trees, 10k estimators, lr=0.01, depth=10, 127 leaves) to predict sEV (specific energy consumption, kWh/km) from 30 features. The model achieves R²=0.77 on a held-out 20% test set.

We then apply **TreeSHAP** (Lundberg & Lee, 2017) — an exact, polynomial-time algorithm for computing Shapley values on tree ensembles. For each trip, SHAP decomposes the prediction into additive feature contributions:

```
predicted_sEV = baseline + Σ SHAP(feature_i)
```

where the baseline is the global mean sEV. We sum the SHAP values of the 14 weather features to get the total weather effect per trip, and define:

```
corrected_sEV = actual_sEV − weather_shap_total
```

This removes the weather-attributable component while preserving all other operational effects.

---

## Pipeline

```
Data/*.xlsx → [ingest] → combined.parquet (5.3M rows)
                              ↓
                        [process] → filtered.parquet (1.38M rows)
                              ↓
                        [SHAP analysis] → weather_attribution_full.csv
                                          monthly_weather_impact.csv
                                          corrected_sEV per trip
```

---

## Setup

```bash
# Install uv (Python package manager) if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# macOS only: LightGBM requires OpenMP
brew install libomp
```

---

## Usage

### 1. Ingest (`main.py`)

Reads all monthly Excel exports from `Data/` and combines them into a single parquet file.

```bash
uv run python main.py
```

- Skips temporary Excel lock files (`~$...`)
- Adds `source_file` and `source_period` metadata
- Output: `Output/ingest/combined.parquet`

### 2. Process (`process.py`)

Applies quality filters, type conversions, and enriches with station elevation data.

```bash
uv run python process.py --sample 1000   # with CSV sample for inspection
uv run python process.py                  # parquet only
```

**Filters applied:**
- Remove rows with null/zero energy values
- Remove null temperatures
- Remove shunting trips (Leist Typ = "S")
- Remove very short trips (TFZ-km ≤ 5)
- Remove sEV outliers outside [6, 50] kWh/km

**Enrichment:**
- Station elevation via SRTM 90m tiles (NASA)
- Elevation difference between departure and arrival
- Delay metrics derived from planned vs actual timestamps

### 3. SHAP Weather Analysis (`analysis_shap.py`)

Trains the model and computes per-trip weather attribution.

```bash
uv run python analysis_shap.py
```

This runs the full pipeline:
1. Loads filtered data (1.38M trips, 30 features)
2. Trains LightGBM with early stopping (80/20 split)
3. Computes SHAP values on 50,000 test samples
4. Outputs per-trip weather decomposition + corrected sEV

Runtime: ~25 minutes (training ~15 min, SHAP ~10 min).

### 4. Merge with Original Data (`merge_shap_original.py`)

Joins the SHAP results back onto the original human-readable columns (station names, train numbers, dates).

```bash
uv run python merge_shap_original.py
```

---

## Output Structure

```
Output/
├── ingest/
│   └── combined.parquet              # raw combined data
├── filtered/
│   ├── filtered.parquet              # cleaned + enriched
│   └── filter_conditions.txt         # applied filter log
├── model/
│   ├── feature_importance.csv        # LightGBM feature importance
│   └── feature_importance.png        # bar chart
└── analysis/shap/
    ├── weather_attribution.csv       # 50k trips with all SHAP values
    ├── weather_attribution_full.csv  # merged with original columns
    ├── monthly_weather_impact.csv    # monthly aggregated weather effect
    ├── monthly_weather_impact.png    # bar chart by month
    ├── shap_summary.png              # beeswarm plot (all features)
    ├── shap_weather_dependence.png   # scatter: weather feature vs SHAP
    ├── weather_vs_nonweather.png     # weather vs non-weather importance
    ├── stats.txt                     # summary statistics
    └── report.md                     # full analysis report
```

---

## Features

| Category | Features |
|----------|----------|
| Operational | Dauer_plan, Dauer_ist, Dauer_kombiniert, Abfahrtzeitrel, Ankunftzeitrel, delay_diff_min |
| Distance | Zugeinheiten Km, TFZ-km, TLKM |
| Route | elevation_ab_m, elevation_an_m, elevation_diff_m |
| Weather (14) | Temperature, snow depth, precipitation, fresh snow, humidity, mean wind, wind gusts — each at departure and arrival station |
| Temporal | year, month, day |
| Categorical | Baureihe Zug (train type) |

---

## Dependencies

- **polars** – fast DataFrame processing
- **lightgbm** – gradient-boosted trees
- **shap** – SHAP value computation
- **scikit-learn** – train/test splitting
- **srtm.py** – elevation data from NASA SRTM tiles
- **matplotlib / seaborn** – visualization
- **scikit-learn** – train/test split, metrics
- **matplotlib / seaborn** – visualization
- **pyarrow** – parquet ↔ pandas conversion
