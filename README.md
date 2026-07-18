# CRO-Heatwave

District-level 10-day maximum temperature forecasting for Uttar Pradesh, India, built on daily meteorological data from the NASA POWER API. The long-term goal is a district-level heatwave early-warning system; the current phase focuses on building a reliable, reproducible data pipeline as the foundation for that.

## Status

**Phase 0 — Data Pipeline: Complete.**

75 UP districts, daily data 1984–2025, pulled from the NASA POWER API, cleaned, and feature-engineered.

**Phase A (model development) has not started.** Baseline models through GRU-based sequence models, walk-forward evaluation, and window-length tuning are next.

See [`docs/PHASE_0_DATA_PIPELINE.md`](docs/PHASE_0_DATA_PIPELINE.md) for this phase's detailed scope and [`docs/MASTER_PROJECT_DOCUMENTATION.md`](docs/MASTER_PROJECT_DOCUMENTATION.md) for the overall system design.

## Design principles

- **Notebooks never own a pipeline step.** All transformation logic lives in `src/`; notebooks, when used, only call into it.
- **Every derived feature's provenance is labeled.** Some features here are standard, literature-established formulas (e.g., the solar clearness index); others are documented engineering assumptions where no established formula exists. The code distinguishes between the two rather than presenting both with equal confidence.
- **Fold-aware statistics only.** Reference statistics like climatological normals are computed per evaluation fold, from that fold's training years only — never from the full dataset — to avoid leaking future information into past evaluations.
- **Config over hardcoding.** District lists, date ranges, and pipeline parameters live in `configs/`, not inline in scripts.

## Repository structure

```
CRO-Heatwave/
├── configs/
│   └── data.yaml              # district centroids, date range, NASA POWER params, missing-value policy
├── data/
│   ├── raw/                   # gitignored — per-district parquet, raw API pulls
│   └── engineered/            # gitignored — per-district parquet, cleaned + feature-engineered
├── src/
│   ├── data/
│   │   ├── fetch_nasa_power.py       # raw data acquisition, idempotent, manifest-tracked
│   │   └── clean_and_engineer.py     # missing-value handling, feature engineering
│   ├── models/
│   │   └── interfaces.py             # common ForecastModel interface (Phase A onward)
│   └── eval/                         # evaluation layer (Phase A-2 onward)
├── scripts/                          # one-off utilities — not part of the runtime pipeline
│   ├── compute_district_centroids.py # GADM boundary -> centroid computation (one-time)
│   ├── generate_data_yaml.py         # centroid CSV -> configs/data.yaml (one-time)
│   ├── validate_schema.py            # schema check across all engineered files
│   └── plausibility_check.py         # domain sanity check (May T2M_MAX by district)
├── notebooks/                        # exploratory only, never owns pipeline logic
├── docs/
│   ├── MASTER_PROJECT_DOCUMENTATION.md
│   └── PHASE_0_DATA_PIPELINE.md
└── train.py                          # entrypoint stub, Phase A onward
```

Module boundaries are enforced by import direction: `src/data/` never imports from `src/models/` or `src/eval/`; `src/models/` never imports from `src/eval/`.

## Setup

```bash
git clone https://github.com/tarun-rai21/CRO-Heatwave.git
cd CRO-Heatwave

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Data pipeline

### 1. District centroids

`configs/data.yaml` contains all 75 UP district centroids, computed from GADM 4.1 administrative boundaries (level 2), reprojected to an equal-area CRS (EPSG:7755) before centroid computation, then reprojected back to WGS84. Naive lat/lon vertex averaging was deliberately avoided, since it produces geometrically incorrect centroids for irregularly shaped districts.

Three districts required reconciling GADM's source naming with current official names:

| GADM name | Current official name |
|---|---|
| Allahabad | Prayagraj |
| Faizabad | Ayodhya |
| Sant Ravi Das Nagar | Bhadohi |

Regeneration (only needed if boundaries or renames change):
```bash
python scripts/compute_district_centroids.py
python scripts/generate_data_yaml.py
```

### 2. Raw data acquisition

```bash
python src/data/fetch_nasa_power.py
```

Pulls daily records for all 75 districts, 1984-01-01 to 2025-12-31, from the NASA POWER Daily Point API. One request per district (centroid point query). Idempotent — safe to re-run; already-fetched districts (tracked via `data/raw/manifest.json`) are skipped unless explicitly forced. NASA POWER's documented fill value (`-999`) is mapped to `NaN` at this stage.

### 3. Cleaning and feature engineering

```bash
python src/data/clean_and_engineer.py
```

Per district, independently (no cross-district contamination):

- **Missing-value policy**: gaps ≤3 days are linearly interpolated; gaps >3 days are excluded and flagged (`has_excluded_gap` column). Every interpolation and exclusion is counted and written to `data/engineered/data_quality_report.csv`.
- **Autoregressive features**: `T2M_MAX_lag1/3/7`, `T2M_MAX_roll7_mean/std`.
- **Circular encodings**: day-of-year and wind-direction sin/cos.
- **Derived land-surface/atmospheric features**, with provenance documented in code:
  - `SOIL_DRYNESS_lag1` — engineering assumption, equal-weighted combination of root-zone and surface soil wetness.
  - `ATM_DRYNESS_lag1` — direct linear transform of relative humidity, a standard atmospheric dryness proxy.
  - `RADIATION_EFFECTIVE_lag1` — clearness index (measured / clear-sky solar radiation), an established solar-radiation-literature concept.
- **No climatological normal** is computed in this file — that's deferred to per-fold computation in a later phase, to keep evaluation folds free of future-information leakage.

## Validation performed

- **Schema validation**: all 75 engineered files checked for correct columns, row count (15,341, matching the exact calendar-day count for 1984–2025 inclusive with leap years), dtypes, and no unexpected NaNs outside the expected lag/rolling-window warm-up period.
- **Plausibility check**: May (pre-monsoon heat season) mean `T2M_MAX` computed per district. Result: 40.9°C (Rampur, near the Himalayan foothills) to 44.0°C (Banda, in the Bundelkhand region) — a geographically coherent gradient matching known UP climate patterns, with no outlier districts.
- **Missing-data audit**: only 6 NaN cells found across ~13.8 million raw data points (75 districts × 15,341 days × 12 variables), all traced to a single coherent event — one radiation-sensor gap on 1994-01-11 affecting 6 geographically adjacent western UP districts.

## Known limitations

- NASA POWER's ~50km grid resolution vs. district polygon boundaries is a genuine resolution mismatch.
- District centroid-point sampling approximates an entire district's climate; larger or irregularly shaped districts are more affected.
- NASA POWER has no fixed published API rate limit; the fetch script uses a conservative fixed delay plus exponential backoff rather than a hardcoded requests-per-minute assumption.

## License

Not yet decided.