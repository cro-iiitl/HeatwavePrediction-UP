# Phase 0 — Data Pipeline Rebuild

**Extends:** `MASTER_PROJECT_DOCUMENTATION.md` — see §4 (Project Structure), §8.7 (Design Decision: re-pull raw data), §11 (Data Pipeline), §15 (Phase Overview)
**Phase objective (per master §15):** Rebuild the data pipeline from raw NASA POWER pulls, with documented, reproducible missing-value handling and feature engineering — replacing the prior project's opaque, undocumented CSV.
**Dependencies:** None (first phase).
**Completion criteria (per master §15):** Raw and engineered data reproducible from config; summary statistics sanity-checked against the prior project's CSV.

This document does not restate design decisions already made in the master document — it implements them. Where this phase requires a decision the master document didn't make, it's called out explicitly as new.

---

## 1. Scope of This Phase

**In scope:**
- Pulling raw daily meteorological records for the 75 districts of Uttar Pradesh, 1984-01-01 to 2025-12-31, from the NASA POWER API.
- Documenting and implementing a missing-value policy.
- Producing all engineered features listed in master §6.1 (lags, rolling stats, circular encodings) from the raw pull — not inheriting any pre-engineered column from the prior project.
- Sanity-checking the new pull's summary statistics against the prior project's CSV.

**Out of scope (deferred to later phases):**
- Windowing/sequence generation (master §11.4 — this is Phase A-1/A-2, not Phase 0).
- Any model training.
- The 50 additional districts needed for spatial modeling (Phase B, per master §8.10).

---

## 2. Inputs and Outputs Contract

| Stage | Input | Output |
|---|---|---|
| `fetch_nasa_power.py` | `configs/data.yaml` (district list with lat/lon, date range, variable list) | One raw file per district: `data/raw/{district_id}.parquet` — untransformed API response, one row per day |
| `clean_and_engineer.py` | `data/raw/*.parquet` + `configs/data.yaml` (missing-value policy parameters) | One engineered file per district: `data/engineered/{district_id}.parquet` — cleaned, feature-engineered, one row per day, ready for windowing (Phase 1) |

Both stages operate **per district independently** — this is deliberate and matches the "no cross-district contamination" principle already fixed for windowing (master §11.4); keeping it consistent from the raw-pull stage onward means a single district can be re-pulled or re-processed without touching the other 74.

---

## 3. `fetch_nasa_power.py` — Raw Data Acquisition

### 3.1 API Details

NASA POWER's **Daily Point API** (`https://power.larc.nasa.gov/api/temporal/daily/point`) is the correct endpoint here, not the regional/gridded API — point queries take a single lat/lon per district centroid and return a daily time series for the full requested date range in one call, which matches "one row per district-day" without needing to resolve a district's shape against a grid.

- **Pending Design Decision (new):** exact rate-limit behavior (requests per minute, cooldown) — needs verification against NASA POWER's current API documentation before implementation, since these limits are set externally and can change; do not hardcode an assumed limit without checking current docs at implementation time.

### 3.4 Manifest for Tracking Completion

A `data/raw/manifest.json` (or equivalent) records, per district: fetch timestamp, date range actually returned, row count, and a checksum/hash of the response. This is what makes the fetch idempotent (§3.2) and gives Phase 0's validation step (§8) something concrete to check against.

---

## 4. Raw Data Schema

One row per (district_id, date), columns are the raw API response fields, untransformed:

| Column | Type | Notes |
|---|---|---|
| district_id | string | Matches the district list in configs/data.yaml |
| date | date | ISO format |
| lat, lon | float | District centroid, from config — not from the API response, attached at write time for traceability |
| T2M_MAX, T2M_MIN | float | °C |
| RH2M | float | Relative humidity, % |
| WS10M, WD10M | float | Wind speed (m/s), wind direction (degrees) — raw, not yet circular-encoded |
| PRECTOTCORR | float | Precipitation, mm/day |
| PS | float | Surface pressure, kPa |
| ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN | float | Radiation, kWh/m²/day |
| CLOUD_AMT | float | % |
| GWETROOT, GWETTOP | float | Soil wetness fraction, 0–1 |

NASA POWER's documented fill value for missing data (historically `-999`) must be mapped to a proper null/NaN at this stage, **not** left as a numeric sentinel — this is the single most important transformation at the raw layer, since a `-999` treated as a real value would silently corrupt every downstream statistic (this is a specific, named suspicion about what may have gone wrong in the prior project's "no missing values" claim).

---

## 5. `clean_and_engineer.py` — Missing-Value Handling and Feature Engineering

### 5.1 Missing-Value Policy (implements master §11.2 direction)

Applied **per district, per variable, in chronological order** — never across districts:

1. Identify gaps (consecutive missing days) after fill-value → NaN conversion (§4).
2. **Gaps ≤ 3 days:** linear interpolation between the last known value before the gap and the first known value after it.
3. **Gaps > 3 days:** not interpolated. The affected date range is flagged in a `data_quality` companion column/table and excluded from any window that would span it (this connects forward to windowing in Phase A-1 — a window must not silently contain an interpolated-past-threshold or excluded region).
4. **Every interpolation and exclusion is logged** — count of interpolated days per district per variable, and count/date-ranges of excluded gaps — written to `data/engineered/data_quality_report.csv`. This is the artifact that replaces the prior project's unverified "no missing values present" claim with an actual, checkable count.

### 5.2 Feature Engineering Steps

In order, matching master §6.1:

1. **Autoregressive lags:** `T2M_MAX_lag1`, `T2M_MAX_lag3`, `T2M_MAX_lag7`, `T2M_MAX_roll7_mean`, `T2M_MAX_roll7_std` — computed per district on the (gap-handled) chronological series.
2. **Land-surface / atmospheric lags** (one-day-lagged, since same-day values aren't available at forecast time beyond the last observed day): soil dryness, atmospheric dryness, radiation, cloud amount lags — derived from the raw variables in §4 (exact derived-variable formulas, e.g. how "soil dryness" and "atmospheric dryness" are computed from `GWETROOT`/`GWETTOP`/`RH2M`, are a **Pending Design Decision (new)** — the prior project used these variable names but the derivation formula was never in the notebooks provided; this needs to be defined explicitly here rather than assumed to match the prior project).
3. **Circular encodings:** day-of-year (sin/cos), wind direction (sin/cos) — computed directly from `date` and `WD10M`.
4. **Climatological normal is not computed in this file** — per master §11.5, it must be computed per walk-forward fold from that fold's training years only, which means it belongs in the fold-splitting stage (Phase A-2), not baked into the shared engineered dataset here. This file must not compute or attach a single fixed climatological normal to the engineered output, to avoid quietly reintroducing the leakage risk flagged in §11.5.

### 5.3 VIF Screening — Explicitly Not Performed in This File

VIF-based collinearity screening (master §6.1, redone correctly per master §8-adjacent finding on the prior project's broken VIF exercise) is an **analysis step for Phase A-2**, run against the full candidate feature set once it exists — not a filtering step baked into `clean_and_engineer.py`. This file's output should retain the **full candidate feature set**; nothing is dropped here based on collinearity. This is a deliberate mirror of the decision in §3.1 to request the full variable superset from the API rather than pre-filtering.

---

## 6. Engineered Data Schema

One row per (district_id, date):

| Column group | Columns | Notes |
|---|---|---|
| Identifiers | district_id, date, lat, lon | Carried through unchanged |
| Target | T2M_MAX | Gap-handled (interpolated where ≤3-day gap, per §5.1) |
| Autoregressive | T2M_MAX_lag1/3/7, T2M_MAX_roll7_mean/std | |
| Atmospheric/land-surface | SOIL_DRYNESS_lag1, ATM_DRYNESS_lag1, RADIATION_EFFECTIVE_lag1, CLOUD_AMT_lag1, etc. | Exact derivation formulas: Pending Design Decision, see §5.2 item 2 |
| Circular encodings | DOY_SIN, DOY_COS, WD10M_SIN, WD10M_COS | |
| Data quality flag | has_excluded_gap (boolean) | True if this row falls within a >3-day excluded gap window for any variable; consumed by windowing (Phase A-1) to exclude affected windows |

No climatological-normal column exists in this file's output (§5.2 item 4).

---

## 7. Configuration Schema (`configs/data.yaml`)

```yaml
districts:
  # list of {district_id, lat, lon} for the 75 UP districts
  # geography-agnostic per master §8.10 — this list is what changes for Phase B, not code
  - district_id: "agra"
    lat: 27.18
    lon: 78.01
  # ...

date_range:
  start: "1984-01-01"
  end: "2025-12-31"

nasa_power:
  community: "AG"
  parameters:
    - T2M_MAX
    - T2M_MIN
    - RH2M
    - WS10M
    - WD10M
    - PRECTOTCORR
    - PS
    - ALLSKY_SFC_SW_DWN
    - CLRSKY_SFC_SW_DWN
    - CLOUD_AMT
    - GWETROOT
    - GWETTOP
  fill_value: -999        # mapped to NaN at raw-ingestion stage
  max_retries: 5
  backoff_base_seconds: 2  # exponential backoff base — rate-limit specifics pending

missing_value_policy:
  interpolation_max_gap_days: 3   # gaps beyond this are flagged and excluded, not filled
```

---

## 8. Validation Strategy

This phase's completion criteria (master §15) require the new pull to be **reproducible from config** and **sanity-checked against the prior project's CSV**. Concretely:

1. **Reproducibility check:** run `fetch_nasa_power.py` twice against the same config; confirm the manifest (§3.4) checksums match and no district requires re-fetching (idempotency, §3.2).
2. **Sanity check against prior CSV:** compare summary statistics (mean/std/min/max of T2M_MAX by month, by district) between the new pull and the prior project's CSV. Expected: broadly consistent seasonal shape (e.g., May ~42°C peak as seen in the prior EDA) — **not** expected to be numerically identical, since station/grid selection or API version differences could cause small divergence. A large, unexplained divergence (e.g., different peak month, wildly different variance) is a signal to investigate before proceeding, not something to average away.
3. **Missing-data report review:** manually review `data_quality_report.csv` (§5.1) for any district with an unusually high interpolation/exclusion rate relative to the others — this would indicate either a genuine data-sparse district or a bug in the fetch/parsing logic, and needs to be distinguished before moving to Phase A.
4. **Schema validation:** automated check that every output file in `data/engineered/` matches the schema in §6 exactly (column names, types, no unexpected nulls outside the documented data-quality flag).

---

## 9. Completion Checklist

- [ ] `configs/data.yaml` fully populated with all 75 district centroids.
- [ ] `fetch_nasa_power.py` implemented, idempotent, resumable, with manifest tracking (§3).
- [ ] Raw pull completed for all 75 districts, 1984–2025, with `-999` correctly mapped to NaN.
- [ ] `clean_and_engineer.py` implemented per §5, including the data-quality report.
- [ ] Missing-value interpolation/exclusion counts reviewed per district (§8.3).
- [ ] Derivation formulas for SOIL_DRYNESS / ATM_DRYNESS / RADIATION_EFFECTIVE explicitly defined (§10.1) — not carried over unexamined from the prior project.
- [ ] Summary-statistic sanity check against prior CSV completed and documented (§8.2).
- [ ] Schema validation passing on all 75 engineered output files.

Phase 0 is not complete until every item above is checked — per master §15, Phase A-1 (window-length ablation) depends on this phase's output.

---

## 10. New Decisions Made in This Phase

### 10.1 Derived-variable formulas must be defined explicitly, not inherited

**Context:** The prior project used feature names like `SOIL_DRYNESS_lag1` and `ATM_DRYNESS_lag1`, but no notebook provided showed the formula that derived these from raw NASA POWER variables (`GWETROOT`, `GWETTOP`, `RH2M`, etc.).

**Decision:** These formulas must be authored fresh and documented inline in `clean_and_engineer.py` (with a docstring citing the physical rationale) before this phase can be marked complete. Do not assume the prior project's undocumented formula was correct or even reconstructible.

**Rationale:** This is a direct instance of the exact failure mode this whole rebuild exists to fix (master §1.3) — an unverifiable derived feature. Inventing an assumed formula here to "match" the prior project would just relocate the same problem.

### 10.2 Full candidate feature superset requested and retained; no filtering before Phase A-2

**Decision:** Both the API pull (§3.1) and the engineered output (§5.3) retain every candidate variable; VIF-based screening happens once, correctly, in Phase A-2, against the complete feature set.

**Rationale:** Prevents repeating the prior project's error of a feature-selection decision made in one place (README) not matching what was actually executed in another (notebook code).

### 10.3 Climatological normal excluded from this phase's output

**Decision:** `clean_and_engineer.py` produces no climatological-normal column; that computation is deferred to per-fold computation in Phase A-2, per master §11.5.

**Rationale:** Computing and attaching a single dataset-wide climatological normal here would risk the exact leakage pattern master §11.5 was written to prevent (a fold's evaluation reference computed using data from outside that fold's training years).

---

## 11. Known Limitations Carried Forward

- NASA POWER's grid resolution vs. UP district boundary mismatch (master §11.6) is not addressed in this phase — each district is represented by its centroid point query, which is the best available resolution from this API but remains a genuine approximation, especially for larger or irregularly-shaped districts.
- Exact NASA POWER rate-limit behavior (§3.3) must be verified against current API documentation at implementation time — this document does not assert a specific numeric limit, since these are set externally and subject to change.

---

## 12. Appendix: NASA POWER Variable Mapping

| NASA POWER parameter | Project feature name(s) derived from it | Notes |
|---|---|---|
| T2M_MAX | T2M_MAX (target), T2M_MAX_lag1/3/7, T2M_MAX_roll7_mean/std | Primary target variable |
| T2M_MIN | (retained, not yet mapped to a named feature) | Candidate for a diurnal-range feature; formula pending (§10.1) |
| RH2M | Input to ATM_DRYNESS_lag1 derivation | Formula pending (§10.1) |
| WS10M | WS10M_lag1 | |
| WD10M | WD10M_SIN, WD10M_COS | Circular-encoded |
| PRECTOTCORR | PRECTOTCORR_lag1 | |
| PS | PS_lag1 | |
| ALLSKY_SFC_SW_DWN | Input to RADIATION_EFFECTIVE_lag1 derivation | Formula pending (§10.1) |
| CLRSKY_SFC_SW_DWN | Input to RADIATION_EFFECTIVE_lag1 derivation | Retained for VIF screening in Phase A-2 (master §6.1) despite prior project's now-invalidated VIF result claiming to drop it |
| CLOUD_AMT | CLOUD_AMT_lag1 | |
| GWETROOT, GWETTOP | Input to SOIL_DRYNESS_lag1 derivation | Formula pending (§10.1) |

`date` → DOY_SIN, DOY_COS (circular day-of-year encoding, not a raw API parameter).