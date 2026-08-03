# Master Project Documentation

## District-Level Heatwave Early Warning System — Uttar Pradesh, India

**Status:** Design phase — rebuild of prior system following methodology audit
**Document type:** System-level technical design specification (top-level; phase documents extend this)
**Audience:** Any engineer joining this project cold, with no prior context

---

## 1. Project Overview

### 1.1 Project Objective

Build a district-level heatwave early warning system for Uttar Pradesh, India, capable of triggering preparedness actions (health advisories, labour restrictions) ahead of extreme heat events, rather than reacting once a heatwave is already underway.

### 1.5 Expected Users

- **Primary (future, out of current scope):** Government of UP disaster management and public health units, via a dashboard/API once the classification phase and deployment phase are complete.
- **Current:** The project author and any future contributor/reviewer of this codebase — this document is written for that audience.

### 1.6 Scope (Current Phase — Phase A)

- **Geography:** 75 districts of Uttar Pradesh only.
- **Target:** `T2M_MAX` regression, 10-day direct forecast horizon.
- **Data source:** NASA POWER API, re-pulled from raw source (not reusing the prior project's opaque CSV).
- **Models:** naive baselines → seasonal-naive → GBM control → LSTM → GRU seq2seq/attention variants.
- **Evaluation:** walk-forward (expanding-window, 3-fold), RMSE, skill score vs. climatology, and formal statistical significance testing (Diebold-Mariano test).

### 1.7 Out of Scope (Current Phase)

- Spatial modeling / the additional 50 districts (Rajasthan, Delhi, MP, Bihar, Uttarakhand, Nepal) — deferred to **Phase B**.
- Binary heatwave classification and severity classification — deferred until the best Phase A regression model is selected.
- Deployment (scheduled inference, dashboard, API) — deferred until model selection is complete; architecture is anticipated in this document (§3, §14) but not built yet.
- Confidence intervals / probabilistic forecasting (quantile regression, conformal prediction) — future work, not designed yet.

---

## 2. System Goals

### 2.1 Functional Goals

- Given the trailing N-day (window length to be finalized via ablation, see §8.6) meteorological history for a district, produce a 10-day-ahead daily `T2M_MAX` forecast.
- Support incremental model comparison: every new model tier must be evaluated against the current best model using the same protocol, not a bespoke one.
- Support walk-forward evaluation across three chronological folds without requiring pipeline changes between folds (fold boundaries are config, not code).

### 2.2 Non-Functional Goals

| Goal | Definition for this project |
|---|---|
| Reproducibility | Any pipeline stage (data pull → feature engineering → windowing → training → evaluation) must be re-runnable from a config and produce the same result. No pipeline step may live only inside a notebook cell. |
| Maintainability | A new contributor should be able to add a new model tier by implementing one interface (see §10) without modifying the data or evaluation layers. |
| Reliability | Data pipeline must explicitly handle and document missing/invalid values from the NASA POWER API rather than silently passing through or silently dropping rows. |
| Scalability | The data layer must be geography-agnostic (district list is a parameter, not hardcoded), so that Phase B's expansion from 75 → 125 districts is a config change, not a rewrite. |
| Explainability | Attention-based models (Tier 4) must expose attention weights for inspection; this is a deliberate design requirement, not an incidental side effect, because the eventual government-facing use case requires interpretable justification for forecasts. |
| Performance expectation | Beat climatological-normal skill score (not just beat persistence) at every forecast horizon day, for the model to be considered to have any real skill (see §13). |

### 2.3 Reliability, Reproducibility, and Explainability as First-Class Requirements

These three are called out separately because the prior version of this project failed on exactly these axes. They are treated in this rebuild as acceptance criteria for every pipeline stage, not aspirational qualities.

---

## 3. High-Level System Architecture

### 3.1 Overall Workflow

```
NASA POWER API (raw pull, per-district daily records)
        │
        ▼
Raw Data Store
        │
        ▼
Cleaning & Missing-Value Handling
        │
        ▼
Feature Engineering (lags, rolling stats, circular encodings)
        │
        ▼
Fold-Aware Windowing (30-day input windows, geography-agnostic)
        │
        ▼
Chronological Fold Splitter (expanding window, 3 non-overlapping test blocks)
        │
        ▼
Model Training (Tier 0 → Tier 4, per fold)
        │
        ├──────────────► Evaluation Layer (RMSE, skill score, DM test, bootstrap) → Experiment Tracking (W&B)
        │
        └──────────────► Artifact Store (Kaggle Output Datasets, versioned)
                                   │
                                   ▼ (future phase)
                          Serving Layer (inference.py + FastAPI)
                                   │
                                   ▼ (future phase)
                          Dashboard / Government Consumers
```

### 3.2 Data Flow Summary

Data flows strictly left-to-right and forward-in-time. No stage downstream of the fold splitter is permitted to see data from a future fold or future date — this applies to climatological-normal computation as much as model training (see §11.5, a specific leakage risk carried over from the prior project's design).

### 3.3 Module Interaction Principle

Every module boundary in §4/§10 exists to prevent one specific failure class from the prior version: **logic must not exist only inside a notebook.** Notebooks are permitted only for exploratory analysis and for calling into `src/`, never for owning a transformation that later stages depend on.

---

## 4. Complete Project Structure (Summary)

- `configs/` — fold boundaries, district lists, hyperparameters, file paths, all as data, not inline in scripts.
- `src/data/` — must accept the district list as a parameter so Phase B's 125-district expansion doesn't require code changes.
- `train.py` — single entrypoint. Given a fold config and a model config, runs the full train → predict → evaluate cycle for that (fold, model) pair. This replaces "three separate copy-pasted notebooks" from the prior project.

---

## 5. Complete System Workflow

### 5.1 End-to-End Execution Order

`configs/*.yaml` → `src/data/*` (fetch_nasa_power.py raw pull; clean_and_engineer.py missing-value handling and features; windowing.py per-district sequences) → `train.py` (fit on fold's training window) → `src/eval/*` (RMSE, skill score vs climatology, Diebold-Mariano test vs current-best model) → W&B (log metrics, significance results, artifacts). Model + scaler persisted to `artifacts/`, mirrored to Kaggle Output.

### 5.2 Per-Fold Execution

For each of the 3 folds (§8.2), the identical sequence above runs once per model tier. The **only** things that change between runs are the config files passed to `train.py` — no notebook-specific or fold-specific code branches are permitted.

### 5.3 Hyperparameter Tuning Is a Separate, Prior Step

Hyperparameter tuning happens **once**, on a fixed reference split (1984–2010 train / 2011–2018 validation / 2019–2025 test — the original single split from the pre-rebuild design), before walk-forward evaluation begins. Walk-forward folds then evaluate the *locked* configuration across time; they do not re-tune. This is a deliberate cost-control decision (§8.3) and also keeps the walk-forward comparison honest — if hyperparameters were re-tuned per fold, differences between folds could reflect tuning luck rather than genuine time-generalization.

---

## 6. Machine Learning Architecture

### 6.4 Training Pipeline

Config-driven via `train.py` (§5). Each (fold, model tier) pair is one run. Hyperparameters are frozen prior to walk-forward (§5.3).

### 6.5 Forecasting Strategy by Tier

| Tier | Multi-step strategy | Mechanism |
|---|---|---|
| 0–3 (baselines, GBM, LSTM) | Direct | Single forward pass, 10 independent outputs from one fixed representation of the input window. |
| 4 (GRU seq2seq/attention) | Decoder-conditioned generation | A decoder generates the 10 output days sequentially, each step conditioned on the decoder's own previous hidden state (and optionally its own previous scalar output as one additional input) — never on a reconstructed exogenous feature vector. This sidesteps the recursive-forecasting blocker (§6.3) while still letting day *d*'s forecast be informed by day *d−1*'s forecast, which direct multi-step cannot do. |

### 6.6 Evaluation Pipeline

See §13. Summary: RMSE per horizon day, skill score against climatological normal (not raw variance), and Diebold-Mariano significance testing of every new tier against the current best model.

### 6.7 Experiment Tracking

Weights & Biases (free tier). Chosen over MLflow for lower setup overhead on a solo project running on ephemeral Kaggle sessions; W&B's hosted dashboard survives session termination, which is the property that matters most here (§8.4).

---

## 7. Model Evolution Roadmap

Each tier exists to test a specific hypothesis about what actually drives forecast skill — the roadmap is designed so that if a later tier wins, you can attribute *why*, not just *that* it won.

### Tier 0 — Naive Baselines
Persistence, rolling-mean, climatological normal.

### Tier 1 — Seasonal-Naive

### Tier 2 — GBM Control
Flattened-window gradient-boosted trees.

### Tier 3 — LSTM
Direct multi-step, single shared hidden-state representation.

### Tier 4 — GRU, split into three sub-tiers specifically so architecture, strategy, and attention effects are not confounded:

- **4a — GRU, direct multi-step:** Isolates GRU vs. LSTM as a pure architecture swap, holding forecasting strategy fixed.
- **4b — GRU, seq2seq, no attention:** Isolates the effect of decoder-conditioned sequential generation (§6.5) alone.
- **4c — GRU, seq2seq, with attention:** Isolates the added effect of attention over encoder states on top of 4b.
- **Motivation:** Without this split, a win at "Tier 4" would be ambiguous between three distinct causes. This is a direct correction of a structural gap in the original plan.
- **Comparison:** 4a vs. Tier 3; 4b vs. 4a; 4c vs. 4b — three separate comparisons, not one.

### Tier 4b (Optional) — Transformer / TCN
**Status:** Pending Design Decision. Only pursued if Tier 4c shows a statistically significant, practically meaningful gain over Tier 3. Not committed to in this document.

### Tier 5 — Spatially-Augmented LSTM/GRU (Phase B, 125 districts)
**Architecture:** Same recurrent core as the winning Phase A tier, with input features augmented by neighboring-district current conditions (e.g., upwind district's current T_max/anomaly).
**Motivation:** Cheap way to test whether most of the "heat propagates from Rajasthan into western UP" spatial signal is captured by simple neighbor features, before committing to full graph machinery.
**Comparison:** Against the winning Phase A tier, now evaluated on the 125-district dataset.

### Tier 6 — Spatiotemporal GNN (Phase B, 125 districts)
**Architecture:** Graph-based spatiotemporal model (e.g., Graph WaveNet / DCRNN-style), graph built from district adjacency or distance-weighted edges.
**Motivation:** Learns spatial propagation directly from data rather than hand-fed neighbor features. Only pursued once Tier 5 establishes whether that additional structure is likely to earn its cost.
**Comparison:** Against Tier 5.

---

## 8. Design Decisions

Each entry: context → alternatives considered → decision → rationale → consequences.

### 8.8 Skill Score Definition
**Decision:** Skill Score = 1 − (RMSE_model² / RMSE_climatology²), using the Tier 0 climatological-normal baseline (§7, Tier 0) as the reference, computed per fold from that fold's training years only.
**Rationale:** This reference already accounts for season, so beating it demonstrates the model is extracting real signal beyond the calendar.
**Consequences:** Raw RMSE is still reported (§13) for interpretability, but the skill score is the metric that determines whether a tier is considered to have "worked."

### 8.9 Statistical Significance: Diebold-Mariano Test, Not a Naive Paired T-Test
**Context:** Per-row daily forecast errors are autocorrelated (both across consecutive days and across horizon days within one forecast), which violates the independence assumption of a plain paired t-test on raw per-row errors; a naive t-test on ~300K correlated rows will produce artificially significant p-values for trivial improvements.
**Alternatives considered:** Paired t-test on raw per-row errors (rejected); Diebold-Mariano test on the pooled loss-differential series with Newey-West/HAC variance correction (adopted as primary); block bootstrap over year/district-year blocks (adopted as secondary cross-check).
**Decision:** DM test (HAC-corrected, lag ≈ horizon length) per horizon day, as the primary significance test for every "does Tier X beat Tier X−1" comparison; block bootstrap as a secondary sanity check.
**Rationale:** DM test is the field-standard tool for exactly this comparison problem in forecast evaluation literature and correctly accounts for the autocorrelation structure.
**Consequences:** Comparisons are run per horizon day (10 tests) for each tier transition; multiple-comparisons correction (Holm or Bonferroni) is required across those 10 tests per comparison (see §13.4).

### 8.10 Geography Scope: 75 Districts Now, Spatial Expansion to 125 Deferred to Phase B
**Context:** Full 125-district dataset (UP + neighboring states + Nepal) is needed for spatial modeling, but building spatial-model scaffolding now, before per-district sequence models are validated, would mean maintaining two data pipelines simultaneously for no near-term benefit.
**Decision:** Phase A restricted to 75 UP districts; data layer built geography-agnostic (district list is a config parameter) so Phase B is a config change, not a rewrite.
**Consequences:** Spatial signal (loo wind propagation from Rajasthan) cannot be modeled at all until Phase B; this is accepted since Phase A's purpose is establishing a validated non-spatial baseline first.

---

## 9. Project Philosophy

- **Notebooks never own a pipeline step.** Every transformation that a later stage depends on must live in `src/`. Notebooks call into `src/`; they are not the source of truth. This is the single principle most directly aimed at preventing the prior project's core failure.
- **Config over hardcoding.** Fold boundaries, district lists, hyperparameters, and file paths live in `configs/`, not inline in scripts — this is what makes walk-forward evaluation "run the same entrypoint three times with three configs" rather than three divergent code paths.
- **Validation-first, not results-first.** Every claim (feature selection rationale, "the model would have caught the 2022 event," a reported improvement number) must be backed by a specific, re-runnable computation before it appears in any documentation. This directly reverses the prior project's pattern of asserting results in prose that weren't actually computed in code.
- **Isolate variables before attributing wins.** Tier 4 is deliberately split into three checkpoints (§7) specifically so that an improvement can be attributed to a specific cause (architecture, strategy, or attention) rather than bundled ambiguously.
- **Every new model must beat a metric that accounts for season and time, not a naive absolute error.** (§8.8)
- **Geography-agnostic data layer.** Enables scaling from 75 to 125 districts without a rewrite (§8.10).
- **Explainability is designed in, not bolted on.** Attention weights are a required output of Tier 4, not an incidental artifact, because the eventual government-facing use case needs interpretable justification.

---

## 10. Repository Architecture

### 10.1 Module Boundaries and Allowed Import Directions

- `configs` can be imported by `data` and `models`, but never imports anything itself.
- `src/data/` must not import from `src/models/` or `src/eval/`.
- `src/models/` must not import from `src/eval/` (evaluation stays model-agnostic).
- `notebooks/` may import from any `src/` module but nothing may import *from* a notebook.

### 10.2 Common Model Interface

Every model tier implements the same interface (`src/models/interfaces.py`) so `train.py` and `src/eval/` never need tier-specific branches:

```python
class ForecastModel(Protocol):
    def fit(self, X_train, y_train) -> None: ...
    def predict(self, X) -> np.ndarray:  # shape (n_samples, 10)
        ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "ForecastModel": ...
```

### 10.3 Coding and Naming Conventions
**Pending Design Decision** — specific linting/formatting tooling (e.g., black, ruff) and naming conventions beyond the module-boundary rules above have not yet been decided.

---

## 11. Data Pipeline

### 11.1 Data Source
NASA POWER API, re-pulled directly (§8.7). 75 districts of Uttar Pradesh, daily resolution, 1984–2025.

### 11.2 Missing-Value Handling Policy
**Pending Design Decision — policy direction stated, exact thresholds not yet finalized.**
Direction: linear interpolation for gaps ≤3 days; longer gaps flagged and excluded rather than silently filled or silently dropped. The number of district-days requiring interpolation must be recorded and reported (this is itself useful methods content, and directly addresses the prior project's unverified "no missing values present" claim).

### 11.3 Train/Validation/Test Split
Two distinct splits serve two distinct purposes (do not conflate them):

1. **Hyperparameter tuning split** (used once, before walk-forward): train 1984–2010, validation 2011–2018, test 2019–2025.
2. **Walk-forward evaluation folds** (used to evaluate the frozen configuration across time, §8.2):

| Fold | Train | Test |
|---|---|---|
| 1 | 1984–2010 | 2011–2015 |
| 2 | 1984–2015 | 2016–2020 |
| 3 | 1984–2020 | 2021–2025 |

### 11.4 Sequence Generation
Per-district sliding window, window length pending (§8.6), 10-day direct output target. No cross-district contamination in window construction — sequences are built independently per district and concatenated only after windowing.

### 11.5 Data Leakage Prevention
- Scaler (standardization) fit only on each fold's training data, applied to that fold's validation/test data — never fit on the full dataset.
- **Climatological normal must be recomputed per fold, from that fold's training years only.** This is called out explicitly because computing it once from the full 1984–2025 range (as the prior project effectively did via a fixed 1984–2015 "baseline period" applied across a differently-dated split) risks leaking information about later years' climate into an earlier fold's baseline.
- Sequence windows never span the train/test boundary — the last training window's target days must fall entirely within the training period.

### 11.6 Assumptions
NASA POWER's per-district value (nearest grid cell or spatial average) is treated as representative of that administrative district — this is a genuine resolution mismatch (~50km reanalysis grid vs. district boundaries) that is accepted as a known limitation, not resolved in this phase.

---

## 12. Experiment Pipeline

- **Training:** `train.py`, config-driven, one run per (fold, tier) pair.
- **Validation:** Used within the single hyperparameter-tuning split only (§11.3, split 1); walk-forward folds do not re-validate/re-tune.
- **Testing:** Each fold's held-out test block, evaluated per §13.
- **Comparison:** Every tier compared against the current best model, not against Tier 0 alone, per the incremental chain established in this document.
- **Metric collection and result storage:** Logged to W&B (§8.4); model artifacts and predictions persisted to `artifacts/`, mirrored to versioned Kaggle Output Datasets given Kaggle's ephemeral sessions.
- **Visualization: Pending Design Decision** — specific plots beyond the MAE/RMSE-vs-horizon and skill-score plots inherited conceptually from the prior project are not yet specified.
- **Reproducibility:** Any (fold, tier) run must be reproducible from its config file and the versioned data artifact alone.

---

## 13. Evaluation Strategy

### 13.1 Primary Metric: RMSE per Horizon Day
Reported for every model, every fold, every horizon day (1 through 10) — not just an averaged single number, since error is known to grow with horizon and averaging can hide horizon-specific behavior.

### 13.2 Reference Metric: Skill Score vs. Climatology
`Skill Score = 1 - (RMSE_model² / RMSE_climatology²)`, per fold, per horizon day (§8.8). This is the metric that determines whether a tier has genuine skill, not just seasonal awareness.

### 13.3 Baseline Models
Tier 0's three baselines (persistence, rolling-mean, climatological normal) plus Tier 1 (seasonal-naive) — see §7. Every later tier is compared against whichever of these is strongest *at the specific horizon day* being evaluated, since persistence and climatology are expected to cross over somewhere mid-horizon.

### 13.4 Statistical Validation
Diebold-Mariano test (HAC-corrected) per horizon day, for each tier-vs-current-best comparison (§8.9), with Holm correction across the 10 per-horizon-day tests within a single tier comparison. Block bootstrap as a secondary cross-check.

### 13.5 Standing Evaluation Slice: 2022 Event
Independent of the fold scheme, a dedicated evaluation slice specifically on March–April 2022 is required for every model tier, since the 3-fold aggregate metric dilutes this single event's contribution to its containing fold (§8.2). This directly closes the gap where the prior project asserted 2022-detection capability without ever computing it.

### 13.6 Error Analysis
**Pending Design Decision** — specific error-analysis breakdowns (e.g., by season, by district, by heatwave-day vs. non-heatwave-day) beyond the 2022 slice above are not yet specified, since heatwave-conditional evaluation is explicitly deferred to the classification phase (§1.7).

---

## 14. Future Roadmap

### 14.1 Planned Work (Committed)
- Phase A: Tiers 0 through 4c, per the roadmap in §7, on 75 districts.
- Phase B: Tiers 5–6, spatial expansion to 125 districts, per §7.
- Post-regression: binary heatwave classification and severity classification, built on top of whichever Phase A/B model is selected.

### 14.2 Speculative / Not Yet Committed
- Tier 4b (Transformer/TCN) — contingent on Tier 4c's results (§7).
- Confidence intervals via quantile regression or conformal prediction.
- Extended forecast horizon beyond 10 days.
- Deployment: scheduled daily inference (e.g., GitHub Actions cron), FastAPI serving layer, dashboard (Streamlit or similar). Architecture anticipated in §4/§3 but not built in this phase.

### 14.3 Known Limitations Carried Forward
- NASA POWER's grid resolution vs. district boundary mismatch (§11.6) is unresolved and will remain a limitation through Phase B.
- The 3-fold walk-forward scheme's confound between training-set size and test-period difficulty (§8.2) is an accepted, permanent limitation of this evaluation design, not something later phases are expected to fix.

---

## 16. Glossary

| Term | Definition |
|---|---|
| T2M_MAX | Daily maximum 2-meter air temperature (°C) — the regression target. |
| Direct multi-step forecasting | Producing all forecast horizon days in a single forward pass from one fixed input representation, as opposed to recursively feeding predictions back in. |
| Recursive forecasting | Feeding a model's own prediction back in as input to predict the next time step. Rejected for this project (§6.3, §8.5) because required exogenous features at future steps are unobservable. |
| Seq2seq (decoder-conditioned generation) | A forecasting strategy where a decoder generates horizon days sequentially, each step conditioned on the decoder's own previous hidden state — not on reconstructed exogenous features. Used in Tier 4. |
| Climatological normal | The historical mean of T2M_MAX for a given district and calendar day-of-year, computed from training years only within a given fold. |
| Skill score | `1 - (RMSE_model² / RMSE_climatology²)` — the primary "did this model actually learn something beyond the season" metric. |
| Diebold-Mariano (DM) test | A statistical test for comparing forecast accuracy between two models, using the pooled loss-differential series with autocorrelation-robust (HAC/Newey-West) variance correction. Chosen over a naive paired t-test because per-row forecast errors are autocorrelated. |
| Walk-forward / expanding-window evaluation | Evaluating a model across multiple chronological folds where the training window expands forward in time and the test window shifts forward accordingly. |
| VIF (Variance Inflation Factor) | A collinearity diagnostic; a feature's VIF must be computed by regressing it against the *entire* retained feature set, not a subset — a rule the prior project violated. |
| IMD | India Meteorological Department — defines the heatwave threshold this project ultimately targets (out of scope for current phase, §1.7). |
| Phase A / Phase B | Phase A: 75-district, non-spatial modeling (this document's primary scope). Phase B: 125-district, spatial modeling (deferred). |