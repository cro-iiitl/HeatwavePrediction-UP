# Phase A — Model Development

**Extends:** `MASTER_PROJECT_DOCUMENTATION.md` — see §6.2 (Sequence Generation), §7 (Model Evolution Roadmap), §8.6 (window length pending), §11.4 (Sequence Generation), §15 (Phase Overview)
**Depends on:** `PHASE_0_DATA_PIPELINE.md` — engineered per-district data
**Status:** Tiers 0–3 complete (baselines, GBM, LSTM), walk-forward evaluation infrastructure built and validated, window length locked, both trainable tiers hyperparameter-tuned. Tier 4 (GRU sub-tiers 4a/4b/4c) not yet started.

This document combines the original Phase A-1 design specification (window-length ablation methodology) with a summary of what was actually executed across all of Phase A to date, including a significant methodological correction discovered partway through. Full numeric results and CSVs referenced throughout live in `configs/model_lstm.yaml`, `configs/model_gbm.yaml`, `docs/model_comparison_report.md`, and `scripts/kaggle_output/` / `scripts/gbm_output/` / `scripts/walk_forward_output/`.

---

# Part 1 — Phase A-1 Design Specification: Input Window-Length Ablation

## 1.1 Scope of This Phase

**In scope:**
- ACF/PACF analysis on a representative sample of districts, on the anomaly-from-climatology series (not raw `T2M_MAX`).
- Training Tier 3 (LSTM, direct multi-step) at four candidate window lengths — {10, 15, 20, 30} days — with every other hyperparameter held fixed.
- Comparing validation RMSE across the four candidates, with explicit weighting toward the later forecast horizon (see §1.5).
- Locking one window length into config for use by every subsequent tier (Tiers 2 through 6).

**Out of scope:**
- Any other hyperparameter tuning (learning rate, hidden units, dropout, etc.) — those are tuned separately, after the window length is locked. Bundling window-length search with full hyperparameter search would confound "does a longer window help" with "did this particular configuration happen to tune well."
- Evaluation of any tier other than Tier 3 — the ablation uses Tier 3 as the vehicle for the decision, but no other tier is trained in this phase.
- Walk-forward folds — this entire phase runs on the fixed hyperparameter-tuning split (1984–2010 train / 2011–2018 validation / 2019–2025 test, per master §5.3 and §11.3), not the 3-fold walk-forward scheme. Locking window length must happen before walk-forward evaluation begins, exactly as hyperparameter tuning does.

## 1.2 Why This Phase Exists

The prior project used a 30-day window with no stated justification (master §1.3, §8.6). This phase exists to replace that assumption with evidence, and — separately — to make sure whatever value is chosen is held **identical across every later tier**, since a difference in window length between tiers would confound any tier-to-tier comparison (master §6.1, §8.6).

## 1.3 Step 1 — ACF/PACF Diagnostic

**Series used:** computed on the **anomaly series** — `T2M_MAX(t) - climatological_normal(day_of_year, district)` — not the raw T2M_MAX series. Rationale: raw T2M_MAX autocorrelation is dominated by the seasonal cycle itself. ACF is retained as a secondary plot for context; PACF is the primary diagnostic.

**Output:** a written note stating the lag beyond which PACF is no longer distinguishable from noise, per sampled district, and whether the sampled districts agree. This note directly informs which of the four candidate window lengths is *expected* to win, before the ablation confirms or contradicts it.

## 1.4 Step 2 — Empirical Ablation

**Method:** Train Tier 3 (LSTM, direct multi-step, per master §6.5) four times, on the fixed reference split, with window length as the only variable. Fixed (non-ablated) hyperparameters for this step: a reasonable placeholder configuration, explicitly **not** the final locked hyperparameter set — full hyperparameter tuning happens afterward, once window length is fixed.

**Metric:** Validation RMSE, computed **per horizon day** (1 through 10) for each of the four runs — not a single averaged number.

## 1.5 Decision Rule

Given the project's stated goal of specifically improving performance near the **later forecast horizon** (day 7–10), window length is chosen using this explicit priority order:

1. **Primary:** lowest average validation RMSE across horizon days 7–10.
2. **Secondary (tie-break):** lowest average validation RMSE across the full horizon (days 1–10), if two candidates are close on the primary criterion.
3. **Tertiary (tie-break):** prefer the **shorter** window length if two candidates are statistically indistinguishable on both criteria above — shorter windows cost less compute across every subsequent tier and fold, and there's no reason to carry unnecessary sequence length forward if it isn't earning its cost.

## 1.6 Validation Strategy

1. **Reproducibility:** re-running the ablation with the same config and same random seed should produce RMSE values within expected run-to-run variance.
2. **Consistency with diagnostic:** the empirically chosen window length should be checked against the ACF/PACF diagnostic's written note — if the ablation picks a window length the diagnostic gave no reason to expect, that disagreement must be investigated and documented, not silently accepted, since it may indicate the diagnostic's linear assumption is missing something the LSTM is genuinely using (PACF is linear; a recurrent model can extract nonlinear signal from lags PACF undervalues), or alternatively that something in the ablation runs isn't behaving as intended.
3. **Downstream propagation check:** once locked, confirm the chosen window length is correctly read from config by the windowing logic for at least one other tier, so the "held identical across every tier" requirement is verified in practice, not just assumed from the config existing.

## 1.7 Known Limitations (as designed)

- PACF is a **linear** diagnostic. An LSTM is a nonlinear model and may extract usable signal from a lag that PACF shows as weak — this is precisely why the empirical ablation exists as a check on the diagnostic, not a replacement for it.
- The "representative sample" of two districts is a compromise, not a full per-district window-length search — a single window length is locked project-wide for practicality (master §8.6), even though it's plausible different districts have genuinely different optimal lookback windows. This is an accepted limitation of a project-wide single window length, not something this phase attempts to resolve per-district.

---

# Part 2 — Phase A Execution Summary (What Was Actually Done)

## 2.1 Window-Length Ablation — Executed

Two districts sampled from actual Phase 0 data (not assumed from geography alone): **Banda** (hottest annual-mean T2M_MAX, 3rd of 75) and **Kushinagar** (coolest cluster, most-humid cluster). PACF cutoff lags disagreed between the two (18 vs. 27) — documented rather than averaged.

The empirical ablation, run on the full 75-district pool, initially favored **window = 15 days**, beating 10, 20, and 30 on the placeholder-hyperparameter config. This result was later re-examined (§2.4 below) and found to be a near-tie with window = 10 once properly stress-tested — window = 15 was retained for its lower run-to-run variance, not because it empirically "won" outright. See `configs/model_lstm.yaml` for the full numeric trail.

## 2.2 Tier 0/1 Baselines — Executed

Persistence, rolling-mean, climatological-normal, and seasonal-naive were implemented as `ForecastModel`-conforming classes (`src/models/baselines.py`), refactored from an initial one-off script into the common interface so the walk-forward runner could treat every tier uniformly. All four consistently underperform Tier 2 and Tier 3, confirming both trainable tiers have genuine skill beyond seasonal awareness (master §2.2's stated performance bar).

## 2.3 Tier 2 (GBM) and Tier 3 (LSTM) — Executed, Tuned Twice

Both tiers were built, and both were tuned **twice** — first against the single fixed hyperparameter-tuning split (per master §5.3's intended sequencing), then a second time against walk-forward-averaged performance once a serious discrepancy was discovered between the two approaches (§2.4).

- **Tier 2 (GBM):** LightGBM, 10 independent models (one per horizon day), operating on a flattened window using `TREE_FEATURE_COLUMNS` (includes T2M_MAX's own lags/rolling stats plus, as of a later experiment, `lat`/`lon` — see §2.5). Final locked config and full tuning history in `configs/model_gbm.yaml`.
- **Tier 3 (LSTM):** Single-layer LSTM, direct multi-step, operating on the raw sequence using `SEQUENCE_FEATURE_COLUMNS` (deliberately excludes T2M_MAX's own lags, since the LSTM already receives the raw sequence and can learn equivalent representations internally; also deliberately excludes lat/lon after a later experiment showed it hurt LSTM performance — see §2.5). Final locked config and full tuning history in `configs/model_lstm.yaml`.

## 2.4 The Fixed-Split-vs-Walk-Forward Discrepancy — A Documented Methodological Correction

This is the single most important finding of Phase A to date, and it is documented in full in `docs/model_comparison_report.md`. Summary:

On the single fixed hyperparameter-tuning split, the tuned LSTM appeared to clearly beat GBM. When both models were then evaluated on the actual 3-fold walk-forward scheme (the project's real, intended evaluation methodology per master §11.3), **GBM won every single fold**, often by a wide margin — the LSTM's fixed-split "win" turned out to be a case of overfitting to that one specific validation window, not a genuine generalization advantage.

Both tiers were subsequently **re-tuned using walk-forward-averaged RMSE as the selection criterion** rather than single-split performance:
- GBM's walk-forward-tuned config was nearly identical to its original fixed-split winner (confirming GBM's original tuning was not overfit to one window the way LSTM's was).
- LSTM's walk-forward-tuned config (lower learning rate, more dropout) improved meaningfully over the original, but still did not close the gap to GBM.

**Current standing result: GBM (Tier 2) is the best-performing individual model tier**, confirmed via Diebold-Mariano significance testing (all differences statistically real, not noise) and a seed-variance check (GBM's own performance is highly stable across random seeds; this specific check was not repeated for LSTM, since GBM's advantage over LSTM's best-observed result already clearly exceeds any plausible seed-noise range).

## 2.5 District-Identity (lat/lon) Feature Experiment

Neither tier originally had any explicit signal for which district a sample belonged to — both were trained on all 75 districts pooled together with random cross-district batch shuffling, inferring local climate context only from the input window's own recent weather values. Adding `lat`/`lon` as explicit features was tested independently for each tier:

- **GBM:** small but consistent improvement across all 3 walk-forward folds — kept permanently in `TREE_FEATURE_COLUMNS`.
- **LSTM:** substantial, consistent **regression** across all 3 folds — reverted, not kept. Likely cause: lat/lon values are static per district but were fed in repeated at every one of the window's timesteps, diluting the gradient signal the LSTM needs to extract genuinely time-varying structure from the other features. This is a concrete illustration that the same feature-engineering decision does not transfer uniformly across architectures with different internal mechanisms for consuming static information.

## 2.6 Walk-Forward Evaluation Infrastructure — Built and Validated

`src/eval/metrics.py` (RMSE per horizon day, skill score vs. climatology), `src/eval/dm_test.py` (Diebold-Mariano test, HAC-corrected, with Holm correction across the 10 per-horizon-day tests), and `src/eval/walk_forward.py` (3-fold definitions, fold-safe window splitting so a window's target span can never cross a fold's train/test boundary) were built once and have now been exercised successfully across two full end-to-end runs, producing consistent qualitative conclusions both times. `scripts/run_walk_forward_evaluation.py` is the entrypoint that trains and evaluates all tiers across all 3 folds in one run.

**Simplification documented, not hidden:** master §13.3 specifies comparing each tier against whichever baseline is strongest *at each individual horizon day* (since baselines cross over mid-horizon). The implemented runner instead uses a simpler incremental-chain comparison (each tier vs. the immediately preceding tier in the roadmap) — this still satisfies "every tier vs. current best" but without per-horizon-day dynamic baseline re-selection.

## 2.7 Honest Open Items Going Into Tier 4

- LSTM's own seed-to-seed variance for its final walk-forward-tuned configuration was never formally quantified with a dedicated 3-seed check (unlike GBM, which was). This is an accepted, documented gap — not resolved, because GBM's advantage over LSTM's best-observed result is large enough that resolving it would not change the current tier ranking.
- Tier 4 (GRU sub-tiers 4a/4b/4c, per master §7) has not been started. Given the fixed-split-vs-walk-forward lesson learned in §2.4, Tier 4's tuning is planned to use walk-forward-averaged performance as the selection criterion **from the start**, rather than repeating the fixed-split-first pattern.
- GBM (2.3904 with lat/lon, walk-forward average) is the benchmark Tier 4 needs to beat to be considered a worthwhile addition to the project, not Tier 3's LSTM number.

See `docs/model_comparison_report.md` for the complete numeric detail behind every claim in Part 2 of this document.