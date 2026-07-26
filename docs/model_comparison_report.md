# CRO-Heatwave: Model Comparison Report — Phase A

## Purpose

This document records how each model tier (Tier 0/1 baselines, Tier 2 GBM,
Tier 3 LSTM) was built, tuned, and evaluated, and explains the full
reasoning behind the final tier ranking. It also documents a real
methodological correction made mid-project: an initial comparison
based on a single fixed validation split gave a misleading result,
which was caught and fixed by re-evaluating everything on proper
walk-forward validation. That correction — not just the final numbers
— is the most important part of this record.

---

## 1. Model Tiers

### Tier 0/1 — Naive Baselines

Four baselines, implementing no learned parameters:

- **Persistence**: predicts every horizon day as equal to the last
  observed `T2M_MAX` in the input window.
- **Rolling mean**: predicts every horizon day as the mean `T2M_MAX`
  over the trailing 7 days of the input window.
- **Climatological normal**: predicts each horizon day using the
  historical mean `T2M_MAX` for that specific (month, day), computed
  from training years only (no leakage from future years).
- **Seasonal-naive**: predicts each horizon day using the exact value
  from the same calendar date one year prior.

These exist to answer one question: does either "real" model (GBM,
LSTM) actually beat a naive, no-learning approach? Without this check,
a low RMSE number is meaningless — it could just reflect the season
being predictable, not any genuine model skill.

### Tier 2 — GBM (LightGBM)

Ten independent LightGBM regressor models, one per forecast horizon
day (a direct multi-step approach — each day gets its own specialized
model rather than one model outputting a shared vector). Operates on
a **flattened window**: the 15-day input window is reshaped into one
flat feature vector per sample, since GBM has no internal concept of
sequence order.

Feature set includes `T2M_MAX`'s own historical lags and rolling
statistics (`T2M_MAX_lag1/3/7`, `T2M_MAX_roll7_mean/std`) — this is
necessary for GBM specifically, since it has no memory of its own and
must be handed explicit historical summary statistics to have any
access to trend/persistence information at all.

### Tier 3 — LSTM

A single-layer LSTM followed by a small dense head, producing all 10
horizon days in one forward pass (direct multi-step, sharing one
learned hidden-state representation across all outputs). Operates on
the **raw sequence** of the 15-day input window — unlike GBM, the LSTM
does not need `T2M_MAX`'s own engineered lags/rolling stats as
separate input features, since it already receives the raw sequence
of past `T2M_MAX` values directly, timestep by timestep, and can learn
equivalent (or better) temporal representations internally. Including
those same engineered lag features for the LSTM would be redundant —
this is why GBM and LSTM deliberately use two different feature sets
(`TREE_FEATURE_COLUMNS` vs. `SEQUENCE_FEATURE_COLUMNS`), not the same
one.

---

## 2. Evaluation Methodology

### The metric

RMSE (root mean squared error) computed independently for each of the
10 forecast horizon days, then averaged specifically over **days 7–10**
as the primary comparison metric — not an average over all 10 days.
This reflects the project's actual goal: a 10-day-ahead early-warning
system's value is concentrated in the later part of the horizon, where
forecast error has had the most room to accumulate and where advance
warning matters most.

### Two very different validation approaches were used — and this is the
### central story of this document

**Approach A — Fixed single split** (used initially for all
hyperparameter tuning): train on 1984–2010, tune/validate on
2011–2018. Every hyperparameter decision (window length, LSTM
architecture, GBM tree parameters) was originally locked using only
this one split.

**Approach B — Walk-forward evaluation** (the project's actual,
intended evaluation methodology): three separate, non-overlapping test
periods, each with an expanding training window:

| Fold | Train | Test |
|---|---|---|
| 1 | 1984–2010 | 2011–2015 |
| 2 | 1984–2015 | 2016–2020 |
| 3 | 1984–2020 | 2021–2025 |

A model is trained fresh for each fold (on that fold's training
years only) and evaluated on that fold's held-out test years, which
it has never seen. The three folds' RMSE values are then averaged.
This is a fundamentally harder, more honest test: a model that performs
well by chance on one specific window (Approach A) will not
necessarily perform well across three separate, genuinely different
multi-year periods (Approach B) unless it has learned something
truly general, not idiosyncratic to one window.

---

## 3. What Went Wrong With Approach A — and How It Was Caught

Under Approach A (single fixed split), extensive tuning was performed
on the LSTM: a two-stage grid search (hidden units × learning rate,
then dropout × dense layer size) selected `hidden_units=32,
learning_rate=0.0003, dropout=0.2, dense_units=32` as the winning
configuration, achieving RMSE(7–10) = **2.2954** on the 2011–2018
validation window — clearly beating GBM's best result on the same
split, **2.3388**.

Based on this, the LSTM appeared to be the stronger model.

**When both models were then run through proper walk-forward
evaluation** (training and testing on the three folds above, using
these same "winning" configurations), the result inverted entirely:

| Tier | Fold 1 | Fold 2 | Fold 3 | Average |
|---|---|---|---|---|
| GBM (Approach-A config) | 2.4030 | 2.3419 | 2.4480 | **2.3976** |
| LSTM (Approach-A config) | 2.6278–2.8113* | 2.6368–2.7059* | 2.8988–3.1054* | **2.72–2.90*** |

*LSTM's fixed-split-tuned config was re-run twice locally (different
random seeds, since no seed was fixed in the original walk-forward
script); results varied by run but were consistently, substantially
worse than GBM in every fold, every time.

**GBM beat the "winning" LSTM configuration in every single fold**,
despite having lost to it on the original fixed split. This is a
direct demonstration of a specific, well-known failure mode: **a
flexible model (LSTM) tuned tightly against one validation window can
overfit to that window's particular characteristics**, achieving a
misleadingly strong score there while failing to generalize to
genuinely different time periods. GBM's simpler, tree-based structure,
combined with its explicit lag/rolling-statistic features (which
encode temporal structure directly rather than requiring it to be
learned), proved substantially more robust across different multi-year
periods.

This was caught specifically because the project's evaluation plan
called for walk-forward validation as the real methodology, with the
fixed split intended only for locking hyperparameters efficiently
before the more expensive walk-forward run. Approach A was never meant
to be the final word — and this incident is the concrete
demonstration of why that distinction matters.

---

## 4. The Correction — Re-Tuning Both Models Against Walk-Forward Validation

To ensure a fair comparison, both GBM and LSTM were then **re-tuned
using walk-forward-averaged RMSE(7-10) as the selection criterion**,
not single-split performance. For each candidate hyperparameter
configuration, the model was trained and evaluated independently on
all three folds, and the average RMSE(7–10) across folds became the
score used to pick a winner.

### GBM — walk-forward retuning result

Five configurations tested (varying `n_estimators`, `max_depth`,
`learning_rate`, `num_leaves`):

| n_estimators | max_depth | learning_rate | num_leaves | Fold 1 | Fold 2 | Fold 3 | Average |
|---|---|---|---|---|---|---|---|
| 300 | 5 | 0.03 | 31 | 2.3994 | 2.3405 | 2.4444 | **2.3948** (best) |
| 200 | 5 | 0.05 | 31 | 2.4030 | 2.3419 | 2.4480 | 2.3976 (original) |
| 200 | 6 | 0.05 | 63 | 2.4101 | 2.3459 | 2.4580 | 2.4047 |
| 150 | 3 | 0.05 | 15 | 2.4118 | 2.3661 | 2.4520 | 2.4100 |
| 100 | 4 | 0.05 | 15 | 2.4192 | 2.3685 | 2.4427 | 2.4101 |

The gap between the best and the original single-split-tuned config
is tiny (0.0028) — well within noise. **GBM's original configuration
was already close to optimal for walk-forward generalization**,
confirming that GBM's tuning was not overfit to the single split in
the same way LSTM's was. The original config
(`n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=31`)
is retained rather than switching to a marginally better but more
expensive alternative.

### LSTM — walk-forward retuning result

Five configurations tested (varying `hidden_units`, `dropout`,
`learning_rate`):

| hidden_units | dropout | learning_rate | dense_units | Fold 1 | Fold 2 | Fold 3 | Average |
|---|---|---|---|---|---|---|---|
| 32 | 0.3 | 0.0001 | 32 | 2.7303 | 2.4640 | 2.7580 | **2.6507** (best) |
| 32 | 0.2 | 0.0003 | 32 | 2.8874 | 2.6045 | 2.6267 | 2.7062 (original) |
| 16 | 0.2 | 0.0003 | 32 | 2.8610 | 2.6315 | 2.8839 | 2.7921 |
| 32 | 0.4 | 0.0003 | 32 | 3.4798 | 3.0887 | 3.2292 | 3.2659 |
| 16 | 0.4 | 0.0003 | 32 | 3.4549 | 3.3285 | 3.3941 | 3.3925 |

A lower learning rate (0.0001 vs. the original 0.0003) combined with
slightly higher dropout (0.3 vs. 0.2) produced a real, if modest,
improvement in walk-forward generalization (2.6507 vs. 2.7062).
Configurations that reduced model capacity further (`hidden_units=16`)
or increased dropout more aggressively (0.4) performed **worse**, not
better — meaning the original hypothesis ("the LSTM overfit because it
had too much capacity/too little regularization") was only partially
correct. Simply shrinking or heavily regularizing the model does not
close the gap to GBM.

---

## 5. Final, Walk-Forward-Validated Result

Confirmed independently on two separate runs — a Kaggle GPU tuning
search (5 configs per tier, selecting the best) and a full local
end-to-end walk-forward evaluation re-run using the selected configs
— with consistent conclusions both times:

| Tier | Configuration | Avg RMSE (7–10), Kaggle search | Avg RMSE (7–10), local confirmation run |
|---|---|---|---|
| **GBM (Tier 2)** | n_estimators=200, max_depth=5, lr=0.05, num_leaves=31 | 2.3976 | **2.3976** |
| LSTM (Tier 3) | hidden_units=32, dropout=0.3, lr=0.0001, dense_units=32 | 2.6507 | 2.6385 |

(The small LSTM discrepancy, 2.6507 vs. 2.6385, is expected run-to-run
training noise — no random seed was fixed for this confirmation run —
and is consistent with the seed-variance behavior already characterized
in Phase A-1. It does not change the conclusion.)

### Full local confirmation run — all tiers, all folds

| Tier | Fold 1 | Fold 2 | Fold 3 | Average |
|---|---|---|---|---|
| GBM | 2.4030 | 2.3419 | 2.4480 | **2.3976** |
| LSTM | 2.5269 | 2.6807 | 2.7079 | 2.6385 |
| Climatological normal | 2.9924 | 2.8615 | 3.0219 | 2.9586 |
| Rolling mean | 3.0624 | 3.0588 | 3.0687 | 3.0633 |
| Persistence | 3.1614 | 3.1760 | 3.1495 | 3.1623 |
| Seasonal-naive | 3.9460 | 4.0237 | 4.3525 | 4.1074 |

**Diebold-Mariano significance (HAC-corrected, Holm-corrected across
the 10 horizon-day tests within each comparison)**: at every fold, GBM
was significantly different from the best available baseline (10/10
horizon days significant), and LSTM was significantly different from
GBM (10/10 horizon days significant) — in GBM's favor, per the RMSE
values above. These are not marginal, noise-sized gaps; the DM tests
confirm the differences are statistically real at every single horizon
day, in every fold.

**GBM is the stronger model on this dataset, at this data volume, with
this feature engineering — and this conclusion holds up under proper
walk-forward validation, confirmed on two independent runs, not just a
single split.** The gap (roughly 0.24–0.25 RMSE) is substantial and
consistent, not a marginal or noise-sized difference.

This does not mean LSTMs are unsuitable for time-series forecasting in
general — it means that, for this specific problem (daily maximum
temperature, 75 districts, ~40 years of daily data, 15-day input
windows), a well-engineered gradient-boosted tree model with explicit
lag/rolling-statistic features currently generalizes better than a
compact single-layer LSTM relying on learned temporal representations
from raw sequences. Plausible contributing factors:

- GBM's explicit lag features encode robust temporal structure
  directly; the LSTM must learn equivalent structure from data, which
  may require more training data or a different architecture to match.
- Tree-based models are structurally less prone to overfitting a
  specific validation window than a flexible recurrent architecture
  tuned tightly against it.
- The problem may not require long-range sequence modeling to the
  degree that would favor LSTM's architectural strengths — consistent
  with the earlier finding (Phase A-1 window-length ablation) that
  very short input windows (10–15 days) already performed as well as
  or better than longer ones.

---

## 6. Honest Limitations of This Comparison

- GBM's walk-forward search covered 5 configurations across a fairly
  narrow hyperparameter range; a wider search was not pursued, since
  results across all 5 configs were already tightly clustered
  (plateaued), suggesting further search would yield diminishing
  returns.
- LSTM's walk-forward search also covered only 5 configurations,
  targeted specifically at the capacity/regularization hypothesis. A
  broader architectural search (e.g., different window lengths
  re-examined jointly with hyperparameters, multi-layer LSTMs, or
  different feature engineering) was not attempted and could plausibly
  change this outcome — this comparison should be read as "GBM beats
  this specific LSTM design," not "GBM is provably superior to any
  possible LSTM design for this problem."
- No random seed was fixed during the original walk-forward runs that
  first revealed this problem, so exact LSTM numbers varied slightly
  run to run (documented in Section 3) — the qualitative conclusion
  (GBM wins every fold, by a wide margin) was consistent across
  multiple runs, but exact RMSE values for the original config should
  be read as approximate.