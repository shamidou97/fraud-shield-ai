# Fraud Shield AI — Performance Report

## 1. Summary

Six models were trained and evaluated on the Fraud Shield credit card
transaction dataset: three supervised baselines (Logistic Regression, Random
Forest, XGBoost), two deep learning models (FNN, LSTM), and a hybrid
ensemble that stacks all five as a meta-learner. All results below are
reported on the true held-out test set (`fraudTest.csv`), which no model saw
during training or validation, and which is chronologically separate from
the training period (train: Jan 2019–Jun 2020; test: Jun–Dec 2020).

**Headline result:** the LSTM was the single strongest model (F1 = 0.93),
outperforming the hybrid ensemble (F1 = 0.81) and every tabular baseline.
The meta-learner correctly identified LSTM as its most trustworthy input,
but the ensemble blend still landed below LSTM's standalone performance —
discussed in Section 4.

## 2. Final held-out test performance

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| **LSTM** | 0.96 | 0.90 | **0.93** | 1.00 |
| Hybrid Ensemble | 0.79 | 0.83 | 0.81 | 1.00 |
| Random Forest | 0.72 | 0.68 | 0.70 | 0.99 |
| XGBoost | 0.70 | 0.68 | 0.69 | 1.00 |
| FNN | 0.72 | 0.65 | 0.68 | 0.99 |
| Logistic Regression | 0.30 | 0.41 | 0.34 | 0.92 |

All metrics use a decision threshold tuned per model to maximize F1 on the
precision-recall curve, rather than a fixed 0.5 cutoff (see Section 3).
Full results, including the tuned threshold per model, are in
`final_test_results.csv`.

Baseline dataset context: fraud rate is 0.579% in train and 0.386% in test
(see `01_eda.ipynb`). A majority-class-only classifier would score ~99.4%
accuracy while catching zero fraud — accuracy is not a meaningful metric for
this problem, which is why precision/recall/F1/AUC-ROC are used throughout.

## 3. Why threshold tuning was necessary

Every model except Logistic Regression's baseline weighting used aggressive
class-imbalance correction during training — `scale_pos_weight` for
XGBoost (an official XGBoost hyperparameter for binary classification with
skewed classes), and an equivalent manually weighted binary cross-entropy
loss for the FNN and LSTM (`pos_weight ≈ 172`, matching the negative:positive
class ratio in the training data).

This correction is necessary for the models to learn from the rare fraud
class at all, but it has a side effect: predicted probabilities are pushed
upward across the board, so the conventional 0.5 decision threshold becomes
miscalibrated. At the default threshold, XGBoost's held-out precision was as
low as 0.25 despite a near-perfect AUC-ROC (~1.00) — the model's *ranking*
of transactions by fraud likelihood was excellent, but the *cutoff* used to
convert that ranking into a yes/no decision was wrong.

Each model's threshold was independently tuned by finding the point on its
own precision-recall curve that maximizes F1. AUC-ROC is threshold-independent
and unaffected by this step, which is why it stays constant between the
default-threshold and tuned-threshold evaluations — a useful internal
consistency check.

## 4. The hybrid ensemble: what it does well, and its limitation here

The meta-learner is a logistic regression trained on the five base models'
output probabilities, chosen deliberately for interpretability — its
coefficients show exactly how much the ensemble trusts each input:

| Base model | Meta-model coefficient |
|---|---|
| LSTM | 8.85 |
| Random Forest | 3.49 |
| XGBoost | 1.95 |
| FNN | 0.89 |
| Logistic Regression | -3.65 |

The ensemble correctly identifies LSTM as by far its most reliable input,
and assigns Logistic Regression (the weakest individual model) a *negative*
weight — effectively discounting or partially inverting its signal rather
than treating it as additive evidence.

Despite this sensible weighting, the ensemble's held-out F1 (0.81) falls
short of LSTM alone (0.93). This is a legitimate and explainable outcome
rather than a flaw in the ensemble logic: stacking helps most when base
models make different, complementary errors. Here, Random Forest, XGBoost,
FNN, and Logistic Regression all operate on the same snapshot-based tabular
features with no access to transaction history, so they tend to make
correlated errors with each other. LSTM's access to each cardholder's recent
transaction sequence gives it a qualitatively different and stronger signal,
one the other four models structurally cannot replicate. Averaging a strong,
differentiated model with four weaker, similar models pulls the blended
result toward the weaker group's ceiling.

**Practical recommendation:** for this dataset and feature set, LSTM alone
is the strongest deployable model. The ensemble remains valuable as a more
robust, less single-point-of-failure option in principle, and would likely
close the gap with LSTM further if the tabular models were given richer,
more differentiated features (e.g. the target-encoded high-cardinality
fields — `merchant`, `job`, `state` — noted as a future direction in
`02_feature_engineering.ipynb`).

## 5. Feature importance and interpretability

XGBoost's feature importances and SHAP values (see `03_baseline_models.ipynb`,
Section 9) show `amt_category_zscore` as the single most important feature
by a wide margin — more than 3x the next-highest feature, and far ahead of
raw `amt` itself.

This feature was engineered directly from an EDA finding: transaction
amount's relationship with fraud is not consistent across merchant
categories. In `shopping_net` and `misc_net`, fraud transactions average
roughly 80–120x larger than legitimate ones — consistent with large,
opportunistic purchases on a compromised card. In `gas_transport` and
`grocery_net`, the pattern inverts: fraud transactions are *smaller* than
legitimate ones, consistent with low-value "card testing" transactions used
to verify a stolen card is still active before a larger purchase elsewhere.
A single global amount feature cannot capture this, since the effect
direction flips by category — `amt_category_zscore` (amount normalized
against its own category's mean and standard deviation) resolves this and
became the model's dominant signal.

SHAP summary plots confirm the direction of this effect: high
`amt_category_zscore` values consistently push predictions toward fraud,
while low or negative values push toward legitimate. The per-cardholder
equivalent (`amt_zscore`) carries comparatively little importance by
comparison, indicating the category-relative framing captures meaningfully
more signal than the cardholder-relative one for this dataset.

## 6. Data notes

- **Class imbalance:** 0.579% fraud in train, 0.386% in test.
- **Temporal split:** train and test do not overlap in time (train ends
  2020-06-21 12:13:37; test begins 2020-06-21 12:14:25), ruling out
  leakage through the rolling transaction-velocity feature.
- **Monthly fraud rate** is roughly stable (0.4%–0.7%) across the training
  period aside from an elevated Jan–Feb 2019, with no strong seasonal
  pattern (e.g. no clear holiday-season spike).
- **Geo-distance:** raw `lat`/`long`/`merch_lat`/`merch_long` were dropped
  from the final feature set in favor of the derived `geo_distance_km`,
  which captures the relevant signal (cardholder-merchant distance) more
  directly and avoids multicollinearity between the coordinate pairs.

## 7. Reproducibility

- `01_eda.ipynb` — exploratory analysis and the findings that motivated
  feature engineering decisions
- `02_feature_engineering.ipynb` — feature construction, imputation,
  encoding; outputs `train_features.parquet` / `test_features.parquet`
- `03_baseline_models.ipynb` — Logistic Regression, Random Forest, XGBoost;
  threshold tuning; SHAP interpretability
- `04_deep_learning_models.ipynb` — FNN, LSTM; sequence construction with
  alignment verification against the tabular split
- `05_hybrid_ensemble.ipynb` — meta-learner training and final held-out
  test evaluation; produces `final_test_results.csv`

All train/validation splits use `random_state=42`, `test_size=0.2`,
stratified on `is_fraud`, applied identically across `03` and `04` so that
validation-row indices align for ensemble stacking.
