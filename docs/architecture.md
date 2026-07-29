# Architecture

This document describes the Fraud Shield AI system as actually implemented,
expanding on the AWS component table in the main [README](../README.md).
See [performance_report.md](performance_report.md) for model results.

## 1. End-to-end data flow

```
fraudTrain.csv / fraudTest.csv (Kaggle)
        |
        v
  data/raw/ (local) ---------------------> S3 raw prefix (AWS, planned)
        |
        v
02_feature_engineering.ipynb
  - geo-distance, transaction velocity, per-cardholder amount z-score,
    per-category amount z-score, one-hot encoding
        |
        v
  data/processed/{train,test}_features.parquet
        |
        +----------------------+----------------------+
        v                      v                       v
03_baseline_models         04_deep_learning        (shared: SMOTE /
  LogReg, RF, XGBoost        FNN, LSTM                class weighting
  (scale_pos_weight)         (weighted BCE loss)       per src/features/
        |                      |                       imbalance.py)
        v                      v
  data/processed/models/*.joblib, *.pt, *_val_predictions.parquet
        |                      |
        +----------+-----------+
                    v
        05_hybrid_ensemble.ipynb
          - stacks 5 base-model probabilities
          - logistic regression meta-learner
          - final evaluation on held-out fraudTest.csv
                    |
                    v
        data/processed/models/meta_model.joblib
        docs/final_test_results.csv
                    |
                    v
        app/inference_client.py (local hybrid ensemble inference)
                    |
                    v
        app/streamlit_app.py (upload CSV -> predictions -> download)
```

## 2. Current implementation state

**Fully implemented and working:**
- Feature engineering pipeline (`src/features/engineering.py`,
  `src/features/imbalance.py`)
- Five trained models: Logistic Regression, Random Forest, XGBoost, FNN, LSTM
- Hybrid ensemble meta-learner, evaluated on the true held-out test set
- Local inference (`app/inference_client.py`) running the full ensemble
  (including LSTM sequence reconstruction) against new transaction data
- Streamlit web app (`app/streamlit_app.py`) for CSV upload, adjustable
  decision threshold, and downloadable results

**Scaffolded but not yet wired to real infrastructure:**
- `src/pipeline/pipeline_definition.py`, `src/pipeline/steps.py` — SageMaker
  Pipeline definition exists but references placeholder image URIs
  (`<processing-image-uri>`, `<training-image-uri>`, `<inference-image-uri>`)
- `src/pipeline/deploy.py` — reads from Model Registry and deploys a
  real-time endpoint, but no model has been registered to AWS yet; all
  training/evaluation so far has run locally against local parquet files
- `infra/cdk/stacks/endpoint_stack.py` — autoscaling endpoint stack exists
  but references a placeholder `model_package_arn`
- `src/features/feature_store.py` — Feature Store ingestion functions exist
  but haven't been exercised against a real AWS account/role

In short: the ML pipeline (data -> features -> models -> ensemble -> local
app) is real and complete. The AWS deployment path (S3 -> SageMaker Pipelines
-> Model Registry -> autoscaling endpoint) is designed and scaffolded per
the component table below, but running it end-to-end against a live AWS
account is the remaining gap between this project and a production
deployment.

## 3. Why a hybrid hierarchy: supervised + deep learning + ensemble

The three supervised baselines (LogReg, RF, XGBoost) work directly on
per-transaction tabular features -- no memory of prior transactions beyond
the engineered `txn_velocity` and z-score features. XGBoost was the
strongest of the three (see performance report), and its SHAP values showed
`amt_category_zscore` -- an engineered feature, not a raw dataset column --
as the single most predictive input.

The deep learning branch adds a second axis: the LSTM consumes each
cardholder's trailing 5-transaction sequence (built via `groupby().shift()`
in `04_deep_learning_models.ipynb`), giving it access to genuine sequential
history the tabular models cannot see directly. This turned out to matter a
great deal -- LSTM was the strongest individual model on the held-out test
set, ahead of every tabular baseline and the ensemble itself.

The meta-learner (logistic regression over the five base models' output
probabilities) was chosen for interpretability over a black-box stacker.
Its learned coefficients (see performance report, Section 4) confirm it
correctly identifies LSTM as its most trustworthy input and assigns the
weakest model (Logistic Regression) a negative weight -- but the blended
ensemble still falls short of LSTM alone, since the other four base models
share correlated blind spots that LSTM does not have. This is documented as
a genuine finding rather than smoothed over: for this dataset and feature
set, LSTM is the strongest deployable model.

## 4. Class imbalance handling

Fraud is rare (0.58% of training transactions, 0.39% of test). Every model
except the baseline Logistic Regression relies on a form of cost-sensitive
training:

- **XGBoost:** `scale_pos_weight` = negative:positive class ratio, an
  official XGBoost hyperparameter for imbalanced binary classification
- **FNN / LSTM:** a manually weighted binary cross-entropy loss
  (`weighted_bce` in `04_deep_learning_models.ipynb`), using the same
  negative:positive ratio, since PyTorch's built-in `pos_weight` argument
  expects raw logits rather than the sigmoid outputs these models produce
- **Random Forest / Logistic Regression:** scikit-learn's built-in
  `class_weight='balanced'`

This correction is necessary for the models to learn from the minority
class at all, but it pushes predicted probabilities upward across the
board, making the default 0.5 decision threshold miscalibrated. Every
model's threshold was independently tuned to the F1-maximizing point on
its own precision-recall curve rather than left at 0.5 -- documented in
detail in the performance report, Section 3.

## 5. AWS component mapping (planned production path)

See the AWS component table in the [README](../README.md) for the full
service list. In brief: S3 for raw/processed storage, Athena for querying,
SageMaker Feature Store for engineered features, SageMaker Pipelines to
orchestrate preprocessing -> training -> evaluation -> registration,
SageMaker Model Registry for versioning, a SageMaker real-time endpoint
with autoscaling for inference, and SageMaker Clarify for the bias/SHAP
interpretability work already prototyped locally in `03_baseline_models.ipynb`.

The local artifacts this project produced (`*.joblib`, `*.pt`,
`meta_model.joblib`) map directly onto this path: they are the same model
objects that `src/pipeline/steps.py`'s `ModelStep` would register, and the
same ensemble logic in `app/inference_client.py` is what would run inside
a SageMaker inference container behind the endpoint in
`infra/cdk/stacks/endpoint_stack.py`.
