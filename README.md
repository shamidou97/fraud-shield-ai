# Fraud Shield AI

Real-time credit card fraud detection system combining supervised learning (Logistic Regression, Random Forest, XGBoost) and deep learning (FNN, LSTM) into a hybrid ensemble, built on AWS.

## Overview

Fraud Shield analyzes individual credit card transactions to flag fraud, surface anomalies, and explain its decisions. It is built as a deployable, end-to-end pipeline rather than a collection of notebooks — data prep, training, evaluation, and serving are all automated and reproducible.

**Dataset:** [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle) — ~1.3M synthetic transaction records.

## Pipeline stages

1. **Data ingestion** — raw CSV landed in S3
2. **EDA & feature engineering** — Athena queries + engineered features (transaction velocity, merchant/cardholder geo-distance, spending patterns) written to SageMaker Feature Store
3. **Class imbalance handling** — SMOTE / undersampling / cost-sensitive weighting
4. **Model training** — supervised branch (LogReg, RF, XGBoost) and deep learning branch (FNN, LSTM) trained in parallel
5. **Hybrid ensemble** — meta-model combines both branches' outputs
6. **Evaluation & interpretability** — Precision, Recall, F1, AUC-ROC, SHAP
7. **Model registry & deployment** — SageMaker Model Registry → real-time autoscaling endpoint
8. **Web app** — Streamlit/Gradio dashboard for uploads, results, and monitoring

## AWS components

| Component | AWS Service | Purpose |
|---|---|---|
| Raw data storage | **S3** | Landing zone for raw transaction CSVs and processed datasets |
| Data querying | **Athena** | SQL queries over raw/curated data for EDA and feature aggregation |
| Data catalog | **Glue Data Catalog** | Schema/table registry so Athena and Feature Store stay in sync |
| Feature storage | **SageMaker Feature Store** | Online/offline store for engineered features (velocity, geo-distance, spend profile) |
| Preprocessing & training | **SageMaker Processing / Training Jobs** | SMOTE/undersampling jobs, model training for LogReg/RF/XGBoost/FNN/LSTM |
| Hyperparameter tuning | **SageMaker Automatic Model Tuning** | Grid/random/Bayesian search across model configs |
| Orchestration | **SageMaker Pipelines** | Chains preprocessing → training → evaluation → registration into one DAG |
| Model versioning | **SageMaker Model Registry** | Tracks model versions, approval status, lineage |
| Real-time inference | **SageMaker Endpoint (autoscaling)** | Low-latency scoring for the web app / API |
| Explainability | **SageMaker Clarify** | Bias metrics and SHAP-based feature attribution |
| Infrastructure as code | **CDK / CloudFormation** | Reproducible deployment of endpoints, autoscaling policies, IAM roles |
| Monitoring | **SageMaker Model Monitor / CloudWatch** | Data drift detection, endpoint metrics, alarms |
| Web front end | **Streamlit / Gradio (on EC2, App Runner, or ECS)** | Upload transactions, view fraud results, dashboards |
| Access control | **IAM** | Least-privilege roles for pipeline execution, endpoint access, and app-to-endpoint calls |

## Project structure

```
fraud-shield-ai/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                        # original Kaggle CSVs (not committed)
│   └── processed/                  # cleaned/feature-engineered outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_deep_learning_models.ipynb
│   └── 05_hybrid_ensemble.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                   # AWS account/region/role config
│   ├── data/
│   │   ├── ingest.py                # S3 upload/download helpers
│   │   └── athena_queries.py        # boto3 Athena query wrappers
│   ├── features/
│   │   ├── engineering.py           # velocity, geo-distance, spend-profile features
│   │   ├── imbalance.py             # SMOTE / undersampling / class weights
│   │   └── feature_store.py         # Feature Store create/ingest/query
│   ├── models/
│   │   ├── supervised.py            # LogReg, Random Forest, XGBoost
│   │   ├── deep_learning.py         # FNN, LSTM (PyTorch/TensorFlow)
│   │   ├── ensemble.py              # hybrid ensemble / stacking logic
│   │   └── evaluate.py              # metrics, SHAP, evaluation reports
│   └── pipeline/
│       ├── pipeline_definition.py   # SageMaker Pipeline DAG
│       ├── steps.py                 # ProcessingStep/TrainingStep/ModelStep defs
│       └── deploy.py                # Model Registry -> endpoint deployment
├── infra/
│   ├── cdk/
│   │   ├── app.py
│   │   └── stacks/
│   │       ├── endpoint_stack.py    # SageMaker endpoint + autoscaling
│   │       └── iam_stack.py         # execution roles, policies
│   └── cloudformation/
│       └── pipeline-stack.yaml
├── app/
│   ├── streamlit_app.py             # web interface (upload, results, dashboards)
│   └── inference_client.py          # calls the SageMaker endpoint
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
└── docs/
    ├── architecture.md
    ├── user_manual.md
    └── performance_report.md
```

## Tools & technologies

- **Language:** Python
- **ML/DL:** Scikit-learn, XGBoost, PyTorch, TensorFlow
- **AWS:** S3, Athena, Glue, SageMaker (Feature Store, Pipelines, Training, Model Registry, Endpoints, Clarify, Model Monitor), CDK/CloudFormation, IAM, CloudWatch
- **Data/EDA:** Pandas, Matplotlib, Seaborn
- **Front end:** Streamlit / Gradio
- **Explainability:** SHAP, SageMaker Clarify

## Setup (local)

```bash
git clone <repo-url>
cd fraud-shield-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set AWS_ACCOUNT_ID, AWS_REGION, SAGEMAKER_ROLE_ARN
```

## Running the pipeline

```bash
python -m src.pipeline.pipeline_definition   # define and upsert the SageMaker Pipeline
python -m src.pipeline.deploy                 # register approved model + deploy endpoint
streamlit run app/streamlit_app.py            # launch the web interface
```

## Evaluation metrics

Reported per model and for the hybrid ensemble: Precision, Recall, F1-score, AUC-ROC, plus SHAP-based feature attribution for interpretability.
