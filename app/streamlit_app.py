"""
Streamlit web app: upload transactions, view fraud predictions, and
monitor fraud trends via the local hybrid ensemble (see
app/inference_client.py -- swap to a deployed SageMaker endpoint later
without changing this file, since predict_batch's signature stays the same).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from app.inference_client import predict_batch

st.set_page_config(page_title="Fraud Shield AI", layout="wide")
st.title("Fraud Shield AI")
st.caption("Upload transactions to flag potential fraud using the hybrid ensemble model.")

with st.expander("Required columns"):
    st.write(
        "The CSV must include: `cc_num`, `event_time`, `amt`, `city_pop`, "
        "`geo_distance_km`, `txn_velocity`, `amt_zscore`, `amt_category_zscore`, "
        "one-hot `category_*` columns, and `gender_F`/`gender_M`. This matches "
        "the output of `02_feature_engineering.ipynb`."
    )

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
threshold = st.slider(
    "Decision threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Transactions with fraud probability at or above this value are flagged. "
         "See docs/performance_report.md for the F1-optimal thresholds found per model.",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} transactions.")

    if st.button("Run fraud detection"):
        with st.spinner("Scoring transactions..."):
            try:
                results = predict_batch(df, threshold=threshold)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        fraud_count = int(results["is_fraud_pred"].sum())

        col1, col2 = st.columns(2)
        col1.metric("Transactions scored", len(results))
        col2.metric("Flagged as fraud", fraud_count)

        st.subheader("Results")
        display_cols = [c for c in results.columns if c not in ("fraud_probability", "is_fraud_pred")]
        display_cols += ["fraud_probability", "is_fraud_pred"]
        st.dataframe(
            results[display_cols].sort_values("fraud_probability", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download results as CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="fraud_shield_results.csv",
            mime="text/csv",
        )
