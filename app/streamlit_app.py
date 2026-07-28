"""
Streamlit web app: upload transactions, view fraud predictions, and
monitor fraud trends via the deployed SageMaker endpoint.
"""
import streamlit as st
import pandas as pd
from app.inference_client import predict_batch

st.set_page_config(page_title="Fraud Shield AI", layout="wide")
st.title("Fraud Shield AI")
st.caption("Upload transactions to flag potential fraud in real time.")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} transactions.")

    if st.button("Run fraud detection"):
        with st.spinner("Scoring transactions..."):
            results = predict_batch(df)
        st.subheader("Results")
        st.dataframe(results)

        fraud_count = int(results["is_fraud_pred"].sum())
        st.metric("Flagged as fraud", fraud_count)
