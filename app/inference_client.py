"""
Client for calling the deployed SageMaker endpoint from the web app.
"""
import json
import boto3
import pandas as pd
from src import config

runtime = boto3.client("sagemaker-runtime", region_name=config.AWS_REGION)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    payload = df.to_json(orient="records")
    response = runtime.invoke_endpoint(
        EndpointName=config.ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )
    predictions = json.loads(response["Body"].read())
    df = df.copy()
    df["is_fraud_pred"] = predictions["predictions"]
    return df
