"""
boto3 wrappers for running Athena queries against the transaction dataset.

Uses raw boto3 rather than the SageMaker SKLearnProcessor container to
avoid known package/version limitations in that container.
"""
import time
import boto3
from src import config

athena = boto3.client("athena", region_name=config.AWS_REGION)

DATABASE = "fraud_shield_db"
OUTPUT_LOCATION = f"s3://{config.S3_BUCKET}/athena-results/"


def run_query(query: str, database: str = DATABASE, poll_interval: float = 2.0) -> str:
    """Submit an Athena query and block until it succeeds. Returns the execution ID."""
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": OUTPUT_LOCATION},
    )
    execution_id = response["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_interval)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query {state}: {reason}")

    return execution_id
