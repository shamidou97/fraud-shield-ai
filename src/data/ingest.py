"""
S3 ingestion helpers: upload the raw Kaggle dataset and download
processed outputs.
"""
import boto3
from src import config

s3 = boto3.client("s3", region_name=config.AWS_REGION)


def upload_raw_dataset(local_path: str, filename: str) -> str:
    """Upload a local CSV to the raw data prefix in S3. Returns the S3 URI."""
    key = f"{config.S3_RAW_PREFIX}/{filename}"
    s3.upload_file(local_path, config.S3_BUCKET, key)
    return f"s3://{config.S3_BUCKET}/{key}"


def download_processed(key: str, local_path: str) -> None:
    """Download a processed artifact from S3 to a local path."""
    s3.download_file(config.S3_BUCKET, f"{config.S3_PROCESSED_PREFIX}/{key}", local_path)
