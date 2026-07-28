"""
Central configuration for Fraud Shield AI.

Loads AWS account, region, and resource names from environment variables
(see .env.example). Import from here instead of hardcoding values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_RAW_PREFIX = "raw"
S3_PROCESSED_PREFIX = "processed"

FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "fraud-shield-features")
MODEL_PACKAGE_GROUP_NAME = os.getenv(
    "MODEL_PACKAGE_GROUP_NAME", "FraudShieldModelPackageGroup"
)
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "fraud-shield-endpoint")

PIPELINE_NAME = "FraudShieldPipeline"
