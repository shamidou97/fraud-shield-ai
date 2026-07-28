"""
Deploys the latest approved model from the Model Registry to a
real-time SageMaker endpoint with autoscaling.
"""
import boto3
import sagemaker
from sagemaker import ModelPackage
from src import config

sm_client = boto3.client("sagemaker", region_name=config.AWS_REGION)
session = sagemaker.Session()


def get_latest_approved_model_package() -> str:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response["ModelPackageSummaryList"]
    if not packages:
        raise RuntimeError("No approved model package found.")
    return packages[0]["ModelPackageArn"]


def deploy_endpoint(instance_type: str = "ml.m5.xlarge", initial_instance_count: int = 1):
    model_package_arn = get_latest_approved_model_package()
    model = ModelPackage(
        role=config.SAGEMAKER_ROLE_ARN,
        model_package_arn=model_package_arn,
        sagemaker_session=session,
    )
    model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=config.ENDPOINT_NAME,
    )
    print(f"Deployed endpoint: {config.ENDPOINT_NAME}")
    # Autoscaling policy is configured separately via infra/cdk/stacks/endpoint_stack.py


if __name__ == "__main__":
    deploy_endpoint()
