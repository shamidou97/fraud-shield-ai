"""
IAM roles and policies for the Fraud Shield pipeline and endpoint.
"""
from aws_cdk import Stack
from aws_cdk import aws_iam as iam
from constructs import Construct


class IamStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.sagemaker_role = iam.Role(
            self,
            "FraudShieldSageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
