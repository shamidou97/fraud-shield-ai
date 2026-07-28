#!/usr/bin/env python3
"""
CDK app entry point for Fraud Shield AI infrastructure:
the SageMaker endpoint + autoscaling stack and the supporting IAM stack.
"""
import aws_cdk as cdk
from stacks.endpoint_stack import EndpointStack
from stacks.iam_stack import IamStack

app = cdk.App()

env = cdk.Environment(account="308436492030", region="us-east-1")

iam_stack = IamStack(app, "FraudShieldIamStack", env=env)
EndpointStack(app, "FraudShieldEndpointStack", env=env, execution_role=iam_stack.sagemaker_role)

app.synth()
