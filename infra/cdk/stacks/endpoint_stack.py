"""
SageMaker real-time endpoint with application autoscaling, deployed via CDK.

Mirrors the HeartDiseaseModelPackageGroup autoscaling setup, adapted for
the Fraud Shield hybrid ensemble model.
"""
from aws_cdk import Stack
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_applicationautoscaling as appscaling
from constructs import Construct


class EndpointStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, execution_role, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        model = sagemaker.CfnModel(
            self,
            "FraudShieldModel",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                model_package_name="<model-package-arn-from-registry>",
            ),
        )

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "FraudShieldEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                    initial_instance_count=1,
                    instance_type="ml.m5.xlarge",
                )
            ],
        )

        endpoint = sagemaker.CfnEndpoint(
            self,
            "FraudShieldEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
            endpoint_name="fraud-shield-endpoint",
        )

        scalable_target = appscaling.CfnScalableTarget(
            self,
            "FraudShieldScalableTarget",
            max_capacity=4,
            min_capacity=1,
            resource_id=f"endpoint/{endpoint.endpoint_name}/variant/AllTraffic",
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            service_namespace="sagemaker",
        )

        appscaling.CfnScalingPolicy(
            self,
            "FraudShieldScalingPolicy",
            policy_name="FraudShieldInvocationsPerInstance",
            policy_type="TargetTrackingScaling",
            scaling_target_id=scalable_target.ref,
            target_tracking_scaling_policy_configuration={
                "targetValue": 1000.0,
                "predefinedMetricSpecification": {
                    "predefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                },
                "scaleInCooldown": 300,
                "scaleOutCooldown": 60,
            },
        )
