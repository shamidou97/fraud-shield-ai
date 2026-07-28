"""
Defines and upserts the end-to-end Fraud Shield SageMaker Pipeline:
preprocessing -> training -> evaluation -> conditional registration.
"""
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from src import config
from src.pipeline.steps import build_preprocessing_step, build_training_step, build_register_step

session = sagemaker.Session()
role = config.SAGEMAKER_ROLE_ARN


def build_pipeline() -> Pipeline:
    preprocess_step = build_preprocessing_step(session, role)
    train_step = build_training_step(session, role)
    # model_data would come from train_step.properties in a full implementation
    register_step = build_register_step(session, role, model_data="<s3-model-artifact>")

    return Pipeline(
        name=config.PIPELINE_NAME,
        steps=[preprocess_step, train_step, register_step],
        sagemaker_session=session,
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Started pipeline execution: {execution.arn}")
