"""
SageMaker Pipeline step definitions: ProcessingStep, TrainingStep, ModelStep.
"""
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from src import config


def build_preprocessing_step(session, role) -> ProcessingStep:
    """Feature engineering + imbalance handling as a SageMaker Processing step."""
    processor = ScriptProcessor(
        image_uri="<processing-image-uri>",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )
    return ProcessingStep(
        name="PreprocessAndEngineerFeatures",
        processor=processor,
        code="src/features/engineering.py",
    )


def build_training_step(session, role) -> TrainingStep:
    """Trains the supervised + deep learning branches (invoked via separate entry points)."""
    estimator = Estimator(
        image_uri="<training-image-uri>",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )
    return TrainingStep(name="TrainFraudModels", estimator=estimator)


def build_register_step(session, role, model_data) -> ModelStep:
    """Registers the approved hybrid ensemble model in the Model Registry."""
    model = Model(
        image_uri="<inference-image-uri>",
        model_data=model_data,
        role=role,
        sagemaker_session=session,
    )
    return ModelStep(
        name="RegisterFraudModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=config.MODEL_PACKAGE_GROUP_NAME,
        ),
    )
