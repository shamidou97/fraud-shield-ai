"""Smoke tests for pipeline step construction (requires AWS credentials to run for real)."""
import pytest


@pytest.mark.skip(reason="Requires live AWS session and role ARN; run in an AWS-configured environment")
def test_build_pipeline_runs():
    from src.pipeline.pipeline_definition import build_pipeline
    pipeline = build_pipeline()
    assert pipeline.name == "FraudShieldPipeline"
