"""
SageMaker Feature Store helpers: create the feature group and ingest
engineered features for online/offline use.
"""
import time
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from src import config

session = sagemaker.Session()


def get_or_create_feature_group(df, feature_group_name: str = config.FEATURE_GROUP_NAME):
    fg = FeatureGroup(name=feature_group_name, sagemaker_session=session)
    fg.load_feature_definitions(data_frame=df)
    fg.create(
        s3_uri=f"s3://{config.S3_BUCKET}/feature-store",
        record_identifier_name="transaction_id",
        event_time_feature_name="event_time",
        role_arn=config.SAGEMAKER_ROLE_ARN,
        enable_online_store=True,
    )
    while fg.describe().get("FeatureGroupStatus") != "Created":
        time.sleep(5)
    return fg


def ingest(fg: FeatureGroup, df) -> None:
    fg.ingest(data_frame=df, max_workers=4, wait=True)
