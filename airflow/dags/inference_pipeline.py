from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_pipeline.pipeline.preprocess_step import PreprocessStep
from ml_pipeline.pipeline.inference_step import InferenceStep
from ml_pipeline.pipeline.feature_engineering_step import FeatureEngineeringStep
from ml_pipeline.pipeline.utils.features_store import FeaturesStore
from ml_pipeline.config import (
    FeatureEngineeringConfig,
    INFERENCE_DATA_PATH,
    PreprocessConfig,
)


inference_mode = True
features_store = FeaturesStore(
    features_dir=FeatureEngineeringConfig.features_dir,
    encoders_path=FeatureEngineeringConfig.encoders_path,
)

# Steps
preprocess_step = PreprocessStep(
    inference_mode=inference_mode, preprocessed_data_dir=PreprocessConfig.data_dir
)
feature_engineering_step = FeatureEngineeringStep(
    features_store=features_store,
    inference_mode=inference_mode,
)
inference_step = InferenceStep()


default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "inference-pipeline",
    default_args=default_args,
    start_date=datetime(2023, 12, 20),
    tags=["inference"],
    schedule=None,
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={"data_path": INFERENCE_DATA_PATH},
    )
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            "batch_path": PreprocessConfig.data_dir / PreprocessConfig.batch_name
        },
    )
    inference_task = PythonOperator(
        task_id="inference",
        python_callable=inference_step,
        op_kwargs={"batch_path": features_store.features_dir / "batch.parquet"},
    )

    preprocessing_task >> feature_engineering_task >> inference_task
