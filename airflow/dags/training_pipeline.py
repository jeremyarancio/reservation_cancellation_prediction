# ML Pipeline using AWS and Airflow: https://aws.amazon.com/fr/blogs/machine-learning/build-end-to-end-machine-learning-workflows-with-amazon-sagemaker-and-apache-airflow/
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_pipeline.pipeline.preprocess_step import PreprocessStep
from ml_pipeline.pipeline.train_step import TrainStep
from ml_pipeline.pipeline.condition_step import ConditionStep
from ml_pipeline.pipeline.feature_engineering_step import FeatureEngineeringStep
from ml_pipeline.pipeline.utils.features_store import FeaturesStore
from ml_pipeline.config import (
    TRAINING_DATA_PATH,
    TrainerConfig,
    ConditionConfig,
    PreprocessConfig,
    FeatureEngineeringConfig,
)


inference_mode = False
feature_store = FeaturesStore(
    features_dir=FeatureEngineeringConfig.features_dir,
    encoders_path=FeatureEngineeringConfig.encoders_path,
)
preprocess_step = PreprocessStep(
    inference_mode=inference_mode, preprocessed_data_dir=PreprocessConfig.data_dir
)
feature_engineering_step = FeatureEngineeringStep(
    features_store=feature_store,
    inference_mode=inference_mode
)
train_step = TrainStep(params=TrainerConfig.params)
condition_step = ConditionStep(
    criteria=ConditionConfig.criteria, metric=ConditionConfig.metric
)

default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "training-pipeline",
    default_args=default_args,
    start_date=datetime(2023, 12, 19),
    tags=["training"],
    schedule=None,
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={"data_path": TRAINING_DATA_PATH},
    )
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            "train_path": PreprocessConfig.data_dir / "train.parquet",
            "test_path": PreprocessConfig.data_dir / "test.parquet",
        },
    )
    training_task = PythonOperator(
        task_id="training",
        python_callable=train_step,
        op_kwargs={
            "train_path": feature_store.features_dir / "train.parquet",
            "test_path": feature_store.features_dir / "test.parquet",
            "target": FeatureEngineeringConfig.target
        },
    )
    vaildation_task = PythonOperator(
        task_id="validation",
        python_callable=condition_step,
        op_kwargs=training_task.output,
    )

    preprocessing_task >> feature_engineering_task >> training_task >> vaildation_task
