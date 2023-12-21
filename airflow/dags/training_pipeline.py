# ML Pipeline using AWS and Airflow: https://aws.amazon.com/fr/blogs/machine-learning/build-end-to-end-machine-learning-workflows-with-amazon-sagemaker-and-apache-airflow/
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_pipeline.pipeline.preprocess_step import PreprocessStep
from ml_pipeline.pipeline.train_step import TrainStep
from ml_pipeline.pipeline.condition_step import ConditionStep
from ml_pipeline.config import (
    DataConfig,
    TrainerConfig,
    ConditionConfig
) 
from ml_pipeline.pipeline.utils.data import TrainingData


data = TrainingData(
    data_path=DataConfig.data_path,
    train_path=DataConfig.train_path,
    test_path=DataConfig.test_path
)

preprocess_step = PreprocessStep(
    data=data
)
train_step = TrainStep(
    params=TrainerConfig.params,
    train_path=data.train_path,
    test_path=data.test_path
)
condition_step = ConditionStep(
    criteria=ConditionConfig.criteria,
    metric=ConditionConfig.metric
)

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'retries': 0,
    "catchup": False
}

with DAG(
    "training-pipeline",
    default_args=default_args,
    start_date=datetime(2023, 12, 19),
    tags=["training"],
    schedule=None
) as dag:

    t1 = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
    )

    t2 = PythonOperator(
        task_id="training",
        python_callable=train_step
    )
    t3 = PythonOperator(
        task_id="validation",
        python_callable=condition_step,
        op_kwargs=t2.output
    )

    t1 >> t2 >> t3
