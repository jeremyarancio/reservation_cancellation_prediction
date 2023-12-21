from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_pipeline.pipeline.preprocess_step import PreprocessStep
from ml_pipeline.pipeline.inference_step import InferenceStep
from ml_pipeline.config import DataConfig 
from ml_pipeline.pipeline.utils.data import BatchInferenceData

data = BatchInferenceData(
    data_path=DataConfig.batch_path,
    preprocessed_batch_data_path=DataConfig.preprocessed_batch_data_path
)
preprocess_step = PreprocessStep(data=data)
inference_step = InferenceStep()

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'retries': 0,
    "catchup": False
}

with DAG(
    "inference-pipeline",
    default_args=default_args,
    start_date=datetime(2023, 12, 20),
    tags=["inference"],
    schedule=None
) as dag:
    
    t1 = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
    )
    t2 = PythonOperator(
        task_id="inference",
        python_callable=inference_step,
        op_args=[data.preprocessed_batch_data_path]
    )

    t1 >> t2

