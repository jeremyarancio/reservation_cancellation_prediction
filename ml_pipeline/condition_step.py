from typing import Dict

import mlflow

from ml_pipeline.config import MlflowConfig
from ml_pipeline.utils.step import Step


mlflow.set_tracking_uri(MlflowConfig.uri)



class ConditionStep(Step):

    def __init__(     
        self,
        criteria: float,
        metric: str
    ) -> None:
        self.criteria = criteria
        self.metric = metric

    def run_step(self):

        run_id = self.inputs

        run = mlflow.get_run(run_id=run_id)
        metric = run.data.metrics[self.metric]

        registered_models = mlflow.search_registered_models()

        if not registered_models:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{MlflowConfig.artifact_path}", 
                name=MlflowConfig.registered_model_name
            )
            return print("New model registered.")

        latest_registered_model = registered_models[0]
        registered_model_run = mlflow.get_run(latest_registered_model.latest_versions[0].run_id) #TODO: Can be improved
        registered_metric = registered_model_run.data.metrics[self.metric]

        if metric > registered_metric*(1 + self.criteria):
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{MlflowConfig.artifact_path}", 
                name=MlflowConfig.registered_model_name
            )
            return print("Model registered.")
        
        self.outputs = None
    