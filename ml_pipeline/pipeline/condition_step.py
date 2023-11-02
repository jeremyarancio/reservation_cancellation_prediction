from typing import Dict
import logging

import mlflow

from ml_pipeline.config import MlflowConfig
from ml_pipeline.pipeline.utils.step import Step


LOGGER = logging.getLogger(__name__)
mlflow.set_tracking_uri(MlflowConfig.uri)


class ConditionStep(Step):
    """Condition to register the model.
    
    Args:
        criteria (float): Coefficient applied to the metric of the model registered in the model registry.
        metric (str): Metric as a reference. Can be `["precision", "recall", or "roc_auc"]`. Default to `"roc_auc"`.
    """

    def __init__(     
        self,
        criteria: float,
        metric: str = "roc_auc"
    ) -> None:
        self.criteria = criteria
        self.metric = metric

    def run_step(self) -> None:
        """
        Compare the metric from the last run to the model in the registry.
        if `metric_run > registered_metric*(1 + self.criteria)`, then the model is registered.
        """

        run_id = self.inputs

        run = mlflow.get_run(run_id=run_id)
        metric = run.data.metrics[self.metric]

        registered_models = mlflow.search_registered_models()

        if not registered_models:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{MlflowConfig.artifact_path}", 
                name=MlflowConfig.registered_model_name
            )
            return LOGGER.info("New model registered.")

        latest_registered_model = registered_models[0]
        registered_model_run = mlflow.get_run(latest_registered_model.latest_versions[0].run_id) #TODO: Can be improved
        registered_metric = registered_model_run.data.metrics[self.metric]

        if metric > registered_metric*(1 + self.criteria):
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/{MlflowConfig.artifact_path}", 
                name=MlflowConfig.registered_model_name
            )
            return LOGGER.info("Model registered as a new version.")
        
        self.outputs = None
    