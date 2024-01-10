import logging
from typing import Literal

import mlflow

from steps.config import MlFlowConfig


LOGGER = logging.getLogger(__name__)


class ConditionStep:
    """Condition to register the model.

    Args:
        criteria (float): Coefficient applied to the metric of the model registered in the model registry.
        metric (str): Metric as a reference. Can be `["precision", "recall", or "roc_auc"]`.
    """

    def __init__(
        self, 
        criteria: float, 
        metric: Literal["roc_auc", "precision", "recall"]
    ) -> None:
        self.criteria = criteria
        self.metric = metric

    def __call__(self, mlflow_run_id: str) -> None:
        """
        Compare the metric from the last run to the model in the registry.
        if `metric_run > registered_metric*(1 + self.criteria)`, then the model is registered.
        """

        LOGGER.info(f"Run_id: {mlflow_run_id}")
        mlflow.set_tracking_uri(MlFlowConfig.uri)

        run = mlflow.get_run(run_id=mlflow_run_id)
        metric = run.data.metrics[self.metric]

        registered_models = mlflow.search_registered_models(
            filter_string=f"name = '{MlFlowConfig.registered_model_name}'"
        )

        if not registered_models:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow_run_id}/{MlFlowConfig.artifact_path}",
                name=MlFlowConfig.registered_model_name,
            )
            LOGGER.info("New model registered.")

        latest_registered_model = registered_models[0]
        registered_model_run = mlflow.get_run(
            latest_registered_model.latest_versions[0].run_id
        )  # TODO: Can be improved
        registered_metric = registered_model_run.data.metrics[self.metric]

        if metric > registered_metric * (1 + self.criteria):
            mlflow.register_model(
                model_uri=f"runs:/{mlflow_run_id}/{MlFlowConfig.artifact_path}",
                name=MlFlowConfig.registered_model_name,
            )
            LOGGER.info("Model registered as a new version.")
