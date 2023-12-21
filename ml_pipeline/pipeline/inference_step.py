import logging
from pathlib import Path

import mlflow
import pandas as pd

from ml_pipeline.config import MlFlowConfig

LOGGER = logging.getLogger(__name__)


class InferenceStep:
    "Get the model from the model registry and predict in batch"

    def __init__(self) -> None:
        """Load the latest version registered model artifact"""
        self.model = self._load_model(registered_model_name=MlFlowConfig.registered_model_name)

    def __call__(self, batch_path: Path):
        """Use the MLFlow artifact built-in predict`"""
        batch = self._load_batch(batch_path)
        if self.model:
            prediction = self.model.predict(batch)
            LOGGER.info(f"Prediction: {prediction}")
            return prediction
        else:
           LOGGER.warning("No model used for prediction. Model registry probably empty.")
    
    @staticmethod
    def _load_model(registered_model_name: str):
        mlflow.set_tracking_uri(MlFlowConfig.uri)
        models = mlflow.search_registered_models(
            filter_string=f"name = '{registered_model_name}'"
        )
        LOGGER.info(f"Models in the model registry: {models}")
        if models:
            latest_model_version = models[0].latest_versions[0].version
            LOGGER.info(f"Latest model version in the model registry used for prediction: {latest_model_version}")
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{registered_model_name}/{latest_model_version}"
            )
            return model
        else:
            LOGGER.warning(f"No model in the model registry under the name: {MlFlowConfig.registered_model_name}.")

    @staticmethod
    def _load_batch(batch_path: Path) -> pd.DataFrame:
        batch = pd.read_parquet(batch_path)
        LOGGER.info(f"Batch: {batch.columns}")
        return batch
                        