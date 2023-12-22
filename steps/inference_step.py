import logging
from pathlib import Path
import json
from typing import List

import mlflow
import pandas as pd

from steps.config import MlFlowConfig

LOGGER = logging.getLogger(__name__)


class InferenceStep:
    "Get the model from the model registry and predict in batch"

    def __call__(self, batch_path: Path) -> List[int]:
        """Use the MLFlow artifact built-in predict.
        
        Args:
            batch_path (Path): Input batch_path

        Return (List[int]):
            Predictions
        """
        model = self._load_model(
            registered_model_name=MlFlowConfig.registered_model_name
        )
        batch = self._load_batch(batch_path)
        if model:
            # Transform np.ndarray into list for serialization
            prediction = model.predict(batch).tolist()
            LOGGER.info(f"Prediction: {prediction}")
            return json.dumps(prediction)
        else:
            LOGGER.warning(
                "No model used for prediction. Model registry probably empty."
            )

    @staticmethod
    def _load_model(registered_model_name: str):
        """Load model from model registry.

        Args:
            registered_model_name (str): Name

        Returns:
            Model artifact
        """
        mlflow.set_tracking_uri(MlFlowConfig.uri)
        models = mlflow.search_registered_models(
            filter_string=f"name = '{registered_model_name}'"
        )
        LOGGER.info(f"Models in the model registry: {models}")
        if models:
            latest_model_version = models[0].latest_versions[0].version
            LOGGER.info(
                f"Latest model version in the model registry used for prediction: {latest_model_version}"
            )
            model = mlflow.sklearn.load_model(
                model_uri=f"models:/{registered_model_name}/{latest_model_version}"
            )
            return model
        else:
            LOGGER.warning(
                f"No model in the model registry under the name: {MlFlowConfig.registered_model_name}."
            )

    @staticmethod
    def _load_batch(batch_path: Path) -> pd.DataFrame:
        """Load dataframe from path"""
        batch = pd.read_parquet(batch_path)
        LOGGER.info(f"Batch columns: {batch.columns}")
        return batch
