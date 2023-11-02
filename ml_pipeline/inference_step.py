from typing import Union

import mlflow
import pandas as pd
import numpy as np

from ml_pipeline.config import MlflowConfig
from ml_pipeline.utils.step import Step


mlflow.set_tracking_uri(MlflowConfig.uri)


class InferenceStep(Step):

    def __init__(self) -> None:
        """"""
        models = mlflow.search_registered_models()
        latest_model_version = models[0].latest_versions[0].version
        self.model = mlflow.sklearn.load_model(
            model_uri=f"models:/{MlflowConfig.registered_model_name}/{latest_model_version}"
        )

    def run_step(self, batch: Union[np.array, pd.DataFrame]):
        """"""
        return self.model.predict(batch)


                        