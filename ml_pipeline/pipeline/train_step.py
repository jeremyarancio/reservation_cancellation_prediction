from typing import Dict, Any
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import pandas as pd
import mlflow

from ml_pipeline.config import TrainerConfig, MlFlowConfig
from ml_pipeline.pipeline.utils.artifact import Artifact
from ml_pipeline.pipeline.feature_engineering_step import FeatureEngineering


class TrainStep:
    """Training step tracking experiments with MLFlow.
    In this case, GradientBoostingClassifier has been picked, and the chosen metrics are:
    * precision
    * recall
    * roc_auc
    
    Args:
        params (Dict[str, Any]): Parameters of the model. Have to match the model arguments.
        model_name (str, optional): Additional information for experiments tracking. Defaults to TrainerConfig.model_name."""

    def __init__(
            self,
            params: Dict[str, Any],
            train_path: Path,
            test_path: Path,
            model_name: str = TrainerConfig.model_name
    ) -> None:
        self.params = params
        self.model_name = model_name
        self.train_path = train_path
        self.test_path = test_path

    def __call__(self) -> None:

        mlflow.set_tracking_uri(MlFlowConfig.uri)
        mlflow.set_experiment(MlFlowConfig.experiment_name)
        
        with mlflow.start_run():

            train_df = pd.read_parquet(self.train_path)
            test_df = pd.read_parquet(self.test_path)

            feature_engineering = FeatureEngineering()
            preprocessed_train_df = feature_engineering.fit_transform(train_df)
            preprocessed_test_df = feature_engineering.transform(test_df)
            
            # Train
            gbc = GradientBoostingClassifier(
                random_state=TrainerConfig.random_state,
                verbose=True,
                **self.params
            )
            model = gbc.fit(
                preprocessed_train_df.drop(feature_engineering.target_name, axis=1),
                preprocessed_train_df[feature_engineering.target_name]
            )

            # Evaluate
            y_test = preprocessed_test_df[feature_engineering.target_name]
            y_pred = model.predict(preprocessed_test_df.drop(feature_engineering.target_name, axis=1))

            # Metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            print(classification_report(y_test, y_pred))

            metrics = {
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }

            # Mlflow
            mlflow.log_params(TrainerConfig.params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag(key="model", value=self.model_name)
            mlflow.pyfunc.log_model(
                artifact_path=MlFlowConfig.artifact_path,
                python_model=Artifact(
                    model=model,
                    ordinal_encoder=feature_engineering.ordinal_encoder,
                    ordinal_encoded_features=feature_engineering.ordinal_encoded_features,
                    target_encoder=feature_engineering.target_encoder,
                    target_encoded_features=feature_engineering.target_encoded_features
                )               
            )

            return {"mlflow_run_id": mlflow.active_run().info.run_id}
