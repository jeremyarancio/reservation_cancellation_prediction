from typing import Dict, Any
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import pandas as pd
import mlflow

from steps.config import TrainerConfig, MlFlowConfig


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
            model_name: str = TrainerConfig.model_name
    ) -> None:
        self.params = params
        self.model_name = model_name

    def __call__(
            self,
            train_path: Path,
            test_path: Path,
            target: str
        ) -> None:

        mlflow.set_tracking_uri(MlFlowConfig.uri)
        mlflow.set_experiment(MlFlowConfig.experiment_name)
        
        with mlflow.start_run():

            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            
            # Train
            gbc = GradientBoostingClassifier(
                random_state=TrainerConfig.random_state,
                verbose=True,
                **self.params
            )
            model = gbc.fit(
                train_df.drop(target, axis=1),
                train_df[target]
            )

            # Evaluate
            y_test = test_df[target]
            y_pred = model.predict(test_df.drop(target, axis=1))

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
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag(key="model", value=self.model_name)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=MlFlowConfig.artifact_path,      
            )

            return {"mlflow_run_id": mlflow.active_run().info.run_id}
