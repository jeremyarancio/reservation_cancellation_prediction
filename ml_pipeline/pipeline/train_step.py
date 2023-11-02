from typing import Dict, Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import pandas as pd
import mlflow

from ml_pipeline.config import TrainerConfig, MlflowConfig
from ml_pipeline.pipeline.utils.step import Step


mlflow.set_tracking_uri(MlflowConfig.uri)
mlflow.set_experiment(MlflowConfig.experiment_name)


class TrainStep(Step):
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
            model_name: str = TrainerConfig.model_name,
    ) -> None:
        self.params = params
        self.model_name = model_name

    def run_step(self):

        with mlflow.start_run():
            
            preprocessed_data_path = self.inputs

            data_df = pd.read_csv(preprocessed_data_path)

            X_train, X_test, y_train, y_test = train_test_split(
                    data_df.drop(["is_canceled"], axis=1),
                    data_df["is_canceled"],
                    shuffle=TrainerConfig.shuffle,
                    random_state=TrainerConfig.random_state,
                    test_size=TrainerConfig.test_size,
                    stratify=data_df[["is_canceled"]]
                )
            
            # Train
            model = GradientBoostingClassifier(
                random_state=TrainerConfig.random_state,
                verbose=True,
                **self.params
            ).fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

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
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path=MlflowConfig.artifact_path,
            )

            run_id = mlflow.active_run().info.run_id
            self.outputs = run_id
    