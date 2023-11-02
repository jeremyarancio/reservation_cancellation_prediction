import os
from pathlib import Path


REPO_DIR = Path(os.path.realpath(""))
DATA_PATH = REPO_DIR / "data/hotel_bookings.csv"
PREPROCESSED_DATA_PATH = REPO_DIR / "data/preprocessed_data.csv"
INFERENCE_DATA_PATH = REPO_DIR / "data/inference/sample_for_inference.csv"


class TrainerConfig:
    model_name ="Gradient Boosting"
    random_state = 42
    test_size = 0.2
    shuffle = True
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }

class ConditionConfig:
    criteria = 0.05
    metric = "roc_auc"

class MlflowConfig:
    uri = "http://0.0.0.0:8000"
    experiment_name = "cancelation_predictor"
    artifact_path = "sklearn-artifact-model"
    registered_model_name = "predictor"
