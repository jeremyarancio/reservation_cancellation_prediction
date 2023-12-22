import os
from pathlib import Path


REPO_DIR = Path(os.path.realpath(""))
INFERENCE_DATA_PATH = REPO_DIR / "data/sample_for_inference.parquet"
TRAINING_DATA_PATH = REPO_DIR / "data/hotel_bookings.parquet"


class PreprocessConfig:
    data_dir = REPO_DIR / "data/preprocessed"
    train_name = "train.parquet"
    test_name = "test.parquet"
    batch_name = "batch.parquet"

class TrainerConfig:
    model_name ="gradient-boosting"
    random_state = 42
    train_size = 0.2
    shuffle = True
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }

class ConditionConfig:
    criteria = 0.05
    metric = "roc_auc"

class MlFlowConfig:
    uri = "http://0.0.0.0:8000"
    experiment_name = "cancelation_estimator"
    artifact_path = "model-artifact"
    registered_model_name = "cancelation_estimator"

class FeatureEngineeringConfig:
    features_dir = REPO_DIR / "data/features_store"
    encoders_path = REPO_DIR / "artifact/encoders.joblib"
    base_features = [
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests"
    ]
    ordinal_features = [
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "customer_type"
    ]
    target_features = [
        "country",
        "booking_changes",
        "agent",
        "company"
    ]
    target = "is_canceled"