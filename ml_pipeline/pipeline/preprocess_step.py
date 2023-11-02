from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder

from ml_pipeline.pipeline.utils.step import Step


def cat_target_encode(cats, target) -> np.ndarray:
    """Target encoding for high cardinality categories.

    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html#sklearn.preprocessing.TargetEncoder

    Args:
        cats (pd.Series): categories
        target (pd.Series): target (is_cancelled in our case)

    Returns:
        _type_: encoded categories
    """
    cats = np.array(cats).reshape(-1, 1)
    target_encoder = TargetEncoder(smooth="auto")
    encoded_cats = target_encoder.fit_transform(cats, target)
    return encoded_cats


def cat_ordinal_encode(cats) -> np.ndarray:
    """Transform cat into labels int""" 
    cats = np.array(cats).reshape(-1, 1)
    ordinal_encoder = OrdinalEncoder()
    encoded_cats = ordinal_encoder.fit_transform(cats)
    return encoded_cats


class PreprocessStep(Step):
    """Preprocessing based on Exploratory Data Analysis done in `notebooks/0_exploratory_data_analysis.ipynb`
    Cat objects are transformed respectively with TargetEncoder or OrdinalEncoder.
    Check the notebook for more details.

    Args:
        data_path (str): data.csv file to preprocess. If `training_mode==False`, switch the preprocessing to inference mode.
        preprocessed_data_path (Optional[Path], optional): Output path of the preprocessed data. Defaults to None.
        training_mode (Optional[bool], optional): Switch to training or inference mode. Defaults to True.
    """

    def __init__(
            self,
            data_path: Path,
            preprocessed_data_path: Optional[Path] = None,
            training_mode: Optional[bool] = True
    ) -> None:
        self.training_mode = training_mode
        self.preprocessed_data_path = preprocessed_data_path
        super().__init__(inputs=data_path)

    def run_step(self):
        """Check notebook: `notebooks/0_exploratory_data_analysis.ipynb`"""
        data_path = self.inputs

        df = pd.read_csv(data_path)

        # Fillna
        df["children"].fillna(0, inplace=True)
        df["country"].fillna("Unknown", inplace=True)
        df["agent"].fillna(0, inplace=True)
        df["company"].fillna(0, inplace=True)

        # Category values
        arrival_date_month = cat_ordinal_encode(cats=df["arrival_date_month"])
        meals = cat_ordinal_encode(cats=df["meal"])
        countries = cat_target_encode(cats=df["country"], target=df["is_canceled"])
        market_segments = cat_ordinal_encode(cats=df["market_segment"])
        distribution_channels = cat_ordinal_encode(cats=df["distribution_channel"])
        reserved_room_types = cat_ordinal_encode(cats=df["reserved_room_type"])
        assigned_room_types = cat_ordinal_encode(cats=df["assigned_room_type"])
        booking_changes = cat_target_encode(cats=df["booking_changes"], target=df["is_canceled"])
        deposit_types = cat_ordinal_encode(cats=df["deposit_type"])
        agents = cat_target_encode(cats=df["agent"], target=df["is_canceled"])
        companies = cat_target_encode(cats=df["company"], target=df["is_canceled"])
        customer_types = cat_ordinal_encode(cats=df["customer_type"])

        preprocessed_df = df[
            [
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
        ]

        preprocessed_df["arrival_date_month"] = arrival_date_month
        preprocessed_df["meal"] = meals
        preprocessed_df["country"] = countries
        preprocessed_df["market_segment"] = market_segments
        preprocessed_df["distribution_channel"] = distribution_channels
        preprocessed_df["reserved_room_type"] = reserved_room_types
        preprocessed_df["assigned_room_type"] = assigned_room_types
        preprocessed_df["booking_changes"] = booking_changes
        preprocessed_df["deposit_type"] = deposit_types
        preprocessed_df["agent"] = agents
        preprocessed_df["company"] = companies
        preprocessed_df["customer_type"] = customer_types

        if self.training_mode:
            # We add the target to the training dataset
            preprocessed_df["is_canceled"] = df["is_canceled"]
            preprocessed_df.to_csv(self.preprocessed_data_path)
            self.outputs = self.preprocessed_data_path

        if not self.training_mode:
            # Inference mode
            self.outputs = preprocessed_df
            



