from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder

from config import DATA_PATH, PREPROCESSED_DATA_PATH



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
    """""" 
    cats = np.array(cats).reshape(-1, 1)
    ordinal_encoder = OrdinalEncoder()
    encoded_cats = ordinal_encoder.fit_transform(cats)
    return encoded_cats

    

if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    # Fillna
    df["children"].fillna(0, inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["agent"].fillna(0, inplace=True)
    df["company"].fillna(0, inplace=True)

    # Category values
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
            "is_canceled",
            "lead_time",
            "arrival_date_year",
            "arrival_date_month",
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

    preprocessed_df.to_csv(PREPROCESSED_DATA_PATH)



