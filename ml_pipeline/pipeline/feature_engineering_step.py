import logging
from typing import Tuple
import pickle

from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
import pandas as pd

from ml_pipeline.pipeline.utils.features_encoder import FeaturesEncoder, FeaturesStore
from ml_pipeline.config import FeatureEngineeringConfig
from ml_pipeline.pipeline.utils.data import Data


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:

    def __init__(
            self,
            features_store: FeaturesStore,
            inference_mode: bool
        ) -> None:
        self.features_store = features_store
        self.inference_mode = inference_mode
    
    def __call__(
        self,
        data: Data
    ) -> None:
        if self.inference_mode:
            data.

    def fit_transform(
        self, 
        df: pd.DataFrame,                                                                                                                                                                                                                            
    ) -> None:
        """Feature engineering methods.

        Args:
            data_path (Path): preprocessed data for feature engineering (.parquet).
            preprocessed_data_path (Path): path of the processed data.
        """
        LOGGER.info("Start features engineering 'fit_transform'.")
        feature_encoders = self._load_features_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(df, features_encoder=feature_encoders)

        ordinal_encoded_data = feature_encoders.ordinal_encoder.fit_transform(ordinal_df)
        target_encoded_data = feature_encoders.target_encoder.fit_transform(target_df, target_col)

        base_df[feature_encoders.ordinal_features] = ordinal_encoded_data
        base_df[feature_encoders.target_features] = target_encoded_data

        # Don't forget to add the target
        base_df[feature_encoders.target] = target_col

        base_df.to_parquet(self.features_store.features_path)
        feature_encoders.to_pickle(path=self.features_store.encoders_path)
        LOGGER.info(f"Features and encoders successfully saved respectively to {self.features_store.features_path} and {self.features_store.encoders_path}")

    def transform(self, df: pd.DataFrame) -> None:
        """
        """
        LOGGER.info("Start features engineering 'tranform'.")
        #TODO: Raise error if fit_transform not instantiated
        features_encoder = self._load_features_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(df, features_encoder=features_encoder)
        
        ordinal_encoded_data = features_encoder.ordinal_encoder.transform(ordinal_df)
        target_encoded_data = features_encoder.target_encoder.transform(target_df)

        base_df[features_encoder.ordinal_features] = ordinal_encoded_data
        base_df[features_encoder.target_features] = target_encoded_data

        if not self.inference_mode:
            base_df[features_encoder.target] = target_col
    
    def _load_features_encoder(self) -> FeaturesEncoder:
        """_summary_

        Args:
            feature_encoders (FeatureEncoders): _description_
        """
        if not self.inference_mode:
            #Init encoders
            ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            ordinal_encoder.transform
            target_encoder = TargetEncoder()
            return FeaturesEncoder(
                ordinal_encoder=ordinal_encoder,
                target_encoder=target_encoder,
                ordinal_features=FeatureEngineeringConfig.ordinal_features,
                target_features=FeatureEngineeringConfig.target_features,
                base_features=FeatureEngineeringConfig.base_features,
                target=FeatureEngineeringConfig.target
            )
        elif self.inference_mode:
            # Load encoders artifact
            with open(self.features_store, "rb") as f:
                features_encoder = pickle.load(f)
            return features_encoder

    def _get_dfs(
        self,
        df: pd.DataFrame,
        features_encoder: FeaturesEncoder
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """"""
        base_df = df[features_encoder.base_features]
        ordinal_df = df[features_encoder.ordinal_features]
        target_df = df[features_encoder.target_features]
        if not self.inference_mode:
            target_col = df[features_encoder.target]
            return base_df, ordinal_df, target_df, target_col
        elif self.inference_mode:
            return base_df, ordinal_df, target_df, None
