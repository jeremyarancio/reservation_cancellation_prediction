import logging
from typing import Tuple, Optional
import joblib
from pathlib import Path

from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
import pandas as pd

from steps.utils.data_classes import FeaturesEncoder, FeaturesEngineeringEData
from steps.config import FeatureEngineeringConfig


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    def __init__(
            self,
            inference_mode: bool,
            feature_engineering_data: FeaturesEngineeringEData
        ) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        batch_path: Optional[Path] = None,
    ) -> None:
        if not self.inference_mode:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            self.fit_transform(
                df=train_df, 
                path=self.feature_engineering_data.train_path
            )
            self.transform(
                df=test_df,
                path=self.feature_engineering_data.test_path
            )

        if self.inference_mode:
            batch_df = pd.read_parquet(batch_path)
            self.transform(
                batch_df,
                path=self.feature_engineering_data.batch_path
            )

    def fit_transform(self, df: pd.DataFrame, path: Path) -> None:
        """Feature engineering methods.

        Args:
            data_path (Path): preprocessed data for feature engineering (.parquet).
            preprocessed_data_path (Path): path of the processed data.
        """
        LOGGER.info("Start features engineering 'fit_transform'.")
        feature_encoders = self._init_features_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(
            df, features_encoder=feature_encoders
        )

        ordinal_encoded_data = feature_encoders.ordinal_encoder.fit_transform(ordinal_df)
        target_encoded_data = feature_encoders.target_encoder.fit_transform(target_df, target_col)

        base_df[feature_encoders.ordinal_features] = ordinal_encoded_data
        base_df[feature_encoders.target_features] = target_encoded_data

        # Don't forget to add the target
        base_df[feature_encoders.target] = target_col

        base_df.to_parquet(path=path)
        feature_encoders.to_joblib(path=self.feature_engineering_data.encoders_path)
        LOGGER.info(f"Features and encoders successfully saved respectively to {str(path)} and {str(self.feature_engineering_data.encoders_path)}")

    def transform(self, df: pd.DataFrame, path: Path) -> None:
        """ """
        LOGGER.info("Start features engineering 'tranform'.")
        # TODO: Raise error if fit_transform not instantiated
        features_encoder = self._load_features_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(
            df, 
            features_encoder=features_encoder
        )

        ordinal_encoded_data = features_encoder.ordinal_encoder.transform(ordinal_df)
        target_encoded_data = features_encoder.target_encoder.transform(target_df)

        base_df[features_encoder.ordinal_features] = ordinal_encoded_data
        base_df[features_encoder.target_features] = target_encoded_data

        if target_col is not None:
            base_df[features_encoder.target] = target_col

        base_df.to_parquet(path=path)
        LOGGER.info(f"Features successfully saved to {str(path)}")

    def _init_features_encoder(self) -> FeaturesEncoder:
        """_summary_

        Args:
            feature_encoders (FeatureEncoders): _description_
        """
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", 
            unknown_value=-1
        )
        target_encoder = TargetEncoder()
        return FeaturesEncoder(
            ordinal_encoder=ordinal_encoder,
            target_encoder=target_encoder,
            ordinal_features=FeatureEngineeringConfig.ordinal_features,
            target_features=FeatureEngineeringConfig.target_features,
            base_features=FeatureEngineeringConfig.base_features,
            target=FeatureEngineeringConfig.target,
        )

    def _load_features_encoder(self) -> FeaturesEncoder:
        # Load encoders artifact
        features_encoder = joblib.load(self.feature_engineering_data.encoders_path)
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
