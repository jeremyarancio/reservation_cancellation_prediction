import logging
from typing import Tuple, Optional
import joblib
from pathlib import Path

from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
import pandas as pd

from ml_pipeline.pipeline.utils.features_store import FeaturesEncoder, FeaturesStore
from ml_pipeline.config import FeatureEngineeringConfig


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    def __init__(self, features_store: FeaturesStore, inference_mode: bool) -> None:
        self.features_store = features_store
        self.inference_mode = inference_mode

    def __call__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        batch_path: Optional[Path] = None,
    ) -> None:
        if not self.inference_mode:
            filename_train = train_path.stem
            filename_test = test_path.stem
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            self.fit_transform(df=train_df, file_name=filename_train)
            self.transform(df=test_df, file_name=filename_test)

        if self.inference_mode:
            filename = batch_path.stem
            batch_df = pd.read_parquet(batch_path)
            self.transform(batch_df, file_name=filename)

    def fit_transform(self, df: pd.DataFrame, file_name: str) -> None:
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

        data_path = self.features_store.features_dir / (file_name + ".parquet")
        base_df.to_parquet(data_path)
        feature_encoders.to_joblib(path=self.features_store.encoders_path)
        LOGGER.info(f"Features and encoders successfully saved respectively to {str(data_path)} and {str(self.features_store.encoders_path)}")

    def transform(self, df: pd.DataFrame, file_name: str) -> None:
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

        data_path = self.features_store.features_dir / (file_name + ".parquet")
        base_df.to_parquet(data_path)
        LOGGER.info(
            f"Features and encoders successfully saved respectively to {str(data_path)} and {str(self.features_store.encoders_path)}"
        )

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
        features_encoder = joblib.load(self.features_store.encoders_path)
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
