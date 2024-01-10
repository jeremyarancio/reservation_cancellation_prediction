from pathlib import Path

import pandas as pd

from steps.config import TrainerConfig
from steps.utils.data_classes import PreprocessingData


class PreprocessStep:
    """Preprocessing based on Exploratory Data Analysis done in `notebooks/0_exploratory_data_analysis.ipynb`
    
    Args:
        inference_mode (bool): Training or inference mode.
        preprocessing_data (PreprocessingData): PreprocessingStep output paths."""

    def __init__(
        self,
        inference_mode: bool,
        preprocessing_data: PreprocessingData
    ) -> None:
        self.inference_mode = inference_mode
        self.preprocessing_data = preprocessing_data

    def __call__(self, data_path: Path) -> None:
        """Data is preprocessed then, regarding if inference=True or False:
            * False: Split data into train and test.
            * True: Data preprocessed then returned simply
        
        Args:
            data_path (Path): Input
        """

        preprocessed_df = pd.read_parquet(data_path)
        preprocessed_df = self._preprocess(preprocessed_df)

        if not self.inference_mode:
            train_df = preprocessed_df.sample(
                frac=TrainerConfig.train_size, random_state=TrainerConfig.random_state
            )
            test_df = preprocessed_df.drop(train_df.index)
            train_df.to_parquet(self.preprocessing_data.train_path, index=False)
            test_df.to_parquet(self.preprocessing_data.test_path, index=False)

        if self.inference_mode:
            preprocessed_df.to_parquet(self.preprocessing_data.batch_path, index=False)

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing."""
        df["children"].fillna(0, inplace=True)
        df["country"].fillna("Unknown", inplace=True)
        df["agent"].fillna(0, inplace=True)
        df["company"].fillna(0, inplace=True)
        return df
