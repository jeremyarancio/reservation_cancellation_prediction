from pathlib import Path

import pandas as pd

from steps.config import TrainerConfig
from steps.utils.data_classes import PreprocessingData


class PreprocessStep:
    """Preprocessing based on Exploratory Data Analysis done in `notebooks/0_exploratory_data_analysis.ipynb`
    Cat objects are transformed respectively with TargetEncoder or OrdinalEncoder.
    Check the notebook for more details.

    Args:
        data_path (str): data.parquet file to preprocess. If `training_mode==False`, switch the preprocessing to inference mode.
        preprocessed_data_path (Optional[Path], optional): Output path of the preprocessed data. Defaults to None.
        training_mode (Optional[bool], optional): Switch to training or inference mode. Defaults to True.
    """

    def __init__(
        self,
        inference_mode: bool,
        preprocessing_data: PreprocessingData
    ) -> None:
        self.inference_mode = inference_mode
        self.preprocessing_data = preprocessing_data

    def __call__(self, data_path: Path) -> None:
        """Check notebook: `notebooks/0_exploratory_data_analysis.ipynb`"""

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
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df["children"].fillna(0, inplace=True)
        df["country"].fillna("Unknown", inplace=True)
        df["agent"].fillna(0, inplace=True)
        df["company"].fillna(0, inplace=True)
        return df
