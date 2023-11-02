"Training script"
import logging

from ml_pipeline.pipeline.pipeline import Pipeline
from ml_pipeline.pipeline.preprocess_step import PreprocessStep
from ml_pipeline.pipeline.train_step import TrainStep
from ml_pipeline.pipeline.condition_step import ConditionStep

from ml_pipeline.config import (
    DATA_PATH, 
    PREPROCESSED_DATA_PATH, 
    TrainerConfig,
    ConditionConfig
) 


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    preprocess_step = PreprocessStep(
        data_path=DATA_PATH,
        preprocessed_data_path=PREPROCESSED_DATA_PATH
    )
    train_step = TrainStep(
        params=TrainerConfig.params
    )
    condition_step = ConditionStep(
        criteria=ConditionConfig.criteria,
        metric=ConditionConfig.metric
    )

    pipeline = Pipeline(
        steps=[preprocess_step, train_step, condition_step]
    )
    pipeline.run()