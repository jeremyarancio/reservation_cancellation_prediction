import logging

from ml_pipeline.pipeline import Pipeline
from ml_pipeline.preprocess_step import PreprocessStep
from ml_pipeline.inference_step import InferenceStep
from ml_pipeline.config import INFERENCE_DATA_PATH


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    
    preprocess_step = PreprocessStep(
        data_path=INFERENCE_DATA_PATH,
        training_mode=False
    )
    inference_step = InferenceStep()

    pipeline = Pipeline(steps=[preprocess_step, inference_step])
    outputs = pipeline.run()
    print(outputs)
