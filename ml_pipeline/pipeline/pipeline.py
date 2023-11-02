from typing import List, Optional
import logging

import numpy as np

from ml_pipeline.pipeline.utils.step import Step


LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Pipeline composed of Steps. Can return an output (for Inference for example)
    
    Args:
        steps (List[Step]): list of Step objects. Each Step has its own specificity like preprocessing, 
    inference, or model training.
    """

    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps
        self.outputs = None

    def run(self) -> Optional[np.ndarray]:
        """Run the pipeline step by step.

        Returns:
            Optional[np.ndarray]: Batch prediction in case of inference pipeline.
        """
        LOGGER.info("Start pipeline.")
        for step in self.steps:
            # First step
            if not self.outputs:
                step.run_step()
                self.outputs = step.outputs
            else:
                step.inputs = self.outputs
                step.run_step()
                if step.outputs:
                    self.outputs = step.outputs

        # Inference
        if self.outputs:
            return self.outputs
