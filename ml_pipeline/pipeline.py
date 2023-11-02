from typing import List, Optional
import logging

import numpy as np

from ml_pipeline.utils.step import Step


LOGGER = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps
        self.outputs = None

    def run(self) -> Optional[np.ndarray]:
        """"""
        LOGGER.info("Start pipeline.")
        for step in self.steps:
            # First step
            if not self.outputs:
                step.run_step()
                self.outputs = step.outputs
            else:
                step.inputs = self.next
                step.run_step()
                if step.outputs:
                    self.outputs = step.outputs
            LOGGER.info("End pipeline")
        # Inference
        if self.outputs:
            return self.outputs
        
        

        