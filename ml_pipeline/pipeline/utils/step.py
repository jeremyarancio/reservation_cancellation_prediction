from abc import ABC, abstractmethod
import logging
from typing import Tuple, Union, Any


LOGGER = logging.getLogger(__name__)


class Step(ABC): 
    """Step object used as a component of the Pipeline."""

    def __init__(
            self,
            inputs: Union[Tuple, Any], 
            outputs: Union[Tuple, Any] = None
    ) -> None:
        """
        Args:
            inputs (Union[Tuple, Any]): Input of the step.
            outputs (Union[Tuple, Any], optional): Outputs of the step. It is possible no output is required, for example 
            at the end of the pipeline.
        """
        self.inputs = inputs
        self.outputs = outputs
    
    @abstractmethod
    def run_step(self):
        pass

