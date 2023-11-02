from abc import ABC, abstractmethod
import logging
from typing import Tuple, Union, Any


LOGGER = logging.getLogger(__name__)


class Step(ABC): 

    def __init__(
            self,
            inputs: Union[Tuple, Any], 
            outputs: Union[Tuple, Any] = None
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
    
    @abstractmethod
    def run_step(self):
        pass

