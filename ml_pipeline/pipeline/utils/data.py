from dataclasses import dataclass
from abc import ABC
from pathlib import Path


@dataclass
class Data(ABC):
    data_path: Path

@dataclass
class TrainingData(Data):
    train_path: Path
    test_path: Path

@dataclass
class BatchInferenceData(Data):
    preprocessed_batch_data_path: Path

 
