from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pickle

from sklearn.base import BaseEstimator

@dataclass
class FeaturesStore:
    features_path: Path
    encoders_path: Path

@dataclass
class FeaturesEncoder:
    ordinal_encoder: BaseEstimator
    target_encoder: BaseEstimator
    base_features: Iterable[str]
    ordinal_features: Iterable[str]
    target_features: Iterable[str]
    target: str

    def to_pickle(
        self, 
        path: Path
    ) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
