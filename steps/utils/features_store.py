from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import joblib

from sklearn.preprocessing import OrdinalEncoder, TargetEncoder


@dataclass
class FeaturesStore:
    features_dir: Path
    encoders_path: Path

@dataclass
class FeaturesEncoder:
    ordinal_encoder: OrdinalEncoder
    target_encoder: TargetEncoder
    base_features: Iterable[str]
    ordinal_features: Iterable[str]
    target_features: Iterable[str]
    target: str

    def to_joblib(self, path: Path) -> None:
        """_summary_

        Args:
            path (Path): _description_
        """
        joblib.dump(self, path)
