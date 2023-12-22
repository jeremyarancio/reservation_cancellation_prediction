from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import joblib

from sklearn.preprocessing import OrdinalEncoder, TargetEncoder


@dataclass
class PreprocessingData:
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    batch_path: Optional[Path] = None


@dataclass
class FeaturesEngineeringEData:
    encoders_path: Path
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    batch_path: Optional[Path] = None


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
