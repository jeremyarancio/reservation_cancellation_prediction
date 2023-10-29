import os
from pathlib import Path


REPO_DIR = Path(os.path.realpath(""))
DATA_PATH = REPO_DIR / "data/hotel_bookings.csv"
PREPROCESSED_DATA_PATH = REPO_DIR / "data/preprocessed_data.csv"