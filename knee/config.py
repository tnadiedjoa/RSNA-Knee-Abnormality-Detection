import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KNEE_DATA_DIR", ROOT_DIR / "data"))
CACHE_DIR = Path(os.environ.get("KNEE_CACHE_DIR", DATA_DIR / "cache"))
CACHE_SIZE = int(os.environ.get("KNEE_CACHE_SIZE", 256))  # pixel size of the slices stored in CACHE_DIR
RUNS_DIR = ROOT_DIR / "runs"

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SEED = 42
