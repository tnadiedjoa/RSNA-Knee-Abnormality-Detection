import pandas as pd

from src.config import LABELS, TEST_CSV, TRAIN_CSV


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_labeled_train() -> pd.DataFrame:
    """Rows of train.csv that have all 12 target labels filled in."""
    df = load_train()
    return df.dropna(subset=LABELS)


def load_unlabeled_train() -> pd.DataFrame:
    """Rows of train.csv with a report but no labels (weak-supervision candidates)."""
    df = load_train()
    return df[df[LABELS].isna().all(axis=1)]


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)
