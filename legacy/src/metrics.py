import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import LABELS


def per_label_auc(oof: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """oof and truth must be aligned on StudyInstanceUID (same rows, same order)."""
    scores = {}
    for label in LABELS:
        y = truth[label].values.astype(int)
        if len(np.unique(y)) < 2:
            scores[label] = float("nan")
            continue
        scores[label] = roc_auc_score(y, oof[label].values)
    return scores


def macro_auc(scores: dict) -> float:
    valid = [s for s in scores.values() if s == s]  # drop NaN
    return float(np.mean(valid)) if valid else float("nan")


def print_scores(scores: dict) -> None:
    for label in LABELS:
        s = scores[label]
        print(f"  {label:<20} {'nan' if s != s else f'{s:.3f}'}")
    print(f"Macro AUC: {macro_auc(scores):.3f}")
