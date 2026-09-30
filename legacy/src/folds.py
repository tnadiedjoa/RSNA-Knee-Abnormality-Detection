"""Shared fold assignment so every model (text, image, ...) is evaluated and
produces out-of-fold predictions on the *same* split of the same 58 labeled
studies — a prerequisite for ensembling their predictions later.

Kept dependency-free (only numpy/sklearn) so it can be copy-pasted verbatim
into the Kaggle notebook, which can't import from `src/`.
"""

import numpy as np
from sklearn.model_selection import KFold

from src.config import RANDOM_STATE


def make_folds(n_rows: int, n_splits: int = 5, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Returns an array of length n_rows giving each row's fold id (0..n_splits-1)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(n_rows, dtype=int)
    for fold_id, (_, val_idx) in enumerate(kf.split(np.arange(n_rows))):
        folds[val_idx] = fold_id
    return folds
