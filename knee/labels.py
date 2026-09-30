"""Report grades (scripts/llm_labels.py) -> soft training labels.

The mapping is fixed a priori (monotone in the grade) instead of fitted on the 58 gold studies,
which stay an independent check. "not_mentioned" gets a per-finding prior: silence usually means
normal. Synovitis is the exception: reports rarely discuss it while annotators mark it on about
half the studies, so an unmentioned synovitis inherits part of the effusion score (the two go
together; a public label set reported the same trick).
"""

import numpy as np
import pandas as pd

from knee.config import DATA_DIR, LABELS

GRADE_SCORE = {"absent": 0.02, "equivocal": 0.25, "mild": 0.6, "moderate": 0.85, "severe": 0.97}
SILENCE_PRIOR = {label: 0.10 for label in LABELS}


def grades_to_soft(grades: pd.DataFrame, synovitis_from_effusion: bool = True) -> pd.DataFrame:
    """grades: StudyInstanceUID + one grade column per label -> soft labels in [0, 1]."""
    out = grades[["StudyInstanceUID"]].copy()
    for label in LABELS:
        g = grades[label]
        out[label] = g.map(GRADE_SCORE).fillna(SILENCE_PRIOR[label]).astype(np.float32)
        out[f"{label}__mentioned"] = g.isin(list(GRADE_SCORE)).astype(np.int8)
    if synovitis_from_effusion:
        silent = out["Synovitis__mentioned"] == 0
        out.loc[silent, "Synovitis"] = (0.1 + 0.5 * out.loc[silent, "Effusion"]).astype(np.float32)
    return out


def load_soft(name: str, **kwargs) -> pd.DataFrame:
    return grades_to_soft(pd.read_parquet(DATA_DIR / "labels" / f"{name}.parquet"), **kwargs)


def gold() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv")
    return train.dropna(subset=LABELS)[["StudyInstanceUID"] + LABELS].reset_index(drop=True)
