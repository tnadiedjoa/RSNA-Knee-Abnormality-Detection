"""5 folds over the training studies, grouped by patient and by (near-)identical report,
stratified on the number of report-positive findings. Saved to data/folds.csv.

    python -m scripts.make_folds <label set name>"""

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from knee.config import DATA_DIR, LABELS, SEED


def report_key(text: str) -> str:
    """Report with digits and dates masked: template reports of different patients collide on purpose."""
    t = re.sub(r"\d+", "#", text.lower())
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().encode()).hexdigest()


def main(labels: str, n_folds: int = 5):
    train = pd.read_csv(DATA_DIR / "train.csv")
    meta = pd.read_parquet(DATA_DIR / "series_meta.parquet")
    patient = meta.groupby("study")["PatientID"].first()
    train["patient"] = train["StudyInstanceUID"].map(patient).fillna(train["StudyInstanceUID"])
    train["report_key"] = train["Report"].map(report_key)

    # union-find over patient and report key -> group id
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p, r in zip("p" + train["patient"], "r" + train["report_key"]):
        parent[find(p)] = find(r)
    train["group"] = ["g" + find("p" + p) for p in train["patient"]]

    grades = pd.read_parquet(DATA_DIR / "labels" / f"{labels}.parquet").set_index("StudyInstanceUID")
    n_pos = grades.reindex(train["StudyInstanceUID"])[LABELS].isin(["moderate", "severe"]).sum(1).to_numpy()
    strat = np.minimum(n_pos, 5) + 10 * train[LABELS[0]].notna().to_numpy()  # also spread the gold studies
    folds = np.full(len(train), -1)
    for k, (_, val) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED)
                                 .split(train, strat, train["group"])):
        folds[val] = k
    out = train[["StudyInstanceUID", "patient", "group"]].assign(fold=folds)
    out.to_csv(DATA_DIR / "folds.csv", index=False)
    print(f"{train['patient'].nunique()} patients, {out['group'].nunique()} groups for {len(out)} studies")
    print(out.groupby("fold").size().to_dict(), "gold per fold:",
          out[train[LABELS[0]].notna()].groupby("fold").size().to_dict())


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
