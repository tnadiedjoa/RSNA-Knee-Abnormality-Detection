"""Entry point: generate submissions/submission.csv from the saved text baseline.

Note: the public test.csv stub has no `Report` column (only StudyInstanceUID) —
the real hidden test set, substituted when this runs as a Kaggle submission
notebook, is expected to include it. If it's missing, we fall back to the
neutral 0.5 baseline so the file stays valid.
"""

import pickle

import pandas as pd

from src.config import LABELS, MODELS_DIR, SUBMISSIONS_DIR
from src.data import load_test
from src.text_baseline import predict


def main():
    test_df = load_test()

    if "Report" not in test_df.columns:
        print("WARNING: no 'Report' column in test.csv — falling back to 0.5 for all labels.")
        for label in LABELS:
            test_df[label] = 0.5
    else:
        with open(MODELS_DIR / "text_baseline.pkl", "rb") as f:
            models = pickle.load(f)
        preds = predict(models, test_df["Report"])
        test_df = pd.concat([test_df[["StudyInstanceUID"]], preds], axis=1)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "submission.csv"
    test_df[["StudyInstanceUID"] + LABELS].to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
