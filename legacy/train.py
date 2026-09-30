"""Entry point: cross-validate the text-only baseline, save its OOF predictions
(for later ensembling) and its final model, fit on all labeled data."""

import pickle

from src.config import MODELS_DIR, OOF_DIR
from src.data import load_labeled_train
from src.folds import make_folds
from src.metrics import per_label_auc, print_scores
from src.text_baseline import compute_oof, train_final_models


def main():
    df = load_labeled_train()
    print(f"Labeled training examples: {len(df)}")

    folds = make_folds(len(df))
    oof = compute_oof(df, folds)
    scores = per_label_auc(oof, df)
    print("\nPer-label cross-validated AUC:")
    print_scores(scores)

    OOF_DIR.mkdir(exist_ok=True)
    oof.to_csv(OOF_DIR / "text_oof.csv", index=False)
    print(f"\nSaved OOF predictions to {OOF_DIR / 'text_oof.csv'}")

    models = train_final_models(df)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "text_baseline.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"Saved models to {MODELS_DIR / 'text_baseline.pkl'}")


if __name__ == "__main__":
    main()
