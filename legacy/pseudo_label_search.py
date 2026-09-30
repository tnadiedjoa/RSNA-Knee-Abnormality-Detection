"""Check whether self-training on the ~4349 unlabeled reports improves the
text baseline's CV macro AUC, and at what confidence threshold."""

from src.data import load_labeled_train, load_unlabeled_train
from src.folds import make_folds
from src.metrics import macro_auc, per_label_auc, print_scores
from src.pseudo_label import compute_oof_pseudo
from src.text_baseline import compute_oof


def main():
    labeled_df = load_labeled_train()
    unlabeled_df = load_unlabeled_train()
    print(f"Labeled: {len(labeled_df)}  Unlabeled: {len(unlabeled_df)}", flush=True)
    folds = make_folds(len(labeled_df))

    baseline_oof = compute_oof(labeled_df, folds)
    baseline_scores = per_label_auc(baseline_oof, labeled_df)
    print(f"\nNo pseudo-labeling (current default pipeline): macro AUC = {macro_auc(baseline_scores):.3f}", flush=True)

    thresholds = (0.95, 0.9, 0.8)
    oofs = compute_oof_pseudo(labeled_df, unlabeled_df, folds, thresholds=thresholds)
    for t in thresholds:
        scores = per_label_auc(oofs[t], labeled_df)
        print(f"\nPseudo-label threshold={t}: macro AUC = {macro_auc(scores):.3f}", flush=True)
        print_scores(scores)


if __name__ == "__main__":
    main()
