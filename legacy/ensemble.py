"""Combine the text and image baselines' out-of-fold predictions (see
train.py and kaggle/image_baseline.ipynb, both using src/folds.make_folds so
they're aligned study-for-study) and check whether a simple average beats
either model alone.
"""

import pandas as pd

from src.config import LABELS, OOF_DIR
from src.data import load_labeled_train
from src.metrics import macro_auc, per_label_auc, print_scores


def main():
    truth = load_labeled_train()
    text_oof = pd.read_csv(OOF_DIR / "text_oof.csv")
    image_oof = pd.read_csv(OOF_DIR / "image_oof.csv")

    merged = truth[["StudyInstanceUID"]].merge(
        text_oof, on="StudyInstanceUID", suffixes=("", "_text")
    ).merge(image_oof, on="StudyInstanceUID", suffixes=("_text", "_image"))
    assert len(merged) == len(truth), "OOF files must cover the same labeled studies"

    ensemble = pd.DataFrame({"StudyInstanceUID": merged["StudyInstanceUID"]})
    for label in LABELS:
        ensemble[label] = (merged[f"{label}_text"] + merged[f"{label}_image"]) / 2

    print("Text-only:")
    text_scores = per_label_auc(text_oof, truth)
    print_scores(text_scores)

    print("\nImage-only:")
    image_scores = per_label_auc(image_oof, truth)
    print_scores(image_scores)

    print("\nEnsemble (50/50 average):")
    ensemble_scores = per_label_auc(ensemble, truth)
    print_scores(ensemble_scores)

    print(f"\nMacro AUC — text: {macro_auc(text_scores):.3f} | "
          f"image: {macro_auc(image_scores):.3f} | "
          f"ensemble (50/50): {macro_auc(ensemble_scores):.3f}")

    print("\nWeight sweep (text_weight, macro AUC) — approximate: weight is picked"
          " by looking at all 58 OOF predictions, not re-nested in CV:")
    best = (0.5, macro_auc(ensemble_scores))
    for text_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = pd.DataFrame({"StudyInstanceUID": merged["StudyInstanceUID"]})
        for label in LABELS:
            blend[label] = (
                text_weight * merged[f"{label}_text"] + (1 - text_weight) * merged[f"{label}_image"]
            )
        macro = macro_auc(per_label_auc(blend, truth))
        print(f"  {text_weight:.1f}  {macro:.3f}")
        if macro > best[1]:
            best = (text_weight, macro)
    print(f"\nBest blend: text_weight={best[0]:.1f} -> macro AUC {best[1]:.3f}")


if __name__ == "__main__":
    main()
