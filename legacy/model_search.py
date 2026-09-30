"""Compare several TF-IDF configs and classifiers under the same 5-fold split,
to pick a better text-model config than the char(3,5)+LogisticRegression default.
Prints macro AUC for each config; does not change any saved model on its own —
update src/text_baseline.build_pipeline's defaults by hand if a config wins.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from src.config import RANDOM_STATE
from src.data import load_labeled_train
from src.folds import make_folds
from src.metrics import macro_auc, per_label_auc
from src.text_baseline import build_pipeline, compute_oof

CONFIGS = {
    "char(3,5) + LogisticRegression [current default]": lambda: build_pipeline(
        analyzer="char_wb", ngram_range=(3, 5)
    ),
    "char(2,4) + LogisticRegression": lambda: build_pipeline(analyzer="char_wb", ngram_range=(2, 4)),
    "char(4,6) + LogisticRegression": lambda: build_pipeline(analyzer="char_wb", ngram_range=(4, 6)),
    "word(1,2) + LogisticRegression": lambda: build_pipeline(analyzer="word", ngram_range=(1, 2)),
    "char(3,5) + MultinomialNB": lambda: build_pipeline(
        analyzer="char_wb", ngram_range=(3, 5), classifier=MultinomialNB()
    ),
    "char(3,5) + RandomForest": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(3, 5),
        classifier=RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(3,5) + SGD(log_loss)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(3, 5),
        classifier=SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(4,6) + SGD(log_loss)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(4, 6),
        classifier=SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(3,6) + SGD(log_loss)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(3, 6),
        classifier=SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(4,6) + SGD(log_loss, alpha=1e-3)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(4, 6),
        classifier=SGDClassifier(
            loss="log_loss", alpha=1e-3, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(4,6) + SGD(log_loss, alpha=1e-2)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(4, 6),
        classifier=SGDClassifier(
            loss="log_loss", alpha=1e-2, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(4,6) + SGD(log_loss, alpha=5e-4)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(4, 6),
        classifier=SGDClassifier(
            loss="log_loss", alpha=5e-4, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "char(4,6) + SGD(log_loss, alpha=2e-3)": lambda: build_pipeline(
        analyzer="char_wb",
        ngram_range=(4, 6),
        classifier=SGDClassifier(
            loss="log_loss", alpha=2e-3, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
}


def main():
    df = load_labeled_train()
    folds = make_folds(len(df))

    results = []
    for name, factory in CONFIGS.items():
        oof = compute_oof(df, folds, pipeline_factory=factory)
        scores = per_label_auc(oof, df)
        macro = macro_auc(scores)
        results.append((macro, name))
        print(f"{macro:.3f}  {name}")

    results.sort(reverse=True)
    print(f"\nBest: {results[0][1]} ({results[0][0]:.3f})")


if __name__ == "__main__":
    main()
