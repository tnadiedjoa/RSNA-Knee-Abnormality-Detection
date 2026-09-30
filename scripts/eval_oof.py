"""Out-of-fold scores of one or more runs, and of their rank-average blend.

    python -m scripts.eval_oof runs/<run1> [runs/<run2> ...] [--labels qwen36_27b_v1]

silver = confident report labels (absent vs moderate/severe) over all training studies,
gold = the 58 expert-labelled studies (each scored by the fold that did not train on it).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from knee.config import DATA_DIR, LABELS
from scripts.train import macro_auc


def targets(labels: str) -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv").set_index("StudyInstanceUID")
    grades = pd.read_parquet(DATA_DIR / "labels" / f"{labels}.parquet").set_index("StudyInstanceUID")[LABELS]
    silver = pd.DataFrame(np.where(grades.isin(["moderate", "severe"]), 1.0, np.where(grades == "absent", 0.0, np.nan)),
                          index=grades.index, columns=LABELS)
    gold = train[LABELS].dropna()
    silver.loc[gold.index] = gold.to_numpy()
    silver["is_gold"] = silver.index.isin(gold.index)
    return silver


def load_oof(run: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(run.glob("oof_fold*.csv"))]).set_index("StudyInstanceUID")


def score(pred: pd.DataFrame, t: pd.DataFrame) -> dict:
    t = t.loc[pred.index]
    s, s_per = macro_auc(t[LABELS].to_numpy(), pred[LABELS].to_numpy())
    g = t["is_gold"].to_numpy()
    gd, g_per = macro_auc(t.loc[g, LABELS].to_numpy(), pred.loc[g, LABELS].to_numpy())
    return {"silver": s, "gold": gd, "n": len(pred), "n_gold": int(g.sum()), "silver_per": s_per, "gold_per": g_per}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", default="qwen36_27b_v1")
    args = p.parse_args()
    t = targets(args.labels)
    preds = {Path(r).name: load_oof(Path(r)) for r in args.runs}
    common = sorted(set.intersection(*(set(p.index) for p in preds.values())))
    rows, per = {}, {}
    for name, pr in preds.items():
        sc = score(pr.loc[common], t)
        rows[name], per[name] = sc, sc["gold_per"]
    if len(preds) > 1:
        blend = sum(pr.loc[common, LABELS].rank(pct=True) for pr in preds.values()) / len(preds)
        sc = score(blend, t)
        rows["rank blend"], per["rank blend"] = sc, sc["gold_per"]
    table = pd.DataFrame({k: {"silver AUC": v["silver"], "gold AUC": v["gold"], "studies": v["n"], "gold n": v["n_gold"]}
                          for k, v in rows.items()}).T
    print(table.round(4).to_string())
    print("\ngold AUC per finding:")
    print(pd.DataFrame(per, index=[l[:10] for l in LABELS]).T.round(3).to_string())
    print("\nsilver AUC per finding:")
    print(pd.DataFrame({k: v["silver_per"] for k, v in rows.items()}, index=[l[:10] for l in LABELS]).T.round(3).to_string())


if __name__ == "__main__":
    main()
