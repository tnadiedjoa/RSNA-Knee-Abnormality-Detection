"""Agreement of report-derived label sets with the 58 gold studies (per-finding AUC).

    python -m scripts.eval_labels qwen36_v1 [more label-set names...]
Public tables in data/public_labels/ are included for reference only (several were produced
with hosted LLM APIs, so they are not used for training).
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.config import DATA_DIR, LABELS
from knee.labels import gold, load_soft


def public_sets() -> dict:
    root = DATA_DIR / "public_labels"
    sets = {}
    for name, path in [("pilkwang_v2", "pilkwang/report_labels_v2.csv"), ("steven_v2", "stevenleehans/llm_labels_v2.csv"),
                       ("lixin_sol56", "lixin73/labels_llm_gpt56sol.csv")]:
        if (root / path).exists():
            df = pd.read_csv(root / path)
            if all(label in df.columns for label in LABELS):
                sets[name] = df[["StudyInstanceUID"] + LABELS]
    return sets


def auc_table(sets: dict) -> pd.DataFrame:
    g = gold()
    rows = {}
    for name, df in sets.items():
        m = g.merge(df, on="StudyInstanceUID", suffixes=("", "_p"), how="left")
        rows[name] = {label: roc_auc_score(m[label], m[f"{label}_p"].fillna(0.5)) for label in LABELS}
    table = pd.DataFrame(rows).T
    table["macro"] = table.mean(axis=1)
    return table


if __name__ == "__main__":
    sets = {name: load_soft(name) for name in sys.argv[1:]} | public_sets()
    pd.set_option("display.width", 250)
    t = auc_table(sets)
    t.columns = [c[:8] for c in t.columns]
    print(t.round(3).to_string())
    if sys.argv[1:]:
        soft = load_soft(sys.argv[1])
        print("\nmention rate (all reports):", soft[[f"{l}__mentioned" for l in LABELS]].mean().round(2).values)
        print("mean soft label:", soft[LABELS].mean().round(2).values)
        print("gold prevalence:", gold()[LABELS].mean().round(2).values)
    _ = np
