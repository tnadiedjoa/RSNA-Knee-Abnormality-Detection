"""Does a test run add to our best CV members? Per fold, on the studies every run scored:

    python -m scripts.cv_compare <test run> [<test run> ...] [--folds 0 1] [--base e79 e46] [--others e42]

silver / gold of each run alone, its rank correlation with the base runs, and the silver / gold of the base rank average
with the test run added at 0.2 and 0.33 (an addition that helps here is a candidate member for the Kaggle blends).
Runs are given by name fragment (e.g. e114): the folder runs/*_<fragment>_* holding oof_fold<k>.csv for that fold.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from knee.config import LABELS, ROOT_DIR
from scripts.eval_oof import score, targets


def fold_oof(fragment: str, fold: int) -> pd.DataFrame | None:
    for run in sorted((ROOT_DIR / "runs").glob(f"*_{fragment}_*")):
        if f"_from_{fragment}" in run.name:  # continuations name their parent: not that run
            continue
        f = run / f"oof_fold{fold}.csv"
        if f.exists():
            return pd.read_csv(f).set_index("StudyInstanceUID")[LABELS]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--base", nargs="+", default=["e79", "e46"])
    p.add_argument("--others", nargs="*", default=["e42"])
    p.add_argument("--labels", default="qwen36_27b_v1")
    args = p.parse_args()
    t = targets(args.labels)
    for fold in args.folds:
        runs = {k: fold_oof(k, fold) for k in args.base + args.others + args.tests}
        missing = [k for k, v in runs.items() if v is None]
        runs = {k: v for k, v in runs.items() if v is not None}
        if not all(k in runs for k in args.base):
            print(f"fold {fold}: base run missing ({missing})")
            continue
        idx = sorted(set.intersection(*[set(v.index) for v in runs.values()]))
        rk = {k: v.loc[idx].rank(pct=True) for k, v in runs.items()}
        print(f"== fold {fold}: {len(idx)} studies" + (f" (no fold file: {', '.join(missing)})" if missing else ""))
        for k, v in rk.items():
            s = score(v, t)
            print(f"  {k:6s} silver {s['silver']:.4f}  gold {s['gold']:.3f}")
        corr = lambda a, b: np.mean([rk[a][c].corr(rk[b][c], method="spearman") for c in LABELS])
        base = sum(rk[k] for k in args.base) / len(args.base)
        sb = score(base, t)
        print(f"  base {' + '.join(args.base)}: silver {sb['silver']:.4f}  gold {sb['gold']:.3f}")
        for k in args.others + args.tests:
            if k not in rk:
                continue
            cs = " ".join(f"{b} {corr(k, b):.3f}" for b in args.base)
            adds = []
            for w in (0.2, 0.33):
                s = score((1 - w) * base + w * rk[k], t)
                adds.append(f"+{w:.2f}: silver {s['silver'] - sb['silver']:+.4f} gold {s['gold'] - sb['gold']:+.3f}")
            print(f"  {k:6s} corr {cs} | " + " | ".join(adds))


if __name__ == "__main__":
    main()
