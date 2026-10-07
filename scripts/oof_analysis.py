"""Where out-of-fold predictions fail: silver AUC by study subgroup, and whether a scanner / sex prior helps.

    python -m scripts.oof_analysis subgroups runs/<run> [runs/<run2> ...]
    python -m scripts.oof_analysis site-prior runs/<run>

subgroups: rank-averaged OOF of the runs, silver AUC by number of series, slice count, field strength, maker and
site proxy (maker | model | series-description style), with the gold studies each site holds.
site-prior: per finding, a logistic regression on the OOF logit + one-hot scanner group (maker | model | field,
groups under 30 studies pooled) + sex, nested over the 5 folds.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from knee.config import DATA_DIR, LABELS
from knee.data import series_table
from scripts.eval_oof import load_oof, score, targets
from scripts.train import macro_auc


def _style(desc) -> str:
    d = str(desc)
    if d.startswith("DummySeriesDesc"):
        return "dummy"
    if re.search(r"_tse_|_vibe_|tse_fs", d):
        return "siemens_style"
    if re.search(r"(RT|LT)\.(Sag|Cor|Tra)_", d):
        return "philips_dot"
    if re.search(r"SPAIR|TSE|W_", d):
        return "philips"
    return "other"


def _study_meta(index) -> pd.DataFrame:
    ser = series_table("train")
    first = ser.groupby("StudyInstanceUID").first()
    meta = pd.DataFrame({
        "maker": first.Manufacturer.fillna("?").str.split().str[0].str.upper(),
        "model": first.ManufacturerModelName.fillna("?"),
        "field": first.MagneticFieldStrength.astype(str),
        "sex": first.PatientSex.fillna("?"),
        "style": ser.groupby("StudyInstanceUID").SeriesDescription.agg(lambda s: _style(s.iloc[0])),
        "n_series": ser.groupby("StudyInstanceUID").size().clip(upper=9),
        "max_slices": pd.cut(ser.groupby("StudyInstanceUID").cached_slices.max(), [0, 30, 45, 80, 2000]),
    })
    return meta.reindex(index)


def subgroups(runs):
    oofs = [load_oof(Path(r)) for r in runs]
    idx = oofs[0].index
    pred = sum(o.loc[idx, LABELS].rank(pct=True) for o in oofs) / len(oofs)
    t = targets("qwen36_27b_v1").loc[idx]
    meta = _study_meta(idx)
    meta["site"] = meta.maker + "|" + meta.model + "|" + meta["style"]
    y, p = t[LABELS].to_numpy(), pred.to_numpy()
    print(f"all {len(y)} studies: silver {macro_auc(y, p)[0]:.4f}")
    for col in ["n_series", "max_slices", "field", "maker"]:
        rows = [(str(k), len(ii), round(macro_auc(y[ii], p[ii])[0], 4))
                for k, ii in meta.groupby(col, observed=True).indices.items() if len(ii) >= 40]
        print(col, rows)
    rows = [(k, len(ii), round(meta.n_series.iloc[ii].mean(), 1), round(macro_auc(y[ii], p[ii])[0], 4),
             int(t["is_gold"].iloc[ii].sum())) for k, ii in meta.groupby("site").indices.items() if len(ii) >= 60]
    print(pd.DataFrame(rows, columns=["site", "studies", "series", "silver", "gold"]).sort_values("silver").to_string(index=False))


def site_prior(run):
    oof = load_oof(Path(run))
    t = targets("qwen36_27b_v1").loc[oof.index]
    folds = pd.read_csv(DATA_DIR / "folds.csv").set_index("StudyInstanceUID").loc[oof.index, "fold"].to_numpy()
    meta = _study_meta(oof.index)
    scanner = meta.maker + "|" + meta.model + "|" + meta.field
    scanner = scanner.where(scanner.map(scanner.value_counts()) >= 30, "other")
    x_meta = pd.get_dummies(pd.DataFrame({"scanner": scanner, "sex": meta.sex})).astype(float).to_numpy()
    adj = oof[LABELS].copy()
    for label in LABELS:
        p = np.clip(oof[label].to_numpy(), 1e-4, 1 - 1e-4)
        x = np.hstack([np.log(p / (1 - p))[:, None], x_meta])
        y = t[label].to_numpy()
        ok, out = ~np.isnan(y), np.zeros(len(y))
        for f in range(5):
            fit = LogisticRegression(C=0.3, max_iter=2000).fit(x[ok & (folds != f)], y[ok & (folds != f)])
            out[folds == f] = fit.decision_function(x[folds == f])
        adj[label] = out
    a, b = score(oof[LABELS], t), score(adj, t)
    print(f"{scanner.nunique()} scanner groups: silver {a['silver']:.4f} -> {b['silver']:.4f}, gold {a['gold']:.4f} -> {b['gold']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["subgroups", "site-prior"])
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    if args.what == "subgroups":
        subgroups(args.runs)
    else:
        site_prior(args.runs[0])


if __name__ == "__main__":
    main()
