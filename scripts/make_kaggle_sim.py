"""Lay out training studies like Kaggle's hidden test set, to run public inference pipelines locally.

    python -m scripts.make_kaggle_sim /tmp/kaggle_sim [--gold] [--studies UID ...] [--workers 8]

Writes <out>/rsna-knee-abnormality-detection/{test.csv, test_series.csv, sample_submission.csv,
test_series/<study>/<series>/<instance>.dcm} for the chosen training studies (default: the 58 gold studies),
fetched from the competition archive by HTTP range requests (knee/remote_zip.py). Local use only: the competition
data never leaves the cluster (no versioning, no upload).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from knee.config import DATA_DIR, LABELS
from knee.labels import gold
from knee.remote_zip import RemoteArchive, fetch_members


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--studies", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    studies = args.studies or gold()["StudyInstanceUID"].tolist()
    root = Path(args.out) / "rsna-knee-abnormality-detection"
    (root / "test_series").mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(DATA_DIR / "train_series.csv")
    series = series[series["StudyInstanceUID"].isin(studies)]
    pd.DataFrame({"StudyInstanceUID": studies}).to_csv(root / "test.csv", index=False)
    series.to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": studies, **{label: 0.5 for label in LABELS}}).to_csv(root / "sample_submission.csv", index=False)
    index = pd.read_parquet(DATA_DIR / "zip_index.parquet")
    dcm = index[index["name"].str.endswith(".dcm") & index["study"].isin(studies)]
    archive = RemoteArchive()

    def one(item):
        (study, ser), members = item
        d = root / "test_series" / study / ser
        if d.is_dir() and len(list(d.iterdir())) == len(members):
            return len(members)
        d.mkdir(parents=True, exist_ok=True)
        for name, data in zip(members["name"], fetch_members(archive, members)):
            (d / name.rsplit("/", 1)[-1]).write_bytes(data)
        return len(members)

    with ThreadPoolExecutor(args.workers) as pool:
        n = sum(pool.map(one, dcm.groupby(["study", "series"])))
    print(f"{len(studies)} studies, {len(series)} series, {n} DICOM files in {root}")


if __name__ == "__main__":
    main()
