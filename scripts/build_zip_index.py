"""Read the competition archive's central directory and save one row per member to
data/zip_index.parquet (name, method, csize, usize, lho, split, study, series)."""

import time

from knee.config import DATA_DIR
from knee.remote_zip import RemoteArchive, read_index


def main():
    t = time.time()
    df = read_index(RemoteArchive())
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(DATA_DIR / "zip_index.parquet", index=False)
    dcm = df[df["name"].str.endswith(".dcm")]
    print(f"{len(df)} members ({len(dcm)} DICOMs, {dcm['study'].nunique()} studies, "
          f"{dcm['series'].nunique()} series) in {time.time() - t:.0f}s")
    print(df.groupby("split")[["csize", "usize"]].sum().div(2**30).round(1).rename(columns=lambda c: c + "_GB"))
    print("compression methods:", df["method"].value_counts().to_dict())


if __name__ == "__main__":
    main()
