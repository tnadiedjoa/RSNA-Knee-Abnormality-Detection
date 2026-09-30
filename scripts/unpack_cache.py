"""Unpack the JPEG cache to raw uint8 arrays on a node's local disk, for GPUs whose jobs have few
CPUs (JPEG decoding is then the bottleneck). Only the series used as model inputs are unpacked.

    python -m scripts.unpack_cache /tmp/knee_raw [--workers 4]
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

from knee.cache_io import SUFFIX, unpack
from knee.config import CACHE_DIR
from knee.data import SLOTS, assign_slots, series_table


def _job(args):
    src, dst = args
    if not dst.exists():
        unpack(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dst")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    slots = assign_slots(series_table("train"))
    jobs = [(CACHE_DIR / st / f"{se}{SUFFIX}", Path(args.dst) / st / f"{se}.npy")
            for _, row in slots.iterrows() for st, se in [(row["StudyInstanceUID"], row[name]) for name, _, _ in SLOTS]
            if isinstance(se, str)]
    t = time.time()
    with Pool(args.workers) as pool:
        pool.map(_job, jobs, chunksize=16)
    print(f"unpacked {len(jobs)} series to {args.dst} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
