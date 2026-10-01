"""Build the image cache straight from the remote archive (nothing but the cache is stored).

For every series: one byte-range request, DICOM decode, canonical frame (knee/dicom_io.py),
150 mm centre crop, 256 px, uint8, slices stored as JPEG q95 (knee/cache_io.py). Resumable:
    <cache>/<study>/<series>.jpgs.npz  slices + positions
    <cache>/<study>/<series>.json      geometry, laterality, header tags
Then data/series_meta.parquet gathers all the JSONs, and --fix-laterality (run after the full
build) gives every series of a study the same knee side.

--convert re-encodes an older zlib cache (<series>.npz with an "x" array) in place.

Another resolution goes to its own folder and leaves data/series_meta.parquet untouched:
    KNEE_CACHE_DIR=data/cache384 python -m scripts.build_cache --size 384 --sides-from-meta --slots-only
(every study's side is read from the existing meta, so there is no --fix-laterality pass).
"""

import argparse
import json
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

from knee.cache_io import SUFFIX, save_series
from knee.config import CACHE_DIR, DATA_DIR
from knee.data import SLOTS, assign_slots, series_table
from knee.dicom_io import load_volume, study_side, to_uint8
from knee.remote_zip import RemoteArchive, fetch_members

_archive = None


def _init():
    global _archive
    _archive = RemoteArchive()


def process(item):
    study, series, members, side = item if len(item) == 4 else (*item, None)
    out = CACHE_DIR / study
    if (out / f"{series}.json").exists() and (side is None or ARGS.sides_from_meta):
        return series, "skip"
    t = time.time()
    try:
        vol = load_volume(fetch_members(_archive, members), side=side)
        x, pos = to_uint8(vol, size=ARGS.size, fov_mm=ARGS.fov, max_slices=ARGS.max_slices)
        out.mkdir(parents=True, exist_ok=True)
        save_series(out / f"{series}{SUFFIX}", x, pos)
        meta = dict(vol.meta, study=study, series=series, plane=vol.plane, laterality=vol.laterality,
                    laterality_source=vol.laterality_source, row_spacing=vol.spacing[0], col_spacing=vol.spacing[1],
                    slice_gap=float(np.median(np.diff(vol.positions))) if len(vol.positions) > 1 else None,
                    cached_slices=len(x), seconds=round(time.time() - t, 2))
        (out / f"{series}.json").write_text(json.dumps(meta, default=str))
        return series, "ok"
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{series}.err").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return series, "error"


def convert(path):
    """Old zlib .npz -> JPEG format; the JSON metadata stays valid."""
    try:
        with np.load(path) as f:
            x, pos = f["x"], f["pos"]
        save_series(path.with_name(path.name[: -len(".npz")] + SUFFIX), x, pos)
        path.unlink()
        return "ok"
    except Exception as e:
        return f"error {path}: {e}"


def study_sides(meta: pd.DataFrame) -> dict:
    infos = meta.assign(side=meta["laterality"], source=meta["laterality_source"])
    return {st: study_side(g[["side", "source", "center_x"]].to_dict("records")) for st, g in infos.groupby("study")}


def fix_laterality(index: pd.DataFrame) -> None:
    """Second pass: one knee side per study (dicom_io.study_side), re-fetching the series whose
    own evidence mirrored them differently. The test pipeline decides the side the same way."""
    meta = pd.read_parquet(DATA_DIR / "series_meta.parquet")
    sides = study_sides(meta)
    want = meta["study"].map(sides)
    todo = meta[(meta["laterality"] == "L") != (want == "L")]
    print(f"re-orienting {len(todo)} series in {todo['study'].nunique()} studies", flush=True)
    dcm = index[index["name"].str.endswith(".dcm")].groupby("series")
    items = [(st, se, dcm.get_group(se)[["lho", "csize", "method"]], sides[st]) for st, se in zip(todo["study"], todo["series"])]
    with Pool(ARGS.workers, initializer=_init) as pool:
        res = pool.map(process, items, chunksize=2)
    print({r: [x[1] for x in res].count(r) for r in {"ok", "error"}}, flush=True)


def gather_meta():
    rows = [json.loads(p.read_text()) for p in CACHE_DIR.glob("*/*.json")]
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "series_meta.parquet", index=False)
    print(f"series_meta.parquet: {len(df)} series")


def main():
    if ARGS.convert:
        old = sorted(p for p in CACHE_DIR.glob("*/*.npz") if not p.name.endswith(SUFFIX))
        old = old[ARGS.shard :: ARGS.nshards]
        print(f"converting {len(old)} series", flush=True)
        with Pool(ARGS.workers) as pool:
            res = pool.map(convert, old, chunksize=16)
        errors = [r for r in res if r != "ok"]
        print(f"{len(res) - len(errors)} converted, {len(errors)} errors", errors[:5], flush=True)
        return
    index = pd.read_parquet(DATA_DIR / "zip_index.parquet")
    dcm = index[index["name"].str.endswith(".dcm")]
    items = [(study, series, m[["lho", "csize", "method"]]) for (study, series), m in dcm.groupby(["study", "series"])]
    if ARGS.sides_from_meta:
        sides = study_sides(pd.read_parquet(DATA_DIR / "series_meta.parquet"))
        items = [(st, se, m, sides[st]) for st, se, m in items if st in sides]
    if ARGS.slots_only:
        slots = assign_slots(series_table("train"))
        keep = {se for name, _, _ in SLOTS for se in slots[name] if isinstance(se, str)}
        items = [it for it in items if it[1] in keep]
    items = items[ARGS.shard :: ARGS.nshards]
    if ARGS.limit:
        items = items[: ARGS.limit]
    if ARGS.fix_laterality:
        fix_laterality(index)
        gather_meta()
        return
    print(f"{len(items)} series, {ARGS.workers} workers", flush=True)
    t, counts = time.time(), {}
    with Pool(ARGS.workers, initializer=_init) as pool:
        for i, (_, status) in enumerate(pool.imap_unordered(process, items, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0 or i == len(items):
                rate = i / (time.time() - t)
                print(f"{i}/{len(items)} {counts} {rate:.1f} series/s, eta {(len(items) - i) / rate / 60:.0f} min", flush=True)
    if not ARGS.sides_from_meta:
        gather_meta()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fov", type=float, default=150.0)
    parser.add_argument("--max-slices", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--fix-laterality", action="store_true", help="after a full build: one side per study")
    parser.add_argument("--sides-from-meta", action="store_true", help="final sides from data/series_meta.parquet")
    parser.add_argument("--slots-only", action="store_true", help="only the train series the model reads")
    ARGS = parser.parse_args()
    main()
