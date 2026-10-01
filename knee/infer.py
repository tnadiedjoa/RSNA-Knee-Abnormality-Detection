"""Test-time pipeline (Kaggle notebook, internet off): DICOM folders -> submission.csv.

Preprocessing is the training code itself (knee/dicom_io.py + knee/cache_io.py), so train and
test inputs are identical. Every series is decoded once into a local cache (one per slice size
the models were trained on, see config "cache_size"), then each model reads its cache.

    from knee.infer import run
    run(data_dir="/kaggle/input/competitions/rsna-knee-abnormality-detection",
        model_dirs=["/kaggle/input/knee-models/models/r34_base"], out_csv="submission.csv")
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import DataLoader

from knee.cache_io import SUFFIX, save_series
from knee.config import LABELS
from knee.data import StudyDataset, assign_slots
from knee.dicom_io import load_volume, series_side_info, study_side, to_uint8
from knee.model import KneeNet


def _side_info(series_dir):
    try:
        headers = [pydicom.dcmread(p, stop_before_pixels=True, force=True) for p in Path(series_dir).glob("*.dcm")]
        return series_side_info(headers)
    except Exception:
        return {"side": "U", "source": "none", "center_x": None}


def size_dir(cache_dir: Path, size: int) -> Path:
    """Test cache of one slice size; the 256 px one keeps the plain path."""
    return cache_dir if size == 256 else cache_dir.with_name(f"{cache_dir.name}{size}")


def _prepare(args):
    series_dir, out_dir, study, series, side, sizes = args
    try:
        blobs = [p.read_bytes() for p in sorted(Path(series_dir).glob("*.dcm"))]
        vol = load_volume(blobs, side=side)
        for size in sizes:  # as scripts/build_cache.py --size
            x, pos = to_uint8(vol, size=size)
            (size_dir(out_dir, size) / study).mkdir(parents=True, exist_ok=True)
            save_series(size_dir(out_dir, size) / study / f"{series}{SUFFIX}", x, pos)
        return {"study": study, "series": series, "MRAcquisitionType": vol.meta.get("MRAcquisitionType"),
                "cached_slices": len(x), "ok": True}
    except Exception as e:  # a broken series must not sink the submission
        return {"study": study, "series": series, "ok": False, "error": repr(e)}


def prepare_cache(data_dir: Path, cache_dir: Path, workers: int, sizes=(256,)) -> pd.DataFrame:
    series = pd.read_csv(data_dir / "test_series.csv")
    dirs = [data_dir / "test_series" / st / se for st, se in zip(series["StudyInstanceUID"], series["SeriesInstanceUID"])]
    t = time.time()
    with ProcessPoolExecutor(workers) as pool:
        # headers first: one knee side per study, exactly as the training cache (build_cache --fix-laterality)
        infos = list(pool.map(_side_info, dirs, chunksize=4))
        by_study: dict = {}
        for st, info in zip(series["StudyInstanceUID"], infos):
            by_study.setdefault(st, []).append(info)
        sides = {st: study_side(v) for st, v in by_study.items()}
        print(f"headers read in {time.time() - t:.0f}s, sides {pd.Series(sides).value_counts().to_dict()}", flush=True)
        jobs = [(d, cache_dir, st, se, sides[st], sizes)
                for d, st, se in zip(dirs, series["StudyInstanceUID"], series["SeriesInstanceUID"])]
        rows = list(pool.map(_prepare, jobs, chunksize=4))
    meta = pd.DataFrame(rows)
    print(f"decoded {int(meta['ok'].sum())}/{len(meta)} series in {time.time() - t:.0f}s", flush=True)
    if (~meta["ok"]).any():
        print(meta.loc[~meta["ok"], "error"].value_counts().head(), flush=True)
    df = series.merge(meta[meta["ok"]], left_on=["StudyInstanceUID", "SeriesInstanceUID"], right_on=["study", "series"])
    df["csv_plane"] = df["Anatomical_Plane"].str.lower()
    return df


@torch.no_grad()
def predict_model(model_dir: Path, slots: pd.DataFrame, cache_dir: Path, device, workers: int,
                  windows: int | None = None) -> np.ndarray:
    cfg = json.loads((model_dir / "config.json").read_text())
    windows = windows or cfg["windows_eval"]  # denser than in training helps a little (experiments.md, E17)
    ds = StudyDataset(slots, n_windows=windows, img_size=cfg["img_size"],
                      span=tuple(cfg.get("span", (0.08, 0.92))), zoom=cfg.get("zoom", 1.0), cache_dir=cache_dir)
    dl = DataLoader(ds, batch_size=max(1, 48 // windows), num_workers=workers)
    models = []
    for ckpt in sorted(model_dir.glob("fold*.pt")):
        model = KneeNet(cfg["backbone"], pretrained=False, dim=cfg["dim"], depth=cfg["depth"], drop_path=0)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        models.append(model.to(device).eval().to(memory_format=torch.channels_last))
    out = [[] for _ in models]
    for batch in dl:  # every batch is read once and goes through all fold networks
        x, pos, mask = batch["x"].to(device), batch["pos"].to(device), batch["mask"].to(device)
        for i, model in enumerate(models):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, pos, mask)
            out[i].append(torch.sigmoid(logits.float()).cpu().numpy())
    print(f"{model_dir.name}: {len(models)} fold networks done", flush=True)
    return np.mean([np.concatenate(o) for o in out], axis=0)


def run(data_dir, model_dirs, out_csv="submission.csv", cache_dir="/tmp/knee_cache", workers=None,
        time_budget_s=None, windows=None):
    """time_budget_s: skip the remaining models (in the given order) once the next one would end past
    this many seconds from the call; the blend then uses the models done so far."""
    t0 = time.time()
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    workers = workers or os.cpu_count()
    test = pd.read_csv(data_dir / "test.csv")
    cfgs = {Path(m): json.loads((Path(m) / "config.json").read_text()) for m in model_dirs}
    sizes = {m: c.get("cache_size", 256) for m, c in cfgs.items()}
    series = prepare_cache(data_dir, cache_dir, workers, sorted(set(sizes.values())))
    slots = assign_slots(series).set_index("StudyInstanceUID").reindex(test["StudyInstanceUID"]).reset_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, spent, pixels = [], 0.0, 0
    for model_dir in map(Path, model_dirs):
        cost = cfgs[model_dir]["img_size"] ** 2 * len(list(model_dir.glob("fold*.pt"))) * (windows or cfgs[model_dir]["windows_eval"])
        if preds and time_budget_s and time.time() - t0 + spent / pixels * cost * 1.2 > time_budget_s:
            print(f"time budget: skipping {model_dir.name}", flush=True)
            continue
        t = time.time()
        p = predict_model(model_dir, slots, size_dir(cache_dir, sizes[model_dir]), device, workers, windows)
        spent, pixels = spent + time.time() - t, pixels + cost
        print(f"{model_dir.name}: {time.time() - t:.0f}s, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
        # several models: average ranks, so no model dominates through its calibration
        preds.append(pd.DataFrame(p).rank(pct=True).to_numpy() if len(model_dirs) > 1 else p)
    sub = pd.DataFrame(np.mean(preds, axis=0), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", test["StudyInstanceUID"].to_numpy())
    sub[LABELS] = sub[LABELS].fillna(0.5)
    sub.to_csv(out_csv, index=False)
    print(sub.head())
    return sub
