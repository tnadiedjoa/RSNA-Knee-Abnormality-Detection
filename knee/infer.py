"""Test-time pipeline (Kaggle notebook, internet off): DICOM folders -> submission.csv.

Preprocessing is the training code itself (knee/dicom_io.py + knee/cache_io.py), so train and
test inputs are identical. Every series is decoded once into a local cache, then each model
reads it.

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


def _prepare(args):
    series_dir, out_dir, study, series, side = args
    try:
        blobs = [p.read_bytes() for p in sorted(Path(series_dir).glob("*.dcm"))]
        vol = load_volume(blobs, side=side)
        x, pos = to_uint8(vol)
        (out_dir / study).mkdir(parents=True, exist_ok=True)
        save_series(out_dir / study / f"{series}{SUFFIX}", x, pos)
        return {"study": study, "series": series, "MRAcquisitionType": vol.meta.get("MRAcquisitionType"),
                "cached_slices": len(x), "ok": True}
    except Exception as e:  # a broken series must not sink the submission
        return {"study": study, "series": series, "ok": False, "error": repr(e)}


def prepare_cache(data_dir: Path, cache_dir: Path, workers: int) -> pd.DataFrame:
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
        jobs = [(d, cache_dir, st, se, sides[st])
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
def predict_model(model_dir: Path, slots: pd.DataFrame, cache_dir: Path, device, workers: int) -> np.ndarray:
    cfg = json.loads((model_dir / "config.json").read_text())
    ds = StudyDataset(slots, n_windows=cfg["windows_eval"], img_size=cfg["img_size"], cache_dir=cache_dir)
    dl = DataLoader(ds, batch_size=4, num_workers=workers)
    preds = []
    for ckpt in sorted(model_dir.glob("fold*.pt")):
        model = KneeNet(cfg["backbone"], pretrained=False, dim=cfg["dim"], depth=cfg["depth"], drop_path=0)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval().to(memory_format=torch.channels_last)
        out = []
        for batch in dl:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["x"].to(device), batch["pos"].to(device), batch["mask"].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        preds.append(np.concatenate(out))
        print(f"{model_dir.name}/{ckpt.name} done", flush=True)
    return np.mean(preds, axis=0)


def run(data_dir, model_dirs, out_csv="submission.csv", cache_dir="/tmp/knee_cache", workers=None):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    workers = workers or os.cpu_count()
    test = pd.read_csv(data_dir / "test.csv")
    series = prepare_cache(data_dir, cache_dir, workers)
    slots = assign_slots(series).set_index("StudyInstanceUID").reindex(test["StudyInstanceUID"]).reset_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for model_dir in map(Path, model_dirs):
        p = predict_model(model_dir, slots, cache_dir, device, workers)
        # several models: average ranks, so no model dominates through its calibration
        preds.append(pd.DataFrame(p).rank(pct=True).to_numpy() if len(model_dirs) > 1 else p)
    sub = pd.DataFrame(np.mean(preds, axis=0), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", test["StudyInstanceUID"].to_numpy())
    sub[LABELS] = sub[LABELS].fillna(0.5)
    sub.to_csv(out_csv, index=False)
    print(sub.head())
    return sub
