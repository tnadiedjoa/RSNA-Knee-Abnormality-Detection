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


def model_span(cfg: dict, span_pad: float = 0.0) -> tuple:
    """Slice range the windows cover: the training span, widened by span_pad at both ends (clipped to the stack)."""
    lo, hi = cfg.get("span", (0.08, 0.92))
    return max(0.0, lo - span_pad), min(1.0, hi + span_pad)


def _prepare(args):
    series_dir, out_dir, study, series, side, sizes = args
    try:
        blobs = [p.read_bytes() for p in sorted(Path(series_dir).glob("*.dcm"))]
        vol = load_volume(blobs, side=side)
        for size in sizes:  # as scripts/build_cache.py --size
            x, pos = to_uint8(vol, size=size)
            (size_dir(out_dir, size) / study).mkdir(parents=True, exist_ok=True)
            save_series(size_dir(out_dir, size) / study / f"{series}{SUFFIX}", x, pos)
        return {"study": study, "series": series, "cached_slices": len(x), "ok": True,  # + header fields of series_meta
                **{t: vol.meta.get(t) for t in ("MRAcquisitionType", "RepetitionTime", "EchoTime", "InversionTime",
                                                "ScanningSequence")}}
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


def model_windows(cfg: dict, windows: int | None = None) -> int:
    """Windows per series at inference: the test-time setting, except stacked-slice models (one image per series)."""
    return 1 if cfg.get("stack_slices") else (windows or cfg["windows_eval"])


def model_dataset(cfg: dict, slots: pd.DataFrame, cache_dir: Path, windows: int | None = None,
                  span_pad: float = 0.0) -> StudyDataset:
    return StudyDataset(slots, n_windows=model_windows(cfg, windows), img_size=cfg["img_size"],
                        span=model_span(cfg, span_pad), zoom=cfg.get("zoom", 1.0), cache_dir=cache_dir,
                        stack=cfg.get("stack_slices", 0))


@torch.no_grad()
def predict_model(model_dir: Path, slots: pd.DataFrame, cache_dir: Path, device, workers: int,
                  windows: int | None = None, span_pad: float = 0.0) -> np.ndarray:
    cfg = json.loads((model_dir / "config.json").read_text())
    windows = model_windows(cfg, windows)  # denser than in training helps a little (experiments.md, E17)
    ds = model_dataset(cfg, slots, cache_dir, windows, span_pad)
    dl = DataLoader(ds, batch_size=max(1, 48 // windows), num_workers=workers)
    models = []
    for ckpt in sorted(model_dir.glob("fold*.pt")):
        model = KneeNet.from_config(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        models.append(model.to(device).eval().to(memory_format=torch.channels_last))
    out = [[] for _ in models]
    for batch in dl:  # every batch is read once and goes through all fold networks
        x, pos, mask, seq = (batch[k].to(device) for k in ("x", "pos", "mask", "seq"))
        for i, model in enumerate(models):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, pos, mask, seq)
            out[i].append(torch.sigmoid(logits.float()).cpu().numpy())
    print(f"{model_dir.name}: {len(models)} fold networks done", flush=True)
    return np.mean([np.concatenate(o) for o in out], axis=0)


# forward cost per image relative to ConvNeXt-tiny at the same slice size (FLOPs), for the time budget
BACKBONE_COST = {"convnext_small": 1.95, "radimagenet_resnet50": 0.92, "resnet50": 0.92, "resnet34": 0.8}


def model_cost(model_dir: Path, cfg: dict, windows: int | None = None) -> float:
    """Relative inference cost of one model per study: pixels x windows x fold networks x backbone factor."""
    factor = next((v for k, v in BACKBONE_COST.items() if cfg["backbone"].startswith(k)), 1.0)
    return cfg["img_size"] ** 2 * model_windows(cfg, windows) * len(list(model_dir.glob("fold*.pt"))) * factor


def input_key(cfg: dict, windows: int | None = None, span_pad: float = 0.0) -> tuple:
    """Models with the same key read identical batches: cache, slice size, windows, span, zoom, stacked slices."""
    return (cfg.get("cache_size", 256), cfg["img_size"], model_windows(cfg, windows),
            model_span(cfg, span_pad), cfg.get("zoom", 1.0), cfg.get("stack_slices", 0))


@torch.no_grad()
def predict_group(model_dirs: list, cfgs: dict, slots: pd.DataFrame, cache_dir: Path, devices: list, workers: int,
                  windows: int | None = None, span_pad: float = 0.0) -> tuple[dict, float, float]:
    """predict_model for several models that read the same input (input_key): each batch is read once and
    goes through every fold network of every model, the networks spread round-robin over the GPUs, so
    consecutive launches alternate between them and both compute at once.
    Returns ({model_dir: mean fold probability}, seconds loading networks, seconds predicting)."""
    cfg = cfgs[model_dirs[0]]
    windows = model_windows(cfg, windows)
    ds = model_dataset(cfg, slots, cache_dir, windows, span_pad)
    dl = DataLoader(ds, batch_size=max(1, 48 // windows), num_workers=workers, pin_memory=devices[0].type == "cuda")
    t = time.time()
    nets = []  # (model_dir, device, network)
    for m in model_dirs:
        c = cfgs[m]
        for ckpt in sorted(m.glob("fold*.pt")):
            dev = devices[len(nets) % len(devices)]
            net = KneeNet.from_config(c)
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
            nets.append((m, dev, net.to(dev).eval().to(memory_format=torch.channels_last)))
    load_s, t = time.time() - t, time.time()
    out = {m: [] for m in model_dirs}
    for batch in dl:
        inputs = {d: [batch[k].to(d, non_blocking=True) for k in ("x", "pos", "mask", "seq")] for d in devices}
        logits = []
        for m, d, net in nets:  # launches only; .cpu() below waits for the GPUs
            with torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
                logits.append((m, net(*inputs[d])))
        folds = {m: [] for m in model_dirs}
        for m, lg in logits:
            folds[m].append(torch.sigmoid(lg.float()).cpu().numpy())
        for m in model_dirs:
            out[m].append(np.mean(folds[m], axis=0))
    for m in model_dirs:
        print(f"{m.name}: {sum(n[0] == m for n in nets)} fold networks done", flush=True)
    return {m: np.concatenate(out[m]) for m in model_dirs}, load_s, time.time() - t


def run(data_dir, model_dirs, out_csv="submission.csv", cache_dir="/tmp/knee_cache", workers=None,
        time_budget_s=None, windows=None, grouped=False, calib_studies=16, span_pad=0.0):
    """time_budget_s: skip the remaining models (in the given order) once the next one would end past
    this many seconds from the call; the blend then uses the models done so far.
    span_pad: widen every model's slice range by this much at both ends (test-time setting, experiments.md).
    grouped: models sharing an input run together on every GPU (predict_group); a first pass over
    calib_studies studies measures the speed, the models that fit the budget are chosen in the given
    order, and before each group the slowest-priority ones are dropped if the measured pace says so."""
    t0 = time.time()
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    workers = workers or os.cpu_count()
    test = pd.read_csv(data_dir / "test.csv")
    model_dirs = [Path(m) for m in model_dirs]
    assert model_dirs, "no model to run (MODELS patterns matched nothing in the dataset?)"
    cfgs = {m: json.loads((m / "config.json").read_text()) for m in model_dirs}
    sizes = {m: c.get("cache_size", 256) for m, c in cfgs.items()}
    series = prepare_cache(data_dir, cache_dir, workers, sorted(set(sizes.values())))
    slots = assign_slots(series).set_index("StudyInstanceUID").reindex(test["StudyInstanceUID"]).reset_index()
    if grouped:
        preds = _run_grouped(model_dirs, cfgs, slots, cache_dir, workers, windows, time_budget_s, t0, calib_studies,
                             span_pad)
        return _write(test, [preds[m] for m in model_dirs if m in preds], len(model_dirs), out_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, spent, pixels = [], 0.0, 0
    for model_dir in map(Path, model_dirs):
        cost = cfgs[model_dir]["img_size"] ** 2 * len(list(model_dir.glob("fold*.pt"))) * model_windows(cfgs[model_dir], windows)
        if preds and time_budget_s and time.time() - t0 + spent / pixels * cost * 1.2 > time_budget_s:
            print(f"time budget: skipping {model_dir.name}", flush=True)
            continue
        t = time.time()
        p = predict_model(model_dir, slots, size_dir(cache_dir, sizes[model_dir]), device, workers, windows, span_pad)
        spent, pixels = spent + time.time() - t, pixels + cost
        print(f"{model_dir.name}: {time.time() - t:.0f}s, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
        preds.append(p)
    return _write(test, preds, len(model_dirs), out_csv)


def _write(test: pd.DataFrame, preds: list, n_models: int, out_csv) -> pd.DataFrame:
    # several models: average ranks, so no model dominates through its calibration
    preds = [pd.DataFrame(p).rank(pct=True).to_numpy() for p in preds] if n_models > 1 else preds
    sub = pd.DataFrame(np.mean(preds, axis=0), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", test["StudyInstanceUID"].to_numpy())
    sub[LABELS] = sub[LABELS].fillna(0.5)
    sub.to_csv(out_csv, index=False)
    print(sub.head())
    return sub


def _run_grouped(model_dirs, cfgs, slots, cache_dir, workers, windows, time_budget_s, t0, calib_studies,
                 span_pad=0.0) -> dict:
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    print(f"devices: {[str(d) for d in devices]}", flush=True)
    costs = {m: model_cost(m, cfgs[m], windows) for m in model_dirs}
    groups: dict = {}  # input key -> its models, in the given order; groups run in the order of their first model
    for m in model_dirs:
        groups.setdefault(input_key(cfgs[m], windows, span_pad), []).append(m)
    n = len(slots)
    keep = list(model_dirs)
    rate, load = None, 0.0  # cost units x studies per second (all GPUs together); seconds to load one network
    if time_budget_s and n > calib_studies:  # speed of the first group on a few studies
        key, ms = next(iter(groups.items()))
        _, load_s, pred_s = predict_group(ms, cfgs, slots.iloc[:calib_studies], size_dir(cache_dir, key[0]), devices,
                                          workers, windows, span_pad)
        rate = sum(costs[m] for m in ms) * calib_studies / pred_s
        load = load_s / sum(len(list(m.glob("fold*.pt"))) for m in ms)
        left = time_budget_s - (time.time() - t0)
        keep, total = [], 0.0
        for m in model_dirs:  # given order = priority
            need = (costs[m] * n / rate + load * len(list(m.glob("fold*.pt")))) * 1.15
            if not keep or total + need <= left:
                keep.append(m)
                total += need
            else:
                print(f"time budget: skipping {m.name}", flush=True)
        print(f"calibration on {calib_studies} studies: {rate:.3g} units/s, {load:.1f} s per network load; "
              f"{total / 60:.0f} min planned for {len(keep)} models, {left / 60:.0f} min left", flush=True)
    preds, done, spent = {}, 0.0, 0.0
    for key, ms in groups.items():
        ms = [m for m in ms if m in keep]
        if preds and time_budget_s and rate:  # measured pace: drop this group's last models if they no longer fit
            left = time_budget_s - (time.time() - t0)
            while ms and sum(costs[m] for m in ms) * n / rate * 1.15 > left:
                print(f"time budget: skipping {ms.pop().name}", flush=True)
        if not ms:
            continue
        t = time.time()
        p, _, pred_s = predict_group(ms, cfgs, slots, size_dir(cache_dir, key[0]), devices, workers, windows, span_pad)
        preds.update(p)
        done, spent = done + sum(costs[m] for m in ms) * n, spent + pred_s
        rate = done / spent
        print(f"group {key[:3]}: {len(ms)} models in {time.time() - t:.0f}s, {(time.time() - t0) / 60:.1f} min elapsed",
              flush=True)
    return preds
