"""Study-level dataset: a few series ("slots") per study, several 2.5D windows per series.

A window is 3 adjacent slices stacked as RGB, so an ImageNet 2D backbone sees local 3D context.
Frames are already canonical (knee/dicom_io.py): no horizontal flip here, it would swap
medial and lateral.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from knee.cache_io import SUFFIX, RawSeriesReader, SeriesReader
from knee.config import CACHE_DIR, DATA_DIR

# (name, plane, fluid-sensitive fat-suppressed?)
SLOTS = [("sag_fs", "sagittal", 1), ("cor_fs", "coronal", 1), ("ax_fs", "axial", 1),
         ("sag_t", "sagittal", 0), ("cor_t", "coronal", 0)]
# weighting of the series in a slot, from its header times (KneeNet --seq-emb): the sag_t slot alone
# holds T1 (39 %), PD (41 %) and T2 (20 %) series of the training set
SEQ_CLASSES = ["unknown", "T1", "PD", "T2", "STIR", "GRE"]


def series_table(split: str = "train") -> pd.DataFrame:
    """Cached series with their CSV descriptors (plane, fat suppression) and geometry meta."""
    meta = pd.read_parquet(DATA_DIR / "series_meta.parquet")
    csv = pd.read_csv(DATA_DIR / f"{split}_series.csv")
    df = csv.merge(meta, left_on=["StudyInstanceUID", "SeriesInstanceUID"], right_on=["study", "series"], how="inner")
    df["csv_plane"] = df["Anatomical_Plane"].str.lower()
    return df


def sequence_classes(series: pd.DataFrame) -> pd.Series:
    """Index into SEQ_CLASSES per series: TI 80-300 ms -> STIR, gradient echo -> GRE, TR < 1000 ms -> T1,
    TE >= 60 ms -> T2, else PD; unknown when TR or TE is missing."""
    def num(col):
        return pd.to_numeric(series[col], errors="coerce") if col in series else pd.Series(float("nan"), index=series.index)
    tr, te, ti = num("RepetitionTime"), num("EchoTime"), num("InversionTime")
    gre = series["ScanningSequence"].astype(str).eq("GR") if "ScanningSequence" in series else False
    cls = np.select([tr.isna() | te.isna(), ti.between(80, 300), gre, tr < 1000, te >= 60], [0, 4, 5, 1, 3], 2)
    return pd.Series(cls, index=series["SeriesInstanceUID"].to_numpy())


def assign_slots(series: pd.DataFrame) -> pd.DataFrame:
    """One series per slot per study (None when the study has no candidate), and its SEQ_CLASSES
    index in "<slot>_seq".

    Candidates match the CSV plane and fat-suppression flag; ties go to 2D acquisitions,
    then to the series with the most slices. An empty slot borrows another unused series
    of the same plane, since a second view of that plane beats nothing."""
    rows = []
    for study, g in series.groupby("StudyInstanceUID"):
        g = g.assign(is2d=(g["MRAcquisitionType"].fillna("2D") == "2D").astype(int))
        g = g.sort_values(["is2d", "cached_slices", "SeriesInstanceUID"], ascending=[False, False, True])
        row, used = {"StudyInstanceUID": study}, set()
        for name, plane, fs in SLOTS:
            cand = g[(g["csv_plane"] == plane) & (g["Fat_Suppression"] == fs) & ~g["SeriesInstanceUID"].isin(used)]
            row[name] = cand["SeriesInstanceUID"].iloc[0] if len(cand) else None
            if row[name]:
                used.add(row[name])
        for name, plane, _ in SLOTS:
            if row[name] is None:
                cand = g[(g["csv_plane"] == plane) & ~g["SeriesInstanceUID"].isin(used)]
                if len(cand):
                    row[name] = cand["SeriesInstanceUID"].iloc[0]
                    used.add(row[name])
        rows.append(row)
    out, seq = pd.DataFrame(rows), sequence_classes(series)
    for name, _, _ in SLOTS:
        out[f"{name}_seq"] = out[name].map(seq).fillna(0).astype(int)
    return out


def window_centers(n_slices: int, n_windows: int, lo: float, hi: float, jitter: float, rng) -> np.ndarray:
    """Evenly spaced slice indices over [lo, hi] of the stack, optionally jittered."""
    q = np.linspace(lo, hi, n_windows)
    if jitter:
        q = np.clip(q + rng.uniform(-jitter, jitter, n_windows), 0, 1)
    return np.clip(np.round(q * (n_slices - 1)).astype(int), 0, n_slices - 1)


def _affine(rng, size_in: int, size_out: int, strength: float, zoom: float = 1.0) -> np.ndarray:
    """Input -> output pixel map: resize to size_out (zoom > 1 keeps only the central 1/zoom of the
    cached field of view), plus a random rotation/zoom/shift in training."""
    angle = rng.uniform(-10, 10) * strength
    scale = math.exp(rng.uniform(-0.1, 0.12) * strength) * size_out / size_in * zoom
    tx, ty = rng.uniform(-0.05, 0.05, 2) * size_out * strength
    m = cv2.getRotationMatrix2D((size_in / 2, size_in / 2), angle, scale)
    m[:, 2] += (size_out - size_in) / 2 + tx, (size_out - size_in) / 2 + ty
    return m


def _is_missing(series) -> bool:
    return series is None or (isinstance(series, float) and np.isnan(series))


class StudyDataset(Dataset):
    def __init__(self, slots: pd.DataFrame, labels: np.ndarray | None = None, weights: np.ndarray | None = None,
                 n_windows: int = 10, img_size: int = 224, train: bool = False, aug_strength: float = 1.0,
                 span: tuple[float, float] = (0.08, 0.92), zoom: float = 1.0, seed: int = 0, cache_dir=CACHE_DIR,
                 raw_dir=None):
        self.slots = slots.reset_index(drop=True)
        self.labels, self.weights = labels, weights
        self.n_windows, self.img_size, self.train = n_windows, img_size, train
        self.aug_strength, self.span, self.zoom, self.seed = aug_strength, span, zoom, seed
        self.cache_dir, self.raw_dir = cache_dir, raw_dir
        self.epoch = 0

    def __len__(self):
        return len(self.slots)

    def _load(self, study: str, series: str):
        if self.raw_dir is not None:  # unpacked copy on local disk (scripts/unpack_cache.py): no JPEG decoding
            return RawSeriesReader(self.raw_dir / study / f"{series}.npy")
        return SeriesReader(self.cache_dir / study / f"{series}{SUFFIX}")

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else np.random.default_rng(i)
        row = self.slots.iloc[i]
        study = row["StudyInstanceUID"]
        n_slots, t, s = len(SLOTS), self.n_windows, self.img_size
        x = np.zeros((n_slots, t, 3, s, s), np.uint8)
        pos = np.zeros((n_slots, t), np.float32)
        mask = np.zeros(n_slots, bool)
        seq = np.zeros(n_slots, np.int64)
        for k, (name, _, _) in enumerate(SLOTS):
            series = row[name]
            if _is_missing(series):
                continue
            seq[k] = 0 if pd.isna(row.get(f"{name}_seq")) else int(row[f"{name}_seq"])
            vol = self._load(study, series)
            n = len(vol)
            centers = window_centers(n, t, *self.span, jitter=0.03 if self.train else 0.0, rng=rng)
            needed = np.unique(np.clip(np.concatenate([centers - 1, centers, centers + 1]), 0, n - 1))
            frames, m = {}, None
            for z in needed:  # each slice decoded and transformed once, even when windows share it
                img = vol.slice(z)
                if m is None:
                    m = _affine(rng, img.shape[-1], s, self.aug_strength if self.train else 0.0, self.zoom)
                frames[z] = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
            for j, c in enumerate(centers):
                for ch, z in enumerate(np.clip([c - 1, c, c + 1], 0, n - 1)):
                    x[k, j, ch] = frames[z]
            if self.train:  # gamma + contrast through a 256-entry lookup table
                gamma = math.exp(rng.uniform(-0.25, 0.25) * self.aug_strength)
                contrast = 1 + rng.uniform(-0.15, 0.15) * self.aug_strength
                lut = np.clip(((np.arange(256) / 255.0) ** gamma - 0.5) * contrast + 0.5, 0, 1)
                x[k] = (lut * 255).astype(np.uint8)[x[k]]
            pos[k] = centers / max(n - 1, 1)
            mask[k] = True
        out = {"x": torch.from_numpy(x), "pos": torch.from_numpy(pos), "mask": torch.from_numpy(mask),
               "seq": torch.from_numpy(seq)}
        if self.labels is not None:
            out["y"] = torch.from_numpy(self.labels[i].astype(np.float32))
            out["w"] = torch.from_numpy(self.weights[i].astype(np.float32))
        return out
