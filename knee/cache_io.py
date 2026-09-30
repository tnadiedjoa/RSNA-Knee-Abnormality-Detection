"""Series cache format: one file per series, every slice a JPEG (quality 95), decoded on demand.

JPEG q95 halves the cache against zlib-compressed arrays (mean error < 1 grey level out of 255,
far below MRI noise), and only the slices a sample needs are decoded. The test pipeline writes
the same format, so train and test inputs go through the same round trip.
"""

from pathlib import Path

import cv2
import numpy as np

JPEG_QUALITY = 95
SUFFIX = ".jpgs.npz"


def save_series(path: Path, x: np.ndarray, pos: np.ndarray) -> None:
    blobs = [cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes() for im in x]
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    np.savez(path, data=np.frombuffer(b"".join(blobs), np.uint8), offsets=offsets, pos=pos.astype(np.float32))


class SeriesReader:
    def __init__(self, path: Path):
        with np.load(path) as f:
            self.data, self.offsets, self.pos = f["data"], f["offsets"], f["pos"]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def slice(self, i: int) -> np.ndarray:
        return cv2.imdecode(self.data[self.offsets[i] : self.offsets[i + 1]], cv2.IMREAD_GRAYSCALE)

    def volume(self) -> np.ndarray:
        return np.stack([self.slice(i) for i in range(len(self))])


class RawSeriesReader:
    """Same interface over an unpacked uint8 array (S, H, W) memory-mapped from local disk."""

    def __init__(self, path: Path):
        self.x = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.x)

    def slice(self, i: int) -> np.ndarray:
        return np.asarray(self.x[i])


def unpack(src: Path, dst: Path) -> None:
    """JPEG series -> raw .npy (decoded pixels are identical to on-the-fly decoding)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npy")
    np.save(tmp, SeriesReader(src).volume())
    tmp.rename(dst)
