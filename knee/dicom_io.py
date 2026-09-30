"""DICOM series -> canonical volume.

Every series is turned into the same frame, whatever the scanner wrote:
- slices sorted by their physical position (ImagePositionPatient projected on a fixed
  patient axis), never by file name or InstanceNumber;
- in-plane axes re-oriented to the standard radiological display of the plane;
- every knee mirrored to look like a RIGHT knee, so that the medial/lateral findings (menisci,
  OA compartments, MCL) always sit on the same side: image x (coronal/axial) and slice index
  (sagittal) both run lateral -> medial.

Patient coordinates are DICOM LPS: +x = patient left, +y = posterior, +z = superior.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pydicom

# canonical (image x, image y, slice order) axes per plane, in LPS
_CANONICAL = {
    "sagittal": (np.array([0, 1, 0]), np.array([0, 0, -1]), np.array([1, 0, 0])),  # x->posterior, y->inferior, slices->left
    "coronal": (np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])),  # x->left, y->inferior, slices->posterior
    "axial": (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])),  # x->left, y->posterior, slices->inferior
}
_PLANE_OF_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}
_LATERALITY_DEAD_ZONE_MM = 20.0
_LEFT_WORDS = re.compile(r"\b(left|links|izq\w*|gauche|sinistr\w*)\b")
_RIGHT_WORDS = re.compile(r"\b(right|rechts|derech\w*|dcha|droit\w*|destr\w*)\b")

HEADER_TAGS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SeriesDescription", "ScanningSequence",
    "SequenceVariant", "ScanOptions", "MRAcquisitionType", "SequenceName", "RepetitionTime", "EchoTime",
    "InversionTime", "FlipAngle", "EchoTrainLength", "SliceThickness", "SpacingBetweenSlices", "ReceiveCoilName",
    "SoftwareVersions", "PatientID", "PatientSex", "PatientAge", "BodyPartExamined", "Laterality", "ImageLaterality",
    "PatientPosition", "PhotometricInterpretation", "BitsStored",
]


@dataclass
class Volume:
    pixels: np.ndarray  # float32 (S, H, W), canonical frame
    spacing: tuple[float, float]  # (row, col) mm per pixel after re-orientation
    positions: np.ndarray  # (S,) slice positions (mm) along the canonical slice axis
    plane: str
    laterality: str  # "R", "L" or "U" (unknown -> left as acquired)
    laterality_source: str
    meta: dict = field(default_factory=dict)


def _tag(ds, name):
    v = getattr(ds, name, None)
    if v is None or v == "":
        return None
    if isinstance(v, pydicom.multival.MultiValue):
        return "\\".join(str(x) for x in v)
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
        return float(v)
    return v if isinstance(v, (int, float)) else str(v)


def read_datasets(blobs: list[bytes]) -> list[pydicom.Dataset]:
    return [pydicom.dcmread(io.BytesIO(b), force=True) for b in blobs]


def _pixels(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multi-frame / RGB: keep the first frame / channel mean
        arr = arr.mean(-1) if arr.shape[-1] in (3, 4) else arr[0]
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _laterality(datasets, center_x: float | None) -> tuple[str, str]:
    for name in ("Laterality", "ImageLaterality"):
        values = {str(getattr(d, name, "") or "").upper()[:1] for d in datasets} - {""}
        if len(values) == 1 and values <= {"L", "R"}:
            return values.pop(), "tag"
    text = " ".join(str(getattr(datasets[0], n, "") or "") for n in ("SeriesDescription", "BodyPartExamined")).lower()
    for pattern, side in ((_LEFT_WORDS, "L"), (_RIGHT_WORDS, "R")):
        if pattern.search(text):
            return side, "description"
    if center_x is not None and abs(center_x) > _LATERALITY_DEAD_ZONE_MM:
        return ("R" if center_x < 0 else "L"), "position"  # the scanned knee sits on its own side (+x = left)
    return "U", "none"


def _sorted_series(datasets: list, require_pixels: bool = True):
    """Dominant (orientation, matrix) group sorted by physical slice position.

    Works on header-only datasets too (require_pixels=False), so the test pipeline can decide
    the knee side of a whole study before decoding any pixel."""
    if require_pixels:
        datasets = [d for d in datasets if "PixelData" in d]
    datasets = [d for d in datasets if "Rows" in d and "Columns" in d]
    if not datasets:
        raise ValueError("no image in series")

    # keep the dominant (orientation, matrix) group: drops localizers and stray images
    def key(d):
        iop = tuple(np.round(np.asarray(d.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float), 2))
        return iop, int(d.Rows), int(d.Columns)

    groups: dict = {}
    for d in datasets:
        groups.setdefault(key(d), []).append(d)
    datasets = max(groups.values(), key=len)

    ref = datasets[0]
    has_geometry = "ImageOrientationPatient" in ref and "ImagePositionPatient" in ref
    iop = np.asarray(ref.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    n = np.cross(iop[:3], iop[3:])
    plane = _PLANE_OF_AXIS[int(np.argmax(np.abs(n)))]

    # slice order: physical position along the canonical slice axis
    if has_geometry:
        through = n if np.dot(n, _CANONICAL[plane][2]) >= 0 else -n
        pos = np.array([np.dot(np.asarray(d.ImagePositionPatient, float), through) for d in datasets])
    else:
        pos = np.array([float(getattr(d, "InstanceNumber", i) or i) for i, d in enumerate(datasets)])
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    datasets = [datasets[i] for i in order]
    keep = np.concatenate([[True], np.diff(pos) > 1e-3]) if has_geometry else np.ones(len(pos), bool)
    datasets = [d for d, k in zip(datasets, keep) if k]  # multi-echo duplicates: keep the first image per position
    return datasets, pos[keep], plane, has_geometry


def _center_x(datasets, has_geometry: bool) -> float | None:
    """Patient x (mm, +x = patient left) of the middle slice's centre."""
    if not has_geometry:
        return None
    mid = datasets[len(datasets) // 2]
    ps = [float(x) for x in mid.get("PixelSpacing", [1.0, 1.0])]
    ipp = np.asarray(mid.ImagePositionPatient, float)
    iop = np.asarray(mid.ImageOrientationPatient, float)
    center = ipp + iop[:3] * int(mid.Columns) / 2 * ps[1] + iop[3:] * int(mid.Rows) / 2 * ps[0]
    return float(center[0])


def series_side_info(datasets: list) -> dict:
    """Knee side evidence of one series, from headers only: {"side", "source", "center_x"}."""
    datasets, _, _, has_geometry = _sorted_series(datasets, require_pixels=False)
    center_x = _center_x(datasets, has_geometry)
    side, source = _laterality(datasets, center_x)
    return {"side": side, "source": source, "center_x": center_x}


def study_side(infos: list[dict]) -> str:
    """One side for the whole study, so all its series are mirrored alike.

    The median knee position over all series comes first (it agrees with the DICOM tag on
    98.8% of tagged training series and outvotes an occasional wrong tag); then the majority
    of tags/descriptions; else "U" (left as acquired)."""
    xs = [i["center_x"] for i in infos if i.get("center_x") is not None and not np.isnan(i["center_x"])]
    if xs and abs(np.median(xs)) > _LATERALITY_DEAD_ZONE_MM:
        return "R" if np.median(xs) < 0 else "L"
    votes = [i["side"] for i in infos if i.get("source") in ("tag", "description")]
    if votes:
        best = max(set(votes), key=votes.count)
        if votes.count(best) > len(votes) / 2:
            return best
    return "U"


def load_volume(blobs: list[bytes], side: str | None = None) -> Volume:
    """Decode and canonicalise one series. `side` (from study_side) overrides the series' own
    laterality evidence."""
    datasets, pos, plane, has_geometry = _sorted_series(read_datasets(blobs))
    ref = datasets[0]
    iop = np.asarray(ref.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    r, c = iop[:3], iop[3:]  # column index grows along r, row index grows along c
    tx, ty, _ = _CANONICAL[plane]

    vol = np.stack([_pixels(d) for d in datasets])
    ps = [float(x) for x in ref.get("PixelSpacing", [1.0, 1.0])]
    row_sp, col_sp = ps[0], ps[1]

    # in-plane re-orientation to the canonical display
    if abs(np.dot(c, tx)) > abs(np.dot(r, tx)):
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
        row_sp, col_sp = col_sp, row_sp
    if np.dot(r, tx) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, ty) < 0:
        vol = vol[:, ::-1, :]

    center_x = _center_x(datasets, has_geometry)
    if side is None:
        side, source = _laterality(datasets, center_x)
    else:
        source = "study"

    # mirror left knees into right knees: patient left-right is image x (coronal/axial)
    # or the slice axis (sagittal)
    if side == "L":
        if plane == "sagittal":
            vol, pos = vol[::-1], -pos[::-1]
        else:
            vol = vol[:, :, ::-1]

    meta = {t: _tag(ref, t) for t in HEADER_TAGS}
    meta.update(n_images=len(blobs), n_slices=len(vol), rows=int(ref.Rows), cols=int(ref.Columns),
                transfer_syntax=str(ref.file_meta.TransferSyntaxUID) if hasattr(ref, "file_meta") else None,
                center_x=center_x, has_geometry=has_geometry)
    return Volume(np.ascontiguousarray(vol, dtype=np.float32), (row_sp, col_sp), pos, plane, side, source, meta)


def to_uint8(vol: Volume, size: int = 256, fov_mm: float = 150.0, max_slices: int = 64,
             pct: tuple[float, float] = (1.0, 99.5)) -> tuple[np.ndarray, np.ndarray]:
    """Centre crop of fov_mm (both axes), resize to size x size, percentile window -> uint8.

    Returns (uint8 volume (S, size, size), kept slice positions)."""
    x = vol.pixels
    if len(x) > max_slices:
        idx = np.round(np.linspace(0, len(x) - 1, max_slices)).astype(int)
        x, positions = x[idx], vol.positions[idx]
    else:
        positions = vol.positions
    s, h, w = x.shape
    ch = min(h, int(round(fov_mm / vol.spacing[0])))
    cw = min(w, int(round(fov_mm / vol.spacing[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    x = x[:, y0 : y0 + ch, x0 : x0 + cw]
    lo, hi = np.percentile(x[:, ::4, ::4], pct)
    # pad to a square physical field so the aspect ratio survives the resize
    ph = int(round(max(ch * vol.spacing[0], cw * vol.spacing[1]) / vol.spacing[0]))
    pw = int(round(max(ch * vol.spacing[0], cw * vol.spacing[1]) / vol.spacing[1]))
    if (ph, pw) != (ch, cw):
        pad = np.zeros((s, ph, pw), np.float32)
        pad[:, (ph - ch) // 2 : (ph - ch) // 2 + ch, (pw - cw) // 2 : (pw - cw) // 2 + cw] = x
        x = pad
    out = np.empty((s, size, size), np.uint8)
    scale = 255.0 / max(hi - lo, 1e-6)
    for i in range(s):
        img = cv2.resize(x[i], (size, size), interpolation=cv2.INTER_AREA)
        out[i] = np.clip((img - lo) * scale, 0, 255).astype(np.uint8)
    return out, positions
