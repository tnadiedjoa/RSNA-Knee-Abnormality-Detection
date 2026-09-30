"""Random access to the competition archive over HTTP range requests.

The full archive is 265 GB zipped (570 GB of DICOMs) - far more than our storage allows -
but Kaggle hands out a signed Google Cloud Storage URL that serves byte ranges. So we read
the zip's central directory once (233 MB, 819,640 members), then fetch only the members
we need, series by series, without ever storing the archive.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COMPETITION = "rsna-knee-abnormality-detection"
URL_CACHE = Path(os.environ.get("KNEE_URL_CACHE", Path.home() / ".cache" / "rsna_knee" / "signed_url.json"))

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")  # 30 bytes
_CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")  # 46 bytes
_LOCAL_SIG, _CENTRAL_SIG = 0x04034B50, 0x02014B50


def _fresh_signed_url() -> str:
    import kaggle
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

    api = kaggle.KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kc:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = COMPETITION
        response = kc.competitions.competition_api_client.download_data_files(request)
        url = response.url
        response.close()
    return url


def signed_url(min_validity_s: int = 3600) -> str:
    """Signed archive URL, reused from the cache while it stays valid for min_validity_s."""
    if URL_CACHE.exists():
        url = json.loads(URL_CACHE.read_text())["url"]
        expires = int(re.search(r"Expires=(\d+)", url).group(1))
        if expires - time.time() > min_validity_s:
            return url
    url = _fresh_signed_url()
    URL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    URL_CACHE.write_text(json.dumps({"url": url}))
    URL_CACHE.chmod(0o600)
    return url


class RemoteArchive:
    """Byte-range reader for the competition zip, with retries and URL refresh."""

    def __init__(self, retries: int = 6):
        self.url = signed_url()
        self.retries = retries
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self.session.mount("https://", adapter)

    def read(self, start: int, length: int) -> bytes:
        end = start + length - 1
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (400, 401, 403):  # signature expired
                    self.url = signed_url(min_validity_s=7200)
                    continue
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError(f"short read: {len(r.content)} of {length} bytes")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def size(self) -> int:
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        return int(r.headers["Content-Range"].split("/")[1])


def _central_directory_location(archive: RemoteArchive) -> tuple[int, int, int]:
    """(entries, cd_size, cd_offset) from the zip64 end-of-central-directory record."""
    size = archive.size()
    tail = archive.read(size - 65536, 65536)
    loc = tail.rfind(b"PK\x06\x07")
    (z64_offset,) = struct.unpack_from("<Q", tail, loc + 8)
    rec = archive.read(z64_offset, 56)
    entries, cd_size, cd_offset = struct.unpack_from("<QQQ", rec, 32)
    return entries, cd_size, cd_offset


def read_index(archive: RemoteArchive | None = None, chunk: int = 16 * 2**20) -> pd.DataFrame:
    """One row per archive member: name, method, csize, usize, lho (local header offset)."""
    archive = archive or RemoteArchive()
    entries, cd_size, cd_offset = _central_directory_location(archive)
    starts = list(range(cd_offset, cd_offset + cd_size, chunk))
    with ThreadPoolExecutor(8) as pool:
        parts = pool.map(lambda s: archive.read(s, min(chunk, cd_offset + cd_size - s)), starts)
        cd = b"".join(parts)

    names, methods, csizes, usizes, offsets = [], [], [], [], []
    pos = 0
    for _ in range(entries):
        (sig, _, _, _, method, _, _, _, csize, usize, name_len, extra_len, comment_len,
         _, _, _, lho) = _CENTRAL_HEADER.unpack_from(cd, pos)
        assert sig == _CENTRAL_SIG, f"bad central header at {pos}"
        name = cd[pos + 46 : pos + 46 + name_len].decode()
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        if 0xFFFFFFFF in (csize, usize, lho):  # zip64 extra field (id 0x0001)
            e = 0
            while e < len(extra):
                eid, esz = struct.unpack_from("<HH", extra, e)
                if eid == 0x0001:
                    vals, v = extra[e + 4 : e + 4 + esz], 0
                    if usize == 0xFFFFFFFF:
                        (usize,) = struct.unpack_from("<Q", vals, v); v += 8
                    if csize == 0xFFFFFFFF:
                        (csize,) = struct.unpack_from("<Q", vals, v); v += 8
                    if lho == 0xFFFFFFFF:
                        (lho,) = struct.unpack_from("<Q", vals, v); v += 8
                e += 4 + esz
        names.append(name); methods.append(method); csizes.append(csize)
        usizes.append(usize); offsets.append(lho)
        pos += 46 + name_len + extra_len + comment_len

    df = pd.DataFrame({"name": names, "method": np.array(methods, np.int16), "csize": np.array(csizes, np.int64),
                       "usize": np.array(usizes, np.int64), "lho": np.array(offsets, np.int64)})
    parts = df["name"].str.split("/", expand=True)
    df["split"] = parts[0]
    is_dcm = df["name"].str.endswith(".dcm")
    df["study"] = parts[1].where(is_dcm)
    df["series"] = parts[2].where(is_dcm)
    return df


def extract_members(buf: bytes, buf_start: int, members: pd.DataFrame) -> list[bytes]:
    """Decompress the given members from a buffer that covers them (buf_start = its archive offset)."""
    out = []
    for lho, csize, method in zip(members["lho"], members["csize"], members["method"]):
        p = lho - buf_start
        sig, *_, name_len, extra_len = _LOCAL_HEADER.unpack_from(buf, p)
        assert sig == _LOCAL_SIG, f"bad local header at {lho}"
        data = buf[p + 30 + name_len + extra_len : p + 30 + name_len + extra_len + csize]
        if method == 0:
            out.append(bytes(data))
        elif method == 8:
            out.append(zlib.decompress(data, -15))
        else:
            raise ValueError(f"unsupported zip method {method}")
    return out


def fetch_members(archive: RemoteArchive, members: pd.DataFrame, max_gap: int = 1 << 20) -> list[bytes]:
    """Fetch members with as few range requests as possible (merging runs closer than max_gap).

    Returns the decompressed bytes in the order of `members`."""
    order = np.argsort(members["lho"].to_numpy())
    sorted_m = members.iloc[order]
    lho = sorted_m["lho"].to_numpy()
    # member end upper bound: local header (30) + name (<=256) + extra (<=64) + data
    end = lho + 30 + 350 + sorted_m["csize"].to_numpy()
    results: list[bytes] = [b""] * len(members)
    i = 0
    while i < len(sorted_m):
        j = i
        while j + 1 < len(sorted_m) and lho[j + 1] - end[j] < max_gap:
            j += 1
        start, stop = lho[i], end[i : j + 1].max()
        buf = archive.read(int(start), int(stop - start))
        for k, data in zip(range(i, j + 1), extract_members(buf, int(start), sorted_m.iloc[i : j + 1])):
            results[order[k]] = data
        i = j + 1
    return results
