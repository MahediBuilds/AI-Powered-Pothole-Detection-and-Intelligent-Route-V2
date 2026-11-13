"""Download selected RDD2022 countries without fetching the whole 13 GB archive.

The official release is one outer zip whose members are STORED (uncompressed)
nested per-country zips, so each nested zip is a contiguous byte range of the
outer file. This module reads the outer central directory with HTTP Range
requests, downloads only the requested byte ranges, verifies size and CRC-32
against the central directory, and extracts the nested zips under ``--dest``.

    python -m roadsense.data.fetch_rdd2022 --countries India Japan [--dest data/raw/rdd2022]
        [--keep-zips] [--force] [--list]

Integrity: sizes and CRCs come from the server's own central directory, so they
detect transfer errors, not a tampered server. The outer MD5 below is recorded
for reference; it cannot be checked without downloading the full archive.

Standard library only. All network access goes through a source object with
``size``, ``read_range(start, end)`` and ``stream_range(start, end)`` (half-open
byte ranges), so tests can substitute a local-file-backed source.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from roadsense.config import project_root

# Rethinking Road Damage Detection 2022 (CRDDC2022), figshare DOI 10.6084/m9.figshare.21431547.v1
DOI = "10.6084/m9.figshare.21431547.v1"
OUTER_URL = "https://ndownloader.figshare.com/files/38030910"
OUTER_NAME = "RDD2022_released_through_CRDDC2022.zip"
OUTER_SIZE = 13_264_172_619
OUTER_MD5 = "b62bd51d2ffcfaa76c60f234f0cc2bb3"
NESTED_PREFIX = "RDD2022/"
DEFAULT_DEST = "data/raw/rdd2022"
SPACE_FACTOR = 2.2  # free space needed relative to bytes downloaded (download + extraction)
CHUNK = 1 << 20

_EOCD_SIG = b"PK\x05\x06"
_LOC64_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
_CD_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_EOCD = struct.Struct("<IHHHHIIH")  # 22 bytes
_LOC64 = struct.Struct("<IIQI")  # 20 bytes
_EOCD64 = struct.Struct("<IQHHIIQQQQ")  # 56 bytes
_CD = struct.Struct("<IHHHHHHIIIHHHHHII")  # 46 bytes
_LOCAL = struct.Struct("<IHHHHHIIIHH")  # 30 bytes
_U32 = 0xFFFFFFFF
_U16 = 0xFFFF


class FetchError(Exception):
    """Acquisition failed; the message says why."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    header_offset: int


# --------------------------------------------------------------------------- sources


class HttpSource:
    """Ranged reads against a URL that supports HTTP Range requests."""

    def __init__(self, url: str = OUTER_URL, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout
        self._size: int | None = None

    @property
    def size(self) -> int:
        if self._size is None:
            with self._open(0, 1) as resp:
                self._size = self._content_range(resp, 0, 1)
                resp.read()
        return self._size

    def _open(self, start: int, end: int):
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "roadsense-fetch"},
        )
        resp = urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 (fixed https URL)
        if resp.status != 206:
            resp.close()
            raise FetchError(
                f"server answered HTTP {resp.status} to a Range request (expected 206); "
                "refusing to read it as a byte range"
            )
        return resp

    @staticmethod
    def _content_range(resp, start: int, end: int) -> int:
        header = resp.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header.strip())
        if not match or int(match[1]) != start or int(match[2]) != end - 1:
            raise FetchError(f"unexpected Content-Range {header!r} for bytes {start}-{end - 1}")
        return int(match[3])

    def read_range(self, start: int, end: int) -> bytes:
        return b"".join(self.stream_range(start, end))

    def stream_range(self, start: int, end: int) -> Iterator[bytes]:
        received = 0
        with self._open(start, end) as resp:
            self._content_range(resp, start, end)
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                received += len(chunk)
                yield chunk
        if received != end - start:
            raise FetchError(f"truncated transfer: got {received} of {end - start} bytes")


# --------------------------------------------------------------------------- zip parsing


def _parse_zip64_extra(
    extra: bytes, usize: int, csize: int, offset: int
) -> tuple[int, int, int]:
    """Replace saturated 32-bit values using the zip64 extra field (id 0x0001)."""
    i = 0
    while i + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, i)
        data = extra[i + 4 : i + 4 + size]
        if header_id == 1:
            pos = 0
            values = []
            for saturated in (usize == _U32, csize == _U32, offset == _U32):
                if saturated:
                    if pos + 8 > len(data):
                        raise FetchError("zip64 extra field is too short")
                    values.append(struct.unpack_from("<Q", data, pos)[0])
                    pos += 8
                else:
                    values.append(None)
            return (
                usize if values[0] is None else values[0],
                csize if values[1] is None else values[1],
                offset if values[2] is None else values[2],
            )
        i += 4 + size
    return usize, csize, offset


def _find_eocd(tail: bytes) -> int:
    pos = tail.rfind(_EOCD_SIG)
    while pos != -1:
        if pos + _EOCD.size <= len(tail):
            comment_len = struct.unpack_from("<H", tail, pos + 20)[0]
            if pos + _EOCD.size + comment_len == len(tail):
                return pos
        pos = tail.rfind(_EOCD_SIG, 0, pos)
    raise FetchError("end of central directory not found; not a zip file?")


def read_central_directory(source) -> list[ZipEntry]:
    """List the entries of the zip behind ``source`` using ranged reads only."""
    size = source.size
    tail_len = min(size, _EOCD.size + _U16 + _LOC64.size)
    tail = source.read_range(size - tail_len, size)
    pos = _find_eocd(tail)
    _, disk, cd_disk, _, n_total, cd_size, cd_offset, _ = _EOCD.unpack_from(tail, pos)
    if disk != 0 or cd_disk != 0:
        raise FetchError("multi-disk zip archives are not supported")

    saturated = n_total == _U16 or cd_size == _U32 or cd_offset == _U32
    if saturated:
        loc_pos = pos - _LOC64.size
        if loc_pos < 0 or tail[loc_pos : loc_pos + 4] != _LOC64_SIG:
            raise FetchError("zip64 values expected but no zip64 locator found")
        _, _, eocd64_offset, _ = _LOC64.unpack_from(tail, loc_pos)
        record = source.read_range(eocd64_offset, eocd64_offset + _EOCD64.size)
        if len(record) != _EOCD64.size or record[:4] != _EOCD64_SIG:
            raise FetchError("zip64 end of central directory record not found")
        (_, _, _, _, disk, cd_disk, _, n_total, cd_size, cd_offset) = _EOCD64.unpack(record)
        if disk != 0 or cd_disk != 0:
            raise FetchError("multi-disk zip archives are not supported")

    if cd_offset + cd_size > size:
        raise FetchError("central directory lies outside the file")
    cd = source.read_range(cd_offset, cd_offset + cd_size)
    entries: list[ZipEntry] = []
    pos = 0
    for _ in range(n_total):
        if pos + _CD.size > len(cd) or cd[pos : pos + 4] != _CD_SIG:
            raise FetchError("corrupt central directory")
        (_, _, _, flags, method, _, _, crc, csize, usize, name_len, extra_len, comment_len,
         _, _, _, offset) = _CD.unpack_from(cd, pos)  # fmt: skip
        pos += _CD.size
        raw_name = cd[pos : pos + name_len]
        extra = cd[pos + name_len : pos + name_len + extra_len]
        pos += name_len + extra_len + comment_len
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        usize, csize, offset = _parse_zip64_extra(extra, usize, csize, offset)
        if _U32 in (usize, csize, offset):
            raise FetchError(f"{name}: zip64 values missing from the extra field")
        entries.append(ZipEntry(name, method, csize, usize, crc, offset))
    return entries


def nested_zips(entries: list[ZipEntry]) -> dict[str, ZipEntry]:
    """Map country name to its nested zip entry (``RDD2022/<Country>.zip``)."""
    found = {}
    for entry in entries:
        name = entry.name
        if name.startswith(NESTED_PREFIX) and name.endswith(".zip") and "/" not in name[len(NESTED_PREFIX):]:
            found[name[len(NESTED_PREFIX) : -len(".zip")]] = entry
    return found


def select_entries(entries: list[ZipEntry], countries: list[str]) -> list[ZipEntry]:
    available = nested_zips(entries)
    unknown = [c for c in countries if c not in available]
    if unknown:
        raise FetchError(
            f"unknown country name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return [available[c] for c in dict.fromkeys(countries)]


def data_range(source, entry: ZipEntry) -> tuple[int, int]:
    """Byte range of a STORED entry's data within the outer file."""
    if entry.method != 0 or entry.compressed_size != entry.uncompressed_size:
        raise FetchError(
            f"{entry.name} is not STORED (method {entry.method}, "
            f"{entry.compressed_size} compressed vs {entry.uncompressed_size} bytes); "
            "it is not a contiguous byte range, so it cannot be fetched on its own"
        )
    header = source.read_range(entry.header_offset, entry.header_offset + _LOCAL.size)
    if len(header) != _LOCAL.size or header[:4] != _LOCAL_SIG:
        raise FetchError(f"{entry.name}: local file header not found at its recorded offset")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = entry.header_offset + _LOCAL.size + name_len + extra_len
    end = start + entry.compressed_size
    if end > source.size:
        raise FetchError(f"{entry.name}: data range ends beyond the end of the archive")
    return start, end


# --------------------------------------------------------------------------- download


def _size_and_crc(path: Path) -> tuple[int, int]:
    crc = 0
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return size, crc


def download_entry(
    source, entry: ZipEntry, part_path: Path, log: Callable[[str], None] = print
) -> Path:
    """Stream the entry to ``part_path`` (resuming), verify, and rename to ``.zip``."""
    start, end = data_range(source, entry)
    expected = entry.compressed_size
    part_path.parent.mkdir(parents=True, exist_ok=True)
    have = part_path.stat().st_size if part_path.exists() else 0
    if have > expected:
        raise FetchError(
            f"{part_path} is larger than expected ({have} > {expected} bytes); delete it and retry"
        )
    if have:
        log(f"  resuming at {have} of {expected} bytes")
    next_report = have + (100 << 20)
    if have < expected:
        with part_path.open("ab") as out:
            done = have
            for chunk in source.stream_range(start + have, end):
                out.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    log(f"  {done / expected:6.1%} ({done} of {expected} bytes)")
                    next_report = done + (100 << 20)
    size, crc = _size_and_crc(part_path)
    if size != expected or crc != entry.crc32:
        raise FetchError(
            f"{part_path.name}: verification failed: size {size} (expected {expected}), "
            f"CRC-32 {crc:#010x} (expected {entry.crc32:#010x}). "
            f"The partial file was kept; delete {part_path} to start over."
        )
    final = part_path.with_name(part_path.name.removesuffix(".part"))
    part_path.replace(final)
    return final


def _check_member_name(name: str, country: str) -> str:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm) or ".." in norm.split("/"):
        raise FetchError(f"unsafe path in archive: {name!r}")
    if norm.split("/")[0] != country:
        raise FetchError(f"archive entry {name!r} is not under {country}/")
    return norm


def safe_extract(zip_path: Path, dest: Path, country: str) -> Path:
    """Extract a nested zip to ``dest/<country>`` with a path-traversal guard.

    Everything is extracted to a staging directory first and moved into place
    only after the whole archive extracted, so ``dest/<country>`` never holds a
    partial extraction.
    """
    staging = dest / f"_extract_{country}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staging_root = staging.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            names = [_check_member_name(i.filename, country) for i in infos]
            targets = []
            for norm in names:
                target = (staging / norm).resolve()
                if not target.is_relative_to(staging_root):
                    raise FetchError(f"archive entry escapes the destination: {norm!r}")
                targets.append(target)
            for info, target in zip(infos, targets):
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        final = dest / country
        if final.exists():
            shutil.rmtree(final)
        (staging / country).replace(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def fetch(
    source,
    countries: list[str],
    dest: Path,
    *,
    keep_zips: bool = False,
    force: bool = False,
    list_only: bool = False,
    expected_size: int = OUTER_SIZE,
    log: Callable[[str], None] = print,
) -> None:
    if source.size != expected_size:
        raise FetchError(
            f"outer archive size is {source.size} but {expected_size} was expected; "
            "the record may have changed, refusing to trust ranged reads"
        )
    entries = read_central_directory(source)
    if list_only:
        for entry in entries:
            log(f"{entry.name}  {entry.uncompressed_size:>16,} bytes  method {entry.method}")
        return

    selected = select_entries(entries, countries)
    todo = []
    for entry in selected:
        country = entry.name[len(NESTED_PREFIX) : -len(".zip")]
        if (dest / country / "train" / "images").exists() and not force:
            log(f"{country}: already present, skipping (use --force to re-fetch)")
        else:
            todo.append((country, entry))
    if not todo:
        return
    total = sum(entry.compressed_size for _, entry in todo)
    free = _free_bytes(dest)
    log(f"to fetch: {total:,} bytes; free space on destination: {free:,} bytes")
    if free < SPACE_FACTOR * total:
        raise FetchError(
            f"not enough free space: need about {SPACE_FACTOR * total:,.0f} bytes "
            f"({SPACE_FACTOR} x download), have {free:,}"
        )
    for country, entry in todo:
        log(f"{country}: downloading {entry.compressed_size:,} bytes")
        part = dest / "_downloads" / f"{country}.zip.part"
        zip_path = download_entry(source, entry, part, log)
        log(f"{country}: extracting")
        safe_extract(zip_path, dest, country)
        if not keep_zips:
            zip_path.unlink()
        log(f"{country}: done")


def main(argv: list[str] | None = None, source=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch selected RDD2022 countries.")
    parser.add_argument("--countries", nargs="+", default=[], help="e.g. India Japan Czech")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="relative paths resolve against the project root")
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-fetch countries already present")
    parser.add_argument("--list", action="store_true", dest="list_only", help="list archive entries and exit")
    ns = parser.parse_args(argv)
    if not ns.list_only and not ns.countries:
        parser.error("--countries is required unless --list is given")
    dest = Path(ns.dest)
    if not dest.is_absolute():
        dest = project_root() / dest
    try:
        fetch(
            source or HttpSource(),
            ns.countries,
            dest,
            keep_zips=ns.keep_zips,
            force=ns.force,
            list_only=ns.list_only,
        )
    except (FetchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
