import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from roadsense.data import fetch_rdd2022 as fetch
from roadsense.data.fetch_rdd2022 import FetchError


class FileSource:
    """Local-file-backed stand-in for HttpSource that records the ranges requested."""

    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.streams = []

    def read_range(self, start, end):
        with self.path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start)

    def stream_range(self, start, end, chunk=1000):
        self.streams.append((start, end))
        pos = start
        while pos < end:
            step = min(chunk, end - pos)
            yield self.read_range(pos, pos + step)
            pos += step


def nested_zip_bytes(country, extra_members=None):
    """A deflated nested zip with the real RDD2022 folder layout and a few tiny files."""
    members = {
        f"{country}/train/images/{country}_000000.jpg": bytes(range(256)) * 12,
        f"{country}/train/annotations/xmls/{country}_000000.xml": b"<annotation></annotation>" * 40,
        f"{country}/test/images/{country}_000001.jpg": bytes(range(255, -1, -1)) * 5,
        **(extra_members or {}),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def make_outer(tmp_path, nested, name="outer.zip", method=zipfile.ZIP_STORED):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for country, data in nested.items():
            zf.writestr(zipfile.ZipInfo(f"RDD2022/{country}.zip"), data, compress_type=method)
    return path


@pytest.fixture
def nested():
    return {c: nested_zip_bytes(c) for c in ("India", "Japan", "Czech")}


@pytest.fixture
def outer(tmp_path, nested):
    return make_outer(tmp_path, nested)


def run_fetch(source, countries, dest, **kw):
    messages = []
    kw.setdefault("expected_size", source.size)
    fetch.fetch(source, countries, dest, log=messages.append, **kw)
    return messages


def test_central_directory_listing(outer, nested):
    entries = fetch.read_central_directory(FileSource(outer))
    assert [e.name for e in entries] == [f"RDD2022/{c}.zip" for c in nested]
    for entry in entries:
        data = nested[entry.name.split("/")[1][:-4]]
        assert entry.method == 0
        assert entry.compressed_size == entry.uncompressed_size == len(data)
        assert entry.crc32 == zlib.crc32(data)
    assert set(fetch.nested_zips(entries)) == {"India", "Japan", "Czech"}


def test_list_only_prints_entries_and_downloads_nothing(outer, tmp_path):
    source = FileSource(outer)
    messages = run_fetch(source, [], tmp_path / "dest", list_only=True)
    assert len(messages) == 3 and "RDD2022/India.zip" in messages[0]
    assert source.streams == []
    assert not (tmp_path / "dest").exists()


def test_fetch_extracts_exact_byte_range(outer, nested, tmp_path):
    dest = tmp_path / "dest"
    source = FileSource(outer)
    run_fetch(source, ["India"], dest, keep_zips=True)
    assert (dest / "_downloads" / "India.zip").read_bytes() == nested["India"]
    assert not list((dest / "_downloads").glob("*.part"))
    with zipfile.ZipFile(io.BytesIO(nested["India"])) as zf:
        for info in zf.infolist():
            assert (dest / info.filename).read_bytes() == zf.read(info)
    assert (dest / "India" / "train" / "images" / "India_000000.jpg").is_file()
    assert not (dest / "Japan").exists()  # only the requested country
    assert len(source.streams) == 1
    start, end = source.streams[0]
    assert end - start == len(nested["India"])
    assert outer.read_bytes()[start:end] == nested["India"]
    assert not any(p.name.startswith("_extract_") for p in dest.iterdir())


def test_zip_removed_unless_keep_zips(outer, tmp_path):
    dest = tmp_path / "dest"
    run_fetch(FileSource(outer), ["India", "Czech"], dest)
    assert not list((dest / "_downloads").glob("*.zip"))
    assert (dest / "Czech" / "train" / "images").is_dir()


def test_skip_existing_country_unless_force(outer, tmp_path):
    dest = tmp_path / "dest"
    run_fetch(FileSource(outer), ["India"], dest)
    marker = dest / "India" / "train" / "images" / "marker.txt"
    marker.write_text("x", encoding="utf-8")
    source = FileSource(outer)
    messages = run_fetch(source, ["India"], dest)
    assert any("skipping" in m for m in messages) and source.streams == [] and marker.exists()
    run_fetch(source, ["India"], dest, force=True)
    assert source.streams and not marker.exists()  # replaced by a fresh extraction


def test_crc_mismatch_detected_and_partial_kept(outer, nested, tmp_path):
    source = FileSource(outer)
    entry = fetch.nested_zips(fetch.read_central_directory(source))["India"]
    start, _ = fetch.data_range(source, entry)
    corrupt = bytearray(outer.read_bytes())
    corrupt[start + 100] ^= 0xFF
    bad = tmp_path / "bad.zip"
    bad.write_bytes(bytes(corrupt))
    dest = tmp_path / "dest"
    with pytest.raises(FetchError) as exc:
        run_fetch(FileSource(bad), ["India"], dest)
    message = str(exc.value)
    assert "CRC-32" in message and f"{entry.crc32:#010x}" in message
    assert f"{zlib.crc32(bytes(corrupt)[start:start + len(nested['India'])]):#010x}" in message
    assert (dest / "_downloads" / "India.zip.part").exists()  # nothing deleted silently
    assert not (dest / "India").exists()


def test_resume_from_partial_file(outer, nested, tmp_path):
    dest = tmp_path / "dest"
    source = FileSource(outer)
    entry = fetch.nested_zips(fetch.read_central_directory(source))["India"]
    start, end = fetch.data_range(source, entry)
    part = dest / "_downloads" / "India.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(nested["India"][:300])
    messages = run_fetch(source, ["India"], dest, keep_zips=True)
    assert source.streams == [(start + 300, end)]  # only the missing tail was requested
    assert (dest / "_downloads" / "India.zip").read_bytes() == nested["India"]
    assert any("resuming at 300" in m for m in messages)


def test_resume_with_complete_partial_skips_network(outer, nested, tmp_path):
    dest = tmp_path / "dest"
    part = dest / "_downloads" / "India.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(nested["India"])
    source = FileSource(outer)
    run_fetch(source, ["India"], dest)
    assert source.streams == []
    assert (dest / "India" / "train").is_dir()


def test_oversized_partial_is_an_error(outer, nested, tmp_path):
    dest = tmp_path / "dest"
    part = dest / "_downloads" / "India.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(nested["India"] + b"extra")
    with pytest.raises(FetchError, match="larger than expected"):
        run_fetch(FileSource(outer), ["India"], dest)
    assert part.exists()


def test_deflated_nested_entry_refused(tmp_path, nested):
    outer = make_outer(tmp_path, {"India": nested["India"]}, method=zipfile.ZIP_DEFLATED)
    # incompressible-enough data is still recorded as deflated, which is what matters
    with pytest.raises(FetchError, match="not STORED"):
        run_fetch(FileSource(outer), ["India"], tmp_path / "dest")
    assert not (tmp_path / "dest" / "_downloads").exists()


@pytest.mark.parametrize(
    "bad_name",
    ["../evil.txt", "/abs.txt", "C:/evil.txt", "India/../../evil.txt", "India/..\\..\\evil.txt", "Other/x.txt"],
)
def test_path_traversal_rejected(tmp_path, bad_name):
    nested_bytes = nested_zip_bytes("India", {bad_name: b"boom"})
    outer = make_outer(tmp_path, {"India": nested_bytes})
    dest = tmp_path / "dest"
    with pytest.raises(FetchError, match="unsafe path|not under India"):
        run_fetch(FileSource(outer), ["India"], dest)
    assert not (dest / "India").exists()
    assert not (tmp_path / "evil.txt").exists() and not (dest / "evil.txt").exists()
    assert not any(p.name.startswith("_extract_") for p in dest.iterdir())


def test_unknown_country_gives_clear_error(outer, tmp_path):
    source = FileSource(outer)
    with pytest.raises(FetchError, match=r"unknown country name\(s\): Atlantis.*Available: Czech, India, Japan"):
        run_fetch(source, ["India", "Atlantis"], tmp_path / "dest")
    assert source.streams == []  # validated before any download


def test_outer_size_mismatch_refused(outer, tmp_path):
    source = FileSource(outer)
    with pytest.raises(FetchError, match="outer archive size"):
        fetch.fetch(source, ["India"], tmp_path / "dest", expected_size=source.size + 1, log=lambda m: None)


def test_insufficient_free_space_aborts_before_download(outer, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_free_bytes", lambda path: 1000)
    source = FileSource(outer)
    with pytest.raises(FetchError, match="not enough free space"):
        run_fetch(source, ["India"], tmp_path / "dest")
    assert source.streams == []


def test_free_space_rule_uses_factor(outer, nested, tmp_path, monkeypatch):
    need = int(fetch.SPACE_FACTOR * len(nested["India"]))
    monkeypatch.setattr(fetch, "_free_bytes", lambda path: need - 1)
    with pytest.raises(FetchError, match="not enough free space"):
        run_fetch(FileSource(outer), ["India"], tmp_path / "dest")
    monkeypatch.setattr(fetch, "_free_bytes", lambda path: need + 1)
    run_fetch(FileSource(outer), ["India"], tmp_path / "dest")


def test_cli_reports_errors_with_exit_code(outer, tmp_path, capsys):
    # The synthetic archive is not the real 13 GB file, so the CLI's size check must refuse it.
    code = fetch.main(["--countries", "India", "--dest", str(tmp_path / "d")], source=FileSource(outer))
    assert code == 2
    assert "outer archive size" in capsys.readouterr().err


def test_cli_requires_countries_unless_list(tmp_path):
    with pytest.raises(SystemExit):
        fetch.main(["--dest", str(tmp_path)], source=object())


# ----------------------------------------------------------------------------- zip64


def build_zip64_outer(name: str, data: bytes) -> bytes:
    """One STORED entry described with zip64 structures only (saturated 32-bit fields)."""
    crc = zlib.crc32(data)
    nb = name.encode()
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 45, 0, 0, 0, 0x21, crc, len(data), len(data), len(nb), 0)
    local += nb + data
    extra = struct.pack("<HHQQQ", 1, 24, len(data), len(data), 0)  # usize, csize, local offset
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII", 0x02014B50, 45, 45, 0, 0, 0, 0x21, crc, 0xFFFFFFFF, 0xFFFFFFFF,
        len(nb), len(extra), 0, 0, 0, 0, 0xFFFFFFFF,
    ) + nb + extra  # fmt: skip
    cd_offset = len(local)
    eocd64_offset = cd_offset + len(central)
    eocd64 = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, 1, 1, len(central), cd_offset)
    locator = struct.pack("<IIQI", 0x07064B50, 0, eocd64_offset, 1)
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return local + central + eocd64 + locator + eocd


def test_zip64_central_directory_and_fetch(tmp_path):
    data = nested_zip_bytes("India")
    path = tmp_path / "zip64.zip"
    path.write_bytes(build_zip64_outer("RDD2022/India.zip", data))
    with zipfile.ZipFile(path) as zf:  # the stdlib must agree the builder made a valid zip64 archive
        assert zf.read("RDD2022/India.zip") == data

    source = FileSource(path)
    (entry,) = fetch.read_central_directory(source)
    assert (entry.name, entry.method, entry.header_offset) == ("RDD2022/India.zip", 0, 0)
    assert entry.compressed_size == entry.uncompressed_size == len(data)
    assert entry.crc32 == zlib.crc32(data)

    dest = tmp_path / "dest"
    run_fetch(source, ["India"], dest, keep_zips=True)
    assert (dest / "_downloads" / "India.zip").read_bytes() == data
    assert (dest / "India" / "train" / "images" / "India_000000.jpg").is_file()


def test_zip64_saturated_without_locator_is_an_error(tmp_path):
    raw = build_zip64_outer("RDD2022/India.zip", b"abc")
    broken = raw[: -22 - 20] + raw[-22:]  # drop the zip64 locator
    path = tmp_path / "broken.zip"
    path.write_bytes(broken)
    with pytest.raises(FetchError, match="zip64 locator"):
        fetch.read_central_directory(FileSource(path))


def test_parse_zip64_extra_only_replaces_saturated_fields():
    extra = struct.pack("<HHQ", 1, 8, 5_000_000_000)  # only the offset is saturated
    assert fetch._parse_zip64_extra(extra, 10, 20, 0xFFFFFFFF) == (10, 20, 5_000_000_000)
    extra = struct.pack("<HHQQ", 1, 16, 7, 9)  # usize and csize saturated
    assert fetch._parse_zip64_extra(extra, 0xFFFFFFFF, 0xFFFFFFFF, 3) == (7, 9, 3)
    assert fetch._parse_zip64_extra(b"", 1, 2, 3) == (1, 2, 3)
    with pytest.raises(FetchError, match="too short"):
        fetch._parse_zip64_extra(struct.pack("<HHQ", 1, 8, 1), 0xFFFFFFFF, 0xFFFFFFFF, 0)
