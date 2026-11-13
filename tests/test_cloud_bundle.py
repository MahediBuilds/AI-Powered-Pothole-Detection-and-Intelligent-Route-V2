"""Tests for scripts/cloud_bundle.py: train+val only, chunking, pinned digest, tamper detection."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cloud_bundle", Path(__file__).resolve().parents[1] / "scripts" / "cloud_bundle.py"
)
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

MANIFEST_HEADER = "source,original_path,split,sha256,group_id,box_count\n"


def make_dataset(tmp_path: Path, entry_style: str = "relative") -> Path:
    root = tmp_path / "ds"
    lists = {"train": [], "val_pooled": [], "rdd2022_val": [], "hrp4k_val": [], "rdd2022_test": [], "hrp4k_test": []}
    manifest = [MANIFEST_HEADER]
    for source in ("rdd2022", "hrp4k"):
        for split in ("train", "val", "test"):
            for i in range(3):
                name = f"{source}_{split}_{i}"
                img = root / "images" / source / split / f"{name}.jpg"
                lab = root / "labels" / source / split / f"{name}.txt"
                img.parent.mkdir(parents=True, exist_ok=True)
                lab.parent.mkdir(parents=True, exist_ok=True)
                img.write_bytes(f"image {name}".encode() * 50)
                lab.write_text("0 0.5 0.5 0.1 0.1\n")
                rel = f"images/{source}/{split}/{name}.jpg"
                if entry_style == "absolute":
                    entry = img.resolve().as_posix()
                elif entry_style == "backslash":
                    entry = rel.replace("/", chr(92))
                elif entry_style == "dotslash":
                    entry = "./" + rel
                else:
                    entry = rel
                if split == "train":
                    lists["train"].append(entry)
                elif split == "val":
                    lists["val_pooled"].append(entry)
                    lists[f"{source}_val"].append(entry)
                else:
                    lists[f"{source}_test"].append(entry)
                manifest.append(f"{source},/orig/{name},{split},abc,g{i},1\n")
    (root / "lists").mkdir()
    for name, entries in lists.items():
        (root / "lists" / f"{name}.txt").write_text("\n".join(entries) + "\n")
    (root / "dataset.yaml").write_text("train: lists/train.txt\nval: lists/val_pooled.txt\nnames:\n  0: pothole\n")
    (root / "manifest.csv").write_text("".join(manifest))
    return root


def build(tmp_path, root, chunk_mb=1):
    out = tmp_path / "out"
    info = cb.build(root, out, chunk_mb, None)
    return out, info


@pytest.mark.parametrize("style", ["relative", "absolute", "backslash", "dotslash"])
def test_round_trip_train_val_only(tmp_path, style):
    root = make_dataset(tmp_path, style)
    out, info = build(tmp_path, root)
    assert info["n_train"] == 6 and info["n_val"] == 6 and info["n_images"] == 12
    # transfer (cp transport) then unpack and verify on the "instance"
    remote = tmp_path / "remote_upload"
    args = cb.argparse.Namespace(out=out, remote_dir=str(remote), transport="cp", streams=4, retries=1, stagger=0)
    assert cb.upload(args) == 0
    dest = tmp_path / "instance"
    dest.mkdir()
    digest = cb.verify(remote, dest, info["digest_sha256"])
    ds = dest / "data/processed/unified_v1"
    files = sorted(p.relative_to(ds).as_posix() for p in ds.rglob("*") if p.is_file())
    assert not [f for f in files if "test" in f.lower().replace("lists/", "")], files  # no test paths
    assert not (ds / "lists" / "rdd2022_test.txt").exists() and not (ds / "lists" / "hrp4k_test.txt").exists()
    assert (ds / "lists" / "train.txt").read_text().splitlines()[0] == "images/rdd2022/train/rdd2022_train_0.jpg"
    manifest = (ds / "manifest.csv").read_text()
    assert ",test," not in manifest and manifest.count("\n") == 1 + 12
    assert digest["counts"]["rdd2022/train"] == {"images": 3, "labels": 3}
    assert "rdd2022/test" not in digest["counts"] and "hrp4k/test" not in digest["counts"]


def test_chunking_splits_the_tar(tmp_path):
    root = make_dataset(tmp_path)
    for p in (root / "images" / "hrp4k" / "train").glob("*.jpg"):
        p.write_bytes(b"x" * 700_000)
    out, info = build(tmp_path, root, chunk_mb=1)
    assert info["n_parts"] >= 2
    sums = (out / cb.PARTS_SUMS).read_text().splitlines()
    assert len(sums) == info["n_parts"]


def test_test_entries_in_train_or_val_lists_are_refused(tmp_path):
    root = make_dataset(tmp_path)
    with open(root / "lists" / "train.txt", "a") as fh:
        fh.write("images/rdd2022/test/rdd2022_test_0.jpg\n")
    with pytest.raises(cb.BundleError, match="test directory"):
        cb.build(root, tmp_path / "out", 1, None)


def test_overlapping_train_and_val_refused(tmp_path):
    root = make_dataset(tmp_path)
    with open(root / "lists" / "train.txt", "a") as fh:
        fh.write("images/rdd2022/val/rdd2022_val_0.jpg\n")
    with pytest.raises(cb.BundleError, match="both train and val"):
        cb.build(root, tmp_path / "out", 1, None)


def test_missing_label_refused(tmp_path):
    root = make_dataset(tmp_path)
    (root / "labels" / "hrp4k" / "val" / "hrp4k_val_1.txt").unlink()
    with pytest.raises(cb.BundleError, match="label file missing"):
        cb.build(root, tmp_path / "out", 1, None)


def test_manifest_without_split_column_refused(tmp_path):
    root = make_dataset(tmp_path)
    (root / "manifest.csv").write_text("source,path\nrdd2022,a\n")
    with pytest.raises(cb.BundleError, match="split"):
        cb.build(root, tmp_path / "out", 1, None)


def _transfer(tmp_path, out):
    remote = tmp_path / "remote_upload"
    cb.upload(cb.argparse.Namespace(out=out, remote_dir=str(remote), transport="cp", streams=2, retries=1, stagger=0))
    dest = tmp_path / "instance"
    dest.mkdir()
    return remote, dest


def test_wrong_pinned_digest_is_refused(tmp_path):
    out, info = build(tmp_path, make_dataset(tmp_path))
    remote, dest = _transfer(tmp_path, out)
    with pytest.raises(cb.BundleError, match="pinned"):
        cb.verify(remote, dest, "0" * 64)
    assert not (dest / "data").exists()


def test_damaged_part_is_refused(tmp_path):
    out, info = build(tmp_path, make_dataset(tmp_path))
    remote, dest = _transfer(tmp_path, out)
    part = sorted(remote.glob(cb.PART_PREFIX + "*"))[0]
    data = bytearray(part.read_bytes())
    data[len(data) // 2] ^= 0xFF
    part.write_bytes(bytes(data))
    with pytest.raises(cb.BundleError, match="damaged parts"):
        cb.verify(remote, dest, info["digest_sha256"])


def test_digest_mismatch_after_unpack_is_refused(tmp_path):
    root = make_dataset(tmp_path)
    out, info = build(tmp_path, root)
    # re-pin a digest that claims different content for one file but is internally consistent
    digest = json.loads((out / cb.DIGEST_NAME).read_text())
    key = next(k for k in digest["files"] if k.startswith("labels/"))
    digest["files"][key] = "f" * 64
    raw = (json.dumps(digest, indent=1, sort_keys=True) + "\n").encode()
    (out / cb.DIGEST_NAME).write_bytes(raw)
    remote, dest = _transfer(tmp_path, out)
    with pytest.raises(cb.BundleError, match="differs from digest"):
        cb.verify(remote, dest, cb.sha256_bytes(raw))


def test_extra_file_in_bundle_dir_detected(tmp_path):
    root = make_dataset(tmp_path)
    out, info = build(tmp_path, root)
    digest = json.loads((out / cb.DIGEST_NAME).read_text())
    gone = next(k for k in digest["files"] if k.startswith("images/hrp4k/val"))
    del digest["files"][gone]
    raw = (json.dumps(digest, indent=1, sort_keys=True) + "\n").encode()
    (out / cb.DIGEST_NAME).write_bytes(raw)
    remote, dest = _transfer(tmp_path, out)
    with pytest.raises(cb.BundleError, match="file set differs"):
        cb.verify(remote, dest, cb.sha256_bytes(raw))


def test_label_for_follows_ultralytics_rule():
    assert cb.label_for("images/rdd2022/train/a.b.jpg") == "labels/rdd2022/train/a.b.txt"
    with pytest.raises(cb.BundleError):
        cb.label_for("pics/a.jpg")


def test_output_dir_must_be_empty(tmp_path):
    root = make_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(cb.BundleError, match="not empty"):
        cb.build(root, out, 1, None)


def test_code_bundle_has_scripts_and_no_data_or_test_material(tmp_path):
    import tarfile

    cb.build_code(tmp_path)
    with tarfile.open(tmp_path / cb.CODE_NAME) as tar:
        names = tar.getnames()
    for needed in ("pyproject.toml", "configs/train_unified_v1.yaml", "scripts/vast_remote_unified.sh",
                   "scripts/cloud_bundle.py", "scripts/export_checkpoints.py",
                   "src/roadsense/training/train.py"):
        assert needed in names, needed
    assert not [n for n in names if n.endswith((".pt", ".jpg", ".sqlite")) or "__pycache__" in n]
    assert (tmp_path / "baseline_yolov8s_rdd2022_best.pt").is_file()  # weights travel beside it


# ----------------------------------------------------------------------------- scp upload robustness
class FakeRemote:
    """Stands in for ssh/scp: a local directory plays the instance; failures can be scripted per file."""

    def __init__(self, remote_dir, fail_first=None, corrupt=None):
        self.dir = Path(remote_dir)
        self.fail_first = dict(fail_first or {})  # name -> number of scp attempts that fail
        self.corrupt = set(corrupt or ())          # names that arrive damaged (same size) on the first send
        self.starts, self.sleeps, self.attempts = [], [], {}

    def run(self, cmd, capture_output=True, text=False, cwd=None, encoding=None, errors=None):
        import subprocess as sp
        import time as _t

        if cmd[0] == "scp":
            name = cmd[-2]
            self.starts.append(_t.monotonic())
            self.attempts[name] = self.attempts.get(name, 0) + 1
            if self.fail_first.get(name, 0) > 0:
                self.fail_first[name] -= 1
                return sp.CompletedProcess(cmd, 1, b"", b"Connection reset")
            self.dir.mkdir(parents=True, exist_ok=True)
            data = (Path(cwd) / name).read_bytes()
            if name in self.corrupt and self.attempts[name] == 1:
                data = bytes([data[0] ^ 1]) + data[1:]
            (self.dir / name).write_bytes(data)
            return sp.CompletedProcess(cmd, 0, b"", b"")
        remote = cmd[-1]
        out = ""
        if "stat -c" in remote:
            self.dir.mkdir(parents=True, exist_ok=True)
            out = "".join(f"{p.name} {p.stat().st_size}\n" for p in self.dir.iterdir() if p.is_file())
        elif "sha256sum -c" in remote:
            for line in (self.dir / "parts.sha256").read_text().splitlines():
                h, _, name = line.partition("  ")
                path = self.dir / name
                ok = path.is_file() and cb.sha256_file(path) == h
                out += f"{name}: {'OK' if ok else 'FAILED'}\n"
        elif "rm -f" in remote:
            (self.dir / remote.rsplit("/", 1)[1]).unlink(missing_ok=True)
        return sp.CompletedProcess(cmd, 0, out, "")


def scp_args(out, remote, **kw):
    return cb.argparse.Namespace(out=out, remote_dir=str(remote), transport="scp", streams=16, retries=4,
                                 stagger=kw.pop("stagger", 0.05), host="h", port=22, user="root",
                                 key="k", known_hosts=None, **kw)


def many_parts_bundle(tmp_path, n_parts=6):
    root = make_dataset(tmp_path)
    for p in (root / "images" / "hrp4k" / "train").glob("*.jpg"):
        p.write_bytes(b"x" * 700_000)
    out, info = build(tmp_path, root, chunk_mb=1)
    cb.build_code(out)
    return out, info


def install_fake(monkeypatch, fake):
    monkeypatch.setattr(cb, "_run", fake.run)
    monkeypatch.setattr(cb, "_sleep", lambda s: fake.sleeps.append(s))


def test_connection_starts_are_staggered_and_everything_arrives(tmp_path, monkeypatch):
    out, _ = many_parts_bundle(tmp_path)
    fake = FakeRemote(tmp_path / "remote")
    install_fake(monkeypatch, fake)
    monkeypatch.setattr(cb, "_sleep", cb.time.sleep)  # real sleeps: the stagger is what is measured
    assert cb.upload(scp_args(out, fake.dir, stagger=0.1)) == 0
    sent = sorted(p.name for p in out.iterdir())
    assert sorted(p.name for p in fake.dir.iterdir()) == sent
    gaps = [b - a for a, b in zip(sorted(fake.starts), sorted(fake.starts)[1:])]
    assert len(fake.starts) == len(sent) and min(gaps) >= 0.06, min(gaps)


def test_failed_part_is_retried_with_backoff_and_others_are_unaffected(tmp_path, monkeypatch):
    out, _ = many_parts_bundle(tmp_path)
    fake = FakeRemote(tmp_path / "remote", fail_first={"data_part_0001": 2})
    install_fake(monkeypatch, fake)
    assert cb.upload(scp_args(out, fake.dir)) == 0
    assert fake.attempts["data_part_0001"] == 3 and fake.attempts["data_part_0000"] == 1
    backoffs = [x for x in fake.sleeps if x >= 1]  # the rest are stagger sleeps
    assert len(backoffs) == 2 and 2 <= backoffs[0] < 4 and backoffs[1] >= 4 and max(backoffs) <= 61
    assert (fake.dir / "data_part_0001").is_file()


def test_a_part_that_never_uploads_does_not_abort_the_others(tmp_path, monkeypatch, capsys):
    out, _ = many_parts_bundle(tmp_path)
    fake = FakeRemote(tmp_path / "remote", fail_first={"data_part_0001": 99})
    install_fake(monkeypatch, fake)
    assert cb.upload(scp_args(out, fake.dir)) == 1
    arrived = {p.name for p in fake.dir.iterdir()}
    assert "data_part_0001" not in arrived
    assert arrived == {p.name for p in out.iterdir()} - {"data_part_0001"}
    assert "data_part_0001" in capsys.readouterr().err
    # re-running resumes: only the missing part is sent, and now succeeds
    fake.fail_first.clear()
    fake.attempts.clear()
    assert cb.upload(scp_args(out, fake.dir)) == 0
    assert list(fake.attempts) == ["data_part_0001"]


def test_resume_skips_complete_parts(tmp_path, monkeypatch):
    out, _ = many_parts_bundle(tmp_path)
    fake = FakeRemote(tmp_path / "remote")
    install_fake(monkeypatch, fake)
    assert cb.upload(scp_args(out, fake.dir)) == 0
    fake.attempts.clear()
    assert cb.upload(scp_args(out, fake.dir)) == 0
    assert fake.attempts == {}


def test_same_size_corruption_is_caught_by_the_remote_sha256_and_resent(tmp_path, monkeypatch, capsys):
    out, _ = many_parts_bundle(tmp_path)
    fake = FakeRemote(tmp_path / "remote", corrupt={"data_part_0000"})
    install_fake(monkeypatch, fake)
    assert cb.upload(scp_args(out, fake.dir)) == 0
    assert fake.attempts["data_part_0000"] == 2
    assert "re-sending once" in capsys.readouterr().out
    assert cb.sha256_file(fake.dir / "data_part_0000") == cb.sha256_file(out / "data_part_0000")


def test_backoff_grows_and_is_capped():
    a = [cb.backoff_seconds(n, jitter=0) for n in (1, 2, 3, 4, 5, 6, 7)]
    assert a == [2, 4, 8, 16, 32, 60, 60]


def test_remote_dir_rewritten_by_git_bash_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "code.tgz").write_bytes(b"x")
    with pytest.raises(cb.BundleError, match="MSYS_NO_PATHCONV"):
        cb.upload(scp_args(out, "C:/Program Files/Git/workspace/roadsense/upload"))


def test_code_bundle_has_lf_line_endings_and_executable_scripts(tmp_path):
    import tarfile

    cb.build_code(tmp_path)
    with tarfile.open(tmp_path / cb.CODE_NAME) as tar:
        for member in tar.getmembers():
            if member.name.endswith((".sh", ".py", ".yaml", ".toml")):
                assert b"\r\n" not in tar.extractfile(member).read(), member.name
        assert tar.getmember("scripts/vast_remote_unified.sh").mode & 0o111
