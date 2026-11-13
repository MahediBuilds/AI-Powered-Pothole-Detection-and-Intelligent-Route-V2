"""Build, upload and verify the cloud upload bundle for the unified_v1 training run.

Stdlib only, so the same file runs on this machine and on the rented instance.

    # local: train+val only, chunked, with a content digest; prints the digest's sha256
    python scripts/cloud_bundle.py build --dataset data/processed/unified_v1 \
        --out data/cloud_upload/unified_v1 [--chunk-mb 128] [--source-digest FILE]
    # local: project code + initial weights as code.tgz (a few tens of MB)
    python scripts/cloud_bundle.py code --out data/cloud_upload/unified_v1
    # local: parallel upload of every file in --out to the instance (--only code.tgz first, if wanted)
    python scripts/cloud_bundle.py upload --out data/cloud_upload/unified_v1 \
        --host HOST --port PORT --key ~/.ssh/vast_id_rsa --streams 12
    # instance (after the chunks arrived): check, unpack and verify against the pinned digest
    python scripts/cloud_bundle.py verify --upload-dir upload --dest . --expect-digest-sha256 HEX

Why chunks and parallel streams: one TCP stream from this machine measured about 0.5 MB/s
while 16 parallel streams reached about 4 MB/s (see docs/cloud_run_unified_v1.md).

The bundle holds ONLY the images and labels named by lists/train.txt and lists/val_pooled.txt,
those two lists plus the two per-source val lists, dataset.yaml and a manifest filtered to
non-test rows. Test lists and test images are never read into it, and ``build`` and ``verify``
both fail if any path under a ``test`` directory is involved. List entries are rewritten as
dataset-root-relative posix paths in the bundle, so the entry format of the source lists
(relative, absolute, either slash style) does not matter on the instance.

The digest (``bundle_digest.json``) covers every file in the bundle (sha256 of each), the
per-source/split counts, the manifest and dataset.yaml hashes. Its own sha256 is printed by
``build`` and passed to ``verify`` on the command line, so a damaged digest cannot approve a
damaged dataset. ``--source-digest`` records the full-dataset digest produced by
scripts/dataset_digest.py (or its successor) for provenance only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATASET_REL = "data/processed/unified_v1"          # where the bundle unpacks, relative to the project root
LISTS_IN_BUNDLE = ("train", "val_pooled", "rdd2022_val", "hrp4k_val")
LISTS_FOR_FILES = ("train", "val_pooled")
DIGEST_NAME = "bundle_digest.json"
PART_PREFIX = "data_part_"
PARTS_SUMS = "parts.sha256"
CODE_NAME = "code.tgz"
CODE_FILES = [
    "pyproject.toml", "README.md",
    "configs/train_unified_v1.yaml",
    "scripts/vast_remote_unified.sh", "scripts/cloud_bundle.py", "scripts/export_checkpoints.py",
]
# Sent beside code.tgz as its own file so that code.tgz stays small (setup needs only the code).
WEIGHTS_REL = "pretrained/baseline_yolov8s_rdd2022_best.pt"
CODE_TREES = ["src/roadsense"]


class BundleError(Exception):
    pass


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_test_dir(rel: str) -> bool:
    parts = PurePosixPath(rel.replace("\\", "/")).parts
    return any(p.lower() == "test" for p in parts[:-1])


def resolve_entry(entry: str, list_path: Path, root: Path) -> Path:
    """Absolute path of one list entry (relative to the dataset root or the list file, or absolute)."""
    norm = Path(entry.replace("\\", "/"))
    for cand in ([norm] if norm.is_absolute() else [root / norm, list_path.parent / norm]):
        if cand.is_file():
            return cand.resolve()
    raise BundleError(f"{list_path}: listed image not found: {entry}")


def read_list(list_path: Path, root: Path) -> list[str]:
    """Dataset-root-relative posix paths named by a list file."""
    out = []
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise BundleError(f"empty list: {list_path}")
    for ln in lines:
        if has_test_dir(ln):
            raise BundleError(f"{list_path}: entry under a test directory: {ln}")
        absolute = resolve_entry(ln, list_path, root)
        try:
            rel = absolute.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise BundleError(f"{list_path}: entry outside the dataset root: {ln}") from exc
        if has_test_dir(rel):
            raise BundleError(f"{list_path}: entry under a test directory: {ln}")
        out.append(rel)
    return out


def label_for(image_rel: str) -> str:
    """Ultralytics' image -> label rule: the last /images/ becomes /labels/, suffix .txt."""
    sa, sb = "/images/", "/labels/"
    path = "/" + image_rel
    idx = path.rfind(sa)
    if idx < 0:
        raise BundleError(f"image path has no /images/ component: {image_rel}")
    return (path[:idx] + sb + path[idx + len(sa):]).rsplit(".", 1)[0][1:] + ".txt"


def compute_digest(dest_dataset: Path, rels: list[str], threads: int = 8) -> dict:
    """Digest of the listed files under ``dest_dataset`` (relative posix paths)."""
    def one(rel: str) -> tuple[str, str]:
        return rel, sha256_file(dest_dataset / rel)

    with cf.ThreadPoolExecutor(threads) as pool:
        files = dict(pool.map(one, sorted(rels)))
    counts: dict[str, dict[str, int]] = {}
    for rel in files:
        parts = PurePosixPath(rel).parts
        if len(parts) == 4 and parts[0] in ("images", "labels"):
            key = f"{parts[1]}/{parts[2]}"
            counts.setdefault(key, {"images": 0, "labels": 0})[parts[0]] += 1
    return {
        "format": 1,
        "n_files": len(files),
        "files": files,
        "counts": dict(sorted(counts.items())),
        "files_sha256": sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()),
    }


class ChunkWriter(io.RawIOBase):
    """File-like that splits what is written into fixed-size part files."""

    def __init__(self, out_dir: Path, chunk_bytes: int):
        self.out_dir, self.chunk_bytes = out_dir, chunk_bytes
        self.index, self.fh, self.used, self.names = 0, None, 0, []

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        view = memoryview(data)
        total = len(view)
        while len(view):
            if self.fh is None or self.used >= self.chunk_bytes:
                self._rotate()
            take = min(len(view), self.chunk_bytes - self.used)
            self.fh.write(view[:take])
            self.used += take
            view = view[take:]
        return total

    def _rotate(self) -> None:
        if self.fh:
            self.fh.close()
        name = f"{PART_PREFIX}{self.index:04d}"
        self.names.append(name)
        self.fh = open(self.out_dir / name, "wb")
        self.index += 1
        self.used = 0

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None
        super().close()


# ----------------------------------------------------------------------------- build
def build(dataset: Path, out: Path, chunk_mb: int, source_digest: Path | None,
          dataset_rel: str = DATASET_REL) -> dict:
    dataset = dataset.resolve()
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"output directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lists_dir = dataset / "lists"
    rel_lists = {name: read_list(lists_dir / f"{name}.txt", dataset) for name in LISTS_IN_BUNDLE
                 if (lists_dir / f"{name}.txt").is_file()}
    for needed in LISTS_FOR_FILES:
        if needed not in rel_lists:
            raise BundleError(f"missing lists/{needed}.txt")
    images = sorted({rel for name in LISTS_FOR_FILES for rel in rel_lists[name]})
    train, val = set(rel_lists["train"]), set(rel_lists["val_pooled"])
    if train & val:
        raise BundleError(f"{len(train & val)} images are in both train and val lists, e.g. {sorted(train & val)[0]}")
    labels = [label_for(i) for i in images]
    for rel in labels:
        if not (dataset / rel).is_file():
            raise BundleError(f"label file missing: {dataset / rel}")
    # in-memory files: rewritten lists, dataset.yaml, filtered manifest
    extra: dict[str, bytes] = {}
    for name, rels in rel_lists.items():
        extra[f"lists/{name}.txt"] = ("\n".join(rels) + "\n").encode()
    extra["dataset.yaml"] = (dataset / "dataset.yaml").read_bytes()
    manifest_in = dataset / "manifest.csv"
    with open(manifest_in, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        split_col = next((c for c in reader.fieldnames or [] if c.lower() == "split"), None)
        if split_col is None:
            raise BundleError("manifest.csv has no 'split' column; cannot filter out test rows")
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        kept = 0
        for row in reader:
            if str(row[split_col]).strip().lower() != "test":
                writer.writerow(row)
                kept += 1
    extra["manifest.csv"] = buf.getvalue().encode()
    # stage a digest from the source files and the in-memory ones
    print(f"hashing {len(images) + len(labels)} files ...", flush=True)
    digest = compute_digest(dataset, images + labels)
    for rel, data in extra.items():
        digest["files"][rel] = sha256_bytes(data)
    digest["n_files"] = len(digest["files"])
    digest["files_sha256"] = sha256_bytes(
        "\n".join(f"{k} {v}" for k, v in sorted(digest["files"].items())).encode())
    digest["n_manifest_rows_kept"] = kept
    digest["dataset_rel"] = dataset_rel
    digest["source_digest"] = json.loads(source_digest.read_text(encoding="utf-8-sig")) if source_digest else None
    # tar: members under data/processed/unified_v1/..., streamed into chunks
    writer_out = ChunkWriter(out, chunk_mb * 1024 * 1024)
    total = 0
    with tarfile.open(fileobj=writer_out, mode="w|") as tar:
        for rel in images + labels:
            tar.add(dataset / rel, arcname=f"{dataset_rel}/{rel}", recursive=False)
            total += (dataset / rel).stat().st_size
        for rel, data in extra.items():
            info = tarfile.TarInfo(f"{dataset_rel}/{rel}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    writer_out.close()
    digest_bytes = (json.dumps(digest, indent=1, sort_keys=True) + "\n").encode()
    (out / DIGEST_NAME).write_bytes(digest_bytes)
    sums = [f"{sha256_file(out / n)}  {n}" for n in writer_out.names]
    (out / PARTS_SUMS).write_text("\n".join(sums) + "\n", encoding="utf-8")
    info = {
        "n_images": len(images), "n_train": len(train), "n_val": len(val), "n_parts": len(writer_out.names),
        "image_bytes": total, "digest_sha256": sha256_bytes(digest_bytes), "counts": digest["counts"],
    }
    print(json.dumps(info, indent=1))
    print(f"PIN THIS: --expect-digest-sha256 {info['digest_sha256']}")
    return info


TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt", ".sql", ".json", ".html", ".js", ".css"}


def _add_code_file(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    """Add a file to the code tarball; text files get LF line endings (a CRLF shell script fails on Linux
    with 'set: pipefail: invalid option name'), shell scripts are executable."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(bytes([13, 10]), bytes([10]))  # CRLF -> LF
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime = len(data), int(path.stat().st_mtime)
    info.mode = 0o755 if path.suffix == ".sh" else 0o644
    tar.addfile(info, io.BytesIO(data))


def build_code(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / CODE_NAME
    with tarfile.open(target, "w:gz") as tar:
        for rel in CODE_FILES:
            if not (ROOT / rel).is_file():
                raise BundleError(f"missing file for the code bundle: {rel}")
            _add_code_file(tar, ROOT / rel, rel)
        for tree in CODE_TREES:
            for path in sorted((ROOT / tree).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    _add_code_file(tar, path, path.relative_to(ROOT).as_posix())
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)  sha256 {sha256_file(target)}")
    weights = ROOT / WEIGHTS_REL
    if not weights.is_file():
        raise BundleError(f"missing initial weights: {weights}")
    shutil.copyfile(weights, out / weights.name)
    print(f"copied {weights.name}  sha256 {sha256_file(out / weights.name)}")
    return target


# ----------------------------------------------------------------------------- upload
_run = subprocess.run  # seams for the tests
_sleep = time.sleep


class Stagger:
    """Spaces connection starts at least ``interval`` seconds apart (sshd MaxStartups defaults to 10:30:100,
    so 16 simultaneous unauthenticated connections can be dropped)."""

    def __init__(self, interval: float):
        self.interval, self.next_start = interval, 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        if start > now:
            _sleep(start - now)


def backoff_seconds(attempt: int, base: float = 2.0, cap: float = 60.0, jitter: float = 1.0) -> float:
    """Exponential backoff (2, 4, 8 ... capped) plus up to ``jitter`` seconds so retries do not synchronise."""
    return min(cap, base * 2 ** (attempt - 1)) + random.uniform(0, jitter)


def _ssh_opts(args) -> list[str]:
    opts = ["-i", str(args.key), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if args.known_hosts:
        opts += ["-o", f"UserKnownHostsFile={args.known_hosts}"]
    return opts


def _ssh(args, remote_cmd: str):
    cmd = ["ssh", *_ssh_opts(args), "-p", str(args.port), f"{args.user}@{args.host}", remote_cmd]
    return _run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _remote_sizes(args) -> dict[str, int]:
    res = _ssh(args, f"mkdir -p {args.remote_dir} && cd {args.remote_dir} && stat -c '%n %s' * 2>/dev/null || true")
    sizes = {}
    for line in res.stdout.splitlines():
        name, _, size = line.rpartition(" ")
        if name and size.isdigit():
            sizes[name] = int(size)
    return sizes


def _remote_check(args) -> list[str]:
    """Names of data parts whose sha256 on the remote side differs from parts.sha256 (or are missing)."""
    res = _ssh(args, f"cd {args.remote_dir} && sha256sum -c parts.sha256 2>&1 || true")
    return sorted({line.split(":")[0] for line in res.stdout.splitlines()
                   if line.rstrip().endswith(("FAILED", "No such file or directory", "FAILED open or read"))
                   and line.startswith(PART_PREFIX)})


def upload(args) -> int:
    """Send the files in ``args.out`` to the instance over several scp connections.

    Resumable: files already complete (same size) on the remote side are skipped. One failed file never
    stops the others: failures are retried with backoff, then reported, and the exit code is 1 (re-run the
    same command to retry only those). After the transfer every data part is checked with sha256sum on
    the remote side against parts.sha256, and parts that differ are re-sent once.
    """
    files = sorted(p for p in args.out.iterdir() if p.is_file())
    if getattr(args, "only", None):
        files = [p for p in files if p.name in set(args.only)]
    if not files:
        raise BundleError(f"nothing to upload in {args.out}")
    stagger = Stagger(getattr(args, "stagger", 0.5))
    if args.transport == "scp" and re.match(r"^[A-Za-z]:/(.*/)?Git/", str(args.remote_dir).replace(chr(92), "/")):
        raise BundleError(
            f"remote dir {args.remote_dir!r} points into the Git for Windows folder: Git Bash rewrote the POSIX path. "
            "Run 'export MSYS_NO_PATHCONV=1' first.")
    if args.transport == "cp":
        Path(args.remote_dir).mkdir(parents=True, exist_ok=True)
        remote_sizes: dict[str, int] = {p.name: p.stat().st_size for p in Path(args.remote_dir).iterdir()}
    else:
        remote_sizes = _remote_sizes(args)
    todo = [p for p in files if remote_sizes.get(p.name) != p.stat().st_size]
    print(f"{len(todo)} files, {sum(p.stat().st_size for p in todo) / 1e6:.0f} MB to send "
          f"({len(files) - len(todo)} already complete), {args.streams} streams", flush=True)
    t0 = time.time()
    failed = _send_all(args, todo, stagger, t0)
    names = {p.name for p in files}
    if args.transport == "scp" and PARTS_SUMS in names and any(n.startswith(PART_PREFIX) for n in names):
        bad = _remote_check(args)
        if bad:
            print(f"remote sha256 differs for {bad}: re-sending once", flush=True)
            for name in bad:
                _ssh(args, f"rm -f {args.remote_dir}/{name}")
            failed = sorted(set(failed) | set(_send_all(args, [args.out / n for n in bad], stagger, time.time())))
            bad = _remote_check(args)
            if bad:
                failed = sorted(set(failed) | set(bad))
        if not failed:
            print("remote sha256 check: all parts match parts.sha256", flush=True)
    if failed:
        print(f"FAILED files: {sorted(set(failed))}; re-run the same command to retry them", file=sys.stderr)
        return 1
    print(f"upload finished in {(time.time() - t0) / 60:.1f} min")
    return 0


def _send_all(args, todo: list[Path], stagger: Stagger, t0: float) -> list[str]:
    # code first (setup needs it), then big files first so the tail of the transfer is small files
    todo = sorted(todo, key=lambda p: (p.name != CODE_NAME, -p.stat().st_size))
    total = sum(p.stat().st_size for p in todo)
    done_bytes, failed = 0, []

    def send(path: Path) -> bool:
        for attempt in range(1, args.retries + 1):
            stagger.wait()
            if args.transport == "cp":
                shutil.copyfile(path, Path(args.remote_dir) / path.name)
                return True
            # bare file name with cwd=out: no drive letter or backslash for scp to misread as host:path
            cmd = ["scp", *_ssh_opts(args), "-P", str(args.port), path.name,
                   f"{args.user}@{args.host}:{args.remote_dir}/"]
            try:
                res = _run(cmd, capture_output=True, cwd=args.out)
                if res.returncode == 0:
                    return True
                err = (res.stderr or b"").decode("utf-8", "replace").strip().splitlines()
                print(f"{path.name}: scp exit {res.returncode}: {err[-1] if err else ''}", flush=True)
            except OSError as exc:
                print(f"{path.name}: {exc}", flush=True)
            if attempt < args.retries:
                wait = backoff_seconds(attempt)
                print(f"retry {attempt}/{args.retries - 1} for {path.name} in {wait:.0f} s", flush=True)
                _sleep(wait)
        return False

    with cf.ThreadPoolExecutor(args.streams) as pool:
        futures = {pool.submit(send, p): p for p in todo}
        for fut in cf.as_completed(futures):
            path = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # one broken file must not stop the rest
                print(f"{path.name}: unexpected error {exc!r}", flush=True)
                ok = False
            if not ok:
                failed.append(path.name)
                print(f"GAVE UP on {path.name}", flush=True)
                continue
            done_bytes += path.stat().st_size
            el = max(time.time() - t0, 1e-6)
            rate = done_bytes / el / 1e6
            eta = (total - done_bytes) / (done_bytes / el) if done_bytes else float("nan")
            print(f"{done_bytes / 1e6:8.0f}/{total / 1e6:.0f} MB  {rate:5.2f} MB/s  eta {eta / 60:5.1f} min", flush=True)
    return failed


# ----------------------------------------------------------------------------- verify
def verify(upload_dir: Path, dest: Path, expect_digest_sha256: str, threads: int = 8) -> dict:
    digest_path = upload_dir / DIGEST_NAME
    if sha256_file(digest_path) != expect_digest_sha256:
        raise BundleError(f"{DIGEST_NAME} sha256 differs from the pinned value {expect_digest_sha256}")
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    parts = sorted(p for p in upload_dir.iterdir() if p.name.startswith(PART_PREFIX))
    expected = {}
    for line in (upload_dir / PARTS_SUMS).read_text(encoding="utf-8").splitlines():
        h, _, name = line.partition("  ")
        expected[name] = h
    if sorted(expected) != [p.name for p in parts]:
        raise BundleError("part files do not match parts.sha256 (missing or extra parts)")
    with cf.ThreadPoolExecutor(threads) as pool:
        got = dict(zip([p.name for p in parts], pool.map(sha256_file, parts)))
    bad = [n for n in got if got[n] != expected[n]]
    if bad:
        raise BundleError(f"damaged parts: {bad}")
    print(f"{len(parts)} parts verified; unpacking", flush=True)
    dest_dataset = dest / digest["dataset_rel"]
    if dest_dataset.exists() and any(dest_dataset.iterdir()):
        raise BundleError(f"{dest_dataset} already exists and is not empty")
    class Joined(io.RawIOBase):
        def __init__(self, paths):
            self.paths, self.cur = list(paths), None

        def readable(self):
            return True

        def readinto(self, b):
            while True:
                if self.cur is None:
                    if not self.paths:
                        return 0
                    self.cur = open(self.paths.pop(0), "rb")
                n = self.cur.readinto(b)
                if n:
                    return n
                self.cur.close()
                self.cur = None

    with tarfile.open(fileobj=io.BufferedReader(Joined(parts)), mode="r|") as tar:
        for member in tar:
            target = (dest / member.name).resolve()
            if dest.resolve() not in target.parents or has_test_dir(member.name):
                raise BundleError(f"refusing member {member.name}")
            if not (member.isreg() or member.isdir()):
                raise BundleError(f"refusing non-file member {member.name}")
            if hasattr(tarfile, "data_filter"):
                tar.extract(member, dest, filter="data")
            else:  # Python without extraction filters: the checks above cover traversal and links
                tar.extract(member, dest)
    present = sorted(p.relative_to(dest_dataset).as_posix() for p in dest_dataset.rglob("*") if p.is_file())
    if present != sorted(digest["files"]):
        extra = sorted(set(present) - set(digest["files"]))[:5]
        missing = sorted(set(digest["files"]) - set(present))[:5]
        raise BundleError(f"file set differs from digest: extra {extra} missing {missing}")
    if any(has_test_dir(p) for p in present):
        raise BundleError("a test directory path is present in the unpacked dataset")
    local = compute_digest(dest_dataset, present, threads)
    if local["files"] != digest["files"] or local["counts"] != digest["counts"]:
        wrong = [k for k in digest["files"] if local["files"].get(k) != digest["files"][k]][:5]
        raise BundleError(f"content differs from digest, e.g. {wrong}")
    print(f"DIGEST OK: {digest['n_files']} files, counts {json.dumps(digest['counts'])}")
    return digest


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--chunk-mb", type=int, default=128)
    b.add_argument("--source-digest", type=Path)
    b.add_argument("--dataset-rel", default=DATASET_REL,
                   help="where the bundle unpacks, relative to the project root (tests/smoke runs only)")
    c = sub.add_parser("code")
    c.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("upload")
    u.add_argument("--out", type=Path, required=True, help="directory holding the files to send")
    u.add_argument("--host")
    u.add_argument("--port", type=int, default=22)
    u.add_argument("--user", default="root")
    u.add_argument("--key", type=Path)
    u.add_argument("--known-hosts", type=Path)
    u.add_argument("--remote-dir", default="/workspace/roadsense/upload")
    u.add_argument("--streams", type=int, default=12)
    u.add_argument("--retries", type=int, default=5, help="attempts per file (backoff 2, 4, 8, 16 s between)")
    u.add_argument("--stagger", type=float, default=0.5, help="minimum seconds between connection starts")
    u.add_argument("--only", nargs="+", help="send only these file names, e.g. code.tgz")
    u.add_argument("--transport", choices=("scp", "cp"), default="scp")
    v = sub.add_parser("verify")
    v.add_argument("--upload-dir", type=Path, required=True)
    v.add_argument("--dest", type=Path, default=Path("."))
    v.add_argument("--expect-digest-sha256", required=True)
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "build":
            build(ns.dataset, ns.out, ns.chunk_mb, ns.source_digest, ns.dataset_rel)
        elif ns.cmd == "code":
            build_code(ns.out)
        elif ns.cmd == "upload":
            if ns.transport == "scp" and not (ns.host and ns.key):
                ap.error("--host and --key are required for scp")
            return upload(ns)
        else:
            verify(ns.upload_dir, ns.dest, ns.expect_digest_sha256)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
