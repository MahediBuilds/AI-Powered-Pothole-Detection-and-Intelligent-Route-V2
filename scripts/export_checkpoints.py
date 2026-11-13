"""Export stripped (optimizer-free) copies of a run's checkpoints for download, with a checksum list.

Runs on the training instance (and locally for tests). The run directory is only read.

    python scripts/export_checkpoints.py RUN_DIR OUT_DIR [--quiet] [--only-epochs]

For every RUN_DIR/weights/*.pt that is not yet exported (or changed since), writes
OUT_DIR/weights/<name>.pt through ultralytics' strip_optimizer (about 22 MB instead of about
68 MB), and keeps OUT_DIR/weights_index.json: for each file the source's size/mtime, the
training epoch stored inside it (0-based, as in the name epochN.pt; -1 once Ultralytics has
stripped last.pt/best.pt at the end of training) and its sha256. --only-epochs exports just the epochN.pt files (they never change once written), which is what the
watchdog uses while training; the final pack exports everything, including best.pt and last.pt.
A checkpoint that is still
being written is skipped (it fails to load) and picked up on the next call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(run_dir: Path, out_dir: Path, quiet: bool = False, only_epochs: bool = False) -> dict:
    import torch
    from ultralytics.utils.torch_utils import strip_optimizer

    weights = out_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "weights_index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    for src in sorted((run_dir / "weights").glob("epoch*.pt" if only_epochs else "*.pt")):
        st = src.stat()
        prev = index.get(src.name)
        if prev and prev["src_size"] == st.st_size and prev["src_mtime"] == st.st_mtime:
            continue
        dst = weights / src.name
        try:
            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            epoch = int(ckpt.get("epoch", -1))
            had_optimizer = ckpt.get("optimizer") is not None
            del ckpt
            strip_optimizer(src, dst)
        except Exception as exc:  # file still being written, or unreadable: try again next call
            if not quiet:
                print(f"skip {src.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        index[src.name] = {
            "src_size": st.st_size, "src_mtime": st.st_mtime, "ckpt_epoch_0based": epoch,
            "had_optimizer": had_optimizer, "size": dst.stat().st_size, "sha256": sha256_file(dst),
        }
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        if not quiet:
            print(f"exported {src.name} (epoch {epoch}) -> {dst}")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--only-epochs", action="store_true", help="export epoch*.pt only (best/last change every epoch)")
    ns = ap.parse_args(argv)
    if not (ns.run_dir / "weights").is_dir():
        print(f"error: no weights directory in {ns.run_dir}", file=sys.stderr)
        return 2
    export(ns.run_dir, ns.out_dir, ns.quiet, ns.only_epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
