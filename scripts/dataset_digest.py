"""Print a digest of a converted dataset so two copies can be compared. Read-only.

Covers the manifest, dataset.yaml, every label file (path and bytes) and the image file names of
each split. Image bytes are not hashed here: the manifest carries a SHA-256 per image and
``verify_converted_dataset.py`` checks them.

Usage:
    python scripts/dataset_digest.py <out_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    out_dir = Path(sys.argv[1])
    result: dict[str, object] = {
        "manifest_sha256": sha256_bytes((out_dir / "manifest.csv").read_bytes()),
        "dataset_yaml_sha256": sha256_bytes((out_dir / "dataset.yaml").read_bytes()),
    }
    for split in SPLITS:
        labels = hashlib.sha256()
        label_files = sorted((out_dir / "labels" / split).glob("*.txt"), key=lambda p: p.name)
        for path in label_files:
            labels.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        names = sorted(p.name for p in (out_dir / "images" / split).iterdir() if p.is_file())
        result[split] = {
            "n_images": len(names),
            "n_labels": len(label_files),
            "image_names_sha256": sha256_bytes("\n".join(names).encode("utf-8")),
            "labels_sha256": labels.hexdigest(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
