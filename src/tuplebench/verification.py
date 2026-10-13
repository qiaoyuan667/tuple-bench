"""Checksum verification for the two merged review archives."""
import hashlib
import json
from pathlib import Path


def verify_release(root="."):
    root = Path(root).resolve()
    count = 0
    for manifest in ("SOFTWARE_SHA256SUMS.json", "DATASET_SHA256SUMS.json"):
        content = json.loads((root / manifest).read_text())
        for name, expected in content.items():
            relative = Path(name)
            path = root / relative
            if relative.is_absolute() or ".." in relative.parts or root not in path.resolve().parents:
                raise ValueError("Unsafe checksum path")
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing or unsafe file: " + name)
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected:
                raise ValueError("Checksum mismatch: " + name)
            count += 1
    return dict(status="verified", verified_files=count)
