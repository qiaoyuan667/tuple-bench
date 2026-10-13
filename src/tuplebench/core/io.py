"""Strict JSON, deterministic hashing, exclusive output paths. No pickle loading."""
from __future__ import annotations
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

class ContractError(ValueError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)

def _pairs(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, f'Duplicate JSON key: {k}')
        d[k] = v
    return d

def parse_json(text: str) -> Any:

    def bad(value):
        raise ContractError(f'Non-finite JSON number: {value}')
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad)

def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            value = parse_json(stream.read())
        return _restore_arrays(value, path)
    return parse_json(path.read_text(encoding='utf-8'))

def _restore_arrays(value, source):
    """Restore lossless numeric storage; object arrays and pickle are forbidden."""
    if isinstance(value, dict):
        if set(value) == {'__numeric_array__', 'key'}:
            import numpy as np
            artifact_root = next((p for p in source.resolve().parents if p.name == 'artifacts'), None)
            require(artifact_root is not None, 'Numeric array reference outside artifact storage')
            relative = Path(value['__numeric_array__'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe numeric array reference')
            file = artifact_root.parent / relative
            require(artifact_root in file.resolve().parents and not file.is_symlink(), 'Unsafe array file')
            with np.load(file, allow_pickle=False) as stored:
                key = value['key']
                require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'Invalid array key')
                array = stored[key]
                require(array.dtype.kind in 'fi' and np.isfinite(array).all(), 'Invalid numeric model array')
                return array
        return {k: _restore_arrays(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_arrays(v, source) for v in value]
    return value

def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = parse_json(line)
                require(isinstance(value, dict), 'JSONL rows must be objects')
                rows.append(value)
            except (ValueError, TypeError) as exc:
                raise ContractError(f'{path}:{i}: {exc}') from exc
    return rows

def dumps(value: Any, pretty: bool=False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2 if pretty else None, separators=None if pretty else (',', ':'))

def digest(value: Any) -> str:
    s = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()

def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def opaque(prefix: str, *parts: Any) -> str:
    return prefix + digest(list(parts))[:24]

def write_json(path: str | Path, value: Any, *, overwrite: bool=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'x'
    with path.open(mode, encoding='utf-8') as f:
        f.write(dumps(value, pretty=True) + '\n')

def write_jsonl(path: str | Path, rows: Iterable[dict], *, overwrite: bool=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if overwrite else 'x', encoding='utf-8') as f:
        for row in rows:
            f.write(dumps(row) + '\n')

def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())

def new_directory(path: str | Path) -> Path:
    p = Path(path)
    require(not p.exists(), f'Refusing to overwrite existing output: {p}')
    p.mkdir(parents=True)
    return p

def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x)
