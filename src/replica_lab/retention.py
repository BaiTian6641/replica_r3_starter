"""Read-only storage inventory and retention policy validation.

This module never deletes, moves or rewrites artifacts. It verifies a declared
policy, measures logical bytes/files and optionally hashes small files so run
payloads and retained evidence remain auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


def _sha256(path, limit):
    if path.stat().st_size > limit:
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def measure(path: Path, *, hash_limit=64 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink():
        return {'kind': 'symlink', 'bytes': 0, 'files': 0, 'sha256': None}
    if path.is_file():
        return {'kind': 'file', 'bytes': path.stat().st_size, 'files': 1,
                'sha256': _sha256(path, hash_limit)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    total = 0
    count = 0
    inventory = []
    for item in sorted(path.rglob('*')):
        if item.is_file() and not item.is_symlink():
            size = item.stat().st_size
            total += size; count += 1
            inventory.append({'path': item.relative_to(path).as_posix(),
                              'bytes': size,
                              'sha256': _sha256(item, hash_limit)})
    return {'kind': 'directory', 'bytes': total, 'files': count,
            'sha256': None, 'inventory': inventory}


def build_manifest(policy: dict, *, root: Path) -> dict:
    root = Path(root).resolve()
    if policy.get('format') != 'replica.retention.policy.v1':
        raise ValueError('unknown retention policy format')
    entries = policy.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('retention entries are required')
    result = []
    for entry in entries:
        path_value = entry.get('path')
        action = entry.get('action')
        if not isinstance(path_value, str) or not path_value:
            raise ValueError('entry path required')
        if action not in {'preserve', 'deletion_candidate'}:
            raise ValueError('unknown retention action')
        target = (root / path_value).resolve()
        if target != root and root not in target.parents:
            raise PermissionError('retention path escapes the repository')
        if not target.exists():
            if action == 'preserve':
                raise FileNotFoundError(f'preserved artifact missing: {path_value}')
            result.append({'path': path_value, 'action': action,
                           'reason': entry.get('reason', ''), 'present': False})
            continue
        measured = measure(target)
        result.append({'path': path_value, 'action': action,
                       'reason': entry.get('reason', ''), 'present': True,
                       **{k: v for k, v in measured.items() if k != 'inventory'},
                       'inventory': measured.get('inventory', [])})
    return {'format': 'replica.retention.manifest.v1',
            'root': str(root), 'total_bytes': sum(x.get('bytes', 0) for x in result),
            'entries': result,
            'deletion_performed': False,
            'notes': 'Read-only inventory; deletion candidates require explicit later approval.'}
