#!/usr/bin/env python3
"""Read-only streaming hash inventory of saved native backbones under runs/.

Writes a small JSON report only. Never imports tensor/model libraries, loads a
model, downloads, removes, clones or rewrites checkpoint files. Duplicate totals
are logical upper bounds, not a measurement of unique APFS extents.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 8 * 1024 * 1024
PREFERRED_CANONICALS = {
    'runs/minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022/checkpoint/base/model.safetensors',
    'runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/checkpoint/base/model.safetensors',
}


def identity(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'Expected regular nonsymlink file: {path}')
    return {key: getattr(value, 'st_' + key) for key in
            ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns', 'nlink', 'blocks')}


def fingerprint(directory):
    return {str(p.relative_to(directory)): identity(p)
            for p in directory.rglob('*') if p.is_file() and not p.is_symlink()}


def audit_file(path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlink excluded from audit: {part}')
        if part == ROOT:
            break
    before = identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = stream.fileno()
        import os
        metadata = os.fstat(opened)
        if (metadata.st_dev, metadata.st_ino) != (before['dev'], before['ino']):
            raise ValueError(f'File changed while opening: {path}')
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Missing safetensors header: {path}')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, before['size'] - 8):
            raise ValueError(f'Invalid safetensors header: {path}')
        header_bytes = stream.read(length)
        header = json.loads(header_bytes)
        digest.update(raw)
        digest.update(header_bytes)
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    if identity(path) != before:
        raise ValueError(f'File changed while hashing: {path}')
    checkpoint = path.parent.parent
    manifest = json.loads((checkpoint / 'adapter_manifest.json').read_text())
    receipt_path = checkpoint / 'checkpoint_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    claim = receipt.get('files_sha256', {}).get('base/' + path.name)
    actual = digest.hexdigest()
    if claim is not None and actual != claim:
        raise ValueError(f'Current bytes differ from checkpoint receipt: {path}')
    item = {'path': str(path.relative_to(ROOT)), 'identity': before,
            'sha256': actual, 'receipt_sha256_claim': claim,
            'regime': manifest['regime'],
            'original_trainable_parameters': manifest['parameters']['original']['trainable'],
            'dtypes': sorted({v['dtype'] for k, v in header.items() if k != '__metadata__'})}
    print(json.dumps({'verified': item['path'], 'sha256': actual}), flush=True)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    # Reports may not overwrite any source, run, script or existing model file.
    if output.parent != ROOT / 'reports' or output.suffix != '.json' or output.is_symlink():
        parser.error('Output must be a nonsymlink JSON file directly under reports/')
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    runs = ROOT / 'runs'
    source = ROOT / 'checkpoints/minicpm5_source'
    before = fingerprint(runs)
    source_before = fingerprint(source)
    retention = json.loads((ROOT / 'reports/checkpoint_retention_inventory_20261022.json').read_text())
    prior_candidates = [Path(x['path']) for x in retention['retention_entries']
                        if x['status'] == 'DELETE_ELIGIBLE_NOT_DELETED']
    paths = sorted(runs.glob('*/*/base/model*.safetensors'))
    if not paths:
        raise ValueError('No saved native backbones found')
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(audit_file, paths))
    if fingerprint(runs) != before or fingerprint(source) != source_before:
        raise ValueError('Checkpoint/source metadata changed during read-only audit')
    groups = defaultdict(list)
    for item in records:
        item['covered_by_prior_directory_cleanup'] = any(
            Path(item['path']).is_relative_to(p) for p in prior_candidates)
        groups[(item['identity']['size'], item['sha256'])].append(item)
    duplicates, unique_groups = [], []
    for (size, digest), members in sorted(groups.items()):
        members.sort(key=lambda x: (x['path'] not in PREFERRED_CANONICALS,
                                   x['covered_by_prior_directory_cleanup'], x['path']))
        canonical, *copies = members
        for item in copies:
            item['canonical_path'] = canonical['path']
            item['action'] = 'DEDUP_ONLY_PRESERVE_PATH_AND_BYTES'
        duplicates.extend(copies)
        canonical['action'] = 'KEEP_ONE_CANONICAL_WEIGHT_FILE'
        unique_groups.append({'sha256': digest, 'bytes_per_file': size,
                              'canonical_path': canonical['path'], 'file_count': len(members),
                              'duplicate_count': len(copies), 'paths': [x['path'] for x in members]})
    overlap = [x for x in duplicates if x['covered_by_prior_directory_cleanup']]
    extra = [x for x in duplicates if not x['covered_by_prior_directory_cleanup']]
    total = lambda rows: sum(x['identity']['size'] for x in rows)
    prior_bytes = retention['totals_nonoverlapping']['by_status']['DELETE_ELIGIBLE_NOT_DELETED']['logical_bytes']
    frozen_digest = '07db90adf475eae2201fc3f3fbe76849c12cf1777c4e271114af142a048dfbb8'
    report = {
        'status': 'verified_exact_weight_duplicates_no_checkpoint_mutation',
        'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': round(time.monotonic() - start, 3),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': {'scope': 'runs/*/{checkpoint,training_checkpoint_NNNN,adapter_export}/base/model*.safetensors present at audit time',
                   'hash': 'SHA-256 of every full file; compare with local checkpoint receipt when available',
                   'workers': 2, 'stream_chunk_bytes_per_worker': CHUNK_BYTES,
                   'model_or_tensor_libraries_used': False},
        'source': {'path': 'checkpoints/minicpm5_source', 'preserved': True,
                   'metadata_unchanged': True, 'downloaded': False,
                   'note': 'Original BF16 source and saved FP32 bases have different bytes/dtypes. Prior numerical comparison is in reports/minicpm5_D1_checkpoint_audit_20261022.json.'},
        'frozen_source_values_fp32_group_sha256': frozen_digest,
        'totals': {'weight_files': len(records), 'logical_weight_bytes': total(records),
                   'distinct_weight_byte_groups': len(unique_groups),
                   'frozen_source_values_fp32_files': sum(x['sha256'] == frozen_digest for x in records),
                   'duplicate_files_beyond_one_per_group': len(duplicates),
                   'duplicate_logical_bytes_upper_bound': total(duplicates),
                   'overlap_with_prior_directory_cleanup': {'files': len(overlap), 'logical_bytes': total(overlap)},
                   'additional_dedup_outside_prior_cleanup': {'files': len(extra), 'logical_bytes': total(extra)},
                   'prior_directory_cleanup_logical_bytes': prior_bytes,
                   'combined_nonoverlapping_cleanup_and_dedup_upper_bound': prior_bytes + total(extra)},
        'groups': unique_groups, 'files': records,
        'limitations': ['Logical totals do not measure unique APFS extents or guaranteed physical recovery.',
                       'Dedup-only candidates must keep existing paths and exact bytes for current loaders/receipts; do not delete their base files.',
                       'Only weight bytes are compared for sharing. Sidecars and optimizer/RNG state stay independent.',
                       'Do not add dedup totals to the prior cleanup total without subtracting the recorded overlap.',
                       'Equal filename, length or source-lock provenance alone does not prove equal current weights.'],
        'checkpoint_and_source_metadata_unchanged': True,
        'checkpoint_deletions': 0, 'checkpoint_replacements': 0,
        'downloads': 0, 'model_runs': 0, 'training_status': 'PAUSED'}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'totals': report['totals']}), flush=True)


if __name__ == '__main__':
    main()
