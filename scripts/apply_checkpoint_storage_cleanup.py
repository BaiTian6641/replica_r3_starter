#!/usr/bin/env python3
"""Apply only the pinned 22 October retention/de-duplication decisions.

Default is a metadata-only plan. --apply verifies bytes, uses APFS clones with
rollback, then deletes exactly the archived obsolete runtime trees. No models
are loaded, trained or downloaded. An atomic progress journal records each step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import uuid

import compact_identical_checkpoints as cow

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
 'reports/checkpoint_retention_inventory_20261022.json': '1d356f3a92eb83bd15ab2b90aefa1a2b5e1ea7aec6c5e0a11986987c8f8c175b',
 'reports/base_checkpoint_dedup_candidates_20261022.json': '298da34a17255ca72d4b91e42e0170c2218f96eccb88d2872eefa82b7d41c249',
 'reports/base_checkpoint_duplication_audit_20261022.json': '147b9bb999b18692049f2ae4dd6a2be5f501631bf4386a591f8895b380cb9043',
}


def sha(path):
    return cow.sha256_file(path)


def write_journal(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    cow._no_symlinks(path.parent)
    with temporary.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_path(name, root=ROOT):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid relative path: {name}')
    result = root / relative
    cow._no_symlinks(result)
    return result


def tree_state(path):
    cow._no_symlinks(path)
    if not path.is_dir():
        raise ValueError(f'Missing directory: {path}')
    files = {}
    for p in path.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'Symlink inside candidate tree: {p}')
        if p.is_file():
            files[str(p.relative_to(path))] = cow._snapshot(p)
    return files


def disk(root=ROOT):
    usage = shutil.disk_usage(root)
    return {'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free,
            'df_k': subprocess.check_output(['/bin/df', '-k', str(root)], text=True).splitlines()}


def quiescent(roots):
    listing = subprocess.check_output(['/bin/ps', '-ww', '-axo', 'pid=,ppid=,command='], text=True)
    cow.assert_quiescent(roots, process_listing=listing)
    for line in listing.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and re.search(
            r'(?:^|[ /])(train_text|evaluate_text|qualify_native_adapter|qualify_augmented_generation)\.py(?:\s|$)', fields[2]):
            raise ValueError(f'Model worker still active: PID {fields[0]}')


def make_plan(root=ROOT):
    loaded = {}
    for name, expected in PINNED.items():
        path = safe_path(name, root)
        if sha(path) != expected:
            raise ValueError(f'Approved inventory changed: {name}')
        loaded[name] = json.loads(path.read_text())
    retention, dedup, audit = loaded.values()
    deletions = [x for x in retention['retention_entries'] if x['status'] == 'DELETE_ELIGIBLE_NOT_DELETED']
    if len(deletions) != 16:
        raise ValueError('Expected exactly 16 approved deletion directories')
    protected = [retention['source_path'], *[x['path'] for x in retention['protected_selected_artifacts']],
                 *retention['preserved_negative_exports']]
    delete_paths = [Path(x['path']) for x in deletions]
    for item in deletions:
        relative = Path(item['path'])
        if len(relative.parts) != 3 or relative.parts[0] != 'runs' or not re.fullmatch(r'training_checkpoint_\d{4,}', relative.name):
            raise ValueError(f'Unexpected deletion form: {relative}')
        if any(relative.is_relative_to(Path(p)) or Path(p).is_relative_to(relative) for p in protected):
            raise ValueError(f'Deletion overlaps protected artifact: {relative}')
        path = safe_path(item['path'], root)
        state = tree_state(path)
        if len(state) != item['file_count'] or sum(x['size'] for x in state.values()) != item['logical_bytes']:
            raise ValueError(f'Deletion inventory changed: {path}')
        item['verified_tree'] = state
        item['directory_identity'] = [path.stat().st_dev, path.stat().st_ino]
    if any(p.is_relative_to(q) for i,p in enumerate(delete_paths) for j,q in enumerate(delete_paths) if i != j):
        raise ValueError('Overlapping deletion paths')
    for p in protected:
        if not safe_path(p, root).is_dir():
            raise ValueError(f'Missing protected path: {p}')
    for record in retention['metadata_archive']['records']:
        for key in ('source_path', 'archive_path'):
            if sha(safe_path(record[key], root)) != record['source_sha256']:
                raise ValueError(f'Metadata archive mismatch: {record[key]}')
    if len(retention['metadata_archive']['records']) != 32:
        raise ValueError('Expected 32 archived metadata files')
    by_path = {x['path']: x for x in audit['files']}
    copies, skipped = [], []
    for item in dedup['candidates']:
        target, canonical = Path(item['path']), Path(item['canonical_path'])
        if any(canonical.is_relative_to(p) for p in delete_paths):
            raise ValueError('Canonical is marked for deletion')
        if any(target.is_relative_to(p) for p in delete_paths):
            skipped.append(item['path'])
            continue
        for name in (item['path'], item['canonical_path']):
            p = safe_path(name, root)
            rel = p.relative_to(root / 'runs')
            if len(rel.parts) != 4 or rel.parts[-2] != 'base' or not cow.MODEL_FILE.fullmatch(p.name):
                raise ValueError(f'Unexpected native file path: {name}')
            cow._native_fp32_header(p)
            previous = by_path[name]['identity']
            current = cow._snapshot(p)
            keys = {'device':'dev', 'inode':'ino', 'size':'size', 'mtime_ns':'mtime_ns', 'ctime_ns':'ctime_ns', 'links':'nlink'}
            if any(current[k] != previous[v] for k,v in keys.items()):
                raise ValueError(f'Weight identity changed since approved audit: {name}')
            if by_path[name]['sha256'] != item['verified_sha256']:
                raise ValueError('Candidate/canonical byte group mismatch')
        copies.append(item)
    if len(copies) != 9 or len(skipped) != 16:
        raise ValueError('Expected 9 retained duplicates and 16 deletion-overlap skips')
    return {'deletions': deletions, 'copies': copies, 'skipped_deleted_weights': skipped,
            'protected_paths': protected, 'pinned_inputs': PINNED}


def clone_one(target, reference, expected, *, clone=cow.clonefile, scratch_path=None):
    """Verify both originals; keep a verified independent rollback clone."""
    before, ref_before = cow._snapshot(target), cow._snapshot(reference)
    if before['device'] != ref_before['device']:
        raise ValueError('Cross-volume clone refused')
    if sha(target) != expected or sha(reference) != expected:
        raise ValueError('Exact byte comparison failed before clone')
    if (target.stat().st_uid, target.stat().st_gid) != (reference.stat().st_uid, reference.stat().st_gid):
        raise ValueError('Reference and target ownership differ')
    if scratch_path is None:
        scratch = Path(tempfile.mkdtemp(prefix='.checkpoint-cleanup-', dir=target.parent))
    else:
        scratch = scratch_path
        cow._no_symlinks(scratch)
        scratch.mkdir(mode=0o700)
    replacement, backup = scratch/'replacement', scratch/'original'
    replaced = False
    preserve = False
    try:
        clone(reference, replacement)
        cloned = cow._snapshot(replacement)
        if cloned['inode'] in (before['inode'], ref_before['inode']) or sha(replacement) != expected:
            raise ValueError('Clone identity/bytes mismatch')
        clone(target, backup)
        if sha(backup) != expected:
            raise ValueError('Rollback clone bytes mismatch')
        os.chmod(replacement, stat.S_IMODE(target.stat().st_mode))
        os.utime(replacement, ns=(target.stat().st_atime_ns, before['mtime_ns']))
        with replacement.open('rb') as stream:
            os.fsync(stream.fileno())
        cow._unchanged(target, before)
        cow._unchanged(reference, ref_before)
        os.replace(replacement, target)
        replaced = True
        if sha(target) != expected:
            raise ValueError('Installed clone bytes mismatch')
        after = cow._snapshot(target)
        if after['inode'] in (before['inode'], ref_before['inode']) or after['links'] != 1:
            raise ValueError('Installed clone must be an independent regular file')
        return {'path': str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target),
                'sha256_before': expected, 'sha256_after': expected, 'before': before, 'after': after}
    except BaseException:
        if replaced:
            try:
                os.replace(backup, target)
            except Exception as rollback_error:
                preserve = True
                raise cow.RollbackError(f'Original retained at {backup}; rollback error: {rollback_error}')
        raise
    finally:
        if not preserve:
            shutil.rmtree(scratch)


def delete_one(path, expected_tree, expected_directory, *, pending_name=None):
    """Recheck one exact archived tree, rename locally, then fd-relative removal."""
    cow._no_symlinks(path)
    if tree_state(path) != expected_tree or [path.stat().st_dev, path.stat().st_ino] != expected_directory:
        raise ValueError(f'Changed deletion candidate: {path}')
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError('This Python lacks fd-safe recursive deletion')
    pending = pending_name or '.cleanup-' + path.name + '-' + uuid.uuid4().hex
    if Path(pending).name != pending or not pending.startswith('.cleanup-'):
        raise ValueError('Invalid quarantine name')
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    renamed = False
    try:
        os.rename(path.name, pending, src_dir_fd=fd, dst_dir_fd=fd)
        renamed = True
        staged = path.parent/pending
        if tree_state(staged) != expected_tree or [staged.stat().st_dev, staged.stat().st_ino] != expected_directory:
            raise ValueError('Renamed tree identity changed')
        shutil.rmtree(pending, dir_fd=fd)
        renamed = False
    finally:
        if renamed and (path.parent/pending).exists() and not path.exists():
            os.rename(pending, path.name, src_dir_fd=fd, dst_dir_fd=fd)
        os.close(fd)


def preserved_state(plan):
    deleted = [Path(x['path']) for x in plan['deletions']]
    changed = {x['path'] for x in plan['copies']}
    state = {}
    roots = [ROOT/'runs', ROOT/'checkpoints/minicpm5_source']
    for directory in roots:
        for p in directory.rglob('*'):
            relative = str(p.relative_to(ROOT))
            if p.is_file() and p.name != '.DS_Store' and relative not in changed and not any(Path(relative).is_relative_to(d) for d in deleted):
                cow._no_symlinks(p)
                state[relative] = cow._snapshot(p)
    for name in ['checkpoints/minicpm5_source.download.json','configs/minicpm5_metadata_lock.local.json','configs/minicpm5_sealed_lock.local.json']:
        state[name] = cow._snapshot(ROOT/name)
    return state


def verify_retained(plan):
    results = []
    for receipt_path in sorted((ROOT/'runs').glob('*/training_checkpoint_*/checkpoint_receipt.json')):
        if any(receipt_path.is_relative_to(ROOT/x['path']) for x in plan['deletions']):
            continue
        receipt = json.loads(receipt_path.read_text())
        for name, expected in receipt['files_sha256'].items():
            if sha(safe_path(str((receipt_path.parent/ name).relative_to(ROOT)))) != expected:
                raise ValueError(f'Retained checkpoint mismatch: {receipt_path}: {name}')
        results.append(str(receipt_path.relative_to(ROOT)))
    spec = importlib.util.spec_from_file_location('source_reuse', ROOT/'scripts/download_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.ensure_snapshot(ROOT/'configs/minicpm5_metadata_lock.local.json', ROOT/'checkpoints/minicpm5_source',
                                    sealed_lock=ROOT/'configs/minicpm5_sealed_lock.local.json')
    return {'training_checkpoint_receipts_verified': results, 'original_source_reuse': source}


def execute(plan, output, *, apply=False):
    roots = sorted({safe_path(x['path']).parent for x in plan['deletions']} |
                   {safe_path(x[k]).parents[2] for x in plan['copies'] for k in ('path','canonical_path')})
    before_state = preserved_state(plan)
    report = {'status':'planned_no_payload_changes','training_status':'PAUSED',
              'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'implementation_sha256':sha(Path(__file__)), 'pinned_inputs':PINNED,
              'apply_requested':apply, 'disk_before':disk(), 'clones':[], 'deleted':[],
              'planned_dedup_files':[x['path'] for x in plan['copies']],
              'planned_deletions':[x['path'] for x in plan['deletions']],
              'skipped_duplicates_in_deleted_trees':plan['skipped_deleted_weights'],
              'logical_deleted_bytes':0,'logical_shared_bytes':0,'model_runs':0,'downloads':0}
    write_journal(output, report)
    if not apply:
        return report
    try:
        quiescent(roots)
        # Verify the original before any mutations as well as at the end.
        spec=importlib.util.spec_from_file_location('source_reuse',ROOT/'scripts/download_model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        report['source_before']=module.ensure_snapshot(ROOT/'configs/minicpm5_metadata_lock.local.json',ROOT/'checkpoints/minicpm5_source',sealed_lock=ROOT/'configs/minicpm5_sealed_lock.local.json')
        for item in plan['copies']:
            quiescent(roots)
            scratch=safe_path(item['path']).parents[2]/('.checkpoint-cleanup-'+uuid.uuid4().hex)
            report['pending']={'action':'clone','path':item['path'],'temporary_directory':str(scratch.relative_to(ROOT))}
            write_journal(output,report)
            result=clone_one(safe_path(item['path']),safe_path(item['canonical_path']),item['verified_sha256'],scratch_path=scratch)
            report['clones'].append(result)
            report['logical_shared_bytes']+=item['logical_bytes']
            report['pending']=None
            write_journal(output,report)
        report['disk_after_dedup']=disk()
        if preserved_state(plan)!=before_state:
            raise ValueError('Protected payload metadata changed before deletion')
        report['retained_before_deletion']=verify_retained(plan)
        write_journal(output,report)
        for item in plan['deletions']:
            quiescent(roots)
            pending='.cleanup-'+Path(item['path']).name+'-'+uuid.uuid4().hex
            quarantine=str(Path(item['path']).parent/pending)
            report['pending']={'action':'delete','path':item['path'],'quarantine_path':quarantine}
            write_journal(output,report)
            delete_one(safe_path(item['path']),item['verified_tree'],item['directory_identity'],pending_name=pending)
            report['deleted'].append({'path':item['path'],'logical_bytes':item['logical_bytes'],'files':item['file_count']})
            report['logical_deleted_bytes']+=item['logical_bytes']
            report['pending']=None
            write_journal(output,report)
        os.sync()
        report['disk_after_delete']=disk()
        if preserved_state(plan)!=before_state:
            raise ValueError('Protected payload metadata changed after deletion')
        report['protected_file_metadata_unchanged']=True
        report['preserved_file_count']=len(before_state)
        report['retained_verification']=verify_retained(plan)
        if any(safe_path(x['path']).exists() for x in plan['deletions']):
            raise ValueError('A marked directory remains')
        report['status']='completed_safe_dedup_and_marked_deletion'
    except BaseException as error:
        report['status']='failed_partial_operation_inspect_journal'
        report['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        report['disk_after']=disk()
        report['observed_free_space_change_bytes']=report['disk_after']['free_bytes']-report['disk_before']['free_bytes']
        report['measurement_note']='Observed system-wide filesystem free-space delta; APFS sharing/snapshots and concurrent system activity can affect recovery.'
        report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        write_journal(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.absolute()
    if output.parent!=ROOT/'reports' or output.suffix!='.json' or output.exists() or output.is_symlink():
        parser.error('Use a fresh JSON report directly under reports/')
    plan=make_plan()
    result=execute(plan,output,apply=args.apply)
    print(json.dumps({'status':result['status'],'clones':len(result['clones']),'deleted':len(result['deleted']),
                      'report':str(output.relative_to(ROOT))}))


if __name__=='__main__':
    main()
