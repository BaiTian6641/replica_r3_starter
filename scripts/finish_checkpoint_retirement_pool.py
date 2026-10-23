#!/usr/bin/env python3
"""Finish only the nine orphan-cache removals after the recorded ctime guard stop.

Keeps the original partial journal; never repeats retired payload deletion.
All full-byte and unlink checks use the unchanged reviewed cleanup helpers.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
import retire_completed_checkpoints as cleanup
safe=cleanup.safe
PLAN=ROOT/'reports/checkpoint_retirement_plan_after_D2_20261022.json'
PARTIAL=ROOT/'reports/checkpoint_retirement_initial_after_D2_20261022.json'
OUTPUT=ROOT/'reports/checkpoint_retirement_after_D2_20261022.json'


def recovery_scope(plan, initial, *, root=ROOT):
    cleanup.validate_scope([r['path'] for r in plan['deletions']],plan['protected'])
    old=initial['deleted_payloads'];done={r['path'] for r in initial['deleted_pool_files']}
    if (initial['status']!='partial_or_failed_inspect_journal' or len(old)!=19
        or {r['path'] for r in old}!=set(cleanup.CANDIDATES) or len(done)!=1
        or len(initial['deleted_pool_files'])!=1):
        raise ValueError('Expected the exact recorded19+1 partial result')
    if initial.get('protected_byte_claims_verified')!=35 or not initial.get('original_source'):
        raise ValueError('Missing completed protected-byte verification')
    orphans={r['path'] for r in plan['pool'] if r['action']=='retire_orphan'}
    if not done.issubset(orphans):raise ValueError('Prior deletion outside reviewed orphans')
    if any((root/r['path']).exists() for r in old) or any((root/n).exists() for n in done):
        raise ValueError('A previously removed payload/cache object reappeared')
    entries=list((root/'runs/.checkpoint_storage/sha256').rglob('*'))
    if any(p.is_symlink() for p in entries):raise ValueError('Pool inventory contains a symlink')
    actual={str(p.relative_to(root)) for p in entries if p.is_file()}
    expected={r['path'] for r in plan['pool']}-done
    if actual!=expected:raise ValueError('Pool inventory changed outside partial retirement')
    return [r for r in plan['pool'] if r['path'] in orphans-done]


def main():
    if OUTPUT.exists():raise ValueError('Preserve partial journal under its pinned archival name first; fresh completion journal required')
    plan=json.loads(PLAN.read_text());initial=json.loads(PARTIAL.read_text())
    if safe.sha(PLAN)!=initial['plan_sha256']:raise ValueError('Original plan changed')
    if any(safe.sha(ROOT/n)!=h for n,h in plan['implementation_sha256'].items()):raise ValueError('Original cleanup implementation changed')
    roots=sorted({ROOT/Path(n).parent for n in cleanup.CANDIDATES})
    with (ROOT/'runs/.D2_mps_worker.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);safe.quiescent(roots)
        remaining=recovery_scope(plan,initial);cleanup.check_preserved(plan)
        live={r['sha256'] for r in plan['surviving_native_weights']}
        for row in plan['surviving_native_weights']:safe.cow._unchanged(ROOT/row['path'],row['identity'])
        journal={**initial,'status':'verifying_remaining_orphan_pool','partial_journal':str(PARTIAL.relative_to(ROOT)),
                 'partial_journal_sha256':safe.sha(PARTIAL),'recovery_started_at_utc':datetime.now(timezone.utc).isoformat(),
                 'recovery_implementation_sha256':safe.sha(Path(__file__)),'resumed_deleted_pool_files':[],
                 'metadata_refreshes':[],'pending':None}
        journal.pop('error',None);journal['disk_at_recovery_start']=safe.disk();safe.write_journal(OUTPUT,journal)
        try:
            claims={r['path']:r['sha256'] for r in plan['surviving_native_weights']}
            for row in plan['pool']:
                if row['action']=='keep_live':claims[row['path']]=row['sha256']
            for name in initial['protected_training_receipts_verified']:
                receipt_path=ROOT/name;receipt=json.loads(receipt_path.read_text())
                for n,h in receipt['files_sha256'].items():claims[str((receipt_path.parent/n).relative_to(ROOT))]=h
            def verify(item):
                n,h=item
                if safe.sha(safe.safe_path(n))!=h:raise ValueError('Protected byte claim differs '+n)
            with ThreadPoolExecutor(max_workers=2) as workers:list(workers.map(verify,claims.items()))
            cleanup.ensure_snapshot(ROOT/'configs/minicpm5_metadata_lock.local.json',ROOT/'checkpoints/minicpm5_source',
                                    sealed_lock=ROOT/'configs/minicpm5_sealed_lock.local.json')
            cleanup.check_preserved(plan)
            for row in remaining:
                safe.quiescent(roots);cleanup.check_preserved(plan)
                path=safe.safe_path(row['path']);fresh=safe.cow._snapshot(path)
                delta={k:[row['identity'][k],fresh[k]] for k in fresh if fresh[k]!=row['identity'][k]}
                if set(delta)-{'ctime_ns'}:raise ValueError('Orphan identity changed beyond ctime')
                journal['pending']={'orphan_pool_file':row['path']};journal['metadata_refreshes'].append({'path':row['path'],'delta':delta})
                safe.write_journal(OUTPUT,journal)
                cleanup.retire_pool_file(path,fresh,live)
                record={'path':row['path'],'sha256':row['sha256'],'logical_bytes':fresh['size']}
                journal['deleted_pool_files'].append(record);journal['resumed_deleted_pool_files'].append(record)
                journal['pending']=None;safe.write_journal(OUTPUT,journal)
                print(json.dumps({'retired_orphan':row['path']}),flush=True)
            cleanup.check_preserved(plan)
            final_pool={p.stem for p in (cleanup.POOL/'sha256').rglob('*.safetensors')}
            if final_pool!=live:raise ValueError('Final pool differs from four live groups')
            journal.update(status='completed_reviewed_retirement_and_orphan_pool_gc',preserved_file_identities_unchanged=True,
                           source_and_acquisition_receipt_preserved=True,archive_files=len(plan['metadata_archive']),
                           recovery_protected_byte_claims_verified=len(claims),final_live_pool_versions=len(final_pool),
                           recovery_note='Initial guard stopped on ctime-only metadata drift in an unused canonical. Fresh full hashes match content addresses; original inode/size/mtime/link/allocation identities are preserved. Cause of ctime drift is unassigned. All original guards remain unchanged.')
        except BaseException as error:
            journal.update(status='partial_recovery_inspect_journal',error={'type':type(error).__name__,'message':str(error)})
            raise
        finally:
            journal['disk_after']=safe.disk();journal['observed_freed_bytes']=journal['disk_after']['free_bytes']-journal['disk_before']['free_bytes']
            journal['completed_at_utc']=datetime.now(timezone.utc).isoformat();safe.write_journal(OUTPUT,journal)
        print(json.dumps({'status':journal['status'],'observed_freed_gib':journal['observed_freed_bytes']/1024**3}),flush=True)

if __name__=='__main__':main()
