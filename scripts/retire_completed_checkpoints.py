#!/usr/bin/env python3
"""Retire only the documented post-D2 payloads; preserve selected models/evidence.

--plan inventories and archives small metadata. --apply verifies protected bytes,
removes exact trees, and prunes only pool weights with no surviving native file.
No tensor/model imports, native computation or network acquisition.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from replica_lab.checkpoint_storage import _hash_lock
import apply_checkpoint_storage_cleanup as safe
from download_model import ensure_snapshot

CANDIDATES = {
    'runs/minicpm5_D2_capacity256_chunk01_20261022/checkpoint': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity256_chunk01_20261022/training_checkpoint_0001': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity256_chunk02_20261022/checkpoint': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity256_chunk02_20261022/training_checkpoint_0002': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity512_chunk01_20261022/checkpoint': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity512_chunk01_20261022/training_checkpoint_0001': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity512_chunk02_20261022/checkpoint': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_capacity512_chunk02_20261022/training_checkpoint_0002': 'Completed audited D2 capacity probe; not a training parent',
    'runs/minicpm5_D2_tiny_chunk01_20261022/checkpoint': 'Update4 superseded by independently verified retained update8',
    'runs/minicpm5_D2_tiny_chunk01_20261022/training_checkpoint_0004': 'Update4 superseded by independently verified retained update8',
    'runs/minicpm5_D1_mask1_B8_lr1e2_chunk01_20261022/checkpoint': 'Intermediate D1 export; final arm retained',
    'runs/minicpm5_D1_mask1_B8_lr1e3_chunk01_20261022/checkpoint': 'Intermediate D1 export; final arm retained',
    'runs/minicpm5_augmented_generation_qualification256_20261022/adapter_export': 'Completed qualification; raw reports retained and source-value native base preserved elsewhere',
    'runs/minicpm5_native_adapter_qualification256_20261022/adapter_export': 'Completed qualification; raw reports retained and source-value native base preserved elsewhere',
    'runs/minicpm5_balanced_mask1_B8_lr1e4_chunk02_20261022/checkpoint': 'D0 update12 continuation probe; selected update8 retained',
    'runs/minicpm5_capacity256_full_muon8bit_chunk01_20261022/checkpoint': 'Completed D0 capacity probe; raw reports retained',
    'runs/minicpm5_capacity256_full_muon8bit_chunk02_20261022/checkpoint': 'Completed D0 capacity probe; raw reports retained',
    'runs/minicpm5_capacity512_full_muon8bit_chunk01_20261022/checkpoint': 'Completed D0 capacity probe; raw reports retained',
    'runs/minicpm5_capacity512_full_muon8bit_chunk02_20261022/checkpoint': 'Completed D0 capacity probe; raw reports retained',
}
PROTECTED = (
    'checkpoints/minicpm5_source',
    'runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/checkpoint',
    'runs/minicpm5_balanced_mask1_B8_lr1e4_chunk01_20261022/training_checkpoint_0008',
    'runs/minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022/checkpoint',
    'runs/minicpm5_D1_mask1_B8_lr1e2_chunk02_20261022/training_checkpoint_0028',
    'runs/minicpm5_D2_tiny_chunk02_20261022/checkpoint',
    'runs/minicpm5_D2_tiny_chunk02_20261022/training_checkpoint_0008',
    'runs/minicpm5_mps_muon8bit_D0_preflight_20261022_01/checkpoint',
    'runs/minicpm5_D1_mask1_B8_lr1e3_chunk02_20261022/checkpoint',
)
IMPLEMENTATIONS = ('scripts/retire_completed_checkpoints.py',
    'scripts/apply_checkpoint_storage_cleanup.py', 'scripts/compact_identical_checkpoints.py',
    'src/replica_lab/checkpoint_storage.py', 'scripts/download_model.py')
POOL = ROOT/'runs/.checkpoint_storage'


def validate_scope(deletions, protected):
    if set(deletions) != set(CANDIDATES) or set(protected) != set(PROTECTED):
        raise ValueError('Only the fixed reviewed post-D2 scope is allowed')
    for name in deletions:
        path = Path(name)
        if len(path.parts) != 3 or path.parts[0] != 'runs':
            raise ValueError('Deletion must be an explicit payload directory')
        if any(path.is_relative_to(Path(p)) or Path(p).is_relative_to(path) for p in protected):
            raise ValueError('Deletion overlaps protected data')


def inventory():
    validate_scope(CANDIDATES, PROTECTED)
    for name in PROTECTED:
        if not safe.safe_path(name).is_dir():raise ValueError('Missing protected '+name)
    old = json.loads((ROOT/'reports/base_checkpoint_duplication_audit_20261022.json').read_text())
    known = {r['path']:r['sha256'] for r in old['files']}
    entries=[];archives=[]
    archive=ROOT/'reports/checkpoint_retirement_metadata_after_D2_20261022'
    if archive.exists():raise ValueError('Use a fresh metadata archive')
    for name,reason in CANDIDATES.items():
        path=safe.safe_path(name);tree=safe.tree_state(path)
        report=path.parent/('qualification_report.json' if path.name=='adapter_export' else 'run_report.json')
        raw=json.loads(report.read_text())
        if not (raw.get('status','').startswith('passed_') or raw.get('status')=='completed_local_training_run'):
            raise ValueError('Candidate has no completed run report')
        for file in sorted(path.rglob('*')):
            if file.is_file() and file.suffix in {'.json','.jinja'} and file.stat().st_size < 512*1024:
                target=archive/file.relative_to(ROOT/'runs');target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target);digest=safe.sha(file)
                if safe.sha(target)!=digest:raise ValueError('Metadata archive bytes differ')
                archives.append({'source':str(file.relative_to(ROOT)),'archive':str(target.relative_to(ROOT)),'sha256':digest})
        entries.append({'path':name,'reason':reason,'tree':tree,'directory_identity':[path.stat().st_dev,path.stat().st_ino],
                        'logical_bytes':sum(x['size'] for x in tree.values()),'run_report':str(report.relative_to(ROOT)),
                        'run_report_sha256':safe.sha(report)})
    native=sorted((ROOT/'runs').glob('*/*/base/model*.safetensors'))
    live=[]
    for path in native:
        if any(path.is_relative_to(ROOT/n) for n in CANDIDATES):continue
        name=str(path.relative_to(ROOT));parent=path.parent.parent
        if str(parent.relative_to(ROOT)) not in PROTECTED:raise ValueError('Unreviewed surviving native payload '+name)
        receipt=parent/'checkpoint_receipt.json';manifest=json.loads((parent/'adapter_manifest.json').read_text())
        expected=json.loads(receipt.read_text())['files_sha256'].get('base/'+path.name) if receipt.exists() else known.get(name)
        if expected is None:
            expected=next((x['sha256'] for x in manifest.get('checkpoint_storage',{}).get('files',[]) if x['file']==path.name),None)
        if expected is None:raise ValueError('No independent native byte claim '+name)
        live.append({'path':name,'sha256':expected,'identity':safe.cow._snapshot(path)})
    if len(live)!=8:raise ValueError('Expected exactly8 protected native weight files')
    live_hashes={r['sha256'] for r in live};pool=[]
    for path in sorted((POOL/'sha256').rglob('*')):
        if path.is_dir():continue
        if not re.fullmatch(r'[0-9a-f]{64}\.safetensors',path.name) or path.parent.name!=path.stem[:2]:
            raise ValueError('Unexpected pool object')
        pool.append({'path':str(path.relative_to(ROOT)),'sha256':path.stem,'identity':safe.cow._snapshot(path),
                     'action':'keep_live' if path.stem in live_hashes else 'retire_orphan'})
    if len(pool)!=14 or len(live_hashes)!=4 or sum(x['action']=='retire_orphan' for x in pool)!=10:
        raise ValueError('Pool liveness differs from reviewed14/4/10 scope')
    all_state=safe.preserved_state({'deletions':entries,'copies':[]})
    for row in pool:
        if row['action']=='retire_orphan':all_state.pop(row['path'])
    return {'status':'reviewed_plan_no_payload_deletion','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'deletions':entries,'protected':list(PROTECTED),'surviving_native_weights':live,'pool':pool,
            'metadata_archive':archives,'preserved_file_state':all_state,'disk_before':safe.disk(),
            'implementation_sha256':{n:safe.sha(ROOT/n) for n in IMPLEMENTATIONS},'model_runs':0,'downloads':0}


def retire_pool_file(path, expected_identity, live_hashes, *, pool=POOL):
    """Remove one fully hashed orphan; never unlink a live hash or prefix bucket."""
    safe.cow._no_symlinks(path)
    digest=path.stem
    if (path.parent.parent!=pool/'sha256' or path.parent.name!=digest[:2]
        or not re.fullmatch(r'[0-9a-f]{64}\.safetensors',path.name) or digest in live_hashes):
        raise ValueError('Pool path is not an exact orphan')
    with _hash_lock(pool,digest):
        safe.cow._unchanged(path,expected_identity)
        if safe.sha(path)!=digest:raise ValueError('Pool bytes differ from content address')
        safe.cow._unchanged(path,expected_identity)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            info=os.stat(path.name,dir_fd=fd,follow_symlinks=False)
            if (info.st_dev,info.st_ino)!=(expected_identity['device'],expected_identity['inode']):
                raise ValueError('Pool entry replaced before unlink')
            os.unlink(path.name,dir_fd=fd)
        finally:os.close(fd)


def check_preserved(plan):
    current=safe.preserved_state({'deletions':plan['deletions'],'copies':[]})
    for row in plan['pool']:
        if row['action']=='retire_orphan':current.pop(row['path'],None)
    if current!=plan['preserved_file_state']:
        raise ValueError('Protected/source/raw file identities changed')


def apply(plan_path, output):
    plan=json.loads(plan_path.read_text());validate_scope([x['path'] for x in plan['deletions']],plan['protected'])
    if any(safe.sha(ROOT/n)!=h for n,h in plan['implementation_sha256'].items()):raise ValueError('Cleanup implementation changed')
    if output.exists():raise ValueError('Fresh journal required; inspect partial progress before retry')
    roots=sorted({ROOT/Path(n).parent for n in CANDIDATES});safe.quiescent(roots)
    for r in plan['metadata_archive']:
        if safe.sha(safe.safe_path(r['source']))!=r['sha256'] or safe.sha(safe.safe_path(r['archive']))!=r['sha256']:
            raise ValueError('Archived metadata changed')
    live_hashes=set();claims={}
    for row in plan['surviving_native_weights']:claims[row['path']]=row['sha256']
    for row in plan['pool']:
        if row['action']=='keep_live':claims[row['path']]=row['sha256']
    receipts=[]
    for p in sorted((ROOT/'runs').glob('*/training_checkpoint_*/checkpoint_receipt.json')):
        if any(p.is_relative_to(ROOT/n) for n in CANDIDATES):continue
        receipt=json.loads(p.read_text());receipts.append(str(p.relative_to(ROOT)))
        for n,h in receipt['files_sha256'].items():claims[str((p.parent/n).relative_to(ROOT))]=h
    journal={'status':'verifying_before_retirement','started_at_utc':datetime.now(timezone.utc).isoformat(),
             'plan':str(plan_path.relative_to(ROOT)),'plan_sha256':safe.sha(plan_path),'disk_before':safe.disk(),
             'deleted_payloads':[],'deleted_pool_files':[],'model_runs':0,'downloads':0,'native_jobs':'IDLE'}
    safe.write_journal(output,journal)
    def verify(item):
        n,h=item
        if safe.sha(safe.safe_path(n))!=h:raise ValueError('Protected byte claim differs '+n)
        return n
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            for n in workers.map(verify,claims.items()):
                print(json.dumps({'verified':n}),flush=True)
        source=ensure_snapshot(ROOT/'configs/minicpm5_metadata_lock.local.json',ROOT/'checkpoints/minicpm5_source',
                               sealed_lock=ROOT/'configs/minicpm5_sealed_lock.local.json')
        for r in plan['surviving_native_weights']:
            safe.cow._unchanged(ROOT/r['path'],r['identity']);live_hashes.add(r['sha256'])
        check_preserved(plan)
        journal.update(status='retiring_reviewed_payloads',protected_byte_claims_verified=len(claims),
                       protected_training_receipts_verified=receipts,original_source=source)
        safe.write_journal(output,journal)
        for row in plan['deletions']:
            safe.quiescent(roots);path=safe.safe_path(row['path']);pending='.cleanup-post-D2-'+uuid.uuid4().hex
            journal['pending']={'path':row['path'],'quarantine':str(path.parent/pending)};safe.write_journal(output,journal)
            safe.delete_one(path,row['tree'],row['directory_identity'],pending_name=pending)
            journal['deleted_payloads'].append({'path':row['path'],'logical_bytes':row['logical_bytes']});journal['pending']=None
            safe.write_journal(output,journal)
        # Exact surviving file set/identities must still bind the original liveness inventory.
        remaining={str(p.relative_to(ROOT)) for p in (ROOT/'runs').glob('*/*/base/model*.safetensors')}
        if remaining!={r['path'] for r in plan['surviving_native_weights']}:raise ValueError('Surviving native inventory changed')
        for row in plan['surviving_native_weights']:
            safe.cow._unchanged(ROOT/row['path'],row['identity'])
        check_preserved(plan)
        journal['disk_after_payloads']=safe.disk();safe.write_journal(output,journal)
        for row in plan['pool']:
            if row['action']!='retire_orphan':continue
            safe.quiescent(roots);journal['pending']={'orphan_pool_file':row['path']};safe.write_journal(output,journal)
            retire_pool_file(safe.safe_path(row['path']),row['identity'],live_hashes)
            journal['deleted_pool_files'].append({'path':row['path'],'sha256':row['sha256'],'logical_bytes':row['identity']['size']})
            journal['pending']=None;safe.write_journal(output,journal)
        check_preserved(plan)
        journal.update(status='completed_reviewed_retirement_and_orphan_pool_gc',preserved_file_identities_unchanged=True,
                       source_and_acquisition_receipt_preserved=True,archive_files=len(plan['metadata_archive']))
    except BaseException as error:
        journal.update(status='partial_or_failed_inspect_journal',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        journal['disk_after']=safe.disk();journal['observed_freed_bytes']=journal['disk_after']['free_bytes']-journal['disk_before']['free_bytes']
        journal['completed_at_utc']=datetime.now(timezone.utc).isoformat();safe.write_journal(output,journal)
    print(json.dumps({'status':journal['status'],'observed_freed_gib':journal['observed_freed_bytes']/1024**3}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--apply',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();plan=a.plan if a.plan.is_absolute() else ROOT/a.plan
    for path in [plan,*([a.output if a.output.is_absolute() else ROOT/a.output] if a.output else [])]:
        safe.cow._no_symlinks(path)
        if path.parent!=ROOT/'reports' or path.suffix!='.json':p.error('Use explicit JSON plan/journal files directly in reports/')
    if a.output and (a.output if a.output.is_absolute() else ROOT/a.output)==plan:p.error('Plan and journal must differ')
    lock=ROOT/'runs/.D2_mps_worker.lock';safe.cow._no_symlinks(lock)
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.apply:
            if not a.output:p.error('--apply requires --output')
            output=a.output if a.output.is_absolute() else ROOT/a.output;apply(plan,output)
        else:
            if plan.exists():p.error('Use a fresh plan path')
            result=inventory();safe.write_journal(plan,result)
            print(json.dumps({'status':result['status'],'payload_dirs':len(result['deletions']),'metadata_archives':len(result['metadata_archive']),'orphan_pool_files':sum(x['action']=='retire_orphan' for x in result['pool'])}))

if __name__=='__main__':main()
