#!/usr/bin/env python3
"""Seed the new APFS pool from retained audited canonicals; no model execution."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replica_lab.checkpoint_storage import deduplicate_native_weights, _state
from apply_checkpoint_storage_cleanup import write_journal, quiescent, disk


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Publish verified COW pool entries after cleanup')
    args=parser.parse_args()
    if not args.apply:
        print(json.dumps({'status':'plan_only_no_payload_changes','canonical_groups':10,'apply_requested':False}))
        return
    audit_path=ROOT/'reports/base_checkpoint_duplication_audit_20261022.json'
    if hashlib.sha256(audit_path.read_bytes()).hexdigest()!='147b9bb999b18692049f2ae4dd6a2be5f501631bf4386a591f8895b380cb9043':
        raise ValueError('Approved canonical inventory changed')
    cleanup=json.loads((ROOT/'reports/checkpoint_cleanup_apply_20261022.json').read_text())
    if cleanup['status']!='completed_safe_dedup_and_marked_deletion' or len(cleanup['deleted'])!=16 or len(cleanup['clones'])!=9:
        raise ValueError('Complete the verified cleanup before seeding the pool')
    pool=ROOT/'runs/.checkpoint_storage'
    output=ROOT/'reports/checkpoint_weight_pool_seed_20261022.json'
    if pool.exists() or pool.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError('Use only once with a fresh pool and report; inspect prior progress before retry')
    audit=json.loads(audit_path.read_text())
    canonicals=[ROOT/x['canonical_path'] for x in audit['groups']]
    roots=sorted({p.parents[2] for p in canonicals})
    quiescent(roots)
    before={str(p.relative_to(ROOT)):_state(p) for p in canonicals}
    report={'status':'seeding_existing_verified_canonical_weights','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256':hashlib.sha256((ROOT/'src/replica_lab/checkpoint_storage.py').read_bytes()).hexdigest(),
            'disk_before':disk(),'entries':[],'model_runs':0,'downloads':0,'training_status':'PAUSED'}
    write_journal(output,report)
    try:
        for group,path in zip(audit['groups'],canonicals):
            quiescent(roots)
            report['pending']=str(path.relative_to(ROOT));write_journal(output,report)
            result=deduplicate_native_weights(path.parent)
            if len(result['files'])!=1:
                raise ValueError('Expected one audited native file per canonical')
            item=result['files'][0]
            if item.get('sha256')!=group['sha256'] or item['action']!='published_pool_clone_of_saved_file' or not item['independent_inode']:
                raise ValueError('Pool entry did not publish as an exact independent COW clone')
            report['entries'].append({'canonical_path':str(path.relative_to(ROOT)),**item})
            report['pending']=None;write_journal(output,report)
        if {str(p.relative_to(ROOT)):_state(p) for p in canonicals}!=before:
            raise ValueError('A retained canonical changed during pool seeding')
        report['canonical_file_identities_unchanged']=True
        report['status']='completed_existing_weights_seeded_as_cow_pool'
    except BaseException as error:
        report['status']='failed_partial_pool_seeding_inspect_journal'
        report['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        report['disk_after']=disk()
        report['observed_free_space_change_bytes']=report['disk_after']['free_bytes']-report['disk_before']['free_bytes']
        report['pool_logical_bytes']=sum(p.stat().st_size for p in pool.rglob('*.safetensors')) if pool.exists() else 0
        report['measurement_note']='Pool entries share existing extents via clonefile; logical file size is not extra physical allocation. Free-space deltas are system-wide observations.'
        report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        write_journal(output,report)
    print(json.dumps({'status':report['status'],'pool_entries':len(report['entries'])}))


if __name__=='__main__':
    main()
