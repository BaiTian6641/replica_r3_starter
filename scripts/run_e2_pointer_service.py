#!/usr/bin/env python3
"""Exercise MemoryAnswerService on the published pointer candidate (no training)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib,json,os,sys,traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import CopySpec,TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.memory_system import EligibilityScope,MemoryAnswerService,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['dataset']['root']
    if sha(data/'records.jsonl')!=protocol['dataset']['records_sha256']: raise ValueError('records changed')
    records=[MemoryRecord(**r) for r in rows(data/'records.jsonl')]
    queries=[q for q in rows(data/'queries.jsonl') if q['split'] in ('dev','heldout')]
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
        copy=CopySpec(max_tokens=512,mode='pointer',max_span=protocol['max_span']))
    receipt=model.load_extension(Path(protocol['extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,extension=receipt)
    ledger=RecordLedger(args.output/'service.sqlite')
    for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    def prep(q):
        scope=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
        ids=encode_prompt(tok,protocol['prompt'].replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=session.index.retrieve(q['query'],**{k:getattr(scope,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16)
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=scope,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',
            enabled_branches=('memory','copy'))
    units = protocol.get('candidate_units', ['token'])
    scopes = protocol.get('record_scopes', ['all'])
    results_by_arm = {}
    exact_results = []
    first_arm = True
    for unit in units:
        for scope in scopes:
            arm = f'{unit}:{scope}'
            pointer_results = []
            for q in queries:
                req = prep(q)
                try:
                    pointer = MemoryAnswerService(session, pointer_enabled=True,
                                                  candidate_unit=unit,
                                                  record_scope=scope).answer(req)
                    exact = MemoryAnswerService(session, pointer_enabled=False).answer(req)
                    pointer_results.append({'case_id':q['case_id'],'split':q['split'],'text':pointer.text,
                        'expected':q['expected_answer'],'correct':pointer.text.casefold()==q['expected_answer'].casefold(),
                        'method':pointer.method,'record_ids':pointer.record_ids,'span_hashes':pointer.span_hashes,
                        'source_ids':pointer.source_ids,'source_hashes':pointer.source_hashes,'producer':pointer.producer,
                        'unit':unit,'record_scope':scope})
                    if first_arm:
                        exact_results.append({'case_id':q['case_id'],'split':q['split'],'method':exact.method,
                            'record_ids':exact.record_ids,'span_hashes':exact.span_hashes,
                            'source_hashes':exact.source_hashes,
                            'text_sha256':hashlib.sha256(exact.text.encode()).hexdigest()})
                finally: req.release()
            results_by_arm[arm] = pointer_results
            first_arm = False
    report['arms'] = results_by_arm; report['exact_source'] = exact_results
    report['metrics'] = {arm: {split: {'count': sum(r['split']==split for r in values),
        'correct': sum(r['correct'] for r in values if r['split']==split),
        'accuracy': sum(r['correct'] for r in values if r['split']==split)/max(1,sum(r['split']==split for r in values))}
        for split in ('dev','heldout')} for arm, values in results_by_arm.items()}
    report['checks']={'all_pointer_method':all(r['method']=='pointer_copy' for values in results_by_arm.values() for r in values),
        'all_pointer_provenance':all(r['record_ids'] and r['span_hashes'] and r['source_hashes'] for values in results_by_arm.values() for r in values),
        'all_exact_fallback':all(r['method']=='exact_source' for r in exact_results)}
    report['status']='completed'; report['claims']='Bounded consumer check of published pointer and exact-source fallback; no training or default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_pointer_service_v1.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),
            'protocol':json.loads(a.protocol.read_text()),'downloads':0,'optimizer_updates':0,'checkpoints_saved':0}
    profiler=RunProfiler('mps',a.output)
    try:
        with exclusive_mps_worker(ROOT): profiler.start(); execute(a,profiler,report)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); raise
    finally:
        profile=profiler.stop(); report['profile']=profile; mem=(profile or {}).get('memory') or {}
        report['costs']={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics'),'checks':report.get('checks')}))
if __name__=='__main__': main()
