#!/usr/bin/env python3
"""Load the saved context head and evaluate the integrated memory answer path."""
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
    from safetensors.torch import load_file
    from replica_lab.architecture.contracts import CopySpec,TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.memory_system import EligibilityScope,MemoryAnswerService,MemoryRecord,RecordLedger
    from replica_lab.memory_system.context import CandidateContextHead
    from replica_lab.memory_system.session import MemorySession
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['eval_dataset']['root']
    if sha(data/'records.jsonl')!=protocol['eval_dataset']['records_sha256']: raise ValueError('records changed')
    ctx_manifest=json.loads((ROOT/protocol['context_head_manifest']).read_text())
    if sha(ROOT/protocol['context_head'])!=ctx_manifest['weights_sha256']: raise ValueError('context head bytes changed')
    pointer_manifest=json.loads((Path(args.model_path).parent.parent/ROOT/protocol['pointer_extension']/'manifest.json').read_text()) if False else json.loads((ROOT/protocol['pointer_extension']/'manifest.json').read_text())
    if pointer_manifest['weights_sha256']!=ctx_manifest['parent_pointer_weights_sha256']: raise ValueError('context head parent pointer differs')
    queries=rows(data/'queries.jsonl'); records=[MemoryRecord(**r) for r in rows(data/'records.jsonl')]
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
        copy=CopySpec(max_tokens=512,mode='pointer',max_span=protocol['max_span']))
    receipt=model.load_extension(Path(protocol['pointer_extension']))
    head=CandidateContextHead().to('mps')
    head.load_state_dict(load_file(str(ROOT/protocol['context_head'])))
    head.eval()
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,pointer=receipt,
                  context_head=ctx_manifest)
    ledger=RecordLedger(args.output/'context_service.sqlite')
    for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    def prep(q):
        sc=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
        ids=encode_prompt(tok,protocol['prompt'].replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=session.index.retrieve(q['query'],**{k:getattr(sc,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16)
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=sc,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',enabled_branches=('memory','copy'))
    parent=[]; candidate=[]
    for q in queries:
        req=prep(q)
        try:
            p_service=MemoryAnswerService(session,pointer_enabled=True,candidate_unit='word',record_scope='top')
            c_service=MemoryAnswerService(session,pointer_enabled=True,candidate_unit='word',record_scope='top')
            c_service.context_head=head
            pa=p_service.answer(req); ca=c_service.answer(req)
            parent.append({'case_id':q['case_id'],'text':pa.text,'expected':q['expected_answer'],
                'correct':pa.text.casefold()==q['expected_answer'].casefold(),'record_ids':pa.record_ids})
            candidate.append({'case_id':q['case_id'],'text':ca.text,'expected':q['expected_answer'],
                'correct':ca.text.casefold()==q['expected_answer'].casefold(),'record_ids':ca.record_ids,
                'source_hashes':ca.source_hashes,'span_hashes':ca.span_hashes})
        finally:req.release()
    report['parent']=parent; report['candidate']=candidate
    report['metrics']={'parent':sum(x['correct'] for x in parent)/len(parent),
                       'candidate':sum(x['correct'] for x in candidate)/len(candidate)}
    report['checks']={'parent_pointer_manifest':True,'context_head_manifest':True,
        'candidate_method':all(x['record_ids'] for x in candidate)}
    report['decision']='accept_opt_in' if report['metrics']['candidate']>report['metrics']['parent'] else 'reject'
    report['deployment']='not_published'; report['status']='completed'
    report['claims']='Integrated context-head consumer check; no training, publication or default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_pointer_context_service_v1.json')
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
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics'),'decision':report.get('decision')}))
if __name__=='__main__': main()
