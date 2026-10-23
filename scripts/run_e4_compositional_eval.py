#!/usr/bin/env python3
"""No-retraining E4 diagnostic: new associations with trained value tokens."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, json, os, sys, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def digest(w): return hashlib.sha256(w.detach().float().cpu().numpy().tobytes()).hexdigest()
def read(model,tokenizer,text):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    from replica_lab.neural import LinearFastMemory
    with torch.no_grad():
        keys,values=model.encode_fast_observation(text,tokenizer,max_tokens=256)
        prim=LinearFastMemory(64,64,learning_rate=0.1,decay=0.0,max_update_norm=1.0)
        state=prim.update_chunk(prim.initial_state(1,device=keys.device,dtype=keys.dtype),keys,values)
        weights=state.weights[:,None].repeat(1,4,1,1).detach()
    return FastMemoryRead(weights,model.fast_producer_key(),digest(weights))
def zero(model):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    w=torch.zeros(1,4,64,64,device='mps')
    return FastMemoryRead(w,model.fast_producer_key(),'zero')
def generate(session,q,template,state):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_system import EligibilityScope
    scope=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'parallel_replica',None)
    ids=encode_prompt(session.tokenizer,template.replace('{query}',q['query']),'mps')
    mask=torch.ones_like(ids,dtype=torch.bool)
    branches=('memory',) if state is None else ('memory','fast')
    req=session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=scope,
        task=TaskGraph(q['case_id'],q['query']),core_impl='none',fast_state=state,
        enabled_branches=branches)
    try:
        out=session.generate(req,max_new_tokens=6); n=int(req.frame.sense.prompt_ids.shape[1])
        ids_out=[int(x) for x in out[0,n:].tolist()]
        ans=session.tokenizer.decode(ids_out,skip_special_tokens=True).strip()
        return {'case_id':q['case_id'],'answer':ans,'expected':q['expected_answer'],
                'correct':ans.casefold()==q['expected_answer'].casefold(),'token_ids':ids_out}
    finally: req.release()
def execute(args,profiler,report):
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['dataset']['root']
    if sha(data/'records.jsonl')!=protocol['dataset']['records_sha256']: raise ValueError('compositional data changed')
    queries=rows(data/'queries.jsonl')
    spec=FastMemorySpec(heads=4,key_size=64,value_size=64)
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,fast_memory=spec)
    receipt=model.load_extension(Path(args.candidate_extension))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,
                  candidate=receipt,parameters=model.parameter_audit())
    ledger=RecordLedger(args.output/'compositional.sqlite')
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    template=('A source page was observed earlier and stored in fast memory. '
              'Answer the query using only that observation. Reply with only the value.\n\nQuery: {query}')
    results={}
    for arm in ('correct','shuffled','zero','disabled'):
        results[arm]=[]
        for i,q in enumerate(queries):
            if arm=='disabled': state=None
            elif arm=='zero': state=zero(model)
            elif arm=='correct': state=read(model,tok,q['observation'])
            else: state=read(model,tok,queries[(i+1)%len(queries)]['observation'])
            results[arm].append(generate(session,q,template,state))
    report['results']=results
    report['metrics']={arm:sum(x['correct'] for x in xs)/len(xs) for arm,xs in results.items()}
    report['controls_equal_disabled']=all(a['answer']==b['answer'] for a,b in zip(results['zero'],results['disabled']))
    report['state_influence']=sum(a['answer']!=b['answer'] for a,b in zip(results['correct'],results['shuffled']))
    report['status']='completed'; report['claims']='No-retraining compositional diagnostic; no capability or deployment claim.'
    report['checks']={'correct_above_controls': report['metrics']['correct']>max(report['metrics']['shuffled'],report['metrics']['zero'],report['metrics']['disabled']),
                      'zero_equals_disabled':report['controls_equal_disabled']}
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e4_compositional_eval_v1.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    p.add_argument('--candidate-extension',type=Path,default=ROOT/'runs/e4_delayed_attempt01_20261023/selected_extension')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),
            'protocol':json.loads(a.protocol.read_text()),'downloads':0,'optimizer_updates':0,'checkpoints_saved':0}
    profiler=RunProfiler('mps',a.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start(); execute(a,profiler,report)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); raise
    finally:
        profile=profiler.stop(); report['profile']=profile
        mem=(profile or {}).get('memory') or {}
        report['costs']={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics'),'checks':report.get('checks')}))
if __name__=='__main__': main()
