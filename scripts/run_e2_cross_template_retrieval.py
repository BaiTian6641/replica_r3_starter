#!/usr/bin/env python3
"""Compare lexical, dense and existing reranker retrieval on cross-template data."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib,json,os,sys,traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def cap(session,sel,max_units=16):
    from replica_lab.memory_system.index import RetrievalSelection
    counts=defaultdict(int)
    for span in session.spans(sel.records): counts[span.record_id]+=1
    rec=[]; sc=[]; used=0
    for r,s in zip(sel.records,sel.scores):
        if used+counts[r.record_id]>max_units: break
        rec.append(r); sc.append(s); used+=counts[r.record_id]
    return RetrievalSelection(tuple(rec),tuple(sc),sel.epoch,sel.ledger_id,sel.scope,sel.query,sel.dense_producer_id)
def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.memory_system import EligibilityScope,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['dataset']['root']
    if sha(data/'records.jsonl')!=protocol['dataset']['records_sha256']: raise ValueError('records changed')
    records=[MemoryRecord(**r) for r in rows(data/'records.jsonl')]; queries=rows(data/'queries.jsonl')
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler)
    ext=model.load_extension(Path(protocol['reranker_extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,extension=ext)
    ledger=RecordLedger(args.output/'retrieval.sqlite')
    for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    template='Return the record that answers the query. Query: {query}'
    results=[]
    for q in queries:
        scope=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
        lexical=cap(session,session.index.retrieve(q['query'],**{k:getattr(scope,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16))
        dense=cap(session,session.dense_candidates(q['query'],scope,limit=16,lexical_weight=0.0))
        ids=encode_prompt(tok,template.replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        req=session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=scope,
            task=TaskGraph(q['case_id'],q['query']),selection=lexical)
        try:
            qf=dict(req.frame.sense.summaries)[session.adapter.ports.injection_layers[0]][:,0]
            valid=req.frame.memory.valid.any(dim=2)
            with torch.no_grad(): scores=session.adapter.reranker(qf,req.frame.memory.slots.mean(dim=2),valid)[0]
            rs={}
            for i,span in enumerate(req.frame.memory.spans):
                if bool(valid[0,i]): rs[span.record_id]=max(rs.get(span.record_id,-1e9),float(scores[i]))
            order=sorted(range(len(lexical.records)),key=lambda i:(-rs.get(lexical.records[i].record_id,-1e9),i))
            reranked=cap(session,type(lexical)(tuple(lexical.records[i] for i in order),
                tuple(rs.get(lexical.records[i].record_id,-1e9) for i in order),lexical.epoch,
                lexical.ledger_id,lexical.scope,lexical.query,lexical.dense_producer_id))
        finally: req.release()
        def rank(sel):
            ids=[r.record_id for r in sel.records]; return ids.index(q['expected_record_id'])+1 if q['expected_record_id'] in ids else None
        results.append({'case_id':q['case_id'],'lexical':rank(lexical),'dense':rank(dense),'rerank':rank(reranked)})
    report['results']=results
    report['metrics']={arm:{'top1':sum(r[arm]==1 for r in results)/len(results),
        'top4':sum(r[arm] is not None and r[arm]<=4 for r in results)/len(results)} for arm in ('lexical','dense','rerank')}
    report['status']='completed'; report['claims']='Cross-template retrieval diagnostic only; no training or default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_cross_template_retrieval_v1.json')
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
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics')}))
if __name__=='__main__': main()
