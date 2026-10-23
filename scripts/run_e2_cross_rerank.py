#!/usr/bin/env python3
"""Train/evaluate a fresh reranker on varied-template retrieval (MPS)."""
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
def cap(session,selection,max_units=16):
    from replica_lab.memory_system.index import RetrievalSelection
    counts=defaultdict(int)
    for span in session.spans(selection.records): counts[span.record_id]+=1
    rec=[]; sc=[]; used=0
    for r,s in zip(selection.records,selection.scores):
        if used+counts[r.record_id]>max_units: break
        rec.append(r); sc.append(s); used+=counts[r.record_id]
    return RetrievalSelection(tuple(rec),tuple(sc),selection.epoch,selection.ledger_id,selection.scope,selection.query,selection.dense_producer_id)
def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import EligibilityScope,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['dataset']['root']
    if sha(data/'records.jsonl')!=protocol['dataset']['records_sha256']: raise ValueError('records changed')
    records=[MemoryRecord(**r) for r in rows(data/'records.jsonl')]; queries=rows(data/'queries.jsonl')
    train=[q for q in queries if q['split']=='train']; dev=[q for q in queries if q['split']=='dev']; held=[q for q in queries if q['split']=='heldout']
    template='Return the record that answers the query. Query: {query}'
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler)
    parent=model.load_extension(Path(args.parent_extension))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,parent=parent)
    ledger=RecordLedger(args.output/'rerank.sqlite')
    for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    def prep(q,training=False):
        sc=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
        ids=encode_prompt(tok,template.replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=cap(session,session.index.retrieve(q['query'],**{k:getattr(sc,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16))
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=sc,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none')
    def rank(q):
        req=prep(q)
        try:
            qf=dict(req.frame.sense.summaries)[session.adapter.ports.injection_layers[0]][:,0]
            valid=req.frame.memory.valid.any(dim=2)
            with torch.no_grad(): scores=session.adapter.reranker(qf,req.frame.memory.slots.mean(dim=2),valid)[0]
            rs={}
            for i,span in enumerate(req.frame.memory.spans):
                if bool(valid[0,i]): rs[span.record_id]=max(rs.get(span.record_id,-1e9),float(scores[i]))
            lex=[r.record_id for r in req.selection.records]
            order=sorted(lex,key=lambda rid:(-rs.get(rid,-1e9),lex.index(rid)))
            return {'case_id':q['case_id'],'split':q['split'],'lexical_rank':lex.index(q['expected_record_id'])+1 if q['expected_record_id'] in lex else None,
                    'rerank_rank':order.index(q['expected_record_id'])+1 if q['expected_record_id'] in order else None}
        finally:req.release()
    before=[rank(q) for q in dev+held]
    previous={n:p.requires_grad for n,p in model.new_named_parameters()}
    for n,p in model.new_named_parameters():p.requires_grad_(n.startswith(('reranker.','sense_ports.13.')))
    opt=build_optimizer(model,'muon8bit',lr_base=0.0,lr_new=protocol['training']['lr_new'],lr_muon_base=0.0,
        lr_muon_new=protocol['training']['lr_muon_new'],weight_decay=0.0)
    if opt is None: raise RuntimeError('no reranker parameters')
    model.train(); losses=[]
    for update in range(1,int(protocol['training']['updates'])+1):
        opt.zero_grad(set_to_none=True); total=0.0
        for q in train:
            req=prep(q)
            try:
                with session.training_request(prompt_ids=req.frame.sense.prompt_ids,prompt_mask=req.frame.sense.prompt_mask,
                        query=q['query'],scope=req.selection.scope,task=TaskGraph(q['case_id'],q['query']),
                        selection=req.selection,core_impl='none',train_sense=True) as active:
                    loss=session.ranking_loss(active,q['expected_record_id'])
                    total+=float(loss.detach()); session.backward(active,loss)
            finally:req.release()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        opt.step(); opt.zero_grad(set_to_none=True); losses.append(total); profiler.mark('cross_rerank_update',update=update,loss=total)
    del opt
    for n,p in model.new_named_parameters():p.requires_grad_(previous[n])
    model.eval(); ext=args.output/'selected_extension'; receipt=model.save_extension(ext)
    replacement=MiniCPM5Adapter(model.base,model.manifest).to('mps').eval(); replacement.seal_original_identity(); replacement.load_extension(ext)
    session=MemorySession(replacement,tok,ledger,max_candidate_units=64)
    after=[rank(q) for q in dev+held]
    def top1(xs,split,key):
        y=[x for x in xs if x['split']==split]; return sum(x[key]==1 for x in y)/len(y)
    report['before']=before; report['after']=after; report['training']={'losses':losses}
    report['extension']={'format':receipt['format'],'weights_sha256':receipt['weights_sha256'],'manifest_digest':receipt['manifest_digest'],'base_saved':False}
    report['metrics']={'lexical':{'dev_top1':top1(before,'dev','lexical_rank'),'heldout_top1':top1(before,'heldout','lexical_rank'),'heldout_top4':sum(x['lexical_rank'] is not None and x['lexical_rank']<=4 for x in before if x['split']=='heldout')/8},
      'candidate':{'dev_top1':top1(after,'dev','rerank_rank'),'heldout_top1':top1(after,'heldout','rerank_rank'),'heldout_top4':sum(x['rerank_rank'] is not None and x['rerank_rank']<=4 for x in after if x['split']=='heldout')/8}}
    c=report['metrics']['candidate']; l=report['metrics']['lexical']
    report['decision']='accept_opt_in' if c['heldout_top1']>l['heldout_top1'] and c['dev_top1']>=l['dev_top1'] else 'reject'
    report['deployment']='opt_in_only' if report['decision']=='accept_opt_in' else 'not_published'
    report['status']='completed'; report['claims']='Cross-template reranker candidate; no answer generation or default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_cross_rerank_v1.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e1_e2_native_attempt02_20261023/selected_extension')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),
            'protocol':json.loads(a.protocol.read_text()),'downloads':0,'optimizer_updates':0,'checkpoints_saved':0}
    profiler=RunProfiler('mps',a.output)
    try:
        with exclusive_mps_worker(ROOT):profiler.start();execute(a,profiler,report)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        prof=profiler.stop();report['profile']=prof;mem=(prof or {}).get('memory') or {}
        report['costs']={'profiled_wall_seconds':(prof or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics'),'decision':report.get('decision')}))
if __name__=='__main__':main()
