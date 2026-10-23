#!/usr/bin/env python3
"""Continue pointer head on top-record whole-word candidates; fresh evaluation."""
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
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import MemoryAnswerService,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    protocol=json.loads(args.protocol.read_text()); train_data=ROOT/protocol['train_dataset']['root']; eval_data=ROOT/protocol['eval_dataset']['root']
    if sha(train_data/'records.jsonl')!=protocol['train_dataset']['records_sha256']: raise ValueError('train records changed')
    if sha(eval_data/'records.jsonl')!=protocol['eval_dataset']['records_sha256']: raise ValueError('eval records changed')
    train_q=[q for q in rows(train_data/'queries.jsonl') if q['split']=='train']; eval_q=rows(eval_data/'queries.jsonl')
    template='Answer the query from stored records. Reply with only the value.\n\nQuery: {query}'
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
        copy=CopySpec(max_tokens=512,mode='pointer',max_span=protocol['max_span']))
    parent=model.load_extension(Path(protocol['parent_extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,parent=parent)
    ledger=RecordLedger(args.output/'scope.sqlite')
    for r in [MemoryRecord(**x) for x in rows(train_data/'records.jsonl')+rows(eval_data/'records.jsonl')]:
        ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    def scope(q):
        from replica_lab.memory_system import EligibilityScope
        return EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
    def prep(q):
        sc=scope(q); ids=encode_prompt(tok,template.replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=session.index.retrieve(q['query'],**{k:getattr(sc,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16)
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=sc,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',enabled_branches=('memory','copy'))
    def evaluate(tag):
        out=[]
        for q in eval_q:
            req=prep(q)
            try:
                a=MemoryAnswerService(session,pointer_enabled=True,candidate_unit='word',record_scope='top').answer(req)
                out.append({'case_id':q['case_id'],'text':a.text,'expected':q['expected_answer'],
                    'correct':a.text.casefold()==q['expected_answer'].casefold(),'method':a.method,
                    'record_ids':a.record_ids,'span_hashes':a.span_hashes,'source_hashes':a.source_hashes})
            finally:req.release()
        return out
    parent_results=evaluate('parent')
    # Continue only the copy head on top-record word candidates.
    previous={n:p.requires_grad for n,p in model.new_named_parameters()}
    for n,p in model.new_named_parameters(): p.requires_grad_(n.startswith('copy_head.'))
    optimizer=build_optimizer(model,'muon8bit',lr_base=0.0,lr_new=protocol['training']['lr'],
        lr_muon_base=0.0,lr_muon_new=protocol['training']['lr_muon'],weight_decay=0.0)
    if optimizer is None: raise RuntimeError('copy head not trainable')
    model.train(); report['training']={'optimizer':optimizer.audit(),'updates':[]}
    for update in range(1,int(protocol['training']['updates'])+1):
        optimizer.zero_grad(set_to_none=True); total=0.0
        for offset in range(int(protocol['training']['examples_per_update'])):
            q=train_q[(4*(update-1)+offset)%len(train_q)]
            probe=prep(q); ids=probe.frame.sense.prompt_ids.clone(); mask=probe.frame.sense.prompt_mask.clone(); sel=probe.selection; probe.release()
            with session.training_request(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=scope(q),
                    task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',
                    enabled_branches=('memory','copy')) as req:
                cand,cmask,valid,_=session.copy_candidates(req,include_provenance=True,unit='word',record_scope='top')
                target=None
                for i in range(cand.shape[1]):
                    seq=tuple(int(x) for x in cand[0,i][cmask[0,i]].tolist())
                    if tok.decode(list(seq),skip_special_tokens=True).strip().casefold()==q['expected_answer'].casefold():
                        target=i; break
                if target is None: raise ValueError(f'target absent {q["case_id"]}')
                scores=model.pointer_scores(req.frame,cand,cmask,valid)
                loss=torch.nn.functional.cross_entropy(scores,torch.tensor([target],device='mps'))
                total+=float(loss.detach()); session.backward(req,loss)
            del loss
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['training']['updates'].append({'update':update,'mean_loss':total}); profiler.mark('scope_update',update=update,loss=total)
    del optimizer
    for n,p in model.new_named_parameters(): p.requires_grad_(previous[n])
    model.eval(); ext=args.output/'selected_extension'; receipt=model.save_extension(ext)
    report['extension']={'format':receipt['format'],'weights_sha256':receipt['weights_sha256'],
        'manifest_digest':receipt['manifest_digest'],'parameters':receipt['parameters'],'base_saved':False,
        'bytes':sum(p.stat().st_size for p in ext.rglob('*') if p.is_file())}
    replacement=MiniCPM5Adapter(model.base,model.manifest).to('mps').eval(); replacement.seal_original_identity(); replacement.load_extension(ext)
    session2=MemorySession(replacement,tok,ledger,max_candidate_units=64)
    def evaluate2():
        out=[]
        for q in eval_q:
            req=prep_with(session2,q)
            try:
                a=MemoryAnswerService(session2,pointer_enabled=True,candidate_unit='word',record_scope='top').answer(req)
                out.append({'case_id':q['case_id'],'text':a.text,'expected':q['expected_answer'],
                    'correct':a.text.casefold()==q['expected_answer'].casefold(),'record_ids':a.record_ids})
            finally:req.release()
        return out
    def prep_with(sess,q):
        sc=scope(q); ids=encode_prompt(tok,template.replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=sess.index.retrieve(q['query'],**{k:getattr(sc,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16)
        return sess.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=sc,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',enabled_branches=('memory','copy'))
    candidate=evaluate2()
    report['parent_results']=parent_results; report['candidate_results']=candidate
    pa=sum(x['correct'] for x in parent_results)/len(parent_results); ca=sum(x['correct'] for x in candidate)/len(candidate)
    report['metrics']={'parent':pa,'candidate':ca}
    report['decision']='accept_opt_in' if ca>pa else 'reject'
    report['deployment']='opt_in_only' if ca>pa else 'not_published'
    report['status']='completed'; report['claims']='Top-record pointer continuation; no default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_pointer_scope_train_v1.json')
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
