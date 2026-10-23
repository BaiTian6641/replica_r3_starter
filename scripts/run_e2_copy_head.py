#!/usr/bin/env python3
"""Train/evaluate the record-token copy head on fresh novel-value data (MPS)."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,os,sys,traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def cap(session, selection, max_units=16):
    from collections import defaultdict
    from replica_lab.memory_system.index import RetrievalSelection
    spans=session.spans(selection.records); counts=defaultdict(int)
    for span in spans: counts[span.record_id]+=1
    records=[]; scores=[]; used=0
    for record,score in zip(selection.records,selection.scores):
        if used+counts[record.record_id]>max_units: break
        records.append(record); scores.append(score); used+=counts[record.record_id]
    return RetrievalSelection(tuple(records),tuple(scores),selection.epoch,selection.ledger_id,
                              selection.scope,selection.query,selection.dense_producer_id)
def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import CopySpec,TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import EligibilityScope,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    protocol=json.loads(args.protocol.read_text()); data=ROOT/protocol['dataset']['root']
    if sha(data/'records.jsonl')!=protocol['dataset']['records_sha256']: raise ValueError('records changed')
    records=[MemoryRecord(**row) for row in rows(data/'records.jsonl')]
    queries=rows(data/'queries.jsonl'); train=[q for q in queries if q['split']=='train']
    dev=[q for q in queries if q['split']=='dev']; held=[q for q in queries if q['split']=='heldout']
    template='Answer the query from stored records. Reply with only the value.\n\nQuery: {query}'
    def scope_for(q): return EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
    def prepare(session,q,*,copy_enabled):
        ids=encode_prompt(session.tokenizer,template.replace('{query}',q['query']),'mps')
        mask=torch.ones_like(ids,dtype=torch.bool)
        selection=cap(session,session.index.retrieve(q['query'],**{k:getattr(scope_for(q),k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16))
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=scope_for(q),
            task=TaskGraph(q['case_id'],q['query']),selection=selection,core_impl='none',
            enabled_branches=('memory','copy') if copy_enabled else ('memory',))
    def evaluate(session,qs,copy_enabled):
        out=[]
        for q in qs:
            req=prepare(session,q,copy_enabled=copy_enabled)
            try:
                gen=session.generate(req,max_new_tokens=8); n=int(req.frame.sense.prompt_ids.shape[1])
                ans=session.tokenizer.decode(gen[0,n:].tolist(),skip_special_tokens=True).strip()
                out.append({'case_id':q['case_id'],'split':q['split'],'answer':ans,'expected':q['expected_answer'],
                            'correct':ans.casefold()==q['expected_answer'].casefold(),
                            'selected':[s.record_id for s in req.frame.memory.spans]})
            finally: req.release()
        return out
    torch.manual_seed(protocol['training']['seed'])
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
                                  copy=CopySpec(max_tokens=protocol['copy']['max_tokens']))
    parent=model.load_extension(Path(args.parent_extension),accept_parent_v1=True)
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,parent=parent,
                  parameters=model.parameter_audit())
    ledger=RecordLedger(args.output/'copy_ledger.sqlite')
    for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    # Freeze everything except the copy head.
    previous={n:p.requires_grad for n,p in model.new_named_parameters()}
    for n,p in model.new_named_parameters(): p.requires_grad_(n.startswith('copy_head.'))
    optimizer=build_optimizer(model,'muon8bit',lr_base=0.0,lr_new=protocol['training']['lr_new'],
        lr_muon_base=0.0,lr_muon_new=protocol['training']['lr_muon_new'],weight_decay=0.0)
    if optimizer is None: raise RuntimeError('copy head is not trainable')
    model.train(); report['training']={'optimizer':optimizer.audit(),'updates':[]}
    for update in range(1,int(protocol['training']['updates'])+1):
        optimizer.zero_grad(set_to_none=True); total=0.0
        for offset in range(int(protocol['training']['examples_per_update'])):
            q=train[(4*(update-1)+offset)%len(train)]
            req=prepare(session,q,copy_enabled=True)
            target=tok(q['expected_answer'],add_special_tokens=False)['input_ids']+[tok.eos_token_id]
            target_ids=torch.tensor([target],dtype=torch.long,device='mps')
            action_ids=torch.cat((req.frame.sense.prompt_ids,target_ids),dim=1)
            labels=action_ids.clone(); labels[:,:req.frame.sense.prompt_ids.shape[1]]=-100
            with session.training_request(prompt_ids=req.frame.sense.prompt_ids,
                    prompt_mask=req.frame.sense.prompt_mask,query=q['query'],scope=scope_for(q),
                    task=TaskGraph(q['case_id'],q['query']),selection=req.selection,
                    core_impl='none',enabled_branches=('memory','copy')) as active:
                out=session.forward(active,action_ids,labels=labels)
                loss=out.loss; total+=float(loss.detach())
                if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite copy loss')
                session.backward(active,loss)
            req.release(); del out,loss
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['training']['updates'].append({'update':update,'mean_loss':total})
        profiler.mark('copy_update',update=update,loss=total)
    del optimizer
    for n,p in model.new_named_parameters(): p.requires_grad_(previous[n])
    model.eval(); extension=args.output/'selected_extension'; receipt=model.save_extension(extension)
    report['extension']={'path':str(extension),'format':receipt['format'],'weights_sha256':receipt['weights_sha256'],
        'manifest_digest':receipt['manifest_digest'],'parameters':receipt['parameters'],'base_saved':False,
        'bytes':sum(p.stat().st_size for p in extension.rglob('*') if p.is_file())}
    # Reload into a fresh adapter for serialization and evaluation.
    replacement=MiniCPM5Adapter(model.base,model.manifest).to('mps').eval(); replacement.seal_original_identity(); replacement.load_extension(extension)
    session2=MemorySession(replacement,tok,ledger,max_candidate_units=64)
    enabled=evaluate(session2,dev+held,copy_enabled=True)
    disabled=evaluate(session2,dev+held,copy_enabled=False)
    report['enabled']=enabled; report['disabled']=disabled
    def acc(xs,split):
        y=[x for x in xs if x['split']==split]; return sum(x['correct'] for x in y)/len(y) if y else None
    report['metrics']={'enabled':{s:acc(enabled,s) for s in ('dev','heldout')},
                       'disabled':{s:acc(disabled,s) for s in ('dev','heldout')},
                       'enabled_answers':Counter(x['answer'] for x in enabled),
                       'disabled_answers':Counter(x['answer'] for x in disabled)}
    if acc(enabled,'heldout')>acc(disabled,'heldout') and acc(enabled,'dev')>=acc(disabled,'dev'):
        decision='accept_opt_in'
    elif acc(enabled,'heldout')<=acc(disabled,'heldout'):
        decision='reject'
    else: decision='defer'
    report['decision']=decision; report['deployment']='not_published' if decision!='accept_opt_in' else 'opt_in_only'
    report['status']='completed'; report['claims']='Copy-head candidate on synthetic novel values; opt-in evaluation only; base/E2 writer frozen.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_copy_head_v1.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e1_e2_native_attempt02_20261023/selected_extension')
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
