#!/usr/bin/env python3
"""Candidate-context scoring for frozen top-record pointer (no pointer training)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib,json,os,sys,traceback
from pathlib import Path
import torch
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def execute(args,profiler,report):
    from replica_lab.architecture.contracts import CopySpec,TaskGraph
    from replica_lab.architecture.loading import encode_prompt,load_local_mps
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import EligibilityScope,MemoryRecord,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    protocol=json.loads(args.protocol.read_text()); train_data=ROOT/protocol['train_dataset']['root']; eval_data=ROOT/protocol['eval_dataset']['root']
    if sha(train_data/'records.jsonl')!=protocol['train_dataset']['records_sha256']: raise ValueError('train records changed')
    if sha(eval_data/'records.jsonl')!=protocol['eval_dataset']['records_sha256']: raise ValueError('eval records changed')
    train_q=[q for q in rows(train_data/'queries.jsonl') if q['split']=='train']; eval_q=rows(eval_data/'queries.jsonl')
    template='Answer the query from stored records. Reply with only the value.\n\nQuery: {query}'
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
        copy=CopySpec(max_tokens=512,mode='pointer',max_span=protocol['max_span']))
    parent=model.load_extension(Path(protocol['pointer_extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,parent=parent)
    ledger=RecordLedger(args.output/'context.sqlite')
    for r in [MemoryRecord(**x) for x in rows(train_data/'records.jsonl')+rows(eval_data/'records.jsonl')]:
        ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    def scope(q):
        return EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
    def prepare(q):
        sc=scope(q); ids=encode_prompt(tok,template.replace('{query}',q['query']),'mps'); mask=torch.ones_like(ids,dtype=torch.bool)
        sel=session.index.retrieve(q['query'],**{k:getattr(sc,k) for k in ('subject_id','branch_id','purpose','as_of','mode','snapshot_at')},limit=16)
        return session.prepare(prompt_ids=ids,prompt_mask=mask,query=q['query'],scope=sc,
            task=TaskGraph(q['case_id'],q['query']),selection=sel,core_impl='none',enabled_branches=('memory','copy'))
    def context_features(req,cand,cmask,prov):
        top=req.frame.memory.spans[0].record_id
        ids=[i for span in req.frame.memory.spans if span.record_id==top for i in span.token_ids]
        tokens=[t.replace('Ġ','').replace('▁','').strip().lower() for t in tok.convert_ids_to_tokens(ids)]
        is_positions=[i for i,t in enumerate(tokens) if t in ('is','are','was','were')]
        features=[]
        for start,length,_ in prov:
            last_is=max([i for i in is_positions if i < start], default=-1)
            after=1.0 if last_is == start-1 else 0.0
            distance=(start-last_is)/max(1,len(ids)) if last_is>=0 else 1.0
            features.append([start/max(1,len(ids)), (start+length)/max(1,len(ids)),
                             after, length/max(1,protocol['max_span']), distance])
        return torch.tensor([features],dtype=torch.float32,device=model.base.get_input_embeddings().weight.device)
    def collect(q,*,need_target):
        req=prepare(q)
        try:
            cand,cmask,valid,prov=session.copy_candidates(req,include_provenance=True,
                unit='word',record_scope='top')
            scores=model.pointer_scores(req.frame,cand,cmask,valid).detach()
            features=context_features(req,cand,cmask,prov)
            target=0
            if need_target:
                target=None
                for i in range(cand.shape[1]):
                    seq=[int(x) for x in cand[0,i][cmask[0,i]].tolist()]
                    if tok.decode(seq,skip_special_tokens=True).strip().casefold()==q['expected_answer'].casefold():
                        target=i; break
                if target is None: raise ValueError(f'target absent {q["case_id"]}')
            return scores,features,target,cand,cmask
        finally:
            req.release()
    train=[]
    for q in train_q: train.append((q,*collect(q,need_target=True)))
    eval_rows=[]
    for q in eval_q: eval_rows.append((q,*collect(q,need_target=False)))
    # Train only a small candidate-context bias on training queries.
    head=torch.nn.Linear(5,1).to(model.base.get_input_embeddings().weight.device)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt=torch.optim.AdamW(head.parameters(),lr=protocol['training']['lr'])
    losses=[]
    for step in range(1,int(protocol['training']['updates'])+1):
        opt.zero_grad(set_to_none=True); loss=torch.zeros((),device='mps')
        for q,scores,feats,target,cand,cmask in train:
            bias=head(feats).squeeze(0)
            loss=loss+torch.nn.functional.cross_entropy((scores+bias).unsqueeze(0),torch.tensor([target],device='mps'))
        loss=loss/len(train); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        if step%100==0: profiler.mark('context_update',step=step,loss=losses[-1])
    head.eval()
    def decode_idx(cand,cmask,idx): return tok.decode([int(x) for x in cand[0,idx][cmask[0,idx]].tolist()],skip_special_tokens=True).strip()
    parent_rows=[]; candidate_rows=[]
    with torch.no_grad():
        for q,scores,feats,_,cand,cmask in eval_rows:
            pi=int(scores.argmax().item()); ci=int((scores+head(feats).squeeze(0)).argmax().item())
            parent_rows.append({'case_id':q['case_id'],'text':decode_idx(cand,cmask,pi),
                'expected':q['expected_answer'],'correct':decode_idx(cand,cmask,pi).casefold()==q['expected_answer'].casefold()})
            candidate_rows.append({'case_id':q['case_id'],'text':decode_idx(cand,cmask,ci),
                'expected':q['expected_answer'],'correct':decode_idx(cand,cmask,ci).casefold()==q['expected_answer'].casefold()})
    pa=sum(x['correct'] for x in parent_rows)/len(parent_rows); ca=sum(x['correct'] for x in candidate_rows)/len(candidate_rows)
    report['parent']=parent_rows; report['candidate']=candidate_rows; report['metrics']={'parent':pa,'candidate':ca}
    report['training']={'losses':losses,'context_head_parameters':sum(p.numel() for p in head.parameters())}
    head_path=args.output/'context_head.safetensors'
    from safetensors.torch import save_file
    save_file({n:p.detach().cpu() for n,p in head.named_parameters()},str(head_path))
    report['context_head']={'path':str(head_path),'sha256':sha(head_path),'parameters':{'weight':list(head.weight.shape),'bias':list(head.bias.shape)},
        'parent_pointer':protocol['pointer_extension'],'features':protocol['features'],'base_saved':False}
    report['decision']='accept_opt_in' if ca>pa else 'reject'; report['deployment']='opt_in_only' if ca>pa else 'not_published'
    report['status']='completed'; report['claims']='Candidate-context bias diagnostic; pointer weights frozen; no default change.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_pointer_context_v1.json')
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
