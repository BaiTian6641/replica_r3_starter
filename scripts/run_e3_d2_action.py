#!/usr/bin/env python3
"""Bounded full-augmented D2 action diagnostic (no export on rejection)."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os, sys, time, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from run_e3_qualification import (admit_policy, build_action_batch, run_episode,
    run_teacher, selection_for)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['mps'],default='mps'); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_d2_action_v1.json')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e3_set_attempt03_20261023/selected_extension')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(a.protocol.read_text()); report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
      'protocol_sha256':sha(a.protocol),'protocol':protocol,'downloads':0,'base_checkpoint_saved':False}
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    profiler=RunProfiler('mps',a.output,sample_interval=0.1)
    try:
        with exclusive_mps_worker(ROOT): profiler.start(); execute(a,protocol,report,profiler)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); raise
    finally:
        profile=profiler.stop(); mem=(profile or {}).get('memory') or {}
        report.update(profile=profile,costs={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),
            'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes'),
            'sampled_peak_rss_bytes':mem.get('rss_peak_sampled_bytes'),'base_checkpoint_saved':False})
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'heldout':report.get('heldout_completions'),'costs':report.get('costs')}))

def execute(args,protocol,report,profiler):
    import torch
    from replica_lab.architecture.contracts import CoreSpec, TaskGraph
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.task_graph import Budget
    from replica_lab.runtime.tools import FixtureWorkspace
    from run_e3_qualification import policy_text
    torch.manual_seed(int(protocol['training']['seed']))
    core=CoreSpec(**protocol['core']); model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,core=core)
    if not args.parent_extension.is_dir(): raise ValueError('parent extension missing')
    receipt=model.load_extension(args.parent_extension)
    model.set_regime('full_augmented')
    train_params=[p for p in model.parameters() if p.requires_grad]
    report.update(manifest=model.manifest.to_dict(),manifest_digest=model.manifest.digest,
      source_lock=lock,parent_extension=str(args.parent_extension),parent_receipt=receipt,
      trainable_parameters=sum(p.numel() for p in train_params),trainable_tensors=len(train_params))
    ledger=RecordLedger(args.output/'ledger.sqlite'); subject='d2-action'; purpose='d2-action'
    root=ROOT/protocol['fixtures']['root']; budgets=protocol['evaluation']
    cases={x['case_id']:x for x in protocol['fixtures']['train']+protocol['fixtures']['dev']+protocol['fixtures']['heldout']}
    # Admit one host policy record per branch; no evaluator/reference-repair text enters memory.
    for cid,role in [(x,'train') for x in protocol['fixtures']['train']]+[(x,'eval') for x in protocol['evaluation']['cases']]:
        tools=FixtureWorkspace(root/cid)
        admit_policy(ledger,tools,subject=subject,branch=cid,purpose=purpose)
    session=MemorySession(model,tok,ledger,max_candidate_units=16)
    scopes={cid:EligibilityScope(subject,cid,purpose,0,'assistant_for_source',None) for cid in cases}
    examples=[]
    for cid in protocol['fixtures']['train']:
        work=args.output/'teacher'/cid
        result=run_teacher(root/cid,work,tokenizer=tok,subject=subject,branch=cid,purpose=purpose,
          max_prompt_tokens=int(protocol['training']['max_prompt_tokens']),
          max_generated_tokens=int(protocol['training']['max_label_tokens']),repair_tool='set_constant')
        if not result['summary']['success']: raise AssertionError('teacher failed '+cid)
        tools=FixtureWorkspace(work)
        for ex in result['examples']:
            ex['case_id']=cid; ex['goal']=tools.goal
            ex['task_graph']=TaskGraph(task_id=cid,goal=tools.goal,state_digest=ex['task_digest'],revision=ex['revision'])
            examples.append(ex)
        report.setdefault('teacher',[]).append({'case_id':cid,'steps':len(result['examples']),'success':result['summary']['success']})
    report['alignment_examples']={'count':len(examples),'max_label_tokens':max(len(x['target_token_ids']) for x in examples)}
    selection_for_case=lambda cid: selection_for(session,scopes[cid],FixtureWorkspace(root/cid).goal)
    alignment=protocol['training']; opt=build_optimizer(model,'muon8bit',lr_base=alignment['lr_base'],lr_new=alignment['lr_new'],
      lr_muon_base=alignment['lr_muon_base'],lr_muon_new=alignment['lr_muon_new'],weight_decay=alignment['weight_decay'])
    if opt is None: raise RuntimeError('optimizer has no trainable parameters')
    report['optimizer_initial']=opt.audit(); model.train(); updates=[]
    for update in range(1,int(alignment['updates'])+1):
        opt.zero_grad(set_to_none=True); total=0.0
        for off in range(int(alignment['examples_per_update'])):
            ex=examples[(2*(update-1)+off)%len(examples)]
            cid=ex['case_id']; action_ids,mask,labels,prompt_tokens=build_action_batch(tok,ex,'mps')
            with session.training_request(prompt_ids=action_ids[:,:prompt_tokens],prompt_mask=mask[:,:prompt_tokens],
                    query=ex['goal'],scope=scopes[cid],task=ex['task_graph'],selection=selection_for_case(cid),
                    task_features=ex['task_features'],task_digest=ex['task_digest'],revision=ex['revision'],
                    previous_core=None,core_impl='recurrent',enabled_branches=('memory','workspace'),
                    train_core=True,train_sense=True) as request:
                output=session.forward(request,action_ids,labels=labels,attention_mask=mask)
                loss=output.loss/int(alignment['examples_per_update']); total+=float(loss.detach()); session.backward(request,loss)
            del output,loss,action_ids,mask,labels
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],alignment['gradient_clip'],error_if_nonfinite=True))
        ratio=min(1.0,update/int(alignment['warmup_updates']))
        for group in opt.param_groups:
            base=group['group_name']=='original'
            group['lr']=ratio*(alignment['lr_muon_base'] if base else alignment['lr_muon_new']) if group['algorithm']=='muon8bit' else ratio*(alignment['lr_base'] if base else alignment['lr_new'])
        opt.step(); opt.zero_grad(set_to_none=True); updates.append({'update':update,'loss':total,'gradient_norm':norm})
        profiler.mark('d2_action_update',update=update,loss=total)
    for group in opt.param_groups: group['lr']=0.0
    del opt; model.eval(); torch.mps.empty_cache(); report['alignment']=updates
    report['optimizer_final_note']='state discarded after in-process diagnostic; no checkpoint exported'
    # Held-out/dev evaluation uses the actual changed model in this process.
    budget=Budget(int(budgets['max_decisions']),int(budgets['max_tool_calls']),
                  int(budgets['max_decisions'])*int(budgets['prompt_tokens']),
                  int(budgets['max_decisions'])*int(budgets['generated_tokens']))
    episodes=[]
    for cid in protocol['evaluation']['cases']:
        work=args.output/'episodes'/cid
        result,_=run_episode(root/cid,work,session=session,tokenizer=tok,mode='full',subject=subject,purpose=purpose,
          budget=budget,max_prompt_tokens=int(budgets['prompt_tokens']),generated_token_budget=int(budgets['generated_tokens']),
          zero_previous=True,mark=lambda name,**kw: profiler.mark(name,**kw))
        episodes.append(result)
        profiler.mark('d2_eval_episode',case=cid,success=result['success'],decisions=result['decision_count'])
    report['episodes']=episodes
    held=[x for x in episodes if x['case_id'] in protocol['fixtures']['heldout']]
    report['heldout_completions']=f"{sum(x['success'] for x in held)}/{len(held)}"
    report['dev_completions']=f"{sum(x['success'] for x in episodes if x['case_id'] in protocol['fixtures']['dev'])}/{len(protocol['fixtures']['dev'])}"
    report['candidate_export']='not_written'
    report['status']='completed'
    report['claims']='Full-backbone action diagnostic only; no export on rejection; no general capability claim.'
    ledger.close()
if __name__=='__main__': main()
