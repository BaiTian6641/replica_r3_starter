#!/usr/bin/env python3
"""Diagnostic: can accepted E3 core states classify the next action kind?

Teacher trajectories provide actual host-authored labels and tool receipts. The
accepted E3 extension computes prompt-only core states. A small classifier is
trained on train cases and evaluated on dev/held-out cases; this is a
representation probe, not a policy-head deployment or task-completion claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib, json, os, sys, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KINDS=('read','run_test','set_constant','stop')

def action_kind(target_text):
    obj=json.loads(target_text)
    if obj.get('kind')=='read': return 'read'
    if obj.get('kind')=='stop': return 'stop'
    if obj.get('kind')=='invoke_tool':
        tool=obj.get('tool'); return tool if tool in ('run_test','set_constant') else tool
    return obj.get('kind')

def parse_examples(role_cases, output, tokenizer, protocol):
    from run_e3_qualification import run_teacher
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.runtime.tools import FixtureWorkspace
    examples=[]; reports=[]
    for case_id in role_cases:
        case_dir=Path(protocol['fixtures']['root'])/case_id
        tools=FixtureWorkspace(case_dir)
        actions = protocol['teacher'].get('pass_actions') if case_id in             protocol['teacher'].get('pass_cases', ()) else None
        result=run_teacher(case_dir, output/'teacher'/case_id, tokenizer=tokenizer,
            subject=protocol['subject'], branch=case_id, purpose=protocol['purpose'],
            max_prompt_tokens=int(protocol['teacher']['max_prompt_tokens']),
            max_generated_tokens=int(protocol['teacher']['max_generated_tokens']),
            repair_tool='set_constant', actions=actions)
        if not result['summary']['success']:
            raise AssertionError(f'teacher failed: {case_id}')
        for item in result['examples']:
            item['case_id']=case_id; item['goal']=tools.goal
            item['role']='train' if case_id in protocol['fixtures']['train'] else (
                'dev' if case_id in protocol['fixtures']['dev'] else 'heldout')
            item['action_kind']=action_kind(item['target_text'])
            item['task_graph']=TaskGraph(case_id, tools.goal,
                state_digest=item['task_digest'], revision=item['revision'])
            examples.append(item)
        reports.append({'case_id':case_id,'steps':len(result['examples']),'success':True})
    return examples,reports

class ProbeHead:
    def __init__(self,width,hidden=64):
        import torch
        from torch import nn
        torch.manual_seed(20261023)
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,len(KINDS)))
    def train(self,x,y,steps=240):
        import torch
        self.net.train(); opt=torch.optim.AdamW(self.net.parameters(),lr=0.01,weight_decay=0.0)
        for _ in range(steps):
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(self.net(x),y)
            loss.backward(); opt.step()
        return float(loss.detach())
    def predict(self,x):
        import torch
        self.net.eval()
        with torch.no_grad(): return self.net(x).argmax(-1)

def vector(core, task_features, *, use_workspace=True, use_task=True):
    import torch
    parts=[]
    if use_workspace:
        w=core.workspace.detach().cpu().float(); v=core.valid.detach().cpu().bool()
        pooled=(w.masked_fill(~v.unsqueeze(-1),0).sum(1)/v.sum(1,keepdim=True).clamp_min(1))
        parts.append(pooled)
    if use_task: parts.append(task_features.detach().cpu().float().reshape(1,-1))
    return torch.cat(parts,dim=-1)[0]

def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import CoreSpec, TaskGraph
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from run_e3_qualification import admit_policy, initial_state, selection_for, tokenize_prompt
    protocol=json.loads(args.protocol.read_text())
    if protocol['source_revision']!='f97400052a43d642bbc6e9975e2397e3ae6a6b52':
        raise ValueError('source revision mismatch')
    if not Path(protocol['extension']).exists():
        raise ValueError('accepted E3 extension missing')
    core_spec=CoreSpec(**protocol['core'])
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,core=core_spec)
    receipt=model.load_extension(Path(protocol['extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,
                  extension=receipt,parameters=model.parameter_audit())
    cases={}
    for role in ('train','dev','heldout'):
        for case_id in protocol['fixtures'][role]: cases[case_id]=role
    examples,teacher_reports=parse_examples(cases,args.output,tok,protocol)
    report['teacher']=teacher_reports
    ledger=RecordLedger(args.output/'policy_probe.sqlite')
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    subject,purpose=protocol['subject'],protocol['purpose']
    scopes={}; selections={}; tools_by_case={}
    from replica_lab.runtime.tools import FixtureWorkspace
    # Admit every policy first; retrieval is epoch-bound and later admissions
    # intentionally invalidate earlier selections.
    for case_id in cases:
        tools=FixtureWorkspace(Path(protocol['fixtures']['root'])/case_id)
        tools_by_case[case_id]=tools
        admit_policy(ledger,tools,subject=subject,branch=case_id,purpose=purpose)
        scopes[case_id]=EligibilityScope(subject,case_id,purpose,0,
                                          'assistant_for_source',None)
    for case_id in cases:
        selections[case_id]=selection_for(session,scopes[case_id],
                                           tools_by_case[case_id].goal)
    rows=[]
    for item in examples:
        ids=tokenize_prompt(tok,item['prompt_text'],'mps')
        mask=torch.ones_like(ids,dtype=torch.bool)
        req=session.prepare(prompt_ids=ids,prompt_mask=mask,query=item['goal'],
            scope=scopes[item['case_id']],task=item['task_graph'],
            selection=selections[item['case_id']],task_features=item['task_features'],
            task_digest=item['task_digest'],revision=item['revision'],core_impl='recurrent',
            enabled_branches=('memory','workspace'),previous_core=None)
        try:
            row={'case_id':item['case_id'],'role':item['role'],'step':item['step'],
                 'label':item['action_kind'],
                 'vector':vector(req.frame.core,torch.tensor(item['task_features'],device='mps')),
                 'prompt_tokens':int(ids.shape[1])}
            row['task_vector']=vector(req.frame.core,torch.tensor(item['task_features'],device='mps'),use_workspace=False)
            row['workspace_vector']=vector(req.frame.core,torch.tensor(item['task_features'],device='mps'),use_task=False)
            rows.append(row)
        finally: req.release()
    report['examples']={'count':len(rows),'roles':dict(Counter(r['role'] for r in rows)),
                        'kinds':dict(Counter(r['label'] for r in rows)),
                        'max_prompt_tokens':max(r['prompt_tokens'] for r in rows)}
    idx={kind:i for i,kind in enumerate(KINDS)}
    x=torch.stack([r['vector'] for r in rows]); y=torch.tensor([idx[r['label']] for r in rows])
    roles=torch.tensor([{'train':0,'dev':1,'heldout':2}[r['role']] for r in rows])
    results={}
    for name,use_w,use_t in (('joint',True,True),('workspace_only',True,False),('task_only',False,True)):
        head=ProbeHead(x.shape[1]); loss=head.train(x[roles==0],y[roles==0])
        results[name]={'train_loss':loss}
        for role,flag in (('train',0),('dev',1),('heldout',2)):
            pred=head.predict(x[roles==flag]);
            truth=y[roles==flag]
            results[name][role]={'count':int((roles==flag).sum()),
                'accuracy':float((pred==truth).float().mean()),
                'predictions':[KINDS[int(v)] for v in pred.tolist()],
                'labels':[KINDS[int(v)] for v in truth.tolist()]}
        results[name]['class_majority']={'train':max(Counter(KINDS[int(v)] for v in y[roles==0].tolist()).values())/int((roles==0).sum()),
                                         'heldout':max(Counter(KINDS[int(v)] for v in y[roles==2].tolist()).values())/int((roles==2).sum())}
    # Step-only baseline: a classifier that knows only the trajectory step.
    # This must be reported because teacher sequences can make step position a
    # shortcut label. It is computed from train labels only.
    from collections import Counter as _Counter
    train_rows=[r for r in rows if r['role']=='train']
    global_label=_Counter(r['label'] for r in train_rows).most_common(1)[0][0]
    by_step={}
    for r in train_rows:
        by_step.setdefault(r['step'],_Counter())[r['label']]+=1
    step_label={step:counts.most_common(1)[0][0] for step,counts in by_step.items()}
    for role,flag in (('train',0),('dev',1),('heldout',2)):
        selected=[r for r in rows if {'train':0,'dev':1,'heldout':2}[r['role']]==flag]
        preds=[step_label.get(r['step'],global_label) for r in selected]
        truth=[r['label'] for r in selected]
        results.setdefault('step_only',{})[role]={
            'count':len(selected),'accuracy':sum(a==b for a,b in zip(preds,truth))/len(selected),
            'predictions':preds,'labels':truth}
    results['step_only']['rule']='train-majority label per teacher step; global train majority for unseen steps'
    report['probe']=results
    report['status']='completed'
    joint=results['joint']['heldout']['accuracy']
    baseline=max(results['joint']['class_majority']['heldout'],results['step_only']['heldout']['accuracy'])
    margin=float(protocol['probe'].get('decision_margin',0.1))
    if joint>0.5 and joint>baseline+margin:
        decision='representation_signal'
    elif joint>baseline:
        decision='inconclusive'
    else:
        decision='representation_negative'
    report['decision']=decision
    report['decision_baseline']={'joint_heldout':joint,'class_majority_heldout':results['joint']['class_majority']['heldout'],
                                  'step_only_heldout':results['step_only']['heldout']['accuracy'],
                                  'required_margin':margin}
    report['claims']='Diagnostic only: prompt-only core states, host-authored teacher labels, no policy deployment, no task completion.'
    ledger.close()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_policy_probe_v1.json')
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
    print(json.dumps({'status':report['status'],'decision':report.get('decision'),
                      'baseline':report.get('decision_baseline'),
                      'probe':{k:{r:(v.get(r,{}).get('accuracy') if isinstance(v.get(r),dict) else None) for r in ('train','dev','heldout')} for k,v in report.get('probe',{}).items()}}))
if __name__=='__main__': main()
