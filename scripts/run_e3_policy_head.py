#!/usr/bin/env python3
"""Separately declared E3 action-kind head + real mixed-outcome episodes.

The head is trained only on mixed train teacher states. At evaluation it selects
one of four kinds; the frozen model then generates the action arguments under a
kind hint. The host validates and executes; no host script chooses the repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib, json, os, sys, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def action_kind(target_text):
    obj=json.loads(target_text)
    if obj.get('kind')=='read': return 'read'
    if obj.get('kind')=='stop': return 'stop'
    if obj.get('kind')=='invoke_tool': return obj.get('tool')
    return obj.get('kind')

class HeadProposer:
    """Select kind with the head, then generate args with the frozen model."""
    def __init__(self, session, tokenizer, head, *, task_id, goal, scope, base):
        self.session, self.tokenizer, self.head = session, tokenizer, head
        self.task_id, self.goal, self.scope, self.base = task_id, goal, scope, base

    def __call__(self, request):
        import torch
        from replica_lab.architecture.contracts import TaskGraph
        from replica_lab.architecture.heads import ACTION_KINDS
        from replica_lab.architecture.loading import encode_prompt
        ids=encode_prompt(self.tokenizer,request.prompt_text,'mps')
        mask=torch.ones_like(ids,dtype=torch.bool)
        task=TaskGraph(self.task_id,self.goal,revision=request.revision,
                       state_digest=request.task_digest)
        probe=self.session.prepare(prompt_ids=ids,prompt_mask=mask,query=self.goal,
            scope=self.scope,task=task,task_features=torch.tensor([request.task_features],
            dtype=torch.float32,device='mps'),task_digest=request.task_digest,
            revision=request.revision,previous_core=request.previous_workspace,
            core_impl='recurrent',enabled_branches=('memory','workspace'))
        try:
            core=probe.frame.core
            with torch.no_grad():
                logits=self.head(torch.tensor([request.task_features],dtype=torch.float32,
                    device='mps'),core.workspace,core.valid)
                kind=ACTION_KINDS[int(logits.argmax(-1)[0])]
        finally:
            probe.release()
        hint=(f'\n\nSelected action kind: {kind}.\n'
              'Return exactly one JSON action object with that kind and no other text.')
        return self.base(replace(request,prompt_text=request.prompt_text+hint))

def core_rows(model,session,tokenizer,examples,scopes,selections):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    rows=[]
    for item in examples:
        ids=encode_prompt(tokenizer,item['prompt_text'],'mps')
        mask=torch.ones_like(ids,dtype=torch.bool)
        task=TaskGraph(item['case_id'],item['goal'],revision=item['revision'],
                       state_digest=item['task_digest'])
        req=session.prepare(prompt_ids=ids,prompt_mask=mask,query=item['goal'],
            scope=scopes[item['case_id']],task=task,
            selection=selections[item['case_id']],task_features=item['task_features'],
            task_digest=item['task_digest'],revision=item['revision'],
            core_impl='recurrent',enabled_branches=('memory','workspace'),
            previous_core=None)
        try:
            core=req.frame.core
            rows.append({'case_id':item['case_id'],'role':item['role'],'step':item['step'],
                'label':item['action_kind'],
                'task_features':core.task_features.detach().cpu().float(),
                'workspace':core.workspace.detach().cpu().float(),
                'valid':core.valid.detach().cpu().bool()})
        finally:req.release()
    return rows

def train_head(model,rows,protocol):
    import torch
    from replica_lab.architecture.heads import ACTION_KINDS, ActionKindHead
    head=ActionKindHead(model.manifest.core).to('mps')
    index={kind:i for i,kind in enumerate(ACTION_KINDS)}
    train=[r for r in rows if r['role']=='train']
    tf=torch.stack([r['task_features'][0] for r in train]).to('mps')
    ws=torch.stack([r['workspace'][0] for r in train]).to('mps')
    valid=torch.stack([r['valid'][0] for r in train]).to('mps')
    y=torch.tensor([index[r['label']] for r in train],dtype=torch.long,device='mps')
    opt=torch.optim.AdamW(head.parameters(),lr=protocol['head']['lr'],
        weight_decay=protocol['head']['weight_decay'])
    losses=[]
    for step in range(1,int(protocol['head']['steps'])+1):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(head(tf,ws,valid),y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,
            error_if_nonfinite=True)
        opt.step(); losses.append(float(loss.detach()))
    return head, {'steps':len(losses),'initial_loss':losses[0],'final_loss':losses[-1],
                  'classes':list(ACTION_KINDS)}

def head_metrics(head,rows):
    import torch
    from replica_lab.architecture.heads import ACTION_KINDS
    out={}
    with torch.no_grad():
        for role in ('train','dev','heldout'):
            selected=[r for r in rows if r['role']==role]
            if not selected: continue
            tf=torch.stack([r['task_features'][0] for r in selected]).to('mps')
            ws=torch.stack([r['workspace'][0] for r in selected]).to('mps')
            valid=torch.stack([r['valid'][0] for r in selected]).to('mps')
            pred=head(tf,ws,valid).argmax(-1).cpu().tolist()
            labels=[r['label'] for r in selected]
            out[role]={'count':len(selected),
                'accuracy':sum(ACTION_KINDS[p]==l for p,l in zip(pred,labels))/len(labels),
                'predictions':[ACTION_KINDS[p] for p in pred],'labels':labels}
    return out

def execute(args,profiler,report):
    import torch
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.heads import ActionKindHead
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.loop import SessionProposer
    from replica_lab.runtime.task_graph import Budget
    from replica_lab.runtime.tools import FixtureWorkspace
    from run_e3_qualification import (admit_policy, initial_state, run_episode,
                                      run_teacher, selection_for, tokenize_prompt)
    protocol=json.loads(args.protocol.read_text())
    if protocol['source_revision']!='f97400052a43d642bbc6e9975e2397e3ae6a6b52':
        raise ValueError('source revision mismatch')
    core_spec=CoreSpec(**protocol['core'])
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,core=core_spec)
    receipt=model.load_extension(Path(protocol['extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,
                  extension=receipt,parameters=model.parameter_audit())
    fixtures=Path(protocol['fixtures']['root'])
    all_cases={}
    for role in ('train','dev','heldout'):
        for case_id in protocol['fixtures'][role]: all_cases[case_id]=role
    pass_cases=set(protocol['head']['pass_cases'])
    pass_actions=protocol['head']['pass_actions']
    examples=[]; teacher=[]
    for case_id,role in all_cases.items():
        tools=FixtureWorkspace(fixtures/case_id)
        actions=pass_actions if case_id in pass_cases else None
        result=run_teacher(fixtures/case_id,args.output/'teacher'/case_id,tokenizer=tok,
            subject=protocol['subject'],branch=case_id,purpose=protocol['purpose'],
            max_prompt_tokens=int(protocol['teacher']['max_prompt_tokens']),
            max_generated_tokens=int(protocol['teacher']['max_generated_tokens']),
            repair_tool='set_constant',actions=actions)
        if not result['summary']['success']: raise AssertionError(f'teacher failed {case_id}')
        for item in result['examples']:
            item['case_id']=case_id; item['role']=role; item['goal']=tools.goal
            item['action_kind']=action_kind(item['target_text'])
            from replica_lab.architecture.contracts import TaskGraph
            item['task_graph']=TaskGraph(case_id,tools.goal,state_digest=item['task_digest'],
                                         revision=item['revision'])
            examples.append(item)
        teacher.append({'case_id':case_id,'role':role,'steps':len(result['examples']),
                        'success':True})
    report['teacher']=teacher
    ledger=RecordLedger(args.output/'head_memory.sqlite'); session=MemorySession(model,tok,ledger,max_candidate_units=64)
    scopes={}; selections={}; tools_by_case={}
    for case_id in all_cases:
        tools=FixtureWorkspace(fixtures/case_id); tools_by_case[case_id]=tools
        admit_policy(ledger,tools,subject=protocol['subject'],branch=case_id,purpose=protocol['purpose'])
        scopes[case_id]=EligibilityScope(protocol['subject'],case_id,protocol['purpose'],0,
                                         'assistant_for_source',None)
    for case_id in all_cases:
        selections[case_id]=selection_for(session,scopes[case_id],tools_by_case[case_id].goal)
    rows=core_rows(model,session,tok,examples,scopes,selections)
    report['examples']={'count':len(rows),'roles':dict(Counter(r['role'] for r in rows)),
                        'labels':dict(Counter(r['label'] for r in rows))}
    head,audit=train_head(model,rows,protocol); report['head_training']=audit
    report['head_metrics']=head_metrics(head,rows)
    head_dir=args.output/'policy_head'; head_dir.mkdir()
    from safetensors.torch import save_file
    save_file({n:p.detach().cpu().contiguous() for n,p in head.named_parameters()},
              str(head_dir/'head.safetensors'))
    manifest={'format':'replica.e3.action_kind_head.v1','created_at_utc':'2026-10-23',
        'parent_extension':protocol['extension'],'parent_manifest_digest':model.manifest.digest,
        'classes':list(head.kinds),'core':asdict(core_spec),
        'data_manifest_sha256':protocol['fixtures']['manifest_sha256'],
        'training_roles':['train'],'parameters':{n:list(p.shape) for n,p in head.named_parameters()},
        'weights_sha256':sha(head_dir/'head.safetensors'),'base_saved':False,
        'scope':'Synthetic mixed-outcome fixtures; proposal prior only.'}
    (head_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report['head_artifact']=manifest
    report['optimizer_updates']=int(protocol['head']['steps'])
    head.eval(); model.eval()
    episodes=[]; controls=[]
    budget=Budget(int(protocol['episodes']['decisions']),int(protocol['episodes']['tool_calls']),
                  int(protocol['episodes']['decisions'])*int(protocol['episodes']['prompt_tokens']),
                  int(protocol['episodes']['decisions'])*int(protocol['episodes']['generated_tokens']))
    for case_id in protocol['fixtures']['heldout']:
        tools=tools_by_case[case_id]
        base=SessionProposer(session,tok,task_id=case_id,goal=tools.goal,
            scope=scopes[case_id],device='mps',zero_previous_workspace=True)
        proposer=HeadProposer(session,tok,head,task_id=case_id,goal=tools.goal,
                              scope=scopes[case_id],base=base)
        result,_=run_episode(fixtures/case_id,args.output/'episodes'/case_id,session=session,
            tokenizer=tok,mode='full',subject=protocol['subject'],purpose=protocol['purpose'],
            budget=budget,max_prompt_tokens=int(protocol['episodes']['prompt_tokens']),
            generated_token_budget=int(protocol['episodes']['generated_tokens']),
            zero_previous=True,proposer=proposer)
        result['arm']='head_conditioned'; episodes.append(result)
        profiler.mark('head_episode',case=case_id,success=result['success'])
    # Null control on one pass and one repair case, using the accepted model
    # without a head hint. Actual tools and the independent evaluator remain.
    for case_id in protocol['episodes']['control_cases']:
        tools=tools_by_case[case_id]
        result,_=run_episode(fixtures/case_id,args.output/'controls'/case_id,session=session,
            tokenizer=tok,mode='full',subject=protocol['subject'],purpose=protocol['purpose'],
            budget=budget,max_prompt_tokens=int(protocol['episodes']['prompt_tokens']),
            generated_token_budget=int(protocol['episodes']['generated_tokens']),
            zero_previous=True)
        result['arm']='head_disabled'; controls.append(result)
        profiler.mark('control_episode',case=case_id,success=result['success'])
    report['episodes']=episodes; report['controls']=controls
    report['episode_metrics']={'head_success':sum(e['success'] for e in episodes),
                               'head_count':len(episodes),
                               'control_success':sum(e['success'] for e in controls),
                               'control_count':len(controls)}
    report['decision']=('accepted_for_followup' if any(e['success'] for e in episodes)
                        and not any(e['success'] for e in controls) else
                        'inconclusive' if any(e['success'] for e in episodes) else 'rejected')
    report['deployment']='not_published'
    report['status']='completed'
    report['claims']='Head-conditioned action generation on synthetic mixed cases; proposal prior only; no task-completion or calibration claim beyond these episodes.'
    ledger.close()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_policy_head_v1.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source')
    p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':sha(a.protocol),'protocol':json.loads(a.protocol.read_text()),
            'downloads':0,'optimizer_updates':0,'checkpoints_saved':0}
    profiler=RunProfiler('mps',a.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start(); execute(a,profiler,report)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); raise
    finally:
        profile=profiler.stop(); report['profile']=profile
        mem=(profile or {}).get('memory') or {}
        report['costs']={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),
                         'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'decision':report.get('decision'),
                      'head':report.get('head_metrics'),'episodes':report.get('episode_metrics')}))
if __name__=='__main__': main()
