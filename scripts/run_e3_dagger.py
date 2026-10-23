#!/usr/bin/env python3
"""DAgger-style action aggregation on model-visited states (modules only)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[k]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from run_e3_qualification import (admit_policy, build_action_batch, canonical, run_episode,
    selection_for)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Collector:
    def __init__(self, proposer):self.proposer=proposer;self.requests=[]
    def __call__(self,request):self.requests.append(request);return self.proposer(request)
def failed(item):
    o=item.get('outcome') or {}; r=o.get('result') or {}; receipt=r.get('receipt') or {}
    code=receipt.get('exit_code') if isinstance(receipt,dict) else None
    return o.get('status')=='failed' or (code is not None and code!=0)
def tool_of(item):
    a=item.get('action') or {}; return a.get('tool') if a.get('kind')=='invoke_tool' else None
def teacher_action(tools, prior):
    read_done=any((x.get('action') or {}).get('kind')=='read' and (x.get('outcome') or {}).get('status')=='completed' for x in prior)
    tests=[x for x in prior if tool_of(x)=='run_test']
    if not read_done:return {'kind':'read','path':tools.workspace_entry}
    if not tests:return {'kind':'invoke_tool','tool':'run_test','arguments':{}}
    last=tests[-1]
    if not failed(last):return {'kind':'stop'}
    repaired=any(tool_of(x)=='set_constant' and (x.get('outcome') or {}).get('status')=='completed' for x in prior[tests.index(last)+1:])
    if repaired:return {'kind':'invoke_tool','tool':'run_test','arguments':{}}
    return {'kind':'invoke_tool','tool':'set_constant','arguments':{'name':tools.variant['constant'],'value':tools.variant['expected_value']}}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_dagger_v1.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source');p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e3_set_attempt03_20261023/selected_extension')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True); protocol=json.loads(a.protocol.read_text())
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'protocol':protocol,'downloads':0,'base_saved':False}
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    prof=RunProfiler('mps',a.output,sample_interval=0.1)
    try:
        with exclusive_mps_worker(ROOT):prof.start();execute(a,protocol,report,prof)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        profile=prof.stop();mem=(profile or {}).get('memory') or {}
        report.update(profile=profile,costs={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),
          'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes'),'sampled_peak_rss_bytes':mem.get('rss_peak_sampled_bytes')})
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'before':report.get('before_completions'),'after':report.get('after_completions'),'costs':report.get('costs')}))
def execute(args,protocol,report,prof):
    import torch
    from replica_lab.architecture.contracts import CoreSpec, TaskGraph
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.loop import SessionProposer
    from replica_lab.runtime.task_graph import Budget
    from replica_lab.runtime.tools import FixtureWorkspace
    torch.manual_seed(int(protocol['training']['seed']))
    core=CoreSpec(**protocol['core']);model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=prof,core=core)
    receipt=model.load_extension(args.parent_extension)
    report.update(manifest=model.manifest.to_dict(),manifest_digest=model.manifest.digest,source_lock=lock,parent_receipt=receipt)
    ledger=RecordLedger(args.output/'ledger.sqlite');subject='dagger';purpose='dagger'
    root=ROOT/protocol['fixtures']['root']; all_cases=protocol['fixtures']['train']+protocol['fixtures']['eval'] if 'eval' in protocol['fixtures'] else protocol['fixtures']['train']+protocol['fixtures']['dev']+protocol['fixtures']['heldout']
    for cid in all_cases:admit_policy(ledger,FixtureWorkspace(root/cid),subject=subject,branch=cid,purpose=purpose)
    session=MemorySession(model,tok,ledger,max_candidate_units=16)
    scopes={cid:EligibilityScope(subject,cid,purpose,0,'assistant_for_source',None) for cid in all_cases}
    eval_cases=protocol['evaluation']['cases'];bud=protocol['evaluation']
    budget=Budget(int(bud['max_decisions']),int(bud['max_tool_calls']),int(bud['max_decisions'])*int(bud['prompt_tokens']),int(bud['max_decisions'])*int(bud['generated_tokens']))
    def evaluate(label):
        out=[]
        for cid in eval_cases:
            r,_=run_episode(root/cid,args.output/label/cid,session=session,tokenizer=tok,mode='full',subject=subject,purpose=purpose,
              budget=budget,max_prompt_tokens=int(bud['prompt_tokens']),generated_token_budget=int(bud['generated_tokens']),zero_previous=True,
              mark=lambda name,**kw:prof.mark(name,**kw))
            out.append(r)
        return out
    model.eval(); report['before']=evaluate('before');report['before_completions']=f"{sum(x['success'] for x in report['before'])}/{len(report['before'])}"
    # DAgger rollouts on train; teacher labels are computed only from prior actual actions/receipts.
    examples=[]; rollouts=[]
    for cid in protocol['fixtures']['train']:
        tools=FixtureWorkspace(root/cid);scope=scopes[cid]
        proposer=Collector(SessionProposer(session,tok,task_id=cid,goal=tools.goal,scope=scope,device='mps',zero_previous_workspace=True))
        result,_=run_episode(root/cid,args.output/'rollout'/cid,session=session,tokenizer=tok,mode='full',subject=subject,purpose=purpose,
          budget=budget,max_prompt_tokens=int(bud['prompt_tokens']),generated_token_budget=int(bud['generated_tokens']),zero_previous=True,proposer=proposer)
        trace=result['summary']['trace']; rollouts.append({'case_id':cid,'success':result['success'],'decisions':len(trace),'requests':len(proposer.requests)})
        for i,request in enumerate(proposer.requests):
            action=teacher_action(tools,trace[:i]); raw=canonical(action)
            ids=tok(raw,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
            prompt_len=len(tok(request.prompt_text,add_special_tokens=False)['input_ids'])
            if prompt_len>int(protocol['training']['max_prompt_tokens']) or len(ids)>int(protocol['training']['max_label_tokens']):
                raise ValueError('DAgger example exceeds declared bounds')
            examples.append({'case_id':cid,'goal':tools.goal,'prompt_text':request.prompt_text,
              'target_text':raw,'target_token_ids':ids,'task_features':list(request.task_features),
              'task_digest':request.task_digest,'revision':request.revision,
              'task_graph':TaskGraph(task_id=cid,goal=tools.goal,state_digest=request.task_digest,revision=request.revision)})
    report['rollouts']=rollouts;report['examples']={'count':len(examples),'targets':[e['target_text'] for e in examples]}
    model.train();alg=protocol['training'];opt=build_optimizer(model,'muon8bit',lr_base=alg['lr_base'],lr_new=alg['lr_new'],lr_muon_base=alg['lr_muon_base'],lr_muon_new=alg['lr_muon_new'],weight_decay=alg['weight_decay'])
    if opt is None:raise RuntimeError('no trainable parameters')
    updates=[]
    for update in range(1,int(alg['updates'])+1):
        opt.zero_grad(set_to_none=True);total=0.0
        for off in range(int(alg['examples_per_update'])):
            e=examples[(2*(update-1)+off)%len(examples)];ids,mask,labels,prompt_tokens=build_action_batch(tok,e,'mps')
            with session.training_request(prompt_ids=ids[:,:prompt_tokens],prompt_mask=mask[:,:prompt_tokens],query=e['goal'],
                    scope=scopes[e['case_id']],task=e['task_graph'],selection=selection_for(session,scopes[e['case_id']],e['goal']),
                    task_features=e['task_features'],task_digest=e['task_digest'],revision=e['revision'],previous_core=None,
                    core_impl='recurrent',enabled_branches=('memory','workspace'),train_core=True,train_sense=True) as request:
                output=session.forward(request,ids,labels=labels,attention_mask=mask);loss=output.loss/int(alg['examples_per_update'])
                total+=float(loss.detach());session.backward(request,loss)
            del output,loss,ids,mask,labels
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],alg['gradient_clip'],error_if_nonfinite=True))
        ratio=min(1.0,update/int(alg['warmup_updates']))
        for group in opt.param_groups:group['lr']=ratio*(alg['lr_muon_new'] if group['algorithm']=='muon8bit' else alg['lr_new'])
        opt.step();opt.zero_grad(set_to_none=True);updates.append({'update':update,'loss':total,'gradient_norm':norm});prof.mark('dagger_update',update=update,loss=total)
    del opt;model.eval();report['updates']=updates
    ext=args.output/'selected_extension';receipt=model.save_extension(ext);report['extension']=receipt
    report['after']=evaluate('after');report['after_completions']=f"{sum(x['success'] for x in report['after'])}/{len(report['after'])}"
    report['deployment']='opt_in_diagnostic';report['status']='completed';report['claims']='DAgger action diagnostic; no general capability or default promotion.'
    ledger.close()
if __name__=='__main__':main()
