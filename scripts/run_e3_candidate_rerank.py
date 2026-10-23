#!/usr/bin/env python3
"""Model-scored action-candidate reranking (no training)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[k]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_e3_qualification import admit_policy, canonical, component_binding, initial_state
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['mps'],default='mps')
 p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_candidate_rerank_v1.json');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source');p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
 p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e3_set_attempt03_20261023/selected_extension')
 p.add_argument('--selection',choices=['model','random'],default='model')
 p.add_argument('--seed',type=int,default=20261023)
 a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
 a.output.mkdir(parents=True,exist_ok=True);protocol=json.loads(a.protocol.read_text())
 report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'protocol':protocol,'selection_mode':a.selection,'seed':a.seed,'downloads':0,'training_updates':0}
 from replica_lab.profiling import RunProfiler
 from replica_lab.architecture.loading import exclusive_mps_worker
 prof=RunProfiler('mps',a.output,sample_interval=0.1)
 try:
  with exclusive_mps_worker(ROOT):prof.start();execute(a,protocol,report,prof)
 except BaseException as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
 finally:
  profile=prof.stop();mem=(profile or {}).get('memory') or {};report.update(profile=profile,costs={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes'),'sampled_peak_rss_bytes':mem.get('rss_peak_sampled_bytes')})
  report['ended_at_utc']=datetime.now(timezone.utc).isoformat();(a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'completions':report.get('completions'),'costs':report.get('costs')}))
def menu(tools,state):
 cands=[{'kind':'read','path':tools.workspace_entry},{'kind':'invoke_tool','tool':'run_test','arguments':{}},{'kind':'stop'}]
 # Only observed failure text and current constants construct candidates.
 text=''
 for outcome in state.outcomes:
  if outcome.status not in ('failed','completed'):continue
  result=dict(outcome.result);receipt=dict(outcome.receipt or {})
  if result.get('kind')=='verify_success' or 'receipt' not in result and not receipt:continue
  text=(receipt.get('stderr_excerpt') or '')+'\n'+(receipt.get('stdout_excerpt') or '')+'\n'+(result.get('stderr_excerpt') or '')+'\n'+(result.get('stdout_excerpt') or '')
  if text.strip():break
 if text.strip():
  try:
   from replica_lab.runtime.procedures import parse_failure_bound
   b=parse_failure_bound(text);name=b['name'];expected=float(b['expected']);actual=float(b['actual'])
   for value in (expected,actual,expected+max(1.0,abs(expected)*0.1)):
    cands.append({'kind':'invoke_tool','tool':'set_constant','arguments':{'name':name,'value':value}})
  except ValueError:pass
 # A second legal constant value prevents a single-option repair menu.
 try:
  for item in tools.constant_candidates():
   if item.get('name') and isinstance(item.get('value'),(int,float)):
    cands.append({'kind':'invoke_tool','tool':'set_constant','arguments':{'name':item['name'],'value':float(item['value'])}})
 except Exception:pass
 out=[];seen=set()
 for c in cands:
  raw=canonical(c)
  if raw not in seen:seen.add(raw);out.append(c)
 return out
class OneAction:
 def __init__(self,raw,prompt_tokens,target_ids):self.raw=raw;self.prompt_tokens=prompt_tokens;self.target_ids=target_ids
 def __call__(self,request):
  from replica_lab.runtime.loop import ProposalOutput
  return ProposalOutput(raw_text=self.raw,prompt_tokens=self.prompt_tokens,generated_token_ids=tuple(self.target_ids),generated_tokens=len(self.target_ids),frame_digest=hashlib.sha256(self.raw.encode()).hexdigest(),workspace=None,workspace_stamp=None,wall_seconds=0.0,notes=('model_scored_candidate',))
def execute(args,protocol,report,prof):
 import torch
 from replica_lab.architecture.contracts import CoreSpec, TaskGraph
 from replica_lab.architecture.loading import encode_prompt,load_local_mps
 from replica_lab.memory_system import EligibilityScope,RecordLedger
 from replica_lab.memory_system.session import MemorySession
 from replica_lab.runtime.journal import TaskJournal
 from replica_lab.runtime.loop import TaskRuntime
 from replica_lab.runtime.task_graph import Budget
 from replica_lab.runtime.tools import FixtureWorkspace
 from replica_lab.runtime.workspace_store import WorkspaceStore
 core=CoreSpec(**protocol['core']);model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=prof,core=core)
 receipt=model.load_extension(args.parent_extension);report.update(manifest=model.manifest.to_dict(),manifest_digest=model.manifest.digest,source_lock=lock,parent_receipt=receipt)
 ledger=RecordLedger(args.output/'ledger.sqlite');subject='candidate-rerank';purpose='candidate-rerank';root=ROOT/protocol['fixtures']['root'];cases=protocol['evaluation']['cases']
 for cid in protocol['fixtures']['dev']+protocol['fixtures']['heldout']:admit_policy(ledger,FixtureWorkspace(root/cid),subject=subject,branch=cid,purpose=purpose)
 session=MemorySession(model,tok,ledger,max_candidate_units=16);episodes=[]
 for cid in cases:
  tools=FixtureWorkspace(root/cid);scope=EligibilityScope(subject,cid,purpose,0,'assistant_for_source',None)
  budget=Budget(int(protocol['evaluation']['max_decisions']),int(protocol['evaluation']['max_tool_calls']),int(protocol['evaluation']['max_decisions'])*int(protocol['evaluation']['prompt_tokens']),int(protocol['evaluation']['max_decisions'])*int(protocol['evaluation']['generated_tokens']))
  work=args.output/'episodes'/cid;work.mkdir(parents=True);import shutil;shutil.copytree(root/cid,work/'case')
  tools=FixtureWorkspace(work/'case');journal=TaskJournal.open(work/'task.sqlite',component_binding(tools,subject=subject,branch=cid,purpose=purpose));store=WorkspaceStore(work/'workspace')
  runtime=TaskRuntime(tools,journal,workspace_store=store,mode='full',max_prompt_tokens=int(protocol['evaluation']['prompt_tokens']),generated_token_budget=int(protocol['evaluation']['generated_tokens']),budget=budget)
  runtime.initialize(initial_state(tools,subject=subject,branch=cid,purpose=purpose,budget=budget))
  traces=[];score_rows=[];rng=__import__('random').Random(args.seed)
  for step in range(int(protocol['evaluation']['max_decisions'])):
   if runtime.state.is_complete(runtime.verifier):break
   cands=menu(tools,runtime.state)
   prompt=runtime._build_prompt(runtime.state)+'\nCANDIDATE ACTIONS (host enumerates; model scores or random control selects):\n'+'\n'.join('- '+canonical(c) for c in cands)
   ids=encode_prompt(tok,prompt,'mps')
   if ids.shape[1]>int(protocol['evaluation']['prompt_tokens']):raise ValueError('candidate prompt exceeds budget')
   scores=[]
   if args.selection=='model':
    sel=session.index.retrieve(tools.goal,**asdict(scope),limit=1)
    task=TaskGraph(cid,tools.goal,state_digest=runtime.state.digest,revision=runtime.state.revision)
    request=session.prepare(prompt_ids=ids,prompt_mask=torch.ones_like(ids,dtype=torch.bool),query=tools.goal,scope=scope,task=task,selection=sel,task_features=runtime.state.features(),task_digest=runtime.state.digest,revision=runtime.state.revision,previous_core=None,core_impl='recurrent',enabled_branches=('memory','workspace'))
    try:
     with torch.no_grad():
      for c in cands:
       raw=canonical(c);target=tok(raw,add_special_tokens=False)['input_ids']+[tok.eos_token_id];t=torch.tensor([target],dtype=torch.long,device='mps');all_ids=torch.cat((ids,t),dim=1);mask=torch.ones_like(all_ids,dtype=torch.bool);labels=all_ids.clone();labels[:,:ids.shape[1]]=-100
       out=session.forward(request,all_ids,labels=labels,attention_mask=mask);scores.append(float(out.loss.detach()))
    finally:request.release()
    best=min(range(len(cands)),key=lambda i:(scores[i],i))
   else:
    best=rng.randrange(len(cands))
   chosen=cands[best];raw=canonical(chosen);score_rows.append({'step':len(traces)+1,'candidates':cands,'scores':scores,'chosen':best});prof.mark('candidate_choice',case=cid,step=len(traces)+1,chosen=best,score=scores[best] if scores else None)
   target=tok(raw,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
   trace=runtime.decide(OneAction(raw,int(ids.shape[1]),target));traces.append(trace)
   if runtime.state.is_complete(runtime.verifier):break
  success=bool(runtime.state.is_complete(runtime.verifier));episodes.append({'case_id':cid,'role':tools.role,'success':success,'decisions':len(traces),'tool_calls':runtime.state.budget.used_tool_calls,'prompt_tokens':runtime.state.budget.used_prompt_tokens,'generated_tokens':runtime.state.budget.used_generated_tokens,'false_success_claims':sum(1 for t in traces if t.success_claimed and not t.success_condition_met),'actions':[(t.action or {}).get('kind') for t in traces],'tools':[((t.action or {}).get('args') or {}).get('tool') for t in traces],'candidate_scores':score_rows,'trace':[asdict(t) for t in traces],'evaluator_pass':tools.verify_success().get('passed')});journal.close()
 report['episodes']=episodes;report['completions']=f"{sum(x['success'] for x in episodes)}/{len(episodes)}";report['heldout_completions']=f"{sum(x['success'] for x in episodes if x['role']=='heldout')}/{sum(1 for x in episodes if x['role']=='heldout')}";report['status']='completed';report['claims']='Model-scored candidate action interface; no training, no default promotion.';ledger.close()
if __name__=='__main__':main()
