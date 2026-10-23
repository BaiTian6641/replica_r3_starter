#!/usr/bin/env python3
"""Source-grounded answer selection by model candidate scoring (no training)."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,traceback
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[k]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['mps'],default='mps');p.add_argument('--protocol',type=Path,default=ROOT/'configs/e2_candidate_answer_v1.json')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--model-path',type=Path,default=ROOT/'checkpoints/minicpm5_source');p.add_argument('--model-lock',type=Path,default=ROOT/'configs/minicpm5_sealed_lock.local.json')
 p.add_argument('--parent-extension',type=Path,default=ROOT/'runs/e3_set_attempt03_20261023/selected_extension');p.add_argument('--selection',choices=['model','random'],default='model');p.add_argument('--seed',type=int,default=20261023)
 p.add_argument('--candidate-order',choices=['canonical','shuffled'],default='canonical')
 p.add_argument('--candidate-context',choices=['menu','independent'],default='menu')
 a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
 a.output.mkdir(parents=True,exist_ok=True);protocol=json.loads(a.protocol.read_text())
 report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'protocol':protocol,'selection_mode':a.selection,'candidate_order':a.candidate_order,'candidate_context':a.candidate_context,'training_updates':0}
 from replica_lab.profiling import RunProfiler
 from replica_lab.architecture.loading import exclusive_mps_worker
 prof=RunProfiler('mps',a.output,sample_interval=0.1)
 try:
  with exclusive_mps_worker(ROOT):prof.start();execute(a,protocol,report,prof)
 except BaseException as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
 finally:
  profile=prof.stop();mem=(profile or {}).get('memory') or {};report.update(profile=profile,costs={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes'),'sampled_peak_rss_bytes':mem.get('rss_peak_sampled_bytes')});report['ended_at_utc']=datetime.now(timezone.utc).isoformat();(a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'accuracy':report.get('accuracy'),'costs':report.get('costs')}))
def execute(args,protocol,report,prof):
 import torch,random
 from replica_lab.architecture.contracts import CoreSpec,TaskGraph
 from replica_lab.architecture.loading import encode_prompt,load_local_mps
 from replica_lab.memory_system import EligibilityScope,MemoryRecord,RecordLedger
 from replica_lab.memory_system.session import MemorySession
 from replica_lab.runtime.candidate_actions import CandidateReranker,canonical_action
 core=CoreSpec(**protocol['core']);model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=prof,core=core)
 receipt=model.load_extension(args.parent_extension);report.update(manifest=model.manifest.to_dict(),manifest_digest=model.manifest.digest,source_lock=lock,parent_receipt=receipt)
 root=ROOT/protocol['dataset']['root'];records=[MemoryRecord(**r) for r in rows(root/'records.jsonl')];queries=rows(root/'queries.jsonl');ledger=RecordLedger(args.output/'ledger.sqlite')
 for r in records:ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
 session=MemorySession(model,tok,ledger,max_candidate_units=16);results=[];rng=random.Random(args.seed)
 for q in queries:
  scope=EligibilityScope(q['subject_id'],q['branch_id'],q['purpose'],q['as_of'],'assistant_for_source',None)
  sel=session.index.retrieve(q['query'],**asdict(scope),limit=4);session.index.verify_selection(sel,expected_scope=scope)
  source='\n'.join(f"- {r.text}" for r in sel.records)
  candidates=[{'kind':'answer','text':x} for x in q['candidates']]
  if args.candidate_order=='shuffled':rng.shuffle(candidates)
  if args.candidate_context=='menu':
   prompt=(f"SOURCE RECORDS:\n{source}\n\nQUESTION: {q['query']}\n\n"
           "CANDIDATE ANSWERS (host enumerates; model scores or random control selects):\n"+
           '\n'.join('- '+canonical_action(c) for c in candidates))
  else:
   prompt=f"SOURCE RECORDS:\n{source}\n\nQUESTION: {q['query']}\n"
  ids=encode_prompt(tok,prompt,'mps')
  if ids.shape[1]>int(protocol['evaluation']['prompt_tokens']):raise ValueError('prompt exceeds budget')
  task=TaskGraph(q['case_id'],q['query'],state_digest=q['case_id'],revision=0)
  features=torch.zeros(1,protocol['core']['task_features'],dtype=torch.float32,device='mps')
  if args.selection=='model':
   picked=CandidateReranker(session,tok).select(prompt_ids=ids,prompt_mask=torch.ones_like(ids,dtype=torch.bool),candidates=candidates,query=q['query'],scope=scope,task=task,task_features=features,task_digest=q['case_id'],revision=0,selection=sel)
   chosen=picked.chosen;index=picked.chosen_index;scores=list(picked.scores)
  else:
   index=rng.randrange(len(candidates));chosen=candidates[index];scores=[]
  answer=chosen['text'];correct=answer==q['expected_answer'];results.append({'case_id':q['case_id'],'expected':q['expected_answer'],'chosen':answer,'correct':correct,'index':index,'scores':scores,'source_ids':[r.record_id for r in sel.records]});prof.mark('candidate_answer',case=q['case_id'],correct=correct,chosen=index)
 report['results']=results;report['accuracy']=f"{sum(x['correct'] for x in results)}/{len(results)}";report['status']='completed';report['claims']='Answer-selection interface only; no training or free-form claim.';ledger.close()
if __name__=='__main__':main()
