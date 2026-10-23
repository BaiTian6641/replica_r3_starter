#!/usr/bin/env python3
"""E3 core/feed-forward/extra-inference controls on fresh prompt-evidence cases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib,json,os,sys,traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[key]='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(args,profiler,report):
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope,RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.task_graph import Budget
    from replica_lab.runtime.tools import FixtureWorkspace
    from run_e3_qualification import admit_policy,run_episode
    protocol=json.loads(args.protocol.read_text())
    if protocol['source_revision']!='f97400052a43d642bbc6e9975e2397e3ae6a6b52': raise ValueError('source mismatch')
    model,tok,lock=load_local_mps(args.model_path,args.model_lock,profiler=profiler,
                                  core=CoreSpec(**protocol['core']))
    receipt=model.load_extension(Path(protocol['extension']))
    report.update(source_lock=lock,manifest_digest=model.manifest.digest,extension=receipt)
    fixtures=Path(protocol['fixtures']['root']); ledger=RecordLedger(args.output/'controls.sqlite')
    session=MemorySession(model,tok,ledger,max_candidate_units=64)
    tools_by_case={}; scopes={}
    for case_id in protocol['fixtures']['heldout']:
        tools=FixtureWorkspace(fixtures/case_id); tools_by_case[case_id]=tools
        admit_policy(ledger,tools,subject=protocol['subject'],branch=case_id,purpose=protocol['purpose'])
        scopes[case_id]=EligibilityScope(protocol['subject'],case_id,protocol['purpose'],0,
                                         'assistant_for_source',None)
    budget=Budget(int(protocol['episodes']['decisions']),int(protocol['episodes']['tool_calls']),
        int(protocol['episodes']['decisions'])*int(protocol['episodes']['prompt_tokens']),
        int(protocol['episodes']['decisions'])*int(protocol['episodes']['generated_tokens']))
    results=[]
    for mode in protocol['modes']:
        for case_id in protocol['fixtures']['heldout']:
            result,_=run_episode(fixtures/case_id,args.output/'episodes'/mode/case_id,
                session=session,tokenizer=tok,mode=mode,subject=protocol['subject'],
                purpose=protocol['purpose'],budget=budget,
                max_prompt_tokens=int(protocol['episodes']['prompt_tokens']),
                generated_token_budget=int(protocol['episodes']['generated_tokens']),
                zero_previous=True)
            result['arm']=mode; results.append(result)
            profiler.mark('core_control',mode=mode,case=case_id,success=result['success'])
    report['results']=results
    report['metrics']={mode:{'count':sum(r['arm']==mode for r in results),
        'success':sum(r['success'] for r in results if r['arm']==mode),
        'malformed':sum(r['malformed_actions'] for r in results if r['arm']==mode),
        'false_success_claims':sum(r['false_success_claims'] for r in results if r['arm']==mode)}
        for mode in protocol['modes']}
    report['status']='completed'
    report['claims']='Core/feed-forward/extra-inference controls on fresh synthetic prompt-evidence cases; no training or general-capability claim.'
    ledger.close()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',choices=['mps'],default='mps')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_mixed_core_controls_v1.json')
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
        with exclusive_mps_worker(ROOT): profiler.start(); execute(a,profiler,report)
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); raise
    finally:
        profile=profiler.stop(); report['profile']=profile; mem=(profile or {}).get('memory') or {}
        report['costs']={'profiled_wall_seconds':(profile or {}).get('elapsed_seconds'),'sampled_peak_driver_bytes':mem.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (a.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'metrics':report.get('metrics')}))
if __name__=='__main__': main()
