#!/usr/bin/env python3
"""Launch a pinned D2 stage offline, with one native MPS worker and no downloads."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STAGES={'capacity512_chunk01':0,'capacity512_chunk02':1,'tiny_chunk01':2,'tiny_chunk02':3}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,required=True)
    parser.add_argument('--run',action='store_true',help='Execute; default only prints the pinned command')
    args=parser.parse_args()
    plan_path=ROOT/'configs/d2_resume_plan_20261022.json'
    plan=json.loads(plan_path.read_text());stage=plan['runs'][STAGES[args.stage]]
    for name,expected in plan['bound_implementation_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Bound training implementation changed: '+name)
    source=ROOT/plan['source']['path']
    if source!=ROOT/'checkpoints/minicpm5_source' or not source.is_dir() or source.is_symlink():
        raise ValueError('Reuse the existing original source; this launcher never acquires weights')
    environment=dict(os.environ)
    environment.update(plan['environment'])
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:
        if environment[key]!='1':raise ValueError('Offline operation is mandatory')
    if environment['PYTORCH_ENABLE_MPS_FALLBACK']!='0':raise ValueError('CPU fallback is disabled')
    threshold=plan['minimum_free_disk_gib'][stage['kind']]*1024**3
    if shutil.disk_usage(ROOT).free<threshold:raise ValueError('Insufficient disk headroom')
    policy=plan['memory_guard']
    command=[sys.executable,str(ROOT/'scripts/run_capacity_preflight.py'),'--report',stage['guard_report'],
             '--driver-limit-gib',str(policy['driver_limit_gib']),'--swap-growth-limit-mib',str(policy['swap_growth_limit_mib']),
             '--minimum-free-percent',str(policy['minimum_free_percent']),'--sustained-host-samples',str(policy['sustained_host_samples']),
             '--',*stage['trainer_arguments']]
    if not args.run:
        print(json.dumps({'status':'plan_only','command':command,'environment':plan['environment']}));return
    report_path=ROOT/f'reports/D2_{args.stage}_launch_20261022.json'
    if report_path.exists() or (ROOT/stage['output']).exists() or (ROOT/stage['guard_report']).exists():
        raise ValueError('Use only fresh stage output/report paths')
    lockpath=ROOT/'runs/.D2_mps_worker.lock'
    lockfd=os.open(lockpath,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=importlib.util.spec_from_file_location('source_reuse',ROOT/'scripts/download_model.py')
        reuse=importlib.util.module_from_spec(spec);spec.loader.exec_module(reuse)
        verified=reuse.ensure_snapshot(ROOT/'configs/minicpm5_metadata_lock.local.json',source,sealed_lock=ROOT/'configs/minicpm5_sealed_lock.local.json')
        report={'status':'started','stage':args.stage,'started_at_utc':datetime.now(timezone.utc).isoformat(),
                'command':command,'offline_environment':plan['environment'],'source_reuse':verified,
                'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'free_disk_before_bytes':shutil.disk_usage(ROOT).free,'downloads':0}
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        try:
            outcome=subprocess.run(command,cwd=ROOT,env=environment,check=False)
            report['exit_code']=outcome.returncode
            report['status']='completed' if outcome.returncode==0 else 'failed'
        except BaseException as error:
            report.update(status='interrupted_or_failed',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
            report['free_disk_after_bytes']=shutil.disk_usage(ROOT).free
            report_path.write_text(json.dumps(report,indent=2)+'\n')
        if outcome.returncode:raise SystemExit(outcome.returncode)
    finally:
        os.close(lockfd)


if __name__=='__main__':main()
