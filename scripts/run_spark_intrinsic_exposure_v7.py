#!/usr/bin/env python3
"""Review, admit and supervise one separately authorized official-BF16 v7 exposure arm.

No default launch, source acquisition or retry. Main and restart workers are
sequential descendants of one externally supervised process group. The same
durable reservation ledger and host guard span both workers.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger, atomic
from replica_lab.intrinsic_memory_execution_v7 import (validate_bindings,validate_authorization,
    bound_json,hash_file,contained)
from replica_lab.capacity_guard import CapacityGuard,CapacityPolicy,parse_host_readings
from replica_lab.wall_deadline_v1 import run_wall_bounded,WallPolicy
from launch_spark_intrinsic_readout_environment_v1 import validate_environment


def environment_only(packet_path):
    packet=json.loads(packet_path.read_text());protocol=bound_json(ROOT,packet['protocol'])
    return validate_environment(bound_json(ROOT,protocol['environment']),root=ROOT)


def host_reading():
    def read(argv):return subprocess.run(argv,capture_output=True,text=True,check=True,timeout=2).stdout
    return parse_host_readings(read(['/usr/sbin/sysctl','-n','vm.swapusage']),read(['/usr/bin/memory_pressure','-Q']))


@contextmanager
def exclusive_worker():
    # Same lock inode as all existing Replica native loaders, held by the parent
    # across admission, main exit and restart. No second worker can overlap.
    path=ROOT/'runs/.D2_mps_worker.lock';path.parent.mkdir(exist_ok=True,parents=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
    finally:os.close(fd)


def artifact_bytes(output):return sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


def supervise_worker(mode,args,output,ledger,guard):
    command=[sys.executable,str(Path(__file__).resolve()),'--packet',str(args.packet),
             '--authorization',str(args.authorization),'--worker',mode]
    heartbeat=output/(mode+'.telemetry.jsonl');samples=output/'host_samples.jsonl'
    offset=0;tail='';last_driver=time.monotonic();next_host=last_driver
    with (output/(mode+'.stdout')).open('x') as stdout,(output/(mode+'.stderr')).open('x') as stderr,samples.open('a') as host:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=os.environ.copy())
        try:
            while True:
                now=time.monotonic()
                if heartbeat.exists():
                    with heartbeat.open() as stream:
                        stream.seek(offset);tail+=stream.read();offset=stream.tell()
                    lines=tail.split('\n');tail=lines.pop()
                    for line in lines:
                        guard.observe_profile(json.loads(line));last_driver=now
                if now>=next_host:
                    reading=host_reading();guard.observe_host(reading)
                    host.write(json.dumps({'monotonic':now,**reading})+'\n');host.flush();next_host=now+2
                doc=json.loads(ledger.path.read_text())
                stages=[v for v in doc['stages'].values() if v['status']=='active']
                reason=guard.stop_reason
                if now>=doc['deadline'] or any(now>=s['deadline'] for s in stages):reason='stage_or_outer_deadline'
                if now-last_driver>30:reason='driver_telemetry_missing_or_stale'
                if artifact_bytes(output)>doc['reserved']['artifact_bytes']:reason='unreserved_artifact_growth'
                if reason:raise RuntimeError(reason)
                if child.poll() is not None:break
                time.sleep(0.25)
            if tail.strip():raise RuntimeError('incomplete telemetry record')
            if child.returncode!=0:raise RuntimeError(mode+' worker failed: '+str(child.returncode))
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
            atomic(output/'capacity_guard.json',{**guard.report(),'artifact_bytes_observed':artifact_bytes(output),
                'driver_sample_seconds':0.25,'host_sample_seconds':2,
                'physical_memory_note':'MPS driver and RSS overlap; host free/swap are system-wide sampled observations'})


def supervised(args):
    packet=json.loads(args.packet.read_text());protocol=bound_json(ROOT,packet['protocol']);arm=protocol['arms'][packet['arm']]
    if os.environ.get('REPLICA_INTRINSIC_V7_SUPERVISED')!=hash_file(args.packet):
        raise ValueError('outer supervisor missing')
    validate_authorization(ROOT,args.packet,args.authorization,allow_existing=True)
    output=contained(ROOT,packet['output_root']);deadline=float(os.environ['REPLICA_INTRINSIC_V7_DEADLINE'])
    ledger=WorkLedger.create(output/'work_ledger.json',identity=hash_file(args.packet),plan=arm['budget'],deadline=deadline)
    ledger.enter('admission')
    # Pre-reserve the audit/store/log envelope for the entire owned tree. Large
    # checkpoint output is separately reserved just before serialization.
    with ledger.work('admission','bounded_audit_and_store_envelope',artifact_bytes=arm['budget']['admission']['artifact_bytes'],writes=1):pass
    guard=CapacityGuard(CapacityPolicy(),host_reading())
    if guard.stop_reason:raise RuntimeError(guard.stop_reason)
    with ledger.work('admission','environment_source_code_data_preflight') as cost:
        environment=environment_only(args.packet)
        _,_,receipt=validate_bindings(ROOT,args.packet,check_source=True)
        cost.update(source_bytes_hashed=receipt['source_bytes_hashed'])
        atomic(output/'admission.json',{'environment':environment,**receipt})
    ledger.leave('admission',status='completed');ledger.enter('load')
    supervise_worker('main',args,output,ledger,guard)
    main=json.loads((output/'main_result.json').read_text())
    if not main.get('original_weights_unchanged'):raise RuntimeError('original integrity not established')
    if main.get('restart_requested'):
        ledger.enter('restart');supervise_worker('restart',args,output,ledger,guard)
    restart=json.loads((output/'restart_result.json').read_text()) if (output/'restart_result.json').exists() else None
    atomic(output/'result.json',{'source_kind':'official_bf16','arm':packet['arm'],
        'packet_sha256':hash_file(args.packet),'protocol_sha256':packet['protocol']['sha256'],'main':main,'restart':restart,
        'capacity':guard.report(),'reservations':ledger.snapshot()['reserved'],
        'native_competence':main['competence_passed'],
        'useful_recall':bool(main['useful_recall_passed'] and restart and restart['fresh_process_logits_exact']),
        'persistent_learning':False,'RSI':False,'promotion':False})


def worker(args):
    packet=json.loads(args.packet.read_text());output=contained(ROOT,packet['output_root'])
    if os.environ.get('REPLICA_INTRINSIC_V7_SUPERVISED')!=hash_file(args.packet):raise ValueError('supervision missing')
    validate_authorization(ROOT,args.packet,args.authorization,allow_existing=True)
    environment_only(args.packet)
    # Recheck after process creation and before importing torch, trusted remote
    # model/configuration code or constructing original weights.
    _,protocol,receipt=validate_bindings(ROOT,args.packet,check_source=True)
    ledger=WorkLedger(output/'work_ledger.json',identity=hash_file(args.packet))
    stage='load' if args.worker=='main' else 'restart'
    with ledger.work(stage,'child_source_revalidation') as event:event['source_bytes_hashed']=receipt['source_bytes_hashed']
    import torch
    import psutil
    stop=threading.Event()
    def telemetry():
        with (output/(args.worker+'.telemetry.jsonl')).open('x') as stream:
            while not stop.is_set():
                sample={'monotonic':time.monotonic(),'pid':os.getpid(),
                        'rss_bytes':psutil.Process().memory_info().rss,
                        'mps_driver_bytes':int(torch.mps.driver_allocated_memory()),
                        'mps_active_bytes':int(torch.mps.current_allocated_memory())}
                stream.write(json.dumps(sample)+'\n');stream.flush();stop.wait(0.25)
    heartbeat=threading.Thread(target=telemetry,daemon=True);heartbeat.start()
    try:
        from replica_lab.intrinsic_memory_native_v7 import load_native,run_main,run_restart
        path,backend,spec,base=load_native(ROOT,protocol,packet['arm'],ledger,stage)
        if args.worker=='main':
            ledger.leave('load',status='completed')
            run_main(ROOT,protocol,packet['arm'],ledger,output,path,backend,spec,base)
        else:run_restart(ROOT,protocol,packet['arm'],ledger,output,path,backend,spec,base)
    except BaseException as error:
        atomic(output/(args.worker+'.failure.json'),{'type':type(error).__name__,'message':str(error),
              'reserved_work_not_refunded':True})
        raise
    finally:stop.set();heartbeat.join(timeout=2)


def launch(args):
    packet=json.loads(args.packet.read_text());protocol=bound_json(ROOT,packet['protocol'])
    receipt=validate_authorization(ROOT,args.packet,args.authorization)
    arm=protocol['arms'][packet['arm']];output=contained(ROOT,packet['output_root'])
    with exclusive_worker():
        # Consumption is a fresh directory under a bound identity, never a
        # resettable receipt flag. Failure in preflight still consumes the attempt.
        output.mkdir(parents=True,exist_ok=False)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
             'PYTORCH_ENABLE_MPS_FALLBACK':'0','HF_MODULES_CACHE':str(output/'hf_modules'),
             'REPLICA_INTRINSIC_V7_SUPERVISED':hash_file(args.packet),
             'REPLICA_INTRINSIC_V7_DEADLINE':str(time.monotonic()+arm['wall_seconds']-11)}
        command=[sys.executable,str(Path(__file__).resolve()),'--packet',str(args.packet),
                 '--authorization',str(args.authorization),'--supervised']
        result=run_wall_bounded(command,cwd=ROOT,output=output/'supervision',
            policy=WallPolicy(arm['wall_seconds']),launch_authorized=True,
            authorization_identity=hash_file(args.authorization),env=env)
        return result.exit_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--authorization',type=Path)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--environment-only',action='store_true');modes.add_argument('--review',action='store_true')
    modes.add_argument('--launch',action='store_true');modes.add_argument('--supervised',action='store_true')
    modes.add_argument('--worker',choices=('main','restart'))
    args=parser.parse_args();args.packet=args.packet.resolve()
    if args.environment_only:print(json.dumps(environment_only(args.packet),indent=2));return 0
    if args.review:
        environment=environment_only(args.packet);_,_,receipt=validate_bindings(ROOT,args.packet)
        print(json.dumps({'environment':environment,**receipt,'launch_authorized':False},indent=2));return 0
    if args.authorization is None:parser.error('an exact external authorization receipt is required')
    args.authorization=args.authorization.resolve()
    if args.launch:return launch(args)
    if args.supervised:supervised(args);return 0
    worker(args);return 0


if __name__=='__main__':raise SystemExit(main())
