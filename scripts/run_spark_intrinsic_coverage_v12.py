#!/usr/bin/env python3
"""Review, admit and supervise one separately authorized official-BF16 v12 arm.

No default launch, source acquisition or retry. Main and restart workers are
sequential descendants of one externally supervised process group. The same
durable v12 reservation journal and host guard span both workers, so the
absolute outer deadline and every charged unit survive the restart.

Modes:

    --environment-only  verify the qualified interpreter/environment only
    --review            verify code, source, data and the comparison; no launch
    --launch            exactly-once authorized execution (guarded; unexecuted)
    --supervised        internal: parent of the supervised workers
    --worker main|restart  internal: one model process

The preparation script writes no authorization receipt. A launch requires an
external receipt whose bytes bind this packet, this protocol, this
implementation closure, this environment manifest and the source-feature
binding.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from replica_lab.intrinsic_memory_budget_v12 import WorkLedger, atomic, audit_io
from replica_lab.intrinsic_memory_coverage_v12 import (
    CONTROL_READ_BYTES_BOUND, ENVIRONMENT_CLOSURE_BYTES, SOURCE_FILE_BYTES)
from replica_lab.intrinsic_memory_execution_v12 import (
    charged_write, observe_bound_reads, record_attempt, validate_authorization,
    validate_bindings)
from replica_lab.intrinsic_memory_execution_v5 import bound_json, contained, hash_file
from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy, parse_host_readings
from replica_lab.wall_deadline_v1 import run_wall_bounded, WallPolicy
from launch_spark_intrinsic_readout_environment_v1 import validate_environment

ENV_PACKET = 'REPLICA_INTRINSIC_V12_PACKET'
ENV_DEADLINE = 'REPLICA_INTRINSIC_V12_DEADLINE'


def environment_only(packet_path):
    packet = json.loads(Path(packet_path).read_text())
    protocol = bound_json(ROOT, packet['protocol'])
    return validate_environment(bound_json(ROOT, protocol['environment']), root=ROOT)


def host_reading():
    def read(argv):
        return subprocess.run(argv, capture_output=True, text=True, check=True,
                              timeout=2).stdout
    return parse_host_readings(read(['/usr/sbin/sysctl', '-n', 'vm.swapusage']),
                               read(['/usr/bin/memory_pressure', '-Q']))


@contextmanager
def exclusive_worker():
    # Same lock inode as all existing Replica native loaders, held by the parent
    # across admission, main exit and restart. No second worker can overlap.
    path = ROOT/'runs/.D2_mps_worker.lock'; path.parent.mkdir(exist_ok=True, parents=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); yield
    finally:
        os.close(fd)


def artifact_bytes(output):
    return sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


def supervise_worker(mode, args, output, ledger, guard, capacity_stage):
    command = [sys.executable, str(Path(__file__).resolve()), '--packet', str(args.packet),
               '--authorization', str(args.authorization), '--worker', mode]
    heartbeat = output/(mode+'.telemetry.jsonl'); samples = output/'host_samples.jsonl'
    offset = 0; tail = ''; last_driver = time.monotonic(); next_host = last_driver
    with (output/(mode+'.stdout')).open('x') as stdout, \
            (output/(mode+'.stderr')).open('x') as stderr, samples.open('a') as host:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 env=os.environ.copy())
        try:
            while True:
                now = time.monotonic()
                if heartbeat.exists():
                    with heartbeat.open() as stream:
                        stream.seek(offset); tail += stream.read(); offset = stream.tell()
                    lines = tail.split('\n'); tail = lines.pop()
                    for line in lines:
                        guard.observe_profile(json.loads(line)); last_driver = now
                if now >= next_host:
                    reading = host_reading(); guard.observe_host(reading)
                    host.write(json.dumps({'monotonic': now, **reading})+'\n'); host.flush()
                    next_host = now+2
                doc = ledger.snapshot(cached=True)
                deadlines = ledger.deadline_state(doc)
                reason = guard.stop_reason
                if (deadlines['outer_remaining_seconds'] <= 0
                        or any(v <= 0 for v in deadlines['stage_remaining_seconds'].values())):
                    reason = 'stage_or_outer_deadline'
                if now-last_driver > 30: reason = 'driver_telemetry_missing_or_stale'
                if artifact_bytes(output) > doc['reserved']['artifact_bytes']:
                    reason = 'unreserved_artifact_growth'
                if reason: raise RuntimeError(reason)
                if child.poll() is not None: break
                time.sleep(0.25)
            if tail.strip(): raise RuntimeError('incomplete telemetry record')
            if child.returncode != 0:
                raise RuntimeError(mode+' worker failed: '+str(child.returncode))
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=2)
            # Each supervision window has its own declared stage and its own
            # receipt path, so a later window never overwrites earlier evidence
            # and no window depends on a stage another receipt consumed.
            entered = False
            try:
                ledger.enter(capacity_stage); entered = True
            except Exception:
                entered = False
            charged_write(ledger, output, 'capacity_guard_'+mode, {
                **guard.report(), 'artifact_bytes_observed': artifact_bytes(output),
                'driver_sample_seconds': 0.25, 'host_sample_seconds': 2,
                'supervision_window': mode,
                'physical_memory_note': ('MPS driver and RSS overlap; host free/swap are '
                                         'system-wide sampled observations')},
                stage=capacity_stage if entered else None)
            if entered:
                try: ledger.leave(capacity_stage, status='completed')
                except Exception: pass


def enter_stage(ledger, stage):
    """Enter a declared parent stage, refusing when another one is active."""
    ledger.enter(stage)
    return stage


def supervised(args):
    packet = json.loads(args.packet.read_text())
    protocol = bound_json(ROOT, packet['protocol']); arm = protocol['arms'][packet['arm']]
    if os.environ.get(ENV_PACKET) != hash_file(args.packet):
        raise ValueError('outer supervisor missing')
    validate_authorization(ROOT, args.packet, args.authorization, allow_existing=True)
    output = contained(ROOT, packet['output_root'])
    deadline = float(os.environ[ENV_DEADLINE])
    ledger = WorkLedger.create(output/'work_ledger.jsonl', identity=hash_file(args.packet),
                               plan=arm['budget'], deadline=deadline)
    ledger.enter('admission')
    # Pre-reserve the audit/store/log envelope for the entire owned tree. Large
    # checkpoint output is separately reserved just before serialization.
    with ledger.work('admission', 'bounded_audit_and_store_envelope',
                     artifact_bytes=arm['budget']['admission']['artifact_bytes'],
                     writes=1):
        pass
    guard = CapacityGuard(CapacityPolicy(), host_reading())
    if guard.stop_reason: raise RuntimeError(guard.stop_reason)
    with ledger.work('admission', 'environment_only',
                     artifact_read_bytes=CONTROL_READ_BYTES_BOUND):
        environment = environment_only(args.packet)
    with ledger.work('admission', 'admission_receipt', writes=1) as event:
        atomic(output/'admission.json', {'environment_validation': environment})
        event['actual'] = {'writes': 1}
    ledger.leave('admission', status='completed'); ledger.enter('preflight')
    observed = []
    with observe_bound_reads(lambda path, count: observed.append(count)):
        with ledger.work('preflight', 'environment_source_code_data_preflight',
                         artifact_read_bytes=(CONTROL_READ_BYTES_BOUND
                                             + SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES)) as cost:
            _, _, receipt = validate_bindings(ROOT, args.packet, check_source=True)
            cost['actual'] = {'artifact_read_bytes':
                              sum(observed) + receipt['source_bytes_hashed']
                              + receipt['environment_closure']['bytes_verified']}
            cost['source_bytes_hashed'] = receipt['source_bytes_hashed']
            cost['environment_closure'] = receipt['environment_closure']
            cost['bound_json_bytes_observed'] = sum(observed)
            with ledger.work('preflight', 'preflight_receipt', writes=1) as event:
                atomic(output/'preflight.json',
                       {'environment_validation': environment, **receipt})
                event['actual'] = {'writes': 1}
    ledger.leave('preflight', status='completed'); ledger.enter('load')
    supervise_worker('main', args, output, ledger, guard, 'capacity_main')
    main = json.loads((output/'main_result.json').read_text())
    if not main.get('original_weights_unchanged'):
        raise RuntimeError('original integrity not established')
    if main.get('restart_requested'):
        ledger.enter('restart')
        supervise_worker('restart', args, output, ledger, guard, 'capacity_restart')
    restart = (json.loads((output/'restart_result.json').read_text())
               if (output/'restart_result.json').exists() else None)
    snapshot = ledger.snapshot()
    ledger.enter('finalize_parent')
    charged_write(ledger, output, 'result', {
        'source_kind': 'official_bf16', 'arm': packet['arm'],
        'mechanism': 'frozen_canonical_source_features_live_contextual_compression_v12',
        'packet_sha256': hash_file(args.packet),
        'protocol_sha256': packet['protocol']['sha256'], 'main': main, 'restart': restart,
        'capacity': guard.report(), 'reservations': snapshot['reserved'],
        'audit_io': audit_io(ledger),
        'deadline_state': ledger.deadline_state(snapshot),
        'native_competence': main['competence_passed'],
        'useful_recall': bool(main['useful_recall_passed'] and restart
                              and restart['fresh_process_logits_exact']),
        'persistent_learning': False, 'RSI': False, 'promotion': False},
        stage='finalize_parent')
    try:
        ledger.leave('finalize_parent', status='completed')
    except Exception as error:
        (output/'finalize_parent_leave_failure.json').write_text(
            json.dumps({'type': type(error).__name__, 'message': str(error)})+'\n')


def worker(args):
    packet = json.loads(args.packet.read_text()); output = contained(ROOT, packet['output_root'])
    if os.environ.get(ENV_PACKET) != hash_file(args.packet):
        raise ValueError('supervision missing')
    # Exact authorization plus the durable attempt marker that launch recorded.
    validate_authorization(ROOT, args.packet, args.authorization, allow_existing=True)
    ledger = WorkLedger(output/'work_ledger.jsonl', identity=hash_file(args.packet))
    stage = 'load' if args.worker == 'main' else 'restart'
    observed = []
    with observe_bound_reads(lambda path, count: observed.append(count)):
        with ledger.work(stage, 'child_source_revalidation',
                         artifact_read_bytes=(SOURCE_FILE_BYTES
                                              + ENVIRONMENT_CLOSURE_BYTES)) as event:
            environment_only(args.packet)
            _, protocol, receipt = validate_bindings(ROOT, args.packet, check_source=True)
            event['actual'] = {'artifact_read_bytes':
                               sum(observed) + receipt['source_bytes_hashed']
                               + receipt['environment_closure']['bytes_verified']}
            event['source_bytes_hashed'] = receipt['source_bytes_hashed']
            event['environment_closure'] = receipt['environment_closure']
            event['bound_json_bytes_observed'] = sum(observed)
    import torch
    import psutil
    stop = threading.Event()

    def telemetry():
        with (output/(args.worker+'.telemetry.jsonl')).open('x') as stream:
            while not stop.is_set():
                sample = {'monotonic': time.monotonic(), 'pid': os.getpid(),
                          'rss_bytes': psutil.Process().memory_info().rss,
                          'mps_driver_bytes': int(torch.mps.driver_allocated_memory()),
                          'mps_active_bytes': int(torch.mps.current_allocated_memory())}
                stream.write(json.dumps(sample)+'\n'); stream.flush(); stop.wait(0.25)

    heartbeat = threading.Thread(target=telemetry, daemon=True); heartbeat.start()
    try:
        from replica_lab.intrinsic_memory_native_v12 import load_and_run
        load_and_run(ROOT, protocol, packet['arm'], ledger, output,
                     stage=stage, mode=args.worker)
    except BaseException as error:
        # charged_write cannot raise, so this records the failure and re-raises
        # the original exception unchanged, including process-control exits.
        charged_write(ledger, output, args.worker+'.failure', {
            'type': type(error).__name__, 'message': str(error),
            'reserved_work_not_refunded': True})
        raise
    finally:
        stop.set(); heartbeat.join(timeout=2)


def launch(args):
    packet = json.loads(args.packet.read_text())
    protocol = bound_json(ROOT, packet['protocol'])
    validate_authorization(ROOT, args.packet, args.authorization)
    arm = protocol['arms'][packet['arm']]; output = contained(ROOT, packet['output_root'])
    marker = contained(ROOT, packet['attempt_marker'])
    if marker.resolve().is_relative_to(output.resolve()):
        raise ValueError('attempt marker must live outside the deletable output tree')
    with exclusive_worker():
        # The durable, append-only marker is written first: deleting the output
        # directory later cannot restore a consumed attempt, and the packet hash is
        # bound into it so a different authorization cannot reuse the slot.
        record_attempt(marker, packet_sha256=hash_file(args.packet),
                       protocol_sha256=packet['protocol']['sha256'], arm=packet['arm'],
                       output_root=packet['output_root'],
                       authorization_sha256=hash_file(args.authorization))
        # Failure in preflight still consumes the attempt; there are no retries.
        output.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT/'src'),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'PYTORCH_ENABLE_MPS_FALLBACK': '0',
               'HF_MODULES_CACHE': str(output/'hf_modules'),
               ENV_PACKET: hash_file(args.packet),
               ENV_DEADLINE: str(time.monotonic()+arm['wall_seconds']-11)}
        command = [sys.executable, str(Path(__file__).resolve()), '--packet', str(args.packet),
                   '--authorization', str(args.authorization), '--supervised']
        result = run_wall_bounded(command, cwd=ROOT, output=output/'supervision',
                                  policy=WallPolicy(arm['wall_seconds']),
                                  launch_authorized=True,
                                  authorization_identity=hash_file(args.authorization),
                                  env=env)
        return result.exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--environment-only', action='store_true')
    modes.add_argument('--review', action='store_true')
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--supervised', action='store_true')
    modes.add_argument('--worker', choices=('main', 'restart'))
    args = parser.parse_args(); args.packet = args.packet.resolve()
    if args.environment_only:
        print(json.dumps(environment_only(args.packet), indent=2)); return 0
    if args.review:
        environment = environment_only(args.packet)
        _, _, receipt = validate_bindings(ROOT, args.packet)
        print(json.dumps({'environment_validation': environment, **receipt,
                          'launch_authorized': False}, indent=2)); return 0
    if args.authorization is None:
        parser.error('an exact external authorization receipt is required')
    args.authorization = args.authorization.resolve()
    if args.launch: return launch(args)
    if args.supervised:
        supervised(args); return 0
    worker(args); return 0


if __name__ == '__main__':
    raise SystemExit(main())
