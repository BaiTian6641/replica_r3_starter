#!/usr/bin/env python3
"""M3 fresh-process extension/depth-image restart qualification on MPS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from run_m2_dynamic import DELAYED_USE_TEMPLATE, generate, rows, sha


def execute(args, profiler, report):
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.fast_memory import (FastMemoryConfig, FastMemoryScope,
                                                  load_fast_depth_image)
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    implementation_pins = protocol.get('implementation_sha256', {})
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M3 implementation bytes changed: {name}')
    report['implementation_sha256'] = implementation_pins
    if protocol.get('format') != 'replica.m3.depth.v1' or args.depth_arm not in protocol['depth_arms']:
        raise ValueError('M3 restart requires a declared M3 depth arm')
    candidate_report_path = args.candidate_run / 'run_report.json'
    candidate_protocol_path = args.protocol
    baseline = protocol.get('baseline_arms', {}).get(args.depth_arm)
    if baseline is not None:
        candidate_protocol_path = ROOT / baseline['protocol_path']
        if (sha(candidate_protocol_path) != baseline['protocol_sha256']
                or candidate_report_path.resolve() != (ROOT / baseline['report']).resolve()):
            raise ValueError('M3 restart baseline protocol or candidate path differs')
    candidate_protocol = json.loads(candidate_protocol_path.read_text())
    if (candidate_protocol['dataset'] != protocol['dataset']
            or candidate_protocol['decision_rule'] != protocol['decision_rule']):
        raise ValueError('M3 restart baseline data or decision rule differs')
    candidate_protocol_sha = sha(candidate_protocol_path)
    candidate = json.loads(candidate_report_path.read_text())
    if (candidate.get('status') != 'completed' or candidate.get('decision') != 'accept'
            or candidate.get('depth_arm') != args.depth_arm
            or candidate.get('protocol_sha256') != candidate_protocol_sha
            or candidate.get('implementation_sha256') != candidate_protocol['implementation_sha256']):
        raise ValueError('M3 restart requires an accepted matching candidate run')
    data = ROOT / protocol['dataset']['root']
    if sha(data / 'manifest.json') != protocol['dataset']['manifest_sha256']:
        raise ValueError('M3 restart dataset manifest differs')
    if (sha(data / 'records.jsonl') != protocol['dataset']['records_sha256']
            or sha(data / 'queries.jsonl') != protocol['dataset']['queries_sha256']):
        raise ValueError('M3 restart dataset differs')
    query = next(row for row in rows(data / 'queries.jsonl') if row['split'] == 'heldout')
    spec = FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2',
        depth_mode=args.depth_arm, depth_writers=True)
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=spec)
    extension = args.candidate_run / 'selected_extension'
    receipt = model.load_extension(extension)
    model.eval()
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  extension_manifest_digest=receipt['manifest_digest'],
                  candidate_run_report=str(candidate_report_path),
                  candidate_run_report_sha256=sha(candidate_report_path),
                  candidate_protocol_path=str(candidate_protocol_path),
                  candidate_protocol_sha256=candidate_protocol_sha,
                  execution_protocol_sha256=sha(args.protocol),
                  candidate_extension=str(extension), depth_arm=args.depth_arm)
    scope = FastMemoryScope(query['subject_id'], query['branch_id'], query['purpose'],
                            'parallel_replica', 0)
    config = FastMemoryConfig(
        key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], heads=protocol['fast_memory']['heads'],
        update_rule='sgdn2', max_tokens_per_chunk=256,
        max_image_bytes=protocol['fast_memory']['max_image_bytes'])
    read, image = load_fast_depth_image(
        args.candidate_run / 'fast_image', config=config, scope=scope,
        producer=model.fast_version_key(), frame_producer=model.fast_producer_key(), device='mps')
    ledger = RecordLedger(args.output / 'restart_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    answer = generate(session, query, DELAYED_USE_TEMPLATE, read, max_prompt_tokens=256)
    ledger.close()
    previous = candidate['restart']['correct_answer']
    report.update(
        status='completed', fresh_process_reload=True,
        source_observations_reencoded=False, image=image,
        previous_answer=previous, reloaded_answer=answer,
        checks={
            'answer_equal': answer['answer'] == previous['answer'],
            'heldout_answer_correct': answer['correct'],
            'state_and_lineage_authenticated': True,
            'no_observation_encoding': True,
        })
    report['decision'] = 'accept' if all(report['checks'].values()) else 'reject'
    elapsed = time.monotonic() - started
    limit = protocol.get('budget', {}).get('restart_max_wall_seconds', 60)
    if elapsed > limit:
        raise TimeoutError('M3 restart wall-time budget exceeded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--depth-arm', choices=['shared', 'private'], required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol), 'downloads': 0,
              'optimizer_updates': 0, 'checkpoints_saved': 0}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop()
        report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
                           'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / ('run_report.json' if report['status'] == 'completed'
                        else 'failure_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'decision': report.get('decision'),
                      'checks': report.get('checks')}))


if __name__ == '__main__':
    main()
