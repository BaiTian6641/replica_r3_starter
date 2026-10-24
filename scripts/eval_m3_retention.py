#!/usr/bin/env python3
"""Inference-only M3 retention ablation on preserved candidates (MPS)."""
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

from run_m2_dynamic import DELAYED_USE_TEMPLATE, generate, make_read, rows, sha, zero_read


def execute(args, profiler, report):
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    implementation_pins = protocol['implementation_sha256']
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'Retention diagnostic implementation changed: {name}')
    report['implementation_sha256'] = implementation_pins
    baseline = protocol['baseline_arms'][args.depth_arm]
    candidate_protocol_path = ROOT / baseline['protocol_path']
    candidate_protocol = json.loads(candidate_protocol_path.read_text())
    candidate_report_path = ROOT / baseline['report']
    candidate = json.loads(candidate_report_path.read_text())
    if (sha(candidate_protocol_path) != baseline['protocol_sha256']
            or candidate_report_path.resolve() != (ROOT / baseline['report']).resolve()
            or candidate.get('status') != 'completed' or candidate.get('decision') != 'accept'
            or candidate.get('protocol_sha256') != baseline['protocol_sha256']
            or candidate.get('implementation_sha256') != candidate_protocol['implementation_sha256']):
        raise ValueError('Retention diagnostic candidate lineage differs')
    identity_fields = ('root', 'manifest_sha256', 'queries_sha256', 'records_sha256')
    if any(candidate_protocol['dataset'][key] != protocol['dataset'][key]
            for key in identity_fields):
        raise ValueError('Retention diagnostic dataset differs')
    report.update(depth_arm=args.depth_arm, candidate_protocol_sha256=baseline['protocol_sha256'],
                  candidate_run_report=str(candidate_report_path),
                  candidate_run_report_sha256=sha(candidate_report_path))
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'), ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'Retention diagnostic dataset changed: {filename}')
    queries = [row for row in rows(data / 'queries.jsonl') if row['split'] == 'retention']
    if len(queries) != protocol['evaluation']['retention_cases']:
        raise ValueError('Retention diagnostic case count differs')
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
                  extension_manifest_digest=receipt['manifest_digest'])
    ledger = RecordLedger(args.output / 'retention_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    private_depth = args.depth_arm == 'private'
    shared_reducer = args.depth_arm == 'shared'
    def depth_read(observations, **kwargs):
        return make_read(model, tokenizer, observations, private_depth=private_depth,
                         shared_reducer=shared_reducer, **kwargs)
    results = {}
    for mode in protocol['evaluation']['arms']:
        results[mode] = []
        for index, query in enumerate(queries):
            enabled_ports = None
            include_observation = mode == 'full_context'
            if mode == 'disabled':
                state = None
            elif mode == 'zero':
                state = zero_read(model, private_depth=private_depth)
            elif mode == 'full_context':
                state = None
            elif mode == 'shuffled':
                state = depth_read(queries[1 - index]['observations'])
            elif mode == 'linear':
                state = depth_read(query['observations'], linear=True)
            elif mode == 'correct':
                state = depth_read(query['observations'])
            else:
                kind, layer_text, suffix = mode.split('_')
                if kind not in {'read', 'write'} or suffix != 'only':
                    raise ValueError(f'Unknown retention arm: {mode}')
                layer = int(layer_text)
                state = depth_read(query['observations'],
                                   write_layers=(layer,) if kind == 'write' else None)
                if kind == 'read':
                    enabled_ports = (layer,)
            value = generate(session, query, DELAYED_USE_TEMPLATE, state,
                             include_observation=include_observation, enabled_ports=enabled_ports)
            value['case_id'] = query['case_id']
            results[mode].append(value)
            if time.monotonic() - started > protocol['budget']['max_wall_seconds_per_arm']:
                raise TimeoutError('Retention diagnostic wall-time budget exceeded')
    report['candidate'] = results
    report['metrics'] = {mode: sum(row['correct'] for row in values) / len(values)
                         for mode, values in results.items()}
    total_answers = sum(len(values) for values in results.values())
    total_new_tokens = sum(row['generated_tokens'] for values in results.values() for row in values)
    report['generation_counts'] = {'answers': total_answers, 'new_tokens': total_new_tokens}
    if (total_answers > protocol['budget']['generation_answers_per_arm']
            or total_new_tokens > protocol['budget']['max_generated_tokens_per_arm']):
        raise ValueError('Retention diagnostic generation budget exceeded')
    correct = report['metrics']['correct']
    answers = lambda mode: [row['answer'] for row in results[mode]]
    report['diagnostic'] = {
        'regression_reproduced': correct < 1.0,
        'zero_recovers': report['metrics']['zero'] > correct,
        'disabled_recovers': report['metrics']['disabled'] > correct,
        'full_context_recovers': report['metrics']['full_context'] > correct,
        'read_13_recovers': report['metrics']['read_13_only'] > correct,
        'read_27_reproduces': answers('read_27_only') == answers('correct'),
        'write_13_recovers': report['metrics']['write_13_only'] > correct,
        'write_27_reproduces': answers('write_27_only') == answers('correct'),
        'interpretation': 'Descriptive ablation only; no repair or promotion admitted by this report.'
    }
    report['status'] = 'completed'
    ledger.close()


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
    print(json.dumps({'status': report['status'], 'metrics': report.get('metrics'),
                      'diagnostic': report.get('diagnostic')}))


if __name__ == '__main__':
    main()
