#!/usr/bin/env python3
"""M4 fresh-process extension, pattern-table and depth-image restart on MPS."""
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
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec, PatternRead, PatternSpec
    from replica_lab.architecture.loading import encode_prompt, load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.fast_memory import FastMemoryConfig, FastMemoryScope, load_fast_depth_image
    from replica_lab.runtime.routing import choose_route
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('format') != 'replica.m4.pattern.v1':
        raise ValueError('M4 restart requires an M4 protocol')
    implementation_pins = protocol['implementation_sha256']
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M4 restart implementation changed: {name}')
    report['implementation_sha256'] = implementation_pins
    candidate_report_path = args.candidate_run / 'run_report.json'
    candidate = json.loads(candidate_report_path.read_text())
    if (candidate.get('status') != 'completed' or candidate.get('decision') != 'accept'
            or candidate.get('protocol_sha256') != sha(args.protocol)
            or candidate.get('implementation_sha256') != implementation_pins):
        raise ValueError('M4 restart requires the accepted matching candidate')
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'), ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'M4 restart dataset changed: {filename}')
    query = next(row for row in rows(data / 'queries.jsonl') if row['split'] == 'heldout')
    fast_spec = FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2',
        depth_mode='shared', depth_writers=True)
    pattern_spec = PatternSpec(**protocol['pattern']['spec'])
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=fast_spec,
                                            pattern=pattern_spec)
    extension = args.candidate_run / 'selected_extension'
    receipt = model.load_extension(extension)
    model.eval()
    if receipt['manifest_digest'] != candidate['extension']['manifest_digest']:
        raise ValueError('M4 restart extension manifest differs')
    if model.pattern_version_key() != candidate['extension']['pattern_version']:
        raise ValueError('M4 restart pattern-table version differs')
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  extension_manifest_digest=receipt['manifest_digest'],
                  candidate_run_report=str(candidate_report_path),
                  candidate_run_report_sha256=sha(candidate_report_path),
                  candidate_protocol_sha256=sha(args.protocol),
                  execution_protocol_sha256=sha(args.protocol),
                  candidate_extension=str(extension))
    scope = FastMemoryScope(query['subject_id'], query['branch_id'], query['purpose'],
                            'parallel_replica', 0)
    config = FastMemoryConfig(key_size=protocol['fast_memory']['key_size'],
                              value_size=protocol['fast_memory']['value_size'],
                              heads=protocol['fast_memory']['heads'], update_rule='sgdn2',
                              max_tokens_per_chunk=256,
                              max_image_bytes=protocol['fast_memory']['max_image_bytes'])
    read, image = load_fast_depth_image(
        args.candidate_run / 'fast_image', config=config, scope=scope,
        producer=model.fast_version_key(), frame_producer=model.fast_producer_key(), device='mps')
    trained_hashes = frozenset(tuple(item) for item in candidate['trained_hash_ids'])
    text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
    ids = encode_prompt(tokenizer, text, 'mps')
    query_hashes = set()
    for order, head, index, available in model.pattern_port.hash_indices(ids, torch.ones_like(ids, dtype=torch.bool)):
        if bool(available.any()):
            query_hashes.update((order, head, int(value)) for value in index[available].tolist())
    choice = choose_route(case_role=query.get('case_role', 'delayed_training'),
                          query_hashes=frozenset(query_hashes), trained_hashes=trained_hashes)
    if choice.kind == 'pattern':
        fast, pattern = None, PatternRead(model.pattern_producer_key(), protocol['pattern']['version'])
    elif choice.kind == 'fast':
        fast, pattern = read, None
    else:
        raise ValueError(f'Unexpected M4 restart route: {choice.kind}')
    ledger = RecordLedger(args.output / 'restart_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    answer = generate(session, query, DELAYED_USE_TEMPLATE, fast, pattern=pattern)
    answer['route'] = choice.kind
    answer['route_reason'] = choice.reason
    ledger.close()
    previous = candidate['restart']['correct_answer']
    report.update(status='completed', fresh_process_reload=True,
                  source_observations_reencoded=False, image=image,
                  previous_answer=previous, reloaded_answer=answer,
                  checks={'answer_equal': answer['answer'] == previous['answer'],
                          'heldout_answer_correct': answer['correct'],
                          'route_equal': answer['route'] == previous.get('route', choice.kind),
                          'state_and_lineage_authenticated': True,
                          'no_observation_encoding': True})
    report['decision'] = 'accept' if all(report['checks'].values()) else 'reject'
    if time.monotonic() - started > protocol['budget']['restart_max_wall_seconds']:
        raise TimeoutError('M4 restart wall-time budget exceeded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path, default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
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
