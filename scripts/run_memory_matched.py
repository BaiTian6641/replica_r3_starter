#!/usr/bin/env python3
"""Guarded native MPS matched-episode memory localisation diagnostic.

Requires ``REPLICA_CAPACITY_GUARD=1``, an admitted preparation, the sealed
MiniCPM5 source, the frozen rejected writer extension and MPS.  Zero optimizer
updates are performed.  The script never trains, never saves a checkpoint and
never changes the capacity guard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.memory_matched import (  # noqa: E402
    ADMISSION_FORMAT,
    PROTOCOL_FORMAT,
    enforce_elapsed_deadline,
    finalize_completed_report,
    run_matched_diagnostic,
    verify_matched_prepared,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + '\n')


def fresh_output(path, parser):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error('Use a fresh run output path')
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_paths(protocol):
    source = protocol.get('source')
    if not isinstance(source, dict):
        raise ValueError('Protocol carries no exact source pins')
    model_path = Path(source['model_path'])
    model_lock = Path(source['model_lock'])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_lock.is_absolute():
        model_lock = ROOT / model_lock
    return model_path, model_lock


def require_admission(prepared):
    path = prepared / 'admission.json'
    if not path.is_file():
        raise RuntimeError(
            'Run requires an admitted preparation: scripts/prepare_memory_matched.py '
            '--admit')
    receipt = json.loads(path.read_text())
    if receipt.get('format') != ADMISSION_FORMAT or receipt.get('status') != 'admitted':
        raise RuntimeError('Preparation admission is not an admitted receipt')
    pins = {
        'prepared_manifest_sha256': prepared / 'manifest.json',
        'episodes_sha256': prepared / 'episodes.json',
        'labels_sha256': prepared / 'labels.json',
        'protocol_sha256': prepared / 'protocol.json',
    }
    for name, artifact in pins.items():
        if not artifact.is_file() or receipt.get(name) != sha(artifact):
            raise RuntimeError(f'Admission pin differs from {artifact.name}')
    return receipt, sha(path)


def checked_extension(args, protocol):
    pin = protocol.get('writer_extension')
    if not isinstance(pin, dict):
        raise ValueError('Protocol does not pin the frozen writer extension')
    path = Path(args.writer_extension) if args.writer_extension else Path(pin['path'])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_dir():
        raise ValueError('Pinned writer extension directory is missing')
    weights = path / 'extensions.safetensors'
    if not weights.is_file() or sha(weights) != pin['weights_sha256']:
        raise ValueError('Writer extension weights differ from the frozen pin')
    manifest = path / 'manifest.json'
    if not manifest.is_file() or sha(manifest) != pin['manifest_sha256']:
        raise ValueError('Writer extension manifest differs from the frozen pin')
    return path


def run_native(args, parser):
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.profiling import RunProfiler

    prepared = Path(args.prepared).resolve()
    if not prepared.is_dir():
        parser.error('Prepared matched input directory does not exist')
    output = fresh_output(args.output, parser)
    protocol_path = prepared / 'protocol.json'
    episodes_path = prepared / 'episodes.json'
    labels_path = prepared / 'labels.json'
    manifest_path = prepared / 'manifest.json'
    protocol = json.loads(protocol_path.read_text())
    data = json.loads(episodes_path.read_text())
    if protocol.get('format') != PROTOCOL_FORMAT:
        raise ValueError('Run requires a matched-memory protocol')
    if args.device != 'mps':
        raise ValueError('The native matched-memory diagnostic requires --device mps')
    if os.environ.get('REPLICA_CAPACITY_GUARD') != '1':
        raise RuntimeError(
            'Launch through scripts/run_capacity_preflight.py --entry-point '
            'run_memory_matched.py; no unguarded native run is allowed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('PYTORCH_ENABLE_MPS_FALLBACK=0 is required')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'{key}=1 is required')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; no CPU native fallback is permitted')
    if protocol.get('zero_updates') is not True or protocol.get('optimizer_updates') != 0:
        raise ValueError('Matched diagnostic must declare zero updates')

    admission, admission_sha256 = require_admission(prepared)
    verified = verify_matched_prepared(prepared)
    if verified['implementation_sha256'] != protocol['implementation_sha256']:
        raise ValueError('Implementation pins differ from the admitted preparation')
    if protocol['episodes']['sha256'] != sha(episodes_path):
        raise ValueError('Episode bytes differ from the protocol pin')
    if protocol['labels']['sha256'] != sha(labels_path):
        raise ValueError('Verifier-label bytes differ from the protocol pin')
    model_path, model_lock = source_paths(protocol)
    source = protocol['source']
    if sha(model_lock) != source['model_lock_sha256']:
        raise ValueError('Sealed model lock bytes changed')
    if source['source_revision'] != source['required_revision']:
        raise ValueError('Protocol source revision is not the sealed revision')
    extension_path = checked_extension(args, protocol)
    budget = protocol.get('budget')
    if not isinstance(budget, dict):
        raise ValueError('Protocol carries no declared wall-clock budget')
    max_wall_seconds = budget.get('max_wall_seconds')
    if type(max_wall_seconds) is not int or max_wall_seconds <= 0:
        raise ValueError('Protocol wall-clock budget must be a positive integer')
    started_monotonic = time.monotonic()

    report = {
        'format': 'replica.memory_matched.run.v1',
        'status': 'started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'native': True, 'device': 'mps', 'fallback_disabled': True,
        'guard_required': 'REPLICA_CAPACITY_GUARD=1',
        'optimizer_updates': 0, 'checkpoints_saved': 0, 'downloads': 0,
        'budget': budget,
        'prepared': {
            'path': str(prepared),
            'protocol_sha256': sha(protocol_path),
            'episodes_sha256': sha(episodes_path),
            'labels_sha256': sha(labels_path),
            'manifest_sha256': sha(manifest_path),
            'admission_sha256': admission_sha256,
        },
        'workload': protocol['planned_workload'],
        'source': source,
    }
    profiler = RunProfiler('mps', output)
    try:
        with exclusive_mps_worker(ROOT):
            engine = protocol.get('engine') or {}
            spec = FastMemorySpec(
                heads=engine.get('heads', 4), key_size=engine.get('key_size', 64),
                value_size=engine.get('value_size', 64),
                update_rule=engine.get('update_rule', 'sgdn2'),
                depth_mode=engine.get('depth_mode', 'shared'),
                depth_writers=engine.get('depth_writers', True))
            profiler.start()
            model, tokenizer, lock = load_local_mps(
                model_path, model_lock, profiler=profiler, fast_memory=spec)
            extension_receipt = model.load_extension(extension_path)
            pin = protocol['writer_extension']
            if extension_receipt['weights_sha256'] != pin['weights_sha256']:
                raise ValueError('Loaded writer extension weights differ from the pin')
            if extension_receipt.get('manifest_digest') != pin.get('manifest_digest'):
                raise ValueError('Loaded writer extension manifest differs from the pin')
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            report['native'] = {
                'device': 'mps', 'fallback_disabled': True,
                'sealed_revision': lock['revision'],
                'source_lock_sha256': sha(model_lock),
                'writer_extension': extension_receipt,
            }
            report['implementation_sha256'] = protocol['implementation_sha256']
            result = run_matched_diagnostic(
                model, tokenizer, protocol, output_dir=output,
                ledger_dir=output / 'ledger', episodes=data['episodes'],
                device='mps', writer_extension_receipt=extension_receipt)
            result['budget'] = budget
            result['started_at_utc'] = report['started_at_utc']
            result['native_run'] = report['native']
            result['prepared'] = report['prepared']
            result['source'] = source
            result['implementation_sha256'] = protocol['implementation_sha256']
            result['raw_readouts_path'] = 'readouts.jsonl'
            result['raw_answers_path'] = 'answers.jsonl'
            result['codebooks_path'] = 'codebooks.json'
            result['rankings_path'] = 'rankings.jsonl'
            result['failures_path'] = 'failures.json'
            if result['readouts_sha256'] != sha(output / 'readouts.jsonl'):
                raise ValueError('Raw readout artifact bytes changed')
            if result['answers_sha256'] != sha(output / 'answers.jsonl'):
                raise ValueError('Raw answer artifact bytes changed')
            if result['counts']['readout_rows'] != len(
                    (output / 'readouts.jsonl').read_text().splitlines()):
                raise ValueError('Raw readout row count differs')
            if result['counts']['answer_rows'] != len(
                    (output / 'answers.jsonl').read_text().splitlines()):
                raise ValueError('Raw answer row count differs')
            result['admission'] = admission
            finalize_completed_report(
                result, started_monotonic=started_monotonic,
                max_wall_seconds=max_wall_seconds,
                stop_profiler=profiler.stop,
                write_report=lambda value: dump(output / 'run_report.json', value),
                label='native matched diagnostic')
            print(json.dumps({
                'status': result['status'],
                'decision': result['decision'],
                'first_failing_stage': result['first_failing_stage'],
                'failed_gates': result['failed_gates'],
                'counts': result['counts'],
                'report': str(output / 'run_report.json'),
            }, sort_keys=True))
            enforce_elapsed_deadline(
                started_monotonic, max_wall_seconds,
                label='native matched diagnostic return')
            return 0
    except BaseException as error:
        failure = dict(report, status='failed',
                       finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic() - started_monotonic,
                       deadline_seconds=max_wall_seconds, deadline_ok=False,
                       error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(),
                       profile=profiler.stop())
        dump(output / 'run_report.json', failure)
        print(json.dumps({'status': 'failed',
                          'error_type': type(error).__name__,
                          'error': str(error),
                          'report': str(output / 'run_report.json')},
                         sort_keys=True), file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('mps',), default='mps')
    parser.add_argument('--writer-extension', type=Path)
    args = parser.parse_args(argv)
    return run_native(args, parser)


if __name__ == '__main__':
    raise SystemExit(main())
