#!/usr/bin/env python3
"""Ascending native MPS capacity schedule for the E2 memory path.

Measures source encoding, writer, sense, action forward and generation for
inference, and prepare/forward/backward for modules-only training. It does not
optimize, download or save a base checkpoint. Every point records actual units
and token lengths; a guard breach preserves the partial profile.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(torch, action):
    torch.mps.synchronize()
    start = time.perf_counter()
    value = action()
    torch.mps.synchronize()
    return value, time.perf_counter() - start


def inference_call(torch, action):
    """Run an inference-only producer without retaining an autograd graph."""
    with torch.no_grad():
        return action()


def make_prompt(tokenizer, target):
    from replica_lab.architecture.loading import encode_prompt
    filler = ('The capacity probe records routine configuration notes, scheduling, '
              'maintenance, review, documentation, audits and recovery details. ')
    text = 'Capacity measurement query. ' + filler * 64
    ids = encode_prompt(tokenizer, text, 'mps')
    if ids.shape[1] < target:
        text += filler * 64
        ids = encode_prompt(tokenizer, text, 'mps')
    if ids.shape[1] < target:
        raise ValueError('Prompt builder did not reach the target token count')
    return ids[:, :target]


def selection_for_units(session, scope, records, units):
    from replica_lab.memory_system.index import RetrievalSelection
    snapshot = session.ledger.snapshot(scope)
    by_id = {record.record_id: record for record in snapshot.records}
    counts = {record.record_id: 0 for record in records}
    spans = session.spans(records)
    for span in spans:
        counts[span.record_id] += 1
    selected, used = [], 0
    for record in records:
        count = counts[record.record_id]
        if used + count > units:
            break
        selected.append(by_id[record.record_id])
        used += count
    if used != units:
        raise ValueError(f'Cannot select exactly {units} units; reached {used}')
    return RetrievalSelection(tuple(selected), tuple(0.0 for _ in selected),
                              snapshot.epoch, snapshot.ledger_id, scope, 'capacity')


def inference_point(torch, model, tokenizer, session, scope, task, records, point,
                    report):
    from replica_lab.architecture.contracts import MemoryRead, RequestContext
    from replica_lab.memory_system.session import pin_selection
    units = int(point['units'])
    selection = selection_for_units(session, scope, records, units)
    spans = session.spans(selection.records)
    if len(spans) != units:
        raise AssertionError('Selected span count differs from declared units')
    prompt_ids = make_prompt(tokenizer, int(point['prompt_tokens']))
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    feature_rows = []

    def encode():
        result = []
        with torch.no_grad():
            for span in spans:
                ids = torch.tensor([span.token_ids], dtype=torch.long, device='mps')
                result.append(model.encode_observations(
                    ids, torch.ones_like(ids, dtype=torch.bool)))
        return result

    features, source_seconds = timed(torch, encode)
    feature_rows = features
    length = max(int(feature.shape[1]) for feature in feature_rows)
    hidden = torch.zeros(1, len(feature_rows), length,
                         model.base.config.hidden_size, device='mps',
                         dtype=feature_rows[0].dtype)
    valid_hidden = torch.zeros(1, len(feature_rows), length, dtype=torch.bool,
                               device='mps')
    for index, feature in enumerate(feature_rows):
        hidden[0, index, :feature.shape[1]] = feature[0]
        valid_hidden[0, index, :feature.shape[1]] = True
    (slots, slots_valid), writer_seconds = timed(
        torch, lambda: inference_call(
            torch, lambda: model.record_writer(hidden, valid_hidden)))
    memory = MemoryRead(spans, slots, slots_valid, model.memory_producer_key())
    sense, sense_seconds = timed(
        torch, lambda: inference_call(
            torch, lambda: model.sense(prompt_ids, prompt_mask)))
    evidence = pin_selection(selection)
    frame = model.make_frame(f'capacity-{point["id"]}', evidence, memory, sense)
    request = RequestContext(f'capacity-{point["id"]}', task, frame, selection)
    try:
        with torch.no_grad():
            forward, forward_seconds = timed(
                torch, lambda: session.forward(request, prompt_ids))
            generated, generation_seconds = timed(
                torch, lambda: session.generate(request,
                                                max_new_tokens=int(point['generate_tokens'])))
        prompt_tokens = int(frame.sense.prompt_ids.shape[1])
        new_ids = [int(token) for token in generated[0, prompt_tokens:].tolist()]
        result = {
            'id': point['id'], 'phase': 'inference',
            'declared_units': units, 'actual_units': len(spans),
            'declared_prompt_tokens': int(point['prompt_tokens']),
            'actual_prompt_tokens': prompt_tokens,
            'generated_tokens': len(new_ids), 'generated_token_ids': new_ids,
            'source_span_tokens': [span.token_end - span.token_start for span in spans],
            'source_span_token_ranges': [(span.token_start, span.token_end) for span in spans],
            'source_encode_seconds': source_seconds,
            'writer_seconds': writer_seconds,
            'sense_seconds': sense_seconds,
            'action_forward_seconds': forward_seconds,
            'writer_requires_grad': bool(slots.requires_grad),
            'sense_requires_grad': any(feature.requires_grad
                                       for _, feature in sense.summaries),
            'generation_seconds': generation_seconds,
            'logits_finite': bool(torch.isfinite(forward.logits).all()),
            'mps_allocated_after_bytes': int(torch.mps.current_allocated_memory()),
            'mps_driver_after_bytes': int(torch.mps.driver_allocated_memory()),
        }
        return result
    finally:
        request.release()


def training_point(torch, model, tokenizer, session, scope, task, records, point,
                   report):
    from replica_lab.architecture.contracts import RequestContext
    units = int(point['units'])
    selection = selection_for_units(session, scope, records, units)
    spans = session.spans(selection.records)
    if len(spans) != units:
        raise AssertionError('Selected span count differs from declared units')
    prompt_ids = make_prompt(tokenizer, int(point['prompt_tokens']))
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    suffix = prompt_ids[:, -int(point['label_tokens']):].clone()
    action_ids = torch.cat((prompt_ids, suffix), dim=1)
    action_mask = torch.ones_like(action_ids, dtype=torch.bool)
    labels = action_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    model.train()
    model.zero_grad(set_to_none=True)

    def prepare():
        return session.prepare(prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                               query='capacity', scope=scope, task=task,
                               selection=selection, training=True, train_sense=False)

    request, prepare_seconds = timed(torch, prepare)
    try:
        output, forward_seconds = timed(
            torch, lambda: session.forward(request, action_ids, labels=labels,
                                           attention_mask=action_mask))
        loss_value = float(output.loss.detach())
        _, backward_seconds = timed(torch, output.loss.backward)
        gradients = model.gradient_audit()
        result = {
            'id': point['id'], 'phase': 'training',
            'declared_units': units, 'actual_units': len(spans),
            'declared_prompt_tokens': int(point['prompt_tokens']),
            'actual_prompt_tokens': int(prompt_ids.shape[1]),
            'label_tokens': int(point['label_tokens']),
            'source_span_tokens': [span.token_end - span.token_start for span in spans],
            'prepare_seconds': prepare_seconds, 'forward_seconds': forward_seconds,
            'backward_seconds': backward_seconds, 'loss': loss_value,
            'new_gradient_present': len(gradients['new']['present']),
            'new_gradient_nonzero': len(gradients['new']['nonzero']),
            'nonfinite_gradients': bool(gradients['new']['nonfinite'])
                                   or bool(gradients['original']['nonfinite']),
            'mps_allocated_after_bytes': int(torch.mps.current_allocated_memory()),
            'mps_driver_after_bytes': int(torch.mps.driver_allocated_memory()),
        }
        return result
    finally:
        request.release()
        model.zero_grad(set_to_none=True)
        model.eval()


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    from replica_lab.memory_system.session import MemorySession

    protocol = json.loads(args.protocol.read_text())
    if protocol['format'] not in ('replica.e2_capacity.qualification.v1',
                                  'replica.e2_capacity.qualification.v2',
                                  'replica.e2_capacity.qualification.v3',
                                  'replica.e2_capacity.qualification.v4',
                                  'replica.e2_capacity.qualification.v5',
                                  'replica.e2_capacity.qualification.v6',
                                  'replica.e2_capacity.qualification.v7',
                                  'replica.e2_capacity.qualification.v8',
                                  'replica.e2_capacity.qualification.v9',
                                  'replica.e2_capacity.qualification.v10',
                                  'replica.e2_capacity.qualification.v11',
                                  'replica.e2_capacity.qualification.v12'):
        raise ValueError('Unknown capacity protocol')
    data_root = ROOT / protocol['dataset']['root']
    manifest = json.loads((data_root / 'manifest.json').read_text())
    if sha(data_root / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('Capacity records differ from the preregistered hash')
    records = [MemoryRecord(**json.loads(line)) for line in
               (data_root / 'records.jsonl').read_text().splitlines() if line.strip()]
    declared_units = int(protocol['scope']['max_units'])
    schedule = protocol['schedule']
    ids = [point['id'] for phase in ('inference', 'training')
           for point in schedule[phase]]
    if len(ids) != len(set(ids)):
        raise ValueError('Schedule ids must be unique')
    for phase in ('inference', 'training'):
        units = [int(point['units']) for point in schedule[phase]]
        if units != sorted(units) or units[-1] > declared_units or units[0] < 1:
            raise ValueError('Each schedule phase must ascend within the declared unit cap')
    parent = (None if protocol.get('parent_extension') is None
              else Path(protocol['parent_extension']))
    from replica_lab.architecture.contracts import PortSpec
    model, tokenizer, source_lock = load_local_mps(
        args.model_path, args.model_lock, profiler=profiler,
        ports=PortSpec(max_units=declared_units))
    parent_receipt = (model.load_extension(parent, accept_parent_v1=True,
                                           port_capacity_migration=True)
                      if parent is not None else None)
    report.update(manifest=model.manifest.to_dict(), manifest_digest=model.manifest.digest,
                  source_lock=source_lock, parameters=model.parameter_audit(),
                  dataset={'manifest': manifest,
                           'records_sha256': protocol['dataset']['records_sha256']},
                  parent_extension=(None if parent is None else {
                                    'path': str(parent.resolve()),
                                    'format': parent_receipt['format'],
                                    'weights_sha256': parent_receipt['weights_sha256']}),
                  checks={}, points=[])
    ledger = RecordLedger(args.output / 'capacity_ledger.sqlite')
    for record in records:
        ledger.admit(record, authorized_subject=record.subject_id,
                     authorized_branch=record.branch_id, source_authorized=False)
    scope = EligibilityScope(records[0].subject_id, records[0].branch_id,
                             records[0].purpose, 10, 'assistant_for_source', None)
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=max(64, declared_units * 2))
    task = TaskGraph('capacity', 'measure bounded memory path')
    all_spans = session.spans(records)
    if len(all_spans) != declared_units:
        raise ValueError(f'Capacity bank has {len(all_spans)} units, expected {declared_units}')
    report['bank'] = {'units': len(all_spans),
                      'source_span_tokens': [span.token_end - span.token_start
                                             for span in all_spans],
                      'record_ids': [span.record_id for span in all_spans]}

    def mark(name, **fields):
        profiler.mark(name, **fields)
        dump(args.output / 'progress.json', report)
        print(json.dumps({'phase': name, **fields}), flush=True)

    overall_start = time.monotonic()
    for phase in ('inference', 'training'):
        for point in schedule[phase]:
            before_alloc = int(torch.mps.current_allocated_memory())
            before_driver = int(torch.mps.driver_allocated_memory())
            started = time.monotonic()
            if phase == 'inference':
                result = inference_point(torch, model, tokenizer, session, scope, task,
                                         records, point, report)
            else:
                result = training_point(torch, model, tokenizer, session, scope, task,
                                        records, point, report)
            result['wall_seconds'] = time.monotonic() - started
            result['mps_allocated_before_bytes'] = before_alloc
            result['mps_driver_before_bytes'] = before_driver
            report['points'].append(result)
            mark(f'capacity_{phase}', point=point['id'], units=result['actual_units'],
                 tokens=result['actual_prompt_tokens'],
                 wall_seconds=result['wall_seconds'])
            torch.mps.empty_cache()
    report['summary'] = {
        'largest_inference_units': max(point['actual_units'] for point in report['points']
                                       if point['phase'] == 'inference'),
        'largest_training_units': max(point['actual_units'] for point in report['points']
                                      if point['phase'] == 'training'),
        'largest_prompt_tokens': max(point['actual_prompt_tokens'] for point in report['points']),
        'largest_source_span_tokens': max(max(point['source_span_tokens'])
                                          for point in report['points']),
        'points': len(report['points']),
        'inference_points': len(schedule['inference']),
        'training_points': len(schedule['training']),
        'wall_seconds': time.monotonic() - overall_start,
        'd2_arm': 'not declared or run',
    }
    report['acceptance'] = {
        'completed_all_points': len(report['points']) == len(ids),
        'inference_producers_detached': all(
            not point['writer_requires_grad'] and not point['sense_requires_grad']
            for point in report['points'] if point['phase'] == 'inference'),
        'schedule_ascending': True,
        'configured_max_not_claimed_as_exercised': True,
        'training_scope': 'modules_only backward; no optimizer step',
        'claims': 'Capacity only; no generation-quality or architecture-benefit claim.'}
    ledger.close()
    mark('capacity_complete', points=len(report['points']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e2_capacity_qualification_v1.json')
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
        parser.error('At least 5 GiB disk headroom required')
    protocol = json.loads(args.protocol.read_text())
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'arguments': {key: str(value) for key, value in vars(args).items()},
              'protocol_sha256': sha(args.protocol), 'protocol': protocol,
              'packages': {name: importlib.metadata.version(name)
                           for name in ('torch', 'transformers', 'safetensors')},
              'implementation_sha256': {
                  str(path.relative_to(ROOT)): sha(path)
                  for path in [Path(__file__),
                               *(ROOT / 'src/replica_lab/memory_system').glob('*.py'),
                               *(ROOT / 'src/replica_lab/architecture').glob('*.py')]},
              'free_disk_before_bytes': shutil.disk_usage(ROOT).free,
              'downloads': 0, 'original_model_loads': 1,
              'environment': {key: os.environ[key] for key in
                              ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                               'HF_DATASETS_OFFLINE', 'PYTORCH_ENABLE_MPS_FALLBACK')}}
    profiler = RunProfiler('mps', args.output, sample_interval=0.1)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop()
        report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {
            'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
            'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes'),
            'sampled_peak_rss_bytes': memory.get('rss_peak_sampled_bytes'),
            'new_original_checkpoints': 0, 'downloads': 0}
        report['free_disk_after_bytes'] = shutil.disk_usage(ROOT).free
        report['artifact_bytes'] = sum(path.stat().st_size for path in args.output.rglob('*')
                                       if path.is_file())
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(args.output / ('run_report.json' if report['status'] == 'completed'
                            else 'failure_report.json'), report)
    print(json.dumps({'status': report['status'],
                      'summary': report.get('summary'),
                      'costs': report.get('costs')}))


if __name__ == '__main__':
    main()
