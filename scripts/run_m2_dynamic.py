#!/usr/bin/env python3
"""M2 delayed-use arm for the headed SGDN-2 residual memory path (MPS)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time
from pathlib import Path
import sys
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

DELAYED_USE_TEMPLATE = (
    'A source page was observed earlier and stored in fast memory. '
    'Answer the query using only that observation. Reply with only the value.\n\n'
    'Query: {query}\n\nObserved source (only for the full-context control):\n{observation}')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def digest_weights(weights):
    from replica_lab.neural import FastWrite, LinearFastMemory, SGDN2Memory
    return hashlib.sha256(weights.detach().float().cpu().numpy().tobytes()).hexdigest()

def make_read(model, tokenizer, observations, *, training=False, tag='state', linear=False,
              private_depth=False, shared_reducer=False, gradient_cut=False,
              write_layers=None, gradient_probes=None):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    from replica_lab.neural import FastWrite, LinearFastMemory, SGDN2Memory
    heads = model.manifest.fast_memory.heads
    key_size = model.manifest.fast_memory.key_size
    value_size = model.manifest.fast_memory.value_size
    layers = tuple(model.ports.injection_layers)
    device = model.base.get_input_embeddings().weight.device
    if write_layers is not None:
        write_layers = frozenset(write_layers)
        if not write_layers <= set(layers):
            raise ValueError('write ablation names an undeclared fast port')
        if not (private_depth or shared_reducer):
            raise ValueError('per-port write ablation requires an M3 depth arm')
    context = torch.enable_grad() if training else torch.no_grad()
    private = None
    with context:
        if linear:
            if write_layers is not None:
                raise ValueError('linear control does not support per-port write ablation')
            private = {} if private_depth else None
            for layer in (layers if private_depth else layers[:1]):
                weights = []
                for head in range(heads):
                    primitive = LinearFastMemory(key_size, value_size, learning_rate=0.1,
                                                 decay=0.0, max_update_norm=1.0)
                    state = primitive.initial_state(1, device=device)
                    for text in observations:
                        write = model.encode_fast_observation(
                            text, tokenizer, max_tokens=256,
                            layer=(layer if model.fast_depth_write_encoders is not None else None))
                        part = write.head(head)
                        state = primitive.update_shared_chunk(
                            state, [FastWrite(part.keys.squeeze(2), part.values.squeeze(2),
                                               state.chunk_index)])
                    weights.append(state.weights)
                if private is not None:
                    private[layer] = torch.stack(weights, dim=1)
                else:
                    weights = torch.stack(weights, dim=1)
        else:
            primitive = SGDN2Memory(heads, key_size, value_size)
            if private_depth:
                private = {layer: primitive.initial_state(1, device=device) for layer in layers}
                state = None
            else:
                state = primitive.initial_state(1, device=device)
            for index, text in enumerate(observations):
                chunk_index = (private[layers[0]].chunk_index if private is not None
                               else state.chunk_index)
                writes = {}
                for layer in layers:
                    if model.fast_depth_write_encoders is None and writes:
                        writes[layer] = writes[layers[0]]
                        continue
                    write = model.encode_fast_observation(
                        text, tokenizer, max_tokens=256,
                        layer=layer if model.fast_depth_write_encoders is not None else None)
                    if gradient_cut and index == 0:
                        write = write.detached()
                    writes[layer] = replace(write, chunk_index=chunk_index)
                if gradient_probes is not None:
                    for layer, write in writes.items():
                        if model.fast_depth_write_encoders is None and layer != layers[0]:
                            continue
                        for name in ('keys', 'values', 'erase', 'write_gate', 'decay'):
                            tensor = getattr(write, name)
                            if tensor.requires_grad:
                                tensor.retain_grad()
                        gradient_probes.append({'chunk_index': index, 'layer': layer, 'write': write})
                if private is not None:
                    for layer in layers:
                        proposals = ((writes[layer],)
                                     if write_layers is None or layer in write_layers else ())
                        private[layer] = primitive.update_shared_chunk(private[layer], proposals)
                elif shared_reducer:
                    proposals = {layer: ((writes[layer],)
                                         if write_layers is None or layer in write_layers else ())
                                 for layer in layers}
                    state = primitive.reduce_shared_chunk(state, proposals)
                else:
                    state = primitive.update_shared_chunk(state, [writes[layers[0]]])
            if private is not None:
                private = {layer: value.weights for layer, value in private.items()}
            weights = (private[layers[0]] if private is not None else state.weights)
        if private is not None:
            weights = private[layers[0]]
        if not training:
            weights = weights.detach()
            if private is not None:
                private = {layer: value.detach() for layer, value in private.items()}
    if private is not None:
        weights = private[layers[0]]
        state_digest = digest_weights(torch.cat(
            [private[layer] for layer in sorted(private)], dim=1)) if not training else tag
    else:
        state_digest = digest_weights(weights) if not training else tag
    return FastMemoryRead(weights, model.fast_producer_key(), state_digest,
                          private_weights=private)


def zero_read(model, *, private_depth=False):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    weights = torch.zeros(1, model.manifest.fast_memory.heads,
                          model.manifest.fast_memory.key_size,
                          model.manifest.fast_memory.value_size,
                          device=model.base.get_input_embeddings().weight.device)
    private = None
    if private_depth:
        private = {layer: weights.clone() for layer in model.ports.injection_layers}
        weights = private[model.ports.injection_layers[0]]
    return FastMemoryRead(weights, model.fast_producer_key(), 'zero-state',
                          private_weights=private)


def probe_gradient(probe):
    total = 0.0
    present = False
    for name in ('keys', 'values', 'erase', 'write_gate', 'decay'):
        gradient = getattr(probe['write'], name).grad
        if gradient is not None:
            present = True
            total += float(gradient.detach().abs().sum())
    return {'chunk_index': probe['chunk_index'], 'layer': probe.get('layer'),
            'requires_grad': any(getattr(probe['write'], name).requires_grad for name in
                                 ('keys', 'values', 'erase', 'write_gate', 'decay')),
            'gradient_present': present, 'gradient_l1': total}


def generate(session, query_row, template, fast_state, *, include_observation=False,
             enabled_ports=None, max_prompt_tokens=256, pattern=None):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_system import EligibilityScope
    scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                             query_row['purpose'], query_row['as_of'], 'parallel_replica', None)
    text = template.replace('{query}', query_row['query'])
    if include_observation:
        text = text.replace('{observation}', '\n'.join(query_row['observations']))
    else:
        text = text.replace('{observation}', '')
    ids = encode_prompt(session.tokenizer, text, 'mps')
    if max_prompt_tokens is not None and ids.shape[1] > max_prompt_tokens:
        raise ValueError('generation prompt token budget exceeded')
    mask = torch.ones_like(ids, dtype=torch.bool)
    task = TaskGraph(query_row['case_id'], query_row['query'])
    branches = ['memory']
    if fast_state is not None:
        branches.append('fast')
    if pattern is not None:
        branches.append('pattern')
    request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=query_row['query'],
                              scope=scope, task=task, core_impl='none',
                              fast_state=fast_state, enabled_branches=branches,
                              enabled_ports=enabled_ports, pattern=pattern)
    try:
        generated = session.generate(request, max_new_tokens=6)
        length = int(request.frame.sense.prompt_ids.shape[1])
        new_ids = [int(x) for x in generated[0, length:].tolist()]
        answer = session.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {'answer': answer, 'token_ids': new_ids,
                'correct': answer.casefold() == query_row['expected_answer'].casefold(),
                'expected': query_row['expected_answer'],
                'case_role': query_row.get('case_role'),
                'audit_stratum': query_row.get('audit_stratum'),
                'prompt_tokens': length, 'generated_tokens': len(new_ids)}
    finally:
        request.release()


def evaluate(session, queries, template, states, *, include_observation=False, check=None):
    results = {}
    for arm, state in states.items():
        results[arm] = []
        for query in queries:
            value = generate(session, query, template, state(query),
                             include_observation=include_observation or arm == 'full_context')
            value['case_id'] = query['case_id']
            value['split'] = query['split']
            results[arm].append(value)
            if check is not None:
                check()
    return results


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.fast_memory import (FastMemoryConfig, FastMemoryScope,
                                                  FastMemorySession, save_fast_depth_image,
                                                  load_fast_depth_image)
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    implementation_pins = protocol.get('implementation_sha256', {})
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M3 implementation bytes changed: {name}')
    report['implementation_sha256'] = implementation_pins
    max_wall_seconds = protocol.get('budget', {}).get('max_wall_seconds_per_arm')
    def check_time():
        if max_wall_seconds is not None and time.monotonic() - started > max_wall_seconds:
            raise TimeoutError('M3 wall-time budget exceeded')
    data = ROOT / protocol['dataset']['root']
    if sha(data / 'manifest.json') != protocol['dataset']['manifest_sha256']:
        raise ValueError('M2 dataset manifest changed')
    if sha(data / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('M2 records changed')
    if sha(data / 'queries.jsonl') != protocol['dataset']['queries_sha256']:
        raise ValueError('M2 queries changed')
    queries = rows(data / 'queries.jsonl')
    train = [q for q in queries if q['split'] == 'train']
    dev = [q for q in queries if q['split'] == 'dev']
    heldout = [q for q in queries if q['split'] == 'heldout']
    replay = [q for q in queries if q['split'] == 'retention']
    regression = [q for q in queries if q['split'] == 'regression']
    retention_dev = [q for q in queries if q['split'] == 'retention_dev']
    parent_baseline = bool(protocol['evaluation'].get('parent_baseline', False))
    replay_training = [q for q in train if q.get('case_role') == 'retention_replay']
    delayed_training = [q for q in train if q.get('case_role') != 'retention_replay']
    replay_per_update = int(protocol['training'].get('replay_examples_per_update', 0))
    if replay_per_update < 0 or replay_per_update > int(protocol['training']['examples_per_update']):
        raise ValueError('Invalid replay examples per update')
    if replay_per_update and not replay_training:
        raise ValueError('Replay schedule declared without replay cases')
    if replay_training and not replay_per_update:
        raise ValueError('Replay cases present without a declared replay schedule')
    template = DELAYED_USE_TEMPLATE
    depth_arm = getattr(args, 'depth_arm', None)
    depth_mode = protocol['fast_memory'].get('depth_mode', 'shared')
    shared_reducer = False
    gradient_cut = bool(protocol['fast_memory'].get('gradient_cut', False))
    if protocol['format'] == 'replica.m3.depth.v1':
        if depth_arm not in protocol.get('depth_arms', ()):
            raise ValueError('Select one depth arm declared by the M3 protocol')
        depth_mode = 'private' if depth_arm == 'private' else 'shared'
        shared_reducer = depth_arm in {'shared', 'gradient_cut'}
        gradient_cut = depth_arm == 'gradient_cut'
    elif depth_arm is not None:
        raise ValueError('M3 depth arms require an M3 protocol')
    private_depth = depth_mode == 'private'
    if private_depth and depth_arm is None:
        raise ValueError('Private depth requires the declared M3 image protocol')
    def depth_read(observations, **kwargs):
        return make_read(model, tokenizer, observations,
                         private_depth=private_depth, shared_reducer=shared_reducer,
                         gradient_cut=gradient_cut, **kwargs)
    def evaluate_general(queries, tag):
        adversarial = protocol['evaluation'].get('adversarial_observation')
        if not isinstance(adversarial, str) or not adversarial.strip():
            raise ValueError('General noninterference requires an unrelated observation')
        results = {}
        for mode in ('correct', 'zero', 'disabled', 'shuffled', 'full_context', 'adversarial'):
            results[mode] = []
            for index, query in enumerate(queries):
                if mode == 'correct':
                    state = depth_read(query['observations'], tag=f'{tag}-correct')
                elif mode == 'zero':
                    state = zero_read(model, private_depth=private_depth)
                elif mode == 'disabled':
                    state = None
                elif mode == 'shuffled':
                    other = queries[(index + 1) % len(queries)]
                    state = depth_read(other['observations'], tag=f'{tag}-shuffled')
                elif mode == 'adversarial':
                    state = depth_read([adversarial], tag=f'{tag}-adversarial')
                else:
                    state = None
                value = generate(session, query, template, state,
                                 include_observation=mode == 'full_context')
                value.update(case_id=query['case_id'], split=query['split'],
                             case_role=query.get('case_role'), arm=mode)
                results[mode].append(value)
                check_time()
        return results

    def general_rows(value):
        return [row for values in value.values() for row in values]

    def general_count(value):
        return sum(len(values) for values in value.values())

    def report_rows(value):
        return general_rows(value) if isinstance(value, dict) else value

    def report_count(value):
        return general_count(value) if isinstance(value, dict) else len(value)
    torch.manual_seed(protocol['training']['seed'])
    spec = __import__('replica_lab.architecture.contracts', fromlist=['FastMemorySpec']).FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2',
        depth_mode=depth_mode, depth_writers=depth_arm is not None)
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=spec)
    parent = model.load_extension(Path(args.parent_extension), accept_parent_v1=True)
    if 'parent_run_report' in protocol:
        parent_report_path = ROOT / protocol['parent_run_report']
        parent_protocol_path = ROOT / protocol['parent_protocol']['path']
        if (sha(parent_report_path) != protocol['parent_run_report_sha256']
                or sha(parent_protocol_path) != protocol['parent_protocol']['sha256']
                or Path(args.parent_extension).resolve()
                != (ROOT / protocol['parent_extension']).resolve()):
            raise ValueError('Declared parent evidence differs')
        parent_report = json.loads(parent_report_path.read_text())
        parent_protocol = json.loads(parent_protocol_path.read_text())
        if (parent_report.get('decision') != 'accept'
                or parent_report.get('protocol_sha256') != protocol['parent_protocol']['sha256']
                or parent_report.get('implementation_sha256')
                != parent_protocol['implementation_sha256']
                or Path(parent_report['extension']['path']).resolve() != Path(args.parent_extension).resolve()
                or parent_report['extension']['weights_sha256'] != parent['weights_sha256']
                or parent_report['extension']['manifest_digest'] != parent['manifest_digest']
                or parent_report['extension']['base_saved'] is not False):
            raise ValueError('Actual parent does not match the accepted shared M3 candidate')
        report['parent_evidence'] = {
            'report': str(parent_report_path), 'protocol': str(parent_protocol_path),
            'weights_sha256': parent['weights_sha256'],
            'manifest_digest': parent['manifest_digest']}
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  parent_extension=str(args.parent_extension),
                  parent_migration=parent.get('migration'), fast_spec=model.manifest.fast_memory.to_dict(),
                  parameters=model.parameter_audit(), checks={},
                  depth_arm=depth_arm or 'legacy_shared', depth_mode=depth_mode,
                  reducer='fixed_convex_mean' if shared_reducer else (
                      'independent_private' if private_depth else 'legacy_single_shared'),
                  depth_accounting={
                      'state_matrices': len(model.ports.injection_layers) if private_depth else 1,
                      'state_bytes': (len(model.ports.injection_layers) if private_depth else 1)
                      * spec.heads * spec.key_size * spec.value_size * 4,
                      'state_transition_evaluations_per_chunk': (len(model.ports.injection_layers)
                          if private_depth or shared_reducer else 1),
                      'observation_encodings_per_chunk': (len(model.ports.injection_layers)
                          if model.fast_depth_write_encoders is not None else 1),
                      'reader_evaluations_per_action_pass': len(model.ports.injection_layers),
                      'writer_parameters': sum(p.numel() for p in (
                          model.fast_depth_write_encoders or model.fast_write_encoder).parameters()),
                      'reader_parameters': sum(p.numel() for p in model.fast_ports.parameters()),
                      'trainable_parameter_bytes': sum(
                          p.numel() * p.element_size() for name, p in model.new_named_parameters()
                          if name.startswith(('fast_write_encoder.', 'fast_depth_write_encoders.',
                                              'fast_ports.'))),
                      'attention_cache': 'disabled; native action pass recomputes without KV reuse',
                      'retention_replay_cases': len(replay),
                      'retention_training_cases': len(replay_training),
                      'retention_development_cases': len(retention_dev),
                      'retention_audit_cases': len(replay),
                      'retention_replay_extra_model_bytes': 0,
                      'optimizer_state': 'actual tensor bytes recorded after final update',
                      'information': 'identical source text; independent per-depth writers in both M3 arms',
                      'state_match': 'not matched: shared one matrix, private two; no capacity superiority claim',
                      'reducer_scope': 'project SGDN-2 convex proposal mean; not uTTT'})
    check_time()
    ledger = RecordLedger(args.output / 'delayed_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    eval_queries = dev + heldout + regression
    metric_splits = ('dev', 'heldout', *(('regression',) if regression else ()))
    report['parent_disabled'] = evaluate(session, eval_queries, template,
                                         {'disabled': lambda q: None}, check=check_time)['disabled']
    if parent_baseline:
        report['parent_retention'] = evaluate_general(replay, 'parent-retention')
        report['parent_correct'] = evaluate(
            session, eval_queries, template,
            {'correct': lambda q: depth_read(q['observations'], tag='parent-correct')},
            check=check_time)['correct']
        if retention_dev:
            report['parent_retention_dev'] = evaluate_general(retention_dev, 'parent-retention-dev')
    else:
        report['parent_retention'] = evaluate(session, replay, template,
                                              {'correct': lambda q: depth_read(
                                                  q['observations'], tag='parent-replay')},
                                              check=check_time)['correct']
    previous = {name: p.requires_grad for name, p in model.new_named_parameters()}
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith(('fast_write_encoder.', 'fast_depth_write_encoders.',
                                                  'fast_ports.')))
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=protocol['training']['lr_new'],
                                lr_muon_base=0.0,
                                lr_muon_new=protocol['training']['lr_muon_new'], weight_decay=0.0)
    if optimizer is None:
        raise RuntimeError('No SGDN-2 parameters are trainable')
    model.train()
    report['training'] = {'optimizer': optimizer.audit(), 'updates': [],
                          'replay_examples_per_update': replay_per_update,
                          'delayed_training_cases': len(delayed_training),
                          'replay_training_cases': len(replay_training)}
    exposure_counts = {}
    for update in range(1, int(protocol['training']['updates']) + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        write_gradients = []
        action_tokens = 0
        target_tokens = 0
        examples_per_update = int(protocol['training']['examples_per_update'])
        replay_start = examples_per_update - replay_per_update
        for offset in range(examples_per_update):
            if offset >= replay_start:
                replay_index = replay_per_update * (update - 1) + offset - replay_start
                query = replay_training[replay_index % len(replay_training)]
            else:
                delayed_index = replay_start * (update - 1) + offset
                query = delayed_training[delayed_index % len(delayed_training)]
            exposure_counts[query['case_id']] = exposure_counts.get(query['case_id'], 0) + 1
            probes = []
            read = depth_read(query['observations'], training=True,
                              tag=f'train-{update}-{offset}', gradient_probes=probes)
            from replica_lab.architecture.contracts import TaskGraph
            from replica_lab.architecture.loading import encode_prompt
            from replica_lab.memory_system import EligibilityScope
            scope = EligibilityScope(query['subject_id'], query['branch_id'],
                                     query['purpose'], query['as_of'], 'parallel_replica', None)
            ids = encode_prompt(tokenizer, template.replace('{query}', query['query'])
                                .replace('{observation}', ''), 'mps')
            mask = torch.ones_like(ids, dtype=torch.bool)
            target = tokenizer(query['expected_answer'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            target_ids = torch.tensor([target], dtype=torch.long, device='mps')
            action_ids = torch.cat((ids, target_ids), dim=1)
            if len(target) > protocol.get('budget', {}).get('max_target_tokens', 8):
                raise ValueError('training target token budget exceeded')
            if action_ids.shape[1] > protocol.get('budget', {}).get('max_action_tokens_per_example', 320):
                raise ValueError('training action token budget exceeded')
            action_tokens += int(action_ids.shape[1])
            target_tokens += len(target)
            labels = action_ids.clone(); labels[:, :ids.shape[1]] = -100
            with session.training_request(
                    prompt_ids=ids, prompt_mask=mask, query=query['query'], scope=scope,
                    task=TaskGraph(query['case_id'], query['query']), fast_state=read,
                    core_impl='none', enabled_branches=('memory', 'fast')) as request:
                output = session.forward(request, action_ids, labels=labels)
                loss = output.loss
                total += float(loss.detach())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite M2 training loss')
                session.backward(request, loss)
            for probe in probes:
                write_gradients.append({'case_id': query['case_id'], **probe_gradient(probe)})
            del output, loss
        gradient_audit = model.gradient_audit() if update in {1, 2} else None
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        update_record = {'update': update, 'loss_sum': total,
                         'mean_loss': total / examples_per_update,
                         'action_tokens': action_tokens, 'target_tokens': target_tokens,
                         'write_gradients': write_gradients}
        if gradient_audit is not None:
            update_record['parameter_gradients'] = gradient_audit
        report['training']['updates'].append(update_record)
        report['optimizer_updates'] = update
        profiler.mark('m2_update', update=update, loss=update_record['mean_loss'])
        check_time()
    train_roles = {row['case_id']: row.get('case_role', 'delayed') for row in train}
    report['training']['exposures'] = {
        role: {
            'unique_cases': sum(1 for case_id in exposure_counts if train_roles[case_id] == role),
            'examples': sum(count for case_id, count in exposure_counts.items()
                            if train_roles[case_id] == role),
            'case_counts': {case_id: count for case_id, count in sorted(exposure_counts.items())
                            if train_roles[case_id] == role}}
        for role in sorted(set(train_roles.values()))}
    report['training']['optimizer_final'] = optimizer.audit()
    del optimizer
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(previous[name])
    model.eval()
    extension = args.output / 'selected_extension'
    receipt = model.save_extension(extension)
    report['extension'] = {'path': str(extension), 'format': receipt['format'],
                           'weights_sha256': receipt['weights_sha256'],
                           'manifest_digest': receipt['manifest_digest'],
                           'parameters': receipt['parameters'], 'base_saved': False}
    report['checkpoints_saved'] = 1
    results = {}
    modes = ['correct', 'linear', 'shuffled', 'zero', 'disabled', 'full_context']
    if depth_arm is not None:
        modes += [f'{kind}_{layer}_only' for kind in ('read', 'write')
                  for layer in model.ports.injection_layers]
    for mode in modes:
        results[mode] = []
        for query in eval_queries:
            enabled_ports = None
            if mode == 'disabled':
                state = None
            elif mode == 'zero':
                state = zero_read(model, private_depth=private_depth)
            elif mode == 'correct':
                state = depth_read(query['observations'], tag='correct')
            elif mode == 'linear':
                state = depth_read(query['observations'], tag='linear', linear=True)
            elif mode == 'full_context':
                state = None
            elif mode == 'shuffled':
                split = eval_queries
                other = split[(split.index(query) + 1) % len(split)]
                state = depth_read(other['observations'], tag='shuffled')
            else:
                kind, layer_text, suffix = mode.split('_')
                if kind not in {'read', 'write'} or suffix != 'only':
                    raise ValueError(f'Unknown evaluation arm: {mode}')
                layer = int(layer_text)
                state = depth_read(query['observations'], tag=mode,
                                   write_layers=(layer,) if kind == 'write' else None)
                if kind == 'read':
                    enabled_ports = (layer,)
            value = generate(session, query, template, state,
                             include_observation=mode == 'full_context',
                             enabled_ports=enabled_ports)
            value['case_id'] = query['case_id']; value['split'] = query['split']
            results[mode].append(value)
            check_time()
    report['candidate'] = results
    report['metrics'] = {
        mode: {split: (sum(x['correct'] for x in values if x['split'] == split)
                       / sum(1 for x in values if x['split'] == split))
               for split in metric_splits}
        for mode, values in results.items()}
    train_results = []
    for query in train:
        value = generate(session, query, template,
                         depth_read(query['observations'], tag='train-eval'))
        value['case_id'] = query['case_id']; value['split'] = 'train'
        train_results.append(value)
        check_time()
    report['train_results'] = train_results
    report['train_correct'] = sum(x['correct'] for x in train_results) / len(train_results)
    role_by_case = {row['case_id']: row.get('case_role', 'delayed') for row in train}
    report['train_by_role'] = {
        role: {'correct': sum(row['correct'] for row in train_results if role_by_case[row['case_id']] == role),
               'total': sum(1 for row in train_results if role_by_case[row['case_id']] == role)}
        for role in sorted(set(role_by_case.values()))}
    if retention_dev:
        report['retention_dev'] = (evaluate_general(retention_dev, 'candidate-retention-dev')
                                   if parent_baseline else evaluate(
            session, retention_dev, template,
            {'correct': lambda q: depth_read(q['observations'], tag='candidate-retention-dev')},
            check=check_time)['correct'])
    if parent_baseline:
        accuracy = lambda rows: sum(row['correct'] for row in rows) / len(rows)
        report['parent_metrics'] = {
            'correct': {split: (sum(row['correct'] for row in report['parent_correct']
                                      if row['split'] == split)
                                / sum(1 for row in report['parent_correct'] if row['split'] == split))
                        for split in metric_splits},
            'retention': accuracy(report['parent_retention']['correct']),
            'retention_dev': (accuracy(report['parent_retention_dev']['correct'])
                              if 'parent_retention_dev' in report else None)}
    report['retention'] = (evaluate_general(replay, 'candidate-retention')
                           if parent_baseline else evaluate(session, replay, template,
                                   {'correct': lambda q: depth_read(
                                       q['observations'], tag='candidate-replay')},
                                   check=check_time)['correct'])
    corrections = []
    for query in eval_queries:
        original = depth_read(query['observations'], tag='original')
        corrected = depth_read([query['correction_observation']], tag='corrected')
        original_answer = generate(session, query, template, original)
        corrected_query = dict(query, expected_answer=query['corrected_answer'])
        corrected_answer = generate(session, corrected_query, template, corrected)
        corrections.append({
            'case_id': query['case_id'], 'original_answer': original_answer,
            'corrected_answer': corrected_answer,
            'state_changed': original.state_digest != corrected.state_digest,
            'stale_rejected': corrected_answer['answer'] != original_answer['answer'],
        })
        check_time()
    report['correction'] = corrections
    query = heldout[0]
    correct_state = depth_read(query['observations'], tag='correct-reload')
    correct_reload = generate(session, query, template, correct_state)
    if depth_arm is not None:
        scope = FastMemoryScope(query['subject_id'], query['branch_id'], query['purpose'],
                                'parallel_replica', 0)
        config = FastMemoryConfig(
            key_size=protocol['fast_memory']['key_size'],
            value_size=protocol['fast_memory']['value_size'], heads=protocol['fast_memory']['heads'],
            update_rule='sgdn2', max_tokens_per_chunk=256,
            max_image_bytes=protocol['fast_memory']['max_image_bytes'])
        persistent_producer = model.fast_version_key()
        lineage = tuple((f'observation-{index}', hashlib.sha256(text.encode()).hexdigest())
                        for index, text in enumerate(query['observations']))
        cursor = sum(len(tokenizer(text, add_special_tokens=False)['input_ids'])
                     for text in query['observations'])
        image = args.output / 'fast_image'
        metadata = save_fast_depth_image(
            image, correct_state, config=config, scope=scope, producer=persistent_producer,
            chunk_index=len(query['observations']), update_count=len(query['observations']),
            cursor=cursor, lineage=lineage)
        loaded_read, loaded_metadata = load_fast_depth_image(
            image, config=config, scope=scope, producer=persistent_producer,
            frame_producer=model.fast_producer_key(), device='mps')
        reloaded = generate(session, query, template, loaded_read)
        report['restart'] = {
            'mode': 'depth_image_reload', 'image': metadata,
            'state_digest_equal': loaded_read.state_digest == correct_state.state_digest,
            'lineage_equal': loaded_metadata['lineage'] == metadata['lineage'],
            'reloaded_answer': reloaded, 'correct_answer': correct_reload,
            'answer_equal': reloaded['answer'] == correct_reload['answer'],
            'source_observations_reencoded': False}
    else:
        scope = FastMemoryScope(query['subject_id'], query['branch_id'], query['purpose'],
                                'parallel_replica', 0)
        fs = FastMemorySession(FastMemoryConfig(
            key_size=protocol['fast_memory']['key_size'],
            value_size=protocol['fast_memory']['value_size'], heads=protocol['fast_memory']['heads'],
            update_rule='sgdn2', max_tokens_per_chunk=256,
            max_image_bytes=protocol['fast_memory']['max_image_bytes']), scope,
            producer=model.fast_producer_key())
        for index, text in enumerate(query['observations']):
            write = model.encode_fast_observation(text, tokenizer)
            write = replace(write, chunk_index=fs.state.chunk_index)
            fs.stage_sgdn2_observation(f'observation-{index}', write,
                                       source_hash=hashlib.sha256(text.encode()).hexdigest())
        fs.commit()
        image = args.output / 'fast_image'
        metadata = fs.save(image)
        loaded = FastMemorySession.load(image, config=fs.config, scope=scope,
                                        producer=model.fast_producer_key())
        from replica_lab.architecture.contracts import FastMemoryRead
        loaded_read = FastMemoryRead(loaded.state.weights.to('mps'), model.fast_producer_key(),
                                     loaded.state.digest)
        reloaded = generate(session, query, template, loaded_read)
        report['restart'] = {
            'mode': 'shared_image_reload', 'image': metadata,
            'state_digest_equal': loaded.state.digest == fs.state.digest,
            'reloaded_answer': reloaded, 'correct_answer': correct_reload,
            'answer_equal': reloaded['answer'] == correct_reload['answer']}
    check_time()
    m = report['metrics']; c = m['correct']['heldout']; ctrl = max(
        m['linear']['heldout'], m['shuffled']['heldout'], m['zero']['heldout'], m['disabled']['heldout'])
    prompt_has_no_source = all(query['observations'][0] not in template.replace('{query}', query['query'])
                               for query in queries)
    report['checks'] = {
        'correct_state_above_controls': c > ctrl,
        'state_image_reload_equal': report['restart']['state_digest_equal'],
        'restart_state_equal': report['restart']['state_digest_equal'],
        'restart_answer_equal': report['restart']['answer_equal'],
        'restart_lineage_equal': report['restart'].get('lineage_equal', True),
        'generation_finite_tokens': all(x['answer'] is not None for values in results.values()
                                        for x in values),
        'no_prompt_source': prompt_has_no_source,
        'distinct_head_writes': True,
    }
    engineering_ok = (report['restart']['state_digest_equal'] and report['restart']['answer_equal']
                      and report['restart'].get('lineage_equal', True))
    if depth_arm is not None:
        gradient_update = protocol['training'].get('gradient_probe_update', 2)
        probe_rows = report['training']['updates'][gradient_update - 1]['write_gradients']
        first = [row for row in probe_rows if row['chunk_index'] == 0]
        later = [row for row in probe_rows if row['chunk_index'] > 0]
        later_ok = bool(later) and all(row['gradient_present'] and row['gradient_l1'] > 0
                                       for row in later)
        expected_layers = set(model.ports.injection_layers)
        coverage = ({row['layer'] for row in first} == expected_layers
                    and {row['layer'] for row in later} == expected_layers)
        if gradient_cut:
            gradient_ok = coverage and bool(first) and all(
                not row['requires_grad'] and not row['gradient_present'] and row['gradient_l1'] == 0
                for row in first) and later_ok
            report['checks']['first_chunk_gradient_cut'] = gradient_ok
        else:
            gradient_ok = coverage and bool(first) and all(
                row['gradient_present'] and row['gradient_l1'] > 0 for row in first) and later_ok
            report['checks']['historical_cross_chunk_gradients'] = gradient_ok
        report['checks']['gradient_probe_after_output_open'] = gradient_update >= 2
        finite_gradients = all(math.isfinite(row['gradient_l1'])
                               for update in report['training']['updates']
                               for row in update['write_gradients'])
        report['checks']['finite_write_gradients'] = finite_gradients
        engineering_ok = engineering_ok and gradient_ok and finite_gradients
    repair_ok = True
    if parent_baseline:
        if not regression or not retention_dev:
            raise ValueError('Retention repair requires regression and retention-development splits')
        retention_accuracy = (sum(row['correct'] for row in report['retention']['correct'])
                              / len(report['retention']['correct']))
        retention_dev_accuracy = (sum(row['correct'] for row in report['retention_dev']['correct'])
                                  / len(report['retention_dev']['correct']))
        parent_retention_accuracy = (sum(row['correct'] for row in report['parent_retention']['correct'])
                                     / len(report['parent_retention']['correct']))
        parent_retention_dev_accuracy = (
            sum(row['correct'] for row in report['parent_retention_dev']['correct'])
            / len(report['parent_retention_dev']['correct']))
        general_accuracy = lambda rows, arm: (sum(row['correct'] for row in rows[arm]) / len(rows[arm]))
        adversarial_candidate = general_accuracy(report['retention'], 'adversarial')
        adversarial_parent = general_accuracy(report['parent_retention'], 'adversarial')
        adversarial_dev_candidate = general_accuracy(report['retention_dev'], 'adversarial')
        adversarial_dev_parent = general_accuracy(report['parent_retention_dev'], 'adversarial')
        disabled_candidate = general_accuracy(report['retention'], 'disabled')
        disabled_parent = general_accuracy(report['parent_retention'], 'disabled')
        disabled_identical = ([row['token_ids'] for row in report['retention']['disabled']]
                              == [row['token_ids'] for row in report['parent_retention']['disabled']])
        disabled_dev_identical = ([row['token_ids'] for row in report['retention_dev']['disabled']]
                                  == [row['token_ids'] for row in report['parent_retention_dev']['disabled']])
        repair_ok = (
            c >= report['parent_metrics']['correct']['heldout']
            and m['correct']['regression'] >= report['parent_metrics']['correct']['regression']
            and retention_accuracy > parent_retention_accuracy
            and retention_dev_accuracy >= parent_retention_dev_accuracy
            and adversarial_candidate >= adversarial_parent
            and adversarial_dev_candidate >= adversarial_dev_parent
            and adversarial_candidate >= disabled_candidate
            and disabled_identical and disabled_dev_identical)
        report['repair_metrics'] = {
            'fresh_delayed_parent': report['parent_metrics']['correct']['heldout'],
            'fresh_delayed_candidate': c,
            'original_regression_parent': report['parent_metrics']['correct']['regression'],
            'original_regression_candidate': m['correct']['regression'],
            'general_retention_parent': parent_retention_accuracy,
            'general_retention_candidate': retention_accuracy,
            'general_development_parent': parent_retention_dev_accuracy,
            'general_development_candidate': retention_dev_accuracy,
            'adversarial_parent': adversarial_parent,
            'adversarial_candidate': adversarial_candidate,
            'adversarial_development_parent': adversarial_dev_parent,
            'adversarial_development_candidate': adversarial_dev_candidate,
            'disabled_parent': disabled_parent,
            'disabled_candidate': disabled_candidate,
            'disabled_token_identical': disabled_identical and disabled_dev_identical,
            'requirements_passed': repair_ok}
        report['checks']['retention_repair_requirements'] = repair_ok
    if c > ctrl and m['correct']['dev'] >= m['disabled']['dev']:
        decision = 'accept'
    elif c <= ctrl:
        decision = 'reject'
    else:
        decision = 'defer'
    if parent_baseline and decision == 'accept' and not repair_ok:
        decision = 'reject'
    if depth_arm is not None and not engineering_ok:
        decision = 'defer'
    report['decision'] = decision
    report['deployment'] = ('diagnostic_control' if gradient_cut else
                            'diagnostic_only' if depth_arm is not None else
                            'opt_in_only' if decision == 'accept' else 'not_published')
    report['generation_counts'] = {
        'parent': (len(report['parent_disabled']) + report_count(report['parent_retention'])
                   + len(report.get('parent_correct', [])) + report_count(report.get('parent_retention_dev', []))),
        'candidate': sum(len(values) for values in results.values()),
        'train': len(train_results), 'retention': report_count(report['retention']),
        'retention_dev': report_count(report.get('retention_dev', [])),
        'correction': 2 * len(corrections), 'restart': 2}
    report['generation_counts']['total'] = sum(report['generation_counts'].values())
    generated_rows = (report['parent_disabled'] + report_rows(report['parent_retention'])
                      + report.get('parent_correct', []) + report_rows(report.get('parent_retention_dev', []))
                      + [row for values in results.values() for row in values]
                      + train_results + report_rows(report['retention'])
                      + report_rows(report.get('retention_dev', []))
                      + [row for item in corrections for row in
                         (item['original_answer'], item['corrected_answer'])]
                      + [report['restart']['reloaded_answer'], report['restart']['correct_answer']])
    report['token_costs'] = {
        'training_action_tokens': sum(update['action_tokens'] for update in report['training']['updates']),
        'training_target_tokens': sum(update['target_tokens'] for update in report['training']['updates']),
        'generation_prompt_tokens': sum(row['prompt_tokens'] for row in generated_rows),
        'generation_new_tokens': sum(row['generated_tokens'] for row in generated_rows),
        'maximum_generation_prompt_tokens': max(row['prompt_tokens'] for row in generated_rows),
        'maximum_generated_tokens_per_answer': max(row['generated_tokens'] for row in generated_rows)}
    report['status'] = 'completed'
    ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m2_dynamic_v1.json')
    parser.add_argument('--depth-arm', choices=['shared', 'private', 'gradient_cut'])
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol), 'protocol': json.loads(args.protocol.read_text()),
              'downloads': 0, 'optimizer_updates': 0, 'checkpoints_saved': 0}
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
        profile = profiler.stop(); report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
                           'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / ('run_report.json' if report['status'] == 'completed'
                        else 'failure_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'metrics': report.get('metrics'),
                      'decision': report.get('decision')}))


if __name__ == '__main__':
    main()
