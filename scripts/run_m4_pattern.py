#!/usr/bin/env python3
"""M4 static causal n-gram pattern path and deterministic routing on MPS."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
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



def collect_query_hashes(model, ids, mask, hash_counts, hash_ngrams=None):
    result = set()
    for order, head, index, available in model.pattern_port.hash_indices(ids, mask):
        for batch, position in available.nonzero().tolist():
            value = int(index[batch, position])
            key = (order, head, value)
            result.add(key)
            hash_counts[key] = hash_counts.get(key, 0) + 1
            if hash_ngrams is not None:
                ngram = tuple(int(token) for token in ids[batch, position - order + 1:position + 1].tolist())
                hash_ngrams.setdefault(key, set()).add(ngram)
    return result


def summarize_collisions(spec, counts, ngrams):
    result = {}
    for order in spec.orders:
        for head in range(spec.heads):
            keys = [key for key in counts if key[0] == order and key[1] == head]
            lookups = sum(counts[key] for key in keys)
            distinct = sum(len(ngrams.get(key, ())) for key in keys)
            colliding = [key for key in keys if len(ngrams.get(key, ())) > 1]
            collision_ngrams = sum(len(ngrams[key]) - 1 for key in colliding)
            repeated = lookups - distinct
            examples = [{'order': key[0], 'head': key[1], 'index': key[2],
                         'ngrams': [list(value) for value in sorted(ngrams[key])]}
                        for key in sorted(colliding)[:4]]
            result[f'{order}_{head}'] = {
                'lookups': lookups,
                'unique_indices': len(keys),
                'distinct_ngrams': distinct,
                'repeated_lookups': repeated,
                'repeat_rate': repeated / lookups if lookups else 0.0,
                'colliding_indices': len(colliding),
                'collision_ngrams': collision_ngrams,
                'collision_rate': collision_ngrams / distinct if distinct else 0.0,
                'table_rows': spec.table_rows,
                'hash_coverage': len(keys) / spec.table_rows,
                'touched_bytes': len(keys) * spec.width * 4,
                'collision_examples': examples,
            }
    return result


def accuracy(rows):
    return sum(row['correct'] for row in rows) / len(rows)


def split_metrics(rows, splits):
    return {split: accuracy([row for row in rows if row['split'] == split]) for split in splits}


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.contracts import (FastMemorySpec, PatternRead, PatternSpec,
                                                   TaskGraph)
    from replica_lab.architecture.loading import encode_prompt, load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.fast_memory import (FastMemoryConfig, FastMemoryScope,
                                                  save_fast_depth_image, load_fast_depth_image)
    from replica_lab.runtime.routing import choose_route
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    implementation_pins = protocol['implementation_sha256']
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M4 implementation bytes changed: {name}')
    report['implementation_sha256'] = implementation_pins
    max_wall = protocol['budget']['max_wall_seconds_per_arm']
    def check_time():
        if time.monotonic() - started > max_wall:
            raise TimeoutError('M4 wall-time budget exceeded')
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'), ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'M4 dataset changed: {filename}')
    queries = rows(data / 'queries.jsonl')
    train = [q for q in queries if q['split'] == 'train' and q.get('case_role') == 'delayed_training']
    dev = [q for q in queries if q['split'] == 'dev']
    heldout = [q for q in queries if q['split'] == 'heldout']
    regression = [q for q in queries if q['split'] == 'regression']
    retention_dev = [q for q in queries if q['split'] == 'retention_dev']
    replay = [q for q in queries if q['split'] == 'retention']
    eval_queries = dev + heldout + regression
    metric_splits = ('dev', 'heldout', 'regression')
    if (len(train), len(dev), len(heldout), len(regression), len(retention_dev), len(replay)) != (24, 4, 8, 8, 2, 4):
        raise ValueError('M4 split sizes differ')
    parent_report_path = ROOT / protocol['parent_run_report']
    parent_protocol_path = ROOT / protocol['parent_protocol']['path']
    if (sha(parent_report_path) != protocol['parent_run_report_sha256']
            or sha(parent_protocol_path) != protocol['parent_protocol']['sha256']
            or Path(args.parent_extension).resolve() != (ROOT / protocol['parent_extension']).resolve()):
        raise ValueError('M4 parent evidence differs')
    parent_report = json.loads(parent_report_path.read_text())
    parent_protocol = json.loads(parent_protocol_path.read_text())
    if (parent_report.get('decision') != 'accept'
            or parent_report.get('protocol_sha256') != protocol['parent_protocol']['sha256']
            or parent_report.get('implementation_sha256') != parent_protocol['implementation_sha256']):
        raise ValueError('M4 parent is not the accepted shared candidate')
    parent_controls = None
    if protocol.get('parent_controls'):
        control_path = ROOT / protocol['parent_controls']['path']
        if sha(control_path) != protocol['parent_controls']['sha256']:
            raise ValueError('M4 parent-control bytes differ')
        parent_controls = json.loads(control_path.read_text())
        control_report_path = Path(parent_controls.get('parent_run_report')
                                   or parent_controls['parent_evidence']['report'])
        control_report_sha = parent_controls.get('parent_run_report_sha256', sha(parent_report_path))
        if (control_report_path.resolve() != parent_report_path.resolve()
                or control_report_sha != protocol['parent_run_report_sha256']
                or Path(parent_controls['parent_extension']).resolve()
                != Path(args.parent_extension).resolve()):
            raise ValueError('M4 parent-control lineage differs')
        for split, expected in (('retention', 4), ('retention_dev', 2)):
            if len(parent_controls[f'parent_{split}']['adversarial']) != expected:
                raise ValueError('M4 parent-control adversarial count differs')
        report['parent_controls'] = {'path': str(control_path),
                                     'sha256': protocol['parent_controls']['sha256'],
                                     'generated_in_this_run': False}
    torch.manual_seed(protocol['training']['seed'])
    fast_spec = FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2',
        depth_mode='shared', depth_writers=True)
    pattern_spec = PatternSpec(**protocol['pattern']['spec'])
    hash_counts, hash_ngrams = {}, {}
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=fast_spec,
                                            pattern=pattern_spec)
    parent = model.load_extension(Path(args.parent_extension), accept_parent_pattern=True)
    if parent_report['extension']['weights_sha256'] != parent['weights_sha256']:
        raise ValueError('M4 parent extension bytes differ')
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  parent_extension=str(args.parent_extension), parent_migration=parent.get('migration'),
                  fast_spec=fast_spec.to_dict(), pattern_spec=protocol['pattern'],
                  parameters=model.parameter_audit(), checks={})
    ledger = RecordLedger(args.output / 'pattern_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    def depth_read(observations, **kwargs):
        return make_read(model, tokenizer, observations, shared_reducer=True, **kwargs)
    def pattern_read():
        return PatternRead(model.pattern_producer_key(), protocol['pattern']['version'])
    def query_hashes(ids, mask):
        return collect_query_hashes(model, ids, mask, hash_counts, hash_ngrams)
    trained_hashes = set()
    for query in train:
        text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
        ids = encode_prompt(tokenizer, text, 'mps')
        trained_hashes.update(query_hashes(ids, torch.ones_like(ids, dtype=torch.bool)))
    training_hash_counts = dict(hash_counts)
    training_hash_ngrams = {key: set(values) for key, values in hash_ngrams.items()}
    report['trained_hash_count'] = len(trained_hashes)
    report['trained_hash_ids'] = sorted(trained_hashes)
    def generate_mode(query, mode, *, other=None, enabled_ports=None):
        if mode == 'pattern':
            fast, pattern = None, pattern_read()
        elif mode == 'both':
            fast, pattern = depth_read(query['observations']), pattern_read()
        elif mode == 'router':
            text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
            ids = encode_prompt(tokenizer, text, 'mps')
            choice = choose_route(case_role=query.get('case_role', 'delayed_training'),
                                  query_hashes=frozenset(query_hashes(ids, torch.ones_like(ids, dtype=torch.bool))),
                                  trained_hashes=frozenset(trained_hashes))
            if choice.kind == 'pattern':
                fast, pattern = None, pattern_read()
            elif choice.kind == 'fast':
                fast, pattern = depth_read(query['observations']), None
            else:
                raise ValueError(f'Unexpected M4 route: {choice.kind}')
            value = generate(session, query, DELAYED_USE_TEMPLATE, fast, pattern=pattern,
                             enabled_ports=enabled_ports)
            value['route'] = choice.kind
            value['route_reason'] = choice.reason
            value['route_known_hashes'] = choice.known_hashes
            value['route_total_hashes'] = choice.total_hashes
            return value
        elif mode == 'fast':
            fast, pattern = depth_read(query['observations']), None
        elif mode == 'linear':
            fast, pattern = depth_read(query['observations'], linear=True), None
        elif mode == 'shuffled':
            fast, pattern = depth_read(other['observations']), None
        elif mode == 'zero':
            fast, pattern = zero_read(model), None
        elif mode == 'disabled':
            fast, pattern = None, None
        elif mode == 'full_context':
            fast, pattern = None, None
        elif mode.startswith('read_'):
            layer = int(mode.split('_')[1])
            fast, pattern = depth_read(query['observations']), None
            enabled_ports = (layer,)
        elif mode.startswith('write_'):
            layer = int(mode.split('_')[1])
            fast, pattern = depth_read(query['observations'], write_layers=(layer,)), None
        else:
            raise ValueError(f'Unknown M4 mode: {mode}')
        return generate(session, query, DELAYED_USE_TEMPLATE, fast, pattern=pattern,
                        include_observation=mode == 'full_context', enabled_ports=enabled_ports)
    def evaluate_main(queries, modes):
        results = {}
        for mode in modes:
            results[mode] = []
            for index, query in enumerate(queries):
                other = queries[(index + 1) % len(queries)] if mode == 'shuffled' else None
                value = generate_mode(query, mode, other=other)
                value.update(case_id=query['case_id'], split=query['split'],
                             case_role=query.get('case_role'), audit_stratum=query.get('audit_stratum'))
                results[mode].append(value)
                check_time()
        return results
    def evaluate_general(queries, modes):
        results = {}
        adversarial = protocol['evaluation']['adversarial_observation']
        for mode in modes:
            results[mode] = []
            for index, query in enumerate(queries):
                route = None
                if mode == 'pattern':
                    fast, pattern = None, pattern_read()
                elif mode == 'router':
                    text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
                    ids = encode_prompt(tokenizer, text, 'mps')
                    choice = choose_route(case_role=query.get('case_role', 'delayed_training'),
                                          query_hashes=frozenset(query_hashes(ids, torch.ones_like(ids, dtype=torch.bool))),
                                          trained_hashes=frozenset(trained_hashes))
                    if choice.kind == 'pattern':
                        fast, pattern = None, pattern_read()
                    elif choice.kind == 'fast':
                        fast, pattern = depth_read(query['observations']), None
                    else:
                        raise ValueError(f'Unexpected M4 general route: {choice.kind}')
                    route = choice
                elif mode == 'fast':
                    fast, pattern = depth_read(query['observations']), None
                elif mode == 'zero':
                    fast, pattern = zero_read(model), None
                elif mode == 'disabled':
                    fast, pattern = None, None
                elif mode == 'shuffled':
                    fast, pattern = depth_read(queries[(index + 1) % len(queries)]['observations']), None
                elif mode == 'adversarial':
                    fast = depth_read([adversarial])
                    pattern = (pattern_read() if protocol['evaluation'].get('adversarial_uses_pattern', False)
                               else None)
                elif mode == 'full_context':
                    fast, pattern = None, None
                else:
                    raise ValueError(f'Unknown M4 general mode: {mode}')
                value = generate(session, query, DELAYED_USE_TEMPLATE, fast, pattern=pattern,
                                 include_observation=mode == 'full_context')
                value.update(case_id=query['case_id'], split=query['split'],
                             case_role=query.get('case_role'), arm=mode)
                if route is not None:
                    value.update(route=route.kind, route_reason=route.reason,
                                 route_known_hashes=route.known_hashes, route_total_hashes=route.total_hashes)
                results[mode].append(value)
                check_time()
        return results
    report['parent'] = {
        'main': evaluate_main(eval_queries, ('fast', 'disabled')),
        'retention': evaluate_general(replay, ('fast', 'disabled')),
        'retention_dev': evaluate_general(retention_dev, ('fast', 'disabled')),
    }
    previous = {name: parameter.requires_grad for name, parameter in model.new_named_parameters()}
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith('pattern_port.'))
    source_lm_updates = int(protocol['training'].get('source_lm_updates', 0))
    report['training'] = {}
    if source_lm_updates:
        source_items = [(query, text) for query in train for text in query['observations']]
        source_optimizer = build_optimizer(
            model, 'muon8bit', lr_base=0.0, lr_new=protocol['training']['lr_new'],
            lr_muon_base=0.0, lr_muon_new=protocol['training']['lr_muon_new'], weight_decay=0.0)
        if source_optimizer is None:
            raise RuntimeError('M4 source-LM optimizer has no trainable parameters')
        model.train()
        report['training']['source_lm'] = {'updates': []}
        for update in range(1, source_lm_updates + 1):
            source_optimizer.zero_grad(set_to_none=True)
            losses = []
            source_tokens = 0
            for offset in range(int(protocol['training']['source_lm_examples_per_update'])):
                query, text = source_items[(12 * (update - 1) + offset) % len(source_items)]
                token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
                if len(token_ids) > protocol['budget']['max_source_tokens_per_chunk']:
                    raise ValueError('M4 source-LM chunk exceeds token bound')
                source_ids = torch.tensor([token_ids], dtype=torch.long, device='mps')
                if tokenizer.bos_token_id is None:
                    raise ValueError('M4 source-LM requires the sealed BOS token')
                prefix = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device='mps')
                ids = torch.cat((prefix, source_ids), dim=1)
                labels = ids.clone()
                labels[:, :prefix.shape[1]] = -100
                scope = EligibilityScope(query['subject_id'], query['branch_id'], query['purpose'],
                                         query['as_of'], 'parallel_replica', None)
                with session.training_request(
                        prompt_ids=prefix, prompt_mask=torch.ones_like(prefix, dtype=torch.bool),
                        query=query['query'], scope=scope,
                        task=TaskGraph(query['case_id'], query['query']), fast_state=None,
                        pattern=pattern_read(), core_impl='none', enabled_branches=('memory', 'pattern')) as request:
                    output = session.forward(request, ids, labels=labels)
                    loss = output.loss
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('nonfinite M4 source-LM loss')
                    session.backward(request, loss)
                losses.append(float(loss.detach()))
                source_tokens += len(token_ids)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           1.0, error_if_nonfinite=True)
            source_optimizer.step(); source_optimizer.zero_grad(set_to_none=True)
            report['training']['source_lm']['updates'].append({
                'update': update, 'loss_sum': sum(losses), 'mean_loss': sum(losses) / len(losses),
                'source_tokens': source_tokens})
            report['optimizer_updates'] += 1
            profiler.mark('m4_source_lm_update', update=update, loss=report['training']['source_lm']['updates'][-1]['mean_loss'])
            check_time()
        report['training']['source_lm']['optimizer_final'] = source_optimizer.audit()
        warmup_extension = args.output / 'warmup_extension'
        warmup_receipt = model.save_extension(warmup_extension)
        report['checkpoints_saved'] += 1
        report['training']['source_lm']['checkpoint'] = {
            'path': str(warmup_extension), 'format': warmup_receipt['format'],
            'weights_sha256': warmup_receipt['weights_sha256'],
            'manifest_digest': warmup_receipt['manifest_digest'], 'base_saved': False}
        del source_optimizer
        torch.manual_seed(protocol['training']['seed'])
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=protocol['training']['lr_new'],
                                lr_muon_base=0.0, lr_muon_new=protocol['training']['lr_muon_new'],
                                weight_decay=0.0)
    if optimizer is None:
        raise RuntimeError('M4 pattern optimizer has no trainable parameters')
    model.train()
    report['training'].update(optimizer=optimizer.audit(), updates=[])
    for update in range(1, int(protocol['training']['updates']) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        action_tokens = target_tokens = 0
        for offset in range(int(protocol['training']['examples_per_update'])):
            query = train[(4 * (update - 1) + offset) % len(train)]
            ids = encode_prompt(tokenizer, DELAYED_USE_TEMPLATE.replace('{query}', query['query'])
                                .replace('{observation}', ''), 'mps')
            mask = torch.ones_like(ids, dtype=torch.bool)
            target = tokenizer(query['expected_answer'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            target_ids = torch.tensor([target], dtype=torch.long, device='mps')
            action_ids = torch.cat((ids, target_ids), dim=1)
            if action_ids.shape[1] > protocol['budget']['max_action_tokens_per_example']:
                raise ValueError('M4 action token budget exceeded')
            labels = action_ids.clone(); labels[:, :ids.shape[1]] = -100
            scope = EligibilityScope(query['subject_id'], query['branch_id'], query['purpose'],
                                     query['as_of'], 'parallel_replica', None)
            with session.training_request(
                    prompt_ids=ids, prompt_mask=mask, query=query['query'], scope=scope,
                    task=TaskGraph(query['case_id'], query['query']), fast_state=None,
                    pattern=pattern_read(), core_impl='none', enabled_branches=('memory', 'pattern')) as request:
                output = session.forward(request, action_ids, labels=labels)
                loss = output.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite M4 loss')
                session.backward(request, loss)
            losses.append(float(loss.detach()))
            action_tokens += int(action_ids.shape[1]); target_tokens += len(target)
        gradient_audit = model.gradient_audit() if update in {1, 2} else None
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       1.0, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        record = {'update': update, 'loss_sum': sum(losses), 'mean_loss': sum(losses) / len(losses),
                  'action_tokens': action_tokens, 'target_tokens': target_tokens}
        if gradient_audit is not None:
            record['parameter_gradients'] = gradient_audit
        report['training']['updates'].append(record)
        report['optimizer_updates'] += 1
        profiler.mark('m4_update', update=update, loss=record['mean_loss'])
        check_time()
    report['training']['optimizer_final'] = optimizer.audit()
    del optimizer
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(previous[name])
    model.eval()
    extension = args.output / 'selected_extension'
    receipt = model.save_extension(extension)
    report['checkpoints_saved'] += 1
    report['extension'] = {'path': str(extension), 'format': receipt['format'],
                           'weights_sha256': receipt['weights_sha256'],
                           'manifest_digest': receipt['manifest_digest'],
                           'parameters': receipt['parameters'], 'base_saved': False,
                           'pattern_version': model.pattern_version_key()}
    modes = ['fast', 'pattern', 'both', 'router', 'linear', 'shuffled', 'zero', 'disabled',
             'full_context', 'read_13_only', 'read_27_only', 'write_13_only', 'write_27_only']
    report['candidate'] = evaluate_main(eval_queries, modes)
    report['general'] = evaluate_general(replay, ('pattern', 'router', 'fast', 'zero', 'disabled',
                                                  'shuffled', 'full_context', 'adversarial'))
    report['general_dev'] = evaluate_general(retention_dev, ('pattern', 'router', 'fast', 'zero',
                                                            'disabled', 'shuffled', 'full_context', 'adversarial'))
    report['metrics'] = {mode: split_metrics(values, metric_splits)
                         for mode, values in report['candidate'].items()}
    report['parent_metrics'] = {mode: split_metrics(values, metric_splits)
                                for mode, values in report['parent']['main'].items()}
    report['general_metrics'] = {mode: accuracy(values) for mode, values in report['general'].items()}
    report['general_dev_metrics'] = {mode: accuracy(values) for mode, values in report['general_dev'].items()}
    report['parent_general_metrics'] = {mode: accuracy(values)
                                        for mode, values in report['parent']['retention'].items()}
    report['parent_general_dev_metrics'] = {mode: accuracy(values)
                                            for mode, values in report['parent']['retention_dev'].items()}
    if parent_controls is not None:
        for split, metric_name in (('retention', 'parent_general_metrics'),
                                   ('retention_dev', 'parent_general_dev_metrics')):
            adversarial_rows = parent_controls[f'parent_{split}']['adversarial']
            report['parent'][split]['adversarial'] = adversarial_rows
            report[metric_name]['adversarial'] = accuracy(adversarial_rows)
    report['train_results'] = [dict(generate_mode(query, 'pattern'), case_id=query['case_id'],
                                    split='train', case_role=query.get('case_role')) for query in train]
    report['train_correct'] = accuracy(report['train_results'])
    corrections = []
    for query in eval_queries:
        original = depth_read(query['observations'])
        corrected = depth_read([query['correction_observation']])
        original_answer = generate(session, query, DELAYED_USE_TEMPLATE, original)
        corrected_answer = generate(session, dict(query, expected_answer=query['corrected_answer']),
                                    DELAYED_USE_TEMPLATE, corrected)
        corrections.append({'case_id': query['case_id'], 'original_answer': original_answer,
                            'corrected_answer': corrected_answer,
                            'state_changed': original.state_digest != corrected.state_digest,
                            'stale_rejected': corrected_answer['answer'] != original_answer['answer']})
        check_time()
    report['correction'] = corrections
    query = heldout[0]
    scope = FastMemoryScope(query['subject_id'], query['branch_id'], query['purpose'],
                            'parallel_replica', 0)
    config = FastMemoryConfig(key_size=protocol['fast_memory']['key_size'],
                              value_size=protocol['fast_memory']['value_size'],
                              heads=protocol['fast_memory']['heads'], update_rule='sgdn2',
                              max_tokens_per_chunk=256,
                              max_image_bytes=protocol['fast_memory']['max_image_bytes'])
    correct_state = depth_read(query['observations'])
    lineage = tuple((f'observation-{index}', hashlib.sha256(text.encode()).hexdigest())
                    for index, text in enumerate(query['observations']))
    cursor = sum(len(tokenizer(text, add_special_tokens=False)['input_ids'])
                 for text in query['observations'])
    image = args.output / 'fast_image'
    metadata = save_fast_depth_image(image, correct_state, config=config, scope=scope,
                                     producer=model.fast_version_key(), chunk_index=len(query['observations']),
                                     update_count=len(query['observations']), cursor=cursor, lineage=lineage)
    loaded_read, loaded_metadata = load_fast_depth_image(
        image, config=config, scope=scope, producer=model.fast_version_key(),
        frame_producer=model.fast_producer_key(), device='mps')
    correct_answer = generate_mode(query, 'router')
    pattern_correct_answer = next(row for row in report['candidate']['pattern']
                                  if row['case_id'] == query['case_id'])
    reloaded_extension = model.load_extension(extension)
    if correct_answer.get('route') == 'pattern':
        reloaded_answer = generate(session, query, DELAYED_USE_TEMPLATE, None, pattern=pattern_read())
    else:
        reloaded_answer = generate(session, query, DELAYED_USE_TEMPLATE, loaded_read)
    pattern_reloaded_answer = generate(session, query, DELAYED_USE_TEMPLATE, None, pattern=pattern_read())
    report['restart'] = {
        'mode': 'depth_image_and_extension_reload', 'image': metadata,
        'state_digest_equal': loaded_read.state_digest == correct_state.state_digest,
        'lineage_equal': loaded_metadata['lineage'] == metadata['lineage'],
        'extension_manifest_equal': reloaded_extension['manifest_digest'] == receipt['manifest_digest'],
        'extension_weights_equal': reloaded_extension['weights_sha256'] == receipt['weights_sha256'],
        'extension': {'path': str(extension), 'weights_sha256': reloaded_extension['weights_sha256'],
                      'manifest_digest': reloaded_extension['manifest_digest']},
        'reloaded_answer': reloaded_answer, 'correct_answer': correct_answer,
        'answer_equal': reloaded_answer['answer'] == correct_answer['answer'],
        'token_ids_equal': reloaded_answer['token_ids'] == correct_answer['token_ids'],
        'pattern_correct_answer': pattern_correct_answer,
        'pattern_reloaded_answer': pattern_reloaded_answer,
        'pattern_answer_equal': pattern_reloaded_answer['answer'] == pattern_correct_answer['answer'],
        'pattern_token_ids_equal': pattern_reloaded_answer['token_ids'] == pattern_correct_answer['token_ids'],
        'source_observations_reencoded': False}
    report['collisions'] = summarize_collisions(pattern_spec, hash_counts, hash_ngrams)
    report['training_collisions'] = summarize_collisions(
        pattern_spec, training_hash_counts, training_hash_ngrams)
    report['collision_scope'] = {
        'collisions': 'All training-inventory and router query-hash calls, including general and restart routes.',
        'training_collisions': 'The 24 delayed-training prompts only.',
        'definition': 'Distinct causal n-grams sharing a table index; repeated occurrences of one n-gram are separate.'}
    report['resource_accounting'] = {
        'table_parameter_bytes': sum(parameter.numel() * parameter.element_size()
                                     for parameter in model.pattern_port.tables.parameters()),
        'dense_gradient_materialization_upper_bound_bytes': sum(
            parameter.numel() * parameter.element_size() for parameter in model.pattern_port.tables.parameters()),
        'optimizer_state_bytes': report['training']['optimizer_final']['state_tensor_bytes'],
        'attention_cache': 'disabled; native action pass recomputes without KV reuse',
    }
    parent_fast = report['parent_metrics']['fast']
    ids = encode_prompt(tokenizer, DELAYED_USE_TEMPLATE.replace('{query}', dev[0]['query'])
                        .replace('{observation}', ''), 'mps')
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        hidden = model.base.get_input_embeddings()(ids)
        timings = []
        for _ in range(3):
            start = time.monotonic()
            model.pattern_port(hidden, ids, mask)
            torch.mps.synchronize()
            timings.append(time.monotonic() - start)
    report['lookup_latency'] = {'cold_seconds': timings[0], 'warm_min_seconds': min(timings[1:])}
    candidate = report['metrics']['router']
    general_parent = report['parent_general_metrics']
    general_candidate = report['general_metrics']
    general_dev_parent = report['parent_general_dev_metrics']
    general_dev_candidate = report['general_dev_metrics']
    disabled_identical = all(
        [row['token_ids'] for row in report['parent']['main']['disabled']]
        == [row['token_ids'] for row in report['candidate']['disabled']]
        for _ in [0])
    general_disabled_identical = all(
        [row['token_ids'] for row in report['parent']['retention']['disabled']]
        == [row['token_ids'] for row in report['general']['disabled']]
        for _ in [0])
    general_dev_disabled_identical = all(
        [row['token_ids'] for row in report['parent']['retention_dev']['disabled']]
        == [row['token_ids'] for row in report['general_dev']['disabled']]
        for _ in [0])
    fast_identical = all(
        [row['token_ids'] for row in report['parent']['main']['fast']]
        == [row['token_ids'] for row in report['candidate']['fast']]
        for _ in [0])
    engineering_ok = (report['restart']['state_digest_equal'] and report['restart']['lineage_equal']
                      and report['restart']['answer_equal'] and disabled_identical
                      and general_disabled_identical and general_dev_disabled_identical and fast_identical)
    gradient_ok = False
    if len(report['training']['updates']) >= 2:
        audit = report['training']['updates'][1].get('parameter_gradients')
        gradient_ok = bool(audit and not audit['new']['nonfinite'] and audit['new']['nonzero'])
    report['checks'] = {
        'correct_router_above_controls': candidate['heldout'] > max(
            report['metrics']['linear']['heldout'], report['metrics']['shuffled']['heldout'],
            report['metrics']['zero']['heldout'], report['metrics']['disabled']['heldout']),
        'fresh_delayed_not_below_parent': candidate['heldout'] >= parent_fast['heldout'],
        'development_not_below_disabled': candidate['dev'] >= report['metrics']['disabled']['dev'],
        'correction_independent_success': all(
            item['state_changed'] and item['stale_rejected'] and item['corrected_answer']['correct']
            for item in corrections),
        'original_regression_not_below_parent': candidate['regression'] >= parent_fast['regression'],
        'general_noninterference': (general_candidate['adversarial'] >= general_parent['adversarial']
                                    and general_candidate['adversarial'] >= general_candidate['disabled']),
        'general_development_noninterference': (
            general_dev_candidate['adversarial'] >= general_dev_parent['adversarial']),
        'disabled_token_identical': disabled_identical and general_disabled_identical and general_dev_disabled_identical,
        'fast_modules_unchanged': fast_identical,
        'post_open_gradients': gradient_ok,
        'restart_state_equal': report['restart']['state_digest_equal'],
        'restart_lineage_equal': report['restart']['lineage_equal'],
        'restart_answer_equal': report['restart']['answer_equal'],
        'restart_token_ids_equal': report['restart']['token_ids_equal'],
        'restart_extension_equal': (report['restart']['extension_manifest_equal']
                                    and report['restart']['extension_weights_equal']),
        'restart_pattern_answer_equal': (report['restart']['pattern_answer_equal']
                                         and report['restart']['pattern_token_ids_equal']),
        'sparse_optimizer_step': report['training']['optimizer_final']['state_tensor_bytes'] > 0,
    }
    if source_lm_updates:
        baseline = protocol['no_warmup_control']
        report['checks'].update({
            'direct_pattern_covered_regression': report['metrics']['pattern']['regression'] > max(
                baseline['pattern_regression'], report['metrics']['linear']['regression'],
                report['metrics']['shuffled']['regression'], report['metrics']['zero']['regression'],
                report['metrics']['disabled']['regression']),
            'direct_pattern_covered_dev': report['metrics']['pattern']['dev'] > max(
                baseline['pattern_dev'], report['metrics']['linear']['dev'],
                report['metrics']['shuffled']['dev'], report['metrics']['zero']['dev'],
                report['metrics']['disabled']['dev']),
            'pattern_adversarial_noninterference': (
                general_candidate['adversarial'] >= general_candidate['disabled']
                and general_dev_candidate['adversarial'] >= general_dev_parent['adversarial']),
        })
    engineering_ok = engineering_ok and gradient_ok and all(report['checks'].values())
    report['decision'] = 'accept' if engineering_ok else 'reject'
    report['deployment'] = 'diagnostic_only'
    all_rows = (report['parent']['main']['fast'] + report['parent']['main']['disabled']
                + report['parent']['retention']['fast'] + report['parent']['retention']['disabled']
                + report['parent']['retention_dev']['fast'] + report['parent']['retention_dev']['disabled']
                + [row for values in report['candidate'].values() for row in values]
                + [row for values in report['general'].values() for row in values]
                + [row for values in report['general_dev'].values() for row in values]
                + report['train_results']
                + [row for item in corrections for row in (item['original_answer'], item['corrected_answer'])]
                + [reloaded_answer, correct_answer, pattern_reloaded_answer])
    report['generation_counts'] = {
        'parent': (len(report['parent']['main']['fast']) + len(report['parent']['main']['disabled'])
                   + len(report['parent']['retention']['fast']) + len(report['parent']['retention']['disabled'])
                   + len(report['parent']['retention_dev']['fast']) + len(report['parent']['retention_dev']['disabled'])),
        'candidate': sum(len(values) for values in report['candidate'].values()),
        'general': sum(len(values) for values in report['general'].values()),
        'general_dev': sum(len(values) for values in report['general_dev'].values()),
        'train': len(report['train_results']), 'correction': 2 * len(corrections), 'restart': 3}
    report['generation_counts']['total'] = sum(report['generation_counts'].values())
    if report['generation_counts']['total'] != len(all_rows):
        raise ValueError('M4 generation accounting mismatch')
    if (report['generation_counts']['total'] > protocol['budget']['generation_answers_per_arm']
            or sum(row['generated_tokens'] for row in all_rows) > protocol['budget']['max_generated_tokens_per_arm']):
        raise ValueError('M4 generation budget exceeded')
    report['token_costs'] = {
        'source_lm_action_tokens': sum(update.get('source_tokens', 0)
                                       for update in report['training'].get('source_lm', {}).get('updates', [])),
        'training_action_tokens': sum(update['action_tokens'] for update in report['training']['updates']),
        'training_target_tokens': sum(update['target_tokens'] for update in report['training']['updates']),
        'generation_prompt_tokens': sum(row['prompt_tokens'] for row in all_rows),
        'generation_new_tokens': sum(row['generated_tokens'] for row in all_rows),
        'maximum_generation_prompt_tokens': max(row['prompt_tokens'] for row in all_rows),
        'maximum_generated_tokens_per_answer': max(row['generated_tokens'] for row in all_rows)}
    report['status'] = 'completed'
    ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path, default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path, required=True)
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
                      'metrics': report.get('metrics'), 'checks': report.get('checks')}))


if __name__ == '__main__':
    main()
