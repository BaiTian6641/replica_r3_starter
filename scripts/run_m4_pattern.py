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



def collect_query_hashes(model, ids, mask, hash_counts):
    result = set()
    for order, head, index, available in model.pattern_port.hash_indices(ids, mask):
        if bool(available.any()):
            for value in index[available].tolist():
                result.add((order, head, int(value)))
                hash_counts[(order, head, int(value))] = hash_counts.get((order, head, int(value)), 0) + 1
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
    torch.manual_seed(protocol['training']['seed'])
    fast_spec = FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2',
        depth_mode='shared', depth_writers=True)
    pattern_spec = PatternSpec(**protocol['pattern']['spec'])
    hash_counts = {}
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
        return collect_query_hashes(model, ids, mask, hash_counts)
    trained_hashes = set()
    for query in train:
        text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
        ids = encode_prompt(tokenizer, text, 'mps')
        trained_hashes.update(query_hashes(ids, torch.ones_like(ids, dtype=torch.bool)))
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
                if mode == 'pattern':
                    fast, pattern = None, pattern_read()
                elif mode == 'router':
                    fast, pattern = None, pattern_read()
                elif mode == 'fast':
                    fast, pattern = depth_read(query['observations']), None
                elif mode == 'zero':
                    fast, pattern = zero_read(model), None
                elif mode == 'disabled':
                    fast, pattern = None, None
                elif mode == 'shuffled':
                    fast, pattern = depth_read(queries[(index + 1) % len(queries)]['observations']), None
                elif mode == 'adversarial':
                    fast, pattern = depth_read([adversarial]), None
                elif mode == 'full_context':
                    fast, pattern = None, None
                else:
                    raise ValueError(f'Unknown M4 general mode: {mode}')
                value = generate(session, query, DELAYED_USE_TEMPLATE, fast, pattern=pattern,
                                 include_observation=mode == 'full_context')
                value.update(case_id=query['case_id'], split=query['split'],
                             case_role=query.get('case_role'), arm=mode)
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
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=protocol['training']['lr_new'],
                                lr_muon_base=0.0, lr_muon_new=protocol['training']['lr_muon_new'],
                                weight_decay=0.0)
    if optimizer is None:
        raise RuntimeError('M4 pattern optimizer has no trainable parameters')
    model.train()
    report['training'] = {'optimizer': optimizer.audit(), 'updates': []}
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
        report['optimizer_updates'] = update
        profiler.mark('m4_update', update=update, loss=record['mean_loss'])
        check_time()
    report['training']['optimizer_final'] = optimizer.audit()
    del optimizer
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(previous[name])
    model.eval()
    extension = args.output / 'selected_extension'
    receipt = model.save_extension(extension)
    report['checkpoints_saved'] = 1
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
    report['train_results'] = [dict(generate_mode(query, 'router'), case_id=query['case_id'],
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
    reloaded_answer = generate(session, query, DELAYED_USE_TEMPLATE, loaded_read)
    report['restart'] = {'mode': 'depth_image_reload', 'image': metadata,
                         'state_digest_equal': loaded_read.state_digest == correct_state.state_digest,
                         'lineage_equal': loaded_metadata['lineage'] == metadata['lineage'],
                         'reloaded_answer': reloaded_answer, 'correct_answer': correct_answer,
                         'answer_equal': reloaded_answer['answer'] == correct_answer['answer'],
                         'source_observations_reencoded': False}
    collisions = {}
    for order in pattern_spec.orders:
        for head in range(pattern_spec.heads):
            indices = [item for item in report['trained_hash_ids'] if item[0] == order and item[1] == head]
            unique = len(indices)
            collisions[f'{order}_{head}'] = {
                'unique_indices': unique,
                'table_rows': pattern_spec.table_rows,
                'hash_coverage': unique / pattern_spec.table_rows,
                'touched_bytes': unique * pattern_spec.width * 4,
            }
    report['collisions'] = collisions
    report['resource_accounting'] = {
        'table_parameter_bytes': sum(parameter.numel() * parameter.element_size()
                                     for parameter in model.pattern_port.tables.parameters()),
        'dense_gradient_materialization_upper_bound_bytes': sum(
            parameter.numel() * parameter.element_size() for parameter in model.pattern_port.tables.parameters()),
        'optimizer_state_bytes': report['training']['optimizer_final']['state_tensor_bytes'],
        'attention_cache': 'disabled; native action pass recomputes without KV reuse',
    }
    ids = encode_prompt(tokenizer, DELAYED_USE_TEMPLATE.replace('{query}', dev[0]['query'])
                        .replace('{observation}', ''), 'mps')
    collisions = {}
    for order in pattern_spec.orders:
        for head in range(pattern_spec.heads):
            counts = [count for (item_order, item_head, _), count in hash_counts.items()
                      if item_order == order and item_head == head]
            lookups = sum(counts)
            unique = len(counts)
            collisions[f'{order}_{head}'] = {
                'lookups': lookups,
                'unique_indices': unique,
                'collisions': lookups - unique,
                'collision_rate': (lookups - unique) / lookups if lookups else 0.0,
                'table_rows': pattern_spec.table_rows,
                'hash_coverage': unique / pattern_spec.table_rows,
                'touched_bytes': unique * pattern_spec.width * 4,
            }
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
        'sparse_optimizer_step': report['training']['optimizer_final']['state_tensor_bytes'] > 0,
    }
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
                + [reloaded_answer, correct_answer])
    report['generation_counts'] = {
        'parent': (len(report['parent']['main']['fast']) + len(report['parent']['main']['disabled'])
                   + len(report['parent']['retention']['fast']) + len(report['parent']['retention']['disabled'])
                   + len(report['parent']['retention_dev']['fast']) + len(report['parent']['retention_dev']['disabled'])),
        'candidate': sum(len(values) for values in report['candidate'].values()),
        'general': sum(len(values) for values in report['general'].values()),
        'general_dev': sum(len(values) for values in report['general_dev'].values()),
        'train': len(report['train_results']), 'correction': 2 * len(corrections), 'restart': 2}
    report['generation_counts']['total'] = sum(report['generation_counts'].values())
    if report['generation_counts']['total'] != len(all_rows):
        raise ValueError('M4 generation accounting mismatch')
    if (report['generation_counts']['total'] > protocol['budget']['generation_answers_per_arm']
            or sum(row['generated_tokens'] for row in all_rows) > protocol['budget']['max_generated_tokens_per_arm']):
        raise ValueError('M4 generation budget exceeded')
    report['token_costs'] = {
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
