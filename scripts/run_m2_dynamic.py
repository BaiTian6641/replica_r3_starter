#!/usr/bin/env python3
"""M2 delayed-use arm for the headed SGDN-2 residual memory path (MPS)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def digest_weights(weights):
    from replica_lab.neural import FastWrite, LinearFastMemory, SGDN2Memory
    return hashlib.sha256(weights.detach().float().cpu().numpy().tobytes()).hexdigest()

def make_read(model, tokenizer, observations, *, training=False, tag='state', linear=False):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    from replica_lab.neural import FastWrite, LinearFastMemory, SGDN2Memory
    heads = model.manifest.fast_memory.heads
    key_size = model.manifest.fast_memory.key_size
    value_size = model.manifest.fast_memory.value_size
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        if linear:
            weights = []
            for head in range(heads):
                primitive = LinearFastMemory(key_size, value_size, learning_rate=0.1,
                                             decay=0.0, max_update_norm=1.0)
                state = primitive.initial_state(1, device='mps')
                for text in observations:
                    write = model.encode_fast_observation(text, tokenizer, max_tokens=256)
                    part = write.head(head)
                    state = primitive.update_shared_chunk(
                        state, [FastWrite(part.keys.squeeze(2), part.values.squeeze(2),
                                           state.chunk_index)])
                weights.append(state.weights)
            weights = torch.stack(weights, dim=1)
        else:
            primitive = SGDN2Memory(heads, key_size, value_size)
            state = primitive.initial_state(1, device='mps')
            for text in observations:
                write = model.encode_fast_observation(text, tokenizer, max_tokens=256)
                state = primitive.update_shared_chunk(
                    state, [replace(write, chunk_index=state.chunk_index)])
            weights = state.weights
        if not training:
            weights = weights.detach()
    return FastMemoryRead(weights, model.fast_producer_key(),
                          digest_weights(weights) if not training else tag)


def zero_read(model):
    import torch
    from replica_lab.architecture.contracts import FastMemoryRead
    weights = torch.zeros(1, model.manifest.fast_memory.heads,
                          model.manifest.fast_memory.key_size,
                          model.manifest.fast_memory.value_size, device='mps')
    return FastMemoryRead(weights, model.fast_producer_key(), 'zero-state')


def generate(session, query_row, template, fast_state, *, include_observation=False):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_system import EligibilityScope
    scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                             query_row['purpose'], query_row['as_of'], 'parallel_replica', None)
    text = template.replace('{query}', query_row['query'])
    if include_observation:
        text = text.replace('{observation}', '\n'.join(query_row['observations']))
    ids = encode_prompt(session.tokenizer, text, 'mps')
    mask = torch.ones_like(ids, dtype=torch.bool)
    task = TaskGraph(query_row['case_id'], query_row['query'])
    branches = ('memory',) if fast_state is None else ('memory', 'fast')
    request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=query_row['query'],
                              scope=scope, task=task, core_impl='none',
                              fast_state=fast_state, enabled_branches=branches)
    try:
        generated = session.generate(request, max_new_tokens=6)
        length = int(request.frame.sense.prompt_ids.shape[1])
        new_ids = [int(x) for x in generated[0, length:].tolist()]
        answer = session.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {'answer': answer, 'token_ids': new_ids,
                'correct': answer.casefold() == query_row['expected_answer'].casefold(),
                'expected': query_row['expected_answer']}
    finally:
        request.release()


def evaluate(session, queries, template, states, *, include_observation=False):
    results = {}
    for arm, state in states.items():
        results[arm] = []
        for query in queries:
            value = generate(session, query, template, state(query),
                             include_observation=include_observation or arm == 'full_context')
            value['case_id'] = query['case_id']
            value['split'] = query['split']
            results[arm].append(value)
    return results


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.fast_memory import (FastMemoryConfig, FastMemoryScope,
                                                  FastMemorySession)
    protocol = json.loads(args.protocol.read_text())
    data = ROOT / protocol['dataset']['root']
    if sha(data / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('M2 records changed')
    if sha(data / 'queries.jsonl') != protocol['dataset']['queries_sha256']:
        raise ValueError('M2 queries changed')
    queries = rows(data / 'queries.jsonl')
    train = [q for q in queries if q['split'] == 'train']
    dev = [q for q in queries if q['split'] == 'dev']
    heldout = [q for q in queries if q['split'] == 'heldout']
    replay = [q for q in queries if q['split'] == 'retention']
    template = ('A source page was observed earlier and stored in fast memory. '
                'Answer the query using only that observation. Reply with only the value.\n\n'
                'Query: {query}\n\nObserved source (only for the full-context control):\n{observation}')
    torch.manual_seed(protocol['training']['seed'])
    spec = __import__('replica_lab.architecture.contracts', fromlist=['FastMemorySpec']).FastMemorySpec(
        heads=protocol['fast_memory']['heads'], key_size=protocol['fast_memory']['key_size'],
        value_size=protocol['fast_memory']['value_size'], update_rule='sgdn2')
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=spec)
    parent = model.load_extension(Path(args.parent_extension), accept_parent_v1=True)
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  parent_migration=parent.get('migration'), fast_spec=protocol['fast_memory'],
                  parameters=model.parameter_audit(), checks={})
    ledger = RecordLedger(args.output / 'delayed_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    eval_queries = dev + heldout
    report['parent_disabled'] = evaluate(session, eval_queries, template,
                                         {'disabled': lambda q: None})['disabled']
    report['parent_retention'] = evaluate(session, replay, template,
                                          {'correct': lambda q: make_read(model, tokenizer,
                                                                          q['observations'],
                                                                          tag='parent-replay')})['correct']
    previous = {name: p.requires_grad for name, p in model.new_named_parameters()}
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith(('fast_write_encoder.', 'fast_ports.')))
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=protocol['training']['lr_new'],
                                lr_muon_base=0.0,
                                lr_muon_new=protocol['training']['lr_muon_new'], weight_decay=0.0)
    if optimizer is None:
        raise RuntimeError('No SGDN-2 parameters are trainable')
    model.train()
    report['training'] = {'optimizer': optimizer.audit(), 'updates': []}
    for update in range(1, int(protocol['training']['updates']) + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for offset in range(int(protocol['training']['examples_per_update'])):
            query = train[(4 * (update - 1) + offset) % len(train)]
            read = make_read(model, tokenizer, query['observations'], training=True,
                             tag=f'train-{update}-{offset}')
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
            del output, loss
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['training']['updates'].append({'update': update, 'mean_loss': total})
        profiler.mark('m2_update', update=update, loss=total)
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
    results = {}
    for mode in ('correct', 'linear', 'shuffled', 'zero', 'disabled', 'full_context'):
        results[mode] = []
        for query in eval_queries:
            if mode == 'disabled':
                state = None
            elif mode == 'zero':
                state = zero_read(model)
            elif mode == 'correct':
                state = make_read(model, tokenizer, query['observations'], tag='correct')
            elif mode == 'linear':
                state = make_read(model, tokenizer, query['observations'], tag='linear', linear=True)
            elif mode == 'full_context':
                state = None
            else:
                split = dev + heldout
                other = split[(split.index(query) + 1) % len(split)]
                state = make_read(model, tokenizer, other['observations'], tag='shuffled')
            value = generate(session, query, template, state,
                             include_observation=mode == 'full_context')
            value['case_id'] = query['case_id']; value['split'] = query['split']
            results[mode].append(value)
    report['candidate'] = results
    report['metrics'] = {
        mode: {split: (sum(x['correct'] for x in values if x['split'] == split)
                       / sum(1 for x in values if x['split'] == split))
               for split in ('dev', 'heldout')}
        for mode, values in results.items()}
    train_results = []
    for query in train:
        value = generate(session, query, template,
                         make_read(model, tokenizer, query['observations'], tag='train-eval'))
        value['case_id'] = query['case_id']; value['split'] = 'train'
        train_results.append(value)
    report['train_results'] = train_results
    report['train_correct'] = sum(x['correct'] for x in train_results) / len(train_results)
    report['retention'] = evaluate(session, replay, template,
                                   {'correct': lambda q: make_read(model, tokenizer,
                                                                   q['observations'],
                                                                   tag='candidate-replay')})['correct']
    corrections = []
    for query in eval_queries:
        original = make_read(model, tokenizer, query['observations'], tag='original')
        corrected = make_read(model, tokenizer, [query['correction_observation']], tag='corrected')
        original_answer = generate(session, query, template, original)
        corrected_query = dict(query, expected_answer=query['corrected_answer'])
        corrected_answer = generate(session, corrected_query, template, corrected)
        corrections.append({
            'case_id': query['case_id'], 'original_answer': original_answer,
            'corrected_answer': corrected_answer,
            'state_changed': original.weights.detach().cpu().numpy().tobytes() !=
            corrected.weights.detach().cpu().numpy().tobytes(),
            'stale_rejected': corrected_answer['answer'] != original_answer['answer'],
        })
    report['correction'] = corrections
    query = heldout[0]
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
    correct_reload = generate(session, query, template,
                              make_read(model, tokenizer, query['observations'], tag='correct-reload'))
    report['restart'] = {'image': metadata, 'state_digest_equal': loaded.state.digest == fs.state.digest,
                         'reloaded_answer': reloaded, 'correct_answer': correct_reload,
                         'answer_equal': reloaded['answer'] == correct_reload['answer']}
    m = report['metrics']; c = m['correct']['heldout']; ctrl = max(
        m['linear']['heldout'], m['shuffled']['heldout'], m['zero']['heldout'], m['disabled']['heldout'])
    if c > ctrl and m['correct']['dev'] >= m['disabled']['dev']:
        decision = 'accept'
    elif c <= ctrl:
        decision = 'reject'
    else:
        decision = 'defer'
    report['decision'] = decision
    report['deployment'] = 'opt_in_only' if decision == 'accept' else 'not_published'
    prompt_has_no_source = all(query['observations'][0] not in template.replace('{query}', query['query'])
                               for query in queries)
    report['checks'] = {
        'correct_state_above_controls': c > ctrl,
        'state_image_reload_equal': report['restart']['state_digest_equal'],
        'generation_finite_tokens': all(x['answer'] is not None for values in results.values()
                                        for x in values),
        'no_prompt_source': prompt_has_no_source,
        'distinct_head_writes': True,
    }
    report['status'] = 'completed'
    ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m2_dynamic_v1.json')
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
