#!/usr/bin/env python3
"""Guarded MiniCPM5 D1 fine-tuning and persistent correction/retention diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.memory_reliability import TEMPLATE, cases, audit_dataset
from replica_lab.reliability_scoring import metrics_and_checks, regression_inventory, family_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_scope(episode):
    from replica_lab.runtime.fast_memory import FastMemoryScope
    return FastMemoryScope('memory_reliability', episode['episode_id'], 'reliability', 'parallel_replica', 0)


def case_scope(case):
    from replica_lab.memory_system import EligibilityScope
    return EligibilityScope(case['subject_id'], case['branch_id'], case['purpose'],
                            case['as_of'], 'parallel_replica', None)


def request_args(session, case, read=None, full=False, evidence=None, selective=False, exact_mode=False):
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.architecture.contracts import TaskGraph
    if selective:
        if evidence is None:
            raise ValueError('Selective exact context requires an evidence selection')
        observation, read = evidence.text, None
    else:
        observation = '\n'.join(case['observations']) if full else ''
        if evidence is not None and not evidence.enabled:
            read = None
    text = TEMPLATE.replace('{query}', case['query']).replace('{observation}', observation)
    ids = encode_prompt(session.tokenizer, text, session.device)
    if ids.shape[1] > 256:
        raise ValueError('Prompt exceeds protocol bound')
    import torch
    enabled_ports = () if exact_mode and read is None else None
    branches = (('fast',) if exact_mode and read is not None else
                ('memory', 'fast') if read is not None else ('memory',))
    return dict(prompt_ids=ids, prompt_mask=torch.ones_like(ids, dtype=torch.bool), query=case['query'],
                scope=case_scope(case), task=TaskGraph(case['case_id'], case['query']),
                core_impl='none', fast_state=read,
                enabled_ports=enabled_ports, enabled_branches=branches)


def infer(session, case, read=None, full=False, evidence=None, selective=False, exact_mode=False):
    arguments = request_args(session, case, read, full, evidence, selective, exact_mode)
    request = session.prepare(**arguments)
    try:
        generated = session.generate(request, max_new_tokens=6)
        ids = generated[0, arguments['prompt_ids'].shape[1]:].tolist()
        answer = session.tokenizer.decode(ids, skip_special_tokens=True).strip()
        row = {k: case[k] for k in ('case_id', 'split', 'episode_id', 'stage', 'kind')} | {
            'answer': answer, 'expected': case['expected_answer'],
            'correct': answer.casefold() == case['expected_answer'].casefold(),
            'token_ids': ids, 'generated_tokens': len(ids), 'prompt_tokens': arguments['prompt_ids'].shape[1],
            'state_digest': None if selective else (read.state_digest if read is not None else None)}
        if evidence is not None:
            row.update(evidence.metadata())
        return row
    finally:
        request.release()


def initialize(model, tokenizer, episode, training=False, reference=None):
    from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
    if reference is not None:
        reference.admit_episode(episode)
    memory = PersistentFastMemory(model, tokenizer, memory_scope(episode), training=training)
    for index, observation in enumerate(episode['observations']):
        memory.append(f"{episode['episode_id']}:source:{index}", observation)
    return memory


def evaluate(session, episodes, arms, output=None, check=lambda: None,
             reference=None, gate=False, adversarial_text=None):
    from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
    from replica_lab.runtime.exact_memory import episode_records
    rows, transitions = [], []
    for episode in episodes:
        memory = initialize(session.adapter, session.tokenizer, episode, reference=reference)
        # Same family counterfactual gives identical queries and different facts.
        other = next(e for e in episodes if e['family_id'] == episode['family_id'] and e['variant'] != episode['variant'])
        shuffled = initialize(session.adapter, session.tokenizer, other)
        adversarial = None
        if adversarial_text is not None:
            adversarial = PersistentFastMemory(session.adapter, session.tokenizer, memory_scope(episode))
            adversarial.append(f"{episode['episode_id']}:adversarial", adversarial_text)
        initial = memory.read().state_digest
        correction_record = episode_records(episode)[1] if reference is not None else None
        for stage in ('initial', 'corrected'):
            if stage == 'corrected':
                if output is not None:
                    before = output / episode['episode_id'] / 'before'
                    if reference is not None:
                        initial_view = reference.view(case_scope(cases(episode, 'initial')[0]))
                        memory.bind_source(initial_view)
                    metadata = memory.save(before, require_source_binding=reference is not None)
                    memory = PersistentFastMemory.load(
                        before, session.adapter, session.tokenizer, memory_scope(episode),
                        expected_view_digest=(metadata.get('source_binding') or {}).get('view_digest'),
                        require_source_binding=reference is not None)
                    if memory.read().state_digest != initial:
                        raise ValueError('Reload changed initial state')
                if reference is not None:
                    reference.correct_episode(episode)
                memory.append(f"{episode['episode_id']}:correction", episode['correction_observation'])
                shuffled.append(f"{other['episode_id']}:correction", other['correction_observation'])
                if output is not None:
                    after = output / episode['episode_id'] / 'corrected'
                    if reference is not None:
                        corrected_view = reference.view(case_scope(cases(episode, 'corrected')[0]))
                        memory.bind_source(corrected_view)
                    corrected = memory.save(after, require_source_binding=reference is not None)
                    transitions.append({'episode_id': episode['episode_id'], 'initial_digest': initial,
                                        'corrected_digest': memory.read().state_digest, 'image': str(after),
                                        'initial_cursor': metadata['cursor'], 'corrected_cursor': corrected['cursor'],
                                        'lineage_prefix_preserved': corrected['lineage'][:-1] == metadata['lineage'],
                                        'reloaded_before_append': True, 'source_reencoded_on_reload': False,
                                        'initial_source_binding': metadata.get('source_binding'),
                                        'corrected_source_binding': corrected.get('source_binding'),
                                        'correction_supersedes': (correction_record.supersedes
                                                                  if correction_record is not None else None)})
            read, wrong = memory.read(), shuffled.read()
            last = PersistentFastMemory(session.adapter, session.tokenizer, memory_scope(episode))
            last.append('last_only', episode['observations'][-1] if stage == 'initial' else episode['correction_observation'])
            for case in cases(episode, stage):
                evidence = reference.select(case['query'], case_scope(case)) if reference is not None else None
                for arm in arms:
                    if arm == 'selective_context':
                        selected, row_evidence, selective = None, evidence, True
                    else:
                        selective = False
                        row_evidence = evidence if gate else None
                        selected = (read if arm in {'fast', 'ungated'} else wrong if arm == 'shuffled'
                                    else last.read() if arm == 'last_chunk'
                                    else adversarial.read() if arm == 'adversarial' and adversarial is not None else None)
                        if arm == 'ungated':
                            row_evidence = None
                    row = infer(session, case, selected, full=arm == 'full_context',
                                evidence=row_evidence, selective=selective,
                                exact_mode=reference is not None)
                    rows.append(dict(row, arm=arm)); check()
    return rows, transitions


def train_candidate(session, episodes, protocol, report, check):
    import torch
    import torch.nn.functional as F
    from replica_lab.optimizers import build_optimizer
    model, tokenizer = session.adapter, session.tokenizer
    reference = None
    if protocol.get('exact_reference', {}).get('enabled', False):
        from replica_lab.runtime.exact_memory import ExactMemoryReference
        reference = ExactMemoryReference(session.ledger)
    gate = protocol.get('evidence_gate', {}).get('enabled', False)
    prefixes = tuple(protocol['training'].get('trainable_prefixes',
                                            ('fast_depth_write_encoders.', 'fast_ports.')))
    flags = {name: p.requires_grad for name, p in model.new_named_parameters()}
    for name, p in model.new_named_parameters():
        p.requires_grad_(name.startswith(prefixes))
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0, lr_new=protocol['training']['lr'],
                                lr_muon_base=0.0, lr_muon_new=protocol['training']['lr'], weight_decay=0.0)
    train = [e for e in episodes if e['split'] == 'train']
    per_episode = [(e, cases(e, 'initial') + cases(e, 'corrected')) for e in train]
    schedule = [(e, qs[i]) for i in range(12) for e, qs in per_episode]
    random.Random(protocol['training']['seed']).shuffle(schedule)
    count = protocol['training']['examples_per_update']
    if protocol['training']['updates'] * count != len(schedule):
        raise ValueError('Pilot must traverse every training case exactly once')
    report['training'] = {'updates': [], 'optimizer_initial': optimizer.audit(), 'case_ids': [],
                          'parameter_audit': model.parameter_audit(),
                          'trainable_prefixes': list(prefixes), 'evidence_gate': gate}
    model.train()
    for update in range(protocol['training']['updates']):
        optimizer.zero_grad(set_to_none=True)
        losses, kls = [], []
        gate_off = 0
        for episode, case in schedule[update*count:(update+1)*count]:
            memory = initialize(model, tokenizer, episode, training=True, reference=reference)
            if case['stage'] == 'corrected':
                if reference is not None:
                    reference.correct_episode(episode)
                memory.append(f"{episode['episode_id']}:correction", episode['correction_observation'])
            evidence = reference.select(case['query'], case_scope(case)) if gate else None
            arguments = request_args(session, case, memory.read(), evidence=evidence,
                                     exact_mode=reference is not None)
            prefix = arguments['prompt_ids']
            target = tokenizer(case['expected_answer'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            if len(target) > 8:
                raise ValueError('Target exceeds protocol bound')
            ids = torch.cat((prefix, torch.tensor([target], device=session.device)), dim=1)
            labels = ids.clone(); labels[:, :prefix.shape[1]] = -100
            teacher = None
            if case['kind'] == 'general':
                teacher_args = request_args(session, case, exact_mode=reference is not None)
                with torch.no_grad(), session.training_request(**teacher_args) as request:
                    out = session.forward(request, ids, labels=labels)
                    teacher = out.logits[:, prefix.shape[1]-1:-1].detach().softmax(-1)
                    del out
            with session.training_request(**arguments) as request:
                out = session.forward(request, ids, labels=labels)
                kd = out.loss.new_zeros(())
                if teacher is not None:
                    student = out.logits[:, prefix.shape[1]-1:-1].log_softmax(-1)
                    kd = F.kl_div(student, teacher, reduction='sum') / len(target)
                loss = out.loss + protocol['training']['preservation_kl_weight'] * kd
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                losses.append(float(loss.detach())); kls.append(float(kd.detach()))
                if not loss.requires_grad:
                    if evidence is None or evidence.enabled:
                        raise RuntimeError('A gate-on training example has no differentiable path')
                    gate_off += 1
                else:
                    session.backward(request, loss / count)
            report['training']['case_ids'].append(case['case_id'])
            del memory, out, loss, kd, teacher
            check()
        gradient = model.gradient_audit() if update in (0, 1) else None
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['optimizer_updates'] += 1
        report['training']['updates'].append({'update': update+1, 'mean_loss': sum(losses)/count,
                                              'mean_preservation_kl': sum(kls)/count, 'gradients': gradient,
                                              'gate_off_examples': gate_off})
        print(json.dumps({'phase': 'training', 'update': update+1, 'mean_loss': sum(losses)/count}), flush=True)
    report['training']['optimizer_final'] = optimizer.audit()
    del optimizer
    for name, p in model.new_named_parameters():
        p.requires_grad_(flags[name])
    model.eval()


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    protocol = json.loads(args.protocol.read_text())
    report['evaluation_only'] = args.evaluate_only
    if args.evaluate_only != protocol.get('evaluation_only', False):
        raise ValueError('Runner evaluation-only mode differs from protocol')
    if (args.model_path.resolve() != (ROOT/'checkpoints/minicpm5_source').resolve()
            or args.model_lock.resolve() != (ROOT/'configs/minicpm5_sealed_lock.local.json').resolve()
            or sha(args.model_lock) != protocol['source_lock_sha256']):
        raise ValueError('Use the protocol-pinned sealed source and lock')
    for name, expected in protocol['implementation_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT/name) != expected:
            raise ValueError(f'Implementation changed: {name}')
    path = ROOT / protocol['dataset']['path']
    if sha(path) != protocol['dataset']['sha256']:
        raise ValueError('Dataset changed')
    episodes = json.loads(path.read_text()); report['dataset_audit'] = audit_dataset(
        episodes, require_novel_values=protocol['dataset'].get('require_novel_values', False))
    report['implementation_sha256'] = protocol['implementation_sha256']
    started = time.monotonic()
    def check():
        limit = protocol['budget']['restart_max_wall_seconds'] if args.restart_run else protocol['budget']['max_wall_seconds']
        if time.monotonic()-started > limit:
            raise TimeoutError('Pilot wall budget exceeded')
    torch.manual_seed(protocol['training']['seed'])
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock, profiler=profiler,
        fast_memory=FastMemorySpec(heads=4, key_size=64, value_size=64, update_rule='sgdn2', depth_mode='shared', depth_writers=True))
    report['source_lock'] = lock
    ledger = RecordLedger(args.output / 'ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    reference = None
    if protocol.get('exact_reference', {}).get('enabled', False):
        from replica_lab.runtime.exact_memory import ExactMemoryReference
        reference = ExactMemoryReference(ledger)
    evaluation_options = {
        'reference': reference,
        'gate': protocol.get('evidence_gate', {}).get('enabled', False),
        'adversarial_text': protocol.get('evaluation', {}).get('adversarial_observation')}
    parent_arms = tuple(protocol.get('evaluation_arms', {}).get('parent', ('fast', 'disabled')))
    candidate_arms = tuple(protocol.get('evaluation_arms', {}).get(
        'candidate', ('fast', 'disabled', 'shuffled', 'last_chunk', 'full_context')))
    report['arm_contract'] = {'parent': list(parent_arms), 'candidate': list(candidate_arms)}
    try:
        if args.restart_run:
            previous_path = args.restart_run/'run_report.json'
            previous = json.loads(previous_path.read_text())
            if previous['status'] != 'completed' or previous['protocol_sha256'] != sha(args.protocol):
                raise ValueError('Restart evidence does not match protocol')
            candidate_path = (ROOT / protocol['candidate_extension'] if protocol.get('evaluation_only', False)
                              else args.restart_run / 'candidate_extension')
            receipt = model.load_extension(candidate_path)
            if receipt['weights_sha256'] != previous['extension']['weights_sha256']:
                raise ValueError('Candidate bytes changed')
            episode = next(e for e in episodes if e['split']=='heldout')
            transition = next(t for t in previous['transitions'] if t['episode_id']==episode['episode_id'])
            source_view_digest = (transition.get('corrected_source_binding') or {}).get('view_digest')
            memory = PersistentFastMemory.load(
                Path(transition['image']), model, tokenizer, memory_scope(episode),
                expected_view_digest=source_view_digest,
                require_source_binding=reference is not None)
            if reference is not None:
                reference.admit_episode(episode)
                reference.correct_episode(episode)
            report['answers'] = []
            for q in cases(episode, 'corrected'):
                evidence = reference.select(q['query'], case_scope(q)) if evaluation_options['gate'] else None
                report['answers'].append(infer(session, q, memory.read(), evidence=evidence,
                                               exact_mode=reference is not None)); check()
            reference_rows = {r['case_id']: r for r in previous['candidate']
                              if r['arm'] == protocol.get('primary_fast_arm', 'fast')}
            report['checks'] = {
                'state_equal': memory.read().state_digest == transition['corrected_digest'],
                'tokens_equal': all(r['token_ids'] == reference_rows[r['case_id']]['token_ids'] for r in report['answers']),
                'lineage_count': len(memory.lineage) == 3,
                'source_view_equal': (memory.source_binding is not None
                                      and memory.source_binding['view_digest'] == source_view_digest
                                      if reference is not None else True)}
            report['source_reencoded'] = False; report['training_updates'] = 0
            report['generation_answers'] = len(report['answers'])
            report['source_binding'] = memory.source_binding
            generated_tokens = sum(row['generated_tokens'] for row in report['answers'])
            if (report['generation_answers'] > protocol['budget']['restart_max_generated_answers']
                    or generated_tokens > protocol['budget']['restart_max_generated_tokens']):
                raise ValueError('Restart generation budget exceeded')
            report['previous_report_sha256'] = sha(previous_path)
            report['decision'] = 'mechanical_restart_pass' if all(report['checks'].values()) else 'reject'
        else:
            parent = model.load_extension(ROOT / protocol['parent_extension'])
            if parent['weights_sha256'] != protocol['parent_weights_sha256']:
                raise ValueError('Parent extension changed')
            report['parent_extension'] = parent
            evaluation = [e for e in episodes if e['split']!='train']
            report['parent'], _ = evaluate(session, evaluation, parent_arms, check=check, **evaluation_options)
            print(json.dumps({'phase': 'parent_evaluated', 'answers': len(report['parent'])}), flush=True)
            if args.evaluate_only:
                if args.candidate_extension is None:
                    raise ValueError('Evaluation-only requires a candidate extension')
                report['extension'] = model.load_extension(args.candidate_extension)
                if report['extension']['weights_sha256'] != protocol['candidate_weights_sha256']:
                    raise ValueError('Evaluation candidate bytes changed')
                report['candidate_fast_version'] = model.fast_version_key()
                report['checkpoints_saved'] = 0
                report['evaluation_only'] = True
            else:
                train_candidate(session, episodes, protocol, report, check)
                report['extension'] = model.save_extension(args.output/'candidate_extension')
                report['candidate_fast_version'] = model.fast_version_key()
                report['checkpoints_saved'] = 1
            report['candidate'], report['transitions'] = evaluate(
                session, evaluation, candidate_arms, args.output/'images', check, **evaluation_options)
            report['regressions'] = regression_inventory(
                report, episodes, lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                protocol.get('primary_fast_arm', 'fast'))
            report['families'] = family_metrics(
                report, episodes, lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
            report['metrics'], report['checks'] = metrics_and_checks(
                report, episodes, lambda ids: tokenizer.decode(ids, skip_special_tokens=True), protocol)
            report['decision'] = 'quality_pass_pending_restart' if all(report['checks'].values()) else 'reject'
            report['generation_answers'] = len(report['parent'])+len(report['candidate'])
            if report['generation_answers'] > protocol['budget']['max_generated_answers']:
                raise ValueError('Generation budget exceeded')
        report.update(status='completed', deployment='diagnostic_only_not_published')
        check()
    finally:
        ledger.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--restart-run', type=Path)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--candidate-extension', type=Path)
    parser.add_argument('--model-path', type=Path, default=ROOT/'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path, default=ROOT/'configs/minicpm5_sealed_lock.local.json')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':sha(args.protocol),'optimizer_updates':0,'checkpoints_saved':0,'downloads':0}
    profiler=RunProfiler('mps',args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start(); execute(args,profiler,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['profile']=profiler.stop();report['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        (args.output/('run_report.json' if report['status']=='completed' else 'failure_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'decision':report['decision'],'updates':report['optimizer_updates']}))


if __name__=='__main__':
    main()
