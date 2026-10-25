"""Bounded tiny-set projector overfit diagnostic.

One train-only world cohort is fitted with the existing projector-only training
path, then evaluated on those same worlds and on fresh disjoint dev worlds with
correct/wrong/absent image controls. This distinguishes capacity/training
failure from overfitting-without-generalization before a V1 retry.
"""
from __future__ import annotations

from pathlib import Path
import json

from .vision_qualification import (
    VISUAL_QUESTION_TEMPLATE, answer_correct, require,
)

OVERFIT_FORMAT = 'replica.vision_overfit_probe.protocol.v1'
OVERFIT_REPORT_FORMAT = 'replica.vision_overfit_probe.report.v1'
OVERFIT_FIELDS = {
    'format', 'root', 'stage', 'vision', 'vision_digest', 'vision_source', 'model',
    'data', 'implementation_sha256', 'limits', 'seed', 'train_worlds', 'dev_worlds',
    'per_world_cases', 'train', 'eval', 'initial_checkpoint',
}
EVAL_CONTROLS = ('correct', 'wrong', 'noimage')


def validate_overfit_protocol(protocol):
    require(isinstance(protocol, dict) and set(protocol) == OVERFIT_FIELDS,
            'Overfit protocol fields differ')
    require(protocol['format'] == OVERFIT_FORMAT, 'Unknown overfit protocol format')
    require(protocol['stage'] == 'v1', 'Overfit probe requires the v1 data stage')
    for name in ('train_worlds', 'dev_worlds', 'per_world_cases'):
        require(type(protocol[name]) is int and protocol[name] >= 1,
                f'{name} must be a positive integer')
    require(protocol['train_worlds'] <= 16 and protocol['dev_worlds'] <= 16,
            'Overfit probe stays a tiny diagnostic cohort')
    train = protocol['train']
    require(type(train['updates']) is int and 1 <= train['updates'] <= 32,
            'Overfit updates stay bounded')
    require(train['grad_accum'] >= 1 and type(train['grad_accum']) is int,
            'Overfit grad_accum must be a positive integer')
    require(train['lr_new'] > 0 and type(train['lr_new']) in (int, float),
            'Overfit learning rate must be positive')
    require(type(train['repeats']) is int and 1 <= train['repeats'] <= 4,
            'Overfit repeats stay bounded at four seeded epochs')
    require(train['seed'] >= 0 and type(train['seed']) is int,
            'Overfit seed must be a nonnegative integer')
    require(train['optimizer'] in ('muon8bit', 'sgd'),
            'Overfit optimizer must be the declared Muon or SGD arm')
    require(protocol['eval']['max_new_tokens'] >= 2,
            'Overfit evaluation needs at least two generated tokens')
    require('max_export_bytes' in protocol['limits']
            and 'max_prompt_tokens' in protocol['limits'],
            'Overfit limits must pin export and prompt budgets')
    require(isinstance(protocol['implementation_sha256'], dict)
            and bool(protocol['implementation_sha256']),
            'Overfit implementation pins are required')
    rows = (protocol['train_worlds'] * protocol['per_world_cases'] * 3
            + protocol['dev_worlds'] * protocol['per_world_cases'] * 3)
    return {'training_updates': protocol['train']['updates'],
            'evaluation_generations': rows, 'optimizer_updates':
                protocol['train']['updates'],
            'max_eval_seconds': protocol['limits']['max_seconds']}


def overfit_training_rows(dataset, cases):
    """Response-supervised rows for the selected train worlds only."""
    rows = []
    for item in cases:
        case_id = item['case_id']
        case = dataset.case(case_id)
        row = item['row']
        response = dataset.label(case_id)
        require(response, f'Empty verifier response: {case_id}')
        require(response not in row['question'].casefold() or len(response) <= 1,
                f'Answer leaked into question: {case_id}')
        rows.append({
            'case_id': case_id,
            'prompt': VISUAL_QUESTION_TEMPLATE.format(question=row['question']),
            'response': response,
            'image': row['image'], 'image_sha256': row['image_sha256'],
            'family': case['group_id'], 'group_id': case['group_id'],
            'split': case['split'],
        })
    require(bool(rows), 'No overfit training rows')
    return rows


def evaluate_cases(model, tokenizer, service, dataset, cases, *, scope, output,
                   max_new_tokens=8, generations=None):
    """Generate correct/wrong/no-image answers with the production session path."""
    from replica_lab.memory_system.ledger import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.architecture.contracts import TaskGraph
    import torch

    ledger = RecordLedger(Path(output) / 'ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger)
    questions = []
    for item in cases:
        case_id = item['case_id']
        for control in EVAL_CONTROLS:
            if control == 'correct':
                image_case = case_id
            elif control == 'wrong':
                image_case = item['wrong_case_id']
            else:
                image_case = None
            question = item['row']['question']
            visual = None
            if image_case is not None:
                visual = service.observe(str(dataset.image_path(image_case)),
                                         source_id='vision-overfit', task_id=case_id,
                                         scope=scope, observed_at=0, source_epoch=0)
            ids = encode_prompt(tokenizer, question, session.device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            request = session.prepare(
                prompt_ids=ids, prompt_mask=mask, query=question, scope=scope,
                task=TaskGraph(task_id=case_id, goal=question),
                training=False, visual=visual, enabled_ports=(),
                core_impl='none', enabled_branches=('memory',))
            try:
                prompt_length = int(request.frame.sense.prompt_ids.shape[1])
                generated_ids = session.generate(request, max_new_tokens=max_new_tokens)
                generated = [int(value) for value in generated_ids[0, prompt_length:].tolist()]
            finally:
                request.release()
            text = session.tokenizer.decode(generated, skip_special_tokens=True)
            questions.append({
                'case_id': case_id, 'control': control,
                'image_case_id': image_case, 'generated_text': text,
                'correct': answer_correct(text, dataset.label(case_id)),
                'generated_tokens': len(generated),
            })
    ledger.close()
    return questions


def summarize(rows):
    summary = {}
    for control in EVAL_CONTROLS:
        selected = [row for row in rows if row['control'] == control]
        summary[control] = {
            'cases': len(selected),
            'correct': sum(row['correct'] for row in selected),
            'accuracy': (sum(row['correct'] for row in selected) / len(selected)
                         if selected else None),
        }
    return summary


def decide_overfit(train_summary, dev_summary):
    correct_train = train_summary['correct']['accuracy']
    correct_dev = dev_summary['correct']['accuracy']
    wrong_dev = dev_summary['wrong']['accuracy']
    noimage_dev = dev_summary['noimage']['accuracy']
    if correct_train < 0.75:
        decision = 'pipeline_capacity_or_training_failure'
        change = ('The same pipeline cannot fit its training worlds; inspect '
                  'supervision/labels, projector gradient path and optimizer '
                  'before changing capacity or data.')
    elif correct_dev <= max(wrong_dev, noimage_dev) + 0.05:
        decision = 'overfit_without_image_grounded_generalization'
        change = ('Training-fit high but fresh-world accuracy does not beat '
                  'wrong/no-image controls; test one connector change (wider '
                  'projector with LayerNorm) instead of more epochs.')
    else:
        decision = 'overfit_cohort_shows_image_grounding'
        change = 'Fresh-world image accuracy exceeds controls; evidence supports a bounded V1 retry.'
    return {'decision': decision, 'justified_next_change': change,
            'train_accuracy': correct_train, 'dev_accuracy': correct_dev,
            'dev_wrong_accuracy': wrong_dev, 'dev_noimage_accuracy': noimage_dev}
