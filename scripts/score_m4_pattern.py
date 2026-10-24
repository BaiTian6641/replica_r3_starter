#!/usr/bin/env python3
"""Independently score a preserved M4 run without retraining (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy(rows):
    return sum(row['correct'] for row in rows) / len(rows)


def split_metrics(rows, splits):
    return {split: accuracy([row for row in rows if row['split'] == split]) for split in splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--parent-controls', type=Path,
                        help='Existing parent adversarial controls with identical renderer and observation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh verdict output path')
    protocol = json.loads(args.protocol.read_text())
    implementation_pins = protocol['implementation_sha256']
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M4 verdict implementation changed: {name}')
    source = json.loads(args.input.read_text())
    source_lm_updates = int(protocol['training'].get('source_lm_updates', 0))
    expected_updates = int(protocol['training']['updates']) + source_lm_updates
    if source.get('status') != 'completed':
        if (source.get('error_type') != 'NameError'
                or source.get('error') != "name 'candidate' is not defined"
                or source.get('optimizer_updates') != 16 or source.get('checkpoints_saved') != 1):
            raise ValueError('M4 verdict input is not the declared completed-evaluation binding failure')
    if source.get('optimizer_updates') != expected_updates:
        raise ValueError('M4 optimizer update count differs from protocol')
    if source.get('checkpoints_saved') != protocol['budget']['checkpoint_saves']:
        raise ValueError('M4 checkpoint count differs from protocol')
    if source_lm_updates:
        source_training = source.get('training', {}).get('source_lm', {})
        if (len(source_training.get('updates', [])) != source_lm_updates
                or not source_training.get('checkpoint', {}).get('weights_sha256')):
            raise ValueError('M4 source-LM phase evidence is incomplete')
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'), ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'M4 dataset changed: {filename}')
    parent_controls = None
    if args.parent_controls is not None:
        parent_controls = json.loads(args.parent_controls.read_text())
        control_report_path = (parent_controls.get('parent_run_report')
                               or parent_controls['parent_evidence']['report'])
        control_report_sha = parent_controls.get('parent_run_report_sha256', sha(ROOT / control_report_path))
        if (Path(control_report_path).resolve() != (ROOT / protocol['parent_run_report']).resolve()
                or control_report_sha != protocol['parent_run_report_sha256']
                or Path(parent_controls['parent_extension']).resolve()
                != (ROOT / protocol['parent_extension']).resolve()):
            raise ValueError('M4 parent-control lineage differs')
        for split in ('retention', 'retention_dev'):
            if 'adversarial' not in parent_controls[f'parent_{split}']:
                raise ValueError(f'M4 parent controls lack adversarial rows for {split}')
            source['parent'][split]['adversarial'] = parent_controls[f'parent_{split}']['adversarial']
        source['parent_general_metrics']['adversarial'] = accuracy(
            parent_controls['parent_retention']['adversarial'])
        source['parent_general_dev_metrics']['adversarial'] = accuracy(
            parent_controls['parent_retention_dev']['adversarial'])
    parent_fast = source['parent_metrics']['fast']
    candidate = source['metrics']['router']
    general_parent = source['parent_general_metrics']
    general_candidate = source['general_metrics']
    general_dev_parent = source['parent_general_dev_metrics']
    general_dev_candidate = source['general_dev_metrics']
    disabled_identical = ([row['token_ids'] for row in source['parent']['main']['disabled']]
                          == [row['token_ids'] for row in source['candidate']['disabled']])
    general_disabled_identical = ([row['token_ids'] for row in source['parent']['retention']['disabled']]
                                  == [row['token_ids'] for row in source['general']['disabled']])
    general_dev_disabled_identical = ([row['token_ids'] for row in source['parent']['retention_dev']['disabled']]
                                      == [row['token_ids'] for row in source['general_dev']['disabled']])
    fast_identical = ([row['token_ids'] for row in source['parent']['main']['fast']]
                      == [row['token_ids'] for row in source['candidate']['fast']])
    audit = source['training']['updates'][1].get('parameter_gradients')
    gradient_ok = bool(audit and not audit['new']['nonfinite'] and audit['new']['nonzero'])
    checks = {
        'correct_router_above_controls': candidate['heldout'] > max(
            source['metrics']['linear']['heldout'], source['metrics']['shuffled']['heldout'],
            source['metrics']['zero']['heldout'], source['metrics']['disabled']['heldout']),
        'fresh_delayed_not_below_parent': candidate['heldout'] >= parent_fast['heldout'],
        'development_not_below_disabled': candidate['dev'] >= source['metrics']['disabled']['dev'],
        'original_regression_not_below_parent': candidate['regression'] >= parent_fast['regression'],
        'general_noninterference': (general_candidate['adversarial'] >= general_parent['adversarial']
                                    and general_candidate['adversarial'] >= general_candidate['disabled']),
        'general_development_noninterference': (
            general_dev_candidate['adversarial'] >= general_dev_parent['adversarial']),
        'disabled_token_identical': disabled_identical and general_disabled_identical and general_dev_disabled_identical,
        'fast_modules_unchanged': fast_identical,
        'post_open_gradients': gradient_ok,
        'restart_state_equal': source['restart']['state_digest_equal'],
        'restart_lineage_equal': source['restart']['lineage_equal'],
        'restart_answer_equal': source['restart']['answer_equal'],
        'correction_independent_success': all(
            item['state_changed'] and item['stale_rejected'] and item['corrected_answer']['correct']
            for item in source['correction']),
        'sparse_optimizer_step': source['training']['optimizer_final']['state_tensor_bytes'] > 0,
    }
    if source_lm_updates:
        baseline = protocol['no_warmup_control']
        checks.update({
            'direct_pattern_covered_regression': source['metrics']['pattern']['regression'] > max(
                baseline['pattern_regression'], source['metrics']['linear']['regression'],
                source['metrics']['shuffled']['regression'], source['metrics']['zero']['regression'],
                source['metrics']['disabled']['regression']),
            'direct_pattern_covered_dev': source['metrics']['pattern']['dev'] > max(
                baseline['pattern_dev'], source['metrics']['linear']['dev'],
                source['metrics']['shuffled']['dev'], source['metrics']['zero']['dev'],
                source['metrics']['disabled']['dev']),
            'pattern_adversarial_noninterference': (
                general_candidate['adversarial'] >= general_candidate['disabled']
                and general_dev_candidate['adversarial'] >= general_dev_parent['adversarial']),
        })
    decision = 'accept' if all(checks.values()) else 'reject'
    all_rows = (source['parent']['main']['fast'] + source['parent']['main']['disabled']
                + source['parent']['retention']['fast'] + source['parent']['retention']['disabled']
                + source['parent']['retention_dev']['fast'] + source['parent']['retention_dev']['disabled']
                + [row for values in source['candidate'].values() for row in values]
                + [row for values in source['general'].values() for row in values]
                + [row for values in source['general_dev'].values() for row in values]
                + source['train_results']
                + [row for item in source['correction'] for row in (item['original_answer'], item['corrected_answer'])]
                + [source['restart']['reloaded_answer'], source['restart']['correct_answer']])
    generation_counts = {
        'parent': (len(source['parent']['main']['fast']) + len(source['parent']['main']['disabled'])
                   + len(source['parent']['retention']['fast']) + len(source['parent']['retention']['disabled'])
                   + len(source['parent']['retention_dev']['fast']) + len(source['parent']['retention_dev']['disabled'])),
        'candidate': sum(len(values) for values in source['candidate'].values()),
        'general': sum(len(values) for values in source['general'].values()),
        'general_dev': sum(len(values) for values in source['general_dev'].values()),
        'train': len(source['train_results']), 'correction': 2 * len(source['correction']), 'restart': 2}
    generation_counts['total'] = sum(generation_counts.values())
    if generation_counts['total'] != len(all_rows):
        raise ValueError('M4 verdict generation accounting mismatch')
    token_costs = {
        'training_action_tokens': sum(update['action_tokens'] for update in source['training']['updates']),
        'source_lm_action_tokens': sum(update.get('source_tokens', 0)
                                       for update in source['training'].get('source_lm', {}).get('updates', [])),
        'generation_prompt_tokens': sum(row['prompt_tokens'] for row in all_rows),
        'generation_new_tokens': sum(row['generated_tokens'] for row in all_rows),
        'maximum_generation_prompt_tokens': max(row['prompt_tokens'] for row in all_rows),
        'maximum_generated_tokens_per_answer': max(row['generated_tokens'] for row in all_rows)}
    if (generation_counts['total'] > protocol['budget']['generation_answers_per_arm']
            or token_costs['generation_new_tokens'] > protocol['budget']['max_generated_tokens_per_arm']):
        raise ValueError('M4 generation budget exceeded')
    verdict = dict(source)
    verdict.update(status='completed', decision=decision, deployment='diagnostic_only',
                   protocol_sha256=sha(args.protocol), implementation_sha256=implementation_pins,
                   checks=checks, generation_counts=generation_counts, token_costs=token_costs,
                   verdict_source=str(args.input), verdict_source_sha256=sha(args.input),
                   original_failure_preserved=source.get('error_type') == 'NameError')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + '\n')
    print(json.dumps({'decision': decision, 'checks_failed': [name for name, value in checks.items() if not value],
                      'generation_answers': generation_counts['total'], 'new_tokens': token_costs['generation_new_tokens']}))


if __name__ == '__main__':
    raise SystemExit(main())
