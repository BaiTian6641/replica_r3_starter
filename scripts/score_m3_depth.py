#!/usr/bin/env python3
"""Independent paired verdict for the preregistered M3 depth arms (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def artifact_bytes(report):
    extension = Path(report['extension']['path'])
    weights = extension.parent / 'fast_image' / 'fast_depth_state.safetensors'
    return {
        'extension_bytes': sum(path.stat().st_size for path in extension.rglob('*') if path.is_file()),
        'depth_image_bytes': (weights.stat().st_size
                              + (weights.parent / 'manifest.json').stat().st_size),
    }


def paired(shared, private):
    left = {row['case_id']: row for row in shared['candidate']['correct']}
    right = {row['case_id']: row for row in private['candidate']['correct']}
    if set(left) != set(right) or not left:
        raise ValueError('M3 correct-arm case sets differ')
    rows = []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if (a['split'], a['expected']) != (b['split'], b['expected']):
            raise ValueError('M3 paired case identity or expectation differs')
        rows.append({'case_id': case_id, 'split': a['split'],
                     'shared_answer': a['answer'], 'private_answer': b['answer'],
                     'shared_correct': a['correct'], 'private_correct': b['correct'],
                     'answers_equal': a['answer'] == b['answer']})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--gradient-cut', type=Path, required=True)
    parser.add_argument('--shared-restart', type=Path, required=True)
    parser.add_argument('--private-restart', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh comparison output path')
    protocol = load(args.protocol)
    implementation_pins = protocol.get('implementation_sha256', {})
    for name, expected in implementation_pins.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError(f'M3 implementation bytes changed: {name}')
    protocol_sha = sha(args.protocol)
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'),
                          ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'M3 dataset differs: {filename}')
    paths = {'shared': args.shared, 'private': args.private, 'gradient_cut': args.gradient_cut}
    reports = {name: load(path) for name, path in paths.items()}
    restarts = {'shared': load(args.shared_restart), 'private': load(args.private_restart)}
    checks = {}
    for name, report in reports.items():
        checks[f'{name}_completed'] = report.get('status') == 'completed'
        checks[f'{name}_protocol_match'] = (report.get('protocol_sha256') == protocol_sha
                                             and report.get('depth_arm') == name)
        checks[f'{name}_updates_and_checkpoint'] = (report.get('optimizer_updates') == 16
                                                     and report.get('checkpoints_saved') == 1
                                                     and report['extension']['base_saved'] is False)
        checks[f'{name}_optimizer_state_measured'] = (
            report['training']['optimizer_final']['state_tensor_bytes'] > 0)
        checks[f'{name}_implementation_match'] = report.get('implementation_sha256') == implementation_pins
        checks[f'{name}_source_and_parent'] = (
            report['source_lock']['revision'] == protocol['source_revision']
            and report['parent_extension'] == protocol['parent_extension']
            and report.get('downloads') == 0)
        checks[f'{name}_generation_budget'] = (
            report['generation_counts']['total'] <= protocol['budget']['generation_answers_per_arm']
            and report['token_costs']['generation_new_tokens']
            <= protocol['budget']['max_generated_tokens_per_arm']
            and report['token_costs']['maximum_generation_prompt_tokens'] <= 256
            and report['token_costs']['maximum_generated_tokens_per_answer'] <= 6)
        checks[f'{name}_time_budget'] = (
            report['costs']['profiled_wall_seconds'] <= protocol['budget']['max_wall_seconds_per_arm'])
        optimizer = report['training']['optimizer']
        expected_rates = {'muon8bit': protocol['training']['lr_muon_new'],
                          'adamw': protocol['training']['lr_new']}
        checks[f'{name}_optimizer_contract'] = (
            optimizer['optimizer'] == 'muon8bit' and optimizer['parameter_dtype'] == 'float32'
            and report['parameters']['original']['trainable'] == 0
            and all(group['lr'] == expected_rates[group['algorithm']] for group in optimizer['groups'])
            and sum(group['parameters'] for group in optimizer['groups'])
            == report['depth_accounting']['writer_parameters']
            + report['depth_accounting']['reader_parameters'])
    checks['shared_private_accepted'] = all(reports[name].get('decision') == 'accept'
                                             for name in ('shared', 'private'))
    checks['gradient_cut_valid'] = (reports['gradient_cut']['checks'].get('first_chunk_gradient_cut') is True
                                    and reports['gradient_cut']['checks'].get('gradient_probe_after_output_open') is True
                                    and reports['gradient_cut'].get('deployment') == 'diagnostic_control')
    for name in ('shared', 'private'):
        restart = restarts[name]
        checks[f'{name}_fresh_restart'] = (
            restart.get('status') == 'completed' and restart.get('decision') == 'accept'
            and restart.get('fresh_process_reload') is True
            and restart.get('source_observations_reencoded') is False
            and restart.get('candidate_run_report_sha256') == sha(paths[name])
            and restart.get('implementation_sha256') == implementation_pins
            and restart.get('costs', {}).get('profiled_wall_seconds', 0)
            <= protocol['budget']['restart_max_wall_seconds']
            and all(restart.get('checks', {}).values()))
    accounting = {name: report['depth_accounting'] for name, report in reports.items()}
    for field in ('writer_parameters', 'reader_parameters', 'trainable_parameter_bytes',
                  'observation_encodings_per_chunk', 'state_transition_evaluations_per_chunk'):
        checks[f'matched_{field}'] = len({accounting[name][field] for name in accounting}) == 1
    checks['state_difference_explicit'] = (
        accounting['private']['state_bytes'] == 2 * accounting['shared']['state_bytes']
        and accounting['gradient_cut']['state_bytes'] == accounting['shared']['state_bytes']
        and 'no capacity superiority claim' in accounting['shared']['state_match'])
    paired_rows = paired(reports['shared'], reports['private'])
    verdict = 'accept_m3_ownership_interfaces' if all(checks.values()) else 'defer'
    costs = {}
    for name, report in reports.items():
        costs[name] = {
            'wall_seconds': report['costs']['profiled_wall_seconds'],
            'peak_driver_bytes': report['costs']['sampled_peak_driver_bytes'],
            'optimizer_state_bytes': report['training']['optimizer_final']['state_tensor_bytes'],
            'optimizer_state_by_dtype': report['training']['optimizer_final']['state_bytes_by_dtype'],
            'state_bytes': report['depth_accounting']['state_bytes'],
            'generation_counts': report['generation_counts'],
            'token_costs': report['token_costs'],
            **artifact_bytes(report),
        }
    result = {
        'schema_version': 'replica.m3.depth.comparison.v1',
        'protocol_sha256': protocol_sha,
        'decision': verdict,
        'checks': checks,
        'reports': {name: str(path) for name, path in paths.items()},
        'restart_reports': {'shared': str(args.shared_restart), 'private': str(args.private_restart)},
        'metrics': {name: report['metrics'] for name, report in reports.items()},
        'paired_correct_cases': paired_rows,
        'paired_answer_agreement': sum(row['answers_equal'] for row in paired_rows) / len(paired_rows),
        'depth_accounting': accounting,
        'costs': costs,
        'total_sequential_wall_seconds': sum(value['wall_seconds'] for value in costs.values()),
        'maximum_single_worker_peak_driver_bytes': max(value['peak_driver_bytes'] for value in costs.values()),
        'scope': ('Engineering qualification of distinct per-depth writers, one-start convex shared '
                  'reduction, private ownership, post-open historical gradients and fresh-process '
                  'image restart. Private has two state matrices versus shared one. Outcomes on the '
                  'exposed M2 family are descriptive; no state-matched, fresh-audit, uTTT, general '
                  'capability, runtime-promotion or shared-superiority claim.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'decision': verdict, 'checks_failed': [name for name, value in checks.items() if not value],
                      'paired_answer_agreement': result['paired_answer_agreement']}))
    return 0 if verdict.startswith('accept') else 1


if __name__ == '__main__':
    raise SystemExit(main())
