#!/usr/bin/env python3
"""Run one isolated bounded procedure-RSI plumbing cycle on CPU.

The run owns three synthetic world families (practice, independent
verification, rejection), executes real fixture tools and evaluators, publishes
one verified procedure into a fresh manifest registry, contains a rejected
candidate, and rolls back to the protected ancestor.  Neural promotion is not
attempted: the gate reads independently pinned evidence and reports blockers.
No real selected manifest, model, checkpoint or baseline environment is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Inherited PYTHONPATH may point at an old export; always prepend this checkout.
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.runtime.bounded_rsi import (  # noqa: E402
    BoundedRsiCycle, BoundedRsiError, EvidenceIntegrityError, HostPolicy,
    ResourceGuard)


def build_policy() -> HostPolicy:
    return HostPolicy(
        subject_id='owned-procedure-subject',
        branch_id='bounded-procedure-sandbox',
        purpose='bounded_procedure_rsi',
        base_revision='procedure-sandbox-r0',
        identity_version='replica.identity.procedure_sandbox.v1',
        evidence_epoch=0,
        source_family='source.queue_policy.v1',
        verification_family='verify.retry_policy.v1',
        rejection_family='reject.dual_constraint.v1',
        guard=ResourceGuard(max_tool_calls=8, max_evaluator_runs=5),
    )


def fresh_directory(path: Path) -> Path:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f'output path exists and is not a directory: {path}')
        if any(path.iterdir()):
            raise SystemExit(
                f'output directory is not empty; refusing to reuse a registry: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    path.write_text(text, encoding='utf-8')
    return text


def summarize(report, output: Path) -> dict:
    data = report.as_dict()
    published = data['publication']
    return {
        'status': 'qualified',
        'qualification_scope': data['qualification']['scope'],
        'integrated_rsi': data['qualification']['integrated_rsi'],
        'acceleration_claim': data['qualification']['acceleration_claim'],
        'isolated_output': str(output),
        'worlds': {
            'source_family': data['worlds']['families']['source'],
            'verification_family': data['worlds']['families']['verification'],
            'rejection_family': data['worlds']['families']['rejection'],
            'source_family_cases': data['worlds']['source_family_cases'],
            'verification_family_cases': data['worlds']['verification_family_cases'],
            'rejection_family_cases': data['worlds']['rejection_family_cases'],
            'case_ids_disjoint': data['worlds']['case_ids_disjoint'],
        },
        'base_manifest_id': data['base']['manifest_id'],
        'published_manifest_id': published['manifest_id'],
        'parent_manifest_id': published['parent'],
        'artifact_id': published['artifact_id'],
        'verification_accepted': data['verification']['accepted'],
        'verification_cases': data['verification']['cases'],
        'rejection': {
            'accepted': data['rejection']['evaluation_accepted'],
            'publish_refused': data['rejection']['publish_refused'],
            'publish_error': data['rejection']['publish_error'],
            'active_unchanged': data['rejection']['active_unchanged'],
            'artifact_status': data['rejection']['artifact_status'],
        },
        'rollback': data['rollback'],
        'procedure_gain': data['procedure_gain'],
        'neural': {
            'decision': data['neural']['decision'],
            'blockers': data['neural']['blockers'],
            'software_promoted': data['neural']['software_promoted'],
            'promotion_authority': data['neural']['promotion_authority'],
        },
        'selector_policy_changes': data['selector_policy']['changes'],
        'replay_digest': data['replay_digest'],
        'report': str(output / 'cycle_report.json'),
        'raw_traces': str(output / 'raw_traces.json'),
    }


def run_one(output: Path, evidence_root: Path):
    fresh_directory(output)
    cycle = BoundedRsiCycle(output, build_policy(), evidence_root=evidence_root)
    return cycle.run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='One isolated bounded procedure-RSI plumbing cycle.')
    parser.add_argument(
        '--output', type=Path, default=None,
        help='fresh output directory (default: runs/bounded_rsi_cycle_<utc>)')
    parser.add_argument(
        '--replay-check', action='store_true',
        help='run a second isolated cycle and require identical replay hashes')
    parser.add_argument(
        '--evidence-root', type=Path, default=ROOT,
        help='repository root holding pinned neural evidence (default: checkout)')
    args = parser.parse_args(argv)

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = ROOT / 'runs' / f'bounded_rsi_cycle_{stamp}'
    else:
        output = args.output.resolve()
    evidence_root = args.evidence_root.resolve()

    try:
        report = run_one(output, evidence_root)
    except EvidenceIntegrityError as error:
        print(json.dumps({'status': 'failed_evidence_integrity',
                          'error': str(error)}, sort_keys=True))
        return 2
    except BoundedRsiError as error:
        print(json.dumps({'status': 'failed_bounded_cycle',
                          'error': str(error)}, sort_keys=True))
        return 4

    data = report.as_dict()
    raw_traces = data.pop('raw_traces')
    write_json(output / 'cycle_report.json', data)
    write_json(output / 'raw_traces.json', raw_traces)
    summary = summarize(report, output)

    if args.replay_check:
        replay_output = output / 'replay'
        try:
            replay = run_one(replay_output, evidence_root)
        except (EvidenceIntegrityError, BoundedRsiError) as error:
            print(json.dumps({'status': 'failed_replay',
                              'error': str(error)}, sort_keys=True))
            return 5
        replay_data = replay.as_dict()
        replay_raw = replay_data.pop('raw_traces')
        write_json(replay_output / 'cycle_report.json', replay_data)
        write_json(replay_output / 'raw_traces.json', replay_raw)
        summary['replay_check'] = {
            'first_digest': data['replay_digest'],
            'second_digest': replay_data['replay_digest'],
            'identical': data['replay_digest'] == replay_data['replay_digest'],
            'second_output': str(replay_output),
        }
        if not summary['replay_check']['identical']:
            summary['status'] = 'failed_replay_mismatch'
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 3

    checks = (
        data['qualification']['scope'] == 'procedure_plumbing_only',
        data['qualification']['integrated_rsi'] is False,
        data['qualification']['acceleration_claim'] is None,
        data['verification']['accepted'] is True,
        data['publication']['parent'] == data['base']['manifest_id'],
        data['rejection']['evaluation_accepted'] is False,
        data['rejection']['publish_refused'] is True,
        data['rejection']['active_unchanged'] is True,
        data['rollback']['base_restored'] is True,
        data['rollback']['revoked'] is True,
        data['rollback']['active_after_rollback'] == data['base']['manifest_id'],
        data['neural']['software_promoted'] is False,
        data['selector_policy']['changes'] == [],
    )
    if not all(checks):
        summary['status'] = 'failed_cycle_invariant'
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 6
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
