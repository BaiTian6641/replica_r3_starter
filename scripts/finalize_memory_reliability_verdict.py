#!/usr/bin/env python3
"""Finalize a reliability verdict after a scorer-only correction.

The raw native reports are never edited. This utility authenticates those reports,
then links a temporary re-pinned scorer artifact back to the original protocol,
guards and implementation without changing the measured decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_metadata(path):
    guard = json.loads(path.read_text())
    return {'path': str(path), 'sha256': sha(path), 'policy': guard['guard']['policy']}


def differences(native, smoke, prefix=''):
    if isinstance(native, dict) and isinstance(smoke, dict):
        missing = sorted(set(native) ^ set(smoke))
        if missing:
            return [f'{prefix}.{name}' if prefix else name for name in missing]
        return [item for name in native for item in differences(native[name], smoke[name], f'{prefix}.{name}' if prefix else name)]
    if isinstance(native, list) and isinstance(smoke, list):
        if len(native) != len(smoke):
            return [prefix]
        return [item for index, (left, right) in enumerate(zip(native, smoke))
                for item in differences(left, right, f'{prefix}.{index}' if prefix else str(index))]
    return [] if native == smoke else [prefix]


def require_canonical_fixture(protocol, run, restart, fixture, native_guard, native_restart_guard):
    fixture_protocol = json.loads((fixture / 'protocol.json').read_text())
    fixture_run = json.loads((fixture / 'run' / 'run_report.json').read_text())
    fixture_restart = json.loads((fixture / 'restart' / 'run_report.json').read_text())
    fixture_guard = json.loads((fixture / 'guard.json').read_text())
    fixture_restart_guard = json.loads((fixture / 'restart_guard.json').read_text())
    if differences(protocol, fixture_protocol) != ['implementation_sha256.scripts/score_memory_reliability.py']:
        raise ValueError('Smoke protocol is not a scorer-only re-pin')
    allowed_run = {'protocol_sha256', 'implementation_sha256.scripts/score_memory_reliability.py'}
    allowed_run.update(f'transitions.{index}.image' for index in range(len(run['transitions'])))
    if set(differences(run, fixture_run)) - allowed_run:
        raise ValueError('Smoke run is not a pin-only transformation of the native run')
    if set(differences(restart, fixture_restart)) - {'protocol_sha256', 'previous_report_sha256'}:
        raise ValueError('Smoke restart is not a pin-only transformation of the native restart')
    if set(differences(native_guard, fixture_guard)) - {'output'}:
        raise ValueError('Smoke guard is not a pin-only transformation')
    if set(differences(native_restart_guard, fixture_restart_guard)) - {'output'}:
        raise ValueError('Smoke restart guard is not a pin-only transformation')
    return fixture_protocol, fixture_run, fixture_restart


def case_scope(case):
    from replica_lab.memory_system import EligibilityScope
    return EligibilityScope(case['subject_id'], case['branch_id'], case['purpose'],
                            case['as_of'], 'parallel_replica', None)


def gate_off_count(episodes):
    from replica_lab.memory_reliability import cases
    from replica_lab.memory_system import RecordLedger
    from replica_lab.runtime.exact_memory import ExactMemoryReference
    reference = ExactMemoryReference(RecordLedger(':memory:'))
    try:
        for episode in episodes:
            reference.admit_episode(episode)
            reference.correct_episode(episode)
        total = 0
        for episode in episodes:
            if episode['split'] != 'train':
                continue
            for stage in ('initial', 'corrected'):
                for case in cases(episode, stage):
                    if not reference.select(case['query'], case_scope(case)).enabled:
                        total += 1
        return total
    finally:
        reference.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--restart-run', type=Path, required=True)
    parser.add_argument('--restart-guard', type=Path, required=True)
    parser.add_argument('--smoke-verdict', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh finalized verdict path')
    protocol = json.loads(args.protocol.read_text())
    run = json.loads((args.run / 'run_report.json').read_text())
    restart = json.loads((args.restart_run / 'run_report.json').read_text())
    smoke = json.loads(args.smoke_verdict.read_text())
    protocol_sha = sha(args.protocol)
    if (run['protocol_sha256'] != protocol_sha or restart['protocol_sha256'] != protocol_sha
            or run['implementation_sha256'] != protocol['implementation_sha256']
            or run['status'] != 'completed' or restart['status'] != 'completed'):
        raise ValueError('Native protocol or implementation lineage differs')
    if not smoke.get('report_sha256') or not smoke['restart'].get('report_sha256'):
        raise ValueError('Scorer smoke is missing report hashes')
    if smoke['guard']['policy'] != guard_metadata(args.guard)['policy']:
        raise ValueError('Scorer smoke guard policy differs')
    if smoke['restart']['guard']['policy'] != guard_metadata(args.restart_guard)['policy']:
        raise ValueError('Scorer smoke restart guard policy differs')
    fixture_protocol, fixture_run, fixture_restart = require_canonical_fixture(
        protocol, run, restart, args.fixture,
        json.loads(args.guard.read_text()), json.loads(args.restart_guard.read_text()))
    if (smoke['report_sha256'] != sha(args.fixture / 'run' / 'run_report.json')
            or smoke['restart']['report_sha256'] != sha(args.fixture / 'restart' / 'run_report.json')
            or smoke['protocol_sha256'] != sha(args.fixture / 'protocol.json')
            or smoke['guard']['sha256'] != sha(args.fixture / 'guard.json')
            or smoke['restart']['guard']['sha256'] != sha(args.fixture / 'restart_guard.json')):
        raise ValueError('Scorer output does not belong to the canonical fixture')
    if smoke['decision'] != run['decision'] or smoke['candidate_decision'] != run['decision']:
        raise ValueError('Scorer decision differs from native decision')
    if smoke['metrics'] != run['metrics']:
        raise ValueError('Scorer metrics differ from native metrics')
    if set(smoke['failed_checks']) != {name for name, value in run['checks'].items() if not value}:
        raise ValueError('Scorer failed gates differ from native gates')
    native_report_sha = sha(args.run / 'run_report.json')
    native_restart_sha = sha(args.restart_run / 'run_report.json')
    finalized = dict(smoke)
    finalized.update(
        report_sha256=native_report_sha,
        protocol_sha256=protocol_sha,
        implementation_sha256=protocol['implementation_sha256'],
        guard=guard_metadata(args.guard),
        restart=dict(smoke['restart'], report_sha256=native_restart_sha,
                     guard=guard_metadata(args.restart_guard),
                     scope='Persisted-image source-binding parity; not live-ledger epoch equality'),
        evidence_scope={
            'novelty': 'Heldout full observed-value novelty; train/dev distractors bronze and silver are shared',
            'restart': 'Persisted-image source-binding parity, not live-ledger epoch equality',
            'trained_parameters': run['training']['parameter_audit']['new']['trainable'],
            'export_time_trainable_flags': run['extension']['parameters']['new']['trainable'],
            'changed_tensors': smoke['checkpoint_changes']['changed_tensors']},
        accounting_errata={
            'gate_off_examples': {
                'reported_total': sum(update.get('gate_off_examples', 0)
                                      for update in run['training']['updates']),
                'independently_recomputed_total': gate_off_count(
                    json.loads((ROOT / protocol['dataset']['path']).read_text())),
                'status': 'unauthenticated',
                'explanation': 'The native counter reports zero while 32 unknown plus 32 general cases are gate-off by construction. Per-example no-grad status is not persisted; the gate-on-only objective cannot be authenticated from that counter.'}},
        metadata_correction={
            'reason': 'Restored args=parser.parse_args() before output-path validation in the independent scorer.',
            'scorer_sha256': sha(ROOT / 'scripts/score_memory_reliability.py'),
            'scorer_smoke': str(args.smoke_verdict),
            'scorer_smoke_sha256': sha(args.smoke_verdict),
            'decision_unchanged': True,
            'corrected_at_utc': datetime.now(timezone.utc).isoformat(),
            'finalizer_script_sha256': sha(Path(__file__)),
            'smoke_fixture': {
                'report_sha256': smoke['report_sha256'],
                'restart_report_sha256': smoke['restart']['report_sha256'],
                'protocol_sha256': smoke['protocol_sha256'],
                'native_protocol_sha256': protocol_sha,
                'native_scorer_sha256': protocol['implementation_sha256']['scripts/score_memory_reliability.py'],
                'corrected_scorer_sha256': sha(ROOT / 'scripts/score_memory_reliability.py'),
                'note': 'Temporary re-pinned scorer copies are smoke fixtures, not native evidence. Native hashes are restored above.'}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finalized, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'decision': finalized['decision'],
                      'reference_decision': finalized['reference_decision'],
                      'candidate_decision': finalized['candidate_decision']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
