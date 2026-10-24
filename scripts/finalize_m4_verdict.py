#!/usr/bin/env python3
"""Finalize M4 verdict metadata without changing raw results or decision.

This host-only utility corrects preservation metadata on an already scored
artifact. It never regenerates answers, reopens a training arm, or edits the
input verdict/report.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-report', type=Path, required=True)
    parser.add_argument('--verdict', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--original-sha256', required=True)
    parser.add_argument('--collision-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh finalized-verdict path')
    protocol = json.loads(args.protocol.read_text())
    run = json.loads(args.run_report.read_text())
    verdict = json.loads(args.verdict.read_text())
    original = json.loads(args.original.read_text())
    audit = json.loads(args.collision_audit.read_text())
    protocol_sha = sha(args.protocol)
    if (run.get('protocol_sha256') != protocol_sha
            or verdict.get('protocol_sha256') != protocol_sha
            or run.get('implementation_sha256') != protocol['implementation_sha256']
            or verdict.get('implementation_sha256') != protocol['implementation_sha256']
            or verdict.get('verdict_source_sha256') != sha(args.run_report)):
        raise ValueError('Finalization lineage does not match the v7 run and protocol')
    if (args.original.resolve() != (ROOT / protocol['no_warmup_control']['report']).resolve()
            or sha(args.original) != args.original_sha256
            or original.get('decision') != 'reject'
            or original.get('original_failure_preserved') is not True):
        raise ValueError('Original M4 rejection artifact is not authenticated')
    if verdict.get('decision') != 'reject' or run.get('decision') != 'reject':
        raise ValueError('Metadata finalization is limited to the preserved reject decision')
    if audit.get('run_report_sha256') != sha(args.run_report):
        raise ValueError('Collision audit belongs to another run')
    if not all(group.get('runner_fields_validated') for group in audit['collisions'].values()):
        raise ValueError('Collision audit has not validated all router-scope groups')
    if not all(group.get('runner_fields_validated') for group in audit['training_collisions'].values()):
        raise ValueError('Collision audit has not validated all training-scope groups')
    trained_parameters = sum(group['parameters'] for group in run['training']['optimizer_final']['groups'])
    finalized = dict(verdict)
    finalized.update(
        original_failure_preserved=True,
        original_artifact={
            'path': protocol['no_warmup_control']['report'],
            'sha256': args.original_sha256,
            'decision': original['decision'],
            'protocol_sha256': original.get('protocol_sha256')},
        metadata_correction={
            'prior_verdict': str(args.verdict),
            'prior_verdict_sha256': sha(args.verdict),
            'reason': 'The prior Boolean inspected the current run error_type; it did not verify the original M4 artifact.',
            'decision_unchanged': True,
            'corrected_at_utc': datetime.now(timezone.utc).isoformat(),
            'corrector_script_sha256': sha(Path(__file__))},
        evidence_scope={
            'direct_pattern': 'All direct-arm development/regression rows; no delayed router row was fully hash-covered, so actually routed covered-case utility is unmeasured.',
            'general_development_noninterference': 'Candidate adversarial >= parent adversarial only; it is not a disabled-base comparison.',
            'pattern_adversarial_noninterference': 'Retention compares against disabled; retention-dev compares against parent adversarial.',
            'latency': 'Single development prompt, three pattern_port forwards; first-call and warm-minimum timings are not end-to-end routing or generation latency.',
            'independent_scoring': 'Independent re-scoring of preserved JSON; answers were not regenerated.',
            'collision_audit': 'Reuses production PatternAccessPort.hash_indices; validates eight numeric fields per group, not every runner field.',
            'trained_parameters': trained_parameters,
            'extension_reported_trainable_parameters': run['parameters']['new']['trainable'],
            'parameter_note': 'The larger extension audit restores pre-run trainability flags before saving; optimizer groups are the actual v7 trained scope.',
            'collision_audit_path': str(args.collision_audit),
            'collision_audit_sha256': sha(args.collision_audit)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finalized, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'decision': finalized['decision'],
                      'original_failure_preserved': True,
                      'trained_parameters': trained_parameters}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
