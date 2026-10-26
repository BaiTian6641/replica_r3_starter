#!/usr/bin/env python3
"""Run the M1 thin procedure compiler and write raw evidence (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.contracts import fingerprint
from replica_lab.learning import ManifestRegistry
from replica_lab.memory_system.ledger import (
    EligibilityScope, MemoryRecord, RecordLedger)
from replica_lab.runtime.artifacts import ArtifactRegistry, NeuralCandidate
from replica_lab.runtime.compilation import (
    CompilationBinding, CompiledProcedure, StaleSourceError)
from replica_lab.runtime.procedures import RepairProcedure
from replica_lab.runtime.tools import FixtureWorkspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_case(source, destination):
    if destination.exists():
        raise ValueError(f'workspace already exists: {destination}')
    shutil.copytree(source, destination)
    return FixtureWorkspace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/m1_procedure_compilation_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    report = {
        'status': 'started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha(args.config),
        'protocol': config,
        'implementation_sha256': {
            name: sha(ROOT / name)
            for name in (
                'scripts/run_m1_compilation.py',
                'src/replica_lab/runtime/compilation.py',
                'src/replica_lab/runtime/artifacts.py',
                'src/replica_lab/runtime/procedures.py',
                'src/replica_lab/runtime/tools.py',
                'src/replica_lab/learning.py',
            )
        },
        'native_model_loads': 0,
        'optimizer_updates': 0,
        'downloads': 0,
    }
    try:
        fixtures = config['fixtures']
        root = ROOT / fixtures['root']
        ledger = RecordLedger(args.output / 'ledger.sqlite')
        tools = copy_case(root / fixtures['train'], args.output / 'workspaces' / 'train')
        failure = tools.run_test({})['receipt']
        if failure['exit_code'] == 0:
            raise ValueError('training fixture did not produce the required observed failure')
        procedure = RepairProcedure('repair-config', 'workspace/app_settings.py',
                                    tuple(config['source_events']))
        source_run = procedure.execute(tools)
        if source_run.status != 'succeeded':
            raise ValueError(f'verified source trajectory failed: {source_run.error}')
        records = (
            MemoryRecord(record_id='train-failure', subject_id=config['binding']['subject_id'],
                         branch_id='train', text=json.dumps(failure, sort_keys=True),
                         observed_at=1, event_at=1, purpose='m1', kind='tool',
                         authenticity='observed_tool'),
            MemoryRecord(record_id='train-repair', subject_id=config['binding']['subject_id'],
                         branch_id='train', text=json.dumps(source_run.receipts, sort_keys=True),
                         observed_at=2, event_at=2, purpose='m1', kind='tool',
                         authenticity='observed_tool'),
        )
        for record in records:
            ledger.admit(record, authorized_subject=record.subject_id,
                         authorized_branch=record.branch_id)
        snapshot = ledger.snapshot(EligibilityScope(
            config['binding']['subject_id'], 'train', 'm1', config['binding']['cutoff']))
        current = {record.record_id: record.record_hash for record in snapshot.records}
        if set(current) != set(config['source_events']):
            raise ValueError('ledger does not contain the complete source trajectory')
        report['source_receipts'] = {'failure': failure, 'repair': list(source_run.receipts)}
        report['source_hashes'] = current

        parent_payload = {
            'base_revision': config['binding']['base_revision'],
            'identity_version': 'baseline_parent',
        }
        manifests = ManifestRegistry(args.output / 'manifests')
        parent = manifests.publish(parent_payload, {
            'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
            'parent_manifest_id': None}, expected_parent=None)
        binding = CompilationBinding(
            config['binding']['subject_id'], 'train', 'm1', config['binding']['cutoff'],
            config['binding']['evidence_epoch'], config['binding']['base_revision'],
            config['binding']['identity_version'], parent['manifest_id'])
        artifacts = ArtifactRegistry()
        compiled = CompiledProcedure.compile(procedure, binding, snapshot.records, artifacts)
        evaluation = compiled.evaluate(artifacts, [
            (fixtures['dev'], 'dev', copy_case(root / fixtures['dev'],
                                               args.output / 'workspaces' / 'dev')),
            *[(case_id, 'heldout', copy_case(root / case_id,
                                              args.output / 'workspaces' / case_id))
              for case_id in fixtures['heldout']],
        ])
        report['evaluation'] = evaluation.publication()
        if not evaluation.accepted:
            raise ValueError('independent procedure evaluation failed')
        manifest = compiled.publish(artifacts, manifests, evaluation, current)
        report['publication'] = {'manifest_id': manifest['manifest_id'], 'parent': manifest['parent']}

        reload_registry = ManifestRegistry(args.output / 'manifests')
        restored = CompiledProcedure.from_manifest(
            reload_registry.active(), ArtifactRegistry(),
            manifests=reload_registry)
        restart_procedure = RepairProcedure(
            restored.artifact.name, 'workspace/app_settings.py', restored.artifact.source_event_ids)
        restart_tools = copy_case(root / fixtures['restart_unseen'],
                                  args.output / 'workspaces' / 'restart_unseen')
        restart_run = restart_procedure.execute(restart_tools)
        report['restart'] = {
            'case_id': fixtures['restart_unseen'], 'status': restart_run.status,
            'final_exit_code': restart_run.final_exit_code,
            'independent_passed': restart_tools.verify_success().get('passed'),
            'receipts': list(restart_run.receipts),
        }
        if restart_run.status != 'succeeded' or restart_run.final_exit_code != 0:
            raise ValueError('restart reconstruction did not execute the accepted procedure')

        try:
            compiled.assert_sources_current({**current, 'train-failure': '0' * 64})
            raise ValueError('changed source was not rejected')
        except StaleSourceError as error:
            report['source_change_rejected'] = str(error)

        child = NeuralCandidate(
            'm1-neural-child', hashlib.sha256(b'parent').hexdigest(), ('core.0.weight',),
            hashlib.sha256(b'weights').hexdigest(), 'modules_only', 'muon8bit',
            {'core': 'v1'}, source_ids=('train-repair',),
            parent_id=compiled.artifact.artifact_id, status='accepted')
        artifacts.add(child)
        affected = artifacts.invalidate_source('train-failure', reason='source revoked')
        report['invalidation'] = {
            'affected': list(affected),
            'procedure_status': artifacts.get(compiled.artifact.artifact_id).status,
            'descendant_status': artifacts.get('m1-neural-child').status,
        }
        if report['invalidation']['procedure_status'] != 'revoked' or \
                report['invalidation']['descendant_status'] != 'revoked':
            raise ValueError('source invalidation did not revoke the artifact and descendant')
        restored_parent = compiled.rollback(artifacts, manifests, parent['manifest_id'])
        report['rollback'] = {
            'active_manifest_id': restored_parent['manifest_id'],
            'procedure_status': artifacts.get(compiled.artifact.artifact_id).status,
        }
        if restored_parent['manifest_id'] != parent['manifest_id']:
            raise ValueError('rollback did not restore the parent')
        catalog = artifacts.save(args.output / 'artifact_catalog.json')
        report['artifact_catalog'] = catalog
        report['acceptance'] = {
            'evaluation_accepted': evaluation.accepted,
            'publication_succeeded': True,
            'restart_unseen_passed': True,
            'changed_source_rejected': True,
            'invalidation_revoked': True,
            'rollback_restored_parent': True,
        }
        report['status'] = 'completed'
        report['claims'] = config['claims']
        ledger.close()
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        (args.output / 'failure_report.json').write_text(json.dumps(report, indent=2) + '\n')
        raise
    finally:
        report['costs'] = {'wall_seconds': time.monotonic() - started,
                           'native_model_loads': 0, 'optimizer_updates': 0,
                           'new_base_copies': 0, 'downloads': 0}
        report['artifact_bytes'] = sum(path.stat().st_size for path in args.output.rglob('*')
                                       if path.is_file())
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        if report['status'] == 'completed':
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'acceptance': report.get('acceptance'),
                      'costs': report.get('costs')}))


if __name__ == '__main__':
    main()
