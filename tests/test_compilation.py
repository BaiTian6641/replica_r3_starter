"""M1: verified-trajectory compilation, publication, restart and invalidation."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from replica_lab.contracts import fingerprint
from replica_lab.learning import ManifestRegistry
from replica_lab.memory_system.ledger import MemoryRecord
from replica_lab.runtime.artifacts import ArtifactRegistry, NeuralCandidate
from replica_lab.runtime.compilation import (
    CompilationBinding, CompiledProcedure, StaleSourceError)
from replica_lab.runtime.procedures import RepairProcedure
from replica_lab.runtime.tools import FixtureWorkspace

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'fixtures/e3_m0a_menus/cases'


def _workspace(tmp_path, case_id, suffix):
    target = tmp_path / f'{case_id}-{suffix}'
    shutil.copytree(CASES / case_id, target)
    return FixtureWorkspace(target)


def _records(tools):
    failure = tools.run_test({})['receipt']
    procedure = RepairProcedure('repair-config', 'workspace/app_settings.py',
                                ('train-failure', 'train-repair'))
    run = procedure.execute(tools)
    assert run.status == 'succeeded'
    records = (
        MemoryRecord(record_id='train-failure', subject_id='subject', branch_id='train',
                     text=json.dumps(failure, sort_keys=True), observed_at=1, event_at=1,
                     purpose='m1', kind='tool', authenticity='observed_tool'),
        MemoryRecord(record_id='train-repair', subject_id='subject', branch_id='train',
                     text=json.dumps(run.receipts, sort_keys=True), observed_at=2, event_at=2,
                     purpose='m1', kind='tool', authenticity='observed_tool'),
    )
    return procedure, records


def _compiled(tmp_path, *, parent_manifest_id=None):
    tools = _workspace(tmp_path, 'm0a_train_queue', 'source')
    procedure, records = _records(tools)
    binding = CompilationBinding('subject', 'train', 'm1', 2, 0,
                                 'minicpm5-test', 'generic_procedure', parent_manifest_id)
    artifacts = ArtifactRegistry()
    compiled = CompiledProcedure.compile(procedure, binding, records, artifacts)
    manifests = ManifestRegistry(tmp_path / 'manifests')
    return compiled, artifacts, manifests, records


def _evaluate(compiled, artifacts, tmp_path):
    return compiled.evaluate(artifacts, [
        ('m0a_dev_retry', 'dev', _workspace(tmp_path, 'm0a_dev_retry', 'dev')),
        ('m0a_heldout_poll', 'heldout', _workspace(tmp_path, 'm0a_heldout_poll', 'heldout')),
    ])


def _hashes(records):
    return {record.record_id: record.record_hash for record in records}


def test_compile_execute_publish_restart_and_rollback(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    assert evaluation.accepted is True
    assert all(case['independent_passed'] for case in evaluation.cases)
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifest['parent'] is None
    assert manifest['payload']['artifact']['status'] == 'accepted'

    # A fresh process reads the same manifest and reconstructs the executable artifact.
    restored_manifest = ManifestRegistry(tmp_path / 'manifests').active()
    restored = CompiledProcedure.from_manifest(restored_manifest, ArtifactRegistry())
    procedure = RepairProcedure(restored.artifact.name, 'workspace/app_settings.py',
                                restored.artifact.source_event_ids)
    run = procedure.execute(_workspace(tmp_path, 'm0a_heldout_heartbeat', 'restart'))
    assert run.status == 'succeeded'
    assert run.final_exit_code == 0

    # Rollback to the original no-parent state is represented by publishing an
    # explicit parent first in a separate registry; stale-parent checks are below.
    with pytest.raises(ValueError, match='rollback target is not an ancestor'):
        compiled.rollback(artifacts, manifests, 'not-an-ancestor')


def test_source_change_before_publication_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    changed = _hashes(records)
    changed['train-failure'] = hashlib.sha256(b'changed evidence').hexdigest()
    with pytest.raises(StaleSourceError):
        compiled.publish(artifacts, manifests, evaluation, changed)
    assert manifests.active() is None


def test_stale_active_parent_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    parent_payload = {'base_revision': 'base', 'identity_version': 'parent'}
    manifests.publish(parent_payload, {
        'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
        'parent_manifest_id': None}, expected_parent=None)
    with pytest.raises(ValueError, match='active parent changed'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))


def test_interrupted_publication_preserves_active_and_allows_retry(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    import replica_lab.learning as learning
    real_replace = learning.os.replace
    calls = {'count': 0}

    def fail_second(source, target):
        calls['count'] += 1
        if calls['count'] == 2:
            raise OSError('simulated interruption before active publication')
        real_replace(source, target)

    monkeypatch.setattr(learning.os, 'replace', fail_second)
    with pytest.raises(OSError, match='simulated interruption'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None
    monkeypatch.setattr(learning.os, 'replace', real_replace)
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active()['manifest_id'] == manifest['manifest_id']


def test_source_invalidation_revokes_procedure_and_descendant(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    child = NeuralCandidate(
        'neural-child', hashlib.sha256(b'parent').hexdigest(), ('core.0.weight',),
        hashlib.sha256(b'weights').hexdigest(), 'modules_only', 'muon8bit',
        {'core': 'v1'}, source_ids=('train-repair',),
        parent_id=compiled.artifact.artifact_id, status='accepted')
    artifacts.add(child)
    affected = artifacts.invalidate_source('train-failure', reason='source revoked')
    assert compiled.artifact.artifact_id in affected
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
    assert artifacts.get('neural-child').status == 'revoked'
    with pytest.raises(PermissionError):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active()['manifest_id'] == manifest['manifest_id']


def test_rollback_to_ancestor_revokes_compiled_artifact(tmp_path):
    parent_payload = {'base_revision': 'base', 'identity_version': 'parent'}
    manifests = ManifestRegistry(tmp_path / 'manifests')
    parent = manifests.publish(parent_payload, {
        'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
        'parent_manifest_id': None}, expected_parent=None)
    compiled, artifacts, _, records = _compiled(
        tmp_path, parent_manifest_id=parent['manifest_id'])
    # Use the same manifest directory created by the parent.
    compiled_manifests = ManifestRegistry(tmp_path / 'manifests')
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    manifest = compiled.publish(artifacts, compiled_manifests, evaluation, _hashes(records))
    assert manifest['parent'] == parent['manifest_id']
    restored = compiled.rollback(artifacts, compiled_manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
