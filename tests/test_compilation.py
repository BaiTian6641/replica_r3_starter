"""M1/M2: payload binding, coordinated publication, rollback and recovery."""
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from replica_lab.contracts import fingerprint
from replica_lab.learning import ManifestRegistry
from replica_lab.memory_system.ledger import MemoryRecord
from replica_lab.runtime.artifacts import (
    ArtifactRegistry, ContextPack, NeuralCandidate, ProcedureArtifact,
    ProcedureStep, artifact_content_digest, artifact_from_dict, artifact_to_dict)
from replica_lab.runtime.compilation import (
    CandidateIntegrityError, CompilationBinding, CompiledProcedure,
    InterruptedTransactionError, JOURNAL_FORMAT, ManifestIntegrityError,
    RecoveryError, StaleSourceError, TransactionError, TransactionJournal,
    WorldInventory, _transaction_record,
    catalog_path_for, journal_path_for, load_catalog, load_manifest,
    recover_pending_transaction)
from replica_lab.runtime.learning_proposal import LearningProposal
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
    # Stable observed text only: raw receipts embed absolute working directories,
    # which would make source hashes (and therefore replay) root-dependent.
    stable_failure = {'exit_code': failure.get('exit_code'),
                      'status': failure.get('status')}
    stable_repair = {'receipts': len(run.receipts),
                     'final_exit_code': run.final_exit_code,
                     'status': run.status}
    records = (
        MemoryRecord(record_id='train-failure', subject_id='subject', branch_id='train',
                     text=json.dumps(stable_failure, sort_keys=True),
                     observed_at=1, event_at=1,
                     purpose='m1', kind='tool', authenticity='observed_tool'),
        MemoryRecord(record_id='train-repair', subject_id='subject', branch_id='train',
                     text=json.dumps(stable_repair, sort_keys=True),
                     observed_at=2, event_at=2,
                     purpose='m1', kind='tool', authenticity='observed_tool'),
    )
    return procedure, records


def _proposal(parent=None, *, sources=('train-failure',), digest_plan=None):
    return LearningProposal(
        'proposal-m1', parent, sources, 'procedure', 'repair-config',
        {'fixture': 'e3_m0a_menus'}, {'cases': 2, 'tool_calls': 4},
        'host_tools', digest_plan or 'held-out fixture completion; no neural update')


def _compiled(tmp_path, *, parent_manifest_id=None, parent_manifest_digest=None,
              proposal=None, subdir='compiled', manifests_dir=None):
    base = tmp_path / subdir
    base.mkdir(parents=True, exist_ok=True)
    tools = _workspace(base, 'm0a_train_queue', 'source')
    procedure, records = _records(tools)
    binding = CompilationBinding('subject', 'train', 'm1', 2, 0,
                                 'minicpm5-test', 'generic_procedure',
                                 parent_manifest_id, parent_manifest_digest)
    artifacts = ArtifactRegistry()
    compiled = CompiledProcedure.compile(
        procedure, binding, records, artifacts,
        proposal=proposal if proposal is not None else _proposal(parent_manifest_id),
        authorized_target='repair-config')
    manifests = ManifestRegistry(manifests_dir if manifests_dir is not None
                                 else base / 'manifests')
    return compiled, artifacts, manifests, records


def _workspaces(tmp_path, tag='a'):
    return [
        ('m0a_dev_retry', 'dev',
         _workspace(tmp_path, 'm0a_dev_retry', f'dev-{tag}')),
        ('m0a_heldout_poll', 'heldout',
         _workspace(tmp_path, 'm0a_heldout_poll', f'heldout-{tag}')),
    ]


def _inventory(workspaces, *, family='verify.m0a.v1', split='verification',
               tweak=None):
    cases = []
    for case_id, role, tools in workspaces:
        definition = fingerprint({'case_id': case_id, 'role': role})
        if tweak is not None and case_id == 'm0a_dev_retry':
            definition = fingerprint({'case_id': case_id, 'role': role, **tweak})
        cases.append({'case_id': case_id, 'role': role,
                      'definition_digest': definition,
                      'workspace_digest': fingerprint(dict(tools.workspace_hashes()))})
    return WorldInventory.build(split, family, cases)


def _evaluate(compiled, artifacts, tmp_path, *, inventory=None, tag='a'):
    workspaces = _workspaces(tmp_path, tag)
    if inventory is None:
        inventory = _inventory(workspaces)
    return compiled.evaluate(artifacts, workspaces, inventory=inventory)


def _hashes(records):
    return {record.record_id: record.record_hash for record in records}


def _publish(compiled, artifacts, manifests, tmp_path, records, *,
             evaluation=None, inventory=None, catalog_path=None):
    if evaluation is None:
        evaluation = _evaluate(compiled, artifacts, tmp_path, inventory=inventory)
    return compiled.publish(artifacts, manifests, evaluation, _hashes(records),
                            world_inventory=inventory, catalog_path=catalog_path)


def test_compile_execute_publish_restart_and_rollback(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    assert evaluation.accepted is True
    assert all(case['independent_passed'] for case in evaluation.cases)
    manifest = _publish(compiled, artifacts, manifests, tmp_path, records,
                        evaluation=evaluation)
    assert manifest['parent'] is None
    assert manifest['payload']['schema'] == 'replica.procedure.manifest.v2'
    assert manifest['payload']['artifact']['status'] == 'validated'
    assert manifest['payload']['proposal']['digest'] == compiled.proposal_digest
    assert manifest['payload']['worlds']['digest'] == evaluation.worlds_digest
    assert manifest['payload']['identity']['digest'] == \
        compiled.binding.identity_digest
    assert artifacts.get(compiled.artifact.artifact_id).status == 'accepted'
    publication = artifacts.publication(compiled.artifact.artifact_id)
    assert publication['manifest_id'] == manifest['manifest_id']
    assert publication['active'] is True
    assert not journal_path_for(manifests).exists()

    # A fresh process reads the same manifest and reconstructs the executable artifact.
    restored_manifest = load_manifest(manifests, manifest['manifest_id'])
    reloaded = ArtifactRegistry()
    restored = CompiledProcedure.from_manifest(restored_manifest, reloaded)
    assert restored.legacy is False
    assert restored.proposal_digest == compiled.proposal_digest
    assert restored.binding.parent_manifest_digest is None
    procedure = RepairProcedure(restored.artifact.name, 'workspace/app_settings.py',
                                restored.artifact.source_event_ids)
    run = procedure.execute(_workspace(tmp_path, 'm0a_heldout_heartbeat', 'restart'))
    assert run.status == 'succeeded'
    assert run.final_exit_code == 0
    # Replaying the reload into a catalog binds the published artifact.
    assert reloaded.get(restored.artifact.artifact_id) is not None
    assert reloaded.assert_consistent() is None

    # Rollback needs the bound parent; a root publication is not reversible here.
    with pytest.raises(PermissionError, match='bound parent'):
        compiled.rollback(artifacts, manifests, 'not-an-ancestor')


def test_source_change_before_publication_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    changed = _hashes(records)
    changed['train-failure'] = hashlib.sha256(b'changed evidence').hexdigest()
    with pytest.raises(StaleSourceError):
        compiled.publish(artifacts, manifests, evaluation, changed)
    assert manifests.active() is None
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'
    assert not journal_path_for(manifests).exists()


def test_stale_active_parent_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    parent_payload = {'base_revision': 'base', 'identity_version': 'parent'}
    manifests.publish(parent_payload, {
        'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
        'parent_manifest_id': None}, expected_parent=None)
    with pytest.raises(ValueError, match='active parent changed'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'


def test_proposal_content_is_bound_not_only_its_id(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)

    # A proposal with different content at compile time is a different candidate
    # and produces a different payload digest.
    other, other_artifacts, _, _ = _compiled(
        tmp_path, subdir='other-proposal',
        proposal=_proposal(None, digest_plan='different plan'))
    other_evaluation = _evaluate(other, other_artifacts, tmp_path, tag='b')
    assert other_evaluation.candidate_digest != evaluation.candidate_digest

    # In-place replacement of the recorded proposal is rejected before publish.
    forged = dict(compiled.proposal)
    forged['evaluation_plan'] = 'forged plan'
    object.__setattr__(compiled, 'proposal', forged)
    with pytest.raises(ValueError, match='does not match its digest'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'


def test_proposal_sources_must_be_bound_evidence(tmp_path):
    with pytest.raises(CandidateIntegrityError, match='proposal sources'):
        _compiled(tmp_path, proposal=_proposal(None, sources=('not-observed',)))


def test_verification_world_inventory_tampering_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    workspaces = _workspaces(tmp_path)
    inventory = _inventory(workspaces)
    evaluation = compiled.evaluate(artifacts, workspaces, inventory=inventory)

    tampered = _inventory(_workspaces(tmp_path, 'b'), tweak={'constants': 'changed'})
    assert tampered.digest != inventory.digest
    with pytest.raises(CandidateIntegrityError, match='world inventory'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records),
                         world_inventory=tampered)
    assert manifests.active() is None
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'

    # Tampering with the recorded inventory inside the evaluation is also caught.
    evaluation.world_inventory['cases'][0]['definition_digest'] = '0' * 64
    with pytest.raises(CandidateIntegrityError, match='world inventory'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records),
                         world_inventory=inventory)
    assert manifests.active() is None


def test_executable_payload_tampering_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    artifact_id = compiled.artifact.artifact_id
    tampered = replace(compiled.artifact, status='validated',
                       tests=('the evaluator passes a different rule',))
    assert artifact_content_digest(tampered) != artifact_content_digest(
        compiled.artifact)
    # Replace the catalog payload after verification, as a hostile writer would.
    artifacts._items[artifact_id] = tampered
    with pytest.raises(CandidateIntegrityError, match='payload changed'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None


def test_identity_snapshot_and_lineage_tampering_is_rejected(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)

    tampered = replace(compiled, binding=replace(
        compiled.binding, identity_version='hostile.identity'))
    with pytest.raises(CandidateIntegrityError, match='does not match the compiled payload'):
        tampered.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None

    # Lineage: a parent digest that does not match the archived parent body.
    parent_payload = {'base_revision': 'base', 'identity_version': 'parent'}
    parent = manifests.publish(parent_payload, {
        'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
        'parent_manifest_id': None}, expected_parent=None)
    child, child_artifacts, child_manifests, child_records = _compiled(
        tmp_path, subdir='lineage', manifests_dir=manifests.root,
        parent_manifest_id=parent['manifest_id'],
        parent_manifest_digest='0' * 64)
    child_evaluation = _evaluate(child, child_artifacts, tmp_path, tag='b')
    with pytest.raises(CandidateIntegrityError, match='lineage digest'):
        child.publish(child_artifacts, child_manifests, child_evaluation,
                      _hashes(child_records))
    assert child_manifests.active()['manifest_id'] == parent['manifest_id']
    assert child_artifacts.get(child.artifact.artifact_id).status == 'validated'


def test_interrupted_publication_preserves_active_and_allows_retry(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    import replica_lab.learning as learning
    real_replace = learning.os.replace
    calls = {'count': 0}

    def fail_second(source, target):
        calls['count'] += 1
        if calls['count'] == 3:
            raise OSError('simulated interruption before active publication')
        real_replace(source, target)

    # 1: durable pre-state catalog, 2: journal, 3: manifest archive.
    monkeypatch.setattr(learning.os, 'replace', fail_second)
    with pytest.raises(OSError, match='simulated interruption'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'
    assert not journal_path_for(manifests).exists()
    monkeypatch.setattr(learning.os, 'replace', real_replace)
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active()['manifest_id'] == manifest['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'accepted'


def test_manifest_failure_leaves_no_accepted_but_unpublished_state(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)

    def explode(self, payload, evaluation, *, expected_parent=None):
        raise RuntimeError('registry refused publication')

    monkeypatch.setattr(ManifestRegistry, 'publish', explode)
    with pytest.raises(RuntimeError, match='refused publication'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None
    assert artifacts.get(compiled.artifact.artifact_id).status == 'validated'
    assert artifacts.publication(compiled.artifact.artifact_id) is None
    assert not journal_path_for(manifests).exists()
    artifacts.assert_consistent()
    monkeypatch.undo()
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active()['manifest_id'] == manifest['manifest_id']


def test_catalog_failure_after_manifest_commit_requires_recovery(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    catalog_path = catalog_path_for(manifests)
    real_save = ArtifactRegistry.save
    calls = {'count': 0}

    def fail_commit(self, path):
        calls['count'] += 1
        if calls['count'] == 2:  # pre-state write succeeds, commit fails
            raise OSError('simulated catalog failure')
        return real_save(self, path)

    monkeypatch.setattr(ArtifactRegistry, 'save', fail_commit)
    with pytest.raises(InterruptedTransactionError, match='recover_pending_transaction'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is not None
    journal = TransactionJournal(journal_path_for(manifests))
    record = journal.read()
    assert record['operation'] == 'publish'
    assert record['phase'] == 'prepared'
    # The durable pre-state catalog still carries the validated artifact.
    on_disk = ArtifactRegistry.load(catalog_path)
    assert on_disk.get(compiled.artifact.artifact_id).status == 'validated'
    assert on_disk.publication(compiled.artifact.artifact_id) is None

    monkeypatch.undo()
    # A new process replays the durable journal explicitly.
    recovered_catalog = ArtifactRegistry.load(catalog_path)
    outcome = recover_pending_transaction(recovered_catalog, manifests, journal,
                                          catalog_path=catalog_path)
    assert outcome['outcome'] == 'completed_publish'
    assert journal.read() is None
    assert recovered_catalog.get(compiled.artifact.artifact_id).status == 'accepted'
    publication = recovered_catalog.publication(compiled.artifact.artifact_id)
    assert publication['manifest_id'] == manifests.active()['manifest_id']
    assert publication['active'] is True
    reloaded = ArtifactRegistry.load(catalog_path)
    assert reloaded.assert_consistent() is None
    assert reloaded.get(compiled.artifact.artifact_id).status == 'accepted'


def test_manifest_commit_despite_error_is_recovered_not_discarded(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    real_publish = ManifestRegistry.publish

    def publish_then_fail(self, payload, evaluation, *, expected_parent=None):
        real_publish(self, payload, evaluation, expected_parent=expected_parent)
        raise OSError('simulated crash after active publication')

    monkeypatch.setattr(ManifestRegistry, 'publish', publish_then_fail)
    with pytest.raises(InterruptedTransactionError, match='recover_pending_transaction'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    # The registry moved even though the call raised, so the journal is kept.
    journal = TransactionJournal(journal_path_for(manifests))
    assert journal.read()['operation'] == 'publish'
    monkeypatch.undo()
    recovered = ArtifactRegistry.load(catalog_path_for(manifests))
    outcome = recover_pending_transaction(recovered, manifests, journal)
    assert outcome['outcome'] == 'completed_publish'
    assert recovered.get(compiled.artifact.artifact_id).status == 'accepted'
    assert recovered.publication(compiled.artifact.artifact_id)['active'] is True
    assert journal.read() is None


def _published_child(tmp_path):
    parent_payload = {'base_revision': 'base', 'identity_version': 'parent'}
    parent_manifests = ManifestRegistry(tmp_path / 'child/manifests')
    parent = parent_manifests.publish(parent_payload, {
        'decision': 'accept', 'candidate_digest': fingerprint(parent_payload),
        'parent_manifest_id': None}, expected_parent=None)
    compiled, artifacts, manifests, records = _compiled(
        tmp_path, subdir='child', parent_manifest_id=parent['manifest_id'],
        parent_manifest_digest=fingerprint(parent))
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    manifest = compiled.publish(artifacts, manifests, evaluation,
                                _hashes(records))
    assert manifest['parent'] == parent['manifest_id']
    return compiled, artifacts, manifests, records, parent, manifest


def test_rollback_is_coordinated_and_retryable(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)

    def fail_rollback(self, manifest_id):
        raise OSError('simulated rollback interruption')

    monkeypatch.setattr(ManifestRegistry, 'rollback', fail_rollback)
    with pytest.raises(OSError, match='simulated rollback'):
        compiled.rollback(artifacts, manifests, parent['manifest_id'])
    # No half-applied state: active is still the publication and the artifact is
    # accepted again; the journal is cleaned up.
    assert manifests.active()['manifest_id'] == manifest['manifest_id']
    artifact = artifacts.get(compiled.artifact.artifact_id)
    assert artifact.status == 'accepted'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is True
    assert not journal_path_for(manifests).exists()
    assert ArtifactRegistry.load(catalog_path_for(manifests)).get(
        compiled.artifact.artifact_id).status == 'accepted'

    monkeypatch.undo()
    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is False
    assert not journal_path_for(manifests).exists()


def test_rollback_completes_when_manifest_rollback_raises_after_committing(
        tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    real_rollback = ManifestRegistry.rollback

    def rollback_then_fail(self, manifest_id):
        real_rollback(self, manifest_id)
        raise OSError('simulated failure after active rewrite')

    monkeypatch.setattr(ManifestRegistry, 'rollback', rollback_then_fail)
    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert manifests.active()['manifest_id'] == parent['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
    assert not journal_path_for(manifests).exists()


def test_restart_recovery_restores_pre_rollback_state_after_crash(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    # Simulate a crash immediately after the catalog was revoked but before the
    # manifest registry moved: rebuild the exact durable transaction record.
    snapshot = artifacts.snapshot()
    record = _transaction_record(
        'rollback', snapshot, compiled.artifact.artifact_id,
        artifact_content_digest(artifacts.get(compiled.artifact.artifact_id)),
        'accepted', 'revoked',
        {'active_before': manifest['manifest_id'],
         'expected_manifest_id': parent['manifest_id'],
         'published_manifest_id': manifest['manifest_id'],
         'payload_digest': fingerprint(manifest['payload'])})
    journal = TransactionJournal(journal_path_for(manifests))
    journal.write(record)
    artifacts.transition(compiled.artifact.artifact_id, 'revoked',
                         reason='manifest rollback')
    artifacts.save(catalog_path_for(manifests))

    # Fresh process: the catalog says revoked, active still publishes it.
    recovered = ArtifactRegistry.load(catalog_path_for(manifests))
    outcome = recover_pending_transaction(recovered, manifests, journal)
    assert outcome['outcome'] == 'aborted_rollback'
    assert manifest['manifest_id'] == manifests.active()['manifest_id']
    artifact = recovered.get(compiled.artifact.artifact_id)
    assert artifact.status == 'accepted'
    assert recovered.publication(compiled.artifact.artifact_id)['active'] is True
    assert journal.read() is None


def test_committed_journal_leftover_is_verified_then_cleared(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    pre_state = artifacts.snapshot()
    manifest = _publish(compiled, artifacts, manifests, tmp_path, records,
                        evaluation=evaluation)
    artifact_id = compiled.artifact.artifact_id
    body = {
        'format': JOURNAL_FORMAT, 'operation': 'publish', 'phase': 'committed',
        'catalog_before': pre_state,
        'artifact': {'artifact_id': artifact_id,
                     'content_digest': artifact_content_digest(
                         artifacts.get(artifact_id)),
                     'from_status': 'validated', 'to_status': 'accepted'},
        'manifest': {'active_before': None,
                     'expected_manifest_id': manifest['manifest_id'],
                     'published_manifest_id': manifest['manifest_id'],
                     'payload_digest': fingerprint(manifest['payload'])},
    }
    record = {'transaction_id': fingerprint(body), **body}
    journal = TransactionJournal(journal_path_for(manifests))
    journal.write(record)
    outcome = recover_pending_transaction(
        ArtifactRegistry.load(catalog_path_for(manifests)), manifests, journal)
    assert outcome['outcome'] == 'cleared_committed'
    assert outcome['artifact_status'] == 'accepted'
    assert journal.read() is None

    wrong = json.loads(json.dumps(record))
    wrong['manifest']['expected_manifest_id'] = '2' * 64
    wrong['transaction_id'] = fingerprint(
        {key: wrong[key] for key in wrong if key != 'transaction_id'})
    journal.path.write_text(json.dumps(wrong))
    with pytest.raises(RecoveryError, match='does not match the active'):
        recover_pending_transaction(
            ArtifactRegistry.load(catalog_path_for(manifests)), manifests, journal)


def test_interrupted_publish_is_not_silently_discarded(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    journal = TransactionJournal(journal_path_for(manifests))
    journal.write(_transaction_record(
        'publish', artifacts.snapshot(), compiled.artifact.artifact_id,
        artifact_content_digest(compiled.artifact), 'validated', 'accepted',
        {'active_before': None, 'expected_manifest_id': '1' * 64,
         'published_manifest_id': '1' * 64,
         'payload_digest': fingerprint({'stale': True})}))
    with pytest.raises(TransactionError, match='recover_pending_transaction'):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active() is None
    assert journal.read() is not None
    # Aborting recovery restores the pre-transaction catalog and clears the slot.
    outcome = recover_pending_transaction(artifacts, manifests, journal)
    assert outcome['outcome'] == 'aborted_publish'
    assert journal.read() is None
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
    publication = artifacts.publication(compiled.artifact.artifact_id)
    assert publication['active'] is False
    with pytest.raises(PermissionError):
        compiled.publish(artifacts, manifests, evaluation, _hashes(records))
    assert manifests.active()['manifest_id'] == manifest['manifest_id']


def test_rollback_to_bound_ancestor_revokes_compiled_artifact(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    assert manifest['parent'] == parent['manifest_id']
    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is False


def test_publication_is_deterministic_across_fresh_roots(tmp_path):
    first_compiled, first_artifacts, first_manifests, first_records = _compiled(
        tmp_path, subdir='replay-a')
    first = _publish(first_compiled, first_artifacts, first_manifests,
                     tmp_path, first_records,
                     evaluation=_evaluate(first_compiled, first_artifacts,
                                          tmp_path, tag='a'))
    second_compiled, second_artifacts, second_manifests, second_records = _compiled(
        tmp_path, subdir='replay-b')
    second = _publish(second_compiled, second_artifacts, second_manifests,
                      tmp_path, second_records,
                      evaluation=_evaluate(second_compiled, second_artifacts,
                                           tmp_path, tag='b'))
    assert first['manifest_id'] == second['manifest_id']
    assert first['payload'] == second['payload']
    assert first['evaluation'] == second['evaluation']


def test_legacy_manifest_requires_explicit_validated_migration(tmp_path):
    compiled, artifacts, _, records = _compiled(tmp_path)
    artifact = replace(compiled.artifact, status='accepted')
    legacy_artifact = artifact_to_dict(artifact)
    legacy_artifact.pop('kind')
    payload = {
        'base_revision': compiled.binding.base_revision,
        'identity_version': compiled.binding.identity_version,
        'owner': compiled.binding.owner(),
        'parent_manifest_id': None,
        'source_hashes': dict(compiled.source_hashes),
        'artifact': legacy_artifact,
    }
    evaluation = {
        'decision': 'accept',
        'candidate_digest': fingerprint(payload),
        'parent_manifest_id': None,
        'artifact_digest': artifact.digest,
        'cases': [],
        'evaluation_scope': 'independent_fixture_execution',
    }
    manifest = {'payload': payload, 'parent': None, 'evaluation': evaluation}
    manifest['manifest_id'] = fingerprint(
        {'payload': payload, 'parent': None, 'evaluation': evaluation})

    registry = ArtifactRegistry()
    restored = CompiledProcedure.from_manifest(manifest, registry)
    assert restored.legacy is True
    assert restored.artifact.status == 'accepted'
    assert restored.binding.parent_manifest_digest is None
    assert restored.proposal is None
    assert registry.publication(restored.artifact.artifact_id)['imported'] is True
    assert registry.assert_consistent() is None

    # Missing or additional fields are schema errors, never silently accepted.
    with pytest.raises(ManifestIntegrityError, match='schema mismatch'):
        broken = {**payload, 'extra': True}
        broker = {'payload': broken, 'parent': None, 'evaluation': evaluation}
        broker['manifest_id'] = fingerprint(
            {'payload': broken, 'parent': None, 'evaluation': evaluation})
        CompiledProcedure.from_manifest(broker, ArtifactRegistry())
    with pytest.raises(ManifestIntegrityError, match='schema mismatch'):
        broken = {key: value for key, value in payload.items() if key != 'owner'}
        broker = {'payload': broken, 'parent': None, 'evaluation': evaluation}
        broker['manifest_id'] = fingerprint(
            {'payload': broken, 'parent': None, 'evaluation': evaluation})
        CompiledProcedure.from_manifest(broker, ArtifactRegistry())
    with pytest.raises(ManifestIntegrityError, match='legacy manifest rejected'):
        CompiledProcedure.from_manifest(manifest, ArtifactRegistry(),
                                        allow_legacy=False)

    # Any bit flip in the payload invalidates the manifest body digest.
    tampered = json.loads(json.dumps(manifest))
    tampered['payload']['owner']['subject_id'] = 'hostile-subject'
    with pytest.raises(ManifestIntegrityError, match='manifest ID'):
        CompiledProcedure.from_manifest(tampered, ArtifactRegistry())


def test_manifest_payload_digest_tampering_is_rejected_on_reload(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    manifest = compiled.publish(artifacts, manifests, evaluation, _hashes(records))

    broken = json.loads(json.dumps(manifest))
    broken['payload']['identity']['subject_id'] = 'hostile-subject'
    with pytest.raises(ManifestIntegrityError, match='manifest ID'):
        CompiledProcedure.from_manifest(broken, ArtifactRegistry())

    rebuilt = json.loads(json.dumps(manifest))
    rebuilt['evaluation']['candidate_digest'] = '0' * 64
    rebuilt['manifest_id'] = fingerprint(
        {'payload': rebuilt['payload'], 'parent': rebuilt['parent'],
         'evaluation': rebuilt['evaluation']})
    with pytest.raises(ManifestIntegrityError, match='payload bytes'):
        CompiledProcedure.from_manifest(rebuilt, ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    broken['payload']['worlds']['cases'][0]['role'] = 'hostile'
    with pytest.raises(ManifestIntegrityError):
        CompiledProcedure.from_manifest(broken, ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    broken['payload']['artifact']['content_digest'] = '0' * 64
    with pytest.raises(ManifestIntegrityError):
        CompiledProcedure.from_manifest(broken, ArtifactRegistry())


def test_journal_schema_is_validated_strictly(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    journal = TransactionJournal(journal_path_for(manifests))
    record = _transaction_record(
        'publish', artifacts.snapshot(), compiled.artifact.artifact_id,
        artifact_content_digest(compiled.artifact), 'proposed', 'accepted',
        {'active_before': None, 'expected_manifest_id': '1' * 64,
         'published_manifest_id': '1' * 64, 'payload_digest': None})
    broken = {**record, 'manifest': {**record['manifest'],
                                     'payload_digest': '2' * 64}}
    with pytest.raises(RecoveryError, match='differs from its ID'):
        journal.write(broken)
    journal.write(record)
    on_disk = json.loads(journal.path.read_text())
    on_disk['artifact']['to_status'] = 'revoked'
    journal.path.write_text(json.dumps(on_disk))
    with pytest.raises(RecoveryError, match='differs from its ID'):
        journal.read()


def test_procedure_artifact_is_reconstructed_with_its_content_digest():
    artifact = ProcedureArtifact(
        'artifact-x', 'repair-x', 1, ('pre',),
        (ProcedureStep('read_file', {'path': 'workspace/app_settings.py'}),),
        ('test',), ('event-1',), {'runtime': 'v1'}, status='validated')
    record = artifact_to_dict(artifact, content_digest=True)
    assert record['content_digest'] == artifact_content_digest(artifact)
    broken = dict(record, content_digest='0' * 64)
    with pytest.raises(ValueError, match='content digest'):
        artifact_from_dict(broken, content_digest=True)


# ---------------------------------------------------------------------------
# review regressions: invalidation rollback, workspace binding, reload,
# publication authentication, uncertain catalog commits, recovery persistence
# ---------------------------------------------------------------------------
def _prepared_rollback(tmp_path, compiled, artifacts, manifests, records, parent,
                       manifest):
    snapshot = artifacts.snapshot()
    record = _transaction_record(
        'rollback', snapshot, compiled.artifact.artifact_id,
        artifact_content_digest(artifacts.get(compiled.artifact.artifact_id)),
        artifacts.get(compiled.artifact.artifact_id).status, 'revoked',
        {'active_before': manifest['manifest_id'],
         'expected_manifest_id': parent['manifest_id'],
         'published_manifest_id': manifest['manifest_id'],
         'payload_digest': fingerprint(manifest['payload'])})
    journal = TransactionJournal(journal_path_for(manifests))
    journal.write(record)
    return snapshot, journal


def test_source_invalidated_publication_can_still_roll_back(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    artifacts.invalidate_source('train-failure', reason='source revoked')
    artifact = artifacts.get(compiled.artifact.artifact_id)
    assert artifact.status == 'revoked'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is False

    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert manifests.active()['manifest_id'] == parent['manifest_id']
    artifact = artifacts.get(compiled.artifact.artifact_id)
    assert artifact.status == 'revoked'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is False
    assert not journal_path_for(manifests).exists()
    durable = ArtifactRegistry.load(catalog_path_for(manifests))
    assert durable.get(compiled.artifact.artifact_id).status == 'revoked'
    assert durable.publication(compiled.artifact.artifact_id)['active'] is False


def test_supplied_workspace_digest_must_match_execution(tmp_path):
    compiled, artifacts, _, _ = _compiled(tmp_path)
    workspaces = _workspaces(tmp_path, 'a')
    honest = _inventory(workspaces)
    forged_cases = []
    for case in honest.cases:
        forged_cases.append({
            'case_id': case.case_id, 'role': case.role,
            'definition_digest': case.definition_digest,
            'workspace_digest': '0' * 64})
    forged = WorldInventory.build(honest.split, honest.family_id, forged_cases)
    with pytest.raises(CandidateIntegrityError, match='executed workspace'):
        compiled.evaluate(artifacts, workspaces, inventory=forged)
    assert artifacts.get(compiled.artifact.artifact_id).status == 'proposed'


def test_active_manifest_reload_restores_acceptance_and_rollback(tmp_path):
    original, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    archived = load_manifest(manifests, manifest['manifest_id'])
    reloaded = ArtifactRegistry()
    restored = CompiledProcedure.from_manifest(archived, reloaded)
    assert restored.artifact.status == 'accepted'
    assert restored.proposal_digest == original.proposal_digest
    publication = reloaded.publication(restored.artifact.artifact_id)
    assert publication['manifest_id'] == manifest['manifest_id']
    assert publication['active'] is True
    assert publication['imported'] is True

    # The reloaded revision can be rolled back to its bound parent.
    rolled = restored.rollback(reloaded, manifests, parent['manifest_id'])
    assert rolled['manifest_id'] == parent['manifest_id']
    assert reloaded.get(restored.artifact.artifact_id).status == 'revoked'
    assert reloaded.publication(restored.artifact.artifact_id)['active'] is False


def test_reload_rejects_a_revoked_publication_flip(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    archived = load_manifest(manifests, manifest['manifest_id'])
    compiled.rollback(artifacts, manifests, parent['manifest_id'])
    with pytest.raises(ManifestIntegrityError, match='not currently publishable'):
        CompiledProcedure.from_manifest(archived, artifacts)


def test_catalog_publication_flips_and_duplicates_are_rejected(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    catalog_path = catalog_path_for(manifests)
    load_catalog(catalog_path, manifests)  # authenticates cleanly

    def rewritten(change):
        value = json.loads(json.dumps(ArtifactRegistry.load(catalog_path).export()))
        change(value)
        return value

    duplicate = rewritten(lambda value: value['publications'].append(
        dict(value['publications'][0])))
    with pytest.raises(ValueError, match='duplicate'):
        ArtifactRegistry.import_state(duplicate)

    flipped = rewritten(
        lambda value: value['publications'][0].__setitem__('payload_digest',
                                                           '0' * 64))
    catalog_path.write_text(json.dumps(flipped))
    with pytest.raises(ManifestIntegrityError, match='payload digest'):
        load_catalog(catalog_path, manifests)

    wrong_manifest = rewritten(
        lambda value: value['publications'][0].__setitem__('manifest_id',
                                                           '1' * 64))
    catalog_path.write_text(json.dumps(wrong_manifest))
    with pytest.raises(ManifestIntegrityError):
        load_catalog(catalog_path, manifests)

    # Restore the genuine catalog for subsequent use by this test.
    ArtifactRegistry.load(catalog_path)
    artifacts.save(catalog_path)
    assert load_catalog(catalog_path, manifests).get(
        compiled.artifact.artifact_id).status == 'accepted'


def test_rollback_survives_fsync_after_replace(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    real_save = ArtifactRegistry.save
    calls = {'count': 0}

    def save_then_fail(self, path):
        calls['count'] += 1
        result = real_save(self, path)
        if calls['count'] == 2:  # the mutation write landed, fsync raised
            raise OSError('simulated fsync-after-replace')
        return result

    monkeypatch.setattr(ArtifactRegistry, 'save', save_then_fail)
    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']
    assert manifests.active()['manifest_id'] == parent['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'revoked'
    assert not journal_path_for(manifests).exists()


def test_rollback_catalog_failure_before_replace_compensates(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    real_save = ArtifactRegistry.save
    calls = {'count': 0}

    def fail_before(self, path):
        calls['count'] += 1
        if calls['count'] == 2:
            raise OSError('simulated pre-replace failure')
        return real_save(self, path)

    monkeypatch.setattr(ArtifactRegistry, 'save', fail_before)
    with pytest.raises(OSError, match='pre-replace'):
        compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert manifests.active()['manifest_id'] == manifest['manifest_id']
    artifact = artifacts.get(compiled.artifact.artifact_id)
    assert artifact.status == 'accepted'
    assert artifacts.publication(compiled.artifact.artifact_id)['active'] is True
    assert not journal_path_for(manifests).exists()
    assert ArtifactRegistry.load(catalog_path_for(manifests)).get(
        compiled.artifact.artifact_id).status == 'accepted'

    monkeypatch.undo()
    restored = compiled.rollback(artifacts, manifests, parent['manifest_id'])
    assert restored['manifest_id'] == parent['manifest_id']


def test_rollback_unknown_catalog_outcome_keeps_the_journal(tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    real_save = ArtifactRegistry.save
    calls = {'count': 0}

    def corrupt_then_fail(self, path):
        calls['count'] += 1
        if calls['count'] == 2:
            Path(path).write_text('{not a catalog', encoding='utf-8')
            raise OSError('simulated unknown catalog outcome')
        return real_save(self, path)

    monkeypatch.setattr(ArtifactRegistry, 'save', corrupt_then_fail)
    with pytest.raises(InterruptedTransactionError,
                       match='catalog commit outcome is unknown'):
        compiled.rollback(artifacts, manifests, parent['manifest_id'])
    journal = TransactionJournal(journal_path_for(manifests))
    assert journal.read() is not None
    assert manifests.active()['manifest_id'] == manifest['manifest_id']
    assert artifacts.get(compiled.artifact.artifact_id).status == 'accepted'

    monkeypatch.undo()
    outcome = recover_pending_transaction(artifacts, manifests, journal)
    assert outcome['outcome'] == 'aborted_rollback'
    assert journal.read() is None
    reloaded = ArtifactRegistry.load(catalog_path_for(manifests))
    assert reloaded.get(compiled.artifact.artifact_id).status == 'accepted'
    assert reloaded.publication(compiled.artifact.artifact_id)['active'] is True


def test_aborted_publish_persists_catalog_before(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    extra = ContextPack('ctx-extra', ('src-1',), 'subject-1', 'branch-a', 'test',
                        10, 'text', hashlib.sha256(b'text').hexdigest(),
                        {'runtime': 'v1'})
    artifacts.add(extra)
    before = artifacts.snapshot()
    journal = TransactionJournal(journal_path_for(manifests))
    journal.write(_transaction_record(
        'publish', before, compiled.artifact.artifact_id,
        artifact_content_digest(compiled.artifact), 'proposed', 'validated',
        {'active_before': None, 'expected_manifest_id': '1' * 64,
         'published_manifest_id': '1' * 64, 'payload_digest': None}))
    # The durable file diverges from the journal's pre-state.
    divergent = ArtifactRegistry()
    divergent.add(compiled.artifact)
    divergent.save(catalog_path_for(manifests))
    outcome = recover_pending_transaction(artifacts, manifests, journal)
    assert outcome['outcome'] == 'aborted_publish'
    assert journal.read() is None
    assert ArtifactRegistry.load(catalog_path_for(manifests)).export() == before


def test_recovery_keeps_journal_when_active_manifest_is_unreadable(tmp_path):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)
    snapshot, journal = _prepared_rollback(
        tmp_path, compiled, artifacts, manifests, records, parent, manifest)
    artifacts.transition(compiled.artifact.artifact_id, 'revoked',
                         reason='manifest rollback')
    artifacts.save(catalog_path_for(manifests))
    active_path = Path(manifests.root) / 'active.json'
    original = active_path.read_text()
    active_path.write_text('{corrupt active')
    with pytest.raises(RecoveryError, match='active manifest is unreadable'):
        recover_pending_transaction(artifacts, manifests, journal)
    assert journal.read() is not None
    active_path.write_text(original)
    outcome = recover_pending_transaction(artifacts, manifests, journal)
    assert outcome['outcome'] == 'aborted_rollback'
    assert journal.read() is None


def test_compiler_rejects_proposals_for_other_targets(tmp_path):
    tools = _workspace(tmp_path, 'm0a_train_queue', 'target')
    procedure, records = _records(tools)
    binding = CompilationBinding('subject', 'train', 'm1', 2, 0,
                                 'minicpm5-test', 'generic_procedure')
    record_proposal = LearningProposal(
        'proposal-record', None, ('train-failure',), 'record',
        'memory-selection', {'fixture': 'x'}, {'cases': 1}, 'none', 'plan')
    with pytest.raises(CandidateIntegrityError, match='procedure target'):
        CompiledProcedure.compile(procedure, binding, records,
                                  ArtifactRegistry(), proposal=record_proposal)
    wrong_target = LearningProposal(
        'proposal-other', None, ('train-failure',), 'procedure',
        'memory-selection', {'fixture': 'x'}, {'cases': 1}, 'host_tools', 'plan')
    with pytest.raises(CandidateIntegrityError, match='host-authorized'):
        CompiledProcedure.compile(procedure, binding, records,
                                  ArtifactRegistry(), proposal=wrong_target,
                                  authorized_target='repair-config')
    with pytest.raises(CandidateIntegrityError, match='requires the host-authorized'):
        CompiledProcedure.compile(procedure, binding, records,
                                  ArtifactRegistry(),
                                  proposal=_proposal(None))


def test_v2_manifest_requires_complete_case_evidence_and_artifact_kind(tmp_path):
    compiled, artifacts, manifests, records = _compiled(tmp_path)
    evaluation = _evaluate(compiled, artifacts, tmp_path)
    manifest = _publish(compiled, artifacts, manifests, tmp_path, records,
                        evaluation=evaluation)

    def reidentify(value):
        value['manifest_id'] = fingerprint({'payload': value['payload'],
                                            'parent': value['parent'],
                                            'evaluation': value['evaluation']})
        return value

    broken = json.loads(json.dumps(manifest))
    del broken['evaluation']['cases'][0]['error']
    with pytest.raises(ManifestIntegrityError, match='case schema'):
        CompiledProcedure.from_manifest(reidentify(broken), ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    broken['evaluation']['cases'][0]['last_receipt']['extra'] = True
    with pytest.raises(ManifestIntegrityError, match='receipt schema'):
        CompiledProcedure.from_manifest(reidentify(broken), ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    broken['evaluation']['cases'][0]['last_receipt'] = None
    with pytest.raises(ManifestIntegrityError, match='receipt evidence'):
        CompiledProcedure.from_manifest(reidentify(broken), ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    del broken['payload']['artifact']['kind']
    with pytest.raises(ManifestIntegrityError, match='kind is required'):
        CompiledProcedure.from_manifest(reidentify(broken), ArtifactRegistry())

    broken = json.loads(json.dumps(manifest))
    broken['payload']['artifact']['kind'] = 'neural_candidate'
    with pytest.raises(ManifestIntegrityError):
        CompiledProcedure.from_manifest(reidentify(broken), ArtifactRegistry())


def test_rollback_keeps_journal_when_manifest_side_becomes_unreadable(
        tmp_path, monkeypatch):
    compiled, artifacts, manifests, records, parent, manifest = \
        _published_child(tmp_path)

    def corrupt_and_fail(self, manifest_id):
        (Path(self.root) / 'active.json').write_text('{corrupt active')
        raise OSError('simulated manifest rollback interruption')

    monkeypatch.setattr(ManifestRegistry, 'rollback', corrupt_and_fail)
    with pytest.raises(InterruptedTransactionError,
                       match='unreadable after a failed rollback'):
        compiled.rollback(artifacts, manifests, parent['manifest_id'])
    journal = TransactionJournal(journal_path_for(manifests))
    assert journal.read() is not None
    durable = ArtifactRegistry.load(catalog_path_for(manifests))
    assert durable.get(compiled.artifact.artifact_id).status == 'revoked'
    with pytest.raises(RecoveryError, match='active manifest is unreadable'):
        recover_pending_transaction(artifacts, manifests, journal)
    assert journal.read() is not None

    # Restoring the active file allows explicit compensation.
    (Path(manifests.root) / 'active.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    outcome = recover_pending_transaction(artifacts, manifests, journal)
    assert outcome['outcome'] == 'aborted_rollback'
    assert journal.read() is None
    reloaded = ArtifactRegistry.load(catalog_path_for(manifests))
    assert reloaded.get(compiled.artifact.artifact_id).status == 'accepted'
    assert reloaded.publication(compiled.artifact.artifact_id)['active'] is True
