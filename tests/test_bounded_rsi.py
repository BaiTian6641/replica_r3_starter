"""Bounded procedural-RSI plumbing: real execution, gates, rollback, replay."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import require_local_artifacts

from replica_lab.contracts import fingerprint
from replica_lab.runtime.bounded_rsi import (
    APP_SETTINGS, BoundedRsiCycle, CandidateControlError, CandidateSubmission,
    DEFAULT_NEURAL_PINS, EvidenceIntegrityError, EvidencePin, GuardedWorkspace,
    HostPolicy, MEMORY_FORMAT, NEURAL_EVIDENCE_FORMAT, NeuralEvidencePins,
    NeuralGate, PERCEPTION_FORMAT, PINNED_MODEL_REVISION,
    ProtectedRevisionError, ResourceGuard, ResourceLimitError, TaskCase,
    WorldCatalog, WorldFamily, build_owned_worlds, materialize_case)
from replica_lab.runtime.artifacts import ArtifactRegistry
from replica_lab.runtime.compilation import CompiledProcedure, StaleSourceError
from replica_lab.runtime.procedures import RepairProcedure
from replica_lab.runtime.tools import FixtureWorkspace

ROOT = Path(__file__).resolve().parents[1]


def policy(**changes) -> HostPolicy:
    values = dict(
        subject_id='owned-procedure-subject',
        branch_id='bounded-procedure-sandbox',
        purpose='bounded_procedure_rsi',
        base_revision='procedure-sandbox-r0',
        identity_version='replica.identity.procedure_sandbox.v1',
        evidence_epoch=0,
        guard=ResourceGuard(max_tool_calls=8, max_evaluator_runs=5),
    )
    values.update(changes)
    return HostPolicy(**values)


def cycle(tmp_path, name='cycle', **changes) -> BoundedRsiCycle:
    evidence_root, pins = unit_evidence(tmp_path)
    return BoundedRsiCycle(tmp_path / name, policy(**changes),
                           evidence_root=evidence_root, neural_gate=NeuralGate(pins))


def published_cycle(tmp_path, name):
    """Run practice + independent verification + publication for adversarial tests."""
    runner = cycle(tmp_path, name)
    base = runner.prepare_base()
    practice = runner.practice()
    candidate = runner.compile_candidate(practice, base['manifest_id'])
    verification = runner.verify(candidate)
    assert verification.accepted is True
    publication = runner.publish(candidate, verification)
    return runner, base, candidate, verification, publication


def write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()



def unit_evidence(tmp_path):
    """Self-contained neural-gate inputs; no native run artifacts are read."""
    root = tmp_path / 'unit_evidence'
    perception_sha = write_json(root / 'perception.json', {
        'format': PERCEPTION_FORMAT, 'engineering_pass': True,
        'quality_pass': None, 'accepted': False,
        'quality_claim': 'synthetic interface-only fixture',
        'model_revision': PINNED_MODEL_REVISION})
    memory_sha = write_json(root / 'memory.json', {
        'format': MEMORY_FORMAT, 'decision': 'reject', 'deployment': 'rejected',
        'failed_checks': ['synthetic_memory_quality']})
    pins = NeuralEvidencePins(
        perception=EvidencePin('perception', 'perception.json',
                               PERCEPTION_FORMAT, perception_sha),
        memory=EvidencePin('memory', 'memory.json', MEMORY_FORMAT, memory_sha),
        correction=DEFAULT_NEURAL_PINS.correction,
        retention=DEFAULT_NEURAL_PINS.retention)
    return root, pins


def test_cycle_publishes_verified_procedure_rejects_and_rolls_back(tmp_path):
    runner = cycle(tmp_path)
    report = runner.run()
    data = report.as_dict()

    assert data['qualification']['scope'] == 'procedure_plumbing_only'
    assert data['qualification']['integrated_rsi'] is False
    assert data['qualification']['acceleration_claim'] is None
    assert data['qualification']['neural_promotions'] == 0

    # Complete, disjoint families; practice never touches verification/rejection.
    assert data['worlds']['source_family_cases'] == 2
    assert data['worlds']['verification_family_cases'] == 2
    assert data['worlds']['rejection_family_cases'] == 1
    assert data['worlds']['case_ids_disjoint'] is True
    source_ids = set(data['worlds']['source_case_ids'])
    verification_ids = set(data['worlds']['verification_case_ids'])
    rejection_ids = set(data['worlds']['rejection_case_ids'])
    assert source_ids.isdisjoint(verification_ids)
    assert source_ids.isdisjoint(rejection_ids)
    assert verification_ids.isdisjoint(rejection_ids)

    # Actual practice and independent verification.
    assert data['practice']['before_successes'] == 0
    assert data['practice']['after_successes'] == 2
    assert len(data['episodic_observations']) == 4
    assert data['verification']['accepted'] is True
    assert all(case['independent_passed'] is True
               for case in data['verification']['cases'])
    assert all(case['final_exit_code'] == 0
               for case in data['verification']['cases'])
    for case in data['verification']['cases']:
        receipt = case['last_receipt']
        assert receipt['kind'] == 'run_test'
        assert receipt['tool_status'] == 'ok'
        assert receipt['exit_code'] == 0
        assert receipt['timed_out'] is False
        assert len(receipt['stdout_sha256']) == 64
        assert receipt['stdout_excerpt'].startswith('policy ok')
    for case in data['rejection']['cases']:
        receipt = case['last_receipt']
        assert receipt['kind'] == 'run_test'
        assert receipt['tool_status'] == 'error'
        assert receipt['exit_code'] == 1
        assert len(receipt['stdout_sha256']) == 64
        assert 'policy requires' in receipt['stdout_excerpt']

    # One verified publication on the protected parent.
    assert data['publication']['parent'] == data['base']['manifest_id']
    assert data['publication']['evidence_epoch'] == 0
    assert data['publication']['base_revision'] == 'procedure-sandbox-r0'
    assert data['publication']['source_hashes'] == \
        data['retrieval']['source_retrieval']['source_hashes']

    # Valid rejection stayed contained.
    assert data['rejection']['evaluation_accepted'] is False
    assert data['rejection']['publish_refused'] is True
    assert 'did not accept' in data['rejection']['publish_error']
    assert data['rejection']['active_unchanged'] is True
    assert data['rejection']['artifact_status'] == 'rejected'

    # Real ancestor rollback, not just an API call.
    assert data['rollback']['ancestor_manifest_id'] == data['base']['manifest_id']
    assert data['rollback']['active_after_rollback'] == data['base']['manifest_id']
    assert data['rollback']['base_restored'] is True
    assert data['rollback']['revoked'] is True
    assert data['rollback']['candidate_artifact_status'] == 'revoked'
    assert runner.manifests.active()['manifest_id'] == data['base']['manifest_id']
    assert runner.artifacts.get(data['publication']['artifact_id']).status == 'revoked'
    assert runner.artifacts.get(data['rejection']['artifact_id']).status == 'rejected'
    assert runner.artifacts.list(kind='neural_candidate') == ()
    archived = json.loads((runner.manifests.root /
                           f"{data['publication']['manifest_id']}.json").read_text())
    restored = CompiledProcedure.from_manifest(archived, ArtifactRegistry())
    assert restored.artifact.artifact_id == data['publication']['artifact_id']
    assert restored.binding.parent_manifest_id == data['base']['manifest_id']
    assert restored.binding.base_revision == 'procedure-sandbox-r0'
    assert restored.binding.evidence_epoch == 0
    assert restored.source_hashes == data['publication']['source_hashes']
    for case in archived['evaluation']['cases']:
        receipt = case['last_receipt']
        assert receipt['exit_code'] == 0
        assert len(receipt['stdout_sha256']) == 64
        assert len(receipt['stderr_sha256']) == 64

    # Neural promotion is truthfully blocked by actual pinned evidence.
    neural = data['neural']
    assert neural['decision'] == 'deferred'
    assert neural['promotable'] is False
    assert 'perception_quality_not_established' in neural['blockers']
    assert neural['gates']['perception']['quality_pass'] is None
    assert 'learned_memory_rejected' in neural['blockers']
    assert neural['gates']['memory']['decision'] == 'reject'
    assert 'correction_evidence_not_pinned' in neural['blockers']
    assert 'retention_evidence_not_pinned' in neural['blockers']
    assert neural['software_promoted'] is False
    assert neural['promotion_authority'] == 'host'
    assert data['selector_policy']['changes'] == []
    assert data['retrieval']['memory_only_recall']['status'] == 'not_exercised'
    raw = data['raw_traces']
    assert raw['practice'] and raw['verification'] and raw['rejection']
    assert any(case['independent_passed'] is False for case in raw['rejection'])
    assert any((receipt.get('receipt') or {}).get('exit_code') == 1
               for entry in raw['practice']
               for receipt in entry['receipts']
               if isinstance(receipt, dict))

    with pytest.raises(ProtectedRevisionError):
        runner.rollback('not-the-protected-ancestor')


def test_forged_gate_evidence_fails_closed(tmp_path):
    root, pins = unit_evidence(tmp_path)
    real_report = json.loads((root / 'perception.json').read_text())
    forged = dict(real_report, quality_pass=True, accepted=True)
    forged_path = root / 'forged_perception.json'
    write_json(forged_path, forged)
    perception_pin = EvidencePin('perception', 'forged_perception.json',
                                 PERCEPTION_FORMAT, pins.perception.sha256)
    gate = NeuralGate(replace(pins, perception=perception_pin))
    with pytest.raises(EvidenceIntegrityError, match='hash differs'):
        gate.assess(root)

    verdict = json.loads((root / 'memory.json').read_text())
    forged_verdict = dict(verdict, decision='accept', deployment='published',
                          failed_checks=[])
    forged_verdict_path = root / 'forged_memory.json'
    write_json(forged_verdict_path, forged_verdict)
    memory_pin = EvidencePin('memory', 'forged_memory.json', MEMORY_FORMAT,
                             pins.memory.sha256)
    gate = NeuralGate(replace(
        pins, perception=replace(pins.perception, sha256=None), memory=memory_pin))
    with pytest.raises(EvidenceIntegrityError, match='hash differs'):
        gate.assess(root)

    # Unpinned evidence is unusable even when a plausible file exists.
    expected_correction = root / pins.correction.relative_path
    write_json(expected_correction, {
        'format': NEURAL_EVIDENCE_FORMAT, 'decision': 'accept',
        'checks': {'corrected': True}})
    gate = NeuralGate(replace(
        pins,
        perception=replace(pins.perception, sha256=None),
        memory=replace(pins.memory, sha256=None)))
    assessment = gate.assess(root)
    assert assessment['decision'] == 'deferred'
    assert 'correction_evidence_not_pinned' in assessment['blockers']
    assert assessment['gates']['correction']['status'] == 'not_pinned'
    assert assessment['software_promoted'] is False


@require_local_artifacts(
    'runs/minicpm5_vision_v0_20261025/score_restart_replay01/score_report.json')
@pytest.mark.integration
def test_historical_native_evidence_remains_explicitly_provisioned():
    assessment = NeuralGate().assess(ROOT)
    assert assessment['decision'] == 'deferred'
    assert 'perception_quality_not_established' in assessment['blockers']
    assert 'learned_memory_rejected' in assessment['blockers']


def test_candidate_claims_and_path_traversal_cannot_control_verdict(tmp_path):
    runner = cycle(tmp_path)
    base = runner.prepare_base()
    practice = runner.practice()
    claimed = CandidateSubmission(
        name='repair-config',
        claimed_acceptance=True,
        claimed_source_hashes={practice.failure_record_ids[0]: '0' * 64})
    candidate = runner.compile_candidate(practice, base['manifest_id'],
                                         submission=claimed)
    assert 'claimed_acceptance' in candidate.claims_ignored
    assert 'claimed_source_hashes' in candidate.claims_ignored
    verification = runner.verify(candidate)
    assert verification.accepted is True
    assert runner.manifests.active()['manifest_id'] == base['manifest_id']

    hostile = runner.compile_candidate(
        practice, base['manifest_id'],
        submission=CandidateSubmission(name='repair-config-hostile',
                                       workspace_entry='../evaluator.py'))
    hostile_verification = runner.verify(hostile)
    assert hostile_verification.accepted is False
    assert any(case['error'] == 'read_failed' for case in hostile_verification.cases)
    with pytest.raises(PermissionError, match='did not accept'):
        runner.publish(hostile, hostile_verification)
    assert runner.manifests.active()['manifest_id'] == base['manifest_id']

    forged_acceptance = replace(
        hostile_verification,
        evaluation=replace(hostile_verification.evaluation, accepted=True))
    with pytest.raises(PermissionError, match='not usable'):
        runner.publish(hostile, forged_acceptance)
    assert runner.manifests.active()['manifest_id'] == base['manifest_id']

    with pytest.raises(CandidateControlError):
        runner.verify(practice)
    with pytest.raises(CandidateControlError):
        runner.publish(hostile, hostile_verification.evaluation)
    with pytest.raises(TypeError):
        CandidateSubmission(name='repair-config', verifier=object())
    with pytest.raises(TypeError):
        CandidateSubmission(name='repair-config', guard=ResourceGuard())
    with pytest.raises(TypeError):
        CandidateSubmission(name='repair-config', base_revision='candidate-chosen')
    with pytest.raises(ProtectedRevisionError):
        runner.compile_candidate(practice, 'foreign-parent')


def test_source_epoch_parent_and_identity_staleness_are_rejected(tmp_path):
    first = cycle(tmp_path, 'source-stale')
    base = first.prepare_base()
    practice = first.practice()
    candidate = first.compile_candidate(practice, base['manifest_id'])
    verification = first.verify(candidate)
    assert verification.accepted is True
    first.ledger.append_correction('source evidence corrected after evaluation')
    with pytest.raises(StaleSourceError, match='epoch changed'):
        first.publish(candidate, verification)
    assert first.manifests.active()['manifest_id'] == base['manifest_id']
    with pytest.raises(StaleSourceError):
        candidate.compiled.assert_sources_current({'forged': 'a' * 64})

    second = cycle(tmp_path, 'parent-stale')
    base = second.prepare_base()
    practice = second.practice()
    candidate = second.compile_candidate(practice, base['manifest_id'])
    verification = second.verify(candidate)
    other_payload = {
        'base_revision': 'procedure-sandbox-r0',
        'identity_version': 'replica.identity.procedure_sandbox.v1',
        'owner': {'subject_id': 'owned-procedure-subject',
                  'branch_id': 'bounded-procedure-sandbox',
                  'purpose': 'bounded_procedure_rsi'},
        'guarded_base': {'format': 'replica.bounded_rsi.base.v1',
                         'evidence_epoch': 0,
                         'source_family': 'source.queue_policy.v1'},
    }
    second.manifests.publish(other_payload, {
        'decision': 'accept',
        'candidate_digest': fingerprint(other_payload),
        'parent_manifest_id': base['manifest_id'],
        'artifact_digest': fingerprint(other_payload),
        'cases': [],
        'evaluation_scope': 'host_reevaluation',
    }, expected_parent=base['manifest_id'])
    with pytest.raises(ValueError, match='active parent changed'):
        second.publish(candidate, verification)

    third = cycle(tmp_path, 'identity-stale')
    base = third.prepare_base()
    practice = third.practice()
    candidate = third.compile_candidate(practice, base['manifest_id'])
    verification = third.verify(candidate)
    wrong_identity = dict(other_payload,
                          identity_version='hostile.identity.revision')
    third.manifests.publish(wrong_identity, {
        'decision': 'accept',
        'candidate_digest': fingerprint(wrong_identity),
        'parent_manifest_id': base['manifest_id'],
        'artifact_digest': fingerprint(wrong_identity),
        'cases': [],
        'evaluation_scope': 'host_reevaluation',
    }, expected_parent=base['manifest_id'])
    with pytest.raises(ProtectedRevisionError, match='identity version'):
        third.publish(candidate, verification)


def test_replaced_and_in_place_mutated_candidates_are_rejected_before_execution(
        tmp_path):
    runner = cycle(tmp_path)
    base = runner.prepare_base()
    practice = runner.practice()
    candidate = runner.compile_candidate(practice, base['manifest_id'])
    artifact_id = candidate.artifact_id

    # dataclasses.replace() creates a foreign candidate even with a genuine
    # artifact, genuine hashes and an altered protected binding.
    replaced = replace(candidate, compiled=replace(
        candidate.compiled,
        binding=replace(candidate.compiled.binding,
                        identity_version='hostile.identity')))
    with pytest.raises(CandidateControlError, match='issued unchanged'):
        runner.verify(replaced)
    assert runner.artifacts.get(artifact_id).status == 'proposed'

    # In-place binding tampering keeps object identity but changes authority.
    in_place = runner.compile_candidate(
        practice, base['manifest_id'],
        submission=CandidateSubmission(name='repair-config-inplace'))
    object.__setattr__(in_place.compiled, 'binding', replace(
        in_place.compiled.binding, base_revision='candidate-chosen'))
    with pytest.raises(CandidateControlError, match='authority changed'):
        runner.verify(in_place)
    assert runner.artifacts.get(in_place.artifact_id).status == 'proposed'

    # The source-hash map is mutable even inside a frozen candidate.
    tampered = runner.compile_candidate(
        practice, base['manifest_id'],
        submission=CandidateSubmission(name='repair-config-hashes'))
    record_id = next(iter(tampered.compiled.source_hashes))
    tampered.compiled.source_hashes[record_id] = '0' * 64
    with pytest.raises(CandidateControlError, match='authority changed'):
        runner.verify(tampered)
    assert runner.artifacts.get(tampered.artifact_id).status == 'proposed'

    # Genuine independent execution still works, but a replacement cannot
    # publish against that verification.
    verification = runner.verify(candidate)
    assert verification.accepted is True
    with pytest.raises(CandidateControlError, match='issued unchanged'):
        runner.publish(replaced, verification)
    assert runner.manifests.active()['manifest_id'] == base['manifest_id']

    # Mutating an already-verified candidate's source map is also rejected.
    publish_tamper = runner.compile_candidate(
        practice, base['manifest_id'],
        submission=CandidateSubmission(name='repair-config-publish-tamper'))
    publish_verification = runner.verify(publish_tamper)
    assert publish_verification.accepted is True
    publish_tamper.compiled.source_hashes[record_id] = '0' * 64
    with pytest.raises(CandidateControlError, match='authority changed'):
        runner.publish(publish_tamper, publish_verification)
    assert runner.manifests.active()['manifest_id'] == base['manifest_id']


def test_prepare_base_metadata_is_detached_from_rollback_authority(tmp_path):
    runner, base, candidate, _verification, publication = published_cycle(
        tmp_path, 'detached-base')
    original_base_id = base['manifest_id']
    base['manifest_id'] = publication['manifest_id']
    with pytest.raises(ProtectedRevisionError, match='not the protected base'):
        runner.rollback(publication['manifest_id'])
    assert runner.manifests.active()['manifest_id'] == \
        publication['manifest_id']
    assert runner.artifacts.get(candidate.artifact_id).status == 'accepted'

    # The private protected snapshot still restores the true ancestor.
    rolled = runner.rollback(original_base_id)
    assert rolled['base_restored'] is True
    assert rolled['active_after_rollback'] == original_base_id


def test_rollback_authenticates_archived_protected_ancestor(tmp_path):
    runner, base, candidate, _verification, publication = published_cycle(
        tmp_path, 'tampered-ancestor')
    base_id = base['manifest_id']
    archive = runner.manifests.root / f'{base_id}.json'
    manifest = json.loads(archive.read_text())
    manifest['payload']['owner']['subject_id'] = 'tampered-subject'
    archive.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    with pytest.raises(ProtectedRevisionError):
        runner.rollback(base_id)
    # No active-manifest mutation and no artifact revocation happened.
    assert runner.manifests.active()['manifest_id'] == \
        publication['manifest_id']
    assert runner.artifacts.get(candidate.artifact_id).status == 'accepted'


def test_replay_is_deterministic_across_isolated_roots(tmp_path):
    first = cycle(tmp_path, 'run-a').run().as_dict()
    second = cycle(tmp_path, 'run-b').run().as_dict()
    assert first['replay_digest'] == second['replay_digest']
    assert first['publication']['manifest_id'] == \
        second['publication']['manifest_id']
    assert first['base']['manifest_id'] == second['base']['manifest_id']
    assert first['verification']['cases'] == second['verification']['cases']
    assert first['rejection']['cases'] == second['rejection']['cases']
    assert first['neural'] == second['neural']
    assert {key: value for key, value in first.items() if key != 'raw_traces'} == \
        {key: value for key, value in second.items() if key != 'raw_traces'}


def test_complete_family_split_precedes_expansion(tmp_path):
    catalog = build_owned_worlds()
    summary = catalog.assert_isolated(policy())
    assert summary['families']['source']['family_id'] == 'source.queue_policy.v1'
    assert summary['families']['verification']['roles'] == ['dev', 'heldout']
    assert summary['independent_world_families'] == 2

    # A verification family without heldout cases is incomplete.
    with pytest.raises(ValueError, match='exactly roles'):
        WorldFamily('verify.incomplete.v1', 'verification', (
            TaskCase('incomplete_dev', 'dev', 'Incomplete.', (('X_BOUND', 1, 2),)),))
    # A case cannot move a verification role into the source family.
    with pytest.raises(ValueError, match='exactly roles'):
        WorldFamily('source.mixed.v1', 'source', (
            TaskCase('mixed_dev', 'dev', 'Mixed.', (('X_BOUND', 1, 2),)),))
    # Expansions stay inside one complete family and cannot duplicate case IDs.
    with pytest.raises(ValueError, match='exactly one complete source family'):
        WorldCatalog(catalog.worlds + (WorldFamily(
            'source.second.v1', 'source', (
                TaskCase('src_train_other', 'train', 'Other.',
                         (('QUEUE_OTHER', 1, 3),)),)),))
    with pytest.raises(ValueError, match='only one complete family'):
        catalog.expanded('source.queue_policy.v1', (
            TaskCase('ver_holdout_window', 'train', 'Duplicate ID.',
                     (('X_BOUND', 1, 2),)),))
    expanded = catalog.expanded('source.queue_policy.v1', (
        TaskCase('src_train_extra', 'train', 'New source expansion.',
                 (('QUEUE_BURST', 2, 4),)),))
    assert expanded.by_split('source').cases[-1].case_id == 'src_train_extra'
    assert expanded.by_split('verification') == catalog.by_split('verification')
    assert len(catalog.by_split('source').cases) == 2

    with pytest.raises(ValueError, match='policy verification family'):
        BoundedRsiCycle(tmp_path / 'bad-catalog', policy(),
                        catalog=WorldCatalog((
                            catalog.by_split('source'),
                            catalog.by_split('rejection'),
                            WorldFamily('verify.wrong.v1', 'verification', (
                                TaskCase('wrong_dev', 'dev', 'Wrong.',
                                         (('X_BOUND', 1, 2),)),
                                TaskCase('wrong_heldout', 'heldout', 'Wrong.',
                                         (('Y_BOUND', 1, 2),)),)),
                        )), evidence_root=ROOT)


def test_host_resource_guard_cannot_be_raised_by_candidate(tmp_path):
    family = build_owned_worlds().by_split('source')
    case = family.cases[0]
    first = materialize_case(family, case, tmp_path / 'guard-tool-calls')
    workspace = FixtureWorkspace(first)
    guarded = GuardedWorkspace(workspace, ResourceGuard(
        max_tool_calls=1, max_evaluator_runs=1))
    procedure = RepairProcedure('repair-config', APP_SETTINGS, ('obs-0001',))
    with pytest.raises(ResourceLimitError, match='tool-call budget'):
        procedure.execute(guarded)
    assert guarded.tool_calls == 2

    second = materialize_case(family, case, tmp_path / 'guard-evaluator-runs')
    guarded = GuardedWorkspace(FixtureWorkspace(second), ResourceGuard(
        max_tool_calls=8, max_evaluator_runs=1))
    with pytest.raises(ResourceLimitError, match='evaluator-run budget'):
        procedure.execute(guarded)
    assert guarded.evaluator_runs == 2


def test_neural_gate_never_promotes_from_software_checks(tmp_path):
    root = tmp_path / 'evidence'
    perception_sha = write_json(root / 'perception.json', {
        'format': PERCEPTION_FORMAT, 'quality_pass': True, 'accepted': True,
        'engineering_pass': True, 'quality_claim': 'test fixture',
        'model_revision': PINNED_MODEL_REVISION})
    memory_sha = write_json(root / 'memory.json', {
        'format': MEMORY_FORMAT, 'decision': 'accept', 'deployment': 'published',
        'checks': {'heldout': True}, 'failed_checks': []})
    correction_sha = write_json(root / 'correction.json', {
        'format': NEURAL_EVIDENCE_FORMAT, 'decision': 'accept',
        'checks': {'corrected_after_feedback': True}})
    retention_sha = write_json(root / 'retention.json', {
        'format': NEURAL_EVIDENCE_FORMAT, 'decision': 'accept',
        'checks': {'heldout_retention': True}})
    pins = NeuralEvidencePins(
        perception=EvidencePin('perception', 'perception.json',
                               PERCEPTION_FORMAT, perception_sha),
        memory=EvidencePin('memory', 'memory.json', MEMORY_FORMAT, memory_sha),
        correction=EvidencePin('correction', 'correction.json',
                               NEURAL_EVIDENCE_FORMAT, correction_sha),
        retention=EvidencePin('retention', 'retention.json',
                              NEURAL_EVIDENCE_FORMAT, retention_sha))
    assessment = NeuralGate(pins).assess(root)
    assert assessment['decision'] == 'deferred_host_review'
    assert assessment['blockers'] == []
    assert assessment['software_promoted'] is False
    assert assessment['promotion_authority'] == 'host'
    assert assessment['neural_training_performed'] is False
    assert assessment['neural_artifact_created'] is False
    assert not hasattr(NeuralGate, 'publish')
    assert not hasattr(BoundedRsiCycle, 'publish_neural')
