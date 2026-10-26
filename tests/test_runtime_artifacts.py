"""Typed E4 artifacts, lifecycle transitions, catalog durability and invalidation."""

import hashlib
import json

import pytest

from replica_lab.runtime.artifacts import (
    ArtifactRegistry, BranchMemoryImage, ContextPack, NeuralCandidate,
    ProcedureArtifact, ProcedureStep, RecordLatent, artifact_from_dict,
    artifact_to_dict, verify_catalog_summary,
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def context(source=('src-1',)):
    text = 'Approved context: the evaluator is authoritative.'
    return ContextPack('ctx-1', source, 'subject-1', 'branch-a', 'test', 10,
                       text, sha(text), {'writer': 'v1'}, status='built')


def latent():
    return RecordLatent('latent-1', 'record-1', sha('source'), sha('span'),
                        4, 256, sha('payload'), 'writer-1', {'writer': 'v1'},
                        source_ids=('src-1',), parent_id='ctx-1')


def image():
    return BranchMemoryImage('image-1', 'branch-a', 'subject-1', 'parallel_replica',
                             sha('fast-state'), 'producer-1', 2, 30, 0,
                             (('obs-1', sha('source')),), sha('payload'),
                             ('src-1',), {'fast': 'v1'}, parent_id='latent-1')


def procedure():
    return ProcedureArtifact('proc-1', 'repair-config', 1, ('a test failed',),
                             (ProcedureStep('run_test', {}),), ('unittest passes',),
                             ('event-1',), {'runtime': 'v1'}, status='proposed')


def candidate(status='proposed'):
    return NeuralCandidate('candidate-1', sha('parent'), ('core.0.weight',),
                           sha('weights'), 'modules_only', 'muon8bit',
                           {'core': 'v1'}, ('src-1',), status=status)


def test_view_and_candidate_lifecycles_are_distinct():
    registry = ArtifactRegistry()
    registry.add(context()); registry.add(latent()); registry.add(image())
    assert registry.transition('ctx-1', 'active', now=1).status == 'active'
    with pytest.raises(ValueError):
        registry.transition('ctx-1', 'accepted')
    registry.add(procedure()); registry.add(candidate())
    assert registry.transition('proc-1', 'validated').status == 'validated'
    assert registry.transition('proc-1', 'accepted').status == 'accepted'
    assert registry.transition('proc-1', 'revoked').status == 'revoked'
    assert registry.transition('candidate-1', 'validated').status == 'validated'
    assert registry.transition('candidate-1', 'rejected').status == 'rejected'


def test_source_invalidation_cascades_to_children_and_candidates():
    registry = ArtifactRegistry()
    registry.add(context()); registry.add(latent()); registry.add(image())
    registry.add(candidate()); registry.add(procedure())
    registry.transition('ctx-1', 'active')
    registry.transition('latent-1', 'active')
    registry.transition('image-1', 'active')
    registry.transition('candidate-1', 'validated')
    registry.invalidate_source('src-1', reason='source revoked')
    assert registry.get('ctx-1').status == 'invalidated'
    assert registry.get('latent-1').status == 'invalidated'
    assert registry.get('image-1').status == 'invalidated'
    assert registry.get('candidate-1').status == 'rejected'
    with pytest.raises(PermissionError):
        registry.payload('latent-1')


def test_artifact_validation_rejects_unbacked_procedure_and_mismatched_hashes():
    with pytest.raises(ValueError):
        ProcedureArtifact('p', 'x', 1, (), (ProcedureStep('run_test', {}),),
                          (), ('event',), {'runtime': 'v1'})
    with pytest.raises(ValueError):
        ContextPack('c', ('src',), 's', 'b', 'p', 1, 'text', '0' * 64,
                    {'runtime': 'v1'})
    with pytest.raises(ValueError):
        RecordLatent('r', 'record', '0' * 63, sha('span'), 4, 256, sha('payload'),
                     'writer', {'runtime': 'v1'}, source_ids=('src',))
    with pytest.raises(ValueError):
        NeuralCandidate('n', '0' * 63, ('x',), sha('w'), 'mod', 'muon8bit',
                        {'runtime': 'v1'}, ('src',))


def test_expiry_and_catalog_manifest(tmp_path):
    registry = ArtifactRegistry()
    item = ContextPack('ctx-2', ('src-1',), 'subject-1', 'branch-a', 'test', 10,
                       'text', sha('text'), {'runtime': 'v1'}, expires_at=5)
    registry.add(item)
    with pytest.raises(ValueError):
        registry.transition('ctx-2', 'active', now=6)
    path = tmp_path / 'catalog.json'
    manifest = registry.save(path)
    assert manifest['format'] == 'replica.artifacts.catalog.v2'
    assert manifest['artifacts'][0]['kind'] == 'context_pack'
    assert manifest['artifacts'][0]['content_digest']
    reloaded = ArtifactRegistry.load(path)
    assert reloaded.get('ctx-2').digest == item.digest
    assert reloaded.export() == manifest

    # A flipped byte in any stored field invalidates the whole catalog.
    tampered = json.loads(path.read_text())
    tampered['artifacts'][0]['text'] = 'different text'
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        ArtifactRegistry.load(path)


def test_legacy_summary_catalog_is_explicitly_validated_not_silently_loaded(tmp_path):
    path = tmp_path / 'artifact_catalog.json'
    path.write_text(json.dumps({
        'format': 'replica.artifacts.catalog.v1',
        'artifacts': [
            {'artifact_id': 'artifact-repair-config', 'kind': 'procedure',
             'status': 'revoked', 'digest': sha('revoked')},
        ]}))
    with pytest.raises(ValueError, match='has no artifact payloads'):
        ArtifactRegistry.load(path)
    migrated = ArtifactRegistry.load_legacy_catalog(path)
    assert migrated['reconstructable'] is False
    assert migrated['artifacts'][0]['status'] == 'revoked'
    # A live registry can emit the historical contract for summary-only readers.
    registry = ArtifactRegistry()
    registry.add(procedure())
    registry.transition('proc-1', 'validated')
    registry.transition('proc-1', 'accepted')
    registry.record_publication('proc-1', 'c' * 64, sha('payload'))
    summary = verify_catalog_summary(registry.summary())
    assert summary[0]['artifact_id'] == 'proc-1'
    assert summary[0]['status'] == 'accepted'
    assert verify_catalog_summary({
        'format': 'replica.artifacts.catalog.v1',
        'artifacts': [{'artifact_id': 'x', 'kind': 'procedure',
                       'status': 'accepted', 'digest': '0' * 64}]})[0][
                           'artifact_id'] == 'x'
    with pytest.raises(ValueError, match='schema mismatch'):
        verify_catalog_summary({'format': 'replica.artifacts.catalog.v1',
                                'artifacts': [{'artifact_id': 'x'}]})
    with pytest.raises(ValueError, match='impossible context_pack status'):
        verify_catalog_summary({
            'format': 'replica.artifacts.catalog.v1',
            'artifacts': [{'artifact_id': 'x', 'kind': 'context_pack',
                           'status': 'accepted', 'digest': '0' * 64}]})
    with pytest.raises(ValueError, match='impossible procedure status'):
        verify_catalog_summary({
            'format': 'replica.artifacts.catalog.v1',
            'artifacts': [{'artifact_id': 'x', 'kind': 'procedure',
                           'status': 'built', 'digest': '0' * 64}]})


def test_accepted_procedure_requires_an_active_publication(tmp_path):
    registry = ArtifactRegistry()
    registry.add(procedure())
    registry.transition('proc-1', 'validated')
    registry.transition('proc-1', 'accepted')
    with pytest.raises(ValueError, match='no active publication'):
        registry.assert_consistent()
    with pytest.raises(ValueError, match='no active publication'):
        registry.save(tmp_path / 'catalog.json')

    registry.record_publication('proc-1', 'a' * 64, sha('payload'))
    registry.assert_consistent()
    path = tmp_path / 'catalog.json'
    registry.save(path)
    reloaded = ArtifactRegistry.load(path)
    published = reloaded.publication('proc-1')
    assert published['manifest_id'] == 'a' * 64
    assert published['active'] is True
    assert published['imported'] is False

    reloaded.transition('proc-1', 'revoked', reason='manifest rollback')
    assert reloaded.publication('proc-1')['active'] is False
    reloaded.assert_consistent()

    # A publication cannot be quietly rewritten for different bytes.
    with pytest.raises(ValueError, match='would be rewritten'):
        reloaded.record_publication('proc-1', 'b' * 64, sha('payload'))

    # Non-procedure artifacts never publish through manifests.
    views = ArtifactRegistry()
    views.add(context())
    with pytest.raises(ValueError, match='only procedures'):
        views.record_publication('ctx-1', 'd' * 64, sha('payload'))


def test_artifact_record_round_trip_preserves_digests():
    successors = {'context_pack': 'active', 'record_latent': 'active',
                  'branch_memory_image': 'active', 'procedure': 'validated',
                  'neural_candidate': 'validated'}
    for item in (context(), latent(), image(), procedure(), candidate()):
        record = artifact_to_dict(item, content_digest=True)
        rebuilt = artifact_from_dict(record, content_digest=True)
        assert rebuilt == item
        assert rebuilt.digest == item.digest
        record['status'] = successors[item.kind]
        with pytest.raises(ValueError, match='digest'):
            artifact_from_dict(record, content_digest=True)
