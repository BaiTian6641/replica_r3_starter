"""Typed E4 artifacts, lifecycle transitions and dependency invalidation."""

import hashlib
import pytest

from replica_lab.runtime.artifacts import (
    ArtifactRegistry, BranchMemoryImage, ContextPack, NeuralCandidate,
    ProcedureArtifact, ProcedureStep, RecordLatent,
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
    assert manifest['format'] == 'replica.artifacts.catalog.v1'
    assert manifest['artifacts'][0]['kind'] == 'context_pack'
