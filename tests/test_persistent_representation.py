"""Deterministic representation, source-view and legacy persistent-state boundaries."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from architecture_helpers import tiny_adapter
from replica_lab.memory_reliability import build_episodes
from replica_lab.runtime.exact_memory import ExactMemoryReference, episode_records
from replica_lab.runtime.fast_memory import FastMemoryScope, save_fast_depth_image
from replica_lab.runtime.memory_representation import FACT_REPRESENTATION, RAW_REPRESENTATION
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
from test_exact_memory import SPEC, Tokenizer, scope


@pytest.fixture
def sources(tmp_path):
    episode = next(e for e in build_episodes() if e['split'] == 'dev')
    reference = ExactMemoryReference(tmp_path / 'ledger.sqlite')
    reference.admit_episode(episode)
    model = tiny_adapter(fast_memory=SPEC).eval()
    owner = FastMemoryScope('memory_reliability', episode['episode_id'], 'reliability', 'parallel_replica', 0)
    try:
        yield model, reference, episode, owner
    finally:
        reference.close()


def state_for(sources, representation):
    model, _, episode, owner = sources
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=representation)
    initial, _ = episode_records(episode)
    for record in initial:
        memory.append_record(record)
    return memory, initial


@pytest.mark.parametrize('representation', [RAW_REPRESENTATION, FACT_REPRESENTATION])
def test_unchanged_ordered_history_save_load_rebuild_is_exact(sources, tmp_path, representation):
    model, reference, episode, owner = sources
    memory, history = state_for(sources, representation)
    before = memory.read().state_digest
    before_key = memory.cache_key
    memory.rebuild(history, semantics='history')
    assert memory.read().state_digest == before
    assert memory.cache_key == before_key
    memory.bind_source(reference, scope=scope(episode))
    metadata = memory.save(tmp_path / 'state', require_source_binding=True)
    assert metadata['persistent']['representation'] == representation
    loaded = PersistentFastMemory.load(
        tmp_path / 'state', model, Tokenizer(), owner, representation=representation,
        require_source_binding=True, source_reference=reference, source_scope=scope(episode))
    assert loaded.read().state_digest == before
    loaded.rebuild(history, semantics='history')
    assert loaded.read().state_digest == before and loaded.source_binding is None
    with pytest.raises(ValueError, match='Source-bound'):
        loaded.save(tmp_path / 'unbound', require_source_binding=True)
    loaded.bind_source(reference, scope=scope(episode))
    loaded.save(tmp_path / 'rebound', require_source_binding=True)


@pytest.mark.parametrize('representation', [RAW_REPRESENTATION, FACT_REPRESENTATION])
def test_current_view_correction_revocation_and_unaffected_sources(sources, representation):
    model, reference, episode, owner = sources
    memory, history = state_for(sources, representation)
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    first = memory.read().state_digest
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    assert memory.read().state_digest == first
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='stale'):
        memory.read()
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode, 'corrected'))
    current = reference.view(scope(episode, 'corrected')).records
    _, correction = episode_records(episode)
    assert correction.supersedes not in [event for event, _ in memory.lineage]
    assert correction.record_id in [event for event, _ in memory.lineage]
    assert set(record.record_id for record in history if record.record_id != correction.supersedes) <= {
        event for event, _ in memory.lineage}
    fresh = PersistentFastMemory(model, Tokenizer(), owner, representation=representation)
    for record in current:
        fresh.append_record(record)
    torch.testing.assert_close(memory.read().weights, fresh.read().weights, atol=0, rtol=0)
    reference.revoke(correction.supersedes, authorized_subject=owner.subject_id,
                     authorized_branch=owner.branch_id)
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode, 'corrected'))
    assert correction.record_id not in [event for event, _ in memory.lineage]
    assert len(memory.lineage) == len(history) - 1


@pytest.mark.parametrize('representation', [RAW_REPRESENTATION, FACT_REPRESENTATION])
def test_failed_history_rebuild_leaves_previous_state_and_binding(sources, monkeypatch, representation):
    model, reference, episode, _ = sources
    memory, history = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    before = memory.read().weights.clone()
    lineage, events, binding = deepcopy(memory.lineage), deepcopy(memory.events), deepcopy(memory.source_binding)
    encoder = model.encode_fast_observation
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('interrupted representation encode')
        return encoder(*args, **kwargs)
    monkeypatch.setattr(model, 'encode_fast_observation', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        memory.rebuild(history, semantics='history')
    torch.testing.assert_close(memory.read().weights, before, atol=0, rtol=0)
    assert memory.lineage == lineage and memory.events == events and memory.source_binding == binding


@pytest.mark.parametrize('representation', [RAW_REPRESENTATION, FACT_REPRESENTATION])
def test_history_rejects_changed_order_hash_and_correction_provenance(sources, representation):
    memory, history = state_for(sources, representation)
    for altered in (history[::-1], (replace(history[0], text='changed'),) + history[1:],
                    (replace(history[0], source_id='other-source'),) + history[1:]):
        with pytest.raises(ValueError, match='History replay'):
            memory.rebuild(altered, semantics='history')
    with pytest.raises(ValueError, match='semantics'):
        memory.rebuild(history, semantics='unspecified')
    with pytest.raises(ValueError, match='authoritative'):
        memory.rebuild(history, semantics='current_view')


def test_representation_producer_cache_and_tokenizer_versions_are_bound(sources, tmp_path):
    model, _, _, owner = sources
    raw, _ = state_for(sources, RAW_REPRESENTATION)
    fact, _ = state_for(sources, FACT_REPRESENTATION)
    assert raw.producer != fact.producer and raw.cache_key != fact.cache_key
    assert raw.read().encoding_identity != fact.read().encoding_identity
    raw.save(tmp_path / 'raw')
    with pytest.raises(ValueError, match='representation|metadata'):
        PersistentFastMemory.load(tmp_path / 'raw', model, Tokenizer(), owner, representation=FACT_REPRESENTATION)
    with pytest.raises(ValueError, match='representation'):
        PersistentFastMemory(model, Tokenizer(), owner, representation='raw_observation.v999')
    raw.tokenizer.new_normalization = 'changed'
    with pytest.raises(ValueError, match='producer'):
        raw.read()
    with torch.no_grad():
        next(model.fast_depth_write_encoders.parameters()).add_(.01)
    with pytest.raises(ValueError, match='producer'):
        fact.read()


def test_mixed_encodings_and_unauthenticated_publication_are_rejected(sources, tmp_path):
    model, reference, episode, owner = sources
    raw, history = state_for(sources, RAW_REPRESENTATION)
    with pytest.raises(ValueError, match='Fact append'):
        raw.append_fact('mixed', episode['facts'][0])
    fact, _ = state_for(sources, FACT_REPRESENTATION)
    with pytest.raises(ValueError, match='Raw append'):
        fact.append('mixed', history[0].text)
    with pytest.raises(PermissionError, match='scope'):
        raw.bind_source(reference, scope=replace(scope(episode), branch_id='foreign'))
    raw.bind_source(reference, scope=scope(episode))
    raw.save(tmp_path / 'bound', require_source_binding=True)
    inspected = PersistentFastMemory.load(tmp_path / 'bound', model, Tokenizer(), owner)
    assert inspected.source_binding is None
    with pytest.raises(ValueError, match='Source-bound'):
        inspected.save(tmp_path / 'unverified', require_source_binding=True)
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='stale'):
        PersistentFastMemory.load(tmp_path / 'bound', model, Tokenizer(), owner,
                                  require_source_binding=True, source_reference=reference,
                                  source_scope=scope(episode, 'corrected'))


@pytest.mark.parametrize('representation', [RAW_REPRESENTATION, FACT_REPRESENTATION])
def test_legacy_requires_explicit_read_only_interpretation_then_source_reencode(sources, tmp_path, representation):
    model, _, _, owner = sources
    memory, history = state_for(sources, representation)
    _, correction = episode_records(sources[2])
    memory.append_record(correction)
    history = history + (correction,)
    legacy = tmp_path / 'legacy'
    save_fast_depth_image(legacy, memory.read(), config=memory.config, scope=owner,
                          producer=model.fast_version_key(), chunk_index=len(history),
                          update_count=len(history), cursor=len(history), lineage=tuple(memory.lineage))
    with pytest.raises(ValueError, match='Legacy snapshot'):
        PersistentFastMemory.load(legacy, model, Tokenizer(), owner)
    loaded = PersistentFastMemory.load(legacy, model, Tokenizer(), owner,
                                      legacy_representation=representation)
    assert loaded.read().state_digest == memory.read().state_digest
    assert loaded.read().cache_key != memory.read().cache_key
    with pytest.raises(ValueError, match='read-only'):
        loaded.rebuild(history, semantics='history')
    with pytest.raises(ValueError, match='read-only'):
        loaded.save(tmp_path / 'silent-relabel')
    migrated = loaded.migrate_legacy(history, semantics='history')
    assert migrated.read().state_digest == memory.read().state_digest
    receipt = migrated.save(tmp_path / 'migrated')
    assert receipt['persistent']['migration']['operation'] == 'explicit_observation_reencode'
    assert receipt['persistent']['migration']['legacy_depth_manifest_sha256']


@pytest.mark.parametrize('mutation', ['representation_version', 'tokenizer', 'engine', 'events',
                                      'model', 'correction', 'manifest_digest'])
def test_snapshot_metadata_cannot_hide_representation_or_producer_change(sources, tmp_path, mutation):
    from replica_lab.architecture.contracts import fingerprint
    model, _, _, owner = sources
    memory, _ = state_for(sources, FACT_REPRESENTATION)
    path = tmp_path / 'state'
    memory.save(path)
    manifest_path = path / 'persistent_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if mutation == 'representation_version':
        manifest['representation_config']['version'] = True
    elif mutation == 'tokenizer':
        manifest['dependencies']['tokenizer'] = '0' * 64
    elif mutation == 'engine':
        manifest['engine_config']['max_tokens_per_chunk'] = 255
    elif mutation == 'events':
        manifest['events'].reverse()
    elif mutation == 'model':
        manifest['model_version'] = '0' * 64
    elif mutation == 'correction':
        manifest['events'][0]['correction'] = True
    else:
        manifest['manifest_digest'] = '0' * 64
    if mutation != 'manifest_digest':
        manifest['manifest_digest'] = fingerprint({key: value for key, value in manifest.items()
                                                   if key != 'manifest_digest'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        PersistentFastMemory.load(path, model, Tokenizer(), owner, representation=FACT_REPRESENTATION)
