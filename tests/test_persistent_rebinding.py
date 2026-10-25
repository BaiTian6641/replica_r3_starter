"""Source rebinding must authorize consumed lineage, not just refresh metadata.

A state whose consumed content is no longer authorized (revoked, superseded
without consuming the correction, unadmitted, or from a future causal view)
must never regain publication rights through bind_source() alone. Only
consuming the authoritative change or an explicit verified rebuild can.
"""
from dataclasses import replace

import pytest

from replica_lab.runtime.exact_memory import episode_records
from replica_lab.runtime.memory_representation import FACT_REPRESENTATION, RAW_REPRESENTATION
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
from test_exact_memory import Tokenizer, scope
from test_persistent_representation import sources, state_for

REPRESENTATIONS = [RAW_REPRESENTATION, FACT_REPRESENTATION]


def revoke_id(reference, episode, record_id):
    reference.revoke(record_id, authorized_subject='memory_reliability',
                     authorized_branch=episode['episode_id'])


def revoke_first(reference, episode, record):
    revoke_id(reference, episode, record.record_id)


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_revocation_blocks_metadata_only_rebind_until_verified_rebuild(sources, tmp_path, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[0])
    with pytest.raises(ValueError, match='stale'):
        memory.read()
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='stale'):
        memory.save(tmp_path / 'stale-contents', require_source_binding=True)
    assert not (tmp_path / 'stale-contents').exists()

    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    lineage = [event_id for event_id, _ in memory.lineage]
    assert initial[0].record_id not in lineage and len(lineage) == len(initial) - 1
    memory.bind_source(reference, scope=scope(episode))
    metadata = memory.save(tmp_path / 'rebuilt', require_source_binding=True)
    loaded = PersistentFastMemory.load(
        tmp_path / 'rebuilt', model, Tokenizer(), owner, representation=representation,
        require_source_binding=True, source_reference=reference, source_scope=scope(episode))
    assert loaded.read().state_digest == metadata['state_digest']
    assert [event_id for event_id, _ in loaded.lineage] == lineage


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_appending_after_revocation_cannot_dilute_stale_contents(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[1])
    survivor = next(record for record in initial if record.record_id != initial[1].record_id)
    extra = replace(survivor, record_id=f'{episode["episode_id"]}:extra',
                    text='The note for project Extra is 7.', event_at=99)
    assert reference.admit((extra,)) == 1
    memory.append_record(extra)
    assert memory.source_binding is None
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_correction_requires_consuming_the_superseding_record_in_order(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    _, correction = episode_records(episode)
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='absent from the authoritative view|never consumed'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))

    memory.append_record(correction)
    memory.bind_source(reference, scope=scope(episode, 'corrected'))
    assert correction.record_id in [event_id for event_id, _ in memory.lineage]

    reversed_state = PersistentFastMemory(model, Tokenizer(), owner, representation=representation)
    reversed_state.append_record(correction)
    reversed_state.append_record(next(record for record in initial
                                      if record.record_id == correction.supersedes))
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        reversed_state.bind_source(reference, scope=scope(episode, 'corrected'))


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_unaffected_records_stay_authorized_across_revocation_and_rebind(sources, tmp_path, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    memory.bind_source(reference, scope=scope(episode))
    revoke_first(reference, episode, initial[2])
    memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    unaffected = [record for record in initial if record.record_id != initial[2].record_id]
    assert sorted(event_id for event_id, _ in memory.lineage) == sorted(r.record_id for r in unaffected)
    memory.bind_source(reference, scope=scope(episode))
    memory.save(tmp_path / 'unaffected', require_source_binding=True)


@pytest.mark.parametrize('representation', REPRESENTATIONS)
def test_unadmitted_and_future_records_cannot_authorize_into_a_view(sources, representation):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, representation)
    _, correction = episode_records(episode)
    memory.append_record(correction)
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))
    memory.bind_source(reference, scope=scope(episode, 'corrected'))


def test_unbacked_revoked_correction_text_cannot_rebind(sources):
    model, reference, episode, owner = sources
    _, correction = episode_records(episode)
    memory = PersistentFastMemory(model, Tokenizer(), owner, representation=RAW_REPRESENTATION)
    memory.append('page0', episode['observations'][0])
    memory.bind_source(reference, scope=scope(episode))
    reference.correct_episode(episode)
    memory.append('page-correction', episode['correction_observation'])
    memory.bind_source(reference, scope=scope(episode, 'corrected'))
    revoke_id(reference, episode, correction.supersedes)
    with pytest.raises(ValueError, match='stale'):
        memory.read()
    with pytest.raises(ValueError, match='revoked source text'):
        memory.bind_source(reference, scope=scope(episode, 'corrected'))


def test_offline_inspection_then_late_append_rebinds_only_fresh_content(sources, tmp_path):
    model, reference, episode, owner = sources
    memory, initial = state_for(sources, RAW_REPRESENTATION)
    memory.bind_source(reference, scope=scope(episode))
    memory.save(tmp_path / 'bound', require_source_binding=True)
    inspected = PersistentFastMemory.load(tmp_path / 'bound', model, Tokenizer(), owner)
    assert inspected.source_binding is None
    with pytest.raises(ValueError, match='Source-bound'):
        inspected.save(tmp_path / 'unverified', require_source_binding=True)

    inspected.append('late', 'The note for project Extra is 7.')
    inspected.bind_source(reference, scope=scope(episode))
    inspected.save(tmp_path / 'late', require_source_binding=True)
    reloaded = PersistentFastMemory.load(
        tmp_path / 'late', model, Tokenizer(), owner, require_source_binding=True,
        source_reference=reference, source_scope=scope(episode))

    revoke_first(reference, episode, initial[0])
    with pytest.raises(ValueError, match='stale'):
        reloaded.read()
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        reloaded.bind_source(reference, scope=scope(episode))
    with pytest.raises(ValueError, match='stale'):
        PersistentFastMemory.load(
            tmp_path / 'late', model, Tokenizer(), owner, require_source_binding=True,
            source_reference=reference, source_scope=scope(episode))


def test_failed_rebuild_preserves_the_last_valid_state_and_binding(sources, monkeypatch):
    memory, initial = state_for(sources, FACT_REPRESENTATION)
    model, reference, episode, owner = sources
    memory.bind_source(reference, scope=scope(episode))
    before = memory.read().state_digest
    lineage, events = list(memory.lineage), list(memory.events)
    binding = dict(memory.source_binding)

    from replica_lab.runtime import persistent_fast_memory as module

    def interrupted(self, records=None, *, semantics, **kwargs):
        raise RuntimeError('injected rebuild failure')

    monkeypatch.setattr(module.PersistentFastMemory, 'rebuild', interrupted)
    revoke_first(reference, episode, initial[0])
    with pytest.raises(RuntimeError, match='injected'):
        memory.rebuild(semantics='current_view', reference=reference, source_scope=scope(episode))
    monkeypatch.undo()
    assert memory.lineage == lineage and memory.events == events
    assert memory.source_binding is None or memory.source_binding == binding
    with pytest.raises(ValueError, match='absent from the authoritative view'):
        memory.bind_source(reference, scope=scope(episode))


