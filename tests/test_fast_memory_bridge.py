"""Observation-to-fast-memory bridge and runtime integration (CPU)."""

import torch

from replica_lab.runtime.fast_memory import FastMemoryConfig, FastMemoryScope, FastMemorySession
from replica_lab.runtime.fast_memory_bridge import DeterministicTextEncoder, FastMemoryBridge


class CharTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return {'input_ids': [3 + ord(character) % 61 for character in text[:64]]}


def bridge():
    config = FastMemoryConfig(key_size=8, value_size=8, heads=2, learning_rate=0.2,
                              max_tokens_per_chunk=512)
    scope = FastMemoryScope('subject-1', 'e3', 'test', 'parallel_replica', 0)
    session = FastMemorySession(config, scope, producer='bridge-producer')
    return FastMemoryBridge(session, DeterministicTextEncoder(
        CharTokenizer(), key_size=8, value_size=8, max_tokens=64))


def test_bridge_stages_and_commits_actual_observation():
    value = bridge()
    assert value.last_commit is None
    receipt = value.observe('obs-1', 'The evaluator failed with code 1.',
                            source_hash='a' * 64)
    assert receipt is not None and receipt.committed
    assert value.observed == 1 and value.last_commit is receipt
    assert value.session.state.chunk_index == 1
    assert torch.isfinite(value.snapshot().weights).all()
    assert value.observe('obs-1', 'The evaluator failed with code 1.',
                         source_hash='a' * 64) is None


def test_runtime_observation_bridge_reaches_next_decision(tmp_path):
    from tests.e3_runtime_helpers import (make_case, make_runtime, read_action,
                                          run_test_action, ScriptedProposer)
    tools = make_case(tmp_path)
    memory = bridge()
    runtime = make_runtime(tools, tmp_path, fast_memory=memory,
                           max_prompt_tokens=512, generated_token_budget=32)
    proposer = ScriptedProposer([read_action(), run_test_action(),
                                 '{"kind": "stop"}'])
    summary = runtime.run(proposer, max_decisions=3)
    assert summary['fast_memory']['observed'] == 2
    assert proposer.requests[1].fast_state is not None   # read content was committed
    assert proposer.requests[2].fast_state is not None   # test receipt was committed
    assert summary['fast_memory']['observed'] == 2
    assert runtime.fast_state is not None


def test_bridge_image_reload_is_exact_and_scope_bound(tmp_path):
    first = bridge()
    first = FastMemoryBridge(first.session, first.encoder,
                             image_root=tmp_path / 'images')
    receipt = first.observe('obs-1', 'The evaluator failed with code 1.',
                            source_hash='a' * 64)
    assert receipt is not None
    image = tmp_path / 'images' / f'image-0-{receipt.chunk_index}'
    loaded = FastMemoryBridge.from_image(
        image, config=first.session.config, scope=first.session.scope,
        producer=first.session.producer, encoder=first.encoder,
        image_root=tmp_path / 'reloaded')
    assert loaded.snapshot().digest == first.snapshot().digest
    assert loaded.observed == 1
    assert loaded.observe('obs-1', 'The evaluator failed with code 1.',
                          source_hash='a' * 64) is None
    second = loaded.observe('obs-2', 'The evaluator passed.', source_hash='b' * 64)
    assert second is not None and second.committed
    assert loaded.session.state.chunk_index == 2
    other_scope = type(first.session.scope)('subject-1', 'other-branch', 'test',
                                            'parallel_replica', 0)
    try:
        FastMemoryBridge.from_image(image, config=first.session.config,
                                    scope=other_scope, producer=first.session.producer,
                                    encoder=first.encoder)
    except ValueError:
        pass
    else:
        raise AssertionError('cross-branch image must be rejected')


def test_runtime_accepts_reloaded_fast_image(tmp_path):
    from tests.e3_runtime_helpers import (make_case, make_runtime, read_action,
                                          ScriptedProposer)
    first = bridge()
    first = FastMemoryBridge(first.session, first.encoder,
                             image_root=tmp_path / 'images')
    receipt = first.observe('obs-1', 'The evaluator failed with code 1.',
                            source_hash='a' * 64)
    image = tmp_path / 'images' / f'image-0-{receipt.chunk_index}'
    loaded = FastMemoryBridge.from_image(
        image, config=first.session.config, scope=first.session.scope,
        producer=first.session.producer, encoder=first.encoder)
    tools = make_case(tmp_path)
    expected_digest = loaded.snapshot().digest
    runtime = make_runtime(tools, tmp_path, fast_memory=loaded,
                           max_prompt_tokens=512, generated_token_budget=32)
    proposer = ScriptedProposer([read_action(), '{"kind": "stop"}'])
    runtime.run(proposer, max_decisions=1)
    assert proposer.requests[0].fast_state is not None
    assert proposer.requests[0].fast_state.digest == expected_digest


def test_runtime_source_correction_invalidates_matching_fast_state_and_workspace(tmp_path):
    from tests.e3_runtime_helpers import make_case, make_runtime, workspace_for, ScriptedProposer, read_action
    tools = make_case(tmp_path)
    first = bridge()
    first = FastMemoryBridge(first.session, first.encoder,
                             image_root=tmp_path / 'images')
    receipt = first.observe('obs-1', 'The evaluator failed with code 1.',
                            source_hash='a' * 64)
    from replica_lab.runtime.workspace_store import WorkspaceStore
    store = WorkspaceStore(tmp_path / 'workspace_store')
    runtime = make_runtime(tools, tmp_path, fast_memory=first,
                           workspace_store=store,
                           max_prompt_tokens=512, generated_token_budget=32)
    request_probe = __import__('types').SimpleNamespace(task_digest=runtime.state.digest, revision=runtime.state.revision)
    runtime.previous_workspace = workspace_for(request_probe)
    # A nonmatching correction leaves the fast state intact but clears dependent workspace.
    nonmatch = runtime.invalidate_source('b' * 64, reason='unrelated correction')
    assert nonmatch['fast_invalidated'] is False
    assert runtime.fast_state.chunk_index == 1
    # A matching correction resets fast state and leaves no live workspace.
    request_probe = __import__('types').SimpleNamespace(task_digest=runtime.state.digest, revision=runtime.state.revision)
    runtime.previous_workspace = workspace_for(request_probe)
    result = runtime.invalidate_source('a' * 64, reason='source corrected')
    assert result['fast_invalidated'] is True
    assert runtime.fast_state.chunk_index == 0
    assert runtime.fast_state.generation == 1
    assert runtime.previous_workspace is None
    assert runtime.workspace_store.load() is None
    assert runtime.journal.events()[-1].kind == 'correction'
