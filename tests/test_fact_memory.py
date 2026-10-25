"""Observed-fact key/value representation and persistence tests."""
import hashlib

import torch

from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import FastMemorySpec
from replica_lab.memory_reliability import build_episodes
from replica_lab.runtime.fact_memory import (binding_diagnostic, encode_fact_binding,
                                              fact_source_text, validate_fact)
from replica_lab.runtime.fast_memory import FastMemoryScope
from replica_lab.runtime.persistent_fast_memory import PersistentFastMemory
from replica_lab.runtime.memory_representation import FACT_REPRESENTATION

SPEC = FastMemorySpec(heads=2, key_size=4, value_size=4, update_rule='sgdn2',
                      depth_mode='shared', depth_writers=True)
SCOPE = FastMemoryScope('memory_reliability', 'fact-test', 'reliability', 'parallel_replica', 0)


class Tokenizer:
    eos_token_id = 63

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False, **kwargs):
        result = {'input_ids': [1 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:8]]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index + 1) for index in range(len(result['input_ids']))]
        return result

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def test_binding_diagnostic_separates_key_and_value_paths():
    model = tiny_adapter(fast_memory=SPEC).eval()
    facts = build_episodes()[0]['facts']
    layer = model.ports.injection_layers[0]
    result = binding_diagnostic(model, Tokenizer(), facts, layer=layer)
    assert result['labels_used'] is False and result['parameters_updated'] is False
    assert result['value_invariant_under_key_change']
    assert result['key_invariant_under_value_change']
    assert result['key_change_norm'] > 0 and result['value_change_norm'] > 0
    assert result['write_shape'][1] == 1


def test_fact_binding_preserves_writer_gradients():
    model = tiny_adapter(fast_memory=SPEC).train()
    fact = build_episodes()[0]['facts'][0]
    layer = model.ports.injection_layers[0]
    write = encode_fact_binding(model, Tokenizer(), fact, layer=layer)
    (write.keys.square().mean() + write.values.square().mean()).backward()
    writer = model.fast_depth_write_encoders[str(layer)]
    assert any(parameter.grad is not None and bool(parameter.grad.ne(0).any())
               for parameter in writer.parameters())


def test_persistent_fact_append_uses_canonical_observed_lineage():
    model = tiny_adapter(fast_memory=SPEC).eval()
    memory = PersistentFastMemory(model, Tokenizer(), SCOPE, representation=FACT_REPRESENTATION)
    episode = build_episodes()[0]
    fact = episode['facts'][0]
    correction = episode['correction']
    memory.append_fact('fact-0', fact)
    memory.append_fact('correction', correction, correction=True)
    assert [event for event, _ in memory.lineage] == ['fact-0', 'correction']
    assert memory.lineage[0][1] == hashlib.sha256(fact_source_text(fact).encode()).hexdigest()
    assert memory.lineage[1][1] == hashlib.sha256(
        fact_source_text(correction, correction=True).encode()).hexdigest()
    assert memory.state.chunk_index == 2
    assert bool(torch.isfinite(memory.read().weights).all())


def test_fact_schema_rejects_answer_labels():
    fact = dict(build_episodes()[0]['facts'][0], expected_answer='label')
    try:
        validate_fact(fact)
    except ValueError as error:
        assert 'exactly' in str(error)
    else:
        raise AssertionError('Answer-labelled fact was accepted')


def test_runner_fact_representation_tracks_lineage_and_controls(tmp_path):
    from replica_lab.memory_system import RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.exact_memory import ExactMemoryReference
    from test_exact_memory import _runner
    run = _runner()
    episodes = build_episodes()
    pair = [e for e in episodes if e['family_id'] == episodes[0]['family_id']]
    model = tiny_adapter(fast_memory=SPEC).eval()
    model.seal_original_identity()
    ledger = RecordLedger(tmp_path / 'facts.sqlite')
    session = MemorySession(model, Tokenizer(), ledger, max_candidate_units=4)
    reference = ExactMemoryReference(ledger)
    rows, transitions = run.evaluate(
        session, pair, ('fast', 'disabled', 'last_chunk', 'full_context', 'selective_context'),
        tmp_path / 'images', reference=reference, gate=True,
        writer_representation=FACT_REPRESENTATION)
    assert len(rows) == 2 * 12 * 5
    assert all(t['initial_cursor'] == 6 and t['corrected_cursor'] == 7 for t in transitions)
    assert all(t['lineage_prefix_preserved'] for t in transitions)
    assert any(row['arm'] == 'selective_context' and row['route'] == 'exact_context' for row in rows)
    ledger.close()
