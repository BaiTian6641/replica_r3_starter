"""Connected E1/E2 host/model contracts on an offline, randomly initialized CPU Llama.

These small software checks establish ownership and gradient/cache behavior;
they do not measure pretrained MiniCPM quality or substitute for the MPS run.
"""

from dataclasses import replace
import hashlib

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.architecture.contracts import ComponentManifest, PendingWrite, PortSpec, TaskGraph
from replica_lab.architecture.minicpm5 import MiniCPM5Adapter, native_abi_digest
from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
from replica_lab.memory_system.session import MemorySession


class CharacterTokenizer:
    """Exact offsets and stable toy token IDs; never accesses a model registry."""

    is_fast = True

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [3 + ord(character) % 61 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def tiny_adapter(regime="modules_only"):
    with torch.random.fork_rng():
        torch.manual_seed(260)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128,
                             attention_dropout=0.0, tie_word_embeddings=False)
        config._attn_implementation = "eager"
        base = LlamaForCausalLM(config)
        ports = PortSpec(injection_layers=(0, 2), observer_layer=3, source_layer=0,
                         memory_width=16, slots_per_unit=2, max_units=4,
                         source_tokens=8, heads=4)
        manifest = ComponentManifest(
            "random-tiny-fixture", "1" * 64, "2" * 64, "3" * 64,
            native_abi_digest(), ports)
        adapter = MiniCPM5Adapter(base, manifest, regime)
    adapter.eval()
    adapter.seal_original_identity()
    return adapter


def source(record_id="r1", text="blue cat", observed_at=1, **changes):
    values = dict(record_id=record_id, subject_id="fixture-person", branch_id="source",
                  text=text, observed_at=observed_at, event_at=0, purpose="test",
                  source_id="synthetic-fixture", authenticity="source_report")
    values.update(changes)
    return MemoryRecord(**values)


def admit(ledger, record):
    return ledger.admit(record, authorized_subject=record.subject_id,
                        authorized_branch=record.branch_id, source_authorized=True)


def scope(**changes):
    values = dict(subject_id="fixture-person", branch_id="replica-a", purpose="test", as_of=20)
    values.update(changes)
    return EligibilityScope(**values)


def prepare_args(**changes):
    values = dict(prompt_ids=torch.tensor([[1, 4, 6]], dtype=torch.long),
                  prompt_mask=torch.ones(1, 3, dtype=torch.bool), query="blue",
                  scope=scope(), task=TaskGraph("tiny-task", "read the admitted source"))
    values.update(changes)
    return values


def action_batch():
    return {"input_ids": torch.tensor([[1, 4, 6, 12, 15]], dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "labels": torch.tensor([[-100, -100, -100, 12, 15]], dtype=torch.long)}


def open_ports(adapter):
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(261)
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_(0.0, 0.03)


@pytest.fixture
def owned_session(tmp_path):
    ledger = RecordLedger(tmp_path / "records.sqlite")
    adapter = tiny_adapter()
    session = MemorySession(adapter, CharacterTokenizer(), ledger,
                            cache_dir=tmp_path / "latents")
    yield session
    session.ledger.close()


def test_prepare_forward_exact_source_and_released_ownership(owned_session):
    session = owned_session
    record = source(text="blue 🙂")
    admit(session.ledger, record)
    request = session.prepare(**prepare_args())
    session.verify(request)
    assert request.frame.memory.slots.shape == (1, 1, 2, 16)
    assert request.frame.memory.spans[0].record_hash == record.record_hash
    assert torch.equal(request.frame.sense.prompt_ids, prepare_args()["prompt_ids"])
    result = session.forward(request, **action_batch())
    assert result.logits.shape == (1, 5, 64)
    assert bool(torch.isfinite(result.loss))
    span = request.frame.memory.spans[0]
    exact = session.exact_text(request, span)
    assert exact.text == record.text
    assert exact.span_hash == hashlib.sha256(record.text.encode()).hexdigest()
    with pytest.raises(PermissionError, match="selected memory"):
        session.exact_text(request, replace(span, char_end=span.char_end - 1))
    request.release()
    assert request.frame is None and request.selection is None and request.native_cache is None
    assert request.state == "released"
    with pytest.raises(RuntimeError, match="released"):
        request.assert_ready()


@pytest.mark.parametrize("regime", ["modules_only", "full_augmented"])
def test_training_request_keeps_source_writer_graph_through_backward_and_releases(tmp_path, regime):
    with RecordLedger(tmp_path / "records.sqlite") as ledger:
        admit(ledger, source())
        adapter = tiny_adapter(regime)
        open_ports(adapter)
        session = MemorySession(adapter, CharacterTokenizer(), ledger, cache_dir=tmp_path / "cache")
        optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=0.001)
        with session.training_request(**prepare_args()) as request:
            assert request.state == "training"
            assert request.frame.memory.slots.requires_grad
            assert not tuple((tmp_path / "cache").glob("*.safetensors"))
            result = session.forward(request, **action_batch())
            session.backward(request, result.loss)
            writer_grads = [p.grad for p in adapter.record_writer.parameters()]
            assert any(g is not None and bool(g.ne(0).any()) for g in writer_grads)
            for port in adapter.memory_ports.values():
                assert port.output_projection.weight.grad is not None
                assert bool(port.output_projection.weight.grad.ne(0).any())
            originals = tuple(adapter.base.parameters())
            if regime == "modules_only":
                assert all(p.grad is None for p in originals)
            else:
                assert adapter.base.get_input_embeddings().weight.grad is not None
                assert bool(adapter.base.get_input_embeddings().weight.grad.ne(0).any())
            assert session._training_leases == {request.request_id}
        assert request.state == "released" and request.frame is None and request.selection is None
        assert not session._training_leases
        optimizer.step()  # publication/update is after the training frame is released
        with session.training_request(**prepare_args()) as following:
            session.backward(following, session.forward(following, **action_batch()).loss)


def test_training_lease_blocks_observation_commit_and_releases_after_exception(owned_session):
    session = owned_session
    admit(session.ledger, source())
    pending = PendingWrite("next", source("next", "blue dog", 2), session.ledger.epoch)
    with pytest.raises(RuntimeError, match="fixture abort"):
        with session.training_request(**prepare_args()) as request:
            with pytest.raises(RuntimeError, match="Finish backward"):
                session.commit_observation(pending, authorized_subject="fixture-person",
                                           authorized_branch="source", source_authorized=True)
            raise RuntimeError("fixture abort")
    assert request.frame is None and request.state == "released"
    assert not session._training_leases
    assert session.commit_observation(pending, authorized_subject="fixture-person",
                                      authorized_branch="source", source_authorized=True)


def test_correction_invalidates_old_frame_exact_fetch_and_cached_payload_after_restart(owned_session):
    session = owned_session
    admit(session.ledger, source())
    old = session.prepare(**prepare_args())
    old_epoch = old.selection.epoch
    old_files = set(session.cache_dir.glob("*.safetensors"))
    assert old_files
    corrected = source("r2", "blue dog", 5, supersedes="r1")
    admit(session.ledger, corrected)
    with pytest.raises(ValueError, match="stale"):
        session.verify(old)
    assert old.state == "invalid" and old.frame is None and old.selection is None
    assert not any(path.exists() for path in old_files)
    path = session.ledger.path
    session.ledger.close()
    session.ledger = RecordLedger(path)
    session.index.ledger = session.ledger
    request = session.prepare(**prepare_args())
    assert request.selection.epoch > old_epoch
    assert tuple(record.record_id for record in request.selection.records) == ("r2",)
    assert session.exact_text(request, request.frame.memory.spans[0]).text == corrected.text


def test_revocation_clears_owned_frame_and_next_request_has_no_evidence(owned_session):
    session = owned_session
    admit(session.ledger, source())
    request = session.prepare(**prepare_args())
    session.ledger.revoke("r1", authorized_subject="fixture-person",
                          authorized_branch="source", source_authorized=True)
    with pytest.raises(ValueError, match="stale"):
        session.forward(request, **action_batch())
    assert request.frame is None and request.selection is None
    next_request = session.prepare(**prepare_args())
    assert next_request.frame.memory.slots.shape == (1, 0, 2, 16)
    assert next_request.frame.memory.spans == ()


def test_scope_filtering_precedes_encoding_and_selection_cannot_change_scope(owned_session):
    session = owned_session
    for record in (source(), source("a", branch_id="replica-a"),
                   source("b", branch_id="replica-b"),
                   source("foreign", subject_id="other"), source("private", purpose="private")):
        admit(session.ledger, record)
    request = session.prepare(**prepare_args())
    assert {span.record_id for span in request.frame.memory.spans} == {"r1", "a"}
    assert {record.record_id for record in request.selection.records} == {"r1", "a"}
    foreign = session.index.retrieve("blue", subject_id="other", branch_id="replica-a",
                                      purpose="test", as_of=20)
    with pytest.raises(PermissionError, match="scope"):
        session.prepare(**prepare_args(selection=foreign))


def test_request_a_b_a_has_independent_evidence_tensors_and_outputs(owned_session):
    session = owned_session
    admit(session.ledger, source("cat", "blue cat"))
    admit(session.ledger, source("dog", "blue dog"))
    open_ports(session.adapter)
    first = session.prepare(**prepare_args(query="cat"))
    second = session.prepare(**prepare_args(query="dog"))
    assert first.frame.memory.slots.data_ptr() != second.frame.memory.slots.data_ptr()
    with torch.no_grad():
        first_logits = session.forward(first, **action_batch()).logits
        second_logits = session.forward(second, **action_batch()).logits
        first_again = session.forward(first, **action_batch()).logits
    torch.testing.assert_close(first_logits, first_again, rtol=0, atol=0)
    assert bool((first_logits - second_logits).abs().max() > 1e-6)
    second.release()
    session.verify(first)


def test_frozen_feature_cache_reuses_native_features_but_reexecutes_changed_writer(owned_session):
    session = owned_session
    admit(session.ledger, source())
    first = session.prepare(training=True, **prepare_args())
    previous = first.frame.memory.slots.detach().clone()
    assert session.cache_stats["feature_misses"] == 1
    first.release()
    with torch.no_grad():
        session.adapter.record_writer.encoder.output_norm.bias.add_(0.125)
    second = session.prepare(training=True, **prepare_args())
    assert session.cache_stats["feature_hits"] == 1
    assert session.cache_stats["feature_misses"] == 1
    assert second.frame.memory.slots.requires_grad
    torch.testing.assert_close(second.frame.memory.slots, previous + 0.125)
    assert session.cache_stats["latent_hits"] == 0
    second.release()


def test_trainable_native_producer_recomputes_source_features_each_request(tmp_path):
    with RecordLedger(tmp_path / "records.sqlite") as ledger:
        admit(ledger, source())
        adapter = tiny_adapter("full_augmented")
        session = MemorySession(adapter, CharacterTokenizer(), ledger)
        for _ in range(2):
            request = session.prepare(training=True, **prepare_args())
            assert request.frame.memory.slots.requires_grad
            request.release()
        assert session.cache_stats["feature_hits"] == 0
        assert session.cache_stats["feature_misses"] == 2
        assert not session._features


def test_persistent_latents_reuse_after_process_style_reconstruction_and_writer_change(owned_session):
    session = owned_session
    admit(session.ledger, source())
    request = session.prepare(**prepare_args())
    expected = request.frame.memory.slots.clone()
    assert session.cache_stats["latent_misses"] == 1
    request.release()
    restored = MemorySession(tiny_adapter(), CharacterTokenizer(), session.ledger,
                             cache_dir=session.cache_dir)
    reused = restored.prepare(**prepare_args())
    assert restored.cache_stats["latent_hits"] == 1
    assert restored.cache_stats["feature_misses"] == 0
    torch.testing.assert_close(reused.frame.memory.slots, expected, rtol=0, atol=0)
    reused.release()
    with torch.no_grad():
        restored.adapter.record_writer.encoder.output_norm.bias.add_(0.25)
    changed = restored.prepare(**prepare_args())
    assert restored.cache_stats["latent_hits"] == 1
    assert restored.cache_stats["latent_misses"] == 1
    torch.testing.assert_close(changed.frame.memory.slots, expected + 0.25)


def test_persistent_latent_payload_tamper_fails_before_reuse(owned_session):
    session = owned_session
    admit(session.ledger, source())
    request = session.prepare(**prepare_args())
    request.release()
    path = next(session.cache_dir.glob("*.safetensors"))
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="cache identity/content"):
        session.prepare(**prepare_args())


def test_all_source_tokens_preserved_as_bounded_chunks_and_candidate_overflow_fails(owned_session):
    session = owned_session
    record = source(text="blue🙂 café 123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    admit(session.ledger, record)
    spans = session.spans((record,))
    assert len(spans) > 1
    assert all(0 < len(span.token_ids) <= 8 for span in spans)
    assert tuple(token for span in spans for token in span.token_ids) == tuple(
        session.tokenizer(record.text)["input_ids"])
    assert spans[0].token_start == 0 and spans[-1].token_end == len(record.text)
    assert all(left.token_end == right.token_start for left, right in zip(spans, spans[1:]))
    assert "".join(record.text[span.char_start:span.char_end] for span in spans) == record.text
    small = MemorySession(session.adapter, session.tokenizer, session.ledger, max_candidate_units=4)
    with pytest.raises(ValueError, match="never silently truncate"):
        small.prepare(**prepare_args())


def test_native_parameter_change_invalidates_frame_and_frozen_feature_cache(owned_session):
    session = owned_session
    admit(session.ledger, source())
    first = session.prepare(training=True, **prepare_args())
    original_features = tuple(session._features.values())[0].clone()
    with torch.no_grad():
        session.adapter.base.get_input_embeddings().weight.add_(0.01)
    with pytest.raises(RuntimeError, match="changed"):
        session.verify(first)
    assert first.frame is None
    assert not session._features
    second = session.prepare(training=True, **prepare_args())
    assert session.cache_stats["feature_hits"] == 0
    assert session.cache_stats["feature_misses"] == 2
    assert not torch.equal(tuple(session._features.values())[0], original_features)
    second.release()


def test_ranking_supervision_trains_reranker_writer_and_optional_sensor_without_target_encoding(
        owned_session, monkeypatch):
    session = owned_session
    cat, dog = source("cat", "blue cat"), source("dog", "blue dog")
    admit(session.ledger, cat)
    admit(session.ledger, dog)
    source_calls = []
    original_encode = session.adapter.encode_observations

    def observed_encode(input_ids, valid):
        source_calls.append(tuple(input_ids[0].tolist()))
        return original_encode(input_ids, valid)

    monkeypatch.setattr(session.adapter, "encode_observations", observed_encode)
    with session.training_request(train_sense=True, **prepare_args()) as request:
        assert set(source_calls) == {
            tuple(session.tokenizer(cat.text)["input_ids"]),
            tuple(session.tokenizer(dog.text)["input_ids"]),
        }
        assert all(summary.requires_grad for _, summary in request.frame.sense.summaries)
        source_count = len(source_calls)
        loss = session.ranking_loss(request, preferred_record_id="cat")
        assert loss.requires_grad and bool(torch.isfinite(loss))
        session.backward(request, loss)
        assert len(source_calls) == source_count  # targets never enter source encoding
        for module in (session.adapter.reranker, session.adapter.record_writer,
                       session.adapter.sense_ports["0"]):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
            assert any(bool(g.ne(0).any()) for g in gradients)
        assert all(p.grad is None for p in session.adapter.base.parameters())


@pytest.mark.parametrize("include_distractor, preferred", [(False, "cat"), (True, "absent")])
def test_ranking_requires_eligible_target_and_a_different_record(
        owned_session, include_distractor, preferred):
    session = owned_session
    admit(session.ledger, source("cat", "blue cat"))
    if include_distractor:
        admit(session.ledger, source("dog", "blue dog"))
    with session.training_request(**prepare_args()) as request:
        assert not any(summary.requires_grad for _, summary in request.frame.sense.summaries)
        with pytest.raises(ValueError, match="eligible preferred record and a distractor"):
            session.ranking_loss(request, preferred_record_id=preferred)


def test_ranking_backward_hook_never_enters_host_storage(owned_session, monkeypatch):
    session = owned_session
    admit(session.ledger, source("r1", "blue cat"))
    admit(session.ledger, source("r2", "blue dog"))
    request = session.prepare(**prepare_args(), training=True, train_sense=True)
    loss = session.ranking_loss(request, "r1")
    def forbidden(*args, **kwargs):
        raise AssertionError("autograd hooks must not access thread-owned ledger state")
    monkeypatch.setattr(session, "verify", forbidden)
    loss.backward()  # host verify happens in session.backward, outside this hook
    assert any(p.grad is not None for p in session.adapter.reranker.parameters())
    request.release()
