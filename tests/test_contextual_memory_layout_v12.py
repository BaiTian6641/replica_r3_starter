"""v12 layout: ownership, live gradients, masks, one snapshot, complete checkpoint."""
from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import load_file

from replica_lab.contextual_feature_identity_v12 import ValueBatch
from replica_lab.contextual_memory import ContextualMemoryError, SLOTS, WIDTH
from replica_lab.contextual_memory_v5 import ContextualReadFailure
from replica_lab.contextual_v12_fixture import (
    SyntheticTokenizer, build_v12_stack, canonical_span, logits_digest,
    model_state_digest, open_v12_store, optimizer_digest, scope, spans_for)
from v12_fixtures import (
    backward_only, grad_state, grads_are_finite, nonzero_gradient_names, step,
    token_inputs)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(**kwargs):
    kwargs.setdefault("layers", 36)
    kwargs.setdefault("sites", (3, 7))
    return build_v12_stack(**kwargs)


def base_reference(adapter, ids):
    return adapter.forward_text(ids, torch.ones_like(ids), use_projector=False,
                                use_ports=False, use_spine=False)


def test_exact_optimizer_ownership_and_no_original_ownership():
    adapter, layout, provider, path = make()
    declared = path.assert_optimizer_ownership()
    assert set(declared) == {"producer", "order", "reader"}
    flattened = [name for names in declared.values() for name in names]
    assert len(flattened) == len(set(flattened))
    assert path.reader._optimizer_owner is path.optimizer
    owned = [p for group in path.optimizer.param_groups for p in group["params"]]
    original = {id(p) for p in adapter.base.parameters()}
    assert not original & {id(p) for p in owned}
    assert not any(p.requires_grad for p in adapter.base.parameters())
    extension_trainable = {id(p) for p in adapter.extension.parameters() if p.requires_grad}
    assert extension_trainable == {id(p) for p in path.reader.parameters() if p.requires_grad}
    assert not original & {id(p) for p in path.encoder.parameters()}
    assert not original & {id(p) for p in path.order.parameters()}
    assert adapter.base_unchanged()
    assert path.encoder.provider is provider
    assert provider.adapter is adapter


def test_partial_freezing_is_refused_by_the_ownership_audit():
    adapter, layout, provider, path = make()
    next(path.order.parameters()).requires_grad_(False)
    with pytest.raises(ContextualMemoryError):
        path.assert_optimizer_ownership()


def test_disabled_path_matches_the_base_exactly_and_skips_extension_work():
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    batch = path.training_values(spans, scope=sc)
    reference = base_reference(adapter, ids)

    def boom(*args, **kwargs):
        raise AssertionError("disabled path computed an extension branch")

    saved = {name: port.forward for name, port in path.reader.ports.items()}
    encoder_forward = path.encoder.forward
    forwards_before = provider.cost["forwards"]
    for name in saved:
        path.reader.ports[name].forward = boom
    path.encoder.forward = boom
    try:
        with torch.no_grad():
            explicit = path.forward(ids, mask, batch, scope=sc, enabled=False)
            absent = path.forward(ids, mask, None, scope=sc)
            empty_sites = path.forward(ids, mask, batch, scope=sc, active_sites=())
    finally:
        for name, value in saved.items():
            path.reader.ports[name].forward = value
        path.encoder.forward = encoder_forward
    assert torch.equal(explicit, reference)
    assert torch.equal(absent, reference)
    assert torch.equal(empty_sites, reference)
    assert path.cost["snapshot_copies"] == 0
    assert provider.cost["forwards"] == forwards_before


def test_step_one_has_zero_upstream_gradient_then_every_path_is_live():
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the bright badge is clearly red",
                       "the bright badge is clearly blue"])

    batch, logits, loss = backward_only(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    assert torch.isfinite(loss)
    assert grad_state(path.encoder) == {"zero": len(list(path.encoder.parameters()))}
    assert grad_state(path.order) == {"zero": 1}
    early = path.reader.ports["3"]
    assert nonzero_gradient_names(early) == {"output_projection.weight"}
    assert nonzero_gradient_names(path.reader.ports["7"]) == {"output_projection.weight"}
    assert batch.slots.grad is not None and not torch.count_nonzero(batch.slots.grad)

    path.optimizer.step()
    batch2, logits2, loss2 = backward_only(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    assert grads_are_finite(path.encoder) and grads_are_finite(path.reader)
    producer_nonzero = nonzero_gradient_names(path.encoder)
    assert "input_projection.weight" in producer_nonzero
    assert "slot_queries" in producer_nonzero
    assert "output.1.weight" in producer_nonzero
    assert nonzero_gradient_names(path.order) == {"weight"}
    assert batch2.slots.grad is not None and bool(torch.count_nonzero(batch2.slots.grad))
    for port in path.reader.ports.values():
        live = nonzero_gradient_names(port)
        assert "key_projection.weight" in live
        assert "value_projection.weight" in live
        assert "query_projection.weight" in live
        assert "output_projection.weight" in live
    assert adapter.base_unchanged()


@pytest.mark.parametrize("sites", [
    (3, 7),
    (3, 7, 11, 15, 19, 23, 27, 31, 35),
    tuple(range(36)),
])
def test_one_immutable_snapshot_serves_all_active_readers(sites):
    adapter, layout, provider, path = make(sites=sites)
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    batch = path.training_values(spans, scope=sc)
    before = dict(path.cost)
    with torch.no_grad():
        context = path.snapshot(ids, batch, scope=sc)
    delta = {key: path.cost[key] - before[key] for key in path.cost}
    assert delta["snapshot_copies"] == 1
    assert delta["value_slots_read"] == 8
    assert delta["retrieval_calls"] == 0 and delta["admission_calls"] == 0
    assert context.spec.sites == tuple(sites)
    assert context.values.shape == (ids.shape[0], ids.shape[1], 8, WIDTH)
    assert context.values.data_ptr() != batch.slots.data_ptr()
    assert path.cost["snapshot_copies"] == 1


def test_active_site_mask_disables_inactive_readers_and_keeps_early_site_live():
    adapter, layout, provider, path = make(sites=(3, 7))
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    _, _, _ = backward_only(path, adapter, spans, ids=ids, labels=labels, sc=sc,
                            active_sites=(3,))
    assert grad_state(path.reader.ports["7"]) == {"none": len(list(path.reader.ports["7"].parameters()))}
    assert "query_projection.weight" in nonzero_gradient_names(path.reader.ports["3"])
    with torch.no_grad():
        context = path.snapshot(ids, path.training_values(spans, scope=sc), scope=sc,
                                active_sites=(3,))
    assert context.spec.sites == (3,)


def test_empty_active_site_iterables_skip_memory_and_match_base():
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    batch = path.training_values(spans_for(["the badge is red"]), scope=sc)
    reference = base_reference(adapter, ids)
    snapshots = path.cost["snapshot_copies"]
    with torch.no_grad():
        for sites in ([], (), iter(()), []):
            actual = path.forward(ids, mask, batch, scope=sc, active_sites=sites)
            assert torch.equal(actual, reference)
    assert path.cost["snapshot_copies"] == snapshots
    with torch.no_grad():
        assert torch.equal(
            path.forward(ids, mask, batch, scope=sc, active_sites=[3]),
            path.forward(ids, mask, batch, scope=sc, active_sites=(3,)))
    with pytest.raises(ContextualMemoryError):
        path.snapshot(ids, batch, scope=sc, active_sites=[])


def test_padded_slots_are_masked_and_do_not_change_logits():
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    one = path.training_values(spans_for(["the badge is red"]), scope=sc)
    assert one.slots.shape == (SLOTS, WIDTH)
    with torch.no_grad():
        zero = path.snapshot(ids, one, scope=sc, padding_value=0.0)
        huge = path.snapshot(ids, one, scope=sc, padding_value=1e6)
        left = path.forward(ids, mask, one, scope=sc)
    values = path.snapshot(ids, one, scope=sc, padding_value=0.0)
    assert values.availability.shape == (ids.shape[0], ids.shape[1], 8)
    assert bool(values.availability[:, :, :SLOTS].all())
    assert not bool(values.availability[:, :, SLOTS:].any())
    assert torch.equal(left, path.forward(ids, mask, one, scope=sc))
    assert not torch.equal(zero.values, huge.values)
    two = path.training_values(spans_for(["the badge is red", "the badge is blue"]), scope=sc)
    assert two.slots.shape == (2 * SLOTS, WIDTH)
    with pytest.raises(ContextualMemoryError):
        path.training_values(spans_for(["a", "b", "c"]), scope=sc)


def test_controls_absent_zero_wrong_and_shuffled_are_distinct_after_updates():
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    correct_spans = spans_for(["the badge is red", "the badge is blue"])
    for _ in range(8):
        step(path, adapter, correct_spans, ids=ids, labels=labels, sc=sc)
    correct = path.training_values(correct_spans, scope=sc)
    wrong = path.training_values(spans_for(["the badge is green", "the badge is black"]), scope=sc)
    shuffled = path.training_values(tuple(reversed(correct_spans)), scope=sc)
    zero = ValueBatch(torch.zeros_like(correct.slots), correct.spans, correct.scope_key,
                      correct.producer_version, training=True)
    digests = {}
    with torch.no_grad():
        for name, batch in (("correct", correct), ("wrong", wrong), ("shuffled", shuffled),
                            ("zero", zero), ("absent", None)):
            digests[name] = logits_digest(path.forward(ids, mask, batch, scope=sc))
    assert len(set(digests.values())) == 5, digests
    assert torch.equal(path.forward(ids, mask, None, scope=sc), base_reference(adapter, ids))


def test_buffer_or_parameter_mutation_clears_c_and_blocks_export(tmp_path):
    adapter, layout, provider, path = make()
    span = canonical_span("the badge is red", record_id="record0")[0]
    provider.features(span)
    assert provider.cost["cache_bytes"] > 0
    adapter.base.register_buffer("review_probe_buffer", torch.zeros(1))
    with torch.no_grad():
        adapter.base.review_probe_buffer.add_(1)
    with pytest.raises(ContextualMemoryError):
        provider.validate_original()
    assert provider.cost["cache_bytes"] == 0
    with pytest.raises(ContextualMemoryError):
        path.save(tmp_path / "buffer_checkpoint")
    assert not (tmp_path / "buffer_checkpoint").exists()

    adapter2, layout2, provider2, path2 = make()
    provider2.features(span)
    with torch.no_grad():
        next(adapter2.base.parameters()).add_(0.5)
    with pytest.raises(ContextualMemoryError):
        path2.save(tmp_path / "parameter_checkpoint")
    assert not (tmp_path / "parameter_checkpoint").exists()


def test_checkpoint_roundtrip_restores_logits_moments_and_no_base_or_cache(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    provider.features(spans[0])  # populate bounded C, which must not be serialized
    manifest = path.save(tmp_path / "checkpoint")
    assert manifest["contains_base_weights"] is False
    assert manifest["feature_cache_serialized"] is False
    assert manifest["task_state_serialized"] is False
    assert manifest["original_weights_serialized"] is False
    assert manifest["projection_cache"] == "none"
    files = {p.name for p in (tmp_path / "checkpoint").iterdir() if p.is_file()}
    assert files == {"manifest.json"}
    assert (tmp_path / "checkpoint" / "components").is_dir()
    components = {p.name for p in (tmp_path / "checkpoint" / "components").iterdir()
                  if p.is_file()}
    assert components == {"extension.safetensors", "optimizer.pt", "manifest.json"}
    tensors = load_file(str(tmp_path / "checkpoint" / "components" / "extension.safetensors"))
    assert set(tensors) == set(path._tensors())
    assert all(name.startswith(("producer.", "order.", "reader.")) for name in tensors)
    assert not any("embedding" in name or name.startswith("base.") for name in tensors)

    adapter2, layout2, provider2, path2 = make()
    path2.load(tmp_path / "checkpoint")
    assert provider2.cost["cache_bytes"] == 0
    assert provider2._cache == {} and provider2._tokens == {}
    component_manifest = json.loads(
        (tmp_path / "checkpoint" / "components" / "manifest.json").read_text())
    assert component_manifest["feature_cache_serialized"] is False
    assert component_manifest["task_state_serialized"] is False
    with torch.no_grad():
        expected = path.forward(ids, mask, path.training_values(spans, scope=sc), scope=sc)
    with torch.no_grad():
        restored = path2.forward(ids, mask, path2.training_values(spans, scope=sc), scope=sc)
    assert torch.equal(expected, restored)
    assert model_state_digest(path) == model_state_digest(path2)
    assert optimizer_digest(path.optimizer) == optimizer_digest(path2.optimizer)


def test_checkpoint_refuses_other_contracts_and_tampering_before_mutation(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red"])
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    path.save(tmp_path / "checkpoint")

    _, _, _, other = make(tokenizer=SyntheticTokenizer(adapter.config.geometry.vocab_size,
                                                       mode="words"))
    before = {name: value.clone() for name, value in other._tensors().items()}
    with pytest.raises(ContextualMemoryError):
        other.load(tmp_path / "checkpoint")
    assert all(torch.equal(value, other._tensors()[name]) for name, value in before.items())

    _, _, _, different_origin = make(seed=18)
    with pytest.raises(ContextualMemoryError):
        different_origin.load(tmp_path / "checkpoint")

    manifest_path = tmp_path / "checkpoint" / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    manifest = json.loads(json.dumps(original_manifest))
    manifest["feature_contract"] = {"format": "different"}
    manifest_path.write_text(json.dumps(manifest))
    _, _, _, tampered = make()
    with pytest.raises(ContextualMemoryError):
        tampered.load(tmp_path / "checkpoint")

    manifest_path.write_text(json.dumps(original_manifest))
    weights = tmp_path / "checkpoint" / "components" / "extension.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(bytes(payload))
    _, _, _, corrupt = make()
    with pytest.raises(ContextualMemoryError):
        corrupt.load(tmp_path / "checkpoint")


def test_training_from_stored_reencodes_a_live_graph_and_refuses_stale_payload(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    step(path, adapter, spans_for(["the badge is red"]), ids=ids, labels=labels, sc=sc)
    span = canonical_span("the badge is red", record_id="record0")[0]
    store = open_v12_store(tmp_path)
    path.admit(store, span, scope=sc)
    durable = path.read(store, "badge", scope=sc)
    fresh = path.training_from_stored(durable, scope=sc)
    assert fresh.training and fresh.slots.requires_grad
    logits = path.forward(ids, mask, fresh, scope=sc)
    loss = adapter.causal_loss(logits, ids, labels)
    loss.backward()
    assert "input_projection.weight" in nonzero_gradient_names(path.encoder)
    tampered = replace(durable, slots=durable.slots + 1.0)
    with pytest.raises(ContextualMemoryError, match="payload"):
        path.training_from_stored(tampered, scope=sc)
    store.close()


def test_exact_fallback_is_separately_charged_and_never_replaces_a_failure(tmp_path):
    adapter, layout, provider, path = make()
    sc = scope()
    span = canonical_span("the badge is red", record_id="record0")[0]
    store = open_v12_store(tmp_path)
    path.admit(store, span, scope=sc)
    batch = path.read(store, "badge", scope=sc)
    before = dict(path.cost)
    text = path.exact_fallback(batch, scope=sc, tokenizer=provider.tokenizer)
    assert text == span.text
    assert path.cost["fallback_calls"] - before["fallback_calls"] == 1
    assert path.cost["fallback_bytes"] - before["fallback_bytes"] == len(span.text.encode())
    assert path.cost["fallback_tokens"] - before["fallback_tokens"] > 0
    before = dict(path.cost)
    with pytest.raises(ContextualReadFailure):
        path.read(store, "badge", scope=scope(as_of=0))
    assert path.cost["fallback_calls"] == before["fallback_calls"]
    store.close()


def test_frozen_readout_saves_and_resumes_moments_without_a_base_export(tmp_path):
    adapter, layout, provider, path = make()
    ids, mask, labels = token_inputs()
    sc = scope()
    spans = spans_for(["the badge is red"])
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    moments = optimizer_digest(path.optimizer)
    path.freeze_readout()
    assert not any(p.requires_grad for module in path.modules().values()
                   for p in module.parameters())
    manifest = path.save(tmp_path / "frozen")
    assert manifest["fitting_frozen"] is True
    _, _, _, resumed = make()
    resumed.load(tmp_path / "frozen")
    assert not any(p.requires_grad for module in resumed.modules().values()
                   for p in module.parameters())
    assert optimizer_digest(resumed.optimizer) == moments
    resumed.resume_training()
    assert all(p.requires_grad for module in resumed.modules().values()
               for p in module.parameters())
