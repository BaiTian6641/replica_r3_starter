"""Offline native E1/E2 integration contracts; no downloaded model or GPU use."""

from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import load_file

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import RequestContext, TaskGraph


def native_logits(adapter, ids, mask):
    # The adapter deliberately uses native generation's left-padding positions.
    # Ordinary native forward otherwise defaults to a single arange for rows.
    positions = mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(~mask.bool(), 1)
    return adapter.base(input_ids=ids, attention_mask=mask, position_ids=positions,
                        use_cache=False).logits


@pytest.mark.parametrize("left_padding", [False, True])
def test_zero_ports_preserve_native_logits_and_response_loss(left_padding):
    adapter = tiny_adapter().eval()
    prefix = torch.tensor([[0, 0, 1, 2, 3]]) if left_padding else torch.tensor([[1, 2, 3]])
    valid = prefix.ne(0)
    with torch.no_grad():
        frame = synthetic_frame(adapter, prompt_ids=prefix, prompt_mask=valid)
        batch = action_batch(frame)
        output = adapter.forward_with_frame(frame=frame, **batch)
        expected = native_logits(adapter, batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(output.logits, expected, atol=0, rtol=0)
    expected_loss = adapter.base.loss_function(
        logits=expected, labels=batch["labels"], vocab_size=adapter.base.config.vocab_size)
    torch.testing.assert_close(output.loss, expected_loss, atol=0, rtol=0)
    assert output.past_key_values is None


def test_source_prefix_matches_complete_native_block_and_known_sense_features():
    adapter = tiny_adapter().eval()
    ids = torch.tensor([[0, 7, 8, 9], [10, 11, 12, 13]])
    valid = ids.ne(0)
    positions = valid.long().cumsum(-1) - 1
    positions = positions.masked_fill(~valid, 1)
    captured = {}
    def capture_last_complete_block(module, args, output):
        captured["last"] = output.detach().clone()
    handle = adapter.base.model.layers[3].register_forward_hook(capture_last_complete_block)
    with torch.no_grad():
        try:
            native = adapter.base.model(input_ids=ids, attention_mask=valid,
                                        position_ids=positions, output_hidden_states=True,
                                        use_cache=False)
        finally:
            handle.remove()
        source = adapter.encode_observations(ids, valid)
        sense = adapter.sense(ids, valid)
    torch.testing.assert_close(source, native.hidden_states[1], atol=0, rtol=0)
    assert tuple(layer for layer, _ in sense.summaries) == (0, 2, 3)
    for layer, summary in sense.summaries:
        # The final tuple entry includes norm, while the observer taps the
        # complete decoder block before the separate final normalization.
        features = captured["last"] if layer == 3 else native.hidden_states[layer + 1]
        expected, _ = adapter.sense_ports[str(layer)](features, valid)
        torch.testing.assert_close(summary, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Chunk source records"):
        adapter.encode_observations(torch.ones(1, 17, dtype=torch.long), torch.ones(1, 17, dtype=torch.bool))


def test_each_nonzero_port_changes_the_native_path_and_disabled_ports_bypass():
    adapter = tiny_adapter().eval()
    with torch.no_grad():
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_(std=0.1)
        outputs = {}
        for enabled in ((), (0,), (2,), (0, 2)):
            frame = synthetic_frame(adapter, enabled_ports=enabled)
            batch = action_batch(frame)
            outputs[enabled] = adapter.forward_with_frame(frame=frame, **batch).logits
        expected = native_logits(adapter, batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(outputs[()], expected, atol=0, rtol=0)
    for enabled in ((0,), (2,), (0, 2)):
        assert torch.isfinite(outputs[enabled]).all()
        assert not torch.equal(outputs[enabled], expected)
    assert not torch.equal(outputs[(0,)], outputs[(2,)])
    assert not torch.equal(outputs[(0, 2)], outputs[(0,)])
    assert not torch.equal(outputs[(0, 2)], outputs[(2,)])


@pytest.mark.parametrize("empty,all_masked", [(True, False), (False, True)])
def test_nonzero_ports_with_unavailable_memory_are_exactly_native(empty, all_masked):
    adapter = tiny_adapter().eval()
    with torch.no_grad():
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_()
        frame = synthetic_frame(adapter, empty=empty, all_masked=all_masked)
        batch = action_batch(frame)
        actual = adapter.forward_with_frame(frame=frame, **batch).logits
        expected = native_logits(adapter, batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("target", ["slots", "valid", "prompt_ids", "prompt_mask", "summary"])
def test_owned_frame_detects_mutated_latents_masks_and_sense(target):
    adapter = tiny_adapter().eval()
    frame = synthetic_frame(adapter)
    batch = action_batch(frame)
    tensor = {
        "slots": frame.memory.slots, "valid": frame.memory.valid,
        "prompt_ids": frame.sense.prompt_ids, "prompt_mask": frame.sense.prompt_mask,
        "summary": frame.sense.summaries[0][1],
    }[target]
    with torch.no_grad():
        if tensor.dtype == torch.bool:
            # Writer masks can be expanded views; one scalar write still
            # mutates their tracked storage without an overlapping bulk op.
            index = (0,) * tensor.ndim
            tensor[index] = ~tensor[index]
        else:
            tensor.add_(1)
    with pytest.raises(RuntimeError, match="DecisionFrame tensor mutated"):
        adapter.forward_with_frame(frame=frame, **batch)


@pytest.mark.parametrize("owner", ["native", "extension"])
def test_parameter_update_invalidates_existing_decision_frame(owner):
    adapter = tiny_adapter().eval()
    frame = synthetic_frame(adapter)
    parameter = (next(adapter.base.parameters()) if owner == "native"
                 else adapter.memory_ports["0"].output_projection.weight)
    with torch.no_grad():
        parameter.add_(0.01)
    with pytest.raises(RuntimeError, match="changed; rebuild"):
        adapter.forward_with_frame(frame=frame, **action_batch(frame))


@pytest.mark.parametrize("target", ["rotary_buffer", "config", "attention_implementation", "rotary_scale"])
def test_native_buffer_and_configuration_changes_invalidate_frames_and_exports(tmp_path, target):
    adapter = tiny_adapter().eval()
    adapter.seal_original_identity()
    frame = synthetic_frame(adapter)
    if target == "rotary_buffer":
        with torch.no_grad():
            adapter.base.model.rotary_emb.inv_freq.add_(0.001)
    elif target == "config":
        adapter.base.config.max_position_embeddings += 1
    elif target == "attention_implementation":
        adapter.base.config._attn_implementation = "sdpa"
    else:
        adapter.base.model.rotary_emb.attention_scaling *= 2
    with pytest.raises(RuntimeError, match="changed; rebuild"):
        adapter.forward_with_frame(frame=frame, **action_batch(frame))
    with pytest.raises(RuntimeError, match="unchanged sealed original"):
        adapter.save_extension(tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_producer_revision_rejects_stale_record_and_sense_features():
    adapter = tiny_adapter().eval()
    frame = synthetic_frame(adapter)
    with torch.no_grad():
        next(adapter.record_writer.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="Memory producer"):
        adapter.make_frame("new", frame.evidence, frame.memory, frame.sense)
    fresh = synthetic_frame(adapter)
    with torch.no_grad():
        next(adapter.sense_ports.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="Sense producer"):
        adapter.make_frame("new", fresh.evidence, fresh.memory, fresh.sense)


def test_request_release_invalidation_and_foreign_cache_are_explicit():
    adapter = tiny_adapter().eval()
    frame = synthetic_frame(adapter)
    request = RequestContext("request", TaskGraph("task", "test"), frame, None)
    request.assert_ready()
    request.native_cache = object()
    with pytest.raises(ValueError, match="cache reuse"):
        request.assert_ready()
    request.native_cache = None
    request.invalidate()
    assert request.frame is None and request.selection is None
    with pytest.raises(RuntimeError, match="invalid"):
        request.assert_ready()
    second = RequestContext("second", TaskGraph("task", "test"), frame, None)
    second.release()
    with pytest.raises(RuntimeError, match="released"):
        second.assert_ready()
    with pytest.raises(ValueError, match="cache reuse"):
        adapter.forward_with_frame(frame=frame, past_key_values=object(), **action_batch(frame))


def test_interleaved_requests_keep_frames_separate_and_generation_verifies_each_step():
    adapter = tiny_adapter().eval()
    with torch.no_grad():
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_(std=0.05)
        first = synthetic_frame(adapter)
        second = synthetic_frame(adapter, prompt_ids=torch.tensor([[12, 13, 14]]))
        expected = adapter.forward_with_frame(frame=first, **action_batch(first)).logits
        adapter.forward_with_frame(frame=second, **action_batch(second))
        repeated = adapter.forward_with_frame(frame=first, **action_batch(first)).logits
    torch.testing.assert_close(repeated, expected, atol=0, rtol=0)
    first.assert_immutable()
    second.assert_immutable()
    request = RequestContext("request", TaskGraph("task", "test"), first, None)
    verified = []
    def verify(current):
        assert current is request
        current.assert_ready()
        verified.append(current.request_id)
    output = adapter.generate_action(request, verify=verify, max_new_tokens=3, eos_token_id=[])
    assert output.shape[1] == first.sense.prompt_ids.shape[1] + 3
    assert request.generated_ids == tuple(output[0, -3:].tolist())
    assert verified == ["request"] * 4
    assert request.native_cache is None
    assert not adapter.training


def test_generation_stop_predicate_runs_after_each_token():
    adapter = tiny_adapter().eval()
    request = RequestContext("request", TaskGraph("task", "test"),
                             synthetic_frame(adapter), None)
    calls = []

    def stop_after_two(ids):
        calls.append(ids)
        return len(ids) >= 2

    output = adapter.generate_action(
        request, verify=lambda current: current.assert_ready(),
        max_new_tokens=4, eos_token_id=[], stop_predicate=stop_after_two)
    assert len(calls) == 2
    assert request.generated_ids == calls[-1]
    assert len(request.generated_ids) == 2
    assert output.shape[1] == request.frame.sense.prompt_ids.shape[1] + 2


def test_known_prefix_target_mask_and_port_map_are_enforced():
    adapter = tiny_adapter().eval()
    frame = synthetic_frame(adapter)
    batch = action_batch(frame)
    changed = {**batch, "input_ids": batch["input_ids"].clone()}
    changed["input_ids"][0, 0] = 15
    with pytest.raises(ValueError, match="exact known prefix"):
        adapter.forward_with_frame(frame=frame, **changed)
    contaminated = {**batch, "labels": batch["labels"].clone()}
    contaminated["labels"][0, 0] = 1
    with pytest.raises(ValueError, match="supervised targets"):
        adapter.forward_with_frame(frame=frame, **contaminated)
    with pytest.raises(ValueError, match="all response logits"):
        adapter.forward_with_frame(frame=frame, logits_to_keep=1, **batch)
    with pytest.raises(ValueError, match="Undeclared port"):
        adapter.make_frame("new", frame.evidence, frame.memory, frame.sense, enabled_ports=(1,))


@pytest.mark.parametrize("regime", ["modules_only", "full_augmented"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_d1_d2_writer_and_native_gradient_ownership_with_checkpointing(regime, checkpointing):
    adapter = tiny_adapter(regime).train()
    if checkpointing:
        adapter.base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        assert adapter.base.is_gradient_checkpointing
    original = {n: p.detach().clone() for n, p in adapter.base.named_parameters()}
    groups = adapter.optimizer_groups(base_lr=1e-3, new_lr=1e-2)
    owned = [id(p) for group in groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(p) for p in adapter.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, foreach=False)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        frame = synthetic_frame(adapter)
        output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
        assert torch.isfinite(output.loss)
        output.loss.backward()
        audit = adapter.gradient_audit()
        assert not audit["original"]["nonfinite"]
        assert not audit["new"]["nonfinite"]
        for port in adapter.memory_ports.values():
            assert port.output_projection.weight.grad.abs().sum() > 0
        if step == 1:
            assert adapter.record_writer.encoder.input_projection.weight.grad.abs().sum() > 0
            for port in adapter.memory_ports.values():
                assert port.query_projection.weight.grad.abs().sum() > 0
        if regime == "modules_only":
            assert not audit["original"]["present"]
        else:
            assert len(audit["original"]["present"]) == len(original)
        optimizer.step()
    if regime == "modules_only":
        for name, parameter in adapter.base.named_parameters():
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    else:
        assert not torch.equal(adapter.base.get_input_embeddings().weight,
                               original["model.embed_tokens.weight"])
        assert not torch.equal(adapter.base.lm_head.weight, original["lm_head.weight"])


def test_mutation_after_forward_is_rejected_before_backward():
    adapter = tiny_adapter().train()
    frame = synthetic_frame(adapter)
    output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
    with torch.no_grad():
        frame.memory.slots.add_(1)
    with pytest.raises(RuntimeError, match="DecisionFrame tensor mutated"):
        output.loss.backward()


def test_extension_only_roundtrip_preserves_native_parameters_and_never_saves_base(tmp_path, monkeypatch):
    adapter = tiny_adapter().eval()
    adapter.seal_original_identity()
    original = {n: p.detach().clone() for n, p in adapter.base.named_parameters()}
    def forbidden_native_save(*args, **kwargs):
        raise AssertionError("Extension export attempted to save the base")
    monkeypatch.setattr(adapter.base, "save_pretrained", forbidden_native_save)
    with torch.no_grad():
        for port in adapter.memory_ports.values():
            port.output_projection.weight.normal_(std=0.04)
        frame = synthetic_frame(adapter)
        expected = adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits
    destination = tmp_path / "extension"
    receipt = adapter.save_extension(destination)
    assert receipt["base_saved"] is False
    assert sorted(p.name for p in destination.iterdir()) == ["extensions.safetensors", "manifest.json"]
    weights = load_file(str(destination / "extensions.safetensors"))
    assert set(weights) == {name for name, _ in adapter.new_named_parameters()}
    assert not any(name.startswith("base.") for name in weights)
    for name, parameter in adapter.base.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    restored = tiny_adapter().eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    with torch.no_grad():
        new_frame = synthetic_frame(restored)
        actual = restored.forward_with_frame(frame=new_frame, **action_batch(new_frame)).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name, parameter in restored.base.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


@pytest.mark.parametrize("change", ["native", "full_scope"])
def test_extension_export_rejects_changed_native_or_full_scope(tmp_path, change):
    adapter = tiny_adapter()
    adapter.seal_original_identity()
    if change == "native":
        with torch.no_grad():
            next(adapter.base.parameters()).add_(0.01)
    else:
        adapter.set_regime("full_augmented")
    with pytest.raises(RuntimeError, match="unchanged sealed original"):
        adapter.save_extension(tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_extension_load_rejects_manifest_and_weight_tampering(tmp_path):
    adapter = tiny_adapter()
    adapter.seal_original_identity()
    destination = tmp_path / "extension"
    adapter.save_extension(destination)
    restored = tiny_adapter()
    restored.seal_original_identity()
    path = destination / "manifest.json"
    original = path.read_text()
    receipt = json.loads(original)
    receipt["manifest"]["source_revision"] = "different-source"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Incompatible or changed"):
        restored.load_extension(destination)
    path.write_text(original)
    with (destination / "extensions.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Incompatible or changed"):
        restored.load_extension(destination)


def test_abi_and_cache_manifest_are_rejected_at_adapter_boundary():
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    adapter = tiny_adapter()
    with pytest.raises(ValueError, match="declared ABI"):
        MiniCPM5Adapter(adapter.base, replace(adapter.manifest, native_abi_digest="wrong"))
    with pytest.raises(ValueError, match="Cached generation"):
        MiniCPM5Adapter(adapter.base, replace(adapter.manifest, cached_generation_supported=True))
