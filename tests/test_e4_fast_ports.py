"""Opt-in native fast-memory read port: null parity, influence and serialization."""

import torch
from dataclasses import replace

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import FastMemoryRead, FastMemorySpec

SPEC = FastMemorySpec(heads=2, key_size=4, value_size=4)


def fast_weights(value=0.0):
    return torch.full((1, SPEC.heads, SPEC.key_size, SPEC.value_size), value)


def fast_read(adapter, value=0.0):
    return FastMemoryRead(fast_weights(value), adapter.fast_producer_key(), 'state-1')


def fast_frame(adapter, *, enabled_branches=None, value=0.0):
    base = synthetic_frame(adapter)
    return adapter.make_frame(base.decision_id, base.evidence, base.memory,
                              base.sense, fast=fast_read(adapter, value),
                              enabled_branches=enabled_branches)


def logits(adapter, frame):
    with torch.no_grad():
        return adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits


def test_fast_ports_are_null_at_init_and_independently_gated():
    adapter = tiny_adapter(fast_memory=SPEC).eval()
    disabled = fast_frame(adapter, enabled_branches=('memory',))
    baseline = logits(adapter, disabled)
    active = fast_frame(adapter, enabled_branches=('memory', 'fast'), value=1.0)
    torch.testing.assert_close(logits(adapter, active), baseline, atol=0, rtol=0)
    with torch.no_grad():
        for port in adapter.fast_ports.values():
            port.output_projection.weight.normal_(0.0, 0.02)
    changed = logits(adapter, fast_frame(adapter, enabled_branches=('memory', 'fast'), value=1.0))
    assert not torch.equal(changed, baseline)
    off = logits(adapter, fast_frame(adapter, enabled_branches=('memory',)))
    torch.testing.assert_close(off, baseline, atol=0, rtol=0)


def test_fast_reader_gradient_and_zero_state():
    adapter = tiny_adapter(fast_memory=SPEC).train()
    with torch.no_grad():
        for port in adapter.fast_ports.values():
            port.output_projection.weight.normal_(0.0, 0.02)
    frame = fast_frame(adapter, enabled_branches=('memory', 'fast'), value=1.0)
    output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
    output.loss.backward()
    for port in adapter.fast_ports.values():
        assert port.output_projection.weight.grad is not None
        assert port.query_projection.weight.grad is not None
        assert torch.isfinite(port.output_projection.weight.grad).all()
    zero_frame = fast_frame(adapter, enabled_branches=('memory', 'fast'), value=0.0)
    with torch.no_grad():
        zero_logits = adapter.forward_with_frame(frame=zero_frame, **action_batch(frame)).logits
    baseline = logits(adapter.eval(), fast_frame(adapter, enabled_branches=('memory',)))
    # A zero fast state is an exact null branch even after port training.
    assert torch.equal(zero_logits, baseline)


def test_fast_extension_v2_roundtrip(tmp_path):
    adapter = tiny_adapter(fast_memory=SPEC).eval()
    adapter.seal_original_identity()
    with torch.no_grad():
        for port in adapter.fast_ports.values():
            port.output_projection.weight.normal_(0.0, 0.02)
    frame = fast_frame(adapter, enabled_branches=('memory', 'fast'), value=1.0)
    expected = logits(adapter, frame)
    destination = tmp_path / 'extension'
    receipt = adapter.save_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v2'
    assert 'fast_ports' in receipt['components']
    restored = tiny_adapter(fast_memory=SPEC).eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    actual = logits(restored, fast_frame(restored, enabled_branches=('memory', 'fast'), value=1.0))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
