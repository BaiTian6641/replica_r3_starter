"""Record-token copy head: null path, influence, gradient and serialization."""
import torch
from dataclasses import replace
from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import CopyRead, CopySpec
from replica_lab.architecture.ports import RecordCopyHead

SPEC = CopySpec(max_tokens=32)

def copy_read(adapter):
    ids = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    return CopyRead(ids, mask, adapter.copy_producer_key(), 'source-digest')

def logits(adapter, frame):
    with torch.no_grad():
        return adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits

def test_copy_head_is_null_at_init_and_can_copy_source_tokens():
    adapter = tiny_adapter(copy=SPEC).eval()
    frame = synthetic_frame(adapter, copy=copy_read(adapter))
    off = synthetic_frame(adapter)
    base = logits(adapter, off)
    zero = logits(adapter, frame)
    torch.testing.assert_close(zero, base, atol=0, rtol=0)
    with torch.no_grad():
        adapter.copy_head.gate_logit.fill_(2.0)
        adapter.copy_head.query_projection.weight.normal_(0.0, 0.05)
        adapter.copy_head.key_projection.weight.normal_(0.0, 0.05)
    # Gate changes are parameter mutations; rebuild the frame so producer and
    # parameter stamps remain valid.
    active_frame = synthetic_frame(adapter, copy=copy_read(adapter))
    active = logits(adapter, active_frame)
    assert not torch.equal(active, base)
    # Candidate token 7 is in the copy set and gets an added logit.
    assert not torch.equal(active[..., 7], base[..., 7])

def test_copy_head_gradient_and_extension_roundtrip(tmp_path):
    adapter = tiny_adapter(copy=SPEC).train()
    with torch.no_grad():
        adapter.copy_head.gate_logit.fill_(1.0)
    frame = synthetic_frame(adapter, copy=copy_read(adapter))
    output = adapter.forward_with_frame(frame=frame, **action_batch(frame))
    output.loss.backward()
    for p in adapter.copy_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    adapter.eval(); adapter.seal_original_identity()
    expected = logits(adapter, synthetic_frame(adapter, copy=copy_read(adapter)))
    destination = tmp_path / 'extension'
    receipt = adapter.save_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v2'
    restored = tiny_adapter(copy=SPEC).eval(); restored.seal_original_identity()
    restored.load_extension(destination)
    actual = logits(restored, synthetic_frame(restored, copy=copy_read(restored)))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
