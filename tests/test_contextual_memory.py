from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from replica_lab.contextual_memory import (
    ContextualMemoryError, ContextualMemoryPath, canonical_spans, tensor_digest,
)
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.optimizers import NativeFusedAdamW
from test_tiered_memory import open_store, scope


def make_path(seed=17):
    torch.manual_seed(seed)
    adapter, *_ = build_contextual_adapter(seed=seed, layers=8, spine_sites=(3,7),
                                           spine_slots=8, memory_size=256)
    return ContextualMemoryPath(adapter, lr=0.01)


def spans(text="The crimson badge means ready.", name="rule", available=10):
    return canonical_spans(text, record_id=name, source_id="verified-source", source_epoch=1,
                           available_at=available)


def step(path, values):
    ids, mask = ids_and_mask()
    path.optimizer.zero_grad(set_to_none=True)
    batch = path.training_values(values, scope=scope())
    logits = path.forward(ids, mask, batch, scope=scope())
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels_from_ids(ids).reshape(-1))
    loss.backward()
    grads = {"producer": sum(float(p.grad.abs().sum()) for p in path.encoder.parameters() if p.grad is not None),
             "reader": sum(float(p.grad.abs().sum()) for p in path.reader.ports.parameters() if p.grad is not None)}
    assert all(torch.isfinite(p.grad).all() for group in path.optimizer.param_groups
               for p in group["params"] if p.grad is not None)
    path.optimizer.step()
    return float(loss.detach()), grads


def test_ordered_bounded_canonical_spans_and_context_sensitive_values():
    text = "λ🙂abc " * 200
    result = spans(text)
    assert "".join(s.text for s in result) == text
    assert all(len(s.text.encode()) <= 256 for s in result)
    assert [s.source_order for s in result] == list(range(len(result)))
    assert all(a.end == b.start for a,b in zip(result, result[1:]))
    path = make_path()
    values = path.encoder(spans())
    assert values.shape == (1,4,256)
    assert not torch.equal(values[:,0], values[:,1])
    changed = path.encoder(spans("The crimson badge means closed."))
    assert not torch.equal(values, changed)


def test_disabled_null_parity_gradients_and_unique_ownership():
    path = make_path()
    assert isinstance(path.optimizer, NativeFusedAdamW)
    ids, mask = ids_and_mask()
    batch = path.training_values(spans(), scope=scope())
    baseline = path.forward(ids, mask, None, scope=scope())
    assert torch.equal(baseline, path.forward(ids, mask, batch, scope=scope(), enabled=False))
    assert torch.equal(baseline, path.forward(ids, mask, batch, scope=scope()))
    _, first = step(path, spans())
    assert first["reader"] > 0 and first["producer"] == 0
    _, second = step(path, spans())
    assert second["producer"] > 0 and second["reader"] > 0
    assert path.adapter.base_unchanged()
    assert all(p.grad is None for p in path.adapter.base.parameters())
    owned = [id(p) for group in path.optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(set(owned))


def test_correct_wrong_shuffled_and_absent_value_controls():
    path = make_path()
    for _ in range(3): step(path, spans())
    ids, mask = ids_and_mask()
    correct = path.training_values(spans(), scope=scope())
    wrong = path.training_values(spans("The silver badge means closed."), scope=scope())
    shuffled = replace(correct, slots=correct.slots.flip(0))
    logits = path.forward(ids, mask, correct, scope=scope())
    for batch in (wrong, shuffled, None):
        assert not torch.equal(logits, path.forward(ids, mask, batch, scope=scope()))
    # This tests causal influence, not answer quality or native capability.


def test_durable_slots_consumed_directly_and_correction_invalidates(tmp_path, monkeypatch):
    path = make_path()
    for _ in range(2): step(path, spans())
    store = open_store(tmp_path)
    span = spans()[0]
    assert path.admit(store, span, scope=scope()).outcome.ok
    batch = path.read(store, "crimson badge", scope=scope())
    assert not batch.slots.requires_grad
    ids, mask = ids_and_mask()
    expected = path.encoder([span])[0].detach()
    assert torch.equal(batch.slots, expected)
    with monkeypatch.context() as patch:
        patch.setattr(path.encoder, "forward", lambda *a: pytest.fail("inference re-encoded text"))
        path.forward(ids, mask, batch, scope=scope())
    text = path.exact_fallback(batch, scope=scope(), tokenizer=lambda s:s.split())
    assert text == span.text and path.cost["fallback_bytes"] == len(text.encode())
    assert path.cost["fallback_tokens"] == len(text.split())
    correction = replace(spans("The crimson badge now means closed.", name="fixed", available=11)[0],
                         source_epoch=2, supersedes=span.record_id)
    assert path.correct(store, span.record_id, correction, scope=scope()).outcome.ok
    with pytest.raises(ContextualMemoryError, match="invalidated"):
        path.forward(ids, mask, batch, scope=scope())
    fresh = path.read(store, "crimson badge", scope=scope())
    assert fresh.spans[0].supersedes == span.record_id
    store.revoke_record(correction.record_id, reason="withdrawn", scope=scope())
    with pytest.raises(ContextualMemoryError, match="invalidated"):
        path.forward(ids, mask, fresh, scope=scope())
    store.close()


def test_stale_release_scope_time_and_generation_refused(tmp_path):
    path = make_path()
    store = open_store(tmp_path)
    path.admit(store, spans()[0], scope=scope())
    batch = path.read(store, "badge", scope=scope())
    with pytest.raises(ContextualMemoryError, match="scope"):
        path.validate(batch, scope("someone_else"))
    with pytest.raises(ContextualMemoryError, match="future"):
        path.validate(batch, scope(as_of=1))
    with pytest.raises(ContextualMemoryError, match="invalidated"):
        path.validate(replace(batch, slots=batch.slots.flip(0)), scope())
    step(path, spans())
    step(path, spans())
    with pytest.raises(ContextualMemoryError, match="stale producer"):
        path.read(store, "badge", scope=scope())
    with pytest.raises(ContextualMemoryError, match="stale producer"):
        path.validate(batch, scope())
    store.close()


def test_extension_only_reload_and_identical_optimizer_continuation(tmp_path):
    path = make_path()
    for _ in range(2): step(path, spans())
    manifest = path.save(tmp_path / "extension")
    assert manifest["contains_base_weights"] is False
    fresh = make_path()
    fresh.load(tmp_path / "extension")
    ids, mask = ids_and_mask()
    a = path.forward(ids, mask, path.training_values(spans(), scope=scope()), scope=scope())
    b = fresh.forward(ids, mask, fresh.training_values(spans(), scope=scope()), scope=scope())
    assert torch.equal(a,b)
    step(path, spans()); step(fresh, spans())
    assert path.encoder.release() == fresh.encoder.release()
    assert tensor_digest(path.order.weight) == tensor_digest(fresh.order.weight)
    fresh.freeze_readout()
    assert fresh.optimizer is None
    assert not any(p.requires_grad for p in fresh.encoder.parameters())
    assert fresh.adapter.base_unchanged()


def test_P_has_one_owner_and_fixed_reader_and_refuses_unqualified_native_learning():
    from replica_lab.contextual_p_learning import FrozenContextualP
    path = make_path()
    with pytest.raises(ContextualMemoryError, match="qualified readout"):
        FrozenContextualP(path, {})
    for _ in range(2): step(path, spans())
    p = FrozenContextualP(path, {}, engineering_fixture=True)
    assert p.bank.expert_count == 1
    producer = path.encoder.release()
    reader = {n:tensor_digest(t) for n,t in path.reader.ports.state_dict().items()}
    ids,mask = ids_and_mask()
    labels = labels_from_ids(ids)
    query = "Which rule applies to the next observation?"
    receipt = p.teach(ids, mask, labels, query, scope=scope(), event_id="teaching-1", lineage=["r1"])
    assert receipt["updated"] and receipt["E"] == receipt["S"] == receipt["C"] == 0
    assert receipt["clock"] == 1
    assert p.teach(ids, mask, labels, query, scope=scope(), event_id="teaching-1", lineage=["r1"])["updated"] is False
    assert path.encoder.release() == producer
    assert reader == {n:tensor_digest(t) for n,t in path.reader.ports.state_dict().items()}
    assert all(t.grad is None for t in path.encoder.parameters())
    assert not torch.equal(p.forward(ids,mask,query,scope=scope()), p.forward(ids,mask,query,scope=scope(),reset=True))
    p.invalidate("r1")
    with pytest.raises(ContextualMemoryError, match="revoked"):
        p.forward(ids,mask,query,scope=scope())
    p.reset(authorization="test_explicit_reset")
    assert p.clock == 0 and not p.quarantined


def test_intrinsic_memory_operates_with_external_spine_removed():
    path = make_path()
    assert "memory" not in path.adapter.extension.spine.ports
    assert not hasattr(path.adapter.extension.spine, "memory_projection")
    assert path.reader is path.adapter.extension.intrinsic_memory
    path.adapter.extension.spine = None
    ids,mask = ids_and_mask()
    batch = path.training_values(spans(),scope=scope())
    baseline = path.forward(ids,mask,None,scope=scope())
    assert torch.equal(baseline,path.forward(ids,mask,batch,scope=scope()))
    step(path,spans())
    _,gradients = step(path,spans())
    assert gradients["producer"] > 0 and gradients["reader"] > 0
    assert not torch.equal(baseline,path.forward(ids,mask,batch,scope=scope()))
    assert path.adapter.base_unchanged()
    assert any(n.startswith("intrinsic_memory.ports.") for n,_ in path.adapter.extension.named_parameters())


def test_explicit_legacy_migration_preserves_values_logits_and_optimizer(tmp_path):
    import hashlib
    import json
    from safetensors.torch import load_file,save_file
    from replica_lab.contextual_memory import LEGACY_FORMAT,FORMAT
    path=make_path()
    step(path,spans());step(path,spans())
    legacy=tmp_path/'legacy'
    path.save(legacy)
    # Synthetic historical v2 layout, byte-equivalent to its reader namespace.
    tensors=load_file(str(legacy/'extension.safetensors'))
    save_file({k.removeprefix('intrinsic_memory.') if k.startswith('intrinsic_memory.reader.') else k:v
               for k,v in tensors.items()},str(legacy/'extension.safetensors'))
    manifest=json.loads((legacy/'manifest.json').read_text())
    manifest['format']=LEGACY_FORMAT
    manifest.pop('owner');manifest.pop('external_spine_required')
    manifest['weights_sha256']=hashlib.sha256((legacy/'extension.safetensors').read_bytes()).hexdigest()
    (legacy/'manifest.json').write_text(json.dumps(manifest))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.iterdir()}
    fresh=make_path()
    with pytest.raises(ContextualMemoryError,match='explicit migration'):fresh.load(legacy)
    receipt=fresh.migrate_legacy(legacy,tmp_path/'intrinsic')
    assert receipt['to_format']==FORMAT and receipt['optimizer_state_preserved']
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.iterdir()}
    fresh.adapter.extension.spine=None
    ids,mask=ids_and_mask()
    assert torch.equal(path.forward(ids,mask,path.training_values(spans(),scope=scope()),scope=scope()),
                       fresh.forward(ids,mask,fresh.training_values(spans(),scope=scope()),scope=scope()))
    step(path,spans());step(fresh,spans())
    assert path.encoder.release()==fresh.encoder.release()
    assert all(torch.equal(v,fresh.reader.state_dict()[k]) for k,v in path.reader.state_dict().items())


def test_intrinsic_sites_do_not_depend_on_external_spine_sites():
    torch.manual_seed(17)
    adapter,*_=build_contextual_adapter(seed=17,layers=8,spine_sites=(1,5),spine_slots=4,memory_size=256)
    path=ContextualMemoryPath(adapter)
    assert path.spec.sites==(3,7) and path.spec.slots==8
    assert adapter.spine_spec.sites==(1,5)
    ids,mask=ids_and_mask()
    path.forward(ids,mask,path.training_values(spans(),scope=scope()),scope=scope())


def test_intrinsic_out_of_decoder_site_refused_before_ownership_transfer():
    adapter, *_ = build_contextual_adapter(
        seed=17, layers=4, spine_sites=(1, 3), spine_slots=8, memory_size=256)
    legacy = adapter.extension.spine
    ports = legacy.ports["memory"]
    with pytest.raises(ValueError, match="outside the decoder"):
        ContextualMemoryPath(adapter)
    assert not hasattr(adapter.extension, "intrinsic_memory")
    assert legacy.ports["memory"] is ports
    assert hasattr(legacy, "memory_projection")
