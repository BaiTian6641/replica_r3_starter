"""Content-key reader mechanics on the tiny CPU fixture; no quality claim."""
import json
import time

import pytest
import torch

from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_v12_fixture import build_v12_stack, canonical_span, scope
from replica_lab.intrinsic_memory_reader_query_v15 import attentive_forward


def make_path(key_mode="ordered", execution="content_key_cpu"):
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="content_key_cpu_fixture", execution_identity=execution,
        objective_identity={"scope": "content_key_mechanics"}, key_mode=key_mode)
    return path


def test_content_snapshot_separates_keys_from_ordered_values_and_preserves_default():
    torch.set_num_threads(1)
    ordered = make_path("ordered")
    content = make_path("content")
    sc = scope("content_keys")
    try:
        ordered_batch = ordered.training_values(
            canonical_span("the badge is red", record_id="badge"), scope=sc)
        content_batch = content.training_values(
            canonical_span("the badge is red", record_id="badge"), scope=sc)
        ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
        mask = torch.ones_like(ids)
        old = ordered.snapshot(ids, ordered_batch, scope=sc)
        new = content.snapshot(ids, content_batch, scope=sc)
        assert old.key_values is None
        assert torch.equal(new.key_values[0, 0, :4], content_batch.slots)
        expected_values = content_batch.slots + content.order(torch.arange(4))
        assert torch.equal(new.values[0, 0, :4], expected_values)
        assert not torch.equal(new.key_values, new.values)
        with torch.no_grad():
            disabled = content.forward(ids, mask, content_batch, scope=sc, enabled=False)
            enabled = content.forward(ids, mask, content_batch, scope=sc)
        assert torch.equal(disabled, enabled)
        before = new.key_values.clone()
        content_batch.slots.add_(1)
        assert torch.equal(before, new.key_values)
    finally:
        ordered.close()
        content.close()


def test_content_key_gradients_and_attention_are_live():
    torch.set_num_threads(1)
    path = make_path("content")
    sc = scope("content_training")
    ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, 0] = -100
    try:
        for _ in range(2):
            path.optimizer.zero_grad(set_to_none=True)
            batch = path.training_values(canonical_span("the badge is red", record_id="badge"),
                                         scope=sc)
            logits = path.forward(ids, mask, batch, scope=sc)
            path.adapter.causal_loss(logits, ids, labels).backward()
            path.optimizer.step()
        path.optimizer.zero_grad(set_to_none=True)
        batch = path.training_values(canonical_span("the badge is red", record_id="badge"),
                                     scope=sc)
        logits, attention = attentive_forward(path, ids, mask, batch, sc, position=3)
        path.adapter.causal_loss(logits, ids, labels).backward()
        for name in ("key_projection", "value_projection", "query_projection", "output_projection"):
            gradient = getattr(path.reader.ports["3"], name).weight.grad
            assert gradient is not None and float(gradient.abs().sum()) > 0, name
        for site in (3, 7):
            selected = attention[site]
            assert torch.allclose(selected["weights"] @ selected["value"], selected["read"],
                                  atol=1e-6, rtol=1e-6)
    finally:
        path.close()


def test_content_interventions_match_native_engine_and_checkpoint_mode(tmp_path):
    from replica_lab.intrinsic_memory_budget_v12 import COUNTERS
    from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
    from replica_lab.intrinsic_memory_counterfactual_v13 import DiagnosticEngineV13
    from test_intrinsic_diagnostic_v5 import FixtureBackend

    torch.set_num_threads(1)
    spec = CompletionObjectiveSpec(64, 256, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")
    plan = {"diagnose": {**{key: 1000000 for key in COUNTERS},
                         "artifact_bytes": 64 * 1024**2, "wall_seconds": 300}}
    ledger = IncrementalWorkLedger.create(tmp_path / "ledger.jsonl", identity="content-controls",
                                          plan=plan, deadline=time.monotonic() + 300)
    ledger.enter("diagnose")
    path = make_path("content")
    engine = DiagnosticEngineV13(path, FixtureBackend(), spec, ledger,
                                 max_new_tokens=0, output=tmp_path)
    engine.stage = "diagnose"
    left_scope, right_scope = scope("left"), scope("right")
    ids = torch.tensor([[3, 5, 7, 9]])
    mask = torch.ones_like(ids)
    try:
        left = path.training_values(canonical_span("the badge is red", record_id="left"),
                                    scope=left_scope)
        right = path.training_values(canonical_span("the badge is blue", record_id="right"),
                                     scope=right_scope)
        for control in ("wrong", "shuffled", "zero"):
            expected = engine._forward(ids, mask, left, left_scope, control=control,
                                       wrong=right, wrong_scope=right_scope)
            actual, _ = attentive_forward(path, ids, mask, left, left_scope, control=control,
                                          wrong=right, wrong_scope=right_scope, position=3)
            assert torch.equal(expected, actual), control
        path.save(tmp_path / "checkpoint")
        manifest = json.loads((tmp_path / "checkpoint/manifest.json").read_text())
        assert manifest["reader_key_mode"] == "content"
    finally:
        path.close()
    restored = make_path("content")
    try:
        restored.load(tmp_path / "checkpoint")
    finally:
        restored.close()
    ordered = make_path("ordered")
    try:
        with pytest.raises(ContextualMemoryError, match="identity differs"):
            ordered.load(tmp_path / "checkpoint")
    finally:
        ordered.close()


def test_legacy_ordered_manifest_loads_with_explicit_default(tmp_path):
    torch.set_num_threads(1)
    path = make_path("ordered", execution="legacy_ordered")
    try:
        path.save(tmp_path / "checkpoint")
        manifest_path = tmp_path / "checkpoint/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest.pop("reader_key_mode") == "ordered"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        path.close()
    restored = make_path("ordered", execution="legacy_ordered")
    try:
        result = restored.load(tmp_path / "checkpoint")
        assert result.get("reader_key_mode", "ordered") == "ordered"
    finally:
        restored.close()

def test_invalid_key_mode_is_refused():
    with pytest.raises(ContextualMemoryError, match="key mode"):
        make_path("invalid")
