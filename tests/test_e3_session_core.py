"""Connected core path through MemorySession: graph ownership and inference detach."""

import pytest
import torch

from architecture_helpers import synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import CoreSpec, CoreState, MemoryRead, TaskGraph
from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
from replica_lab.memory_system.session import MemorySession

SPEC = CoreSpec(width=16, slots=4, steps=2, heads=4, task_features=24,
                ffn_hidden=64, control_hidden=29)
FEATURES = [0.0] * 24


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [3 + ord(character) % 61 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def owned_session(tmp_path):
    ledger = RecordLedger(tmp_path / "records.sqlite")
    adapter = tiny_adapter(core=SPEC)
    ledger.admit(MemoryRecord("r1", "fixture", "source", "blue cat", 1, 0, "test",
                              source_id="fixture", authenticity="source_report"),
                 authorized_subject="fixture", authorized_branch="source",
                 source_authorized=True)
    return MemorySession(adapter, CharacterTokenizer(), ledger), ledger


def prepare_args():
    return dict(prompt_ids=torch.tensor([[1, 4, 6]], dtype=torch.long),
                prompt_mask=torch.ones(1, 3, dtype=torch.bool), query="blue",
                scope=EligibilityScope("fixture", "replica-a", "test", 20),
                task=TaskGraph("tiny-task", "read the admitted source"),
                task_features=torch.tensor([FEATURES], dtype=torch.float32),
                task_digest="state-digest-1", revision=4)


def action_batch():
    return {"input_ids": torch.tensor([[1, 4, 6, 12, 15]], dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "labels": torch.tensor([[-100, -100, -100, 12, 15]], dtype=torch.long)}


def test_training_core_path_reaches_sensors_core_and_workspace_readers(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.train()
        with torch.no_grad():
            # Open both zero-init branches so upstream factors receive a first-step gradient.
            for port in adapter.workspace_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
            for port in adapter.memory_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
        args = prepare_args()
        args['task_features'] = torch.linspace(0.05, 0.95, 24).unsqueeze(0)
        request = session.prepare(training=True, train_core=True, train_sense=True,
                                  **args)
        try:
            assert request.frame.branch_enabled("memory")
            assert request.frame.branch_enabled("workspace")
            assert request.frame.core.workspace.requires_grad
            output = session.forward(request, **action_batch())
            assert torch.isfinite(output.loss)
            session.backward(request, output.loss)
            audit = adapter.gradient_audit()
            assert not audit["new"]["nonfinite"]
            for prefix in ("core.", "core.task_encoder.", "workspace_ports.0.",
                           "workspace_ports.2.", "sense_ports.0.", "sense_ports.3."):
                assert any(name.startswith(prefix) for name in audit["new"]["nonzero"]), prefix
            assert adapter.core.slot_queries.grad.abs().sum() > 0
            assert adapter.core.task_encoder.projection.weight.grad.abs().sum() > 0
            assert adapter.workspace_ports["0"].query_projection.weight.grad.abs().sum() > 0
            assert adapter.workspace_ports["2"].query_projection.weight.grad.abs().sum() > 0
            assert adapter.sense_ports["0"].projection.weight.grad.abs().sum() > 0
            assert request.frame.core.task_digest == "state-digest-1"
            assert request.frame.core.revision == 4
            assert not adapter.predict_success(request.frame.core).requires_grad
        finally:
            request.release()
        adapter.zero_grad(set_to_none=True)
    finally:
        ledger.close()


def test_training_without_train_core_detaches_core_and_keeps_sensors_gradient_free(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.train()
        with torch.no_grad():
            for port in adapter.workspace_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
        request = session.prepare(training=True, train_core=False, train_sense=False,
                                  **prepare_args())
        try:
            assert not request.frame.core.workspace.requires_grad
            output = session.forward(request, **action_batch())
            session.backward(request, output.loss)
            # A frozen core contributes a constant read; the workspace readers
            # remain part of the action graph and still receive the loss.
            assert adapter.core.slot_queries.grad is None
            assert adapter.core.task_encoder.projection.weight.grad is None
            assert adapter.sense_ports["0"].projection.weight.grad is None
            assert adapter.workspace_ports["0"].query_projection.weight.grad is not None
            assert adapter.workspace_ports["0"].output_projection.weight.grad is not None
        finally:
            request.release()
        adapter.zero_grad(set_to_none=True)
    finally:
        ledger.close()


def test_inference_detaches_a_gradient_bearing_previous_workspace(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.eval()
        memory = MemoryRead((), torch.zeros(1, 0, 4, 16), torch.zeros(1, 0, 4, dtype=torch.bool),
                            adapter.memory_producer_key())
        sense = synthetic_frame(adapter).sense
        with torch.enable_grad():
            previous = adapter.build_core_state(
                sense, memory, torch.tensor([FEATURES]),
                task_digest="previous", revision=3)
        assert previous.workspace.requires_grad
        request = session.prepare(training=False, previous_core=previous, **prepare_args())
        try:
            assert not request.frame.core.workspace.requires_grad
            assert request.frame.core.revision == 4
            with torch.no_grad():
                logits = session.forward(request, **action_batch()).logits
            assert torch.isfinite(logits).all()
        finally:
            request.release()
    finally:
        ledger.close()


def test_train_sense_without_train_core_reaches_sensors_but_not_core(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.train()
        with torch.no_grad():
            for port in adapter.workspace_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
            for port in adapter.memory_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
        request = session.prepare(training=True, train_core=False, train_sense=True,
                                  **prepare_args())
        try:
            # The sensor graph passes through a frozen core so the monitor path
            # stays out of the core parameters while sensors remain trainable.
            assert request.frame.core.workspace.requires_grad
            assert all(p.requires_grad for p in adapter.core.parameters())
            output = session.forward(request, **action_batch())
            session.backward(request, output.loss)
            assert adapter.core.slot_queries.grad is None
            assert adapter.core.task_encoder.projection.weight.grad is None
            assert adapter.sense_ports["0"].projection.weight.grad is not None
            assert adapter.sense_ports["0"].projection.weight.grad.abs().sum() > 0
            assert adapter.sense_ports["3"].projection.weight.grad.abs().sum() > 0
            for layer in ("0", "2"):
                assert adapter.workspace_ports[layer].query_projection.weight.grad is not None
                assert adapter.workspace_ports[layer].output_projection.weight.grad is not None
        finally:
            request.release()
        adapter.zero_grad(set_to_none=True)
    finally:
        ledger.close()


def test_core_impl_none_skips_core_and_builds_memory_only(tmp_path, monkeypatch):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.eval()

        def forbidden(*args, **kwargs):
            raise AssertionError('core_impl=none must not build or run the core')

        monkeypatch.setattr(adapter, 'build_core_state', forbidden)
        args = prepare_args()
        for key in ('task_features', 'task_digest', 'revision'):
            args.pop(key)
        request = session.prepare(training=False, core_impl='none', **args)
        try:
            assert request.frame.core is None
            assert request.frame.enabled_branches == ('memory',)
            assert not request.frame.branch_enabled('workspace')
            with torch.no_grad():
                logits = session.forward(request, **action_batch()).logits
            assert torch.isfinite(logits).all()
        finally:
            request.release()
    finally:
        ledger.close()


def test_core_impl_none_rejects_core_only_inputs(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        with pytest.raises(ValueError, match='does not accept task_features'):
            session.prepare(training=False, core_impl='none', **prepare_args())
        args = prepare_args()
        for key in ('task_features', 'task_digest', 'revision'):
            args.pop(key)
        with pytest.raises(ValueError, match='workspace branch'):
            session.prepare(training=False, core_impl='none',
                            enabled_branches=('workspace',), **args)
        adapter = session.adapter.eval()
        empty = MemoryRead((), torch.zeros(1, 0, 4, 16),
                           torch.zeros(1, 0, 4, dtype=torch.bool),
                           adapter.memory_producer_key())
        previous = adapter.build_core_state(synthetic_frame(adapter).sense, empty,
                                            torch.tensor([FEATURES]), task_digest='p', revision=1)
        with pytest.raises(ValueError, match='does not accept a previous core'):
            session.prepare(training=False, core_impl='none', previous_core=previous, **args)
    finally:
        ledger.close()


def test_memory_only_branch_with_core_present_needs_no_previous_workspace(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        request = session.prepare(training=False, enabled_branches=('memory',),
                                  **prepare_args())
        full = None
        try:
            assert request.frame.core is not None
            assert request.frame.enabled_branches == ('memory',)
            with torch.no_grad():
                memory_only = session.forward(request, **action_batch()).logits
            full = session.prepare(training=False, **prepare_args())
            with torch.no_grad():
                both = session.forward(full, **action_batch()).logits
            # The workspace reader is exactly null at initialization, so the
            # branch subset must not perturb the memory-only logits.
            torch.testing.assert_close(memory_only, both, atol=0, rtol=0)
        finally:
            request.release()
            if full is not None:
                full.release()
    finally:
        ledger.close()


def test_training_never_carries_a_live_previous_workspace_graph(tmp_path):
    session, ledger = owned_session(tmp_path)
    try:
        adapter = session.adapter.train()
        with torch.no_grad():
            for port in adapter.workspace_ports.values():
                port.output_projection.weight.normal_(0.0, 0.03)
        frame = synthetic_frame(adapter)
        leaf = torch.randn(1, SPEC.slots, SPEC.width, requires_grad=True)
        previous = CoreState(leaf, torch.ones(1, SPEC.slots, dtype=torch.bool),
                             torch.tensor([FEATURES]), adapter.core_producer_key(),
                             'previous-revision', 3, SPEC.steps)
        request = session.prepare(training=True, train_core=True, previous_core=previous,
                                  **prepare_args())
        try:
            output = session.forward(request, **action_batch())
            session.backward(request, output.loss)
            assert adapter.core.slot_queries.grad.abs().sum() > 0
            # A previous decision's workspace is a task-owned value; the action
            # graph must never reach the caller's live tensor.
            assert leaf.grad is None
        finally:
            request.release()
        adapter.zero_grad(set_to_none=True)
    finally:
        ledger.close()
