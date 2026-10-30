"""Independent contract reproductions for the v12 review; source-free CPU only."""
import time
import types
from dataclasses import replace
from contextlib import contextmanager

import pytest
import torch

from replica_lab.intrinsic_memory_budget_v12 import BudgetError, WorkLedger
from replica_lab.intrinsic_memory_coverage_v12 import FIT_CASES, plan
from replica_lab.intrinsic_memory_diagnostic_v12 import DiagnosticEngineV12
from replica_lab.contextual_spark_fixture import ids_and_mask
from replica_lab.contextual_v12_fixture import build_v12_stack, canonical_span, scope as v12_scope
from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.intrinsic_memory_density_v12 import (
    DensityError, SiteInvocationProfiler, assert_no_read_write_effects, cost_snapshot)
from replica_lab.successors.chunks import ChunkError, ChunkSession, ChunkStore, MAX_TOKENS
from replica_lab.successors.formation import ScopeViolation
from replica_lab.successors.tiers import TieredBlobStore
from test_intrinsic_memory_density_v12_profiler import make as make_profile_path
from test_intrinsic_memory_density_v12_profiler import training_batch
from test_successor_formation_v12 import commit, make_store, observation, pin
from test_successor_chunks_v12 import make_session, tensors
from test_tiered_memory import scope as tier_scope
import profile_intrinsic_memory_density_v12 as profile
import run_intrinsic_memory_density_v12_cost_pilot as cost_pilot


def test_restart_prompt_budget_covers_teacher_revalidation_and_update(tmp_path):
    ledger = WorkLedger.create(tmp_path / "restart.jsonl", identity="independent-review",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    ledger.enter("restart")
    for _ in range(FIT_CASES * 2 + 3):
        token = ledger.reserve("restart", "prompt_tokenization", prompt_tokenizer_calls=1)
        ledger.finish(token)


def test_nonterminating_generation_is_marked_truncated(tmp_path):
    engine = DiagnosticEngineV12.__new__(DiagnosticEngineV12)
    engine.max_new_tokens = 2
    engine.spec = types.SimpleNamespace(context_tokens=16, pad_token_id=0,
                                       allowed_stop_token_ids=(1,), vocab_size=8)
    engine.path = types.SimpleNamespace(validate=lambda batch, scope: None,
        adapter=types.SimpleNamespace(base=types.SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1)]))))
    engine.backend = types.SimpleNamespace(decode=lambda ids: "red")
    engine.tokens = lambda row, exact=False: [2]
    engine.sync = lambda: None
    def forward(ids, mask, batch, scope, **kwargs):
        logits = torch.zeros(1, ids.shape[1], 8)
        logits[:, :, 3] = 1
        return logits
    engine.forward = forward
    @contextmanager
    def measured(operation, **amounts):
        yield {}
    engine.measured = measured
    result = engine.generate({"text": "badge red", "question": "color", "answer": "red"},
                             None, "scope")
    assert result["terminated"] is False
    assert result["truncated"] is True


def test_chunk_recompute_reproduces_committed_digest(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        session = make_session(store)
        session.begin_chunk([20, 21])
        keys, values = tensors()
        session.propose(keys, values)
        receipt = session.commit().receipt
        assert session.recompute(keys, values)[1] == receipt.proposal_digest


def test_pending_chunk_cannot_be_replaced(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        session = make_session(store)
        session.begin_chunk([20])
        session.propose(*tensors())
        with pytest.raises(ChunkError):
            session.begin_chunk([21])


def test_invalidate_restores_last_committed_state(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        session = make_session(store)
        original = session.state.detach().clone()
        session.begin_chunk([20])
        session.propose(*tensors())
        session.invalidate()
        assert torch.equal(session.state, original)


def test_initial_prefix_respects_global_token_bound(tmp_path):
    with ChunkStore(tmp_path, subject_id="subject-1") as store:
        with pytest.raises(ValueError):
            store.open_sequence("spark", "branch-a", "seq-1",
                                initial_prefix=range(MAX_TOKENS + 1))


def test_cold_resume_revalidates_external_authority(tmp_path):
    store = TieredBlobStore(tmp_path, io_delay_seconds=0.01)
    store.publish("record", b"stale", generation=1, kind="e", producer="p",
                  representation="r", scope_digest="s", tier="cold")
    store._hot.clear()
    authority = {"eligible": True, "generation": 1, "eligibility_revision": 0}
    pending = store.read("record", authority=lambda record_id: dict(authority),
                         expected_generation=1, budget_seconds=0)
    assert pending.status == "pending"
    authority["eligible"] = False
    result = store.resume(pending.ticket_id)
    assert result.status != "ready"


def test_p_clock_can_advance(tmp_path):
    store = TieredBlobStore(tmp_path)
    store.publish_p("expert", weights=b"w1", moments=b"m1", clock=1,
                    encoder="e1", lineage=["r1"])
    store.publish_p("expert", weights=b"w2", moments=b"m2", clock=2,
                    encoder="e2", lineage=["r1", "r2"])
    assert store.read_p("expert")["clock"] == 2


def test_pin_is_immutable_under_later_correction(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, pin(store), [observation("base", "dated fact")])
        record_id = base.receipt.record_ids[0]
        snapshot = pin(store)
        assert record_id in {item.record_id for item in store.visible(snapshot)}
        store.correct(record_id, text="corrected later", reason="late evidence",
                      pin=pin(store), verifier_receipt_id="verifier-late")
        assert record_id in {item.record_id for item in store.visible(snapshot)}


def test_cross_branch_supersede_is_refused(tmp_path):
    with make_store(tmp_path) as store:
        other = commit(store, pin(store, branch="branch-b"),
                       [observation("other", "branch b fact")])
        foreign_id = other.receipt.record_ids[0]
        with pytest.raises(ScopeViolation):
            commit(store, pin(store, branch="branch-a"),
                   [observation("child", "invalid cross branch", supersedes=foreign_id)])


def test_rejected_profiler_instrument_is_atomic():
    path = make_profile_path("two")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, training_batch(path), scope=tier_scope())
        with pytest.raises(DensityError):
            profiler.instrument(snapshot, forward_id="f-9", active_sites=(7, 3))
        assert "f-9" not in profiler.summary()["forward_ids"]
    finally:
        profiler.close()
        path.close()


def test_durable_recompute_permits_authority_revalidation(tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        path.admit(store, profile.spans()[0], scope=profile.scope())
        batch = path.read(store, "Badge", scope=profile.scope())
        ids, mask = ids_and_mask(4)
        snapshot = path.snapshot(ids, batch, scope=profile.scope())
        first = profiler.instrument(snapshot, forward_id="f-1")
        path.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                  use_spine=False, intrinsic_memory=first)
        before = cost_snapshot(path)
        second = profiler.instrument(snapshot, phase="recompute", recompute_of="f-1")
        path.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                  use_spine=False, intrinsic_memory=second)
        assert_no_read_write_effects(before, cost_snapshot(path))
    finally:
        profiler.close()
        store.close()
        path.close()


def test_cost_pilot_zeroes_gradients_for_each_declared_update(monkeypatch):
    original_build = cost_pilot.build_path
    calls = []
    def counted_build(arm):
        path = original_build(arm)
        original_zero = path.optimizer.zero_grad
        def counted_zero(*args, **kwargs):
            calls.append(1)
            return original_zero(*args, **kwargs)
        monkeypatch.setattr(path.optimizer, "zero_grad", counted_zero)
        return path
    monkeypatch.setattr(cost_pilot, "build_path", counted_build)
    cost_pilot.run_pass("two", 2)
    assert len(calls) == 2



def test_buffer_only_source_mutation_invalidates_feature_authority():
    adapter, _, provider, path = build_v12_stack()
    try:
        adapter.base.register_buffer("independent_review_buffer", torch.zeros(1))
        with torch.no_grad():
            adapter.base.independent_review_buffer.add_(1)
        with pytest.raises(ContextualMemoryError):
            provider.validate_original()
    finally:
        path.close()


def test_token_cache_binds_actual_canonical_text():
    _, _, provider, path = build_v12_stack()
    try:
        first = canonical_span("badge red", record_id="same")[0]
        second = replace(first, text="badge tan")
        provider.token_ids(first)
        actual = provider.tokenizer(second.text, add_special_tokens=False,
                                    truncation=False)["input_ids"]
        assert provider.token_ids(second) == actual
    finally:
        path.close()


def test_empty_list_active_sites_skips_memory_and_matches_base():
    adapter, _, _, path = build_v12_stack(sites=(3, 7))
    try:
        spans = canonical_span("badge red", record_id="badge")
        batch = path.training_values(spans, scope=v12_scope())
        ids, mask = ids_and_mask(4)
        with torch.no_grad():
            actual = path.forward(ids, mask, batch,
                                  scope=v12_scope(),
                                  active_sites=[])
            expected = adapter.forward_text(ids, mask, use_projector=False,
                                            use_ports=False, use_spine=False)
        assert torch.equal(actual, expected)
    finally:
        path.close()
