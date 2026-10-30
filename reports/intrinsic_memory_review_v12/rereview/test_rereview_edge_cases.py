"""New independent edge reproductions at c95d3e23; CPU/source-free only."""
import sqlite3
import time

import pytest

from replica_lab.intrinsic_memory_budget_v12 import BudgetError, WorkLedger
from replica_lab.intrinsic_memory_coverage_v12 import plan
from replica_lab.intrinsic_memory_density_v12 import SiteInvocationProfiler
from replica_lab.intrinsic_memory_density_protocols_v12 import (
    cost_tolerance, total_cost_protocol, validate_protocol)
from replica_lab.contextual_memory_v5 import PendingContextualRead
from replica_lab.contextual_memory_v12 import ContextualReadFailure
from replica_lab.successors.adapters import formation_authority
from replica_lab.successors.tiers import ReadContract, TieredBlobStore
from test_successor_formation_v12 import commit, make_store, observation, pin
import profile_intrinsic_memory_density_v12 as profile


def test_ledger_refuses_append_after_torn_tail(tmp_path):
    ledger = WorkLedger.create(tmp_path / "ledger.jsonl", identity="rereview",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    ledger.enter("baseline")
    with (tmp_path / "ledger.jsonl").open("ab") as stream:
        stream.write(bytes([123, 34, 116, 34, 58, 34, 114, 101, 115, 101, 114, 118, 101, 34]))
    with pytest.raises(BudgetError):
        ledger.reserve("baseline", "decode", generated_tokens=1)


def test_recovery_detects_deleted_revocation_tombstone(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, pin(store), [observation("base", "fact")])
        record_id = base.receipt.record_ids[0]
        store.revoke(record_id, reason="revoked")
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("DELETE FROM tombstones WHERE record_id=?", (record_id,))
    connection.commit()
    connection.close()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status != "ok"
        assert record_id not in {item.record_id for item in store.visible(pin(store))}


def test_revoked_p_is_not_ready_through_generic_tier_read(tmp_path):
    store = TieredBlobStore(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="e1", lineage=["r1"])
    state = store.read_p("expert")
    manifest = state["manifest"]
    store.revoke_p("expert", reason="revoked", authorized_by="host")
    contract = ReadContract(manifest["producer"], manifest["representation"],
                            manifest["scope_digest"])
    result = store.read(state["record_id"], contract=contract)
    assert result.status != "ready"


def test_failed_cold_resume_preserves_typed_failure(monkeypatch, tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        path.admit(store, profile.spans()[0], scope=profile.scope(), placement="cold")
        pending = path.read(store, "Badge", scope=profile.scope())
        assert isinstance(pending, PendingContextualRead)
        store.revoke_record("r1.span0", reason="rereview", scope=profile.scope())
        def fail_eviction(store, doc):
            raise RuntimeError("secondary eviction failure")
        monkeypatch.setattr(path, "_evict_failed_records", fail_eviction)
        with pytest.raises(ContextualReadFailure):
            path.resume(store, pending, scope=profile.scope(), wait_seconds=1)
    finally:
        store.close()
        path.close()


def test_failed_resume_does_not_re_read_healthy_payloads_uncharged(monkeypatch, tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        for index in range(2):
            path.admit(store, profile.spans(name=f"r{index}")[0],
                       scope=profile.scope(), placement="cold")
        pending = path.read(store, "Badge", scope=profile.scope())
        assert isinstance(pending, PendingContextualRead)
        store.revoke_record("r0.span0", reason="rereview", scope=profile.scope())
        import replica_lab.contextual_memory_v12 as module
        calls = []
        real = module._read_verified
        def counted(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)
        monkeypatch.setattr(module, "_read_verified", counted)
        with pytest.raises(ContextualReadFailure):
            path.resume(store, pending, scope=profile.scope(), wait_seconds=1)
        assert calls == []
    finally:
        store.close()
        path.close()


def test_density_profiler_empty_list_uses_disabled_parity():
    path = profile.build_path("two")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        ids, mask = profile.ids_and_mask(4)
        batch = path.training_values(profile.spans(), scope=profile.scope())
        result = profiler.forward(path, ids, mask, batch, scope=profile.scope(),
                                  active_sites=[])
        expected = path.forward(ids, mask, batch, scope=profile.scope(), active_sites=[])
        import torch
        assert torch.equal(result, expected)
    finally:
        profiler.close()
        path.close()


def test_configured_tolerance_is_refused_and_execution_is_bound():
    protocol = total_cost_protocol()
    protocol["selection"]["tolerance"]["relative"] = 0.5
    with pytest.raises(ValueError, match="selection/tolerance/rounding"):
        validate_protocol(protocol, comparison="preregistered_measured_total_cost")
    assert cost_tolerance(1000.0, 1.0, relative=0.5) == 500.0
    assert cost_tolerance(1000.0, 1.0, relative=0.02) == 20.0
