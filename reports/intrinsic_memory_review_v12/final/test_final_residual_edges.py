"""Final independent residual reproductions; no development data or weights."""
import time
import json

import pytest

from replica_lab.intrinsic_memory_budget_v12 import WorkLedger
from replica_lab.intrinsic_memory_coverage_v12 import plan
from replica_lab.intrinsic_memory_diagnostic_v12 import DiagnosticEngineV12, StageRef
from replica_lab.intrinsic_memory_density_protocols_v12 import (
    measured_cost_target, total_cost_protocol, validate_protocol)
from replica_lab.successors.tiers import ProjectionCache, ReadContract, TieredBlobStore
import profile_intrinsic_memory_density_v12 as profile


def test_fitting_closed_receipt_precedes_stage_close(tmp_path):
    ledger = WorkLedger.create(tmp_path / "ledger.jsonl", identity="final-review",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    ledger.enter("fit_probes_512")
    engine = DiagnosticEngineV12.__new__(DiagnosticEngineV12)
    engine.ledger, engine.output = ledger, tmp_path
    engine._stage_ref = StageRef()
    engine.stage = "fit_probes_512"
    import types
    import torch
    engine.path = types.SimpleNamespace(adapter=types.SimpleNamespace(
        base=types.SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))))
    engine.write_receipt("fitting_closed", {"closed": True})
    ledger.leave("fit_probes_512", status="completed")
    assert (tmp_path / "fitting_closed.json").is_file()
    assert ledger.snapshot()["stages"]["fit_probes_512"]["status"] == "completed"


def test_external_p_revocation_blocks_second_instance_generic_read(tmp_path):
    first = TieredBlobStore(tmp_path)
    first.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="e1", lineage=["r1"])
    state = first.read_p("expert")
    manifest = state["manifest"]
    second = TieredBlobStore(tmp_path)
    first.revoke_p("expert", reason="external", authorized_by="host")
    contract = ReadContract(manifest["producer"], manifest["representation"],
                            manifest["scope_digest"])
    assert second.read(state["record_id"], contract=contract).status != "ready"


def test_revoke_p_clears_clocked_projection_and_replay(tmp_path):
    store = TieredBlobStore(tmp_path)
    store.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="e1", lineage=["r1"])
    record_id = store.read_p("expert")["record_id"]
    key = ProjectionCache.key(record_id, generation=1, producer="p:expert",
                              representation="e1", reader_release="r1", site=3,
                              eligibility_revision=0)
    store.projections.put(key, object())
    store.replay.register(record_id, generation=1, producer="p:expert")
    store.revoke_p("expert", reason="revoked", authorized_by="host")
    assert store.projections.get(key) is None
    assert not store.replay.eligible(record_id, generation=1, producer="p:expert")


def test_minimal_authority_cannot_bypass_manifest_contract(tmp_path):
    store = TieredBlobStore(tmp_path)
    store.publish("record", b"payload", generation=1, kind="e", producer="expected",
                  representation="r1", scope_digest="scope", valid_to=1)
    result = store.read("record", authority=lambda record_id: {"eligible": True})
    assert result.status != "ready"


def test_noninteger_exposure_is_refused_during_validation():
    protocol = total_cost_protocol()
    protocol["selection"]["declared_common_updates"] = 2.0
    with pytest.raises(ValueError):
        validate_protocol(protocol, comparison="preregistered_measured_total_cost")


def test_receipt_creation_does_not_leave_partial_final_file(monkeypatch, tmp_path):
    import fcntl
    from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
    destination = tmp_path / "receipt.json"
    real_fcntl = fcntl.fcntl
    def interrupted(descriptor, request):
        if request == getattr(fcntl, "F_FULLFSYNC", 51):
            raise RuntimeError("interrupted F_FULLFSYNC")
        return real_fcntl(descriptor, request)
    monkeypatch.setattr(fcntl, "fcntl", interrupted)
    with pytest.raises(RuntimeError, match="interrupted F_FULLFSYNC"):
        write_new_json(destination, {"complete": True})
    assert not destination.exists()
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []
    monkeypatch.setattr(fcntl, "fcntl", real_fcntl)
    write_new_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}


def test_cold_staging_propagates_process_control_exceptions(monkeypatch):
    import replica_lab.contextual_memory_v12 as module
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "_payload_read_verified", interrupted)
    ticket = {"pinned": [{"record_id": "r", "text_path": "missing",
                         "text_sha256": "0" * 64, "text_bytes": 1, "values": []}]}
    with pytest.raises(KeyboardInterrupt):
        module._cold_payloads(profile.ROOT, [ticket])
