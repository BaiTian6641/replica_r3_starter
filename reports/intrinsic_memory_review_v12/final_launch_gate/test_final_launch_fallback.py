"""Failed-persist fallback accounting reproduction; no model or labels."""
import json

import pytest

from replica_lab.contextual_memory_v12 import ContextualReadFailure
import profile_intrinsic_memory_density_v12 as profile


def test_failed_persist_fallback_charges_quota_and_receipt_bytes(monkeypatch, tmp_path):
    path = profile.build_path("two")
    store = profile.open_store(tmp_path / "store")
    try:
        path.admit(store, profile.spans()[0], scope=profile.scope(), placement="cold")
        pending = path.read(store, "Badge", scope=profile.scope())
        store.revoke_record("r1.span0", reason="launch_gate_test", scope=profile.scope())
        original = path._persist
        calls = {"count": 0}
        def fail_once(store, token, doc):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("injected charged persistence failure")
            return original(store, token, doc)
        monkeypatch.setattr(path, "_persist", fail_once)
        with pytest.raises(ContextualReadFailure):
            path.resume(store, pending, scope=profile.scope(), wait_seconds=1)
        ticket_path = path._ticket_path(store, pending.ticket_id)
        receipt = json.loads(ticket_path.read_text())
        actual = ticket_path.stat().st_size
        assert receipt["status"] == "failed"
        assert receipt["costs"]["ticket_write_calls"] == 3
        assert receipt["costs"]["ticket_write_bytes_reserved"] >= actual
        assert path.cost["durable_write_bytes"] >= actual
        charged = store.connection.execute(
            "SELECT bytes FROM tier_usage WHERE component_id=?",
            (f"contextual_ticket:{pending.ticket_id}:write",)).fetchone()
        assert charged["bytes"] == actual
    finally:
        store.close()
        path.close()
