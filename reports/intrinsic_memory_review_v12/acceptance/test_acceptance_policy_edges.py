"""Acceptance residual authority policy reproductions; CPU only."""
import json
import time
from replica_lab.successors.tiers import TieredBlobStore
from replica_lab.intrinsic_memory_density_v12 import assert_no_read_write_effects
from replica_lab.intrinsic_memory_budget_v12 import WorkLedger
from replica_lab.intrinsic_memory_coverage_v12 import plan
from run_spark_intrinsic_coverage_v12 import charged_write


def test_partial_authority_still_bypasses_manifest_identity(tmp_path):
    store = TieredBlobStore(tmp_path)
    store.publish("record", b"payload", generation=1, kind="e", producer="expected",
                  representation="r1", scope_digest="scope", valid_to=1)
    result = store.read("record", authority=lambda record_id: {
        "eligible": True, "producer": "expected"})
    assert result.status != "ready"


def test_cross_instance_restore_is_not_sticky(tmp_path):
    first = TieredBlobStore(tmp_path)
    first.publish_p("expert", weights=b"w", moments=b"m", clock=1,
                    encoder="e1", lineage=["r1"])
    second = TieredBlobStore(tmp_path)
    first.revoke_p("expert", reason="revoked", authorized_by="host")
    try:
        second.read_p("expert")
    except PermissionError:
        pass
    first.restore_p("expert", authorized_by="host")
    assert second.read_p("expert")["clock"] == 1



def test_source_recomputation_is_not_a_forbidden_side_effect():
    before = {"format": "review", "memory": {}, "source": {
        "requests": 0, "forwards": 0, "cache_misses": 0}}
    after = {"format": "review", "memory": {}, "source": {
        "requests": 1, "forwards": 1, "cache_misses": 1}}
    result = assert_no_read_write_effects(before, after)
    assert result["source"]["forwards"] == 1



def test_terminal_parent_receipts_remain_charged(tmp_path):
    ledger = WorkLedger.create(tmp_path / "ledger.jsonl", identity="acceptance-review",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    for stage in ("finalize_worker", "finalize_parent"):
        ledger.enter(stage)
        ledger.leave(stage, status="completed")
    charged = charged_write(ledger, tmp_path, "capacity_guard", {"window": "restart"})
    assert charged is True
    document = json.loads((tmp_path / "capacity_guard.json").read_text())
    assert document["receipt_charging"]["charged"] is True
