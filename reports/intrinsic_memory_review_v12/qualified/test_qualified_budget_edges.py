"""Qualified native budget-edge reproductions; no model or labels."""
import time
import types

from replica_lab.intrinsic_memory_coverage_v12 import FIT_CASES, plan
from replica_lab.intrinsic_memory_budget_v12 import WorkLedger
from replica_lab.intrinsic_memory_diagnostic_v12 import DiagnosticEngineV12, StageRef
from replica_lab.contextual_memory_v5 import PendingContextualRead


def test_final_probe_cap_covers_fitting_close_and_source_receipt():
    required = FIT_CASES * (4 + 3 + 2) + 4
    assert plan("two")["fit_probes_512"]["writes"] >= required


def test_read_resume_write_reservation_covers_four_waits(tmp_path):
    ledger = WorkLedger.create(tmp_path / "ledger.jsonl", identity="budget-review",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    ledger.enter("prepare")
    pending = PendingContextualRead("a" * 32, str(tmp_path), {}, "0" * 64, {})
    class Path:
        release_corpus = None
        def __init__(self):
            self.calls = 0
            self.cost = {}
            self.adapter = types.SimpleNamespace(base=types.SimpleNamespace(
                parameters=lambda: iter([__import__("torch").zeros(1)])))
        def next(self):
            self.calls += 1
            if self.calls <= 4:
                return pending
            return types.SimpleNamespace(read_costs={})
        def read(self, store, question, *, scope, wait_seconds=0):
            return self.next()
        def resume(self, store, ticket, *, scope, wait_seconds=0):
            return self.next()
    engine = DiagnosticEngineV12.__new__(DiagnosticEngineV12)
    engine.ledger = ledger
    engine._stage_ref = StageRef()
    engine.stage = "prepare"
    engine.path = Path()
    engine.read(None, "scope", "question")
    reserved = ledger.snapshot()["reserved"]["writes"]
    assert reserved <= 5
