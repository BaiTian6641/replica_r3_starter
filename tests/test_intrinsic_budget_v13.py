"""Incremental v13 ledger parity, recovery and scaling tests; CPU only."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from replica_lab.intrinsic_memory_budget_v12 import BudgetError, COUNTERS, WorkLedger
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger


def plan():
    return {"stage": {**{key: 10000 for key in COUNTERS}, "wall_seconds": 120}}


def create(path, identity="incremental-fixture"):
    return IncrementalWorkLedger.create(path, identity=identity, plan=plan(),
                                        deadline=time.monotonic() + 120)


def test_own_mutations_do_not_replay_growing_journal(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = create(path)
    calls = 0
    original = ledger._read
    def counted():
        nonlocal calls
        calls += 1
        return original()
    monkeypatch.setattr(ledger, "_read", counted)
    ledger.enter("stage")
    for index in range(100):
        token = ledger.reserve("stage", f"operation-{index}", writes=1)
        ledger.finish(token, actual={"writes": 1})
    assert calls == 1  # the initial plan read only; every append is incremental
    assert ledger.snapshot(cached=True)["reserved"]["writes"] == 100
    assert len(ledger.snapshot(cached=True)["events"]) == 100
    with pytest.raises(BudgetError, match="already terminal"):
        ledger.finish(0)
    ledger.leave("stage")
    # Independent complete v12 replay must agree with the incremental state.
    assert WorkLedger(path, identity="incremental-fixture").snapshot() == \
        ledger.snapshot(cached=True)


def test_external_process_append_is_visible_and_charged_once(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = create(path, "cross-process")
    ledger.enter("stage")
    code = """import sys
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
ledger = IncrementalWorkLedger(sys.argv[1], identity='cross-process')
token = ledger.reserve('stage', 'child', writes=1)
ledger.finish(token, actual={'writes': 1})
"""
    environment = {**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=Path(__file__).parents[1],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    snapshot = ledger.snapshot(cached=True)
    assert snapshot["reserved"]["writes"] == 1
    assert snapshot["events"][0]["status"] == "completed"
    with pytest.raises(BudgetError, match="already terminal"):
        ledger.finish(0)


def test_replacement_truncation_and_in_place_mutation_are_refused(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = create(path, "replacement")
    ledger.enter("stage")
    ledger.snapshot(cached=True)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(BudgetError, match="replaced"):
        ledger.snapshot(cached=True)

    second = create(tmp_path / "second.jsonl", "truncation")
    second.enter("stage")
    second.snapshot(cached=True)
    os.truncate(second.path, 16)
    with pytest.raises(BudgetError, match="truncated"):
        second.snapshot(cached=True)

    third = create(tmp_path / "third.jsonl", "mutation")
    third.enter("stage")
    third.snapshot(cached=True)
    stat = third.path.stat()
    with third.path.open("r+b") as stream:
        stream.seek(stat.st_size - 2)
        stream.write(b" \n")
    with pytest.raises(BudgetError, match="modified in place"):
        third.snapshot(cached=True)


def test_torn_tail_requires_explicit_recovery_and_preserves_charge(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = create(path, "recovery")
    ledger.enter("stage")
    token = ledger.reserve("stage", "interrupted", writes=1)
    with path.open("ab") as stream:
        stream.write(b'{"t":"reserve"')
    with pytest.raises(BudgetError, match="torn journal tail"):
        ledger.snapshot()
    with pytest.raises(BudgetError, match="operator recovery"):
        ledger.recover_interrupted()
    receipt = ledger.recover_interrupted(accept_torn_tail=True)
    assert receipt["charged_events"] == [token]
    assert receipt["interrupted_stages"] == ["stage"]
    assert not receipt["refunded"] and receipt["torn_tail"]["bytes"] > 0
    snapshot = ledger.snapshot(cached=True)
    assert snapshot["events"][token]["status"] == "failed"
    assert snapshot["stages"]["stage"]["status"] == "interrupted"
    assert snapshot["reserved"]["writes"] == 1
    assert len(snapshot["torn_records"]) == 1


def test_restart_replays_exactly_once_without_duplicate_charging(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = create(path, "restart")
    ledger.enter("stage")
    for index in range(25):
        token = ledger.reserve("stage", f"event-{index}", generated_tokens=1)
        ledger.finish(token, actual={"generated_tokens": 1})
    before = ledger.snapshot(cached=True)
    restored = IncrementalWorkLedger(path, identity="restart")
    after = restored.snapshot(cached=True)
    assert after == before
    assert after["reserved"]["generated_tokens"] == 25
    with pytest.raises(BudgetError, match="already terminal"):
        restored.finish(0)
