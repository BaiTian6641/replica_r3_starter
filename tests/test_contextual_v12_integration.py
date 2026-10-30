"""Integrated v12 CPU evidence: fresh process, work ledger, 36-block path, arms."""
import json
import time

import pytest
import torch

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_source_features_v12 import WorkLedgerSourceAccounting
from replica_lab.contextual_v12_fixture import (
    ENGINEERING_LABEL, SyntheticWorkAccounting, build_v12_stack, canonical_span,
    logits_digest, model_state_digest, open_v12_store, optimizer_digest, scope,
    spans_for)
from replica_lab.intrinsic_memory_budget_v5 import COUNTERS, WorkLedger
from replica_lab.tiered_memory import MemoryScope
from v12_fixtures import run_child, step, token_inputs


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(**kwargs):
    kwargs.setdefault("layers", 36)
    kwargs.setdefault("sites", (3, 7))
    return build_v12_stack(**kwargs)


def inputs():
    return token_inputs()


def test_fresh_process_restores_logits_and_matches_uninterrupted_update(tmp_path):
    ids, mask, labels = inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    adapter, layout, provider, path = make()
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    path.save(tmp_path / "checkpoint")
    with torch.no_grad():
        expected = logits_digest(path.forward(
            ids, mask, path.training_values(spans, scope=sc), scope=sc))
    loss = step(path, adapter, spans, ids=ids, labels=labels, sc=sc)[2]
    parent = {"pre_logits": expected, "loss": float(loss.detach()),
              "state": model_state_digest(path), "moments": optimizer_digest(path.optimizer)}
    (tmp_path / "parent.json").write_text(json.dumps(parent))
    path.close()

    code = '''
import json, sys, torch
from pathlib import Path
torch.set_num_threads(1)
from replica_lab.contextual_v12_fixture import (build_v12_stack, spans_for, scope,
    logits_digest, model_state_digest, optimizer_digest)
root = Path(sys.argv[1])
ids = torch.tensor([[3,5,7,9,11,13]], dtype=torch.long); mask = torch.ones_like(ids)
labels = ids.clone(); labels[:, 0] = -100
spans = spans_for(["the badge is red", "the badge is blue"])
adapter, layout, provider, path = build_v12_stack(layers=36, sites=(3, 7))
path.load(root / "checkpoint")
with torch.no_grad():
    pre = logits_digest(path.forward(ids, mask,
        path.training_values(spans, scope=scope()), scope=scope()))
path.optimizer.zero_grad(set_to_none=True)
batch = path.training_values(spans, scope=scope())
logits = path.forward(ids, mask, batch, scope=scope())
loss = adapter.causal_loss(logits, ids, labels)
loss.backward(); path.optimizer.step()
root.joinpath("child.json").write_text(json.dumps({"pre_logits": pre,
    "loss": float(loss.detach()), "state": model_state_digest(path),
    "moments": optimizer_digest(path.optimizer)}))
'''
    result = run_child(code, tmp_path)
    assert result.returncode == 0, result.stderr
    child = json.loads((tmp_path / "child.json").read_text())
    assert child["pre_logits"] == parent["pre_logits"]
    assert child["loss"] == parent["loss"]
    assert child["state"] == parent["state"]
    assert child["moments"] == parent["moments"]


def test_work_ledger_admits_source_work_before_dispatch_and_refuses_over_cap(tmp_path):
    plan = {"mechanics": {**{key: 10000 for key in COUNTERS},
                          "artifact_bytes": 16 * 1024**2, "wall_seconds": 120}}
    ledger = WorkLedger.create(tmp_path / "ledger.json", identity="v12_cpu_ledger",
                               plan=plan, deadline=time.monotonic() + 120)
    ledger.enter("mechanics")
    accounting = WorkLedgerSourceAccounting(ledger, "mechanics")
    ids, mask, labels = inputs()
    sc = scope()
    adapter, layout, provider, path = make(accounting=accounting)
    step(path, adapter, spans_for(["the badge is red"]), ids=ids, labels=labels, sc=sc)
    ledger.leave("mechanics")
    events = ledger.snapshot()["events"]
    operations = [event["operation"] for event in events]
    assert "source_tokenize" in operations and "source_forward" in operations
    for event in events:
        assert event["status"] == "completed"
        assert set(event["actual"]).issubset(event["reserved"])
        assert all(event["actual"][key] <= event["reserved"][key]
                   for key in event["actual"])
    reserved = ledger.snapshot()["reserved"]
    assert reserved["source_encodes"] >= 1 and reserved["forwards"] >= 1
    assert reserved["input_tokens"] >= 1

    capped = {"mechanics": {**{key: 10000 for key in COUNTERS},
                            "artifact_bytes": 16 * 1024**2, "wall_seconds": 120}}
    capped["mechanics"]["forwards"] = 0
    second = WorkLedger.create(tmp_path / "ledger2.json", identity="v12_cpu_capped",
                               plan=capped, deadline=time.monotonic() + 120)
    second.enter("mechanics")
    adapter2, layout2, provider2, path2 = make(
        accounting=WorkLedgerSourceAccounting(second, "mechanics"))
    with pytest.raises(Exception):
        provider2.features(canonical_span("the badge is red", record_id="left")[0])
    assert provider2.cost["forwards"] == 0
    second.leave("mechanics")


def test_three_density_arms_share_producer_and_keep_one_owner(tmp_path):
    ids, mask, labels = inputs()
    sc = scope()
    spans = spans_for(["the badge is red", "the badge is blue"])
    arms = [(3, 7), (3, 7, 11, 15, 19, 23, 27, 31, 35), tuple(range(36))]
    producer_states, order_states = [], []
    for sites in arms:
        adapter, layout, provider, path = make(sites=sites)
        declared = path.assert_optimizer_ownership()
        assert set(declared) == {"producer", "order", "reader"}
        assert path.reader._optimizer_owner is path.optimizer
        assert adapter.base_unchanged()
        producer_states.append({name: value.clone()
                                for name, value in path.encoder.state_dict().items()})
        order_states.append(path.order.weight.detach().clone())
        batch = path.training_values(spans, scope=sc)
        before = path.cost["snapshot_copies"]
        with torch.no_grad():
            path.forward(ids, mask, batch, scope=sc)
        assert path.cost["snapshot_copies"] - before == 1
        assert path.cost["value_slots_read"] == 8
        step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
        assert path.cost["retrieval_calls"] == 0 and path.cost["admission_calls"] == 0
    for state in producer_states[1:]:
        assert set(state) == set(producer_states[0])
        assert all(torch.equal(value, producer_states[0][name])
                   for name, value in state.items())
    for order in order_states[1:]:
        assert torch.equal(order, order_states[0])


def test_36_block_end_to_end_synthetic_protocol(tmp_path):
    ids, mask, labels = inputs()
    sc = scope()
    sites = tuple(range(36))
    adapter, layout, provider, path = make(sites=sites)
    spans = spans_for(["the bright badge is clearly red",
                       "the bright badge is clearly blue"])
    for _ in range(8):
        step(path, adapter, spans, ids=ids, labels=labels, sc=sc)
    assert adapter.base_unchanged()
    assert all(torch.isfinite(value).all() for value in path._tensors().values())
    assert path.cost["snapshot_copies"] == 8

    correct = path.training_values(spans, scope=sc)
    shuffled = path.training_values(tuple(reversed(spans)), scope=sc)
    absent = None
    with torch.no_grad():
        digests = {logits_digest(path.forward(ids, mask, batch, scope=sc))
                   for batch in (correct, shuffled, absent)}
    assert len(digests) == 3

    store = open_v12_store(tmp_path)
    path.admit(store, spans[0], scope=sc)
    durable = path.read(store, "badge", scope=sc)
    with torch.no_grad():
        fresh = path.training_values((spans[0],), scope=sc)
        assert torch.equal(path.forward(ids, mask, fresh, scope=sc),
                           path.forward(ids, mask, durable, scope=sc))
    path.save(tmp_path / "checkpoint")
    _, _, _, reloaded = make(sites=sites)
    reloaded.load(tmp_path / "checkpoint")
    with torch.no_grad():
        assert torch.equal(path.forward(ids, mask, fresh, scope=sc),
                           reloaded.forward(ids, mask,
                                            reloaded.training_values((spans[0],), scope=sc),
                                            scope=sc))
    _, _, _, other = make(layers=36, sites=(3, 7), tokenizer_mode="words")
    with pytest.raises(ContextualMemoryError):
        other.load(tmp_path / "checkpoint")
    store.close()
    assert ENGINEERING_LABEL == "engineering_cpu_v12_contextual_fixture_only"
    assert not hasattr(path, "native_admissible")
