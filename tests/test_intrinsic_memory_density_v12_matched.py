"""Real v12 producer/store with exactly two canonical shared reader groups."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from replica_lab.contextual_memory import ContextualMemoryError, canonical_spans
from replica_lab.contextual_spark_fixture import (
    build_contextual_adapter, ids_and_mask, labels_from_ids,
)
from replica_lab.contextual_source_features_v12 import (
    FrozenSourceFeatures, SourceFeatureSpec, adapter_contract_digest,
    tokenizer_contract_digest,
)
from replica_lab.intrinsic_memory_density_v12 import (
    SiteInvocationProfiler, cost_snapshot, ledger_delta, parameter_audit,
)
from replica_lab.intrinsic_memory_matched_reader_v12 import (
    MatchedContextualMemoryV12, matched_v12_layout,
)
from test_intrinsic_memory_density_v12_producer import (
    FixtureAccounting, FixtureTokenizer, forward, make as make_independent,
    spans, step, training_batch,
)
from test_contextual_memory_layout_v4 import assert_tree_equal
from test_tiered_memory import open_store, scope

ARMS = {"two": (3, 7),
        "full_attention_nine": (3, 7, 11, 15, 19, 23, 27, 31, 35),
        "all": tuple(range(36))}
EXPECTED = {name: len(sites) for name, sites in ARMS.items()}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(arm="all"):
    adapter, *_ = build_contextual_adapter(
        seed=17, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256)
    adapter.base.eval()
    for parameter in adapter.base.parameters():
        parameter.requires_grad_(False)
    tokenizer, accounting = FixtureTokenizer(), FixtureAccounting()
    spec = SourceFeatureSpec(
        original_parameters_sha256=hashlib.sha256(b"cpu_density_v12_original").hexdigest(),
        source_execution_sha256=adapter_contract_digest(adapter),
        tokenizer_sha256=tokenizer_contract_digest(tokenizer),
        hidden_size=adapter.config.geometry.hidden_size,
        max_tokens=64, cache_bytes=1024 * 1024, cache_entries=8)
    provider = FrozenSourceFeatures(adapter, tokenizer, spec, accounting=accounting)
    path = MatchedContextualMemoryV12(
        adapter, matched_v12_layout(arm, hidden_size=8),
        source_identity="cpu_density_v12_fixture", feature_provider=provider,
        lr=0.001, initialization_seed=17,
        execution_identity="CPU_density_v12_engineering")
    path.adapter.extension.spine = None
    return path


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_matched_v12_has_exactly_two_groups_no_alias_state_or_ownership(arm):
    path = make(arm)
    try:
        assert path.spec.sites == ARMS[arm]
        audit = parameter_audit(path)
        assert audit["ownership_exact"] is True
        assert audit["duplicate_ownership"] == 0
        assert audit["missing_ownership"] == 0
        assert audit["extra_ownership"] == 0
        assert audit["duplicate_intended_registration"] == 0
        assert audit["reader_groups"]["groups"] == 2
        assert audit["reader_groups"]["alias_duplicated_state_tensors"] == 0
        assert audit["reader_groups"]["state_tensors"] == audit["reader_groups"][
            "physical_state_tensors"] == 22
        assert audit["reader_parameters"] == path.spec.reader_parameter_count()
        assert audit["reader_parameters"] == 272_418
        assert set(path.reader.state_dict()) == {
            f"groups.{group}.{name}"
            for group in (0, 1) for name in path.reader.groups[str(group)].state_dict()}
        assert set(path.reader.groups) == {"0", "1"}
        assert len({id(module) for module in path.reader.groups.values()}) == 2
        assert len(path.reader.ports) == EXPECTED[arm]
        assert len({id(module) for module in path.reader.ports.values()}) == 2
        assert {str(site): (site // 4) % 2 for site in path.spec.sites} == (
            {str(site): (site // 4) % 2 for site in ARMS[arm]})
        assert all(p.grad is None for p in path.adapter.base.parameters())
        assert path.adapter.base_unchanged()
    finally:
        path.close()


@pytest.mark.parametrize("arm", ("two", "full_attention_nine"))
def test_matched_v12_two_spans_use_one_snapshot_and_eight_slots(arm):
    path = make(arm)
    try:
        first = spans("Badge means red.", name="m1", available=5)[0]
        second = spans("Badge expires tomorrow.", name="m2", available=6)[0]
        batch = path.training_values((first, second), scope=scope())
        ids, mask = ids_and_mask(6)
        profiler = SiteInvocationProfiler(path, expected_groups=2)
        try:
            before = cost_snapshot(path)
            logits = profiler.forward(path, ids, mask, batch, scope=scope())
            after = cost_snapshot(path)
            assert tuple(logits.shape)[:2] == (1, 6)
            delta = ledger_delta(before, after)
            assert delta["memory"]["snapshot_copies"] == 1
            assert after["memory"]["value_slots_read"] - before["memory"]["value_slots_read"] == 8
            summary = profiler.summary()
            assert summary["total_site_invocations"] == EXPECTED[arm]
            assert len(summary["group_invocations"]) == 2
            assert all(event["output_requires_grad"] for event in summary["events"])
        finally:
            profiler.close()
    finally:
        path.close()


@pytest.mark.parametrize("arm", ("two", "full_attention_nine", "all"))
def test_matched_v12_boundary_accepts_canonical_sharing(arm):
    from replica_lab.intrinsic_memory_density_v12 import DensityProducerBoundary
    path = make(arm)
    try:
        boundary = DensityProducerBoundary(path, comparison="matched_unique_parameters")
        assert boundary.comparison == "matched_unique_parameters"
        assert len(boundary.group_names) == 2
        assert boundary.audit["reader_groups"]["alias_duplicated_state_tensors"] == 0
        assert boundary.identity()["optimizer_ownership"]["ownership_exact"] is True
    finally:
        path.close()


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_matched_v12_all_site_gradients_after_zero_projection_open(arm):
    path = make(arm)
    step(path)
    step(path)
    ids, mask = ids_and_mask(4)
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    activations, addresses = [], []

    def retain(module, args, output):
        output.retain_grad()
        activations.append(output)
        addresses.append(args[1].data_ptr())

    hooks = [module.register_forward_hook(retain)
             for module in path.reader.groups.values()]
    try:
        live = training_batch(path)
        live.slots.retain_grad()
        path.optimizer.zero_grad(set_to_none=True)
        logits = profiler.forward(path, ids, mask, live, scope=scope())
        F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
        profiler.record_backward(calls=1, processed_positions=int(ids.numel()))
        assert live.slots.grad is not None and live.slots.grad.abs().sum() > 0
        assert len(activations) == EXPECTED[arm]
        assert len(set(addresses)) == 1
        assert all(value.grad is not None and value.grad.abs().sum() > 0
                   for value in activations)
        summary = profiler.summary()
        assert summary["total_site_invocations"] == EXPECTED[arm]
        assert summary["backward_work"]["calls"] == 1
        assert parameter_audit(path)["ownership_exact"] is True
        assert all(p.grad is None for p in path.adapter.base.parameters())
        assert path.adapter.base_unchanged()
    finally:
        for hook in hooks:
            hook.remove()
        profiler.close()
        path.close()


def test_matched_v12_two_site_equals_independent_v12_initialization_and_updates():
    shared, reference = make("two"), make_independent("two")
    try:
        assert_tree_equal(shared.encoder.state_dict(), reference.encoder.state_dict())
        assert_tree_equal(shared.order.state_dict(), reference.order.state_dict())
        for group, site in ((0, 3), (1, 7)):
            assert_tree_equal(shared.reader.groups[str(group)].state_dict(),
                              reference.reader.ports[str(site)].state_dict())
        batch = training_batch(shared)
        reference_batch = training_batch(reference)
        assert torch.equal(forward(shared, batch), forward(reference, reference_batch))
        step(shared); step(reference)
        step(shared); step(reference)
        assert torch.equal(forward(shared, batch), forward(reference, reference_batch))
        assert shared.adapter.base_unchanged() and reference.adapter.base_unchanged()
    finally:
        shared.close(); reference.close()


@pytest.mark.parametrize("change", ("correction", "revocation"))
def test_matched_v12_stale_snapshot_refused_after_authority_change(tmp_path, change):
    path, store = make("two"), open_store(tmp_path)
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        original = spans("Badge means red.", name="r1")[0]
        assert path.admit(store, original, scope=scope()).outcome.ok
        batch = path.read(store, "Badge", scope=scope())
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, batch, scope=scope())
        proxy = profiler.instrument(snapshot, forward_id="authority")
        proxy.delta(3, torch.zeros(1, 4, 8))
        if change == "correction":
            corrected = replace(
                spans("Badge means blue.", name="r2", available=6)[0],
                source_epoch=2, supersedes=original.record_id)
            assert path.correct(store, original.record_id, corrected,
                                scope=scope()).outcome.ok
        else:
            store.revoke_record(original.record_id, reason="withdrawn", scope=scope())
        before = cost_snapshot(path)
        with pytest.raises(ContextualMemoryError):
            proxy.delta(7, torch.zeros(1, 4, 8))
        after = cost_snapshot(path)
        assert ledger_delta(before, after)["memory"]["retrieval_calls"] == 0
        assert ledger_delta(before, after)["memory"]["snapshot_copies"] == 0
        assert parameter_audit(path)["ownership_exact"] is True
    finally:
        profiler.close(); path.close(); store.close()


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_matched_v12_fresh_process_exact_continuation(tmp_path, arm):
    path = make(arm)
    step(path)
    batch = training_batch(path)
    checkpoint = tmp_path / "matched"
    path.save(checkpoint)
    before = forward(path, batch).detach()
    step(path)
    expected_weights = {key: value.clone() for key, value in path._tensors().items()}
    expected_optimizer = path.optimizer.state_dict()
    path.close()
    output = tmp_path / "continued.pt"
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys, torch
from test_intrinsic_memory_density_v12_matched import make, step, forward, training_batch
torch.set_num_threads(1)
p=make(sys.argv[1]);p.load(sys.argv[2])
b=training_batch(p);logits=forward(p,b).detach();step(p)
assert p.adapter.base_unchanged();p.assert_optimizer_ownership()
torch.save({'logits':logits,'weights':p._tensors(),'optimizer':p.optimizer.state_dict()},sys.argv[3])
p.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", child, arm, str(checkpoint), str(output)],
        cwd=root,
        env={**os.environ,
             "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "tests"),
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    produced = torch.load(output, weights_only=True)
    assert torch.equal(produced["logits"], before)
    assert_tree_equal(produced["weights"], expected_weights)
    assert_tree_equal(produced["optimizer"], expected_optimizer)


def test_matched_v12_outer_checkpoint_and_independent_v12_are_not_interchangeable(tmp_path):
    matched = make("two")
    try:
        matched.save(tmp_path / "matched")
    finally:
        matched.close()
    with pytest.raises(ContextualMemoryError):
        make_independent("two").load(tmp_path / "matched")
    independent, checkpoint = make_independent("two"), tmp_path / "independent"
    try:
        independent.save(checkpoint)
    finally:
        independent.close()
    with pytest.raises(ContextualMemoryError):
        make("two").load(checkpoint)
