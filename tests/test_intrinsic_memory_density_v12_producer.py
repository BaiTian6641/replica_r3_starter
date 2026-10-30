"""Direct binding of the real v12 producer/store to density instrumentation.

This is the acceptance path the density track must exercise: the reviewed
``contextual_v12_fixture`` builds the frozen-source provider, the actual v12
producer and a tiered store; ``DensityProducerBoundary`` validates the public
surface, feature contract and exact optimizer ownership, and
``SiteInvocationProfiler`` records site dispatches against physical groups.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
from replica_lab.contextual_memory import ContextualMemoryError, canonical_spans
from replica_lab.contextual_memory_v12 import ContextualMemoryV12
from replica_lab.contextual_source_features_v12 import (
    FrozenSourceFeatures, SourceFeatureSpec, adapter_contract_digest,
    tokenizer_contract_digest,
)
from replica_lab.contextual_spark_fixture import (
    build_contextual_adapter, ids_and_mask, labels_from_ids,
)
from replica_lab.intrinsic_memory_density_v12 import (
    DensityProducerBoundary, ProducerBoundaryViolation, SiteInvocationProfiler,
    cost_snapshot, ledger_delta, parameter_audit,
)
from replica_lab.intrinsic_memory_density_protocols_v12 import (
    fixed_site_protocol, matched_protocol,
)
from test_contextual_memory_layout_v4 import assert_tree_equal
from test_tiered_memory import open_store, scope

ARMS = {"two": (3, 7),
        "full_attention_nine": tuple(range(3, 36, 4)),
        "all": tuple(range(36))}
EXPECTED = {name: len(sites) for name, sites in ARMS.items()}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixtureTokenizer:
    """Deterministic synthetic tokenizer with an immutable declared identity."""

    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        ids = [3 + (ord(character) % 29) for character in text]
        return {"input_ids": ids}

    def identity(self):
        return {"format": "replica.cpu_density_v12.synthetic_tokenizer.v1",
                "vocab_size": 32, "special_tokens": [0, 1, 2]}


class FixtureAccounting:
    """Explicit pre-dispatch source work accounting for the CPU fixture."""

    def __init__(self):
        self.counter = 0
        self.requests = []

    def reserve(self, request):
        self.counter += 1
        self.requests.append(request.operation)
        return self.counter

    def complete(self, token, *, status="completed", actual=None, detail=None):
        if status not in ("completed", "failed"):
            raise ContextualMemoryError("invalid fixture accounting status")


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
    path = ContextualMemoryV12(
        adapter, spark_layout(arm, hidden_size=adapter.config.geometry.hidden_size),
        source_identity="cpu_density_v12_fixture", feature_provider=provider,
        lr=0.001, initialization_seed=17,
        execution_identity="CPU_density_v12_engineering")
    path.adapter.extension.spine = None
    return path


def spans(text="Badge means red.", *, name="r1", available=5):
    return canonical_spans(text, record_id=name, source_id="source",
                           source_epoch=1, available_at=available)


def training_batch(path, text="Badge means red.", name="r1"):
    return path.training_values(spans(text, name=name), scope=scope())


def step(path, batch=None):
    ids, mask = ids_and_mask(4)
    path.optimizer.zero_grad(set_to_none=True)
    live = (training_batch(path) if batch is None
            else path.training_from_stored(batch, scope=scope()))
    live.slots.retain_grad()
    logits = path.forward(ids, mask, live, scope=scope())
    F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
    path.optimizer.step()
    return live.slots.grad


def forward(path, batch):
    ids, mask = ids_and_mask(4)
    return path.forward(ids, mask, batch, scope=scope())


def test_real_v12_independent_reader_refuses_a_matched_claim():
    path = make("all")
    try:
        with pytest.raises(ProducerBoundaryViolation, match="physical groups"):
            DensityProducerBoundary(path, comparison="matched_unique_parameters")
        assert DensityProducerBoundary(
            path, comparison="fixed_per_site").comparison == "fixed_per_site"
    finally:
        path.close()


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_real_v12_boundary_parameter_counts_and_profiled_sites(arm):
    path = make(arm)
    boundary = DensityProducerBoundary(path, comparison="fixed_per_site")
    try:
        audit = boundary.audit
        assert audit["ownership_exact"] is True
        assert audit["duplicate_ownership"] == 0
        assert audit["missing_ownership"] == 0
        assert audit["extra_ownership"] == 0
        assert audit["base_parameters_optimizer_owned"] == 0
        assert audit["base_parameters_trainable"] == 0
        assert audit["reader_groups"]["groups"] == EXPECTED[arm]
        assert audit["reader_parameters"] == path.spec.reader_parameter_count()
        assert audit["intended_parameter_tensors"] == len(
            [p for module in path.modules().values() for p in module.parameters()])
        assert audit["optimizer_parameter_tensors"] == audit["intended_parameter_tensors"]
        assert audit["base_parameters"] > 0
        assert boundary.identity()["feature_contract"]["original_updates"] == 0
        assert boundary.identity()["comparison"] == "fixed_per_site"
        batch = training_batch(path)
        profiler = boundary.profiler(expected_groups=EXPECTED[arm])
        try:
            before = cost_snapshot(path)
            logits = profiler.forward(path, *ids_and_mask(4), batch, scope=scope())
            after = cost_snapshot(path)
            assert tuple(logits.shape)[:2] == (1, 4)
            summary = profiler.summary()
            assert summary["physical_groups"] == EXPECTED[arm]
            assert summary["total_site_invocations"] == EXPECTED[arm]
            assert all(value == 1 for value in summary["site_invocations"].values())
            assert len(summary["group_invocations"]) == EXPECTED[arm]
            delta = ledger_delta(before, after)
            assert delta["memory"]["snapshot_copies"] == 1
            assert delta["memory"]["retrieval_calls"] == 0
            assert delta["memory"]["admission_calls"] == 0
            assert delta["memory"]["durable_write_bytes"] == 0
            assert summary["callbacks_per_physical_group"] == 2
            assert summary["cache_bytes"]["available"] is True
            assert summary["cache_bytes"]["max_observed"] > 0
            assert summary["allocator"]["available"] is False
        finally:
            profiler.close()
    finally:
        path.close()


@pytest.mark.parametrize("arm", ("two", "full_attention_nine"))
def test_real_v12_store_read_profile_correction_and_revocation(tmp_path, arm):
    path = make(arm)
    store = open_store(tmp_path)
    profiler = SiteInvocationProfiler(path, expected_groups=EXPECTED[arm])
    try:
        original = spans("Badge means red.", name="r1")[0]
        assert path.admit(store, original, scope=scope()).outcome.ok
        batch = path.read(store, "Badge", scope=scope())
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, batch, scope=scope())
        proxy = profiler.instrument(snapshot, forward_id="durable-1")
        logits = path.adapter.forward_text(
            ids, ids_and_mask(4)[1], use_projector=False, use_ports=False,
            use_spine=False, intrinsic_memory=proxy)
        assert tuple(logits.shape)[:2] == (1, 4)
        assert profiler.summary()["total_site_invocations"] == EXPECTED[arm]
        corrected = replace(
            spans("Badge means blue.", name="r2", available=6)[0],
            source_epoch=2, supersedes=original.record_id)
        assert path.correct(store, original.record_id, corrected, scope=scope()).outcome.ok
        before = cost_snapshot(path)
        with pytest.raises(ContextualMemoryError):
            snapshot.delta(3, torch.zeros(1, 4, 8))
        assert ledger_delta(before, cost_snapshot(path))["memory"]["retrieval_calls"] == 0
        fresh = path.read(store, "Badge", scope=scope())
        assert any("blue" in span.text for span in fresh.spans)
        store.revoke_record(corrected.record_id, reason="withdrawn", scope=scope())
        with pytest.raises(ContextualMemoryError):
            path.read(store, "Badge", scope=scope())
    finally:
        profiler.close()
        path.close()
        store.close()


@pytest.mark.parametrize("arm", ("two", "full_attention_nine"))
def test_real_v12_two_spans_fill_eight_slots_with_one_snapshot(arm):
    path = make(arm)
    try:
        one = spans("Badge means red.", name="a1", available=5)[0]
        two = spans("Badge expires tomorrow.", name="a2", available=6)[0]
        batch = path.training_values((one, two), scope=scope())
        ids, _ = ids_and_mask(6)
        profiler = SiteInvocationProfiler(path, expected_groups=EXPECTED[arm])
        try:
            before = cost_snapshot(path)
            logits = profiler.forward(path, ids, ids_and_mask(6)[1], batch,
                                      scope=scope())
            after = cost_snapshot(path)
            assert tuple(logits.shape)[:2] == (1, 6)
            assert ledger_delta(before, after)["memory"]["snapshot_copies"] == 1
            assert after["memory"]["value_slots_read"] - before["memory"]["value_slots_read"] == 8
            assert profiler.summary()["total_site_invocations"] == EXPECTED[arm]
        finally:
            profiler.close()
    finally:
        path.close()


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_real_v12_all_site_gradients_after_projection_open(arm):
    path = make(arm)
    step(path)
    step(path)
    ids, mask = ids_and_mask(4)
    profiler = SiteInvocationProfiler(path, expected_groups=EXPECTED[arm])
    activations = []

    def retain(module, args, output):
        output.retain_grad()
        activations.append(output)

    hooks = [module.register_forward_hook(retain)
             for module in profiler.groups.values()]
    try:
        live = training_batch(path)
        live.slots.retain_grad()
        path.optimizer.zero_grad(set_to_none=True)
        logits = profiler.forward(path, ids, mask, live, scope=scope())
        F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
        profiler.record_backward(calls=1, processed_positions=int(ids.numel()))
        assert live.slots.grad is not None and live.slots.grad.abs().sum() > 0
        assert profiler.summary()["backward_work"]["calls"] == 1
        assert len(activations) == EXPECTED[arm]
        assert all(value.grad is not None and value.grad.abs().sum() > 0
                   for value in activations)
        audit = parameter_audit(path)
        assert audit["ownership_exact"] is True
        assert all(p.grad is None for p in path.adapter.base.parameters())
        assert path.adapter.base_unchanged()
    finally:
        for hook in hooks:
            hook.remove()
        profiler.close()
        path.close()


def test_protocol_parameter_counts_match_the_bound_real_reader_formula():
    protocol = fixed_site_protocol()
    per_site = 2 * 2560 * 256 + 2 * 256 * 256 + 2 * 2560 + 4 * 256 + 1
    for arm in ARMS:
        row = protocol["parameter_accounting"]["arms"][arm]
        assert row["unique_reader_parameters"] == per_site * len(ARMS[arm])
        assert row["reader_parameter_tensors"] == 11 * len(ARMS[arm])
    matched = matched_protocol()["parameter_accounting"]
    assert matched["unique_reader_parameters"] == 2_895_874
    assert matched["site_aliases_by_arm"] == {
        "two": 2, "full_attention_nine": 9, "all": 36}


@pytest.mark.parametrize("arm", ("two", "all"))
def test_real_v12_fresh_process_checkpoint_continuation(tmp_path, arm):
    path = make(arm)
    step(path)
    batch = training_batch(path)
    checkpoint = tmp_path / "extension"
    path.save(checkpoint)
    before = forward(path, batch).detach()
    step(path)
    expected_weights = {k: v.clone() for k, v in path._tensors().items()}
    expected_optimizer = path.optimizer.state_dict()
    path.close()
    output = tmp_path / "continued.pt"
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys, torch
from test_intrinsic_memory_density_v12_producer import make, step, forward, training_batch
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
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    produced = torch.load(output, weights_only=True)
    assert torch.equal(produced["logits"], before)
    assert_tree_equal(produced["weights"], expected_weights)
    assert_tree_equal(produced["optimizer"], expected_optimizer)
