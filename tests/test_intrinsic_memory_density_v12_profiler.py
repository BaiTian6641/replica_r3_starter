"""Invocation profiler checks on the reviewed matched-reader CPU fixture.

These tests exercise the existing v5 producer/matched reader, not a stub.  They
verify that one hook pair per physical group records all site invocations, that
forward/recompute identity is honest, and that recomputation creates no
retrieval/write/snapshot side effects.  They are engineering checks on a random
36-block fixture; they are not native density, competence or performance.
"""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_spark_fixture import (
    build_contextual_adapter, ids_and_mask, labels_from_ids,
)
from replica_lab.intrinsic_memory_density_v12 import (
    DensityError, SiteInvocationProfiler, assert_no_read_write_effects,
    cost_snapshot, ledger_delta, parameter_audit,
)
from replica_lab.intrinsic_memory_matched_reader_v1 import (
    ContextualMemoryMatchedV1, matched_layout,
)
from replica_lab.optimizers import NativeFusedAdamW
from test_contextual_memory_v5 import spans
from test_tiered_memory import open_store, scope

ARMS = ("two", "full_attention_nine", "all")
EXPECTED = {"two": 2, "full_attention_nine": 9, "all": 36}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(arm="all"):
    adapter, *_ = build_contextual_adapter(
        seed=17, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256)
    path = ContextualMemoryMatchedV1(
        adapter, matched_layout(arm, hidden_size=8),
        source_identity="random_fixture17", lr=0.001)
    adapter.extension.spine = None
    return path


def training_batch(path):
    return path.training_values(spans(), scope=scope())


def step(path, batch=None):
    ids, mask = ids_and_mask(4)
    path.optimizer.zero_grad(set_to_none=True)
    live = training_batch(path) if batch is None else path.training_from_stored(
        batch, scope=scope())
    live.slots.retain_grad()
    logits = path.forward(ids, mask, live, scope=scope())
    F.cross_entropy(logits.flatten(0, 1), labels_from_ids(ids).flatten()).backward()
    path.optimizer.step()
    return live.slots.grad


def forward(path, batch, **kwargs):
    ids, mask = ids_and_mask(4)
    return path.forward(ids, mask, batch, scope=scope(), **kwargs)


@pytest.mark.parametrize("arm", ARMS)
def test_one_hook_pair_per_physical_group_records_every_site(arm):
    path = make(arm)
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        empty = profiler.summary()
        assert empty["physical_groups"] == 2
        assert empty["callbacks_per_physical_group"] == 2
        assert empty["total_site_invocations"] == 0
        assert all(row["pre_hooks"] == 1 and row["post_hooks"] == 1
                   for row in empty["hook_registrations"])
        assert all(row["hooks_now"] == 2 for row in empty["hook_registrations"])
        batch = training_batch(path)
        before = cost_snapshot(path)
        logits = profiler.forward(path, *ids_and_mask(4), batch, scope=scope())
        after = cost_snapshot(path)
        summary = profiler.summary()
        assert tuple(logits.shape)[:2] == (1, 4)
        assert summary["total_site_invocations"] == EXPECTED[arm]
        assert summary["phase_site_invocations"] == {"forward": EXPECTED[arm],
                                                     "recompute": 0}
        assert summary["site_invocations"] == {
            str(site): 1 for site in path.spec.sites}
        assert summary["physical_groups"] == 2
        assert sum(summary["group_invocations"].values()) == EXPECTED[arm]
        assert len(summary["group_invocations"]) == 2
        assert len({event["group"] for event in summary["events"]}) == 2
        assert all(event["output_requires_grad"] for event in summary["events"])
        assert all(event["input_bytes"] > 0 and event["output_bytes"] > 0
                   for event in summary["events"])
        delta = ledger_delta(before, after)
        assert delta["memory"]["snapshot_copies"] == 1
        assert delta["memory"]["retrieval_calls"] == 0
        assert delta["memory"]["admission_calls"] == 0
        assert delta["memory"]["durable_write_bytes"] == 0
        assert summary["memory_side_effects_not_created_by_profiler"] == {
            "retrieval_calls": 0, "admission_calls": 0,
            "snapshot_copies": 0, "durable_write_bytes": 0}
        assert summary["timing"]["single_forward_fixture_only"] is True
        assert summary["timing"]["group_wall_ns_total"] >= 0
        assert summary["activation_bytes"]["input_total"] > 0
    finally:
        profiler.close()
    for module in path.reader.groups.values():
        assert getattr(module, "_density_profiler_owner", None) is None
        assert not module._forward_pre_hooks and not module._forward_hooks


@pytest.mark.parametrize("arm", ARMS)
def test_recompute_identity_reuses_one_snapshot_and_has_no_memory_side_effects(arm):
    path = make(arm)
    ids, mask = ids_and_mask(4)
    batch = training_batch(path)
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        snapshot = path.snapshot(ids, batch, scope=scope())
        first = profiler.instrument(snapshot, forward_id="f-1", sequence_id=4)
        baseline = path.adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False, use_spine=False,
            intrinsic_memory=first)
        before = cost_snapshot(path)
        second = profiler.instrument(snapshot, phase="recompute",
                                     recompute_of="f-1", sequence_id=4)
        recomputed = path.adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False, use_spine=False,
            intrinsic_memory=second)
        after = cost_snapshot(path)
        assert torch.equal(baseline, recomputed)
        assert_no_read_write_effects(before, after)
        summary = profiler.summary()
        assert summary["total_site_invocations"] == 2 * EXPECTED[arm]
        assert summary["phase_site_invocations"] == {
            "forward": EXPECTED[arm], "recompute": EXPECTED[arm]}
        assert summary["recompute_passes"] == {"recompute-1": "f-1"}
        recompute_events = [event for event in summary["events"]
                            if event["phase"] == "recompute"]
        assert all(event["recompute_of"] == "f-1" for event in recompute_events)
        with pytest.raises(DensityError):
            profiler.instrument(snapshot, forward_id="f-1")
        other = path.snapshot(ids, batch, scope=scope())
        with pytest.raises(DensityError, match="one snapshot"):
            profiler.instrument(other, phase="recompute", recompute_of="f-1")
        with pytest.raises(DensityError, match="unknown forward"):
            profiler.instrument(other, phase="recompute", recompute_of="missing")
    finally:
        profiler.close()
        path.close()


def test_duplicate_profiler_registration_is_refused():
    path = make("two")
    first = SiteInvocationProfiler(path)
    try:
        with pytest.raises(DensityError, match="already"):
            SiteInvocationProfiler(path)
        assert first.summary()["duplicate_registration_refusals"] == 0
    finally:
        first.close()
        path.close()


def test_group_call_outside_site_dispatch_is_attributed_as_error():
    path = make("two")
    profiler = SiteInvocationProfiler(path)
    group = next(iter(profiler.groups.values()))
    try:
        hidden = torch.zeros(1, 4, 8)
        slots = torch.zeros(1, 1, 8, 256)
        available = torch.zeros(1, 1, 8, dtype=torch.bool)
        with pytest.raises(DensityError, match="without a site dispatch"):
            group(hidden, slots, available)
    finally:
        profiler.close()
        path.close()


def test_active_mask_and_disabled_pass_skip_physical_invocations():
    path = make("all")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        batch = training_batch(path)
        ids, mask = ids_and_mask(4)
        baseline = profiler.forward(path, ids, mask, None, scope=scope())
        before = cost_snapshot(path)
        assert torch.equal(
            baseline, profiler.forward(path, ids, mask, replacement_batch(batch),
                                       scope=scope(), enabled=False))
        assert torch.equal(
            baseline, profiler.forward(path, ids, mask, replacement_batch(batch),
                                       scope=scope(), active_sites=()))
        after = cost_snapshot(path)
        assert ledger_delta(before, after)["memory"]["snapshot_copies"] == 0
        assert profiler.summary()["total_site_invocations"] == 0
        profiler.forward(path, ids, mask, batch, scope=scope(), active_sites=(0, 35))
        summary = profiler.summary()
        assert summary["total_site_invocations"] == 2
        assert summary["site_invocations"]["0"] == 1
        assert summary["site_invocations"]["35"] == 1
        assert sum(summary["group_invocations"].values()) == 2
        assert len(summary["group_invocations"]) == 1
    finally:
        profiler.close()
        path.close()


def replacement_batch(batch):
    # A disabled branch must not validate this deliberately invalid batch.
    return replace(batch, scope_key="different")


@pytest.mark.parametrize("arm", ARMS)
def test_all_sites_retain_gradient_after_outputs_open_with_exact_ownership(arm):
    path = make(arm)
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
        assert summary["backward_work"] == {
            "calls": 1, "processed_positions": int(ids.numel()),
            "caller_measured": True}
        assert all(event["output_requires_grad"] for event in summary["events"])
        audit = parameter_audit(path)
        assert audit["ownership_exact"] is True
        assert audit["duplicate_ownership"] == 0
        assert audit["missing_ownership"] == 0
        assert audit["extra_ownership"] == 0
        assert audit["base_parameters_optimizer_owned"] == 0
        assert audit["base_parameters_trainable"] == 0
        assert audit["reader_groups"]["groups"] == 2
        assert audit["reader_parameters"] == path.spec.reader_parameter_count()
        assert isinstance(path.optimizer, NativeFusedAdamW)
        assert all(p.grad is None for p in path.adapter.base.parameters())
        assert path.adapter.base_unchanged()
    finally:
        for hook in hooks:
            hook.remove()
        profiler.close()
        path.close()


@pytest.mark.parametrize("change", ("correction", "revocation"))
def test_profiled_snapshot_refuses_stale_authority_without_new_read(tmp_path, change):
    path, store = make("two"), open_store(tmp_path)
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        original = spans()[0]
        path.admit(store, original, scope=scope())
        batch = path.read(store, "Badge", scope=scope())
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, batch, scope=scope())
        proxy = profiler.instrument(snapshot, forward_id="f-stale")
        proxy.delta(3, torch.zeros(1, 4, 8))
        if change == "correction":
            corrected = replace(spans("Badge means blue.", name="new", available=6)[0],
                                source_epoch=2, supersedes=original.record_id)
            assert path.correct(store, original.record_id, corrected, scope=scope()).outcome.ok
        else:
            store.revoke_record(original.record_id, reason="withdrawn", scope=scope())
        before = cost_snapshot(path)
        with pytest.raises(ContextualMemoryError, match="invalidated"):
            proxy.delta(7, torch.zeros(1, 4, 8))
        after = cost_snapshot(path)
        assert ledger_delta(before, after)["memory"]["retrieval_calls"] == 0
        assert ledger_delta(before, after)["memory"]["snapshot_copies"] == 0
    finally:
        profiler.close()
        path.close()
        store.close()
