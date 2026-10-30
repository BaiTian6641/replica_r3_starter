"""Regressions for the independent v12 density repair findings D1-D5."""
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from replica_lab.intrinsic_memory_density_receipts_v12 import write_new_json
import replica_lab.intrinsic_memory_density_v12 as density_module
from replica_lab.intrinsic_memory_density_v12 import (
    DensityError, DensityProducerBoundary, ProducerBoundaryViolation,
    SiteInvocationProfiler, assert_no_read_write_effects,
    require_timing_synchronization, synchronize_timing,
)
from test_intrinsic_memory_density_v12_profiler import make, training_batch
from test_tiered_memory import scope
from replica_lab.contextual_spark_fixture import ids_and_mask

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import profile_intrinsic_memory_density_v12 as profile
import run_intrinsic_memory_density_v12_cost_pilot as cost_pilot


def test_cost_pilot_zeroes_gradients_before_each_declared_pass(monkeypatch):
    original_build = cost_pilot.build_path
    calls = []

    def counted_build(arm):
        path = original_build(arm)
        original_zero = path.optimizer.zero_grad

        def counted_zero(*args, **kwargs):
            calls.append(1)
            return original_zero(*args, **kwargs)

        monkeypatch.setattr(path.optimizer, "zero_grad", counted_zero)
        return path

    monkeypatch.setattr(cost_pilot, "build_path", counted_build)
    cost_pilot.run_pass("two", 2)
    assert len(calls) == 2


def test_empty_list_active_sites_uses_disabled_parity_no_work():
    path = profile.build_path("two")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        ids, mask = profile.ids_and_mask(4)
        batch = profile.training_batch(path)
        result = profiler.forward(path, ids, mask, batch, scope=profile.scope(),
                                  active_sites=[])
        expected = path.forward(ids, mask, batch, scope=profile.scope(),
                                active_sites=[])
        assert torch.equal(result, expected)
        assert profiler.summary()["total_site_invocations"] == 0
    finally:
        profiler.close()
        path.close()


def test_rejected_instrument_does_not_register_forward_or_recompute_identity():
    path = make("two")
    profiler = SiteInvocationProfiler(path, expected_groups=2)
    try:
        ids, _ = ids_and_mask(4)
        snapshot = path.snapshot(ids, training_batch(path), scope=scope())
        with pytest.raises(DensityError):
            profiler.instrument(snapshot, forward_id="f-9", active_sites=(7, 3))
        summary = profiler.summary()
        assert summary["forward_ids"] == []
        assert summary["recompute_passes"] == {}
        # The rejected id remains usable for a valid retry.
        proxy = profiler.instrument(snapshot, forward_id="f-9", active_sites=(3, 7))
        assert proxy.forward_id == "f-9"
        assert profiler.summary()["forward_ids"] == ["f-9"]
    finally:
        profiler.close()
        path.close()


def test_authority_revalidation_is_reported_separately_from_side_effects():
    before = {"format": "test", "memory": {
        "retrieval_calls": 4, "admission_calls": 2, "snapshot_copies": 1,
        "source_revalidations": 10}, "source": {}}
    after = {"format": "test", "memory": {
        "retrieval_calls": 4, "admission_calls": 2, "snapshot_copies": 1,
        "source_revalidations": 13}, "source": {}}
    before["source"] = {"requests": 1, "forwards": 1, "cache_misses": 1}
    after["source"] = {"requests": 2, "forwards": 2, "cache_misses": 2}
    delta = assert_no_read_write_effects(before, after)
    assert delta["authority_validation"] == {"source_revalidations": 3}
    assert delta["permitted_source_recomputation"] == {
        "source.requests": 1, "source.forwards": 1, "source.cache_misses": 1}
    bad = {"format": "test", "memory": {
        "retrieval_calls": 5, "source_revalidations": 13},
        "source": {"requests": 2, "forwards": 2, "cache_misses": 2}}
    with pytest.raises(DensityError, match="retrieval/write/copy"):
        assert_no_read_write_effects(before, bad)


def test_partial_hook_registration_rolls_back_earlier_groups():
    path = make("two")
    first, second = path.reader.groups["0"], path.reader.groups["1"]
    sentinel = object()
    second._density_profiler_owner = sentinel
    try:
        with pytest.raises(DensityError, match="already"):
            SiteInvocationProfiler(path, expected_groups=2)
        assert getattr(first, "_density_profiler_owner", None) is None
        assert not first._forward_pre_hooks and not first._forward_hooks
        assert getattr(second, "_density_profiler_owner", None) is sentinel
    finally:
        del second._density_profiler_owner
        path.close()


def test_boundary_refuses_foreign_optimizer_and_null_release():
    path = make("two")
    try:
        original = path.optimizer
        path.optimizer = object()
        with pytest.raises(ProducerBoundaryViolation, match="NativeFusedAdamW"):
            DensityProducerBoundary(path, comparison="fixed_per_site")
        path.optimizer = original

        original_release = path.encoder.release
        path.encoder.release = lambda: None
        with pytest.raises(ProducerBoundaryViolation, match="release"):
            DensityProducerBoundary(path, comparison="fixed_per_site")
        path.encoder.release = original_release
    finally:
        path.close()


def test_write_new_json_refuses_existing_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    write_new_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"value": 2})
    assert path.read_text() == '{\n  "value": 1\n}\n'


def test_profile_and_pilot_temp_dirs_are_scoped_and_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    profile.profile_arm("two", include_store=True)
    assert not list(tmp_path.glob("density-v12-store-*"))
    cost_pilot.run_pass("two", 0)
    assert not list(tmp_path.glob("density-cost-pilot-*"))


def test_cpu_timing_synchronization_boundary_is_explicit():
    assert synchronize_timing(torch.device("cpu")) == "cpu_synchronous"
    assert synchronize_timing(None) == "cpu_synchronous"


def test_unavailable_backend_sync_fails_closed(monkeypatch):
    assert require_timing_synchronization(torch.device("cpu")) == "cpu_synchronous"
    monkeypatch.setattr(density_module, "synchronize_timing",
                        lambda device=None: "mps_unavailable")
    with pytest.raises(DensityError, match="did not execute"):
        require_timing_synchronization(torch.device("mps"))


def test_unavailable_cuda_sync_fails_closed():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    assert synchronize_timing(torch.device("cuda")) == "cuda_unavailable"
    with pytest.raises(DensityError, match="did not execute"):
        require_timing_synchronization(torch.device("cuda"))


def test_unknown_accelerator_type_is_refused(monkeypatch):
    from types import SimpleNamespace
    unknown = SimpleNamespace(type="xla")
    assert synchronize_timing(unknown) == "unsupported_accelerator:xla"
    with pytest.raises(DensityError, match="did not execute"):
        require_timing_synchronization(unknown)


def test_mps_requires_fallback_disabled(monkeypatch):
    import torch as torch_module
    if not hasattr(torch_module, "mps"):
        pytest.skip("torch.mps unavailable")
    calls = []
    monkeypatch.setattr(torch_module.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch_module.mps, "synchronize", lambda: calls.append(1))
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert synchronize_timing(torch.device("mps")) == "mps_cpu_fallback_enabled"
    with pytest.raises(DensityError, match="did not execute"):
        require_timing_synchronization(torch.device("mps"))
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert require_timing_synchronization(torch.device("mps")) == "mps_synchronize"
    assert calls == [1]


def test_device_strings_and_typeless_objects_fail_closed(monkeypatch):
    assert synchronize_timing("cpu") == "cpu_synchronous"
    for invalid in ("xla", "not-a-device", object()):
        label = synchronize_timing(invalid)
        assert label.startswith("unsupported_accelerator:")
        with pytest.raises(DensityError, match="did not execute"):
            require_timing_synchronization(invalid)
