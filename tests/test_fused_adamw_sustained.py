"""CPU-only checks for the guarded fused-AdamW native probe package."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fused_sustained", ROOT / "scripts/probe_fused_adamw_sustained.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
PROTOCOL = json.loads(
    (ROOT / "configs/fused_adamw_sustained_probe_v1.json").read_text())


def test_protocol_preserves_full_fp32_scope_fused_backend_and_guard():
    assert PROTOCOL["optimizer"]["selection"] == "adamw_fused"
    assert PROTOCOL["optimizer"]["class"] == "torch.optim.AdamW"
    assert PROTOCOL["optimizer"]["fused"] is True
    assert PROTOCOL["optimizer"]["fallback"] is False
    assert PROTOCOL["scope"]["trainable_tensors"] == 381
    assert PROTOCOL["scope"]["trainable_parameters"] == 2_516_756_480
    assert PROTOCOL["scope"]["includes_embeddings_output_head_and_final_norm"] is True
    assert PROTOCOL["scope"]["resident_dtype"] == "float32"
    assert PROTOCOL["scope"]["compute_dtype"] == "float32"
    assert PROTOCOL["scope"]["effective_batch"] == 8
    assert PROTOCOL["scope"]["physical_microbatch"] == 1
    assert PROTOCOL["sustained"]["total_updates"] == 12
    assert PROTOCOL["sustained"]["checkpoint_or_export"] is False
    assert PROTOCOL["smoke"]["fresh_process_resume"] is True
    assert PROTOCOL["resource_policy"]["driver_limit_bytes"] == 50 * 1024**3
    assert PROTOCOL["resource_policy"]["cpu_fallback"] is False
    assert PROTOCOL["resource_policy"]["native_jobs_at_once"] == 1
    assert "do not rerun Muon" in PROTOCOL["retention"]["historical_muon"]


def test_summary_reports_actual_tokens_and_phase_costs():
    records = [{
        "step": index,
        "total_seconds": 2.0,
        "processed_tokens": 10,
        "supervised_tokens": 4,
        "forward_backward_seconds": 1.0,
        "gradient_audit_seconds": 0.25,
        "gradient_clip_seconds": 0.25,
        "optimizer_seconds": 0.5,
    } for index in range(1, 4)]
    summary = PROBE._summary(records)
    assert summary["updates"] == 3
    assert summary["seconds"] == 6.0
    assert summary["updates_per_second"] == 0.5
    assert summary["processed_tokens_per_second"] == 5.0
    assert summary["supervised_tokens_per_second"] == 2.0
    assert summary["mean_seconds_by_phase"] == {
        "forward_backward_seconds": 1.0,
        "gradient_audit_seconds": 0.25,
        "gradient_clip_seconds": 0.25,
        "optimizer_seconds": 0.5,
    }


def test_execute_refuses_to_run_without_capacity_guard(monkeypatch):
    monkeypatch.delenv("REPLICA_CAPACITY_GUARD", raising=False)
    with pytest.raises(RuntimeError, match="capacity guard"):
        PROBE.execute(SimpleNamespace(mode="sustained"), profiler=None)


def test_gradient_audit_matches_actual_adapter_contract():
    valid = {"original": {"nonzero_gradient_tensors": 2,
                          "missing_gradient_names": []}}
    PROBE._require_full_gradient_audit(valid, 2)
    with pytest.raises(RuntimeError, match="gradient audit"):
        PROBE._require_full_gradient_audit(
            {"original": {"nonzero_gradient_tensors": 1,
                          "missing_gradient_names": ["head"]}}, 2)
