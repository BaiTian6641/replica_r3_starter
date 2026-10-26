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


def test_execute_requires_offline_environment(monkeypatch):
    monkeypatch.setenv("REPLICA_CAPACITY_GUARD", "1")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE=1"):
        PROBE.execute(SimpleNamespace(mode="sustained"), profiler=None)


COMPARE_SPEC = importlib.util.spec_from_file_location(
    "fused_compare", ROOT / "scripts/compare_fused_continuation.py")
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(COMPARE)


def _norm_summary(l2):
    return {"l1_norm": l2, "l2_norm": l2, "max_abs": l2, "finite": True}


def _update(step, loss, gradient_norm, parameter_l2, gradient_l2, moment_l2):
    return {
        "step": step, "loss": loss, "lr_factor": 1.0,
        "learning_rates": [1e-4],
        "gradient_norm_before_clipping": gradient_norm,
        "processed_tokens": 440, "supervised_tokens": 80,
        "training_seconds": 8.0,
        "parameter_norms": _norm_summary(parameter_l2),
        "gradient_norms_pre_clip": _norm_summary(gradient_l2),
        "optimizer_moment_norms": _norm_summary(moment_l2),
    }


def _final_fingerprint(parameter_l2, moment_l2, parameter_hash):
    entry = {"l1_norm": parameter_l2, "l2_norm": parameter_l2, "max_abs": parameter_l2,
             "value_sha256": parameter_hash, "dtype": "torch.float32",
             "device": "cpu", "shape": [2], "storage_bytes": 8,
             "storage_offset": 0, "stride": [1], "is_contiguous": True,
             "finite": True}
    moment = {**entry, "value_sha256": moment_l2.__repr__()}
    return {
        "format": "replica.training_fingerprint.v1",
        "parameters": {"weight": {**entry, "value_sha256": parameter_hash}},
        "buffers": {"buffers": {}},
        "optimizer": {"state": {"weight": {"exp_avg": moment}}},
    }


def _report(mode, updates, output_loss, output_hash, parameter_hash,
            *, checkpoint=None, boundary=None, accounting=None,
            before_hash="initial"):
    return {
        "status": "completed",
        "execution_mode": mode,
        "changed_tensors": 381,
        "model_exported": False,
        "protocol_sha256": "protocol",
        "implementation_sha256": {"probe": "implementation"},
        "parameter_hashes_before": {"weight": before_hash},
        "parameter_hashes_after": {"weight": parameter_hash},
        "checkpoint": checkpoint,
        "steps": 1 if mode == "first" else 2,
        "start_step": 0 if mode != "resume" else 1,
        "optimizer_state_step_values": [1.0] if mode == "first" else [2.0],
        "updates": updates,
        "output_fingerprint": {"loss": output_loss, "logits_sha256": output_hash},
        "checkpoint_reload_validated": mode == "resume",
        "resume_start_matches_first_process": mode == "resume",
        "boundary_fingerprint": boundary,
        "final_fingerprint": _final_fingerprint(
            updates[-1]["parameter_norms"]["l2_norm"],
            updates[-1]["optimizer_moment_norms"]["l2_norm"],
            f"final-{parameter_hash}"),
        "resume_accounting": accounting,
    }


def _guard():
    return {
        "status": "completed_within_declared_sampled_limits",
        "elapsed_seconds": 10.0,
        "guard": {
            "policy": {"driver_limit_bytes": 50 * 1024**3,
                       "swap_growth_limit_bytes": 512 * 1024**2,
                       "minimum_free_percent": 15.0},
            "stop_reason": None,
            "peak_driver_bytes": 40 * 1024**3,
            "peak_swap_growth_bytes": 0,
            "minimum_observed_free_percent": 30.0,
        },
    }


def _boundary(parameter_hash="boundary", moment_hash="moment"):
    entry = {"l1_norm": 1.0, "l2_norm": 1.0, "max_abs": 1.0, "finite": True,
             "dtype": "torch.float32", "device": "cpu", "shape": [2],
             "storage_bytes": 8, "storage_offset": 0, "stride": [1],
             "is_contiguous": True, "value_sha256": parameter_hash}
    return {
        "format": "replica.training_fingerprint.v1",
        "train_mode": {"model_training": True, "mode_sha256": "mode",
                       "module_count": 1, "training_modules": 1,
                       "modules": [{"name": "", "training": True}]},
        "buffers": {"count": 1, "nonpersistent_names": ["rotary"],
                    "buffers": {"rotary": {**entry, "value_sha256": "rotary"}}},
        "parameters": {"weight": entry},
        "optimizer": {"parameter_count": 1, "group_count": 1,
                      "state_tensors": 1, "state_bytes": 8,
                      "state_bytes_by_dtype": {"torch.float32": 8},
                      "step_values": [1.0],
                      "groups": [{"group_name": "original", "algorithm": "adamw_fused",
                                  "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                                  "weight_decay": 0.0, "fused": True,
                                  "tensor_count": 1, "parameter_count": 2}],
                      "state": {"weight": {"exp_avg": {**entry,
                                                       "value_sha256": moment_hash}}}},
        "rng": {"python_sha256": "python", "torch_cpu": {**entry,
                                                           "value_sha256": "rng"},
                "devices": ["cpu"]},
        "schedule_state": {"completed_steps": 1, "base_rates": [1e-4]},
        "data_state": {"next_step": 1, "batch_plan_sha256": "plan"},
    }


def _comparison_fixture(resume_loss=1.3497597854584455,
                        resume_output_loss=3.1678595542907715,
                        resume_boundary=None, arms=2):
    uninterrupted = []
    for index in range(arms):
        loss = (1.3497576033696532, 1.3497593011707067)[index % 2]
        gradient_norm = (154.01649475097656, 154.0172119140625)[index % 2]
        output_loss = (3.1678543090820312, 3.1678714752197266)[index % 2]
        uninterrupted.append(_report(
            "continuous",
            [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0),
             _update(2, loss, gradient_norm,
                     10.1 + index * 0.000001, 20.1 + index * 0.000001,
                     30.1 + index * 0.000001)],
            output_loss, f"logits-{index}", f"final-{index}"))
    first = _report(
        "first",
        [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0)],
        0.06391267478466034, "first-logits", "first", checkpoint={"step": 1},
        boundary=_boundary())
    resume = _report(
        "resume",
        [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0),
         _update(2, resume_loss, 154.0167694091797, 10.1000001, 20.1000001,
                 30.1000001)],
        resume_output_loss, "resume-logits", "resume", before_hash="first",
        boundary=resume_boundary or _boundary())
    return uninterrupted, first, resume


def test_comparison_separates_aa_variation_from_exact_failures():
    uninterrupted, first, resume = _comparison_fixture()
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard(),
        policy="none", reference_index=1)
    assert result["exact_status"] == "failed"
    assert result["variation_status"] == "within_aa_envelope"
    assert result["status"] == "failed"
    assert result["policy"]["applied"] is False
    assert result["aa_variation"]["available"] is True
    assert result["aa_variation"]["earliest_divergence"]["step"] == 2
    assert result["ab_variation"]["earliest_divergence"]["step"] == 2
    assert result["ab_variation"]["earliest_divergence"]["field"] == "loss"
    assert result["earliest_divergence"]["boundary"] is None
    assert result["boundary_fingerprint_comparison"]["status"] == "matched"
    assert result["final_parameter_hash_difference_count"] == 1


def test_comparison_policy_is_explicit_and_never_relaxes_exact_status():
    uninterrupted, first, resume = _comparison_fixture()
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard(),
        policy="aa-envelope-v1", reference_index=1)
    assert result["exact_status"] == "failed"
    assert result["variation_status"] == "within_aa_envelope"
    assert result["status"] == "failed_exact_within_declared_variation"
    assert result["policy"]["applied"] is True
    assert result["policy"]["definition"]["name"] == "aa-envelope-v1"
    assert result["policy"]["does_not_relax_exact_status"] is True


def test_comparison_flags_divergence_outside_aa_envelope():
    uninterrupted, first, resume = _comparison_fixture(resume_loss=2.3497597854584455)
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard(),
        policy="aa-envelope-v1", reference_index=1)
    assert result["variation_status"] == "divergent"
    assert "update_2.loss" in result["ab_variation"]["exceeding_fields"]
    assert result["status"] == "failed"


def test_comparison_localizes_boundary_fingerprint_mismatch():
    uninterrupted, first, resume = _comparison_fixture(
        resume_boundary=_boundary(parameter_hash="wrong"))
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard(),
        policy="aa-envelope-v1", reference_index=1)
    assert result["checks"]["boundary_fingerprint_identical"] is False
    assert result["boundary_fingerprint_comparison"]["matched"] is False
    assert result["boundary_fingerprint_comparison"]["earliest_divergence"]["section"] == "parameters"
    assert result["variation_status"] == "divergent"
    assert "boundary_fingerprint" in result["ab_variation"]["exceeding_fields"]


def test_comparison_is_unqualified_without_aa_arms():
    uninterrupted, first, resume = _comparison_fixture()
    result = COMPARE.compare_reports(
        uninterrupted[:1], first, resume,
        uninterrupted_guards=[_guard()],
        first_guard=_guard(), resume_guard=_guard())
    assert result["variation_status"] == "unqualified"
    assert result["aa_variation"]["available"] is False


def test_comparison_carries_resume_accounting_and_guard_stop():
    uninterrupted, first, resume = _comparison_fixture()
    resume["resume_accounting"] = {
        "path": "native",
        "full_cpu_model_copy": False,
        "model": {"loaded": {"bytes": 1000},
                  "temporary_checkpoint_tensor_bytes": 1000},
        "optimizer": {"staged_device": "cpu",
                      "staged": {"bytes": 2000, "tensor_count": 4,
                                 "bytes_by_dtype": {"torch.float32": 2000}},
                      "migration": {"migrated_tensor_count": 4,
                                    "migrated_bytes": 2000,
                                    "max_single_tensor_bytes": 500,
                                    "peak_driver_bytes": 3000}},
        "swap_used_bytes_start": 100,
        "swap_used_bytes_end": 400,
        "swap_growth_bytes": 300,
    }
    failing_guard = _guard()
    failing_guard["guard"]["stop_reason"] = "sustained_host_swap_growth"
    failing_guard["guard"]["peak_swap_growth_bytes"] = 1019677245
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=failing_guard)
    accounting = result["resume_accounting"]
    assert accounting["staged_cpu_optimizer_bytes"] == 2000
    assert accounting["staged_cpu_optimizer_tensors"] == 4
    assert accounting["optimizer_restore_strategy"] == "cpu_staged_manual_migration"
    assert accounting["optimizer_restore_device"] == "cpu"
    assert accounting["restored_optimizer_bytes"] is None
    assert accounting["restored_optimizer_device"] is None
    assert accounting["optimizer_migration_applied"] is True
    assert accounting["migrated_tensor_count"] == 4
    assert accounting["full_cpu_model_copy"] is False
    assert accounting["swap_growth_bytes"] == 300
    assert accounting["guard_stop_reason"] == "sustained_host_swap_growth"
    assert "interrupted_resume_swap_within_limit" in result["failures"]


def test_boundary_match_gate_reports_earliest_divergence():
    comparison = {"matched": False,
                  "earliest_divergence": {"section": "parameters",
                                          "name": "weight",
                                          "fields": ["value_sha256"]}}
    with pytest.raises(RuntimeError, match="parameters"):
        PROBE._require_boundary_match(comparison, "resume")
    PROBE._require_boundary_match({"matched": True}, "resume")


def test_summary_excludes_fingerprint_instrumentation_time():
    record = {
        "step": 1, "total_seconds": 10.0, "training_seconds": 8.0,
        "instrumentation_seconds": 2.0,
        "processed_tokens": 10, "supervised_tokens": 4,
        "forward_backward_seconds": 5.0, "gradient_audit_seconds": 1.0,
        "gradient_clip_seconds": 1.0, "optimizer_seconds": 1.0,
    }
    summary = PROBE._summary([record])
    assert summary["seconds"] == 8.0
    assert summary["instrumentation_seconds"] == 2.0
    assert summary["updates_per_second"] == 0.125


def test_comparison_reports_direct_optimizer_restore_without_cpu_staging():
    uninterrupted, first, resume = _comparison_fixture()
    resume["resume_accounting"] = {
        "path": "native",
        "full_cpu_model_copy": False,
        "model": {"loaded": {"bytes": 1000},
                  "temporary_checkpoint_tensor_bytes": 1000,
                  "checkpoint_tensors_adopted_by_model": True},
        "optimizer": {
            "restore_strategy": "direct_target_device",
            "target_device": "mps",
            "restored_device": "mps",
            "staged_device": None,
            "staged": {"tensor_count": 0, "bytes": 0, "bytes_by_dtype": {}},
            "staged_cpu_optimizer_bytes": 0,
            "restored": {"tensor_count": 2, "bytes": 2000,
                         "bytes_by_dtype": {"torch.float32": 2000}},
            "migration": None,
        },
        "swap_used_bytes_start": 100,
        "swap_used_bytes_end": 250,
        "swap_growth_bytes": 150,
        "resident_estimate": {"staged_cpu_optimizer_bytes": 0,
                              "restored_optimizer_bytes": 2000},
    }
    result = COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard())
    accounting = result["resume_accounting"]
    assert accounting["available"] is True
    assert accounting["path"] == "native"
    assert accounting["full_cpu_model_copy"] is False
    assert accounting["optimizer_restore_strategy"] == "direct_target_device"
    assert accounting["optimizer_restore_device"] == "mps"
    assert accounting["staged_cpu_optimizer_bytes"] == 0
    assert accounting["staged_cpu_optimizer_tensors"] == 0
    assert accounting["restored_optimizer_bytes"] == 2000
    assert accounting["restored_optimizer_tensors"] == 2
    assert accounting["restored_optimizer_bytes_by_dtype"] == {"torch.float32": 2000}
    assert accounting["restored_optimizer_device"] == "mps"
    assert accounting["optimizer_migration_applied"] is False
    assert accounting["migrated_tensor_count"] is None
    assert accounting["migrated_bytes"] is None
    assert accounting["max_single_tensor_bytes"] is None
    assert accounting["migration_peak_driver_bytes"] is None
    assert accounting["resident_estimate"]["restored_optimizer_bytes"] == 2000


def test_direct_restore_protocol_predeclares_fresh_aa_budget_and_stops():
    protocol = json.loads(
        (ROOT / "configs/fused_continuation_direct_restore_v1.json").read_text())
    assert protocol["format"] == (
        "replica.fused_continuation_direct_restore.protocol.v1")
    comparison = protocol["continuation_comparison"]
    assert [arm["id"] for arm in comparison["arms"]] == [
        "uninterrupted_a", "uninterrupted_b", "first", "resume"]
    envelope = comparison["aa_envelope"]
    assert envelope["source"] == "the two fresh uninterrupted arms only"
    assert envelope["historical_envelope_allowed"] is False
    assert envelope["extra_margin"] == 0.0
    assert envelope["update_1_loss_and_gradient_norm_exact"] is True
    assert envelope["boundary_fingerprint_exact"] is True
    assert envelope["pass_label_forbidden_unless_exact_status_passes"] is True
    assert comparison["required_accounting"] == {
        "staged_cpu_optimizer_bytes": 0,
        "restored_optimizer_bytes_positive": True,
        "restore_strategy": "direct_target_device",
    }
    storage = protocol["storage_policy"]
    assert storage["checkpoint_writes"] == 1
    assert storage["final_model_exports"] == 0
    assert storage["duplicate_base_saves"] == 0
    assert storage["max_new_checkpoint_bytes"] == 31_000_000_000
    assert storage["max_new_artifact_bytes_including_checkpoint"] == (
        34_000_000_000)
    assert protocol["resource_policy"]["native_jobs_at_once"] == 1
    assert protocol["resource_policy"]["native_model_loads"] == 4
    assert protocol["resource_policy"]["cpu_fallback"] is False
    assert protocol["resource_policy"]["downloads"] == 0
    assert "any capacity-guard stop" in comparison["stop_conditions"]
    assert "any first/resume boundary fingerprint mismatch" in (
        comparison["stop_conditions"])
